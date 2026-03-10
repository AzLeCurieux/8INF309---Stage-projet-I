"""
Promo Dashboard – Flask backend
Scraper  : Scrapling (AsyncFetcher → DynamicFetcher/Playwright fallback)
Discovery: keyword-scored link following – finds promo sub-pages automatically
LLM      : GPT-4o-mini (primary, cheap) → GPT-4o (retry 3, higher quality)
Scheduler: APScheduler – auto-scrape every N hours
Jobs     : fire-and-forget – POST /crawl/<id> returns job_id, GET /job/<id> polls
"""
import asyncio
import json
import logging
import os
import re
import sys
import threading
import traceback
import uuid
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from urllib.parse import urljoin, urlparse

import mysql.connector
import numpy as np
from bs4 import BeautifulSoup
from openai import OpenAI as OpenAIClient
from dotenv import load_dotenv
from flask import Flask, jsonify, redirect, render_template, request, url_for

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    SentenceTransformer = None
    cosine_similarity = None

_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(_env_path, override=True)

DATABASE_HOST     = os.environ.get("DB_HOST", "127.0.0.1")
DATABASE_PORT     = int(os.environ.get("DB_PORT", "3306"))
DATABASE_USER     = os.environ.get("DB_USER", "root")
DATABASE_PASSWORD = os.environ.get("DB_PASSWORD", "1234")
DATABASE_NAME     = os.environ.get("DB_NAME", "promotions_db")
OPENAI_API_KEY    = os.environ.get("OPENAI_API_KEY", "")
# Optional HTTP/SOCKS5 proxy for Cloudflare-protected sites
# Format: "http://user:pass@host:port"  or  "socks5://host:port"
SCRAPER_PROXY     = os.environ.get("SCRAPER_PROXY", "")

# LLM models: mini is ~16x cheaper, used for attempts 1-2; full for attempt 3
EXTRACT_MODEL_PRIMARY  = "gpt-4o-mini"
EXTRACT_MODEL_FALLBACK = "gpt-4o"

SIMILARITY_THRESHOLD  = 0.92
INACTIVE_AFTER_DAYS   = 7
SCRAPE_INTERVAL_HOURS = int(os.environ.get("SCRAPE_INTERVAL_HOURS", "6"))
MAX_DISCOVERY_PAGES   = 3   # extra pages to follow per restaurant (reduced for perf)
GENERIC_PROMO_IMAGE   = "https://placehold.co/600x400/121220/f5a623?text=Promo"

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")

# Prevents multiple simultaneous browser instances that would crash the machine.
_playwright_lock = threading.Lock()

# Contexte local pour capturer les logs par thread/job
_thread_context = threading.local()

_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()

class JobLogHandler(logging.Handler):
    """Handler qui capture TOUS les logs système et les redirige vers le bon job."""
    def emit(self, record):
        jid = getattr(_thread_context, 'job_id', None)
        if jid:
            msg = self.format(record)
            with _jobs_lock:
                if jid in _jobs:
                    if "logs" not in _jobs[jid]: _jobs[jid]["logs"] = []
                    _jobs[jid]["logs"].append(msg)

def _new_job(name: str) -> str:
    jid = str(uuid.uuid4())[:8]
    with _jobs_lock:
        _jobs[jid] = {"status": "pending", "name": name, "result": None,
                      "error": None, "started_at": datetime.now().isoformat(),
                      "pages_crawled": [], "finished_at": None, "logs": []}
    return jid

def _set_job(jid: str, status: str, result=None, error=None, pages=None):
    with _jobs_lock:
        if jid in _jobs:
            _jobs[jid].update({
                "status": status, "result": result, "error": error,
                "finished_at": datetime.now().isoformat(),
            })
            if pages is not None:
                _jobs[jid]["pages_crawled"] = pages


def get_db():
    return mysql.connector.connect(
        host=DATABASE_HOST, port=DATABASE_PORT,
        user=DATABASE_USER, password=DATABASE_PASSWORD,
        database=DATABASE_NAME,
    )


if os.environ.get("SKIP_EMBEDDING_INIT") == "1" or SentenceTransformer is None:
    embedding_model = None
    logging.info("Embedding model skipped (SKIP_EMBEDDING_INIT=1)")
else:
    try:
        embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        logging.info("Embedding model loaded.")
    except Exception as exc:
        logging.warning(f"Embedding model load failed: {exc}")
        embedding_model = None

def get_embedding(text: str):
    if not text or embedding_model is None:
        return None
    try:
        return np.array(embedding_model.encode([text])[0], dtype=np.float32)
    except Exception as exc:
        logging.warning(f"Embedding error: {exc}")
        return None


GRADE_BASELINE = 20.0
CATEGORY_KEYWORDS = [
    ("Duo",            ["duo", "couple", "2 person", "2 personnes", "pour deux"]),
    ("Famille",        ["famille", "family", "4 person", "pour la famille"]),
    ("Solo",           ["solo", "individuel", "individual", "1 person", "pour un"]),
    ("Happy Hour",     ["happy hour", "heure heureuse", "5@7", "5 à 7"]),
    ("Spécial Midi",   ["midi", "lunch", "déjeuner"]),
    ("Spécial Poulet", ["poulet", "chicken", "wings", "ailes"]),
    ("Offre Locale",   ["local", "offre locale", "local offer"]),
    ("Spécial du Jour",["du jour", "journée", "daily", "daily offer"]),
]

def _parse_price_float(s) -> float | None:
    if not s or str(s).strip().lower() in ("not provided", "n/a", ""):
        return None
    m = re.search(r"(\d+(?:[.,]\d+)?)", str(s))
    return float(m.group(1).replace(",", ".")) if m else None

def classify_promotion(price_str, promo_type="", promo_details=""):
    p = _parse_price_float(price_str)
    if p is None:
        grade, savings = "N/A", 0.0
    elif p < 7:
        grade, savings = "A+", GRADE_BASELINE - p
    elif p < 12:
        grade, savings = "A",  GRADE_BASELINE - p
    elif p < 16:
        grade, savings = "B",  GRADE_BASELINE - p
    elif p < 20:
        grade, savings = "C",  GRADE_BASELINE - p
    else:
        grade, savings = "D",  max(0.0, GRADE_BASELINE - p)
    text = f"{promo_type} {promo_details}".lower()
    category = "Autre"
    for cat, kws in CATEGORY_KEYWORDS:
        if any(k in text for k in kws):
            category = cat
            break
    return grade, round(savings, 2), category


_IMG_URL_PROMO_KW = [
    "banner", "promo", "promotion", "offre", "deal", "special", "feature",
    "vedette", "hero", "slide", "carousel", "offer", "highlight", "featured",
    "ad-", "-ad", "annonce", "publicite", "pub-",
]
_IMG_CTX_PROMO_KW = [
    "promo", "promotion", "offre", "offer", "special", "spécial", "rabais",
    "réduction", "deal", "discount", "combo", "économie", "vedette", "featured",
    "happy hour", "limited", "limité", "midi", "lunch", "prix", "price",
    "save", "gratuit", "free", "%",
]
_IMG_SKIP_KW = [
    "logo", "icon", "favicon", "sprite", "avatar", "profile", "thumb-", "-tiny",
    "1x1", "pixel", "track", "blank", "placeholder", ".svg",
]

def _find_candidate_promo_images(text: str, max_candidates: int = 35) -> list[dict]:
    """
    Scan combined page text for [IMG_N:url ALT:text] markers and score each.
    Returns top candidates sorted by score.
    """
    # Updated regex to match the new [IMG_N:URL ALT:TEXT] format
    pattern = re.compile(r'\[IMG_\d+:([^\]\s]+)(?: ALT:([^\]]*))?\]')
    candidates: list[dict] = []
    seen: set[str] = set()

    for m in pattern.finditer(text):
        url = m.group(1).strip()
        alt = (m.group(2) or "").strip()
        if url in seen:
            continue
        url_lower = url.lower()

        # Skip obvious non-promo images
        if any(skip in url_lower for skip in _IMG_SKIP_KW):
            continue

        score = 0
        # ALT text is a HUGE indicator
        if alt:
            alt_lower = alt.lower()
            for kw in _IMG_CTX_PROMO_KW:
                if kw in alt_lower: score += 15
            if re.search(r"\$\s*\d+|\d+[.,]\d{2}", alt_lower):
                score += 30

        # URL keyword bonus
        for kw in _IMG_URL_PROMO_KW:
            if kw in url_lower:
                score += 15
                break

        # Context window around the marker
        start = m.start()
        ctx_s = max(0, start - 500)
        ctx_e = min(len(text), start + len(m.group(0)) + 500)
        ctx   = text[ctx_s:ctx_e].lower()

        for kw in _IMG_CTX_PROMO_KW:
            if kw in ctx: score += 10
        if re.search(r"\$\s*\d+|\d+[.,]\d{2}", ctx):
            score += 25

        # Lowered minimum bar to be more proactive (from 8 to 5)
        if score < 5:
            continue

        seen.add(url)
        ctx_snippet = text[ctx_s:ctx_e].replace(m.group(0), " ").strip()[:300]
        candidates.append({"url": url, "score": score, "context": ctx_snippet, "alt": alt})

    candidates.sort(key=lambda x: -x["score"])
    logging.info(f"Candidate promo images found: {len(candidates)}")
    return candidates[:max_candidates]


def _html_to_text(html: str, base_url: str = "") -> str:
    """
    Convert HTML to clean text for LLM processing.
    Image URLs are embedded as [IMG_N:url ALT:text] markers.
    """
    html = html[:800_000]  # Slightly higher cap
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside",
                     "noscript", "form", "svg", "iframe", "button", "input"]):
        tag.decompose()

    # Embed meta og:image as a high-priority image
    og_img = soup.find("meta", property="og:image")
    if og_img and og_img.get("content"):
        content = og_img.get("content").strip()
        if content.startswith("http"):
            soup.insert(0, soup.new_tag("img", src=content, alt="featured_og"))

    img_count = 0
    for img in soup.find_all("img"):
        src = (img.get("src") or img.get("data-src") or
               img.get("data-lazy-src") or img.get("data-original") or
               img.get("data-srcset", "").split()[0] or "").strip()
        
        style = img.get("style", "")
        if not src and "background-image" in style:
            import re
            m = re.search(r"url\(['\"]?(.*?)['\"]?\)", style)
            if m: src = m.group(1).strip()

        if src and not src.startswith("data:") and len(src) > 5:
            if base_url and src.startswith("/"):
                parsed = urlparse(base_url)
                src = f"{parsed.scheme}://{parsed.netloc}{src}"
            elif base_url and not src.startswith("http"):
                src = urljoin(base_url, src)
            
            img_count += 1
            alt = (img.get("alt") or img.get("title") or "").strip()
            img.replace_with(f" [IMG_{img_count}:{src} ALT:{alt}] ")
        else:
            img.decompose()

    lines = [l.strip() for l in soup.get_text(separator="\n").splitlines()]
    return "\n".join(l for l in lines if l)


# Keywords that suggest a link leads to promotions / offers
_PROMO_LINK_KW = [
    # French – promos / offres
    "promo", "promotion", "offre", "offres", "spécial", "special", "vedette",
    "rabais", "réduction", "économie", "commander", "commande", "commander-en",
    "nouveau", "nouveauté", "limité", "journée", "midi", "happy", "aubaine",
    "featured", "featuring", "menu-vedette",
    # French – menu / plats (souvent des promos sur les pages vedettes)
    "menu", "plat", "repas", "mets", "assiette", "dejeuner", "dîner", "souper",
    "brunch", "lunch", "soir", "cuisine", "chef", "signature",
    # English – promos / deals
    "offer", "deal", "discount", "featured", "specials", "weekly", "daily",
    "order", "combo", "bundle", "value", "savings", "limited", "spotlight",
    # English – food pages that often carry featured promos
    "food", "eats", "dishes", "meals", "plate", "dinner", "supper",
    "seasonal", "feature", "today", "week",
]
# Link patterns that are almost certainly NOT promotions
_PROMO_LINK_EXCLUDE = [
    "facebook", "twitter", "instagram", "youtube", "linkedin", "tiktok",
    "tel:", "mailto:", "careers", "emploi", "franchise", "investor",
    "privacy", "terms", "conditions", "legal", "about-us", "a-propos",
    "history", "histoire", "team", "equipe", "login", "signin", "register",
    "carte-cadeau", "giftcard", "location", "trouver", "find-us",
    # Ordering/checkout flows – not promotion pages
    "/order", "/commande", "commander", "checkout", "panier", "cart",
    "livraison", "delivery", "/gift", "/gift-card",
    # Full menu directories (too much content, not promo pages)
    "/menus/", "cocktail", "boisson", "drinks", "wine", "vin",
]

def _score_link(url: str, anchor_text: str) -> int:
    combined = (url + " " + (anchor_text or "")).lower()
    if any(exc in combined for exc in _PROMO_LINK_EXCLUDE):
        return 0
    score = sum(10 for kw in _PROMO_LINK_KW if kw in combined)
    # High-value patterns in the URL path itself (not just anchor text)
    url_lower = url.lower()
    if any(kw in url_lower for kw in ["promo", "promotion", "offre", "deal", "special", "specials"]):
        score += 30   # very likely a promo page
    if "#/" in url:          # SPA hash-route → likely deep page with content
        score += 5
    if "/subcategory/" in url:
        score += 15
    return min(score, 100)

def _discover_promo_links(html: str, base_url: str) -> list[str]:
    """
    Extract links from the page that are likely to lead to promotions.
    Returns up to MAX_DISCOVERY_PAGES URLs, scored by keyword relevance.
    Hash-only fragments (#section) are excluded – they point to the same page.
    """
    base = urlparse(base_url)
    base_no_frag = base._replace(fragment="").geturl()
    soup = BeautifulSoup(html, "html.parser")
    scored: list[tuple[int, str]] = []
    seen: set[str] = {base_no_frag}

    for a in soup.find_all("a", href=True):
        href = (a.get("href") or "").strip()
        if not href or href.startswith("javascript:") or href.startswith("mailto:"):
            continue
        if href.startswith("#"):
            continue   # pure fragment – same page, skip
        full = urljoin(base_url, href)
        parsed = urlparse(full)
        # Strip fragment and normalize trailing slash for deduplication
        no_frag = parsed._replace(fragment="")
        path = no_frag.path.rstrip("/") or "/"
        full_no_frag = no_frag._replace(path=path).geturl()
        # Same domain only (allow different subdomains of same root)
        base_root = ".".join(base.netloc.rsplit(".", 2)[-2:])
        link_root = ".".join(parsed.netloc.rsplit(".", 2)[-2:])
        if base_root != link_root:
            continue
        if full_no_frag in seen:
            continue
        seen.add(full_no_frag)
        text = a.get_text(strip=True)
        score = _score_link(full_no_frag, text)
        if score > 0:
            scored.append((score, full_no_frag))

    scored.sort(key=lambda x: -x[0])
    result = [url for _, url in scored[:MAX_DISCOVERY_PAGES]]
    if result:
        logging.info(f"Link discovery found: {result}")
    return result


def _is_cloudflare_block(html: str) -> bool:
    lh = html.lower()
    return (
        "cloudflare" in lh and (
            "you have been blocked" in lh or
            "attention required" in lh or
            "enable cookies" in lh or
            "cf-error" in lh
        )
    )

def _content_looks_static(text: str) -> bool:
    if len(text) < 2500:
        return False
    has_price = bool(re.search(r"\$\s*\d+|\d+[.,]\d{2}", text))
    promo_kws = ["promo", "offre", "offer", "special", "rabais",
                 "discount", "deal", "saving", "happy hour", "combo",
                 "featuring", "vedette", "limité", "limited"]
    has_kw = any(k in text.lower() for k in promo_kws)
    return has_price or has_kw

async def _scroll_and_wait(page):
    """Scroll pour déclencher le lazy-load."""
    try:
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        await page.wait_for_timeout(1000)
    except Exception:
        pass


def _curl_cffi_fetch_sync(url: str, proxy: str = "") -> tuple[str, str]:
    """
    Fetch synchrone via curl_cffi avec impersonation Chrome (meilleur TLS fingerprint).
    Optionnellement passe par un proxy HTTP/SOCKS5.
    Retourne (text, html).
    """
    try:
        from curl_cffi import requests as cf
        kwargs = dict(
            impersonate="chrome131",
            timeout=20,
            allow_redirects=True,
            headers={
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
                "Accept-Language": "fr-CA,fr;q=0.9,en;q=0.8",
                "Accept-Encoding": "gzip, deflate, br",
                "Cache-Control": "max-age=0",
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Site": "none",
                "Sec-Fetch-User": "?1",
                "Upgrade-Insecure-Requests": "1",
            },
        )
        if proxy:
            kwargs["proxies"] = {"https": proxy, "http": proxy}
        r = cf.get(url, **kwargs)
        if r.status_code == 200 and not _is_cloudflare_block(r.text):
            text = _html_to_text(r.text, url)
            label = f"proxy={proxy[:20]}…" if proxy else "direct"
            logging.info(f"curl_cffi({label}) → {url} ({len(text)} chars)")
            return text, r.text
        if _is_cloudflare_block(r.text or ""):
            logging.warning(f"Cloudflare hard-block ({r.status_code}) for {url}")
        return "", ""
    except Exception as exc:
        logging.warning(f"curl_cffi failed for {url}: {exc}")
        return "", ""


async def _fetch_single_page(url: str) -> tuple[str, str]:
    """
    Pipeline de fetch en 3 étapes :
      1. curl_cffi (Chrome TLS impersonation, synchrone dans executor)
      2. Si CF-bloqué et SCRAPER_PROXY configuré → curl_cffi + proxy
      3. DynamicFetcher (Playwright/patchright) pour les sites JS-rendered
    Retourne (text, html).
    """
    from scrapling.fetchers import DynamicFetcher

    html = text = ""
    loop = asyncio.get_running_loop()

    # Étape 1 : curl_cffi direct
    text, html = await loop.run_in_executor(None, _curl_cffi_fetch_sync, url, "")

    # Étape 2 : retry via proxy si Cloudflare bloqué
    if not html and SCRAPER_PROXY:
        logging.info(f"Retrying {url} with proxy {SCRAPER_PROXY[:30]}…")
        text, html = await loop.run_in_executor(None, _curl_cffi_fetch_sync, url, SCRAPER_PROXY)

    # Étape 3 : DynamicFetcher pour JS-rendered
    # Skip si curl_cffi a déjà récupéré un HTML substantiel (>15k) avec du texte (>500)
    is_hard_antibot  = "doordash.com" in url or "ubereats.com" in url or "timhortons.ca" in url
    curl_got_content = len(html) > 15_000 and len(text) > 500
    need_playwright  = is_hard_antibot or (not curl_got_content and not _content_looks_static(text))

    if need_playwright:
        logging.info(f"Playwright needed ({len(text)} chars) for {url}")
        await loop.run_in_executor(None, _playwright_lock.acquire)
        try:
            proxy_cfg = {"server": SCRAPER_PROXY} if SCRAPER_PROXY and not html else None
            pw_kwargs = dict(
                headless=True, network_idle=True,
                timeout=45_000, wait=5000,
                useragent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
                retries=1, page_action=_scroll_and_wait,
            )
            if proxy_cfg:
                pw_kwargs["proxy"] = proxy_cfg
            resp  = await DynamicFetcher.async_fetch(url, **pw_kwargs)
            html2 = resp.html_content or ""
            if html2 and not _is_cloudflare_block(html2):
                text2 = _html_to_text(html2, url)
                if len(text2) > len(text):
                    html, text = html2, text2
                    logging.info(f"Playwright → {url} ({len(text)} chars)")
            elif _is_cloudflare_block(html2):
                logging.warning(
                    f"Cloudflare blocked {url}. "
                    f"{'Set SCRAPER_PROXY in .env to bypass.' if not SCRAPER_PROXY else 'Even proxy was blocked.'}"
                )
        except Exception as exc:
            logging.warning(f"Playwright failed for {url}: {exc}")
        finally:
            _playwright_lock.release()

    return text, html


async def _smart_crawl(url: str) -> tuple[str, list[str]]:
    """
    1. Fetch the given URL
    2. If the page has no promo content, also try the site root to discover links
    3. Discover promotion-related links and fetch up to MAX_DISCOVERY_PAGES extra
    4. Return combined text and list of crawled URLs
    """
    text, html = await _fetch_single_page(url)
    crawled = [url]

    # If the starting URL yielded nothing useful, try the site homepage too
    if not html or len(text) < 300:
        parsed = urlparse(url)
        root = f"{parsed.scheme}://{parsed.netloc}/"
        if root != url and root + "/" != url:
            logging.info(f"Starting URL thin, trying site root: {root}")
            root_text, root_html = await _fetch_single_page(root)
            if root_html and len(root_text) > len(text):
                html = root_html
                if root_text:
                    text = root_text
                crawled.append(root)

    if not html:
        return text, crawled

    candidate_links = _discover_promo_links(html, url)
    all_parts = [text] if text else []

    for link in candidate_links:
        if len(crawled) > MAX_DISCOVERY_PAGES + 1:
            break
        try:
            sub_text, _ = await _fetch_single_page(link)
            if sub_text and len(sub_text) > 200:
                all_parts.append(f"\n\n=== Contenu supplémentaire : {link} ===\n{sub_text}")
                crawled.append(link)
                logging.info(f"Sub-page {link}: {len(sub_text)} chars added")
        except Exception as exc:
            logging.warning(f"Sub-page fetch failed {link}: {exc}")

    return "\n".join(all_parts), crawled


def _extract_promos_sync(text: str, restaurant_name: str, page_url: str) -> list[dict]:
    """
    Send combined page text to GPT-4o-mini (attempts 1–2) then GPT-4o (attempt 3).
    Retries if empty result (model is occasionally non-deterministic).
    Called SYNCHRONOUSLY with NO asyncio event loop active in the thread.
    """
    content = text[:45_000]  # Increased from 15k to 45k to capture more data
    prompt = f"""You are a promotion extractor for a restaurant analytics system.

Restaurant: {restaurant_name}
Source URL: {page_url}

TASK: Extract every promotional item, special offer, combo, or featured dish from the text.
The text contains markers like [IMG_1:url ALT:description]. 

For each item return a JSON object with:
  - "promo_type"    : "Duo", "Famille", "Solo", "Happy Hour", "Spécial du Jour", "Combo", "Other"
  - "promo_details" : full description
  - "price"         : price like "12.99" or "Not Provided"
  - "promo_date"    : validity or "Not Provided"
  - "link"          : direct URL or "{page_url}"
  - "image_url"     : Select the ABSOLUTE BEST [IMG_N:url] for this promo. 
                      CRITICAL: 
                      1. Check the ALT text of [IMG_N] for keywords matching the promo name.
                      2. If no match, check the [IMG_N] immediately PRECEDING the promo text block.
                      3. If the page is a grid, the image is usually RIGHT ABOVE the price/title.
                      4. Return only the URL. "Not Provided" if none.

Rules:
- Include ALL items even if they look like regular menu items on this page.
- For image_url: Be extremely precise. Prefer high-index images for items at the bottom of the page.
- Return ONLY a valid JSON array.

Page content:
{content}"""

    models = [EXTRACT_MODEL_PRIMARY, EXTRACT_MODEL_PRIMARY, EXTRACT_MODEL_FALLBACK]
    for attempt, model in enumerate(models, 1):
        raw = ""
        try:
            client = OpenAIClient(api_key=OPENAI_API_KEY)
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4096,
                temperature=0 if attempt == 1 else 0.3,
                timeout=60,
            )
            raw = resp.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            parsed = json.loads(raw)
            if parsed:
                logging.info(f"Extracted {len(parsed)} promos for {restaurant_name} "
                             f"(attempt {attempt}, model={model})")
                return parsed
            logging.warning(f"Empty result for {restaurant_name} "
                            f"(attempt {attempt}/{len(models)}), retrying…")
        except json.JSONDecodeError as exc:
            logging.error(f"JSON parse error attempt {attempt} for {restaurant_name}: "
                          f"{exc}\nRaw: {raw[:200]!r}")
        except Exception as exc:
            logging.error(f"LLM error attempt {attempt} for {restaurant_name}: {exc}")
    logging.warning(f"All {len(models)} attempts failed for {restaurant_name}")
    return []


_MAX_IMAGES_PER_BATCH = 15   # Increased from 5 to 15 to capture more banners

def _analyze_images_with_vision(images: list[dict], restaurant_name: str, page_url: str) -> list[dict]:
    """
    Send each candidate image to GPT-4o Vision (detail=low).
    """
    if not images or not OPENAI_API_KEY:
        return []

    all_promos: list[dict] = []
    client = OpenAIClient(api_key=OPENAI_API_KEY)
    # Filter to ensure we only send real image URLs and prioritize high scores
    batch = [img for img in images if img.get("url")][:_MAX_IMAGES_PER_BATCH]

    for img in batch:
        url     = img.get("url", "")
        context = img.get("context", "")[:400]
        if not url:
            continue
        
        # Skip small icons or tracker pixels that might have leaked through
        if "1x1" in url or "pixel" in url or "favicon" in url:
            continue

        logging.info(f"[Vision] Analyzing image: {url[:80]} (Score: {img.get('score')})")
        try:
            resp = client.chat.completions.create(
                model=EXTRACT_MODEL_FALLBACK,   # GPT-4o (vision-capable)
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"Restaurant: {restaurant_name}\n"
                                f"Page: {page_url}\n"
                                f"Text near image: {context}\n\n"
                                "Analyze this image and extract ANY visible promotions, "
                                "special deals, prices, combos, or featured items.\n"
                                "Return a JSON ARRAY. Each element:\n"
                                '  "promo_type": "Duo"/"Famille"/"Solo"/"Happy Hour"/'
                                '"Spécial du Jour"/"Limited Time"/"Featured"/"Combo"/"Other"\n'
                                '  "promo_details": full description\n'
                                '  "price": "12.99" or "Not Provided"\n'
                                '  "promo_date": validity or "Not Provided"\n'
                                f'  "link": "{page_url}"\n'
                                f'  "image_url": "{url}"\n'
                                "If the image has NO promotional content (logo, decoration, "
                                "photo only), return [].\n"
                                "Return ONLY valid JSON array, no markdown."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": url, "detail": "low"},
                        },
                    ],
                }],
                max_tokens=1024,
                temperature=0,
                timeout=30,
            )
            raw = resp.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            parsed = json.loads(raw)
            if isinstance(parsed, list) and parsed:
                logging.info(f"[Vision] Found {len(parsed)} promo(s) in {url[:60]}")
                all_promos.extend(parsed)
        except json.JSONDecodeError as exc:
            logging.warning(f"[Vision] JSON parse error for {url[:60]}: {exc}")
        except Exception as exc:
            logging.warning(f"[Vision] Failed for {url[:60]}: {exc}")

    return all_promos


def _scrape_sync(url: str, restaurant_name: str, jid: str = None) -> tuple[list[dict], list[str]]:
    """
    Two-phase pipeline (safe for background threads):
    1. Async fetch + discovery (isolated event loop, then closed)
    2. Sync LLM extraction (NO event loop active → avoids httpx/anyio bug)
    Returns (promo_list, crawled_url_list).
    """
    msg = f"Scraping {restaurant_name} @ {url}"
    logging.info(msg)

    # Phase 1: async web crawl
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        text, crawled = loop.run_until_complete(_smart_crawl(url))
        logging.info(f"Crawl finished. Pages: {len(crawled)}")
    finally:
        loop.close()
        asyncio.set_event_loop(None)   # ← MUST clear before calling OpenAI

    if not text or len(text) < 50:
        msg = f"No content scraped for {restaurant_name}"
        logging.warning(msg)
        return [], [url], []

    # Detect candidate promotional images before passing to LLM
    candidate_images = _find_candidate_promo_images(text)

    # Phase 2: LLM extraction (synchronous, no event loop)
    logging.info(f"Starting LLM extraction ({len(text)} chars)...")
    promos = _extract_promos_sync(text, restaurant_name, url)
    logging.info(f"LLM finished. Promos found: {len(promos)}")
    return promos, crawled, candidate_images


# ... (rest of methods)


def _price_sim(p1, p2) -> float:
    if p1 is None or p2 is None:
        return 0.5
    try:
        if p1 == p2: return 1.0
        maxp = max(abs(p1), abs(p2), 1e-6)
        return float(1.0 - min(1.0, abs(p1 - p2) / maxp))
    except Exception:
        return 0.5

def save_promos_to_db(restaurant_name: str, new_promos: list[dict]) -> dict:
    if not new_promos:
        return {"inserted": 0, "updated": 0, "skipped": 0}
    use_embeddings = embedding_model is not None
    PRICE_WEIGHT = 0.3
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        db = get_db()
        cur = db.cursor(dictionary=True)
        cur.execute(
            "SELECT id, promo_details, embedding, price FROM promotions_table "
            "WHERE restaurant = %s AND is_active = 1", (restaurant_name,))
        existing = []
        for row in cur.fetchall():
            if row.get("promo_details"):
                ex = {"id": row["id"], "det": row["promo_details"].strip().lower(),
                      "emb": None, "price": row.get("price")}
                if row.get("embedding"):
                    ex["emb"] = np.frombuffer(row["embedding"], dtype=np.float32).reshape(1, -1)
                existing.append(ex)

        inserted = updated = skipped = 0
        for promo in new_promos:
            det = (promo.get("promo_details") or "").strip()
            if not det or det.lower() in ("not provided", "extraction failed", "n/a"):
                skipped += 1; continue
            det_lower = det.lower()
            new_price = _parse_price_float(promo.get("price"))
            best_id, best_score = None, 0.0

            if use_embeddings:
                emb = get_embedding(det)
                emb_arr = emb.reshape(1, -1) if emb is not None else None
            else:
                emb = emb_arr = None

            if emb_arr is not None:
                for x in existing:
                    if x["emb"] is not None:
                        cos = float(cosine_similarity(emb_arr, x["emb"])[0][0])
                        score = (1 - PRICE_WEIGHT) * cos + PRICE_WEIGHT * _price_sim(
                            new_price, _parse_price_float(x["price"]))
                        if score > best_score:
                            best_score, best_id = score, x["id"]
                    elif x["det"] == det_lower:
                        best_score, best_id = SIMILARITY_THRESHOLD, x["id"]; break
            else:
                for x in existing:
                    if x["det"] == det_lower:
                        best_score, best_id = SIMILARITY_THRESHOLD, x["id"]; break

            if best_score >= SIMILARITY_THRESHOLD and best_id:
                cur.execute("UPDATE promotions_table SET last_seen=%s, is_active=1 WHERE id=%s",
                            (now, best_id))
                db.commit(); updated += 1; continue

            grade, savings, category = classify_promotion(
                promo.get("price"), promo.get("promo_type", ""), det)
            emb_bytes = emb_arr.tobytes() if emb_arr is not None else None
            _img = (promo.get("image_url") or "").strip()
            if not _img or _img.lower() in ("not provided", "n/a", "none"):
                _img = None
            try:
                cur.execute(
                    "INSERT INTO promotions_table "
                    "(restaurant, promo_type, promo_details, price, promo_date, link, image_url, "
                    "saved_date_time, embedding, grade, savings_estimate, category, is_active, last_seen) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,1,%s)",
                    (restaurant_name, promo.get("promo_type","Other"), det,
                     promo.get("price"), promo.get("promo_date"),
                     promo.get("link"), _img,
                     now, emb_bytes, grade, savings, category, now))
                db.commit()
                existing.append({"id": cur.lastrowid, "det": det_lower,
                                  "emb": emb_arr, "price": promo.get("price")})
                inserted += 1
                logging.info(f"Inserted: {restaurant_name} | {det[:55]} | {grade}")
            except mysql.connector.Error as err:
                logging.error(f"Insert error: {err}"); db.rollback()

        cur.close(); db.close()
        return {"inserted": inserted, "updated": updated, "skipped": skipped}
    except mysql.connector.Error as err:
        logging.error(f"save_promos_to_db error: {err}")
        return {"inserted": 0, "updated": 0, "skipped": 0, "error": str(err)}

def mark_inactive_promos(restaurant_name: str) -> int:
    cutoff = (datetime.now() - timedelta(days=INACTIVE_AFTER_DAYS)).strftime("%Y-%m-%d %H:%M:%S")
    try:
        db = get_db()
        cur = db.cursor()
        cur.execute(
            "UPDATE promotions_table SET is_active=0 "
            "WHERE restaurant=%s AND is_active=1 AND last_seen < %s",
            (restaurant_name, cutoff))
        count = cur.rowcount
        db.commit(); cur.close(); db.close()
        if count: logging.info(f"Marked {count} promos inactive for {restaurant_name}")
        return count
    except mysql.connector.Error as err:
        logging.error(f"mark_inactive_promos error: {err}"); return 0


# Seuils de similarité pour la déduplication fuzzy
_DEDUP_TEXT_THRESHOLD  = 0.82   # textes similaires → duplicate
_DEDUP_PRICE_THRESHOLD = 0.68   # seuil plus bas si le prix est aussi identique
_NON_PROMO_VERY_SHORT  = 14     # détails trop courts → clairement pas une promo

# Mots/phrases de navigation fréquemment extraits par erreur
_NAV_PHRASES = {
    "view more", "see more", "learn more", "read more", "click here",
    "see all", "show all", "back", "next", "previous", "share", "follow",
    "contact", "about", "home", "order", "checkout", "find us", "locations",
    "gift card", "loyalty", "sign in", "log in", "register",
    "voir plus", "lire la suite", "en savoir plus", "retour", "suivant",
    "précédent", "fermer", "partager", "nous contacter", "à propos",
    "trouver un resto", "emploi", "carte cadeau", "commander", "livraison",
    "télécharger", "download", "reserve", "réserver",
}

# Mots-clés qui confirment qu'il s'agit d'une vraie promo
_PROMO_CONFIRM_KW = [
    "$", "%", "promo", "offre", "deal", "special", "spécial", "combo",
    "rabais", "gratuit", "free", "save", "économ", "vedette", "featured",
    "happy hour", "pizza", "burger", "poulet", "chicken", "ailes", "wings",
    "plat", "repas", "meal", "trio", "duo", "famille", "family", "entrée",
    "sandwich", "soupe", "dessert", "boisson", "drink", "assiette", "wrap",
]


def _normalize_for_dedup(text: str) -> str:
    """Normalise le texte pour comparaison : minuscules, sans ponctuation, sans stop-words."""
    t = re.sub(r"[^\w\s]", " ", (text or "").lower())
    for sw in ("avec", "et", "de", "la", "le", "les", "un", "une", "du", "des",
               "au", "aux", "the", "a", "an", "and", "with", "of", "in", "for",
               "your", "our", "this", "that", "these", "those"):
        t = re.sub(rf"\b{sw}\b", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _is_not_promo(det: str, price: str) -> bool:
    """Retourne True si l'entrée est clairement une non-promotion."""
    det = (det or "").strip()
    if len(det) < _NON_PROMO_VERY_SHORT:
        return True
    det_l = det.lower()
    # Phrase de navigation exacte
    if det_l in _NAV_PHRASES:
        return True
    # Commence par une phrase de nav (avec peu de texte en plus)
    for phrase in _NAV_PHRASES:
        if det_l.startswith(phrase) and len(det_l) < len(phrase) + 12:
            return True
    # URL seule
    if re.match(r"^https?://\S+$", det_l):
        return True
    # Pas de prix ET pas de mot-clé promo ET texte court
    has_price = (price or "Not Provided").strip().lower() not in (
        "not provided", "n/a", "", "none")
    has_kw = any(k in det_l for k in _PROMO_CONFIRM_KW)
    if not has_price and not has_kw and len(det) < 55:
        return True
    return False


def _find_duplicate_ids(promos: list[dict]) -> set[int]:
    """
    Compare toutes les paires de promos (O(n²)).
    Retourne les IDs à désactiver (on garde le plus récemment vu de chaque paire).
    """
    to_remove: set[int] = set()
    for i, a in enumerate(promos):
        if a["id"] in to_remove:
            continue
        for b in promos[i + 1:]:
            if b["id"] in to_remove:
                continue
            na = _normalize_for_dedup(a.get("promo_details") or "")[:300]
            nb = _normalize_for_dedup(b.get("promo_details") or "")[:300]
            if not na or not nb:
                continue
            sim = SequenceMatcher(None, na, nb).ratio()
            price_eq = (a.get("price") or "") == (b.get("price") or "")
            threshold = _DEDUP_PRICE_THRESHOLD if price_eq else _DEDUP_TEXT_THRESHOLD
            if sim >= threshold:
                # Garder le plus récemment vu (last_seen), sinon le plus grand id
                def _ts(p):
                    return str(p.get("last_seen") or ""), p["id"]
                keep_id   = a["id"] if _ts(a) >= _ts(b) else b["id"]
                remove_id = b["id"] if keep_id == a["id"] else a["id"]
                to_remove.add(remove_id)
                logging.debug(f"[Dedup] sim={sim:.2f} keep={keep_id} drop={remove_id}")
    return to_remove


def clean_promos_sync(restaurant_name: str) -> dict:
    """
    Nettoyage en 2 phases :
      1. Suppression rule-based des non-promotions (rapide, sans coût LLM)
      2. Déduplication fuzzy des promos restantes (SequenceMatcher)
    Désactive les entrées problématiques (is_active=0), ne les supprime pas.
    """
    try:
        db  = get_db()
        cur = db.cursor(dictionary=True)
        cur.execute(
            "SELECT id, promo_type, promo_details, price, last_seen "
            "FROM promotions_table WHERE restaurant = %s AND is_active = 1",
            (restaurant_name,),
        )
        promos = cur.fetchall()
        if not promos:
            cur.close(); db.close()
            return {"non_promo_removed": 0, "deduped": 0, "total_checked": 0}

        logging.info(f"[Clean] {restaurant_name}: {len(promos)} promos actives à analyser")

        non_promo_ids = [
            p["id"] for p in promos
            if _is_not_promo(p.get("promo_details"), p.get("price"))
        ]
        if non_promo_ids:
            fmt = ",".join(["%s"] * len(non_promo_ids))
            cur.execute(
                f"UPDATE promotions_table SET is_active=0 WHERE id IN ({fmt})",
                non_promo_ids,
            )
            db.commit()
            logging.info(f"[Clean] {restaurant_name}: {len(non_promo_ids)} non-promos supprimées")

        remaining = [p for p in promos if p["id"] not in set(non_promo_ids)]
        dup_ids   = _find_duplicate_ids(remaining)
        if dup_ids:
            fmt = ",".join(["%s"] * len(dup_ids))
            cur.execute(
                f"UPDATE promotions_table SET is_active=0 WHERE id IN ({fmt})",
                list(dup_ids),
            )
            db.commit()
            logging.info(f"[Clean] {restaurant_name}: {len(dup_ids)} doublons supprimés")

        cur.close(); db.close()
        return {
            "non_promo_removed": len(non_promo_ids),
            "deduped":           len(dup_ids),
            "total_checked":     len(promos),
        }
    except Exception as exc:
        logging.error(f"clean_promos_sync error for {restaurant_name}: {exc}")
        return {"error": str(exc), "non_promo_removed": 0, "deduped": 0, "total_checked": 0}


def _background_clean(jid: str, restaurant_name: str):
    _thread_context.job_id = jid
    _set_job(jid, "running")
    try:
        logging.info(f"[Clean] Job {jid} démarré pour {restaurant_name}")
        stats = clean_promos_sync(restaurant_name)
        stats["clean"] = True
        _set_job(jid, "done", result=stats)
        logging.info(f"[Clean] Job {jid} terminé: {stats}")
    except Exception as exc:
        logging.error(f"[Clean] Job {jid} erreur: {exc}")
        _set_job(jid, "error", error=str(exc))


AUTO_CLEAN_AFTER_SCRAPE = os.environ.get("AUTO_CLEAN_AFTER_SCRAPE", "1") == "1"

def _run_scrape_blocking(restaurant_name: str, url: str, jid: str = None) -> dict:
    promos, crawled, _ = _scrape_sync(url, restaurant_name, jid=jid)
    stats = save_promos_to_db(restaurant_name, promos)
    stats["marked_inactive"] = mark_inactive_promos(restaurant_name)
    stats["pages_crawled"]   = crawled
    if AUTO_CLEAN_AFTER_SCRAPE:
        clean_stats = clean_promos_sync(restaurant_name)
        stats["auto_cleaned"] = clean_stats
    return stats

def _background_scrape(jid: str, restaurant_name: str, url: str, rid: int = None):
    _thread_context.job_id = jid  # Lie ce thread au JID pour capturer les logs
    _set_job(jid, "running")
    try:
        promos, crawled, candidate_images = _scrape_sync(url, restaurant_name, jid=jid)
        stats = save_promos_to_db(restaurant_name, promos)
        stats["marked_inactive"]   = mark_inactive_promos(restaurant_name)
        stats["pages_crawled"]     = crawled
        stats["candidate_images"]  = candidate_images
        if rid is not None:
            stats["restaurant_id"] = rid
        if AUTO_CLEAN_AFTER_SCRAPE:
            stats["auto_cleaned"]  = clean_promos_sync(restaurant_name)
        _set_job(jid, "done", result=stats, pages=crawled)
        logging.info(f"Job {jid} finished for {restaurant_name}")
    except Exception as exc:
        logging.error(f"Job {jid} error: {exc}")
        _set_job(jid, "error", error=str(exc))


app = Flask(__name__)


def _auto_scrape_job():
    logging.info("[Scheduler] Auto-scrape starting…")
    try:
        db = get_db(); cur = db.cursor(dictionary=True)
        cur.execute("SELECT id, name, url FROM restaurants")
        restaurants = cur.fetchall()
        cur.close(); db.close()
    except Exception as exc:
        logging.error(f"[Scheduler] DB error: {exc}"); return
    for r in restaurants:
        try:
            result = _run_scrape_blocking(r["name"], r["url"])
            logging.info(f"[Scheduler] {r['name']}: {result}")
        except Exception as exc:
            logging.error(f"[Scheduler] Error {r['name']}: {exc}")

def _start_scheduler():
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        scheduler = BackgroundScheduler(timezone="America/Montreal")
        scheduler.add_job(_auto_scrape_job, "interval",
                          hours=SCRAPE_INTERVAL_HOURS, id="auto_scrape", replace_existing=True)
        scheduler.start()
        logging.info(f"[Scheduler] Auto-scrape every {SCRAPE_INTERVAL_HOURS}h – started.")
        return scheduler
    except Exception as exc:
        logging.warning(f"[Scheduler] Could not start: {exc}"); return None


@app.route("/")
def dashboard():
    try:
        db = get_db(); cur = db.cursor(dictionary=True)
        # 1. Fetch Restaurants
        cur.execute("""
            SELECT r.id, r.name, r.url, r.scraper_type,
                   COUNT(p.id)                                       AS total_promos,
                   SUM(CASE WHEN p.is_active=1 THEN 1 ELSE 0 END)   AS active_promos,
                   SUM(CASE WHEN p.grade = 'A+' AND p.is_active=1 THEN 1 ELSE 0 END) AS a_plus_count,
                   SUM(CASE WHEN p.grade IN ('A+','A') AND p.is_active=1 THEN 1 ELSE 0 END) AS top_promos,
                   MAX(p.last_seen)                                  AS last_scraped
            FROM restaurants r
            LEFT JOIN promotions_table p ON p.restaurant = r.name
            GROUP BY r.id, r.name, r.url, r.scraper_type
            ORDER BY a_plus_count DESC, top_promos DESC, r.name ASC
        """)
        restaurants = cur.fetchall()

        # 2. Selection of the Day: 6 random active A+/A promotions
        cur.execute("""
            SELECT p.*, r.id as restaurant_id, r.url as restaurant_url
            FROM promotions_table p
            JOIN restaurants r ON p.restaurant = r.name
            WHERE p.is_active = 1 AND p.grade IN ('A+', 'A')
            ORDER BY RAND()
            LIMIT 6
        """)
        selection = cur.fetchall()

        cur.close(); db.close()
    except Exception as exc:
        restaurants = []; selection = []; logging.error(f"dashboard error: {exc}")
    return render_template("index.html", restaurants=restaurants,
                           selection=selection,
                           scrape_interval_hours=SCRAPE_INTERVAL_HOURS)


@app.route("/restaurant/<int:rid>")
def restaurant_detail(rid):
    cat_filter   = request.args.get("category", "")
    grade_filter = request.args.get("grade", "")
    
    # If it's "A ", it's almost certainly "A+" that was incorrectly decoded
    if grade_filter == "A ":
        grade_filter = "A+"
    
    show_all     = request.args.get("show_all", "0") == "1"
    try:
        db = get_db(); cur = db.cursor(dictionary=True)
        cur.execute("SELECT * FROM restaurants WHERE id = %s", (rid,))
        rest = cur.fetchone()
        if not rest: cur.close(); db.close(); return "Restaurant not found", 404

        query = "SELECT * FROM promotions_table WHERE restaurant = %s"
        params = [rest["name"]]
        if not show_all: query += " AND is_active = 1"
        if cat_filter:   query += " AND category = %s"; params.append(cat_filter)
        if grade_filter: query += " AND grade = %s";    params.append(grade_filter)
        
        # Sort by: Active first, then Grade (A+ -> D -> N/A), then most recent
        query += " ORDER BY is_active DESC, FIELD(grade, 'A+', 'A', 'B', 'C', 'D', 'N/A') ASC, last_seen DESC"
        
        cur.execute(query, params); promos = cur.fetchall()

        cur.execute("SELECT DISTINCT category FROM promotions_table "
                    "WHERE restaurant = %s AND category IS NOT NULL", (rest["name"],))
        categories = [r["category"] for r in cur.fetchall()]
        cur.execute("SELECT SUM(is_active=0) inactive FROM promotions_table WHERE restaurant=%s",
                    (rest["name"],))
        row = cur.fetchone()
        inactive_count = int(row["inactive"] or 0) if row else 0
        cur.close(); db.close()
    except Exception as exc:
        logging.error(f"restaurant_detail error: {exc}"); return f"Error: {exc}", 500
    return render_template("restaurant.html", restaurant=rest, promos=promos,
                           categories=categories, category_filter=cat_filter,
                           grade_filter=grade_filter, show_all=show_all,
                           inactive_count=inactive_count)


@app.route("/about")
def about():
    return render_template("about.html", inactive_days=INACTIVE_AFTER_DAYS,
                           scrape_interval_hours=SCRAPE_INTERVAL_HOURS,
                           primary_model=EXTRACT_MODEL_PRIMARY,
                           fallback_model=EXTRACT_MODEL_FALLBACK)


@app.route("/add_restaurant", methods=["POST"])
def add_restaurant():
    name = request.form.get("name", "").strip()
    url  = request.form.get("url", "").strip()
    if not name or not url: return "Missing name or url", 400
    try:
        db = get_db(); cur = db.cursor()
        cur.execute(
            "INSERT INTO restaurants (name, url, scraper_type) VALUES (%s,%s,'scrapling') "
            "ON DUPLICATE KEY UPDATE url=VALUES(url), scraper_type='scrapling'",
            (name, url))
        db.commit(); cur.close(); db.close()
    except Exception as exc: return f"Error: {exc}", 500
    return redirect(url_for("dashboard"))


@app.route("/delete_restaurant/<int:rid>", methods=["POST"])
def delete_restaurant(rid):
    try:
        db = get_db(); cur = db.cursor(dictionary=True)
        cur.execute("SELECT name FROM restaurants WHERE id = %s", (rid,))
        rest = cur.fetchone()
        if rest:
            cur.execute("DELETE FROM promotions_table WHERE restaurant = %s", (rest["name"],))
        cur.execute("DELETE FROM restaurants WHERE id = %s", (rid,))
        db.commit(); cur.close(); db.close()
    except Exception as exc: return f"Error: {exc}", 500
    return redirect(url_for("dashboard"))


@app.route("/crawl/<int:rid>", methods=["POST"])
def crawl_one(rid):
    try:
        db = get_db(); cur = db.cursor(dictionary=True)
        cur.execute("SELECT * FROM restaurants WHERE id = %s", (rid,))
        rest = cur.fetchone(); cur.close(); db.close()
    except Exception as exc: return jsonify({"error": str(exc)}), 500
    if not rest: return jsonify({"error": "Restaurant not found"}), 404
    jid = _new_job(rest["name"])
    threading.Thread(target=_background_scrape,
                     args=(jid, rest["name"], rest["url"]),
                     kwargs={"rid": rest["id"]}, daemon=True).start()
    return jsonify({"job_id": jid, "status": "pending", "name": rest["name"],
                    "restaurant_id": rest["id"]}), 202


@app.route("/crawl_all", methods=["POST"])
def crawl_all():
    try:
        db = get_db(); cur = db.cursor(dictionary=True)
        cur.execute("SELECT * FROM restaurants"); restaurants = cur.fetchall()
        cur.close(); db.close()
    except Exception as exc: return jsonify({"error": str(exc)}), 500
    jobs = []
    for rest in restaurants:
        jid = _new_job(rest["name"])
        threading.Thread(target=_background_scrape,
                         args=(jid, rest["name"], rest["url"]),
                         kwargs={"rid": rest["id"]}, daemon=True).start()
        jobs.append({"job_id": jid, "name": rest["name"], "status": "pending",
                     "restaurant_id": rest["id"]})
    return jsonify({"jobs": jobs}), 202


@app.route("/job/<jid>")
def job_status(jid):
    with _jobs_lock:
        job = _jobs.get(jid)
    if not job: return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


@app.route("/api/job/<jid>/logs")
def job_logs(jid):
    with _jobs_lock:
        job = _jobs.get(jid)
    if not job: return jsonify({"error": "Job not found"}), 404
    return jsonify({"logs": job.get("logs", [])})


@app.route("/api/analyze_images", methods=["POST"])
def analyze_images():
    """
    Approuver l'extraction des promotions depuis des images bannières.
    Body JSON: { restaurant_id: int, images: [{url, score, context}, …] }
    Retourne un job_id à poller.
    """
    data = request.get_json(force=True) or {}
    rid    = data.get("restaurant_id")
    images = data.get("images", [])
    if not rid or not images:
        return jsonify({"error": "restaurant_id and images are required"}), 400

    try:
        db = get_db(); cur = db.cursor(dictionary=True)
        cur.execute("SELECT * FROM restaurants WHERE id = %s", (rid,))
        rest = cur.fetchone(); cur.close(); db.close()
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    if not rest:
        return jsonify({"error": "Restaurant not found"}), 404

    jid = _new_job(f"[Vision] {rest['name']}")

    def _run_image_job():
        _thread_context.job_id = jid
        _set_job(jid, "running")
        try:
            logging.info(f"[Vision] Starting image analysis: {len(images[:_MAX_IMAGES_PER_BATCH])} images for {rest['name']}")
            promos = _analyze_images_with_vision(images, rest["name"], rest["url"])
            stats  = save_promos_to_db(rest["name"], promos) if promos else \
                     {"inserted": 0, "updated": 0, "skipped": 0}
            stats["images_analyzed"] = len(images[:_MAX_IMAGES_PER_BATCH])
            _set_job(jid, "done", result=stats)
            logging.info(f"[Vision] Job {jid} done: {stats}")
        except Exception as exc:
            logging.error(f"[Vision] Job {jid} error: {exc}")
            _set_job(jid, "error", error=str(exc))

    threading.Thread(target=_run_image_job, daemon=True).start()
    return jsonify({
        "job_id":       jid,
        "name":         rest["name"],
        "image_count":  len(images[:_MAX_IMAGES_PER_BATCH]),
    }), 202


@app.route("/clean/<int:rid>", methods=["POST"])
def clean_restaurant(rid):
    """Lance un job de nettoyage (dédoublonnage + suppression non-promos) en arrière-plan."""
    try:
        db = get_db(); cur = db.cursor(dictionary=True)
        cur.execute("SELECT * FROM restaurants WHERE id = %s", (rid,))
        rest = cur.fetchone(); cur.close(); db.close()
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    if not rest:
        return jsonify({"error": "Restaurant not found"}), 404

    jid = _new_job(f"[Clean] {rest['name']}")
    threading.Thread(target=_background_clean,
                     args=(jid, rest["name"]), daemon=True).start()
    return jsonify({"job_id": jid, "name": rest["name"]}), 202


@app.route("/classify_all", methods=["POST"])
def classify_all():
    try:
        db = get_db(); cur = db.cursor(dictionary=True)
        cur.execute("SELECT id, price, promo_type, promo_details FROM promotions_table "
                    "WHERE grade IS NULL OR grade = 'N/A'")
        rows = cur.fetchall(); updated = 0
        for row in rows:
            grade, savings, category = classify_promotion(
                row["price"], row.get("promo_type",""), row.get("promo_details",""))
            cur.execute("UPDATE promotions_table SET grade=%s, savings_estimate=%s, "
                        "category=%s WHERE id=%s", (grade, savings, category, row["id"]))
            updated += 1
        db.commit(); cur.close(); db.close()
        return jsonify({"classified": updated})
    except Exception as exc: return jsonify({"error": str(exc)}), 500


@app.route("/reactivate/<int:pid>", methods=["POST"])
def reactivate_promo(pid):
    try:
        db = get_db(); cur = db.cursor()
        cur.execute("UPDATE promotions_table SET is_active=1, last_seen=NOW() WHERE id=%s", (pid,))
        db.commit(); cur.close(); db.close()
        return jsonify({"ok": True})
    except Exception as exc: return jsonify({"error": str(exc)}), 500


@app.route("/api/promotions")
def api_promotions():
    restaurant = request.args.get("restaurant", "")
    active_only = request.args.get("active", "1") == "1"
    try:
        db = get_db(); cur = db.cursor(dictionary=True)
        q = ("SELECT id, restaurant, promo_type, promo_details, price, promo_date, "
             "link, image_url, saved_date_time, last_seen, grade, savings_estimate, "
             "category, is_active FROM promotions_table WHERE 1=1")
        params = []
        if restaurant: q += " AND restaurant=%s"; params.append(restaurant)
        if active_only: q += " AND is_active=1"
        q += " ORDER BY last_seen DESC LIMIT 200"
        cur.execute(q, params); rows = cur.fetchall()
        cur.close(); db.close()
        for r in rows:
            for k in ("saved_date_time", "last_seen"):
                if r.get(k): r[k] = str(r[k])
        return jsonify(rows)
    except Exception as exc: return jsonify({"error": str(exc)}), 500


@app.route("/api/restaurants")
def api_restaurants():
    try:
        db = get_db(); cur = db.cursor(dictionary=True)
        cur.execute("SELECT id, name, url, scraper_type FROM restaurants ORDER BY name")
        rows = cur.fetchall(); cur.close(); db.close()
        return jsonify(rows)
    except Exception as exc: return jsonify({"error": str(exc)}), 500


@app.route("/api/clear_promotions", methods=["POST"])
def clear_promotions():
    try:
        db = get_db(); cur = db.cursor()
        cur.execute("DELETE FROM promotions_table")
        db.commit()
        deleted = cur.rowcount
        cur.close(); db.close()
        logging.info(f"DB cleared: {deleted} promotions deleted.")
        return jsonify({"deleted": deleted})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/fix_images", methods=["POST"])
def fix_images():
    """Replace NULL / 'Not Provided' image_url with the generic placeholder."""
    try:
        db = get_db(); cur = db.cursor()
        cur.execute(
            "UPDATE promotions_table SET image_url=%s "
            "WHERE image_url IS NULL OR TRIM(image_url)='' "
            "OR LOWER(TRIM(image_url)) IN ('not provided','n/a','none')",
            (GENERIC_PROMO_IMAGE,),
        )
        db.commit()
        updated = cur.rowcount
        cur.close(); db.close()
        return jsonify({"updated": updated})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/ping")
def ping(): return "pong", 200


@app.route("/api/scheduler/status")
def scheduler_status():
    return jsonify({"interval_hours": SCRAPE_INTERVAL_HOURS,
                    "inactive_after_days": INACTIVE_AFTER_DAYS,
                    "primary_model": EXTRACT_MODEL_PRIMARY,
                    "fallback_model": EXTRACT_MODEL_FALLBACK})


# ─── Analytics ────────────────────────────────────────────────────────────────

@app.route("/analytics")
def analytics():
    return render_template("analytics.html")


@app.route("/api/analytics/stats")
def api_analytics_stats():
    try:
        db = get_db(); cur = db.cursor(dictionary=True)

        cur.execute("""SELECT COALESCE(grade,'N/A') as grade, COUNT(*) as count
            FROM promotions_table WHERE is_active=1 GROUP BY grade
            ORDER BY FIELD(grade,'A+','A','B','C','D','N/A')""")
        grade_dist = cur.fetchall()

        cur.execute("""SELECT COALESCE(category,'Other') as category, COUNT(*) as count
            FROM promotions_table WHERE is_active=1
            GROUP BY category ORDER BY count DESC LIMIT 10""")
        cat_dist = cur.fetchall()

        cur.execute("""SELECT restaurant,
            COUNT(*) as total,
            SUM(is_active=1) as active,
            SUM(CASE WHEN grade='A+' AND is_active=1 THEN 1 ELSE 0 END) as a_plus,
            ROUND(AVG(CASE WHEN savings_estimate IS NOT NULL AND savings_estimate>0
                THEN savings_estimate END), 2) as avg_savings
            FROM promotions_table GROUP BY restaurant ORDER BY active DESC""")
        rest_comparison = cur.fetchall()

        cur.execute("""SELECT DATE_FORMAT(saved_date_time,'%Y-%m') as month, COUNT(*) as count
            FROM promotions_table
            WHERE saved_date_time >= DATE_SUB(NOW(), INTERVAL 12 MONTH)
            GROUP BY month ORDER BY month ASC""")
        timeline = cur.fetchall()

        cur.execute("""SELECT COALESCE(category,'Other') as category,
            ROUND(AVG(savings_estimate),2) as avg_savings,
            ROUND(MAX(savings_estimate),2) as max_savings, COUNT(*) as count
            FROM promotions_table WHERE is_active=1 AND savings_estimate > 0
            GROUP BY category ORDER BY avg_savings DESC""")
        savings_by_cat = cur.fetchall()

        cur.execute("""SELECT SUM(is_active=1) as active, SUM(is_active=0) as inactive,
            COUNT(*) as total FROM promotions_table""")
        active_ratio = cur.fetchone()

        cur.execute("""SELECT restaurant, promo_type, promo_details, price, grade,
            savings_estimate, category
            FROM promotions_table WHERE is_active=1 AND savings_estimate IS NOT NULL
            ORDER BY savings_estimate DESC LIMIT 5""")
        top_savings = cur.fetchall()

        cur.close(); db.close()
        return jsonify({
            "grade_dist": grade_dist,
            "cat_dist": cat_dist,
            "rest_comparison": rest_comparison,
            "timeline": timeline,
            "savings_by_cat": savings_by_cat,
            "active_ratio": active_ratio,
            "top_savings": top_savings,
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ─── Seasonal Calendar ────────────────────────────────────────────────────────

_SEASONAL_EVENTS = [
    {"month": 1,  "name_fr": "Nouvel An",             "name_en": "New Year's",       "emoji": "🥂",  "keywords": ["nouvel an","new year","janvier","january","hiver","winter"]},
    {"month": 2,  "name_fr": "Saint-Valentin",         "name_en": "Valentine's Day",   "emoji": "❤️",  "keywords": ["valentin","valentine","amour","love","février","february"]},
    {"month": 3,  "name_fr": "St-Patrick / Printemps", "name_en": "St. Patrick's Day", "emoji": "☘️",  "keywords": ["patrick","paddy","mars","march","printemps","spring"]},
    {"month": 4,  "name_fr": "Pâques",                 "name_en": "Easter",            "emoji": "🐣",  "keywords": ["pâques","easter","avril","april","spring","printemps"]},
    {"month": 5,  "name_fr": "Fête des Mères",         "name_en": "Mother's Day",      "emoji": "🌸",  "keywords": ["mère","mother","mom","maman","mai","may"]},
    {"month": 6,  "name_fr": "Fête des Pères / St-Jean","name_en": "Father's Day",     "emoji": "👨",  "keywords": ["père","father","dad","papa","st-jean","jean","juin","june","summer","été"]},
    {"month": 7,  "name_fr": "Fête du Canada",         "name_en": "Canada Day",        "emoji": "🍁",  "keywords": ["canada","juillet","july","été","summer","bbq"]},
    {"month": 8,  "name_fr": "Fin d'été",              "name_en": "Late Summer",       "emoji": "☀️",  "keywords": ["été","summer","août","august","bbq"]},
    {"month": 9,  "name_fr": "Fête du Travail / Rentrée","name_en": "Labour Day",      "emoji": "🎒",  "keywords": ["travail","labour","labor","rentrée","back to school","septembre","september"]},
    {"month": 10, "name_fr": "Halloween / Action de grâce","name_en": "Halloween & Thanksgiving","emoji": "🎃","keywords": ["halloween","thanksgiving","action de grâce","citrouille","pumpkin","octobre","october"]},
    {"month": 11, "name_fr": "Vendredi Fou",           "name_en": "Black Friday",      "emoji": "🛍️", "keywords": ["black friday","vendredi fou","novembre","november","cyber"]},
    {"month": 12, "name_fr": "Noël / Fêtes",           "name_en": "Christmas / Holidays","emoji": "🎄","keywords": ["noël","christmas","noel","décembre","december","fêtes","holiday","hiver"]},
]


@app.route("/calendar")
def promo_calendar():
    return render_template("calendar.html")


@app.route("/api/calendar")
def api_calendar():
    year = int(request.args.get("year", datetime.now().year))
    try:
        db = get_db(); cur = db.cursor(dictionary=True)
        result = []
        for event in _SEASONAL_EVENTS:
            kw_clause = " OR ".join(
                ["(LOWER(promo_details) LIKE %s OR LOWER(promo_type) LIKE %s)"]
                * len(event["keywords"])
            )
            kw_params = []
            for kw in event["keywords"]:
                kw_params.extend([f"%{kw}%", f"%{kw}%"])
            cur.execute(
                f"""SELECT id, restaurant, promo_type, promo_details, price, grade,
                    category, saved_date_time, is_active
                    FROM promotions_table
                    WHERE ({kw_clause})
                    ORDER BY FIELD(grade,'A+','A','B','C','D','N/A') ASC,
                             saved_date_time DESC LIMIT 20""",
                kw_params,
            )
            promos = cur.fetchall()
            for p in promos:
                if p.get("saved_date_time"): p["saved_date_time"] = str(p["saved_date_time"])
            result.append({**event, "promos": promos, "promo_count": len(promos)})
        cur.close(); db.close()
        return jsonify({"year": year, "events": result})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ─── Promotions Verification ──────────────────────────────────────────────────

def _background_verify(jid: str, restaurant_name: str, url: str, rid: int):
    """Fresh scrape → compare with DB → report accuracy (no DB writes)."""
    _thread_context.job_id = jid
    try:
        logging.info(f"[verify] Starting verification for {restaurant_name}")
        fresh_promos, pages, _ = _scrape_sync(url, restaurant_name, jid)

        db = get_db(); cur = db.cursor(dictionary=True)
        cur.execute(
            "SELECT id, promo_type, promo_details, price, grade, category "
            "FROM promotions_table WHERE restaurant=%s AND is_active=1",
            (restaurant_name,),
        )
        db_promos = cur.fetchall()
        cur.close(); db.close()

        matched, new_promos = [], []
        used_db_ids = set()

        for fp in fresh_promos:
            det = (fp.get("promo_details") or "").strip().lower()
            if not det: continue
            best_ratio, best_dp = 0.0, None
            for dp in db_promos:
                if dp["id"] in used_db_ids: continue
                dp_det = (dp.get("promo_details") or "").strip().lower()
                ratio = SequenceMatcher(None, det, dp_det).ratio()
                if ratio > best_ratio:
                    best_ratio, best_dp = ratio, dp
            if best_ratio >= 0.72 and best_dp:
                used_db_ids.add(best_dp["id"])
                matched.append({**fp, "db_id": best_dp["id"],
                                "db_grade": best_dp.get("grade"),
                                "similarity": round(best_ratio, 2)})
            else:
                new_promos.append(fp)

        stale = [dp for dp in db_promos if dp["id"] not in used_db_ids]
        total = len(fresh_promos)
        accuracy = round(len(matched) / total * 100) if total > 0 else 0

        _set_job(jid, "done", result={
            "scraped_count": total,
            "matched_count": len(matched),
            "new_count": len(new_promos),
            "stale_count": len(stale),
            "db_total": len(db_promos),
            "accuracy": accuracy,
            "matched": matched,
            "new": new_promos,
            "stale": stale,
            "pages": pages,
            "restaurant": restaurant_name,
            "rid": rid,
        })
        logging.info(f"[verify] Done – accuracy={accuracy}% matched={len(matched)} new={len(new_promos)} stale={len(stale)}")
    except Exception as exc:
        logging.exception(f"[verify] Error: {exc}")
        _set_job(jid, "error", error=str(exc))


@app.route("/verify/<int:rid>")
def verify_restaurant(rid):
    try:
        db = get_db(); cur = db.cursor(dictionary=True)
        cur.execute("SELECT * FROM restaurants WHERE id=%s", (rid,))
        rest = cur.fetchone(); cur.close(); db.close()
        if not rest: return "Restaurant not found", 404
    except Exception as exc:
        return f"Error: {exc}", 500
    return render_template("verify.html", restaurant=rest)


@app.route("/api/verify/<int:rid>", methods=["POST"])
def api_verify_restaurant(rid):
    try:
        db = get_db(); cur = db.cursor(dictionary=True)
        cur.execute("SELECT * FROM restaurants WHERE id=%s", (rid,))
        rest = cur.fetchone(); cur.close(); db.close()
        if not rest: return jsonify({"error": "not found"}), 404
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    jid = _new_job(f"Verify {rest['name']}")
    t = threading.Thread(target=_background_verify,
                         args=(jid, rest["name"], rest["url"], rid), daemon=True)
    t.start()
    return jsonify({"job_id": jid})


@app.errorhandler(Exception)
def handle_error(err):
    from werkzeug.exceptions import HTTPException
    if isinstance(err, HTTPException): return err
    logging.exception("Unhandled error: %s", err)
    return jsonify({"error": "internal-server-error", "detail": str(err)}), 500


_DEFAULT_RESTAURANTS = [
    ("Benny and Co", "https://bennyandco.ca/en/promotions"),
    ("Boston Pizza",  "https://www.bostonpizza.com/en/specials"),
    ("Mike",          "https://toujoursmikes.ca/offres-promotions"),
    ("Normandin",     "https://www.restaurantnormandin.com/fr/promotions"),
]

def _init_db():
    """
    Create tables if they don't exist and seed the default restaurants.
    Retries up to 30 times (2 s apart) to allow MySQL to start in Docker.
    """
    import time
    for attempt in range(30):
        try:
            db = get_db()
            cur = db.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS restaurants (
                    id           INT AUTO_INCREMENT PRIMARY KEY,
                    name         VARCHAR(255) NOT NULL UNIQUE,
                    url          VARCHAR(500) NOT NULL,
                    scraper_type VARCHAR(50) DEFAULT 'scrapling'
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS promotions_table (
                    id               INT AUTO_INCREMENT PRIMARY KEY,
                    restaurant       VARCHAR(255),
                    promo_type       VARCHAR(100),
                    promo_details    TEXT,
                    price            VARCHAR(50),
                    promo_date       VARCHAR(100),
                    link             VARCHAR(500),
                    image_url        VARCHAR(500),
                    saved_date_time  DATETIME,
                    embedding        BLOB,
                    grade            VARCHAR(10),
                    savings_estimate FLOAT,
                    category         VARCHAR(100),
                    is_active        TINYINT DEFAULT 1,
                    last_seen        DATETIME
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)
            for name, url in _DEFAULT_RESTAURANTS:
                cur.execute(
                    "INSERT IGNORE INTO restaurants (name, url, scraper_type) "
                    "VALUES (%s, %s, 'scrapling')",
                    (name, url),
                )
            db.commit()
            cur.close()
            db.close()
            logging.info("DB initialized (tables ready, default restaurants seeded).")
            return
        except mysql.connector.Error as err:
            logging.warning(f"DB not ready (attempt {attempt + 1}/30): {err}")
            time.sleep(2)
    logging.error("Could not initialize DB after 30 attempts – check DB_HOST/DB_USER/DB_PASSWORD.")


if __name__ == "__main__":
    import hypercorn.asyncio
    import hypercorn.config
    
    # Configuration avancée des logs pour tout capturer (Scrapling, Playwright, etc.)
    job_handler = JobLogHandler()
    job_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(job_handler)
    logging.getLogger().setLevel(logging.INFO)

    _init_db()
    _scheduler = _start_scheduler()
    cfg = hypercorn.config.Config()
    cfg.bind = ["0.0.0.0:5000"]
    logging.info("Starting Promo Dashboard on http://0.0.0.0:5000")
    try:
        asyncio.run(hypercorn.asyncio.serve(app, cfg))
    finally:
        if _scheduler: _scheduler.shutdown(wait=False)
