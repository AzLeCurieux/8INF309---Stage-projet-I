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
import requests as _requests_lib
from bs4 import BeautifulSoup
from openai import OpenAI as OpenAIClient
from dotenv import load_dotenv
from flask import Flask, jsonify, redirect, render_template, request, url_for, abort
from flask_login import LoginManager, login_required, current_user
from flask_mail import Mail

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
GOOGLE_MAPS_API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY", "")
# Optional HTTP/SOCKS5 proxy for Cloudflare-protected sites
# Format: "http://user:pass@host:port"  or  "socks5://host:port"
SCRAPER_PROXY     = os.environ.get("SCRAPER_PROXY", "")

# LLM models: mini is ~16x cheaper, used for attempts 1-2; full for attempt 3
EXTRACT_MODEL_PRIMARY  = "gpt-4o-mini"
EXTRACT_MODEL_FALLBACK = "gpt-4o"

SIMILARITY_THRESHOLD  = 0.92
INACTIVE_AFTER_DAYS   = 7
SCRAPE_INTERVAL_HOURS = int(os.environ.get("SCRAPE_INTERVAL_HOURS", "6"))
MAX_DISCOVERY_PAGES   = 8   # extra pages to follow per restaurant
# Circuit breaker: abort crawl after 8 min, abort full job after 10 min
SCRAPE_CRAWL_TIMEOUT           = int(os.environ.get("SCRAPE_CRAWL_TIMEOUT", "480"))
SCRAPE_CIRCUIT_BREAKER_TIMEOUT = int(os.environ.get("SCRAPE_CIRCUIT_BREAKER_TIMEOUT", "600"))
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


def _best_srcset_url(srcset: str) -> str:
    """Pick the highest-resolution URL from a srcset attribute string."""
    if not srcset:
        return ""
    parts = [p.strip() for p in srcset.split(",") if p.strip()]
    if not parts:
        return ""
    # last entry is usually the largest; take the URL part (before optional descriptor)
    return parts[-1].split()[0].strip()


def _resolve_url(src: str, base_url: str) -> str:
    if not src or src.startswith("data:") or len(src) < 6:
        return ""
    if base_url and src.startswith("//"):
        scheme = urlparse(base_url).scheme or "https"
        return f"{scheme}:{src}"
    if base_url and src.startswith("/"):
        p = urlparse(base_url)
        return f"{p.scheme}://{p.netloc}{src}"
    if base_url and not src.startswith("http"):
        return urljoin(base_url, src)
    return src


def _html_to_text(html: str, base_url: str = "") -> str:
    """
    Convert HTML to clean text for LLM processing.
    Image URLs are embedded as [IMG_N:url ALT:text] markers.
    Captures: <img>, <picture>/<source>, srcset, data-src variants,
    data-bg/data-background/data-background-image on any element,
    inline background-image styles.
    """
    html = html[:800_000]
    soup = BeautifulSoup(html, "html.parser")

    # Embed og:image BEFORE stripping, so we capture it even if in <head>
    og_img = soup.find("meta", property="og:image")
    if og_img and og_img.get("content"):
        src = og_img["content"].strip()
        if src.startswith("http"):
            soup.insert(0, soup.new_tag("img", src=src, alt="og_featured"))

    # ── Pass 1: lift best <source srcset> into the sibling <img> ──────────
    # (must run BEFORE stripping to capture <picture> anywhere in the page)
    for pic in soup.find_all("picture"):
        best = ""
        for source in pic.find_all("source"):
            cand = _best_srcset_url(source.get("srcset", ""))
            if cand:
                best = cand
                break           # first <source> is usually the best format (webp/avif)
        if best:
            img_inside = pic.find("img")
            if img_inside and not (img_inside.get("src") or "").strip():
                img_inside["src"] = best
            elif not img_inside:
                pic.insert(0, soup.new_tag("img", src=best, alt=""))
        pic.unwrap()            # keep children (the <img>) in place

    # ── Pass 2: inject data-bg / data-background images from any element ──
    _BG_ATTRS = ("data-bg", "data-background", "data-background-image",
                 "data-lazy-background", "data-src-bg")
    for attr in _BG_ATTRS:
        for el in soup.find_all(attrs={attr: True}):
            src = _resolve_url(el.get(attr, "").strip(), base_url)
            if src:
                new_img = soup.new_tag("img", src=src, alt=el.get("data-alt", ""))
                el.insert(0, new_img)

    # ── Pass 3: replace all <img> with [IMG_N:url ALT:text] markers ───────
    img_count = 0
    for img in soup.find_all("img"):
        # Collect src from every known lazy-load / srcset attribute
        src = ""
        for attr in ("src", "data-src", "data-lazy-src", "data-original",
                     "data-lazy", "data-image", "data-img"):
            src = (img.get(attr) or "").strip()
            if src and not src.startswith("data:"): break

        if not src:
            src = _best_srcset_url(img.get("srcset", "") or img.get("data-srcset", ""))

        if not src:
            style = img.get("style", "")
            if "background-image" in style:
                m = re.search(r"url\(['\"]?(.*?)['\"]?\)", style)
                if m: src = m.group(1).strip()

        src = _resolve_url(src, base_url)
        if not src:
            img.decompose()
            continue

        img_count += 1
        alt = (img.get("alt") or img.get("title") or img.get("data-alt") or "").strip()
        img.replace_with(f" [IMG_{img_count}:{src} ALT:{alt}] ")

    # ── Pass 4: strip noise tags (navigation, forms, layout boilerplate) ──
    for tag in soup(["script", "style", "nav", "footer", "header", "aside",
                     "noscript", "form", "svg", "iframe", "button", "input"]):
        tag.decompose()

    lines = [l.strip() for l in soup.get_text(separator="\n").splitlines()]
    return "\n".join(l for l in lines if l)


# Keywords that suggest a link leads to promotions / offers
_PROMO_LINK_KW = [
    # French – promos / offres
    "promo", "promotion", "offre", "offres", "spécial", "special", "vedette",
    "rabais", "réduction", "économie", "nouveau", "nouveauté", "limité",
    "journée", "midi", "happy", "aubaine", "featured", "featuring",
    "menu-vedette", "sur-mesure", "exclusif", "exclusive", "solde", "vente",
    # French – menu / plats (souvent des promos sur les pages vedettes)
    "menu", "plat", "repas", "mets", "assiette", "dejeuner", "dîner", "souper",
    "brunch", "lunch", "soir", "cuisine", "chef", "signature", "commander",
    "commande", "livraison",
    # English – promos / deals
    "offer", "deal", "discount", "featured", "specials", "weekly", "daily",
    "order", "combo", "bundle", "value", "savings", "limited", "spotlight",
    "sale", "clearance", "flash", "exclusive",
    # English – food pages that often carry featured promos
    "food", "eats", "dishes", "meals", "plate", "dinner", "supper",
    "seasonal", "feature", "today", "week", "wings", "chicken", "poulet",
    "ailes", "burger", "sandwich", "pizza", "poutine",
]
# Link patterns that are almost certainly NOT promotions
_PROMO_LINK_EXCLUDE = [
    "facebook", "twitter", "instagram", "youtube", "linkedin", "tiktok",
    "tel:", "mailto:", "careers", "emploi", "franchise", "investor",
    "privacy", "terms", "conditions", "legal", "about-us", "a-propos",
    "history", "histoire", "team", "equipe", "login", "signin", "register",
    "carte-cadeau", "giftcard", "find-us",
    # Checkout/payment flows only (not all ordering pages)
    "checkout", "panier", "cart", "/gift-card",
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

def _link_context(a_tag) -> str:
    """Extract surrounding text for a link: parent + siblings (up to 120 chars)."""
    parts = []
    parent = a_tag.parent
    if parent:
        # Section/nav label (grandparent heading)
        gp = parent.parent
        if gp:
            for hdr in gp.find_all(["h1","h2","h3","h4"], limit=1):
                t = hdr.get_text(strip=True)
                if t: parts.append(t)
        # Sibling text near the link
        for sib in list(a_tag.previous_siblings)[-2:] + list(a_tag.next_siblings)[:2]:
            if hasattr(sib, "get_text"):
                t = sib.get_text(strip=True)[:60]
                if t: parts.append(t)
    return " | ".join(parts)[:120]


def _discover_promo_links(html: str, base_url: str) -> tuple[list[str], list[dict]]:
    """
    Extract links from the page.
    Returns:
      - high_conf: links with keyword score > 0, sorted by score
      - uncertain: same-domain links with score 0, with anchor+context (for LLM review)
    Hash-only fragments (#section) are excluded.
    """
    base = urlparse(base_url)
    base_no_frag = base._replace(fragment="").geturl()
    soup = BeautifulSoup(html, "html.parser")
    scored: list[tuple[int, str]] = []
    uncertain: list[dict] = []
    seen: set[str] = {base_no_frag}

    for a in soup.find_all("a", href=True):
        href = (a.get("href") or "").strip()
        if not href or href.startswith("javascript:") or href.startswith("mailto:"):
            continue
        if href.startswith("#"):
            continue
        full = urljoin(base_url, href)
        parsed = urlparse(full)
        no_frag = parsed._replace(fragment="")
        path = no_frag.path.rstrip("/") or "/"
        full_no_frag = no_frag._replace(path=path).geturl()
        base_root = ".".join(base.netloc.rsplit(".", 2)[-2:])
        link_root = ".".join(parsed.netloc.rsplit(".", 2)[-2:])
        if base_root != link_root:
            continue
        if full_no_frag in seen:
            continue
        seen.add(full_no_frag)
        anchor = a.get_text(strip=True)
        score = _score_link(full_no_frag, anchor)
        if score > 0:
            scored.append((score, full_no_frag))
        else:
            # Collect for LLM review – skip pure utility links already in exclude list
            combined = (full_no_frag + " " + anchor).lower()
            if not any(exc in combined for exc in _PROMO_LINK_EXCLUDE):
                uncertain.append({
                    "url": full_no_frag,
                    "anchor": anchor[:80],
                    "context": _link_context(a),
                })

    scored.sort(key=lambda x: -x[0])
    high_conf = [url for _, url in scored[:MAX_DISCOVERY_PAGES]]
    if high_conf:
        logging.info(f"Link discovery (keyword): {high_conf}")
    return high_conf, uncertain[:40]  # cap uncertain at 40 for LLM


def _llm_filter_links(candidates: list[dict], page_url: str) -> list[str]:
    """
    Send uncertain links to GPT-mini in one batch call.
    Returns the subset of URLs that GPT thinks could lead to promotional content.
    """
    if not candidates or not OPENAI_API_KEY:
        return []
    lines = "\n".join(
        f"{i+1}. [{c['anchor']}] {c['url']}"
        + (f"  (contexte: {c['context']})" if c['context'] else "")
        for i, c in enumerate(candidates)
    )
    prompt = (
        f"Tu analyses le site {page_url} d'un restaurant.\n"
        "Voici des liens trouvés sur la page. Identifie ceux qui mènent probablement "
        "à des promotions, offres spéciales, menus vedettes, combos ou nouveautés.\n\n"
        f"{lines}\n\n"
        "Réponds UNIQUEMENT avec les numéros des liens pertinents, séparés par des virgules. "
        "Si aucun n'est pertinent, réponds 'aucun'."
    )
    try:
        client = OpenAIClient(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model=EXTRACT_MODEL_PRIMARY,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=80, temperature=0,
        )
        answer = resp.choices[0].message.content.strip().lower()
        if answer == "aucun":
            return []
        picked = []
        for part in re.split(r"[,\s]+", answer):
            part = part.strip()
            if part.isdigit():
                k = int(part) - 1
                if 0 <= k < len(candidates):
                    picked.append(candidates[k]["url"])
        logging.info(f"LLM link filter picked {len(picked)}/{len(candidates)}: {picked}")
        return picked
    except Exception as exc:
        logging.warning(f"_llm_filter_links error: {exc}")
        return []


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

def _robots_txt_allows(url: str, user_agent: str = "*") -> bool:
    """Check robots.txt for the given URL.
    Returns True (allow) when the check is disabled (default) or when permitted.
    Fails open: any exception → allow."""
    from models import get_setting
    if get_setting("robots_txt_check_enabled", "0") != "1":
        return True
    try:
        from urllib.robotparser import RobotFileParser
        parsed = urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        rp = RobotFileParser()
        rp.set_url(robots_url)
        rp.read()
        allowed = rp.can_fetch(user_agent, url)
        if not allowed:
            logging.info(f"[robots.txt] Disallowed by policy: {url}")
        return allowed
    except Exception as exc:
        logging.warning(f"[robots.txt] Check failed for {url}: {exc}")
        return True  # fail open


async def _scroll_and_wait(page):
    """Scroll progressif pour déclencher tout le lazy-load."""
    try:
        # Scroll progressif en 4 étapes pour déclencher les images/sections lazy
        for fraction in [0.25, 0.5, 0.75, 1.0]:
            await page.evaluate(
                f"window.scrollTo(0, document.body.scrollHeight * {fraction})"
            )
            await page.wait_for_timeout(600)
        # Revenir en haut puis attendre les éventuels appels réseaux
        await page.evaluate("window.scrollTo(0, 0)")
        await page.wait_for_timeout(800)
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

    # Respect robots.txt when the check is enabled (default: OFF)
    if not _robots_txt_allows(url):
        return "", ""

    html = text = ""
    loop = asyncio.get_running_loop()

    # Étape 1 : curl_cffi direct
    text, html = await loop.run_in_executor(None, _curl_cffi_fetch_sync, url, "")

    # Étape 2 : retry via proxy si Cloudflare bloqué
    if not html and SCRAPER_PROXY:
        logging.info(f"Retrying {url} with proxy {SCRAPER_PROXY[:30]}…")
        text, html = await loop.run_in_executor(None, _curl_cffi_fetch_sync, url, SCRAPER_PROXY)

    # Étape 3 : DynamicFetcher pour JS-rendered
    # On skip Playwright UNIQUEMENT si curl a récupéré un contenu riche ET avec des promos
    is_hard_antibot   = "doordash.com" in url or "ubereats.com" in url or "timhortons.ca" in url
    curl_got_content  = len(html) > 15_000 and len(text) > 800
    has_promo_content = _content_looks_static(text)  # a des prix / mots-clés promo
    need_playwright   = is_hard_antibot or not (curl_got_content and has_promo_content)

    if need_playwright:
        logging.info(f"Playwright needed ({len(text)} chars) for {url}")
        await loop.run_in_executor(None, _playwright_lock.acquire)
        try:
            proxy_cfg = {"server": SCRAPER_PROXY} if SCRAPER_PROXY and not html else None
            pw_kwargs = dict(
                headless=True, network_idle=True,
                timeout=60_000, wait=6000,
                useragent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
                retries=2, page_action=_scroll_and_wait,
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
    3. Discover promotion-related links from root + sub-pages (2-level discovery)
    4. Return combined text and list of crawled URLs
    """
    text, html = await _fetch_single_page(url)
    crawled = [url]
    crawled_set: set[str] = {url}

    # If the starting URL yielded nothing useful, try the site homepage too
    parsed_base = urlparse(url)
    root = f"{parsed_base.scheme}://{parsed_base.netloc}/"
    if (not html or len(text) < 500) and root not in crawled_set:
        logging.info(f"Starting URL thin, trying site root: {root}")
        root_text, root_html = await _fetch_single_page(root)
        if root_html and len(root_text) > len(text):
            html = root_html
            text = root_text
        elif root_html:
            html = root_html  # use root html for link discovery even if less text
        crawled.append(root)
        crawled_set.add(root)

    if not html:
        return text, crawled

    all_parts = [text] if text else []

    # Level-1 discovery: keyword-scored links + LLM-filtered uncertain links
    high_conf, uncertain = _discover_promo_links(html, url)

    # Ask LLM to pick relevant links among those with no keyword match
    slots_for_llm = max(0, MAX_DISCOVERY_PAGES - len(high_conf))
    llm_links: list[str] = []
    if uncertain and slots_for_llm > 0:
        loop = asyncio.get_running_loop()
        llm_links = await loop.run_in_executor(
            None, _llm_filter_links, uncertain[:30], url)
        llm_links = [l for l in llm_links if l not in crawled_set][:slots_for_llm]

    candidate_links = high_conf + llm_links

    for link in candidate_links:
        if len(crawled) >= MAX_DISCOVERY_PAGES + 1:
            break
        if link in crawled_set:
            continue
        try:
            sub_text, sub_html = await _fetch_single_page(link)
            crawled_set.add(link)
            if sub_text and len(sub_text) > 200:
                all_parts.append(f"\n\n=== Page : {link} ===\n{sub_text}")
                crawled.append(link)
                logging.info(f"Sub-page {link}: {len(sub_text)} chars added")

                # Level-2 discovery: look for more promo links inside this sub-page
                if sub_html and len(crawled) < MAX_DISCOVERY_PAGES + 1:
                    sub_high, sub_uncertain = _discover_promo_links(sub_html, link)
                    for sub_link in sub_high[:3]:
                        if sub_link not in crawled_set and len(crawled) < MAX_DISCOVERY_PAGES + 1:
                            candidate_links.append(sub_link)
        except Exception as exc:
            logging.warning(f"Sub-page fetch failed {link}: {exc}")

    return "\n".join(all_parts), crawled


def _extract_promos_sync(text: str, restaurant_name: str, page_url: str) -> list[dict]:
    """
    Send combined page text to GPT-4o-mini (attempts 1–2) then GPT-4o (attempt 3).
    Retries if empty result (model is occasionally non-deterministic).
    Called SYNCHRONOUSLY with NO asyncio event loop active in the thread.
    """
    content = text[:50_000]
    # Pre-extract all image markers so the LLM has an explicit list
    img_markers = re.findall(r'\[IMG_\d+:[^\]]+\]', content)
    img_list_hint = "\n".join(img_markers[:60]) if img_markers else "(none found)"

    prompt = f"""You are a promotion extractor for a restaurant analytics system.

Restaurant: {restaurant_name}
Source URL: {page_url}

── IMAGE MARKERS ──────────────────────────────────────────────────────────────
The page text embeds image tags as: [IMG_N:URL ALT:description]
Images found on this page ({len(img_markers)} total, first 60 shown):
{img_list_hint}

── HOW TO ASSIGN AN IMAGE TO EACH PROMO ──────────────────────────────────────
RULE 1 – PROXIMITY (most reliable):
  The [IMG_N:...] marker that appears IMMEDIATELY BEFORE a promo title/price
  in the text is almost always that promo's image. Restaurant pages are grids:
  each card has [image] then [title] then [description] then [price].

RULE 2 – ALT TEXT MATCH:
  If the ALT text of an [IMG_N] contains words from the promo name/type,
  that image belongs to this promo.

RULE 3 – NEVER LEAVE image_url EMPTY:
  If you cannot identify a specific image, use the [IMG_1:URL] (first image)
  or the og_featured image. NEVER return "Not Provided" for image_url.
  Every promo MUST have an image_url that is a real absolute URL from the list above.

── TASK ───────────────────────────────────────────────────────────────────────
Extract ONLY items that represent a real promotion or limited-time offer.
A real promotion MUST have at least one of:
  - An explicit discount (%, $off, rabais, save, économi)
  - A bundle deal (combo/duo/trio/famille at a special grouped price)
  - A time-limited or seasonal special explicitly marketed as such
  - A price explicitly labelled "special", "promo", or reduced vs normal price

DO NOT extract:
  - Regular menu items at standard prices (e.g. a plain burger or wings listed on the menu)
  - Descriptions without any deal/offer/discount component
  - Navigation elements, headers, or generic section titles

For each qualifying promo, return a JSON object with EXACTLY these fields:
  "promo_type"    : one of "Duo","Famille","Solo","Happy Hour","Spécial du Jour","Combo","Other"
  "promo_details" : full description (keep original language)
  "price"         : numeric string like "12.99", or "Not Provided"
  "promo_date"    : validity date/period, or "Not Provided"
  "link"          : most specific URL for this promo, or "{page_url}"
  "image_url"     : REQUIRED – absolute URL of the image for this promo (see rules above)

Return ONLY a valid JSON array. No markdown, no comments.

── PAGE CONTENT ───────────────────────────────────────────────────────────────
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


def _assign_missing_images(promos: list[dict], page_text: str) -> list[dict]:
    """
    Post-processing pass: for promos still missing an image_url, find the best
    match from the page's [IMG_N:url ALT:text] markers using two strategies:
      1. Alt-text keyword overlap with promo_type + promo_details
      2. Positional proximity – nearest image marker before the promo text
    Falls back to the first non-logo image on the page.
    """
    img_pattern = re.compile(r'\[IMG_\d+:([^\]\s]+)(?: ALT:([^\]]*))?\]')
    _skip = ("logo", ".svg", "icon", "favicon", "sprite", "1x1", "pixel", "blank")
    all_imgs = [
        (m.start(), m.group(1).strip(), (m.group(2) or "").strip())
        for m in img_pattern.finditer(page_text)
        if not any(s in m.group(1).lower() for s in _skip)
    ]
    if not all_imgs:
        return promos

    first_valid_url = all_imgs[0][1] if all_imgs else None
    _stopwords = {"the","le","la","les","des","pour","avec","dans","une","et","ou"}

    for promo in promos:
        img = (promo.get("image_url") or "").strip()
        if img and img.lower() not in ("not provided", "n/a", "none", ""):
            continue  # already has a real image

        search = f"{promo.get('promo_type','')} {promo.get('promo_details','')}".lower()
        words = {w for w in re.findall(r'\w{3,}', search) if w not in _stopwords}

        # Strategy 1: best alt-text keyword overlap
        best_score, best_url = 0, None
        for _, url, alt in all_imgs:
            score = sum(1 for w in words if w in alt.lower())
            if score > best_score:
                best_score, best_url = score, url

        if best_url and best_score > 0:
            promo["image_url"] = best_url
            continue

        # Strategy 2: closest image marker BEFORE the promo text in page
        snippet = search[:50]
        pos = page_text.lower().find(snippet)
        if pos > 0:
            before = [(p, u) for p, u, _ in all_imgs if p < pos]
            if before:
                promo["image_url"] = max(before, key=lambda x: x[0])[1]
                continue

        # Strategy 3: global fallback – first non-logo image
        if first_valid_url:
            promo["image_url"] = first_valid_url

    return promos


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

    # Phase 1: async web crawl (circuit breaker: abort after SCRAPE_CRAWL_TIMEOUT)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        text, crawled = loop.run_until_complete(
            asyncio.wait_for(_smart_crawl(url), timeout=SCRAPE_CRAWL_TIMEOUT)
        )
        logging.info(f"Crawl finished. Pages: {len(crawled)}")
    except asyncio.TimeoutError:
        logging.error(
            f"[CircuitBreaker] Crawl for {restaurant_name} timed out after {SCRAPE_CRAWL_TIMEOUT}s"
        )
        raise RuntimeError(
            f"Circuit breaker: crawl exceeded {SCRAPE_CRAWL_TIMEOUT}s for {restaurant_name}"
        )
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

    # Phase 3: fill missing images via alt-text/proximity matching
    missing_before = sum(1 for p in promos if not (p.get("image_url") or "").strip()
                         or (p.get("image_url") or "").lower() in ("not provided","n/a","none"))
    if missing_before:
        promos = _assign_missing_images(promos, text)
        missing_after = sum(1 for p in promos if not (p.get("image_url") or "").strip()
                            or (p.get("image_url") or "").lower() in ("not provided","n/a","none"))
        logging.info(f"Image assignment: {missing_before} missing → {missing_after} remaining after fallback")

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
_DEDUP_TEXT_THRESHOLD       = 0.92  # textes très similaires sans égalité de prix
_DEDUP_PRICE_THRESHOLD      = 0.72  # même prix → seuil plus bas (était 0.88)
_DEDUP_PRICE_LOOSE_THRESHOLD= 0.50  # même prix + l'un contient l'autre → doublon évident
_NON_PROMO_VERY_SHORT       = 14    # détails trop courts → clairement pas une promo

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

# Indicateurs d'une vraie promotion (réduction, offre groupée, spécial limité)
# NB: les noms d'aliments seuls (pizza, wings, burger…) ne sont PAS des indicateurs
#     de promotion — un item du menu normal peut les contenir.
_PROMO_DEAL_KW = [
    "$", "%", "promo", "offre", "deal", "spécial", "special",
    "rabais", "économ", "economis", "save", "savings",
    "gratuit", "free", "2 pour", "2 for", "buy one", "achetez",
    "happy hour", "combo", "duo", "trio", "famille", "family",
    "forfait", "package", "bundle", "limité", "limited", "exclusif",
    "à partir de", "starting at", "seulement", "only",
    "spécial du", "special of", "du jour", "of the day",
    "semaine", "week", "midi", "lunch", "soir", "evening",
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
    """Retourne True si l'entrée est clairement une non-promotion.

    Logique :
      - Trop court → pas une promo
      - Phrase de navigation → pas une promo
      - URL seule → pas une promo
      - Aucun prix ET aucun mot de deal → item menu normal, pas une promo
    Les noms d'aliments seuls (pizza, wings…) ne suffisent plus à confirmer
    une promo : il faut un vrai indicateur de réduction/offre/bundle.
    """
    det = (det or "").strip()
    if len(det) < _NON_PROMO_VERY_SHORT:
        return True
    det_l = det.lower()
    if det_l in _NAV_PHRASES:
        return True
    for phrase in _NAV_PHRASES:
        if det_l.startswith(phrase) and len(det_l) < len(phrase) + 12:
            return True
    if re.match(r"^https?://\S+$", det_l):
        return True
    has_price = (price or "Not Provided").strip().lower() not in (
        "not provided", "n/a", "", "none")
    has_deal_kw = any(k in det_l for k in _PROMO_DEAL_KW)
    # Sans prix ET sans indicateur de deal → item menu ordinaire
    if not has_price and not has_deal_kw:
        return True
    return False


def _find_duplicate_ids(promos: list[dict]) -> set[int]:
    """
    Compare toutes les paires de promos (O(n²)).
    Retourne les IDs à désactiver (on garde le plus récemment vu de chaque paire).

    Trois niveaux de détection :
      1. Textes presque identiques (>= TEXT_THRESHOLD) sans condition de prix
      2. Même prix ET similarité >= PRICE_THRESHOLD (plus lâche)
      3. Même prix ET l'un contient l'autre (>= PRICE_LOOSE_THRESHOLD) → évident
    """
    to_remove: set[int] = set()

    def _ts(p):
        return str(p.get("last_seen") or ""), p["id"]

    def _price_normalized(p) -> str:
        raw = (p.get("price") or "").strip().lower()
        if raw in ("", "not provided", "n/a", "none"):
            return ""
        # Normalize "$10.95" / "10.95" / "10,95" → "10.95"
        m = re.search(r"(\d+)[.,](\d{2})", raw)
        if m:
            return f"{m.group(1)}.{m.group(2)}"
        m2 = re.search(r"(\d+)", raw)
        return m2.group(1) if m2 else raw

    for i, a in enumerate(promos):
        if a["id"] in to_remove:
            continue
        na = _normalize_for_dedup(a.get("promo_details") or "")[:350]
        pa = _price_normalized(a)
        if not na:
            continue
        for b in promos[i + 1:]:
            if b["id"] in to_remove:
                continue
            nb = _normalize_for_dedup(b.get("promo_details") or "")[:350]
            pb = _price_normalized(b)
            if not nb:
                continue

            price_eq = bool(pa and pb and pa == pb)
            sim = SequenceMatcher(None, na, nb).ratio()

            is_dup = False
            if sim >= _DEDUP_TEXT_THRESHOLD:
                is_dup = True
            elif price_eq and sim >= _DEDUP_PRICE_THRESHOLD:
                is_dup = True
            elif price_eq and sim >= _DEDUP_PRICE_LOOSE_THRESHOLD:
                # Containment check: short form ⊂ long form → same promo, more text
                shorter, longer = (na, nb) if len(na) <= len(nb) else (nb, na)
                if shorter and shorter in longer:
                    is_dup = True

            if is_dup:
                keep_id   = a["id"] if _ts(a) >= _ts(b) else b["id"]
                remove_id = b["id"] if keep_id == a["id"] else a["id"]
                to_remove.add(remove_id)
                logging.debug(f"[Dedup] sim={sim:.2f} price_eq={price_eq} keep={keep_id} drop={remove_id}")
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


def _llm_reverify_restaurant(restaurant_name: str) -> dict:
    """
    Re-vérifie via GPT-4o-mini toutes les promos actives d'un restaurant.
    Demande au LLM d'identifier celles qui NE sont PAS de vraies promotions
    (items menu normaux, texte de navigation, descriptions génériques, etc.)
    et les désactive.
    Retourne {"checked": N, "deactivated": M}.
    """
    try:
        db = get_db(); cur = db.cursor(dictionary=True)
        cur.execute(
            "SELECT id, promo_type, promo_details, price FROM promotions_table "
            "WHERE restaurant=%s AND is_active=1",
            (restaurant_name,),
        )
        promos = cur.fetchall()
        cur.close(); db.close()
    except Exception as exc:
        logging.error(f"[Reverify] DB error for {restaurant_name}: {exc}")
        return {"checked": 0, "deactivated": 0, "error": str(exc)}

    if not promos:
        return {"checked": 0, "deactivated": 0}

    BATCH = 30
    to_deactivate: list[int] = []

    for i in range(0, len(promos), BATCH):
        batch = promos[i:i + BATCH]
        items_text = "\n".join(
            f'ID {p["id"]}: [{p["promo_type"]}] {p["promo_details"]}'
            + (f' — Prix: {p["price"]}' if p.get("price") and
               str(p["price"]).lower() not in ("not provided", "n/a", "", "none") else "")
            for p in batch
        )
        prompt = (
            f"Tu analyses les entrées de la base de données de promotions du restaurant \"{restaurant_name}\".\n\n"
            "Pour chaque entrée ci-dessous, détermine si c'est une vraie promotion ou non.\n\n"
            "Une VRAIE PROMOTION doit avoir au moins un de :\n"
            "  - Un rabais explicite (%, $, économi, save, gratuit, free)\n"
            "  - Un forfait bundle (combo/duo/trio/famille à prix groupé)\n"
            "  - Un spécial limité dans le temps explicitement présenté comme tel\n"
            "  - Un prix clairement présenté comme réduit ou spécial\n\n"
            "N'EST PAS une promotion :\n"
            "  - Un item du menu normal à prix standard (ex: 'Ailes de poulet 12.99$')\n"
            "  - Une description générique sans composante d'offre/réduction\n"
            "  - Du texte de navigation ou des titres de section\n\n"
            f"Entrées :\n{items_text}\n\n"
            "Réponds avec un JSON : {\"not_promos\": [liste des IDs qui ne sont PAS de vraies promotions]}\n"
            "Si toutes sont valides, réponds {\"not_promos\": []}."
        )
        try:
            client = OpenAIClient(api_key=OPENAI_API_KEY)
            resp = client.chat.completions.create(
                model=EXTRACT_MODEL_PRIMARY,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
                temperature=0,
                timeout=30,
            )
            raw = resp.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            result = json.loads(raw)
            bad_ids = [int(x) for x in result.get("not_promos", [])
                       if any(p["id"] == int(x) for p in batch)]
            to_deactivate.extend(bad_ids)
            logging.info(f"[Reverify] {restaurant_name} batch {i//BATCH+1}: "
                         f"{len(bad_ids)} non-promos détectées sur {len(batch)}")
        except Exception as exc:
            logging.error(f"[Reverify] LLM error for {restaurant_name} batch {i//BATCH+1}: {exc}")

    if to_deactivate:
        try:
            db = get_db(); cur = db.cursor()
            fmt = ",".join(["%s"] * len(to_deactivate))
            cur.execute(
                f"UPDATE promotions_table SET is_active=0 WHERE id IN ({fmt})",
                to_deactivate,
            )
            db.commit(); cur.close(); db.close()
            logging.info(f"[Reverify] {restaurant_name}: {len(to_deactivate)} non-promos désactivées")
        except Exception as exc:
            logging.error(f"[Reverify] DB write error for {restaurant_name}: {exc}")

    return {"checked": len(promos), "deactivated": len(to_deactivate)}


LLM_DEDUP_THRESHOLD = 0.75  # cosine sim minimum pour grouper des candidats doublon


def _llm_dedup_restaurant(restaurant_name: str) -> int:
    """
    Déduplication IA en 2 étapes :
      1. Clustering par embedding cosine >= LLM_DEDUP_THRESHOLD (union-find)
      2. Pour chaque cluster >= 2 promos, GPT-4o-mini décide lesquelles supprimer
    Retourne le nombre de promos désactivées.
    """
    if embedding_model is None or cosine_similarity is None:
        return 0
    try:
        db  = get_db()
        cur = db.cursor(dictionary=True)
        cur.execute(
            "SELECT id, promo_type, promo_details, price, embedding FROM promotions_table "
            "WHERE restaurant=%s AND is_active=1 AND embedding IS NOT NULL",
            (restaurant_name,))
        rows = cur.fetchall()
        if len(rows) < 2:
            cur.close(); db.close(); return 0

        embs = np.vstack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
        sims = cosine_similarity(embs)

        # Union-find clustering
        parent = list(range(len(rows)))
        def _find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]; x = parent[x]
            return x
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                if sims[i][j] >= LLM_DEDUP_THRESHOLD:
                    root_i, root_j = _find(i), _find(j)
                    if root_i != root_j:
                        parent[root_i] = root_j

        clusters: dict[int, list[int]] = {}
        for idx in range(len(rows)):
            clusters.setdefault(_find(idx), []).append(idx)

        to_deactivate: list[int] = []
        client = OpenAIClient(api_key=OPENAI_API_KEY)

        for cluster_indices in clusters.values():
            if len(cluster_indices) < 2:
                continue
            cluster_rows = [rows[i] for i in cluster_indices]
            items = "\n".join(
                f'{k+1}. [{r["promo_type"]}] {r["promo_details"]} | Prix: {r["price"] or "N/A"}'
                for k, r in enumerate(cluster_rows)
            )
            prompt = (
                "Tu es un expert en promotions de restaurant. "
                "Voici un groupe de promotions potentiellement similaires ou dupliquées :\n\n"
                f"{items}\n\n"
                "Réponds UNIQUEMENT avec les numéros (séparés par des virgules) des promotions "
                "qui sont de VRAIS DOUBLONS à supprimer (garde la plus complète/spécifique). "
                "Si aucune n'est un doublon, réponds 'aucun'."
            )
            try:
                resp = client.chat.completions.create(
                    model=EXTRACT_MODEL_PRIMARY,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=60, temperature=0)
                answer = resp.choices[0].message.content.strip().lower()
                if answer == "aucun":
                    continue
                for part in re.split(r"[,\s]+", answer):
                    part = part.strip()
                    if part.isdigit():
                        k = int(part) - 1
                        if 0 <= k < len(cluster_rows):
                            to_deactivate.append(cluster_rows[k]["id"])
            except Exception as e:
                logging.warning(f"[LLM-dedup] API error: {e}")

        if to_deactivate:
            fmt = ",".join(["%s"] * len(to_deactivate))
            cur.execute(
                f"UPDATE promotions_table SET is_active=0 WHERE id IN ({fmt})",
                to_deactivate)
            db.commit()
            logging.info(f"[LLM-dedup] {restaurant_name}: {len(to_deactivate)} doublons désactivés")

        cur.close(); db.close()
        return len(to_deactivate)
    except Exception as exc:
        logging.error(f"_llm_dedup_restaurant error: {exc}")
        return 0


def _llm_dedup_no_embeddings(restaurant_name: str) -> int:
    """
    Déduplication LLM sans embeddings — fonctionne même avec SKIP_EMBEDDING_INIT=1.

    Algorithme :
      1. Charge toutes les promos actives du restaurant
      2. Groupe par prix normalisé (ex: "$5", "$6", "$10.95", "" pour sans prix)
      3. Dans chaque groupe de prix, identifie les sous-groupes de candidats doublons
         via SequenceMatcher(ratio >= 0.45) — seuil très lâche exprès
      4. Envoie chaque sous-groupe (≥2) à GPT-4o-mini pour décision finale
      5. GPT retourne les numéros à SUPPRIMER (en gardant le plus complet)
    Retourne le nombre de promos désactivées.
    """
    if not OPENAI_API_KEY:
        logging.warning("[LLM-dedup-noemb] OpenAI key manquante — skip")
        return 0
    try:
        db = get_db(); cur = db.cursor(dictionary=True)
        cur.execute(
            "SELECT id, promo_type, promo_details, price, saved_date_time, last_seen "
            "FROM promotions_table WHERE restaurant=%s AND is_active=1",
            (restaurant_name,))
        rows = cur.fetchall()
        if len(rows) < 2:
            cur.close(); db.close(); return 0

        logging.info(f"[LLM-dedup-noemb] {restaurant_name}: {len(rows)} promos actives")

        # ── Normaliser le prix pour regroupement ──────────────────────────────
        def _norm_price(p):
            raw = (p.get("price") or "").strip().lower()
            if raw in ("", "not provided", "n/a", "none"): return "__nopr__"
            m = re.search(r"(\d+)[.,](\d{2})", raw)
            if m: return f"{m.group(1)}.{m.group(2)}"
            m2 = re.search(r"(\d+)", raw)
            return m2.group(1) if m2 else raw

        # ── Grouper par prix ──────────────────────────────────────────────────
        by_price: dict[str, list] = {}
        for r in rows:
            by_price.setdefault(_norm_price(r), []).append(r)

        # ── Dans chaque groupe de prix, trouver des sous-clusters candidats ───
        LLM_CANDIDATE_SIM = 0.45   # très lâche — LLM tranche ensuite
        all_clusters: list[list] = []

        for price_key, group in by_price.items():
            if len(group) < 2:
                continue
            # Union-Find pour clusteriser par similarité de texte
            ids = [g["id"] for g in group]
            parent = list(range(len(group)))
            def _find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]; x = parent[x]
                return x
            for i in range(len(group)):
                ni = _normalize_for_dedup(group[i].get("promo_details") or "")[:300]
                for j in range(i+1, len(group)):
                    nj = _normalize_for_dedup(group[j].get("promo_details") or "")[:300]
                    if not ni or not nj: continue
                    sim = SequenceMatcher(None, ni, nj).ratio()
                    # Also check containment
                    shorter, longer = (ni, nj) if len(ni) <= len(nj) else (nj, ni)
                    contained = bool(shorter and shorter in longer)
                    if sim >= LLM_CANDIDATE_SIM or contained:
                        ri, rj = _find(i), _find(j)
                        if ri != rj: parent[ri] = rj
            clusters: dict[int, list] = {}
            for idx in range(len(group)):
                clusters.setdefault(_find(idx), []).append(group[idx])
            for cluster in clusters.values():
                if len(cluster) >= 2:
                    all_clusters.append(cluster)

        if not all_clusters:
            logging.info(f"[LLM-dedup-noemb] {restaurant_name}: aucun cluster à analyser")
            cur.close(); db.close(); return 0

        logging.info(f"[LLM-dedup-noemb] {restaurant_name}: {len(all_clusters)} clusters à analyser via LLM")

        # ── Envoyer chaque cluster au LLM ─────────────────────────────────────
        client = OpenAIClient(api_key=OPENAI_API_KEY)
        to_deactivate: list[int] = []

        for cluster in all_clusters:
            items_text = "\n".join(
                f"{k+1}. [{r['promo_type'] or 'Other'}] {r['promo_details']} | Prix: {r['price'] or 'N/A'} | Vu le: {str(r.get('saved_date_time',''))[:10]}"
                for k, r in enumerate(cluster)
            )
            try:
                resp = client.chat.completions.create(
                    model=EXTRACT_MODEL_PRIMARY,
                    messages=[{
                        "role": "system",
                        "content": (
                            "Tu es un expert en promotions de restaurant québécois. "
                            "Ton rôle est d'identifier les doublons dans une liste de promotions."
                        )
                    }, {
                        "role": "user",
                        "content": (
                            f"Voici des promotions de '{restaurant_name}' qui semblent similaires :\n\n"
                            f"{items_text}\n\n"
                            "Ces promotions décrivent-elles essentiellement la MÊME offre (même deal, même prix, même concept) "
                            "exprimée différemment à des dates différentes ?\n\n"
                            "Si OUI : réponds avec les numéros des entrées à SUPPRIMER, séparés par des virgules "
                            "(garde la plus complète ou la plus récente — si toutes sont identiques, garde la #1). "
                            "Exemple de réponse : '2,3,4'\n"
                            "Si NON (promotions distinctes) : réponds exactement 'aucun'.\n\n"
                            "IMPORTANT : ne supprime PAS si les promotions sont clairement différentes "
                            "(ex: une pour enfants vs une pour adultes, ou des plats différents)."
                        )
                    }],
                    max_tokens=80,
                    temperature=0,
                )
                answer = resp.choices[0].message.content.strip().lower()
                if answer in ("aucun", "none", "no", "non"):
                    continue
                for part in re.split(r"[,\s]+", answer):
                    part = part.strip()
                    if part.isdigit():
                        k = int(part) - 1
                        if 0 <= k < len(cluster):
                            rid = cluster[k]["id"]
                            if rid not in to_deactivate:
                                to_deactivate.append(rid)
            except Exception as e:
                logging.warning(f"[LLM-dedup-noemb] API error on cluster: {e}")

        if to_deactivate:
            fmt = ",".join(["%s"] * len(to_deactivate))
            cur.execute(
                f"UPDATE promotions_table SET is_active=0 WHERE id IN ({fmt})",
                to_deactivate)
            db.commit()
            logging.info(f"[LLM-dedup-noemb] {restaurant_name}: {len(to_deactivate)} doublons désactivés par LLM")

        cur.close(); db.close()
        return len(to_deactivate)
    except Exception as exc:
        logging.error(f"_llm_dedup_no_embeddings error: {exc}")
        return 0


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


def _background_deep_clean(jid: str, restaurant_name: str):
    """Deep clean = nettoyage fuzzy standard + déduplication LLM sans embeddings."""
    _thread_context.job_id = jid
    _set_job(jid, "running")
    try:
        logging.info(f"[DeepClean] Job {jid} démarré pour {restaurant_name}")
        stats = clean_promos_sync(restaurant_name)    # Phase 1: fuzzy + non-promos
        llm_removed = _llm_dedup_no_embeddings(restaurant_name)  # Phase 2: LLM
        stats["llm_deduped"] = llm_removed
        stats["deep"] = True
        _set_job(jid, "done", result=stats)
        logging.info(f"[DeepClean] Job {jid} terminé: {stats}")
    except Exception as exc:
        logging.error(f"[DeepClean] Job {jid} erreur: {exc}")
        _set_job(jid, "error", error=str(exc))


AUTO_CLEAN_AFTER_SCRAPE = os.environ.get("AUTO_CLEAN_AFTER_SCRAPE", "1") == "1"

def _run_scrape_blocking(restaurant_name: str, url: str, jid: str = None) -> dict:
    promos, crawled, _ = _scrape_sync(url, restaurant_name, jid=jid)
    stats = save_promos_to_db(restaurant_name, promos)
    # N'archiver que si le scrape a réellement trouvé des promos :
    # si le scraper retourne 0 résultat, c'est probablement un échec réseau/site,
    # pas la disparition des promos → on ne touche pas à is_active.
    stats["marked_inactive"] = mark_inactive_promos(restaurant_name) if promos else 0
    stats["pages_crawled"]   = crawled
    if AUTO_CLEAN_AFTER_SCRAPE:
        stats["auto_cleaned"] = clean_promos_sync(restaurant_name)
    return stats

def _queue_promo_notification(rid: int, restaurant_name: str, new_promos: list):
    """Build an email notification for new promos and add it to the queue."""
    try:
        from models import get_setting, queue_notification
        if not new_promos:
            return
        count = len(new_promos)
        subject = f"🍗 {count} new promo{'s' if count != 1 else ''} at {restaurant_name}"
        html = render_template(
            "email/promo_notification.html",
            restaurant_name=restaurant_name,
            promos=new_promos[:8],  # cap at 8 in email
            promo_count=count,
        )
        # Auto-approve when admin approval is not required
        status = "pending" if get_setting("admin_approval_required", "0") == "1" else "approved"
        queue_notification(rid, restaurant_name, subject, html, count, status)
        logging.info(f"[Notifications] Queued notification for {restaurant_name} ({count} new promos, status={status})")
    except Exception as exc:
        logging.warning(f"[Notifications] Failed to queue notification for {restaurant_name}: {exc}")


def _send_approved_notifications():
    """Send all approved notification queue entries to eligible subscribers."""
    from models import get_approved_notifications, update_notification_status, get_restaurant_subscribers
    import requests as _req
    notifications = get_approved_notifications()
    if not notifications:
        return
    api_key = os.environ.get("MAIL_PASSWORD", "")
    sender = os.environ.get("MAIL_DEFAULT_SENDER", "noreply@chickenwings.local")
    if not api_key:
        logging.warning("[Notifications] MAIL_PASSWORD not set — skipping notification send")
        return
    for notif in notifications:
        rid = notif["restaurant_id"]
        if not rid:
            update_notification_status(notif["id"], "sent")
            continue
        subscribers = get_restaurant_subscribers(rid)
        if not subscribers:
            update_notification_status(notif["id"], "sent")
            continue
        sent_count = 0
        for sub in subscribers:
            try:
                resp = _req.post(
                    "https://api.resend.com/emails",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json={
                        "from": sender,
                        "to": [sub["email"]],
                        "subject": notif["subject"],
                        "html": notif["html_content"],
                    },
                    timeout=15,
                )
                if resp.status_code in (200, 201):
                    sent_count += 1
                else:
                    logging.warning(f"[Notifications] Resend error {resp.status_code} for {sub['email']}")
            except Exception as exc:
                logging.warning(f"[Notifications] Failed to send to {sub['email']}: {exc}")
        update_notification_status(notif["id"], "sent")
        logging.info(f"[Notifications] Sent notification '{notif['subject']}' to {sent_count}/{len(subscribers)} subscribers")


def _background_scrape(jid: str, restaurant_name: str, url: str, rid: int = None):
    _thread_context.job_id = jid
    _set_job(jid, "running")

    _result: list = [None]
    _error:  list = [None]
    _done = threading.Event()

    def _work():
        _thread_context.job_id = jid  # propagate job context to inner thread
        try:
            _result[0] = _scrape_sync(url, restaurant_name, jid=jid)
        except Exception as exc:
            _error[0] = exc
        finally:
            _done.set()

    threading.Thread(target=_work, daemon=True).start()

    if not _done.wait(timeout=SCRAPE_CIRCUIT_BREAKER_TIMEOUT):
        msg = (f"Circuit breaker: scrape for {restaurant_name} exceeded "
               f"{SCRAPE_CIRCUIT_BREAKER_TIMEOUT}s — aborting")
        logging.error(f"[CircuitBreaker] {msg}")
        _set_job(jid, "error", error=msg)
        return

    if _error[0] is not None:
        logging.error(f"Job {jid} error: {_error[0]}")
        _set_job(jid, "error", error=str(_error[0]))
        return

    promos, crawled, candidate_images = _result[0]
    try:
        stats = save_promos_to_db(restaurant_name, promos)
        stats["marked_inactive"]   = mark_inactive_promos(restaurant_name) if promos else 0
        stats["pages_crawled"]     = crawled
        stats["candidate_images"]  = candidate_images
        if rid is not None:
            stats["restaurant_id"] = rid
        if AUTO_CLEAN_AFTER_SCRAPE:
            cleaned = clean_promos_sync(restaurant_name)
            stats["auto_cleaned"] = cleaned
            removed = cleaned.get("non_promo_removed", 0) + cleaned.get("deduped", 0)
            stats["inserted"] = max(0, stats["inserted"] - removed)
        new_count = stats.get("inserted", 0)
        if new_count > 0 and rid is not None:
            threading.Thread(
                target=_queue_promo_notification,
                args=(rid, restaurant_name, promos[:new_count]),
                daemon=True,
            ).start()
        _set_job(jid, "done", result=stats, pages=crawled)
        logging.info(f"Job {jid} finished for {restaurant_name}")
    except Exception as exc:
        logging.error(f"Job {jid} post-scrape error: {exc}")
        _set_job(jid, "error", error=str(exc))


app = Flask(__name__)

# Secret key (required for sessions & itsdangerous tokens)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-change-in-production")

# Flask-Login
login_manager = LoginManager(app)
login_manager.login_view = "auth.login"
login_manager.login_message = ""  # suppress default flash message

from models import get_user_by_id as _get_user_by_id  # noqa: E402

@login_manager.user_loader
def _user_loader(user_id):
    return _get_user_by_id(int(user_id))

# Flask-Mail
app.config.update(
    MAIL_SERVER   = os.environ.get("MAIL_SERVER", ""),
    MAIL_PORT     = int(os.environ.get("MAIL_PORT", "587")),
    MAIL_USE_TLS  = os.environ.get("MAIL_USE_TLS", "true").lower() == "true",
    MAIL_USERNAME = os.environ.get("MAIL_USERNAME", ""),
    MAIL_PASSWORD = os.environ.get("MAIL_PASSWORD", ""),
    MAIL_DEFAULT_SENDER = os.environ.get("MAIL_DEFAULT_SENDER", "noreply@chickenwings.local"),
)
if app.config["MAIL_SERVER"]:
    mail = Mail(app)
    app.extensions["mail"] = mail

# OAuth SSO
from oauth_client import oauth as _oauth  # noqa: E402
_oauth.init_app(app)
_oauth.register(
    "google",
    client_id     = os.environ.get("GOOGLE_CLIENT_ID", ""),
    client_secret = os.environ.get("GOOGLE_CLIENT_SECRET", ""),
    server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs = {"scope": "openid email profile"},
)
_oauth.register(
    "github",
    client_id          = os.environ.get("GITHUB_CLIENT_ID", ""),
    client_secret      = os.environ.get("GITHUB_CLIENT_SECRET", ""),
    api_base_url       = "https://api.github.com/",
    access_token_url   = "https://github.com/login/oauth/access_token",
    authorize_url      = "https://github.com/login/oauth/authorize",
    client_kwargs      = {"scope": "user:email"},
)

# Blueprints
from auth import auth as auth_blueprint    # noqa: E402
from admin import admin as admin_blueprint  # noqa: E402
from decorators import admin_required       # noqa: E402
app.register_blueprint(auth_blueprint)
app.register_blueprint(admin_blueprint)

# Expose _jobs to admin blueprint via app attribute
app.jobs = _jobs

# Require login for all non-auth, non-static, non-ping routes
@app.before_request
def _require_login():
    open_prefixes = ("auth.", "static")
    open_endpoints = ("ping",)
    ep = request.endpoint or ""
    if any(ep.startswith(p) for p in open_prefixes) or ep in open_endpoints:
        return None
    if not current_user.is_authenticated:
        return redirect(url_for("auth.login", next=request.url))


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
        # Send approved notifications every Monday at 9:00 AM
        scheduler.add_job(_send_approved_notifications, "cron",
                          day_of_week="mon", hour=9, minute=0,
                          id="send_notifications", replace_existing=True)
        scheduler.start()
        logging.info(f"[Scheduler] Auto-scrape every {SCRAPE_INTERVAL_HOURS}h – started.")
        logging.info("[Scheduler] Notification digest every Monday 09:00 – started.")
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
@admin_required
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
@admin_required
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
@admin_required
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
@admin_required
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
@admin_required
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


@app.route("/api/admin/deep-clean/<int:rid>", methods=["POST"])
@admin_required
def deep_clean_restaurant(rid):
    """Deep clean : nettoyage fuzzy + déduplication LLM sans embeddings."""
    try:
        db = get_db(); cur = db.cursor(dictionary=True)
        cur.execute("SELECT * FROM restaurants WHERE id=%s", (rid,))
        rest = cur.fetchone(); cur.close(); db.close()
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    if not rest:
        return jsonify({"error": "Restaurant not found"}), 404
    jid = _new_job(f"[DeepClean] {rest['name']}")
    threading.Thread(target=_background_deep_clean,
                     args=(jid, rest["name"]), daemon=True).start()
    return jsonify({"job_id": jid, "name": rest["name"]}), 202


@app.route("/api/admin/deep-clean-all", methods=["POST"])
@admin_required
def deep_clean_all():
    """Deep clean de tous les restaurants séquentiellement dans un seul job."""
    try:
        db = get_db(); cur = db.cursor(dictionary=True)
        cur.execute("SELECT id, name FROM restaurants ORDER BY name")
        rests = cur.fetchall(); cur.close(); db.close()
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    jid = _new_job("[DeepClean-ALL]")

    def _run_all(jid, rests):
        _thread_context.job_id = jid
        _set_job(jid, "running")
        total = {"non_promo_removed": 0, "deduped": 0, "llm_deduped": 0}
        try:
            for rest in rests:
                logging.info(f"[DeepClean-ALL] Traitement de {rest['name']}")
                stats = clean_promos_sync(rest["name"])
                llm_n = _llm_dedup_no_embeddings(rest["name"])
                total["non_promo_removed"] += stats.get("non_promo_removed", 0)
                total["deduped"]           += stats.get("deduped", 0)
                total["llm_deduped"]       += llm_n
            _set_job(jid, "done", result=total)
            logging.info(f"[DeepClean-ALL] Terminé: {total}")
        except Exception as exc:
            logging.error(f"[DeepClean-ALL] Erreur: {exc}")
            _set_job(jid, "error", error=str(exc))

    threading.Thread(target=_run_all, args=(jid, rests), daemon=True).start()
    return jsonify({"job_id": jid, "restaurants": len(rests)}), 202


@app.route("/clean/<int:rid>", methods=["POST"])
@admin_required
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
@admin_required
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
@admin_required
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
@admin_required
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
@admin_required
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

# Simple in-memory cache for analytics (5-minute TTL)
import time as _time_mod
_analytics_cache: dict = {"data": None, "expires": 0.0}
_analytics_cache_lock = threading.Lock()


@app.route("/analytics")
def analytics():
    return render_template("analytics.html")


@app.route("/api/analytics/cache/invalidate", methods=["POST"])
def api_analytics_invalidate():
    with _analytics_cache_lock:
        _analytics_cache["expires"] = 0.0
    return jsonify({"ok": True})


@app.route("/api/analytics/stats")
def api_analytics_stats():
    now = _time_mod.time()
    with _analytics_cache_lock:
        if _analytics_cache["data"] and now < _analytics_cache["expires"]:
            return jsonify(_analytics_cache["data"])
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

        # Weekly trend (last 8 weeks) for finer granularity
        cur.execute("""SELECT DATE_FORMAT(saved_date_time,'%Y-%u') as week,
            MIN(DATE(saved_date_time)) as week_start, COUNT(*) as count
            FROM promotions_table
            WHERE saved_date_time >= DATE_SUB(NOW(), INTERVAL 8 WEEK)
            GROUP BY week ORDER BY week ASC""")
        weekly_trend = cur.fetchall()
        for w in weekly_trend:
            if w.get("week_start"):
                w["week_start"] = str(w["week_start"])

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
            savings_estimate, category, link
            FROM promotions_table WHERE is_active=1 AND savings_estimate IS NOT NULL
            ORDER BY savings_estimate DESC LIMIT 10""")
        top_savings = cur.fetchall()

        # Price range distribution (bucketed)
        cur.execute("""SELECT
            CASE
                WHEN CAST(REGEXP_REPLACE(price, '[^0-9.]','') AS DECIMAL(10,2)) < 5  THEN 'Under $5'
                WHEN CAST(REGEXP_REPLACE(price, '[^0-9.]','') AS DECIMAL(10,2)) < 10 THEN '$5–$10'
                WHEN CAST(REGEXP_REPLACE(price, '[^0-9.]','') AS DECIMAL(10,2)) < 15 THEN '$10–$15'
                WHEN CAST(REGEXP_REPLACE(price, '[^0-9.]','') AS DECIMAL(10,2)) < 20 THEN '$15–$20'
                ELSE '$20+'
            END as bucket, COUNT(*) as count
            FROM promotions_table WHERE is_active=1 AND price IS NOT NULL AND price != ''
              AND price NOT IN ('not provided','N/A')
            GROUP BY bucket ORDER BY MIN(CAST(REGEXP_REPLACE(price,'[^0-9.]','') AS DECIMAL(10,2)))""")
        price_dist = cur.fetchall()

        # Recent activity: last 5 scrape events (newest promos)
        cur.execute("""SELECT restaurant, promo_type, promo_details, grade, price,
            saved_date_time, link
            FROM promotions_table
            ORDER BY saved_date_time DESC LIMIT 8""")
        recent_activity = cur.fetchall()
        for r in recent_activity:
            if r.get("saved_date_time"):
                r["saved_date_time"] = str(r["saved_date_time"])

        # Insights: price promo (explicit numeric price) vs branding (no price)
        cur.execute("""SELECT
            SUM(CASE WHEN price IS NOT NULL AND price != ''
                 AND LOWER(price) NOT IN ('not provided','n/a')
                 AND price REGEXP '^[0-9]' THEN 1 ELSE 0 END) AS price_promos,
            SUM(CASE WHEN price IS NULL OR price = ''
                 OR LOWER(price) IN ('not provided','n/a')
                 OR NOT price REGEXP '^[0-9]' THEN 1 ELSE 0 END) AS branding_promos
            FROM promotions_table WHERE is_active=1""")
        insights_split = cur.fetchone()

        # Monthly pivot: promos per restaurant per month (last 12 months)
        cur.execute("""SELECT restaurant, DATE_FORMAT(saved_date_time,'%Y-%m') AS month,
            COUNT(*) AS cnt
            FROM promotions_table
            WHERE saved_date_time >= DATE_SUB(NOW(), INTERVAL 12 MONTH)
            GROUP BY restaurant, month
            ORDER BY month ASC, restaurant ASC""")
        pivot_data = cur.fetchall()

        cur.close(); db.close()
        payload = {
            "grade_dist": grade_dist,
            "cat_dist": cat_dist,
            "rest_comparison": rest_comparison,
            "timeline": timeline,
            "weekly_trend": weekly_trend,
            "savings_by_cat": savings_by_cat,
            "active_ratio": active_ratio,
            "top_savings": top_savings,
            "price_dist": price_dist,
            "recent_activity": recent_activity,
            "insights_split": insights_split,
            "pivot_data": pivot_data,
            "cached_at": datetime.now().isoformat(),
        }
        with _analytics_cache_lock:
            _analytics_cache["data"] = payload
            _analytics_cache["expires"] = _time_mod.time() + 300  # 5-minute TTL
        return jsonify(payload)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ─── Marketing Study Module ───────────────────────────────────────────────────

@app.route("/marketing")
@login_required
def marketing():
    return render_template("marketing.html")


@app.route("/api/marketing/compare")
@login_required
def api_marketing_compare():
    """Return per-restaurant price data for cross-restaurant comparison."""
    try:
        db = get_db(); cur = db.cursor(dictionary=True)
        cur.execute("""
            SELECT restaurant,
                ROUND(AVG(CASE WHEN price IS NOT NULL AND price NOT IN ('','not provided','N/A')
                    THEN CAST(REGEXP_REPLACE(price,'[^0-9.]','') AS DECIMAL(10,2)) END), 2) as avg_price,
                ROUND(MIN(CASE WHEN price IS NOT NULL AND price NOT IN ('','not provided','N/A')
                    THEN CAST(REGEXP_REPLACE(price,'[^0-9.]','') AS DECIMAL(10,2)) END), 2) as min_price,
                ROUND(MAX(CASE WHEN price IS NOT NULL AND price NOT IN ('','not provided','N/A')
                    THEN CAST(REGEXP_REPLACE(price,'[^0-9.]','') AS DECIMAL(10,2)) END), 2) as max_price,
                COUNT(*) as total_promos,
                SUM(is_active=1) as active_promos,
                SUM(CASE WHEN grade='A+' THEN 1 ELSE 0 END) as aplus_count,
                SUM(CASE WHEN grade IN ('A+','A') THEN 1 ELSE 0 END) as top_grade_count,
                ROUND(AVG(savings_estimate),2) as avg_savings
            FROM promotions_table
            WHERE price IS NOT NULL AND price NOT IN ('','not provided','N/A')
            GROUP BY restaurant ORDER BY avg_price ASC""")
        restaurants = cur.fetchall()

        # Per-category breakdown per restaurant
        cur.execute("""
            SELECT restaurant, COALESCE(category,'Autre') as category,
                COUNT(*) as count,
                ROUND(AVG(CASE WHEN price NOT IN ('','not provided','N/A')
                    THEN CAST(REGEXP_REPLACE(price,'[^0-9.]','') AS DECIMAL(10,2)) END),2) as avg_price,
                ROUND(AVG(savings_estimate),2) as avg_savings
            FROM promotions_table WHERE is_active=1
            GROUP BY restaurant, category ORDER BY restaurant, count DESC""")
        by_category = cur.fetchall()

        # Lowest price active promos per restaurant (top value deals)
        cur.execute("""
            SELECT p.restaurant, p.promo_type, p.promo_details, p.price, p.grade,
                p.savings_estimate, p.category, p.link,
                CAST(REGEXP_REPLACE(p.price,'[^0-9.]','') AS DECIMAL(10,2)) as price_num
            FROM promotions_table p
            WHERE p.is_active=1 AND p.price NOT IN ('','not provided','N/A')
              AND p.price IS NOT NULL
            ORDER BY price_num ASC LIMIT 30""")
        best_deals = cur.fetchall()

        cur.close(); db.close()
        return jsonify({
            "restaurants": restaurants,
            "by_category": by_category,
            "best_deals": best_deals,
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/marketing/analyze", methods=["POST"])
@login_required
def api_marketing_analyze():
    """Use LLM to analyze price gaps and generate promotion recommendations."""
    if not OPENAI_API_KEY:
        return jsonify({"error": "OpenAI API key not configured"}), 503
    try:
        db = get_db(); cur = db.cursor(dictionary=True)
        cur.execute("""
            SELECT restaurant,
                ROUND(AVG(CASE WHEN price NOT IN ('','not provided','N/A')
                    THEN CAST(REGEXP_REPLACE(price,'[^0-9.]','') AS DECIMAL(10,2)) END),2) as avg_price,
                ROUND(MIN(CASE WHEN price NOT IN ('','not provided','N/A')
                    THEN CAST(REGEXP_REPLACE(price,'[^0-9.]','') AS DECIMAL(10,2)) END),2) as min_price,
                COUNT(*) as total_promos, SUM(is_active=1) as active_promos
            FROM promotions_table
            WHERE price IS NOT NULL AND price NOT IN ('','not provided','N/A')
            GROUP BY restaurant ORDER BY avg_price ASC""")
        rest_data = cur.fetchall()

        cur.execute("""
            SELECT restaurant, promo_type, promo_details, price, grade, savings_estimate, category
            FROM promotions_table WHERE is_active=1 AND grade IN ('A+','A')
            ORDER BY savings_estimate DESC LIMIT 20""")
        top_promos = cur.fetchall()
        cur.close(); db.close()

        data_summary = json.dumps({
            "restaurants": rest_data,
            "top_active_promos": top_promos,
        }, ensure_ascii=False, default=str)

        client = OpenAIClient(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model=EXTRACT_MODEL_PRIMARY,
            response_format={"type": "json_object"},
            messages=[{
                "role": "system",
                "content": (
                    "You are a restaurant marketing analyst. "
                    "Analyze competitive pricing data and return actionable JSON recommendations."
                )
            }, {
                "role": "user",
                "content": f"""Analyze this restaurant promotion data and identify price gaps.

DATA:
{data_summary}

Return a JSON object with these exact keys:
{{
  "market_leader": "restaurant with lowest avg price",
  "price_gap_summary": "2-3 sentence summary of price landscape",
  "recommendations": [
    {{
      "restaurant": "restaurant name",
      "issue": "brief description of pricing weakness",
      "action": "specific promotion or discount to implement",
      "suggested_price": "$X.XX",
      "expected_impact": "brief expected outcome"
    }}
  ],
  "quick_wins": ["list of 3-5 specific immediate actions"],
  "market_insights": ["list of 3-5 key insights about the competitive landscape"]
}}

Provide 3-5 concrete recommendations targeting restaurants with higher prices than competitors."""
            }],
            max_tokens=1200,
            temperature=0.4,
        )
        result = json.loads(resp.choices[0].message.content)
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ─── Google Reviews Module ────────────────────────────────────────────────────

@app.route("/reviews")
@login_required
def reviews():
    return render_template("reviews.html", google_maps_api_key=GOOGLE_MAPS_API_KEY)


@app.route("/api/reviews/search")
@login_required
def api_reviews_search():
    """Search for restaurant locations using Google Places Text Search API."""
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify({"error": "Query required"}), 400
    if not GOOGLE_MAPS_API_KEY:
        return jsonify({"error": "Google Maps API key not configured"}), 503
    try:
        resp = _requests_lib.get(
            "https://maps.googleapis.com/maps/api/place/textsearch/json",
            params={"query": query, "type": "restaurant", "key": GOOGLE_MAPS_API_KEY},
            timeout=10,
        )
        data = resp.json()
        if data.get("status") not in ("OK", "ZERO_RESULTS"):
            return jsonify({"error": f"Places API error: {data.get('status')}"}), 502
        results = []
        for p in data.get("results", [])[:10]:
            results.append({
                "place_id": p.get("place_id"),
                "name": p.get("name"),
                "address": p.get("formatted_address"),
                "rating": p.get("rating"),
                "user_ratings_total": p.get("user_ratings_total"),
                "lat": p["geometry"]["location"]["lat"],
                "lng": p["geometry"]["location"]["lng"],
                "types": p.get("types", []),
            })
        return jsonify({"results": results})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/reviews/details")
@login_required
def api_reviews_details():
    """Fetch reviews for a place and generate an LLM summary."""
    place_id = request.args.get("place_id", "").strip()
    if not place_id:
        return jsonify({"error": "place_id required"}), 400
    if not GOOGLE_MAPS_API_KEY:
        return jsonify({"error": "Google Maps API key not configured"}), 503
    try:
        resp = _requests_lib.get(
            "https://maps.googleapis.com/maps/api/place/details/json",
            params={
                "place_id": place_id,
                "fields": "name,rating,user_ratings_total,reviews,formatted_address,url,opening_hours,price_level",
                "key": GOOGLE_MAPS_API_KEY,
                "language": "fr",
            },
            timeout=10,
        )
        data = resp.json()
        if data.get("status") != "OK":
            return jsonify({"error": f"Places API error: {data.get('status')}"}), 502

        result = data.get("result", {})
        reviews_raw = result.get("reviews", [])
        reviews = []
        for r in reviews_raw:
            reviews.append({
                "author": r.get("author_name"),
                "rating": r.get("rating"),
                "text": r.get("text", ""),
                "time": r.get("relative_time_description"),
                "profile_photo": r.get("profile_photo_url"),
            })

        place_info = {
            "name": result.get("name"),
            "address": result.get("formatted_address"),
            "rating": result.get("rating"),
            "user_ratings_total": result.get("user_ratings_total"),
            "url": result.get("url"),
            "price_level": result.get("price_level"),
            "opening_hours": result.get("opening_hours", {}).get("weekday_text", []),
            "reviews": reviews,
        }

        # LLM summary if we have reviews and OpenAI key
        llm_summary = None
        if reviews and OPENAI_API_KEY:
            try:
                reviews_text = "\n".join([
                    f"[{r['rating']}/5] {r['author']}: {r['text'][:400]}"
                    for r in reviews if r.get("text")
                ])
                client = OpenAIClient(api_key=OPENAI_API_KEY)
                llm_resp = client.chat.completions.create(
                    model=EXTRACT_MODEL_PRIMARY,
                    response_format={"type": "json_object"},
                    messages=[{
                        "role": "system",
                        "content": "You are a restaurant review analyst. Summarize customer feedback concisely."
                    }, {
                        "role": "user",
                        "content": f"""Analyze these Google reviews for {result.get('name')} and return JSON:

REVIEWS:
{reviews_text}

Return JSON with these exact keys:
{{
  "overall_sentiment": "Positif / Mitigé / Négatif",
  "best_aspects": ["list of 3 top praised aspects"],
  "worst_aspects": ["list of 3 most criticized aspects"],
  "summary_fr": "2-3 sentence summary in French",
  "summary_en": "2-3 sentence summary in English",
  "highlights": [
    {{"type": "positive", "text": "specific positive quote or paraphrase"}},
    {{"type": "negative", "text": "specific negative quote or paraphrase"}}
  ],
  "recommendation": "One-sentence recommendation for the restaurant owner"
}}"""
                    }],
                    max_tokens=700,
                    temperature=0.3,
                )
                llm_summary = json.loads(llm_resp.choices[0].message.content)
            except Exception as llm_exc:
                logging.warning(f"LLM review summary failed: {llm_exc}")

        place_info["llm_summary"] = llm_summary
        return jsonify(place_info)
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
@admin_required
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


@app.route("/api/reverify-all", methods=["POST"])
@admin_required
def api_reverify_all():
    """
    Lance en arrière-plan la re-vérification LLM de toutes les promos actives
    pour chaque restaurant : détecte et désactive les non-promotions (items
    menu normaux, texte de navigation, descriptions génériques).
    Combine le nettoyage rule-based (clean_promos_sync) et la vérification LLM
    (_llm_reverify_restaurant) pour chaque restaurant.
    """
    def _run():
        try:
            db = get_db(); cur = db.cursor(dictionary=True)
            cur.execute("SELECT name FROM restaurants")
            restaurants = [r["name"] for r in cur.fetchall()]
            cur.close(); db.close()
        except Exception as exc:
            logging.error(f"[reverify-all] DB error: {exc}"); return

        total_deactivated = 0
        for name in restaurants:
            logging.info(f"[reverify-all] Processing {name}…")
            clean_stats = clean_promos_sync(name)
            llm_stats   = _llm_reverify_restaurant(name)
            deactivated = (clean_stats.get("non_promo_removed", 0) +
                           clean_stats.get("deduped", 0) +
                           llm_stats.get("deactivated", 0))
            total_deactivated += deactivated
            logging.info(f"[reverify-all] {name}: rule={clean_stats}, llm={llm_stats}")

        logging.info(f"[reverify-all] Done. Total désactivées : {total_deactivated}")

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return jsonify({"status": "started", "message": "Re-vérification lancée en arrière-plan"})


@app.errorhandler(403)
def forbidden(e):
    return render_template("errors/403.html"), 403


@app.errorhandler(Exception)
def handle_error(err):
    from werkzeug.exceptions import HTTPException
    if isinstance(err, HTTPException): return err
    logging.exception("Unhandled error: %s", err)
    return jsonify({"error": "internal-server-error", "detail": str(err)}), 500


# ---------------------------------------------------------------------------
# User dashboard
# ---------------------------------------------------------------------------

@app.route("/dashboard")
@login_required
def user_dashboard():
    try:
        db = get_db(); cur = db.cursor(dictionary=True)
        cur.execute("SELECT id, name FROM restaurants ORDER BY name")
        restaurants = cur.fetchall()
        cur.close(); db.close()
    except Exception:
        restaurants = []
    from models import get_user_subscriptions
    user_subs = {s["restaurant_id"]: s["frequency"] for s in get_user_subscriptions(current_user.id)}
    return render_template("dashboard.html", restaurants=restaurants, user_subs=user_subs)


# ---------------------------------------------------------------------------
# Admin API endpoints
# ---------------------------------------------------------------------------

@app.route("/api/admin/stats")
@admin_required
def api_admin_stats():
    from models import get_user_count
    try:
        db = get_db(); cur = db.cursor(dictionary=True)
        cur.execute("SELECT COUNT(*) AS total FROM promotions_table")
        total_promos = (cur.fetchone() or {}).get("total", 0)
        cur.execute("SELECT COUNT(*) AS active FROM promotions_table WHERE is_active=1")
        active_promos = (cur.fetchone() or {}).get("active", 0)
        cur.execute("SELECT COUNT(*) AS cnt FROM restaurants")
        restaurants = (cur.fetchone() or {}).get("cnt", 0)
        cur.close(); db.close()
    except Exception as exc:
        logging.error(f"admin_stats: {exc}")
        total_promos = active_promos = restaurants = 0

    from models import get_activity_logs
    recent_logs = get_activity_logs(limit=20)

    # Convert datetime objects to strings for JSON
    for log in recent_logs:
        for k, v in log.items():
            if hasattr(v, 'isoformat'):
                log[k] = v.isoformat()

    user_counts = get_user_count()
    return jsonify({
        "users":        user_counts,
        "promos":       {"total": total_promos, "active": active_promos},
        "restaurants":  restaurants,
        "recent_logs":  recent_logs,
    })


@app.route("/api/admin/users")
@admin_required
def api_admin_users():
    from models import get_all_users
    users = get_all_users()
    for u in users:
        for k, v in u.items():
            if hasattr(v, 'isoformat'):
                u[k] = v.isoformat()
    return jsonify({"users": users})


@app.route("/api/admin/users/<int:uid>/role", methods=["POST"])
@admin_required
def api_admin_set_role(uid):
    from models import set_user_role, log_activity
    data = request.get_json(force=True) or {}
    role = data.get("role", "user")
    ok = set_user_role(uid, role)
    if ok:
        log_activity(current_user.id, current_user.email,
                     "admin_set_role", f"uid={uid} role={role}", request.remote_addr)
    return jsonify({"ok": ok})


@app.route("/api/admin/users/<int:uid>", methods=["DELETE"])
@admin_required
def api_admin_delete_user(uid):
    from models import delete_user as _delete_user, log_activity
    if uid == current_user.id:
        return jsonify({"ok": False, "error": "Cannot delete yourself"}), 400
    ok = _delete_user(uid)
    if ok:
        log_activity(current_user.id, current_user.email,
                     "admin_delete_user", f"uid={uid}", request.remote_addr)
    return jsonify({"ok": ok})


@app.route("/api/admin/logs")
@admin_required
def api_admin_logs():
    from models import get_activity_logs
    logs = get_activity_logs(limit=500)
    for l in logs:
        for k, v in l.items():
            if hasattr(v, 'isoformat'):
                l[k] = v.isoformat()
    return jsonify({"logs": logs})


@app.route("/api/admin/jobs")
@admin_required
def api_admin_jobs():
    with _jobs_lock:
        jobs = [{"id": jid, **{k: v for k, v in data.items() if k != "logs"}}
                for jid, data in _jobs.items()]
    jobs.sort(key=lambda j: j.get("started_at", ""), reverse=True)
    return jsonify({"jobs": jobs[:100]})


@app.route("/api/admin/restaurants")
@admin_required
def api_admin_restaurants():
    try:
        db = get_db(); cur = db.cursor(dictionary=True)
        cur.execute("""
            SELECT r.id, r.name, r.url,
                   COUNT(p.id) AS total_promos,
                   SUM(CASE WHEN p.is_active=1 THEN 1 ELSE 0 END) AS active_promos,
                   SUM(CASE WHEN p.grade IN ('A+','A') AND p.is_active=1 THEN 1 ELSE 0 END) AS top_promos,
                   MAX(p.last_seen) AS last_scraped
            FROM restaurants r
            LEFT JOIN promotions_table p ON p.restaurant = r.name
            GROUP BY r.id, r.name, r.url
            ORDER BY r.name
        """)
        rests = cur.fetchall()
        cur.close(); db.close()
        for r in rests:
            for k, v in r.items():
                if hasattr(v, 'isoformat'):
                    r[k] = v.isoformat()
        return jsonify({"restaurants": rests})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/admin/subscribers")
@admin_required
def api_admin_subscribers():
    from models import get_subscribers, get_user_count
    subs = get_subscribers()
    for s in subs:
        for k, v in s.items():
            if hasattr(v, 'isoformat'):
                s[k] = v.isoformat()
    return jsonify({"subscribers": subs, "stats": get_user_count()})


# ---------------------------------------------------------------------------
# Admin – Notifications
# ---------------------------------------------------------------------------

@app.route("/api/admin/notifications")
@admin_required
def api_admin_notifications():
    from models import get_notifications
    return jsonify({"notifications": get_notifications(limit=200)})


@app.route("/api/admin/notifications/<int:nid>/approve", methods=["POST"])
@admin_required
def api_approve_notification(nid: int):
    from models import update_notification_status
    ok = update_notification_status(nid, "approved")
    if ok:
        log_activity(current_user.id, current_user.email, "notification_approved",
                     f"nid={nid}", request.remote_addr)
    return jsonify({"ok": ok})


@app.route("/api/admin/notifications/<int:nid>/reject", methods=["POST"])
@admin_required
def api_reject_notification(nid: int):
    from models import update_notification_status
    ok = update_notification_status(nid, "rejected")
    if ok:
        log_activity(current_user.id, current_user.email, "notification_rejected",
                     f"nid={nid}", request.remote_addr)
    return jsonify({"ok": ok})


@app.route("/api/admin/notifications/send-now", methods=["POST"])
@admin_required
def api_send_notifications_now():
    """Immediately process all approved notifications."""
    threading.Thread(target=_send_approved_notifications, daemon=True).start()
    log_activity(current_user.id, current_user.email, "notifications_sent_now", "", request.remote_addr)
    return jsonify({"ok": True, "message": "Sending in background…"})


# ---------------------------------------------------------------------------
# Admin – Settings (robots.txt toggle, approval mode)
# ---------------------------------------------------------------------------

@app.route("/api/admin/settings", methods=["GET"])
@admin_required
def api_get_settings():
    from models import get_all_settings
    defaults = {"robots_txt_check_enabled": "0", "admin_approval_required": "0"}
    settings = get_all_settings()
    defaults.update(settings)
    return jsonify({"settings": defaults})


@app.route("/api/admin/settings", methods=["POST"])
@admin_required
def api_update_settings():
    from models import set_setting
    data = request.get_json(force=True) or {}
    allowed_keys = {"robots_txt_check_enabled", "admin_approval_required"}
    updated = {}
    for key in allowed_keys:
        if key in data:
            val = "1" if data[key] in (True, "1", 1) else "0"
            set_setting(key, val)
            updated[key] = val
    log_activity(current_user.id, current_user.email, "settings_updated",
                 str(updated), request.remote_addr)
    return jsonify({"ok": True, "updated": updated})


# ---------------------------------------------------------------------------
# Restaurant subscriptions (user-facing API)
# ---------------------------------------------------------------------------

@app.route("/api/subscriptions")
@login_required
def api_get_subscriptions():
    from models import get_user_subscriptions
    subs = get_user_subscriptions(current_user.id)
    return jsonify({"subscriptions": {s["restaurant_id"]: s["frequency"] for s in subs}})


@app.route("/api/subscriptions/<int:rid>", methods=["POST"])
@login_required
def api_toggle_subscription(rid: int):
    from models import subscribe_restaurant, unsubscribe_restaurant, log_activity as _log
    data = request.get_json(force=True) or {}
    subscribed = bool(data.get("subscribed", False))
    frequency = data.get("frequency", "weekly")
    if frequency not in ("instant", "weekly", "monthly"):
        frequency = "weekly"
    if subscribed:
        subscribe_restaurant(current_user.id, rid, frequency)
        action = "restaurant_subscribed"
    else:
        unsubscribe_restaurant(current_user.id, rid)
        action = "restaurant_unsubscribed"
    from models import log_activity
    log_activity(current_user.id, current_user.email, action,
                 f"rid={rid},freq={frequency}", request.remote_addr)
    return jsonify({"ok": True})


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
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id                   INT AUTO_INCREMENT PRIMARY KEY,
                    email                VARCHAR(255) NOT NULL UNIQUE,
                    password_hash        VARCHAR(255) NOT NULL,
                    first_name           VARCHAR(100),
                    last_name            VARCHAR(100),
                    role                 ENUM('user','admin') DEFAULT 'user',
                    is_verified          TINYINT(1) DEFAULT 0,
                    newsletter_subscribed TINYINT(1) DEFAULT 1,
                    unsubscribe_token    VARCHAR(100),
                    created_at           DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_login           DATETIME,
                    email_verified_at    DATETIME
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS activity_logs (
                    id          INT AUTO_INCREMENT PRIMARY KEY,
                    user_id     INT,
                    user_email  VARCHAR(255),
                    action      VARCHAR(100) NOT NULL,
                    details     TEXT,
                    ip_address  VARCHAR(50),
                    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS app_settings (
                    `key`       VARCHAR(100) PRIMARY KEY,
                    value       TEXT,
                    updated_at  DATETIME DEFAULT CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS restaurant_subscriptions (
                    id            INT AUTO_INCREMENT PRIMARY KEY,
                    user_id       INT NOT NULL,
                    restaurant_id INT NOT NULL,
                    frequency     ENUM('instant','weekly','monthly') DEFAULT 'weekly',
                    created_at    DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE KEY uq_user_rest (user_id, restaurant_id),
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY (restaurant_id) REFERENCES restaurants(id) ON DELETE CASCADE
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS notification_queue (
                    id              INT AUTO_INCREMENT PRIMARY KEY,
                    restaurant_id   INT,
                    restaurant_name VARCHAR(255),
                    subject         VARCHAR(500),
                    html_content    LONGTEXT,
                    promo_count     INT DEFAULT 0,
                    status          ENUM('pending','approved','rejected','sent') DEFAULT 'pending',
                    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
                    sent_at         DATETIME,
                    FOREIGN KEY (restaurant_id) REFERENCES restaurants(id) ON DELETE SET NULL
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)
            # Performance indexes (CREATE INDEX IF NOT EXISTS requires MySQL 8.0+)
            _indexes = [
                ("idx_promos_active",     "promotions_table", "is_active"),
                ("idx_promos_saved",      "promotions_table", "saved_date_time"),
                ("idx_promos_restaurant", "promotions_table", "restaurant(100)"),
                ("idx_promos_grade",      "promotions_table", "grade"),
                ("idx_promos_category",   "promotions_table", "category(50)"),
            ]
            for idx_name, tbl, cols in _indexes:
                try:
                    cur.execute(f"CREATE INDEX {idx_name} ON {tbl}({cols})")
                except mysql.connector.Error:
                    pass  # index already exists

            # Migrations: add OAuth columns if not present
            for col, ddl in [
                ("oauth_provider", "ALTER TABLE users ADD COLUMN oauth_provider VARCHAR(50)"),
                ("oauth_id",       "ALTER TABLE users ADD COLUMN oauth_id VARCHAR(255)"),
            ]:
                try:
                    cur.execute(ddl)
                except mysql.connector.Error:
                    pass  # column already exists
            try:
                cur.execute("ALTER TABLE users MODIFY COLUMN password_hash VARCHAR(255) NULL")
            except mysql.connector.Error:
                pass

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
    cfg.trusted_hosts = ["*"]
    logging.info("Starting Promo Dashboard on http://0.0.0.0:5000")
    try:
        asyncio.run(hypercorn.asyncio.serve(app, cfg))
    finally:
        if _scheduler: _scheduler.shutdown(wait=False)
