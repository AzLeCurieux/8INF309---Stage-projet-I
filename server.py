"""
Promo Dashboard – Flask backend
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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

# ── Load .env ─────────────────────────────────────────────────────────────────
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(_env_path, override=True)

# ── Configuration ─────────────────────────────────────────────────────────────
DATABASE_HOST     = os.environ.get("DB_HOST", "127.0.0.1")
DATABASE_PORT     = int(os.environ.get("DB_PORT", "3306"))
DATABASE_USER     = os.environ.get("DB_USER", "root")
DATABASE_PASSWORD = os.environ.get("DB_PASSWORD", "1234")
DATABASE_NAME     = os.environ.get("DB_NAME", "promotions_db")
OPENAI_API_KEY    = os.environ.get("OPENAI_API_KEY", "")

# LLM models: mini is ~16x cheaper, used for attempts 1-2; full for attempt 3
EXTRACT_MODEL_PRIMARY  = "gpt-4o-mini"
EXTRACT_MODEL_FALLBACK = "gpt-4o"

SIMILARITY_THRESHOLD  = 0.92
INACTIVE_AFTER_DAYS   = 7
SCRAPE_INTERVAL_HOURS = int(os.environ.get("SCRAPE_INTERVAL_HOURS", "6"))
MAX_DISCOVERY_PAGES   = 3   # extra pages to follow per restaurant (reduced for perf)

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")

# ── Playwright concurrency lock (max 1 Chromium at a time) ────────────────────
# Prevents multiple simultaneous browser instances that would crash the machine.
_playwright_lock = threading.Lock()

# ── In-memory job store ───────────────────────────────────────────────────────
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()

def _new_job(name: str) -> str:
    jid = str(uuid.uuid4())[:8]
    with _jobs_lock:
        _jobs[jid] = {"status": "pending", "name": name, "result": None,
                      "error": None, "started_at": datetime.now().isoformat(),
                      "pages_crawled": [], "finished_at": None}
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


# ── DB helper ─────────────────────────────────────────────────────────────────
def get_db():
    return mysql.connector.connect(
        host=DATABASE_HOST, port=DATABASE_PORT,
        user=DATABASE_USER, password=DATABASE_PASSWORD,
        database=DATABASE_NAME,
    )


# ── Embedding model ───────────────────────────────────────────────────────────
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


# ── Classification ─────────────────────────────────────────────────────────────
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


# ── HTML → text ───────────────────────────────────────────────────────────────
def _html_to_text(html: str, base_url: str = "") -> str:
    """
    Convert HTML to clean text for LLM processing.
    Image URLs are embedded as [IMAGE:url] markers so the LLM can associate
    them with nearby promotions.
    """
    html = html[:400_000]  # Hard cap: prevent OOM on JS-heavy pages (400KB is plenty)
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside",
                     "noscript", "form", "svg"]):
        tag.decompose()

    # Embed image URLs as text markers before stripping HTML
    for img in soup.find_all("img"):
        src = (img.get("src") or img.get("data-src") or
               img.get("data-lazy-src") or img.get("data-original") or
               img.get("data-srcset", "").split()[0] or "").strip()
        if src and not src.startswith("data:") and len(src) > 5:
            # Convert relative URLs to absolute
            if base_url and src.startswith("/"):
                parsed = urlparse(base_url)
                src = f"{parsed.scheme}://{parsed.netloc}{src}"
            elif base_url and not src.startswith("http"):
                src = urljoin(base_url, src)
            img.replace_with(f" [IMAGE:{src}] ")
        else:
            img.decompose()

    lines = [l.strip() for l in soup.get_text(separator="\n").splitlines()]
    return "\n".join(l for l in lines if l)


# ── Smart link discovery ───────────────────────────────────────────────────────
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


# ── Single-page fetch (AsyncFetcher → DynamicFetcher fallback) ────────────────
def _content_looks_static(text: str) -> bool:
    if len(text) < 2500:   # Raised: avoid Playwright when HTTP fetch already got useful content
        return False
    has_price = bool(re.search(r"\$\s*\d+|\d+[.,]\d{2}", text))
    promo_kws = ["promo", "offre", "offer", "special", "rabais",
                 "discount", "deal", "saving", "happy hour", "combo",
                 "featuring", "vedette", "limité", "limited"]
    has_kw = any(k in text.lower() for k in promo_kws)
    return has_price or has_kw

async def _scroll_and_wait(page):
    """Scroll to trigger lazy-loading of text content (resources already blocked at browser level)."""
    try:
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        await page.wait_for_timeout(1000)
    except Exception:
        pass


async def _fetch_single_page(url: str) -> tuple[str, str]:
    """
    Fetch one URL: try AsyncFetcher (HTTP) first, DynamicFetcher (Playwright)
    if content looks like a JS shell or has no promo signals.
    Returns (text, html).
    """
    from scrapling.fetchers import AsyncFetcher, DynamicFetcher

    html = text = ""
    try:
        fetcher = AsyncFetcher()
        resp = await fetcher.get(url, stealthy_headers=True, follow_redirects=True, timeout=20)
        html = resp.html_content or ""
        text = _html_to_text(html, url)
        logging.info(f"AsyncFetcher → {url} ({len(text)} chars)")
    except Exception as exc:
        logging.warning(f"AsyncFetcher failed for {url}: {exc}")

    if not _content_looks_static(text):
        logging.info(f"Content looks JS-rendered ({len(text)} chars) – trying Playwright for {url}")
        # Acquire the global lock so only ONE Chromium instance runs at a time.
        # Use run_in_executor to avoid blocking the event loop while waiting.
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _playwright_lock.acquire)
        try:
            resp = await DynamicFetcher.async_fetch(
                url,
                headless=True,
                network_idle=True,    # OK – settles fast because resources are blocked
                timeout=25_000,       # 25 s (reduced from 30 s)
                retries=1,            # Minimum allowed (disable_resources keeps it light)
                disable_resources=True,  # Block images/fonts/media/stylesheets → huge I/O reduction
                blocked_domains={     # Block analytics/tracking that never add promo content
                    "google-analytics.com", "analytics.google.com",
                    "googletagmanager.com", "doubleclick.net",
                    "googlesyndication.com", "facebook.net",
                    "connect.facebook.net", "hotjar.com",
                    "mixpanel.com", "segment.com", "intercom.io",
                    "clarity.ms", "sentry.io",
                },
                page_action=_scroll_and_wait,
            )
            html2 = resp.html_content or ""
            text2 = _html_to_text(html2, url)
            if len(text2) > len(text):
                html, text = html2, text2
                logging.info(f"Playwright → {url} ({len(text)} chars)")
            elif html2:
                # Playwright content not longer, but may have image src attributes
                # that the static HTML lacks. Append any new image URLs to the text.
                from bs4 import BeautifulSoup as _BS
                _soup = _BS(html2[:400_000], "html.parser")
                _imgs = []
                for _img in _soup.find_all("img"):
                    _src = (_img.get("src") or _img.get("data-src") or
                            _img.get("data-lazy-src") or "").strip()
                    if _src and not _src.startswith("data:") and "http" in _src:
                        if _src.startswith("/"):
                            from urllib.parse import urlparse as _up
                            _p = _up(url)
                            _src = f"{_p.scheme}://{_p.netloc}{_src}"
                        _imgs.append(f"[IMAGE:{_src}]")
                if _imgs:
                    text = text + "\n" + " ".join(_imgs[:40])
                    logging.info(f"Playwright images appended ({len(_imgs)} imgs) for {url}")
        except Exception as exc:
            logging.warning(f"Playwright failed for {url}: {exc}")
        finally:
            _playwright_lock.release()

    return text, html


# ── Smart multi-page crawl ────────────────────────────────────────────────────
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


# ── LLM extraction (GPT-4o-mini primary → GPT-4o fallback) ───────────────────
def _extract_promos_sync(text: str, restaurant_name: str, page_url: str) -> list[dict]:
    """
    Send combined page text to GPT-4o-mini (attempts 1–2) then GPT-4o (attempt 3).
    Retries if empty result (model is occasionally non-deterministic).
    Called SYNCHRONOUSLY with NO asyncio event loop active in the thread.
    """
    content = text[:10_000]  # ~2.5k tokens input, leaves room for large JSON responses
    prompt = f"""You are a promotion extractor for a restaurant analytics system.

Restaurant: {restaurant_name}
Source URL: {page_url}

TASK: Extract every promotional item, special offer, featured dish, limited-time deal,
combo, or highlighted menu item from the text below. This page IS a promotions page,
so treat any priced item or featured item as a promotion.

For each item return a JSON object with exactly these keys:
  - "promo_type"    : category — "Duo", "Famille", "Solo", "Happy Hour",
                      "Spécial du Jour", "Limited Time", "Featured", "Combo", "Other"
  - "promo_details" : full description (name, ingredients, quantities, everything)
  - "price"         : price string like "12.99" or "Not Provided" if absent
  - "promo_date"    : validity or "Not Provided"
  - "link"          : direct URL or "{page_url}"
  - "image_url"     : the closest [IMAGE:url] marker found BEFORE or AFTER this item
                      in the text; use the full URL. "Not Provided" only if no [IMAGE:…] nearby.

Rules:
- Include ALL priced items visible on this promotions page, even regular menu items
- Convert prices like "13,00$" → "13.00", "13 $" → "13.00"
- Do NOT skip items just because they look like menu items — on a promo page they ARE promos
- For image_url: scan the surrounding lines for [IMAGE:https://…] and copy the full URL
- Return ONLY a valid JSON array, no markdown fences, no extra text
- If the page truly has zero food/drink items at all, return []

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


# ── Sync wrapper: Phase 1 async fetch → Phase 2 sync LLM ─────────────────────
def _scrape_sync(url: str, restaurant_name: str) -> tuple[list[dict], list[str]]:
    """
    Two-phase pipeline (safe for background threads):
    1. Async fetch + discovery (isolated event loop, then closed)
    2. Sync LLM extraction (NO event loop active → avoids httpx/anyio bug)
    Returns (promo_list, crawled_url_list).
    """
    logging.info(f"Scraping {restaurant_name} @ {url}")

    # Phase 1: async web crawl
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        text, crawled = loop.run_until_complete(_smart_crawl(url))
    finally:
        loop.close()
        asyncio.set_event_loop(None)   # ← MUST clear before calling OpenAI

    if not text or len(text) < 50:
        logging.warning(f"No content scraped for {restaurant_name}")
        return [], [url]

    # Phase 2: LLM extraction (synchronous, no event loop)
    promos = _extract_promos_sync(text, restaurant_name, url)
    return promos, crawled


# ── DB: dedup + save ──────────────────────────────────────────────────────────
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
                for ex in existing:
                    if ex["emb"] is not None:
                        cos = float(cosine_similarity(emb_arr, ex["emb"])[0][0])
                        score = (1 - PRICE_WEIGHT) * cos + PRICE_WEIGHT * _price_sim(
                            new_price, _parse_price_float(ex["price"]))
                        if score > best_score:
                            best_score, best_id = score, ex["id"]
                    elif ex["det"] == det_lower:
                        best_score, best_id = SIMILARITY_THRESHOLD, ex["id"]; break
            else:
                for ex in existing:
                    if ex["det"] == det_lower:
                        best_score, best_id = SIMILARITY_THRESHOLD, ex["id"]; break

            if best_score >= SIMILARITY_THRESHOLD and best_id:
                cur.execute("UPDATE promotions_table SET last_seen=%s, is_active=1 WHERE id=%s",
                            (now, best_id))
                db.commit(); updated += 1; continue

            grade, savings, category = classify_promotion(
                promo.get("price"), promo.get("promo_type", ""), det)
            emb_bytes = emb_arr.tobytes() if emb_arr is not None else None
            try:
                cur.execute(
                    "INSERT INTO promotions_table "
                    "(restaurant, promo_type, promo_details, price, promo_date, link, image_url, "
                    "saved_date_time, embedding, grade, savings_estimate, category, is_active, last_seen) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,1,%s)",
                    (restaurant_name, promo.get("promo_type","Other"), det,
                     promo.get("price"), promo.get("promo_date"),
                     promo.get("link"), promo.get("image_url"),
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


# ── Background scrape thread ──────────────────────────────────────────────────
def _run_scrape_blocking(restaurant_name: str, url: str) -> dict:
    promos, crawled = _scrape_sync(url, restaurant_name)
    stats = save_promos_to_db(restaurant_name, promos)
    stats["marked_inactive"] = mark_inactive_promos(restaurant_name)
    stats["pages_crawled"] = crawled
    return stats

def _background_scrape(jid: str, restaurant_name: str, url: str):
    _set_job(jid, "running")
    try:
        result = _run_scrape_blocking(restaurant_name, url)
        _set_job(jid, "done", result=result, pages=result.get("pages_crawled", []))
        logging.info(f"[Job {jid}] Done {restaurant_name}: {result}")
    except Exception as exc:
        logging.error(f"[Job {jid}] Error {restaurant_name}: {exc}\n{traceback.format_exc()}")
        _set_job(jid, "error", error=str(exc))


# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)


# ── Scheduler ─────────────────────────────────────────────────────────────────
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


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def dashboard():
    try:
        db = get_db(); cur = db.cursor(dictionary=True)
        cur.execute("""
            SELECT r.id, r.name, r.url, r.scraper_type,
                   COUNT(p.id)                                       AS total_promos,
                   SUM(CASE WHEN p.is_active=1 THEN 1 ELSE 0 END)   AS active_promos,
                   SUM(CASE WHEN p.grade IN ('A+','A') AND p.is_active=1 THEN 1 ELSE 0 END) AS top_promos,
                   MAX(p.last_seen)                                  AS last_scraped
            FROM restaurants r
            LEFT JOIN promotions_table p ON p.restaurant = r.name
            GROUP BY r.id, r.name, r.url, r.scraper_type
            ORDER BY r.name
        """)
        restaurants = cur.fetchall()
        cur.close(); db.close()
    except Exception as exc:
        restaurants = []; logging.error(f"dashboard error: {exc}")
    return render_template("index.html", restaurants=restaurants,
                           scrape_interval_hours=SCRAPE_INTERVAL_HOURS)


@app.route("/restaurant/<int:rid>")
def restaurant_detail(rid):
    cat_filter   = request.args.get("category", "")
    grade_filter = request.args.get("grade", "")
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
        query += " ORDER BY is_active DESC, last_seen DESC"
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
        db = get_db(); cur = db.cursor()
        cur.execute("DELETE FROM restaurants WHERE id = %s", (rid,))
        db.commit(); cur.close(); db.close()
    except Exception as exc: return f"Error: {exc}", 500
    return redirect(url_for("dashboard"))


# ── Fire-and-forget crawl ─────────────────────────────────────────────────────
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
                     args=(jid, rest["name"], rest["url"]), daemon=True).start()
    return jsonify({"job_id": jid, "status": "pending", "name": rest["name"]}), 202


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
                         args=(jid, rest["name"], rest["url"]), daemon=True).start()
        jobs.append({"job_id": jid, "name": rest["name"], "status": "pending"})
    return jsonify({"jobs": jobs}), 202


@app.route("/job/<jid>")
def job_status(jid):
    with _jobs_lock:
        job = _jobs.get(jid)
    if not job: return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


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


# ── JSON API ──────────────────────────────────────────────────────────────────
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


@app.route("/ping")
def ping(): return "pong", 200


@app.route("/api/scheduler/status")
def scheduler_status():
    return jsonify({"interval_hours": SCRAPE_INTERVAL_HOURS,
                    "inactive_after_days": INACTIVE_AFTER_DAYS,
                    "primary_model": EXTRACT_MODEL_PRIMARY,
                    "fallback_model": EXTRACT_MODEL_FALLBACK})


@app.errorhandler(Exception)
def handle_error(err):
    from werkzeug.exceptions import HTTPException
    if isinstance(err, HTTPException): return err
    logging.exception("Unhandled error: %s", err)
    return jsonify({"error": "internal-server-error", "detail": str(err)}), 500


# ── DB initialisation (tables + default restaurants) ─────────────────────────
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


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import hypercorn.asyncio
    import hypercorn.config
    _init_db()
    _scheduler = _start_scheduler()
    cfg = hypercorn.config.Config()
    cfg.bind = ["0.0.0.0:5000"]
    logging.info("Starting Promo Dashboard on http://0.0.0.0:5000")
    try:
        asyncio.run(hypercorn.asyncio.serve(app, cfg))
    finally:
        if _scheduler: _scheduler.shutdown(wait=False)
