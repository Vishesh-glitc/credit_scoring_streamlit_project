# backend.py
from __future__ import annotations
import os
import re
import json
import requests
import feedparser
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
from openai import OpenAI
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from statistics import mean
from dotenv import load_dotenv
load_dotenv()  # load keys from .env


# ------------------------------------------------------------------
# API KEYS (hardcoded defaults for quick testing; env vars override if set)
# ------------------------------------------------------------------
# Hardcoded defaults for quick testing; environment variables will override if set
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CX      = os.getenv("GOOGLE_CX")

# ------------------------------------------------------------------
# Hardcoded VAT ID overrides for specific companies
# ------------------------------------------------------------------
VAT_OVERRIDES: Dict[str, str] = {
    "Amobee Asia Pte Ltd": "UEN 201617863D",
    "Winner Studio Co., Limited": "GB153458112",
}

# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------
def _get_openai_client():
    return OpenAI(api_key=OPENAI_API_KEY)

def _now() -> datetime:
    return datetime.now()

def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")

def _safe_get(url: str, timeout: int = 12) -> Optional[str]:
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=timeout)
        if r.status_code == 200:
            return r.text
    except Exception:
        return None
    return None

def _extract_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return soup.get_text(" ", strip=True)


# ------------------------------------------------------------------
# Logo helpers: discover from site, with a neutral default fallback
# ------------------------------------------------------------------
DEFAULT_LOGO_DATA_URI = (
    "data:image/svg+xml;utf8,"
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 128 128'>"
    "<rect width='128' height='128' fill='%23EEF2F7'/>"
    "<g fill='%23677A98'>"
    "<rect x='18' y='62' width='24' height='40' rx='2'/>"
    "<rect x='52' y='46' width='24' height='56' rx='2'/>"
    "<rect x='86' y='30' width='24' height='72' rx='2'/>"
    "<rect x='16' y='102' width='96' height='6'/>"
    "</g>"
    "</svg>"
)

def _standard_logo() -> str:
    """
    Returns a neutral, brand-agnostic placeholder logo as a data URI (SVG).
    This avoids external dependencies like Clearbit and always works offline.
    """
    return DEFAULT_LOGO_DATA_URI

def _discover_logo_from_html(base_url: Optional[str], html: Optional[str]) -> Optional[str]:
    """
    Try to extract a representative logo/icon URL from a site's HTML.
    Order of preference:
      1) <meta property="og:logo"> (rare but explicit)
      2) <meta property="og:image"> / <meta name="twitter:image">
      3) <link rel="icon"> / <link rel="shortcut icon"> / <link rel="apple-touch-icon">
      4) <img> with src containing 'logo'/'brand'/'mark'
    Returns an absolute URL string if found, else None.
    """
    if not html:
        return None
    try:
        soup = BeautifulSoup(html, "html.parser")
        def _abs(u: Optional[str]) -> Optional[str]:
            if not u or str(u).strip() == "":
                return None
            try:
                return urljoin(base_url or "", u)
            except Exception:
                return u
        # 1) og:logo (non-standard but check)
        m = soup.find("meta", attrs={"property": "og:logo"})
        if m and m.get("content"):
            return _abs(m["content"])
        # 2) og:image / twitter:image
        for sel in [
            ("meta", {"property": "og:image"}),
            ("meta", {"name": "twitter:image"}),
            ("meta", {"property": "og:image:url"}),
        ]:
            tag = soup.find(*sel)
            if tag and tag.get("content"):
                return _abs(tag["content"])
        # 3) icons
        icon_rels = ["icon", "shortcut icon", "apple-touch-icon", "apple-touch-icon-precomposed", "mask-icon"]
        for rel in icon_rels:
            link = soup.find("link", attrs={"rel": lambda v: v and rel in (v if isinstance(v, list) else str(v).lower())})
            if link and link.get("href"):
                return _abs(link["href"])
        # 4) <img> heuristics
        img = soup.find("img", src=re.compile(r"(logo|brand|logomark|logotype)", re.I))
        if img and img.get("src"):
            return _abs(img["src"])
    except Exception:
        return None
    return None

# ------------------------------------------------------------------
# Google CSE: best-effort image logo search
# ------------------------------------------------------------------
def google_logo_image(company_name: str, website: Optional[str] = None) -> Optional[str]:
    """
    Fetch a likely logo using Google Custom Search (images) with a single query: "<company> logo".
    Returns an absolute image URL or None.
    """
    if not (GOOGLE_API_KEY and GOOGLE_CX) or "REPLACE_ME" in (GOOGLE_API_KEY or "") or "REPLACE_ME" in (GOOGLE_CX or ""):
        return None
    q = f"{company_name} logo"
    url = (
        "https://www.googleapis.com/customsearch/v1?"
        f"key={GOOGLE_API_KEY}&cx={GOOGLE_CX}&q={requests.utils.quote(q)}"
        "&searchType=image&num=8"
    )
    try:
        data = requests.get(url, timeout=12).json()
        items = data.get("items", []) if isinstance(data, dict) else []
    except Exception:
        items = []
    if not items:
        return None
    def _score_item(it: Dict[str, Any]) -> int:
        link = (it.get("link") or "").lower()
        mime = (it.get("mime") or "").lower()
        score = 0
        # prefer vector/transparent formats
        if "svg" in mime or link.endswith(".svg"): score += 5
        if "png" in mime or link.endswith(".png"): score += 3
        # size
        try:
            w = int(it.get("image", {}).get("width", 0)); h = int(it.get("image", {}).get("height", 0))
            if min(w, h) >= 128: score += 2
        except Exception:
            pass
        # reputable domains
        domain = urlparse(link).netloc
        if any(d in domain for d in ["wikimedia.org","wikipedia.org","brand","assets","static","press"]): score += 2
        # de-prioritize social thumbs
        if any(d in domain for d in ["linkedin.","facebook.","x.com","twitter.com","instagram.com"]): score -= 2
        return score
    items = sorted(items, key=_score_item, reverse=True)
    best = items[0]
    return best.get("link")

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

# ------------------------------------------------------------------
# Helper: detect sparse metadata (for fallback)
# ------------------------------------------------------------------
def _is_sparse(meta: dict) -> bool:
    keys = ["full_name","website","logo_url","founded_year","headquarters","industry","countries","vat_id","description"]
    missing = 0
    for k in keys:
        v = meta.get(k)
        if not v or v == []:
            missing += 1
    return missing >= 6

# ------------------------------------------------------------------
# Google: find official website (simple ranking to avoid social sites)
# ------------------------------------------------------------------
def google_official_site(query: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    if not (GOOGLE_API_KEY and GOOGLE_CX) or "REPLACE_ME" in GOOGLE_API_KEY or "REPLACE_ME" in GOOGLE_CX:
        return None, None, None
    url = f"https://www.googleapis.com/customsearch/v1?key={GOOGLE_API_KEY}&cx={GOOGLE_CX}&q={requests.utils.quote(query)}"
    # print("Google CSE URL:", url)  # Debug log
    try:
        data = requests.get(url, timeout=12).json()
        items = data.get("items", []) if isinstance(data, dict) else []
        if not items:
            return None, None, None
        bad_hosts = {"linkedin.com","www.linkedin.com","twitter.com","x.com","facebook.com","glassdoor.com","crunchbase.com","apps.apple.com","play.google.com"}
        def score_item(it):
            link = it.get("link","")
            host = urlparse(link).netloc.lower()
            s = 0
            if host not in bad_hosts: s += 2
            if any(k in link.lower() for k in ["/about","/company","/en/"]): s += 1
            return s
        best = sorted(items, key=score_item, reverse=True)[0]
        return best.get("title"), best.get("snippet"), best.get("link")
    except Exception:
        return None, None, None

# ------------------------------------------------------------------
# Companies House (UK) fallback (no API key; HTML scrape)
# ------------------------------------------------------------------
def uk_companies_house_fallback(company_name: str) -> Tuple[Optional[str], Optional[str]]:
    try:
        q = company_name.replace(" ", "+")
        search_url = f"https://find-and-update.company-information.service.gov.uk/search?q={q}"
        html = _safe_get(search_url)
        if not html:
            return None, None
        soup = BeautifulSoup(html, "html.parser")
        first = soup.select_one('a[href^="/company/"]')
        if not first:
            return None, None
        comp_url = urljoin("https://find-and-update.company-information.service.gov.uk", first.get("href"))
        comp_html = _safe_get(comp_url)
        if not comp_html:
            return None, None
        text = _extract_text(comp_html)[:200000]
        return comp_url, text
    except Exception:
        return None, None

# ------------------------------------------------------------------
# LLM extraction (one-shot JSON)
# ------------------------------------------------------------------
def extract_metadata_with_llm(company_name: str, website: Optional[str], page_text: str) -> dict:
    """Return normalized company metadata from raw page text using an LLM.
    Fields: full_name, founded_year, headquarters, industry, countries, vat_id, website, description
    """
    if not OPENAI_API_KEY or "REPLACE_WITH_YOUR_OPENAI_KEY" in (OPENAI_API_KEY or ""):
        return {
            "name": company_name,
            "full_name": company_name,
            "website": website,
            # placeholder; may be overridden by discovery in get_company_metadata
            "logo_url": _standard_logo(),
            "founded_year": None,
            "headquarters": None,
            "industry": None,
            "countries": [],
            "vat_id": None,
            "description": None,
        }

    client = _get_openai_client()
    sys_prompt = (
        "You are a precise company information extractor. "
        "Read the company text and return a single JSON object with keys: "
        "full_name, founded_year, headquarters, industry, countries, vat_id, website, description. "
        "Rules: (1) Use null for unknowns. (2) countries must be an array of country names. "
        "(3) description must be exactly 2 neutral sentences stating what the company does and where it is based. No marketing language."
    )
    user_prompt = f"Company name: {company_name}\nWebsite: {website or 'null'}\n\nCompany text:\n{page_text}\n\nReturn ONLY valid JSON. No backticks, no commentary."

    try:
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )
        s = (resp.choices[0].message.content or "").strip()
        if not (s.startswith("{") and s.endswith("}")):
            i, j = s.find("{"), s.rfind("}")
            s = s[i:j+1] if (i != -1 and j != -1 and j > i) else "{}"
        data = json.loads(s)
    except Exception:
        data = {}

    out = {
        "name": company_name,
        "full_name": data.get("full_name") or company_name,
        "website": data.get("website") or website,
        # placeholder; may be overridden by discovery in get_company_metadata
        "logo_url": _standard_logo(),
        "founded_year": str(data.get("founded_year")) if data.get("founded_year") else None,
        "headquarters": data.get("headquarters"),
        "industry": data.get("industry"),
        "countries": data.get("countries") or [],
        "vat_id": data.get("vat_id"),
        "description": data.get("description"),
    }
    return out

# ------------------------------------------------------------------
# Public: fetch company metadata (Google -> scrape -> LLM)
# ------------------------------------------------------------------
def get_company_metadata(company_name: str) -> dict:
    # 1) Try Google Custom Search to find an official site
    title, snippet, link = google_official_site(company_name)
    website = link or None
    page_text = ""
    html = None
    if website:
        html = _safe_get(website)
        if html:
            page_text = _extract_text(html)[:200000]

    # 2) LLM extraction over whatever we scraped
    meta = extract_metadata_with_llm(company_name, website, page_text)
    # Try Google Images "<company> logo" first
    g_logo = google_logo_image(company_name, website)
    if g_logo:
        meta["logo_url"] = g_logo
    elif not meta.get("logo_url"):
        meta["logo_url"] = _standard_logo()

    # 3) If metadata is sparse, try UK Companies House fallback (common for LTDs)
    if _is_sparse(meta):
        ch_url, ch_text = uk_companies_house_fallback(company_name)
        if ch_url and ch_text:
            meta = extract_metadata_with_llm(company_name, ch_url, ch_text)
            # Use Google Images "<company> logo" for fallback logo
            g_logo2 = google_logo_image(company_name, None)
            if g_logo2:
                meta["logo_url"] = g_logo2
            elif not meta.get("logo_url"):
                meta["logo_url"] = _standard_logo()

    # Apply hardcoded VAT overrides if available
    if company_name in VAT_OVERRIDES:
        meta["vat_id"] = VAT_OVERRIDES[company_name]
    return meta

# ------------------------------------------------------------------
# Scoring helpers: Country, Industry, Reputation
# ------------------------------------------------------------------
COUNTRY_RISK_0_TO_10 = {
    "united kingdom": 8.5,
    "united states": 8.5,
    "singapore": 9.0,
    "japan": 8.5,
    "finland": 9.0,
    "spain": 8.0,
    "australia": 8.5,
    "israel": 7.5,
    "saudi arabia": 7.0,
    "hong kong": 7.5,
    "china": 6.5,
    "india": 6.5,
    "u.k.": 8.5,
    "uk": 8.5,
    "u.s.": 8.5,
    "usa": 8.5,
}

def _norm_country(x: Optional[str]) -> str:
    return (x or "").strip().lower()

def infer_primary_country(metadata: dict) -> Tuple[str, bool]:
    countries = metadata.get("countries") or []
    if isinstance(countries, list) and countries:
        for c in countries:
            if isinstance(c, str) and len(c) >= 3:
                return c, False
    hq = metadata.get("headquarters") or ""
    if hq:
        toks = [t.strip() for t in hq.replace("/", ",").split(",") if t.strip()]
        if toks:
            cand = toks[-1].replace("SAR", "").strip()
            return cand, False
    return "Unknown", True

def country_risk_score(country_name: str) -> Tuple[float, bool]:
    key = _norm_country(country_name)
    if key in COUNTRY_RISK_0_TO_10:
        return COUNTRY_RISK_0_TO_10[key], False
    aliases = {
        "united kingdom of great britain and northern ireland": "united kingdom",
        "people's republic of china": "china",
        "pr china": "china",
        "u.k": "uk",
        "u.s": "usa",
        "u.s.a": "usa",
        "great britain": "united kingdom",
        "hong kong sar": "hong kong",
    }
    if key in aliases:
        return COUNTRY_RISK_0_TO_10.get(aliases[key], 5.0), False
    return 5.5, True

INDUSTRY_RISK_0_TO_10 = {
    "banking": 8.0,
    "insurance": 7.5,
    "utilities": 7.5,
    "telecom": 7.0,
    "software": 6.5,
    "it services": 6.5,
    "technology": 6.0,
    "manufacturing": 6.0,
    "pharmaceuticals": 6.5,
    "healthcare": 6.5,
    "retail": 5.0,
    "hospitality": 5.0,
    "media": 5.5,
    "advertising": 5.5,
    "construction": 5.5,
    "real estate": 5.5,
    "mining": 5.0,
}

INDUSTRY_KEYWORDS = [
    ("bank", "banking"), ("finance", "banking"), ("insur", "insurance"),
    ("utility", "utilities"), ("telecom", "telecom"), ("software", "software"),
    ("saas", "software"), ("it service", "it services"), ("tech", "technology"),
    ("manufactur", "manufacturing"), ("pharma", "pharmaceuticals"), ("health", "healthcare"),
    ("retail", "retail"), ("hospitality", "hospitality"), ("media", "media"),
    ("advertis", "advertising"), ("adtech", "advertising"), ("construction", "construction"),
    ("real estate", "real estate"), ("mining", "mining"),
]

# ------------------------------------------------------------------
# Benchmarks seed (MVP fallback; replace with external data later)
# ------------------------------------------------------------------
BENCHMARKS_SEED: List[Dict[str, Any]] = [
    # United Kingdom
    {"industry": "Textiles", "country": "United Kingdom", "avgRevenue": 1_800_000, "avgMargin": 5.0, "avgD2E": 1.1, "updatedAt": "2025-09-01"},
    {"industry": "Retail",   "country": "United Kingdom", "avgRevenue": 2_500_000, "avgMargin": 7.0, "avgD2E": 0.9, "updatedAt": "2025-09-01"},
    # United States
    {"industry": "Consumer Goods", "country": "United States", "avgRevenue": 4_000_000, "avgMargin": 7.0, "avgD2E": 1.0, "updatedAt": "2025-09-01"},
    # Japan
    {"industry": "Wholesale",          "country": "Japan", "avgRevenue": 5_000_000, "avgMargin": 4.0,  "avgD2E": 1.3, "updatedAt": "2025-09-01"},
    {"industry": "Automotive Trading", "country": "Japan", "avgRevenue": 12_000_000,"avgMargin": 5.5,  "avgD2E": 1.5, "updatedAt": "2025-09-01"},
    # Hong Kong
    {"industry": "Media & Technology",   "country": "Hong Kong", "avgRevenue": 3_000_000, "avgMargin": 10.0, "avgD2E": 0.8, "updatedAt": "2025-09-01"},
    {"industry": "Gaming & Software",    "country": "Hong Kong", "avgRevenue": 8_000_000, "avgMargin": 15.0, "avgD2E": 0.6, "updatedAt": "2025-09-01"},
    {"industry": "Retail",               "country": "Hong Kong", "avgRevenue": 2_800_000, "avgMargin": 6.5,  "avgD2E": 0.9, "updatedAt": "2025-09-01"},
    {"industry": "Creative Media",       "country": "Hong Kong", "avgRevenue": 3_000_000, "avgMargin": 9.5,  "avgD2E": 0.8, "updatedAt": "2025-09-01"},
    {"industry": "Electronics Trading",  "country": "Hong Kong", "avgRevenue": 7_000_000, "avgMargin": 6.0,  "avgD2E": 1.2, "updatedAt": "2025-09-01"},
]

def _slug(s: str) -> str:
    return re.sub(r'[^a-z0-9]+', '-', (s or '').strip().lower()).strip('-')

# Hints to ensure the listed companies get appropriate (industry, country)
COMPANY_BENCHMARK_HINTS: Dict[str, Tuple[str, str]] = {
    "Amobee Asia Pte Ltd": ("Textiles", "United Kingdom"),
    "Elite Zone Ltd": ("Retail", "United Kingdom"),
    "Sticky Hands Inc.": ("Consumer Goods", "United States"),
    "Colonne K.K.": ("Wholesale", "Japan"),
    "Honda Shoji Co., LTD": ("Automotive Trading", "Japan"),
    "Fortune Media Technology": ("Media & Technology", "Hong Kong"),
    "Sandsoft International Tech Ltd.": ("Gaming & Software", "Hong Kong"),
    "Funny Global Limited": ("Retail", "Hong Kong"),
    "Winner Studio Co., Limited": ("Creative Media", "Hong Kong"),
    "Restar Limited": ("Electronics Trading", "Hong Kong"),
}

def get_benchmark_pair_for_company(company_name: str, company_obj: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """
    Returns (industry, country) for benchmarks. Uses explicit hints first; falls back to inferred fields.
    """
    # 1) Exact hint match
    if company_name in COMPANY_BENCHMARK_HINTS:
        return COMPANY_BENCHMARK_HINTS[company_name]

    # 2) Fallback to company fields
    ind = company_obj.get("industry")
    ctry = company_obj.get("country")
    if ind and ctry:
        return ind, ctry

    # 3) Try metadata-style fields
    countries = company_obj.get("countries") or []
    ctry2 = ctry or (countries[0] if countries else None)
    return ind, ctry2

def fetch_industry_benchmarks_by_pair(industry: Optional[str], country: Optional[str]) -> List[Dict[str, Any]]:
    if not industry or not country:
        return []
    return [b for b in BENCHMARKS_SEED if b["industry"] == industry and b["country"] == country]

def fetch_company_benchmark_entry(industry: Optional[str], country: Optional[str]) -> Optional[Dict[str, Any]]:
    if not industry or not country:
        return None
    # Use a deterministic id for UI linking
    return {"benchmarkId": f"{_slug(industry)}-{_slug(country)}-2025-09"}

def infer_industry_term(industry_text: Optional[str]) -> Tuple[str, bool]:
    txt = (industry_text or "").lower()
    for needle, canon in INDUSTRY_KEYWORDS:
        if needle in txt:
            return canon, False
    if "shop" in txt or "store" in txt:
        return "retail", True
    if "app" in txt:
        return "technology", True
    if "content" in txt:
        return "media", True
    return "unknown", True

def industry_risk_score(industry_text: Optional[str]) -> Tuple[float, bool, str]:
    canon, defaulted = infer_industry_term(industry_text)
    if canon in INDUSTRY_RISK_0_TO_10:
        return INDUSTRY_RISK_0_TO_10[canon], defaulted, canon
    return 5.5, True, canon

# Reputation (News Sentiment via Google News RSS + optional LLM classification)
def fetch_news_items(company_name: str, max_items: int = 5) -> List[Dict[str, Any]]:
    """
    Fetch up to `max_items` recent news items via Google News RSS using multiple disambiguated queries.
    Locale is set to India English to improve relevance for the user's region.
    Falls back to the company's newsroom feed for well-known cases (e.g., Apple).
    """
    def gnews(search_q: str) -> List[Dict[str, Any]]:
        q = requests.utils.quote(search_q)
        url = f"https://news.google.com/rss/search?q={q}&hl=en-IN&gl=IN&ceid=IN:en"
        feed = feedparser.parse(url)
        rows: List[Dict[str, Any]] = []
        for e in getattr(feed, "entries", [])[:max_items*2]:
            dt = e.get("published_parsed") or e.get("updated_parsed")
            when = datetime(*dt[:6]) if dt else _now()
            # Source robustly
            src = None
            if hasattr(e, "source") and isinstance(getattr(e, "source"), dict):
                src = getattr(e, "source", {}).get("title")
            if not src:
                src = getattr(e, "author", None) or "Google News"
            rows.append({
                "observedAt": _iso(when),
                "score": 0.0,
                "headline": getattr(e, "title", "") or "",
                "source": src or "Google News",
                "url": getattr(e, "link", None),
            })
        return rows

    # Try a few disambiguated queries to avoid fruit/ambiguous names, and capture real company news.
    queries: List[str] = [
        f"\"{company_name}\" company",
        f"\"{company_name}\" (Inc OR Ltd OR PLC OR Limited)",
        f"\"{company_name}\" (revenue OR profit OR lawsuit OR acquisition OR partnership OR expansion OR upgrade)",
        f"{company_name} stock",
    ]
    # Extra hints for very common names
    if company_name.lower() in {"apple", "apple inc", "apple inc."}:
        queries.insert(0, "\"Apple Inc\" OR AAPL")
    if company_name.lower() in {"meta", "meta platforms"}:
        queries.insert(0, "\"Meta Platforms\" OR META")
    if company_name.lower() in {"alphabet", "google"}:
        queries.insert(0, "\"Alphabet Inc\" OR GOOGL")

    seen_links: set = set()
    out: List[Dict[str, Any]] = []
    for q in queries:
        for row in gnews(q):
            link = row.get("url")
            if not link or link in seen_links:
                continue
            seen_links.add(link)
            out.append(row)
            if len(out) >= max_items:
                break
        if len(out) >= max_items:
            break

    # Fallback to known newsroom feed for Apple if still empty
    if not out and company_name.lower().startswith("apple"):
        try:
            rss = "https://www.apple.com/newsroom/rss-feed.rss"
            feed = feedparser.parse(rss)
            for e in getattr(feed, "entries", [])[:max_items]:
                dt = e.get("published_parsed") or e.get("updated_parsed")
                when = datetime(*dt[:6]) if dt else _now()
                out.append({
                    "observedAt": _iso(when),
                    "score": 0.0,
                    "headline": getattr(e, "title", "") or "",
                    "source": "Apple Newsroom",
                    "url": getattr(e, "link", None),
                })
        except Exception:
            pass

    return out[:max_items]

def fetch_news_sentiment(company_name: str) -> Tuple[float, bool, List[Dict[str, Any]]]:
    """
    Classify top 5 news headlines as good (+1), neutral (0), or bad (-1).
    Returns (aggregate_0_to_10, used_defaults, items_with_scores)
    - If OpenAI is unavailable, falls back to a lightweight keyword heuristic.
    - The UI uses items[*].score and the aggregate mapped to 0..10.
    """
    items = fetch_news_items(company_name, max_items=5)
    if not items:
        return 5.0, True, []

    def heuristic_score(text: str) -> int:
        txt = (text or "").lower()
        neg_kw = ["fraud", "scandal", "lawsuit", "probe", "recall", "layoff", "bankruptcy", "defaults", "debt crisis", "fine", "penalty", "data breach", "hack", "downturn", "decline", "drop", "falls", "plunge", "warning"]
        pos_kw = ["wins", "expands", "growth", "record", "beats", "surge", "raises", "profit", "milestone", "partnership", "award", "acquires", "launches", "approval", "upgrade"]
        score = 0
        if any(k in txt for k in neg_kw): score -= 1
        if any(k in txt for k in pos_kw): score += 1
        # clamp to -1..1
        return -1 if score < 0 else (1 if score > 0 else 0)

    used_defaults = False

    # Try OpenAI for classification if available
    if OPENAI_API_KEY and "REPLACE_WITH_YOUR_OPENAI_KEY" not in (OPENAI_API_KEY or ""):
        try:
            client = _get_openai_client()
            for it in items:
                h = it["headline"]
                prompt = (
                    "Classify this headline for credit risk perspective as Good (+1), Neutral (0), or Bad (-1). "
                    "Respond with exactly one number from the set {-1,0,1}.\n\n"
                    f"Headline: {h}"
                )
                r = client.chat.completions.create(
                    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                    messages=[
                        {"role": "system", "content": "You are a sober, conservative credit risk classifier."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0,
                )
                val = (r.choices[0].message.content or "0").strip()
                try:
                    it["score"] = float(int(val))
                except Exception:
                    it["score"] = float(heuristic_score(h))
                    used_defaults = True
        except Exception:
            # Fallback to heuristic for all
            for it in items:
                it["score"] = float(heuristic_score(it["headline"]))
            used_defaults = True
    else:
        # No OpenAI key: heuristic
        for it in items:
            it["score"] = float(heuristic_score(it["headline"]))
        used_defaults = True

    # Aggregate to 0..10: map avg(-1..1) -> 0..10
    vals = [it.get("score", 0.0) for it in items]
    avg = sum(vals) / len(vals) if vals else 0.0
    aggregate = round((avg + 1.0) * 5.0, 2)
    return aggregate, used_defaults, items

# ------------------------------------------------------------------
# Scoring (pillars)
# ------------------------------------------------------------------
def score_company(metadata: dict) -> dict:
    scores = {"Financials": 5.0, "Governance": 5.0, "Industry": 5.0}

    # Financials by age (proxy)
    fy = metadata.get("founded_year")
    if fy:
        try:
            age = max(0, _now().year - int(fy))
            scores["Financials"] = 8 if age >= 20 else 7 if age >= 10 else 6 if age >= 5 else 4
        except Exception:
            pass

    # Governance: HQ + VAT presence
    gov = 5 + (2 if metadata.get("headquarters") else 0) + (2 if metadata.get("vat_id") else 0)
    scores["Governance"] = float(min(gov, 10))

    # Industry via mapping
    i_score, i_def, _canon = industry_risk_score(metadata.get("industry"))
    scores["Industry"] = round(i_score, 2)

    # Country
    pc, c_def = infer_primary_country(metadata)
    c_score, c_def2 = country_risk_score(pc)

    # Reputation (headlines sentiment)
    rep_score, rep_def, news_items = fetch_news_sentiment(metadata.get("name", ""))

    overall = round((scores["Financials"] + scores["Governance"] + scores["Industry"] + c_score + rep_score) / 5.0, 2)
    return {
        "pillar_scores": scores,
        "country_score": round(c_score, 2),
        "reputation_score": round(rep_score, 2),
        "news_items": news_items,
        "overall_score": overall,
        "flags": {
            "industry_defaulted": i_def,
            "country_defaulted": (c_def or c_def2),
            "reputation_defaulted": rep_def,
        },
    }

# ------------------------------------------------------------------
# Public entry point (backend-internal)
# ------------------------------------------------------------------
def run_analysis(company_name: str):
    meta = get_company_metadata(company_name)
    scoring = score_company(meta)
    return meta, scoring

# ------------------------------------------------------------------
# Streamlit UI adapter: fetch_company_report expected by app.py
# ------------------------------------------------------------------
def fetch_company_report(name: str) -> Dict[str, Any]:
    meta, scoring = run_analysis(name)

    # Map to UI fields
    # Score: scale 0..10 -> 0..100
    score_0_100 = int(round(scoring["overall_score"] * 10))
    # Legacy numeric band for backward compatibility
    level = "LOW" if score_0_100 >= 75 else ("MEDIUM" if score_0_100 >= 55 else "HIGH")

    # New classification for Risk field
    if score_0_100 >= 75:
        risk_class = "Low 🟢"
    elif score_0_100 >= 55:
        risk_class = "Moderate 🟡"
    else:
        risk_class = "High 🔴"

    # Company block
    primary_country, _ = infer_primary_country(meta)
    website = meta.get("website")
    logo = meta.get("logo_url") or _standard_logo()

    company = {
        "name": meta.get("name") or name,
        "legalName": meta.get("full_name") or name,
        "registrationId": None,
        "industry": meta.get("industry"),
        "country": primary_country if primary_country != "Unknown" else None,
        "city": None,
        "website": website,
        "employeeCount": None,
        "description": meta.get("description"),
        "foundedYear": meta.get("founded_year"),
        "headquarters": meta.get("headquarters"),
        "countries": meta.get("countries"),
        "vatId": meta.get("vat_id"),
    }

    # --- Benchmarks wiring ---
    bench_industry, bench_country = get_benchmark_pair_for_company(name, company)
    bench_rows = fetch_industry_benchmarks_by_pair(bench_industry, bench_country)
    company_bm = fetch_company_benchmark_entry(bench_industry, bench_country)

    # Subscores mapped to existing UI keys
    fin_sub   = scoring["pillar_scores"]["Financials"]
    gov_sub   = scoring["pillar_scores"]["Governance"]
    ind_sub   = scoring["pillar_scores"]["Industry"]
    geo_sub   = scoring["country_score"]
    rep_sub   = scoring["reputation_score"]

    risk = {
        "scoredAt": risk_class,
        "score": score_0_100,
        "level": level,
        "financial": round(fin_sub, 2),
        "payments": None,              # not derived yet; leave empty
        "news": round(rep_sub, 2),
        "industryRisk": round(ind_sub, 2),
        "geoRisk": round(geo_sub, 2),
        "delta": 0.0,
        "pd": round(_clamp(1.0 - score_0_100/100.0, 0.01, 0.25), 3)
    }
    # Ensure no empty fields in risk; default to 5 if missing/None
    for k in ["financial", "payments", "news", "industryRisk", "geoRisk"]:
        if risk.get(k) is None:
            risk[k] = 5

    # News table rows (observedAt, score, headline, source, url)
    news_rows = scoring.get("news_items", [])

    result: Dict[str, Any] = {
        "company": company,
        "riskScore": risk,
        # Tables (keep keys; leave empty when unknown)
        "financials": [],
        "payments": [],
        "newsSentiment": news_rows,
        "socialSignals": [],
        "registries": [],
        "taxIds": [],
        "filings": [],
        "publicRecords": [],
        "litigation": [],
        "benchmarks": bench_rows,
        "macro": [],
        "documents": [],
        "notes": [],
        # Other sections expected by the current UI
        "alerts": [],
        "alertConfigs": [],
        "dataSources": [],
        "ingestionJobs": [],
        "creditApplications": [],
        "companyBenchmark": company_bm or {},
        "logoUrl": logo,
    }
    return result

# ------------------------------------------------------------------
# PDF export used by the UI
# ------------------------------------------------------------------
def generate_pdf(data: Dict[str, Any]) -> str:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
        from reportlab.lib.units import cm
        from reportlab.lib import colors
    except Exception as e:
        raise RuntimeError(f"reportlab not available: {e}")

    PLACEHOLDER = "Yet to be fetched"
    def _fmt(v):
        if v is None or (isinstance(v, str) and v.strip() == ""):
            return PLACEHOLDER
        if isinstance(v, list):
            return ", ".join([str(x) for x in v]) if v else PLACEHOLDER
        return v

    comp = data.get("company", {}) or {}
    risk = data.get("riskScore", {}) or {}
    name = comp.get("name", "Company")
    website = comp.get("website")
    desc = (comp.get("description") or "").strip()
    logo_url = data.get("logoUrl")

    out_dir = os.path.abspath("./_reports")
    os.makedirs(out_dir, exist_ok=True)
    safe_name = re.sub(r"\s+", "_", name)
    out_path = os.path.join(out_dir, f"{safe_name}_report.pdf")

    # Try to download logo for embedding
    logo_path = None
    if logo_url:
        try:
            resp = requests.get(logo_url, timeout=8)
            if resp.status_code == 200:
                logo_path = os.path.join(out_dir, f"{safe_name}_logo.png")
                with open(logo_path, "wb") as lf:
                    lf.write(resp.content)
        except Exception:
            logo_path = None

    c = canvas.Canvas(out_path, pagesize=A4)
    width, height = A4

    # Theme
    ACCENT = colors.HexColor("#0E6DFD")  # blue accent
    TEXT = colors.HexColor("#222222")
    SUB = colors.HexColor("#555555")
    LINE = colors.HexColor("#E6EAF2")

    y = height - 2*cm

    # Header band
    c.setFillColor(ACCENT)
    c.rect(0, y + 0.8*cm, width, 0.2*cm, stroke=0, fill=1)

    # Logo
    if logo_path:
        try:
            c.drawImage(logo_path, 2*cm, y-0.2*cm, width=3.0*cm, height=3.0*cm, preserveAspectRatio=True, mask='auto')
        except Exception:
            pass

    # Company title & link
    c.setFillColor(TEXT); c.setFont("Helvetica-Bold", 18)
    c.drawString(2*cm + (3.2*cm if logo_path else 0), y+0.6*cm, name)
    c.setFillColor(SUB); c.setFont("Helvetica", 10)
    c.drawString(2*cm + (3.2*cm if logo_path else 0), y-0.1*cm, f"Website: {_fmt(website)}")

    # Description (wrap)
    def wrap_text(s: str, max_chars: int = 95):
        words = s.split()
        line, lines = "", []
        for w in words:
            if len(line) + len(w) + 1 <= max_chars:
                line = (line + " " + w).strip()
            else:
                lines.append(line); line = w
        if line: lines.append(line)
        return lines[:4]  # cap to 4 lines

    y -= 1.2*cm
    if desc:
        c.setFillColor(TEXT); c.setFont("Helvetica-Oblique", 11)
        for t in wrap_text(desc, 95):
            c.drawString(2*cm, y, t); y -= 0.55*cm
    else:
        y -= 0.2*cm

    # Divider
    c.setFillColor(LINE); c.rect(2*cm, y, width-4*cm, 1, stroke=0, fill=1)
    y -= 0.6*cm

    # Quick facts (like the site chips)
    facts = [
        ("Industry", _fmt(comp.get("industry"))),
        ("Country", _fmt(comp.get("country"))),
        ("HQ", _fmt(comp.get("headquarters"))),
        ("Founded", _fmt(comp.get("foundedYear"))),
        ("VAT ID", _fmt(comp.get("vatId"))),
    ]
    c.setFillColor(TEXT); c.setFont("Helvetica-Bold", 12)
    c.drawString(2*cm, y, "Company & Core Info")
    y -= 0.7*cm
    c.setFont("Helvetica", 10)
    col_x = [2*cm, 8*cm, 14*cm]
    for i, (label, value) in enumerate(facts):
        cx = col_x[i % 3]
        c.setFillColor(SUB); c.drawString(cx, y, label)
        c.setFillColor(TEXT); c.drawString(cx, y-0.45*cm, str(value))
        if i % 3 == 2:
            y -= 1.1*cm
    y -= 1.2*cm

    # Risk summary row
    c.setFillColor(TEXT); c.setFont("Helvetica-Bold", 12)
    c.drawString(2*cm, y, "Risk Summary")
    y -= 0.7*cm
    c.setFont("Helvetica", 10)
    subs = [
        ("Score", risk.get("score", "—")),
        ("Level", risk.get("level", "—")),
        ("Financial", risk.get("financial", "—")),
        ("News", risk.get("news", "—")),
        ("Industry", risk.get("industryRisk", "—")),
        ("Geo", risk.get("geoRisk", "—")),
    ]
    for i, (label, value) in enumerate(subs):
        cx = col_x[i % 3]
        c.setFillColor(SUB); c.drawString(cx, y, label)
        c.setFillColor(TEXT); c.drawString(cx, y-0.45*cm, str(value))
        if i % 3 == 2:
            y -= 1.1*cm

    c.showPage(); c.save()
    return out_path