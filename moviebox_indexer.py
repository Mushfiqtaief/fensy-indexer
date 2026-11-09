#!/usr/bin/env python3
"""
Fensy MovieBox Indexer & ML Vector Compiler
Crawls the MovieBox (com.community.oneroom) private mobile API with the
HMAC-MD5 request-signing scheme, compiles the same portable SQLite schema as
the CircleFTP indexer (media_items + categories + FTS5 + 256-byte embedding
BLOBs), and publishes moviebox.db.gz + .sha256 for the Fensy client to sync.

Usage:
    python moviebox_indexer.py --outdir dist
"""

import argparse
import base64
import gzip
import hashlib
import hmac
import json
import os
import random
import re
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional
import requests
from hash_embeddings import compute_embedding_bytes
from imdb_matcher import enrich_with_imdb

# ---------------------------------------------------------------------------
# Signing (transcribed from mesamirh/MovieBox-Tui crypto.rs, MIT/Apache)
# ---------------------------------------------------------------------------

SECRET_B64 = "76iRl07s0xSN9jqmEWAt79EBJZulIQIsV64FZr2O"
CERT_SHA = "03d4b741632d136bc23e2de4c6adcdb7712f1980b619c183221801730bbd0410"
SECRET_BYTES = base64.b64decode(SECRET_B64)

USER_AGENT = (
    "com.community.oneroom/50020121 (Linux; U; Android 13; en_US; "
    "23078RKD5C; Build/TQ2A.230405.003; Cronet/135.0.7012.3)"
)

HOSTS = [
    "https://api6.aoneroom.com",
    "https://api5.aoneroom.com",
    "https://api4.aoneroom.com",
    "https://api4sg.aoneroom.com",
    "https://api6sg.aoneroom.com",
    "https://api3.aoneroom.com",
    "https://api.inmoviebox.com",
]

BFF = "/wefeed-mobile-bff"

PAGE_WORKERS = 8
DETAIL_WORKERS = 12
PAGE_SIZE = 100
MAX_PAGES_PER_SOURCE = 30
REQUEST_TIMEOUT = 20
RETRY_STATUSES = {403, 406, 407, 429, 500, 502, 503, 504}

# Search expansion: keyword seeds swept through search/v2 to reach the full
# catalog (ranking lists only surface ~700 curated titles). Per-seed yields
# measured live: 'a'=446, 'love'=451, 'k'=480, '2024'=412. 26 letters + digits
# + common-word seeds reach the tens of thousands. Search caps at perPage=20
# (30+ returns HTTP 400) and paginates until an empty page.
SEARCH_SEEDS = [chr(c) for c in range(ord('a'), ord('z') + 1)] + [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'the', 'love', 'king', 'star', 'man', 'girl', 'life', 'night', 'day',
    'house', 'story', 'war', 'last', 'one', 'two', 'my', 'no', 'new', 'old',
    'big', 'little', 'young', 'dead', 'dark', 'light', 'lost', 'secret',
    'movie', 'film', 'hindi', 'movie hindi', 'south', 'bengali', 'punjabi',
    'hollywood', 'bollywood', 'tamil', 'telugu', '2020', '2021', '2022',
    '2023', '2024', '2025', '2026', 'game', 'mission', 'city', 'home',
    'heart', 'fire', 'blood', 'gold', 'money', 'world', 'time', 'power',
    'blue', 'black', 'red', 'white', 'green',
]
SEARCH_PAGE_SIZE = 20
SEARCH_MAX_PAGES = 120
SEARCH_WORKERS = 6
SEARCH_PAGE_DELAY = 0.12

_SESSION_LOCAL = threading.local()
_host_lock = threading.Lock()
_active_host_idx = 0


def _md5_hex(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def _sorted_query(path_and_query: str) -> str:
    """path?a=1&b=2 -> path?a=1&b=2 with params re-sorted by key (stable)."""
    if "?" not in path_and_query:
        return path_and_query
    path, query = path_and_query.split("?", 1)
    pairs = []
    for pair in query.split("&"):
        if not pair:
            continue
        key, _, value = pair.partition("=")
        pairs.append((key, value))
    pairs.sort(key=lambda kv: kv[0])
    # Keep multi-value encounter order per key: stable sort handles it.
    return path + "?" + "&".join(f"{k}={v}" for k, v in pairs)


def _build_client_info(build: str) -> str:
    device_id = "".join(random.choice("0123456789abcdef") for _ in range(32))
    info = json.dumps({
        "package_name": "com.community.oneroom",
        "version_name": "4.0.01.0813.03",
        "version_code": 50020121,
        "os": "android",
        "os_version": "13",
        "install_ch": "ps",
        "device_id": device_id,
        "install_store": "ps",
        "gaid": "8f0ba5c5-24a7-4643-a4d5-26d5f3f5ac2c",
        "brand": "Redmi",
        "model": "23078RKD5C",
        "system_language": "en",
        "net": "NETWORK_WIFI",
        "region": "US",
        "timezone": "Asia/Kolkata",
        "sp_code": "40401",
        "X-Play-Mode": "2",
        "X-Client-Build": build,
    }, separators=(",", ":"))
    return info


def _build_headers(
    method: str,
    path_with_query: str,
    body: Optional[str],
    token: Optional[str],
) -> Dict[str, str]:
    ts = int(time.time() * 1000)
    ms = ts % 1000
    ts_sec = ts // 1000
    rand = random.randint(0, 999)

    body_bytes = body.encode("utf-8") if body else b""
    body_len = str(len(body_bytes)) if body else ""
    body_md5 = _md5_hex(body_bytes) if body else ""

    canonical = "\n".join([
        method.upper(),
        "application/json",
        "application/json",
        body_len,
        str(ts),
        body_md5,
        _sorted_query(path_with_query),
    ])
    signature = base64.b64encode(
        hmac.new(SECRET_BYTES, canonical.encode("utf-8"), hashlib.md5).digest()
    ).decode()

    stamp = f"{ts_sec}{ms:03d}000{rand}"
    build_hash = _md5_hex((CERT_SHA + "oneroom" + stamp).encode("utf-8"))
    build_tag = f"{stamp}.{build_hash}"

    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
        "Content-Type": "application/json",
        "x-client-token": f"{ts},{_md5_hex(str(ts)[::-1].encode('utf-8'))}",
        "x-tr-signature": f"{ts}|2|{signature}",
        "x-client-build": build_tag,
        "x-client-status": "0",
        "x-client-info": _build_client_info(build_tag),
        "x-forwarded-for": f"{random.randint(49, 117)}.{random.randint(1, 254)}.{random.randint(1, 254)}.{random.randint(1, 254)}",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


class MovieBoxClient:
    """Signed client with host rotation + bearer bootstrap + one re-login."""

    def __init__(self) -> None:
        self._token: Optional[str] = None
        self._token_lock = threading.Lock()

    # -- token ------------------------------------------------------------

    def _login(self, session: requests.Session) -> str:
        path = f"{BFF}/user-api/visitor-login"
        last_err: Optional[Exception] = None
        for _ in range(len(HOSTS)):
            host = self._next_host()
            try:
                r = session.post(
                    host + path,
                    data="{}",
                    headers=_build_headers("POST", path, "{}", None),
                    timeout=REQUEST_TIMEOUT,
                )
                if r.status_code == 200:
                    token = (r.json().get("data") or {}).get("token")
                    if token:
                        return token
                last_err = RuntimeError(f"login HTTP {r.status_code}")
            except Exception as e:
                last_err = e
        raise ConnectionError(f"visitor-login failed on all hosts: {last_err}")

    def ensure_token(self, session: requests.Session) -> str:
        with self._token_lock:
            if not self._token:
                self._token = self._login(session)
            return self._token

    def invalidate(self) -> None:
        with self._token_lock:
            self._token = None

    # -- host rotation ------------------------------------------------------

    @staticmethod
    def _next_host() -> str:
        global _active_host_idx
        with _host_lock:
            host = HOSTS[_active_host_idx % len(HOSTS)]
            _active_host_idx += 1
            return host

    # -- requests -----------------------------------------------------------

    def get(self, path_with_query: str) -> Dict[str, Any]:
        return self._request("GET", path_with_query, None)

    def post(self, path_with_query: str, body: str) -> Dict[str, Any]:
        return self._request("POST", path_with_query, body)

    def _request(self, method: str, path_with_query: str, body: Optional[str]) -> Dict[str, Any]:
        session = _thread_session()
        last_err: Optional[Exception] = None
        for attempt in range(len(HOSTS) + 2):
            host = self._next_host()
            try:
                token = self.ensure_token(session)
                r = session.request(
                    method,
                    host + path_with_query,
                    data=body,
                    headers=_build_headers(method, path_with_query, body, token),
                    timeout=REQUEST_TIMEOUT,
                )
                if r.status_code in (401, 403):
                    self.invalidate()
                    continue
                if r.status_code in RETRY_STATUSES:
                    last_err = RuntimeError(f"HTTP {r.status_code}")
                    time.sleep(0.3)
                    continue
                if r.status_code == 200:
                    payload = r.json()
                    return payload.get("data") if isinstance(payload.get("data"), (dict, list)) else payload
                last_err = RuntimeError(f"HTTP {r.status_code}")
            except Exception as e:
                last_err = e
                time.sleep(0.3)
        raise ConnectionError(f"{method} {path_with_query} failed: {last_err}")


def _thread_session() -> requests.Session:
    session = getattr(_SESSION_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=4, pool_maxsize=4, max_retries=0
        )
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        _SESSION_LOCAL.session = session
    return session


# ---------------------------------------------------------------------------
# Crawling: ranking-list (8 categories) + tab-operating (homepage groups)
# ---------------------------------------------------------------------------

PLACEHOLDER_MARKERS = (
    "1c7de0bd3393702d9191801f15f88f8d",
    "9a0461bc39da389663bf3dbb17091d3f",
    "b164fbfb4347792950bdfbfb563d39d9",
    "/notice.mp4",
)


def crawl_categories(client: MovieBoxClient) -> List[Dict[str, Any]]:
    data = client.get(f"{BFF}/tab/ranking-list?tabId=0&categoryType=0&page=1&perPage={PAGE_SIZE}")
    cats = data.get("categoryList") or []
    out = [{"id": str(c.get("type")), "name": c.get("name", "")} for c in cats]
    print(f"[Crawler] {len(out)} ranking categories")
    return out


def crawl_ranking_category(client: MovieBoxClient, cat_id: str, cat_name: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for page in range(1, MAX_PAGES_PER_SOURCE + 1):
        data = client.get(
            f"{BFF}/tab/ranking-list?tabId=0&categoryType={cat_id}&page={page}&perPage={PAGE_SIZE}"
        )
        subjects = data.get("subjects") or []
        items.extend(subjects)
        if not (data.get("pager") or {}).get("hasMore") or not subjects:
            break
        time.sleep(0.15)
    print(f"[Crawler] ranking '{cat_name}': {len(items)} items")
    return items


def crawl_homepage(client: MovieBoxClient) -> List[Dict[str, Any]]:
    data = client.get(f"{BFF}/tab-operating?tab=home")
    groups = data.get("items") or []
    subjects: List[Dict[str, Any]] = []
    for g in groups:
        if not isinstance(g, dict):
            continue
        for source in [
            *(g.get("subjects") or []),
            *[
                b.get("subject")
                for b in ((g.get("banner") or {}).get("banners") or [])
                if isinstance(b, dict) and isinstance(b.get("subject"), dict)
            ],
            *[
                c.get("subject")
                for c in ((g.get("customData") or {}).get("items") or [])
                if isinstance(c, dict) and isinstance(c.get("subject"), dict)
            ],
        ]:
            if isinstance(source, dict):
                subjects.append(source)
    print(f"[Crawler] homepage groups: {len(groups)}, subjects: {len(subjects)}")
    return subjects


def crawl_subject_details(client: MovieBoxClient, subject_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """Fetch subject-api/get for description/dubs (embedding corpus depth)."""
    details: Dict[str, Dict[str, Any]] = {}

    def fetch(sid: str) -> Optional[Dict[str, Any]]:
        try:
            return client.get(f"{BFF}/subject-api/get?subjectId={sid}")
        except Exception as e:
            print(f"[Detail] {sid} failed: {e}")
            return None

    with ThreadPoolExecutor(max_workers=DETAIL_WORKERS) as pool:
        futures = {pool.submit(fetch, sid): sid for sid in subject_ids}
        for fut in as_completed(futures):
            d = fut.result()
            if d:
                details[futures[fut]] = d
    print(f"[Crawler] details fetched: {len(details)}/{len(subject_ids)}")
    return details


def _extract_search_subjects(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    subjects: List[Dict[str, Any]] = []
    results = payload.get("results") or payload.get("list")
    if isinstance(results, list):
        for group in results:
            if isinstance(group, dict):
                subjects.extend(s for s in (group.get("subjects") or []) if isinstance(s, dict))
            elif isinstance(group, dict) is False and isinstance(group, list):
                subjects.extend(s for s in group if isinstance(s, dict))
    if not subjects and isinstance(payload.get("list"), list):
        subjects.extend(s for s in payload["list"] if isinstance(s, dict))
    return subjects


def crawl_search_seed(client: MovieBoxClient, seed: str) -> List[Dict[str, Any]]:
    collected: List[Dict[str, Any]] = []
    for page in range(1, SEARCH_MAX_PAGES + 1):
        try:
            data = client.post(
                f"{BFF}/subject-api/search/v2",
                json.dumps({"keyword": seed, "page": page, "perPage": SEARCH_PAGE_SIZE, "subjectType": 0}),
            )
        except Exception:
            break
        subjects = _extract_search_subjects(data)
        if not subjects:
            break
        collected.extend(subjects)
        time.sleep(SEARCH_PAGE_DELAY)
    return collected


def crawl_search_sweep(client: MovieBoxClient) -> Dict[str, Dict[str, Any]]:
    """Sweep keyword seeds through search/v2; dedup by subjectId."""
    by_id: Dict[str, Dict[str, Any]] = {}
    started = time.time()
    with ThreadPoolExecutor(max_workers=SEARCH_WORKERS) as pool:
        futures = {pool.submit(crawl_search_seed, client, seed): seed for seed in SEARCH_SEEDS}
        done = 0
        for fut in as_completed(futures):
            seed = futures[fut]
            try:
                subjects = fut.result()
            except Exception as e:
                print(f"[Search] seed '{seed}' failed: {e}")
                subjects = []
            new = 0
            for s in subjects:
                sid = str(s.get("subjectId") or "")
                if sid and sid not in by_id:
                    by_id[sid] = s
                    new += 1
            done += 1
            print(f"[Search] '{seed}': {len(subjects)} raw, +{new} new (unique {len(by_id)}) [{done}/{len(SEARCH_SEEDS)}]")
    print(f"[Search] sweep complete: {len(by_id)} unique subjects in {time.time() - started:.0f}s")
    return by_id


# ---------------------------------------------------------------------------
# ML embeddings + SQLite (same schema as the CircleFTP indexer)
# ---------------------------------------------------------------------------


def compute_ml_embeddings(rows: List[Dict[str, Any]], dim: int = 64) -> List[bytes]:
    print(f"\n[ML Compiler] Generating {dim}-dim universal feature-hash embeddings for {len(rows)} items...")
    packed_blobs = [
        compute_embedding_bytes(
            title=r.get("title", "") or "",
            synopsis=r.get("synopsis", "") or None,
            genres=r.get("genres", "") or None,
            category=r.get("category_id", "") or None,
        )
        for r in rows
    ]
    print(f"[ML Compiler] Compiled {len(packed_blobs)} vectors (256 bytes each).")
    return packed_blobs


def compile_sqlite_database(
    db_path: str,
    categories: List[Dict[str, Any]],
    rows: List[Dict[str, Any]],
    embeddings: List[bytes],
) -> None:
    print(f"\n[Database] Compiling SQLite at {db_path}...")
    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode = WAL;")
    cur.execute("PRAGMA synchronous = NORMAL;")

    cur.execute("""
        CREATE TABLE plugin_info (key TEXT PRIMARY KEY, value TEXT NOT NULL);
    """)
    cur.execute("INSERT INTO plugin_info VALUES ('id', 'moviebox.aoneroom');")
    cur.execute("INSERT INTO plugin_info VALUES ('name', 'MovieBox');")
    cur.execute(f"INSERT INTO plugin_info VALUES ('compiled_at', '{int(time.time())}');")
    cur.execute(f"INSERT INTO plugin_info VALUES ('total_items', '{len(rows)}');")
    cur.execute("INSERT INTO plugin_info VALUES ('vector_format', 'md5_64');")

    cur.execute("""
        CREATE TABLE categories (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            parent_id TEXT,
            item_count INTEGER DEFAULT 0
        );
    """)
    for cat in categories:
        cur.execute(
            "INSERT OR REPLACE INTO categories VALUES (?, ?, ?, ?)",
            (cat["id"], cat["name"], None, 0),
        )

    cur.execute("""
        CREATE TABLE media_items (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            type TEXT NOT NULL,
            category_id TEXT NOT NULL,
            sub_category_id TEXT,
            poster_url TEXT,
            banner_url TEXT,
            year TEXT,
            quality TEXT,
            duration_text TEXT,
            synopsis TEXT,
            rating REAL DEFAULT 0.0,
            genres TEXT,
            created_at INTEGER NOT NULL,
            extra_json TEXT,
            embedding BLOB NOT NULL,
            imdb_id TEXT
        );
    """)

    cur.execute("""
        CREATE VIRTUAL TABLE media_fts USING fts5(
            title, synopsis, genres,
            content='media_items',
            content_rowid='rowid'
        );
    """)

    print("[Database] Inserting media records...")
    now = int(time.time())
    for idx, row in enumerate(rows):
        cur.execute(
            """
            INSERT OR REPLACE INTO media_items (
                id, title, type, category_id, sub_category_id,
                poster_url, banner_url, year, quality, duration_text,
                synopsis, rating, genres, created_at, extra_json, embedding, imdb_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["id"],
                row["title"],
                row["type"],
                row["category_id"],
                None,
                row["poster"],
                None,
                row["year"],
                row["quality"],
                row["duration"],
                row["synopsis"],
                row["rating"],
                row["genres"],
                now,
                json.dumps({"viewers": row.get("viewers"), "country": row.get("country")}),
                embeddings[idx],
                row.get("imdb_id"),
            ),
        )
        cur.execute(
            "INSERT INTO media_fts(rowid, title, synopsis, genres) VALUES (last_insert_rowid(), ?, ?, ?)",
            (row["title"], row["synopsis"], row["genres"]),
        )

    cur.execute("CREATE INDEX idx_media_cat ON media_items(category_id, type, created_at DESC);")
    cur.execute("CREATE INDEX idx_media_type ON media_items(type, created_at DESC);")
    conn.commit()
    conn.close()
    print(f"[Database] Built: {os.path.getsize(db_path) / (1024 * 1024):.2f} MB")


def to_rows(
    raw_items: List[Dict[str, Any]],
    details: Dict[str, Dict[str, Any]],
    category_by_subject: Dict[str, str],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen: set = set()
    for s in raw_items:
        sid = str(s.get("subjectId") or "")
        if not sid or sid in seen:
            continue
        title = (s.get("title") or "").strip()
        if not title:
            continue
        detail = details.get(sid) or {}
        genre = (s.get("genre") or detail.get("genre") or "").strip()
        description = (detail.get("description") or s.get("description") or "").strip()
        # Subject-type 9 is FightZone/sports clips, 6 is music videos — keep
        # movie/series only, mirroring the client's MediaFilter.
        stype = s.get("subjectType")
        m_type = "series" if stype == 2 else "movie"
        if stype not in (1, 2):
            continue
        rating = 0.0
        for key in ("imdbRatingValue", "imdbRate", "rate"):
            try:
                value = float(s.get(key) or 0)
                if value > 0:
                    rating = value
                    break
            except (TypeError, ValueError):
                continue
        duration_seconds = s.get("seconds") or detail.get("durationSeconds") or 0
        try:
            duration_seconds = int(duration_seconds)
        except (TypeError, ValueError):
            duration_seconds = 0
        duration = f"{duration_seconds // 60}m" if duration_seconds > 0 else ""
        cover = ((s.get("cover") or {}).get("url") or "").strip()
        category_id = category_by_subject.get(sid, "0")
        rows.append({
            "id": sid,
            "title": title,
            "type": m_type,
            "category_id": category_id,
            "poster": cover or None,
            "year": (s.get("releaseDate") or "")[:10],
            "quality": f"IMDb {rating:.1f}" if rating > 0 else "",
            "duration": duration,
            "synopsis": description,
            "rating": rating,
            "genres": genre,
            "viewers": s.get("viewers") or 0,
            "country": s.get("countryName") or "",
        })
        seen.add(sid)
    return rows


def compress_database(db_path: str) -> str:
    gz_path = f"{db_path}.gz"
    print(f"\n[Compressor] {db_path} -> {gz_path}")
    with open(db_path, "rb") as f_in, gzip.open(gz_path, "wb", compresslevel=9) as f_out:
        f_out.writelines(f_in)
    print(f"[Compressor] {os.path.basename(gz_path)}: {os.path.getsize(gz_path) / (1024 * 1024):.2f} MB")
    return gz_path


def compute_sha256(file_path: str) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    digest = h.hexdigest().lower()
    with open(f"{file_path}.sha256", "w") as f:
        f.write(f"{digest}  {os.path.basename(file_path)}\n")
    print(f"[Checksum] {os.path.basename(file_path)}: {digest}")
    return digest


def main() -> None:
    parser = argparse.ArgumentParser(description="Fensy MovieBox Indexer & ML Compiler")
    parser.add_argument("--outdir", default="dist")
    parser.add_argument("--skip-details", action="store_true", help="Skip subject detail fetch (faster, thinner corpus)")
    parser.add_argument("--skip-search", action="store_true", help="Skip search-seed sweep (curated ranking+homepage only)")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    client = MovieBoxClient()

    # 1. Crawl: ranking categories + homepage groups, dedup by subjectId.
    categories = crawl_categories(client)

    raw_by_id: Dict[str, Dict[str, Any]] = {}
    category_by_subject: Dict[str, str] = {}

    with ThreadPoolExecutor(max_workers=PAGE_WORKERS) as pool:
        futures = {
            pool.submit(crawl_ranking_category, client, c["id"], c["name"]): c
            for c in categories
        }
        for fut in as_completed(futures):
            cat = futures[fut]
            try:
                for s in fut.result():
                    sid = str(s.get("subjectId") or "")
                    if sid:
                        raw_by_id.setdefault(sid, s)
                        category_by_subject.setdefault(sid, cat["id"])
            except Exception as e:
                print(f"[Crawler] category {cat['id']} failed: {e}")

    try:
        for s in crawl_homepage(client):
            sid = str(s.get("subjectId") or "")
            if sid:
                raw_by_id.setdefault(sid, s)
    except Exception as e:
        print(f"[Crawler] homepage crawl failed: {e}")

    ranking_count = len(raw_by_id)
    print(f"\n[Summary] Curated subjects (ranking + homepage): {ranking_count}")

    # 1b. Search-seed sweep: expands past the curated ~700 to the full catalog.
    if not args.skip_search:
        try:
            for sid, s in crawl_search_sweep(client).items():
                if sid not in raw_by_id:
                    raw_by_id[sid] = s
        except Exception as e:
            print(f"[Crawler] search sweep failed (continuing with curated): {e}")
    print(f"[Summary] Unique subjects after sweep: {len(raw_by_id)}")

    raw_items = list(raw_by_id.values())

    # 2. Details for embedding corpus (descriptions are empty on cards).
    details: Dict[str, Dict[str, Any]] = {}
    if not args.skip_details:
        try:
            details = crawl_subject_details(client, [s for s in raw_by_id.keys()])
        except Exception as e:
            print(f"[Crawler] detail pass failed (continuing without): {e}")

    rows = to_rows(raw_items, details, category_by_subject)
    if not rows:
        print("[Error] No rows compiled. Exiting.")
        sys.exit(1)
    print(f"[Summary] Rows after filters: {len(rows)}")

    def _apply_imdb(row: Dict[str, Any], imdb_id: str, genres: Optional[str], year: Optional[int]) -> None:
        row["imdb_id"] = imdb_id
        if genres and not row.get("genres"):
            row["genres"] = genres
        if year and not row.get("year"):
            row["year"] = str(year)

    enrich_with_imdb(
        rows,
        get_title=lambda r: r.get("title", "") or "",
        get_year=lambda r: r.get("year"),
        is_series_fn=lambda r: r.get("type") == "series",
        on_match=_apply_imdb,
    )

    embeddings = compute_ml_embeddings(rows, dim=64)

    # 4. SQLite + compression + checksum.
    db_path = os.path.join(args.outdir, "moviebox.db")
    compile_sqlite_database(db_path, categories, rows, embeddings)
    gz_path = compress_database(db_path)
    compute_sha256(gz_path)
    compute_sha256(db_path)

    print("\n[Fensy MovieBox Indexer] Complete!")


if __name__ == "__main__":
    main()
