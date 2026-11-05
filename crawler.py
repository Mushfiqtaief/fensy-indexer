#!/usr/bin/env python3
"""
Fensy Universal Media Indexer & Machine Learning Vector Compiler
Crawls BDIX / FTP catalog sources through socks5 / Mihomo proxy,
generates 64-dimensional semantic recommendation vectors, and compiles
a lightweight, compressed SQLite database for 0ms client mounting.
"""

import os
import re
import sys
import time
import json
import sqlite3
import hashlib
import struct
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Tuple

import requests
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

# Fallback proxies if local Mihomo daemon is not running
DIRECT_SOCKS_PROXIES = [
    "socks5://103.186.238.118:11911",  # Node 11
    "socks5://103.244.49.181:11911",  # Node 8
    "socks5://175.29.161.201:11911",  # Node 1
]

BASE_URL = "http://new.circleftp.net:5000"

PAGE_SIZE = 100
PAGE_WORKERS = 8
CATEGORY_WORKERS = 4
PAGE_RETRIES = 3
PAGE_TIMEOUT = 20
SESSION_LOCAL = threading.local()


def get_active_session() -> requests.Session:
    """Detects and configures an active proxy connection to CircleFTP."""
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) FensyIndexer/2.0",
        "Connection": "keep-alive",
        "Accept": "application/json",
    })

    # 1. Try local Mihomo proxy if running
    for local_proxy in ["http://127.0.0.1:7890", "http://127.0.0.1:30609"]:
        try:
            print(f"[Proxy] Testing local Mihomo proxy at {local_proxy}...")
            r = requests.get(
                f"{BASE_URL}/api/categories",
                proxies={"http": local_proxy, "https": local_proxy},
                timeout=4,
            )
            if r.status_code == 200 and len(r.json()) > 0:
                print(f"[Proxy] Connected successfully via Mihomo ({local_proxy})!")
                _configure_session(session, {"http": local_proxy, "https": local_proxy})
                return session
        except Exception:
            pass

    # 2. Try direct SOCKS5 proxies
    for socks_proxy in DIRECT_SOCKS_PROXIES:
        try:
            print(f"[Proxy] Testing direct SOCKS5 proxy at {socks_proxy}...")
            r = requests.get(
                f"{BASE_URL}/api/categories",
                proxies={"http": socks_proxy, "https": socks_proxy},
                timeout=5,
            )
            if r.status_code == 200 and len(r.json()) > 0:
                print(f"[Proxy] Connected successfully via SOCKS5 ({socks_proxy})!")
                _configure_session(session, {"http": socks_proxy, "https": socks_proxy})
                return session
        except Exception as e:
            print(f"[Proxy] {socks_proxy} unavailable: {e}")

    # 3. Direct connection attempt (if running in Bangladesh network)
    try:
        print("[Proxy] Testing direct connection without proxy...")
        r = requests.get(f"{BASE_URL}/api/categories", timeout=3)
        if r.status_code == 200:
            print("[Proxy] Connected directly!")
            return session
    except Exception:
        pass

    raise ConnectionError("Could not establish proxy connection to CircleFTP.")


_PROXY_CONFIG: Dict[str, str] = {}


def _configure_session(session: requests.Session, proxies: Dict[str, str]) -> None:
    """Applies proxies plus a connection pool sized for parallel page fetches."""
    global _PROXY_CONFIG
    _PROXY_CONFIG = dict(proxies)
    session.proxies.update(proxies)
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=PAGE_WORKERS * 2,
        pool_maxsize=PAGE_WORKERS * 2,
        max_retries=0,
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)


def thread_session() -> requests.Session:
    """One pooled session per worker thread (requests sessions are not thread-safe)."""
    session = getattr(SESSION_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) FensyIndexer/2.0",
            "Connection": "keep-alive",
            "Accept": "application/json",
        })
        if _PROXY_CONFIG:
            session.proxies.update(_PROXY_CONFIG)
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=4, pool_maxsize=4, max_retries=0
        )
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        SESSION_LOCAL.session = session
    return session



BLOCKED_CATEGORY_KEYWORDS = {
    'software', 'softwares', 'game', 'games', 'pc games', 'pc game',
    'console games', 'linux games', 'game mods',
    'apk', 'android apps', 'android games', 'app', 'apps', 'application', 'applications',
    'mobile apps', 'iphone apps',
    'pdf', 'book', 'books', 'ebook', 'ebooks', 'magazine', 'magazines',
    'comic', 'comics', 'tutorial', 'tutorials', 'operating system', 'os', 'windows os',
    'tools', 'utilities', 'audiobook', 'audiobooks', 'firmware'
}

BLOCKED_CATEGORY_IDS = {'11', '12', '13', '14', '36'}

BLOCKED_EXTENSIONS = (
    '.apk', '.exe', '.pdf', '.iso', '.zip', '.rar', '.7z', '.dmg', '.pkg',
    '.epub', '.mobi', '.msi', '.bin', '.tar', '.gz', '.cbr', '.cbz'
)

# Titles that are software/games/books without a file-extension hint, e.g.
# "Circle Network Apk", "Smart Switch PC Setup Samsung", "ModelSim (PC)".
BLOCKED_TITLE_KEYWORDS = (
    'apk', 'apks', 'pc software', 'software', 'portable', 'repack', 'repacks',
    'multiboot', 'keymaker', 'keygen', 'crack', 'cracked', 'activator',
    'driver', 'drivers', 'setup', 'installer', 'toolkit',
    'pc game', 'pc games', 'android app', 'android apps',
    'android game', 'android games',
    'windows os', 'windows 7', 'windows 8', 'windows 10', 'windows 11',
    'macos', 'pcmover', 'pctrans', 'after effects', 'premiere pro',
    'photoshop', 'illustrator', 'modelsim', 'matlab', 'autocad',
    'easeus', 'probook', 'notebook pc', 'iso',
)

_BLOCKED_TITLE_RE = re.compile(
    r'\b(' + '|'.join(re.escape(k) for k in BLOCKED_TITLE_KEYWORDS) + r')\b',
    re.IGNORECASE,
)
_BLOCKED_PC_TAG_RE = re.compile(r'\(pc\)', re.IGNORECASE)


def is_blocked_category(name: str) -> bool:
    lower = (name or "").lower().strip()
    if not lower:
        return False
    for kw in BLOCKED_CATEGORY_KEYWORDS:
        if re.search(r'\b' + re.escape(kw) + r'\b', lower):
            return True
    return False


def is_blocked_category_id(cat_id: Any) -> bool:
    return str(cat_id) in BLOCKED_CATEGORY_IDS if cat_id is not None else False


def is_blocked_media(title: str, synopsis: str = "") -> bool:
    t = (title or "").lower()
    s = (synopsis or "").lower()
    for ext in BLOCKED_EXTENSIONS:
        if t.endswith(ext) or f"{ext} " in t or f"{ext}]" in t or s.endswith(ext):
            return True
    if _BLOCKED_TITLE_RE.search(t) or _BLOCKED_PC_TAG_RE.search(t):
        return True
    return False



def crawl_categories(session: requests.Session) -> List[Dict[str, Any]]:
    print("\n[Crawler] Fetching categories...")
    r = session.get(f"{BASE_URL}/api/categories", timeout=10)
    r.raise_for_status()
    raw_categories = r.json()
    
    categories = []
    for cat in raw_categories:
        name = cat.get("name", "")
        if is_blocked_category(name):
            continue
        
        filtered_sub = [
            sub for sub in cat.get("subCategory", [])
            if not is_blocked_category(sub.get("name", ""))
        ]
        cat["subCategory"] = filtered_sub
        categories.append(cat)
        
    print(f"[Crawler] Found {len(categories)} root video categories (filtered non-video).")
    return categories


def _fetch_page(cat_id: Any, page: int) -> Optional[Dict[str, Any]]:
    url = f"{BASE_URL}/api/posts"
    params = {
        "categoryExact": cat_id,
        "page": page,
        "limit": PAGE_SIZE,
        "order": "desc",
    }
    for attempt in range(PAGE_RETRIES):
        try:
            r = thread_session().get(url, params=params, timeout=PAGE_TIMEOUT)
            if r.status_code == 200:
                return r.json()
            print(f"  [Warning] HTTP {r.status_code} on page {page}, retrying...")
        except Exception as e:
            print(f"\n  [Error] Failed page {page}: {e}. Retrying...")
        time.sleep(0.5 * (attempt + 1))
    return None


def crawl_category_posts(
    session: requests.Session,
    cat_id: Any,
    cat_name: str,
    max_pages: Optional[int] = None,
) -> List[Dict[str, Any]]:
    print(f"\n[Crawler] Crawling category [{cat_id}] '{cat_name}'...")

    first = _fetch_page(cat_id, 1)
    if not first:
        print(f"  -> Failed category '{cat_name}'.")
        return []

    items = list(first.get("posts") or [])
    pagination = first.get("pagination") or {}
    total_pages = int(pagination.get("pages", 1) or 1)
    total_items = pagination.get("total", len(items))
    if max_pages:
        total_pages = min(total_pages, max_pages)

    if total_pages > 1:
        with ThreadPoolExecutor(max_workers=PAGE_WORKERS) as pool:
            futures = {
                pool.submit(_fetch_page, cat_id, p): p
                for p in range(2, total_pages + 1)
            }
            for fut in as_completed(futures):
                data = fut.result()
                if data:
                    items.extend(data.get("posts") or [])

    # Page order is lost when parallel; restore newest-first via id.
    items.sort(key=lambda p: int(p.get("id") or 0), reverse=True)
    print(f"\n  -> Finished category '{cat_name}': {len(items)}/{total_items} items across {total_pages} pages.")
    return items



def compute_ml_embeddings(items: List[Dict[str, Any]], dim: int = 64) -> List[bytes]:
    """
    Computes a 64-dimensional semantic embedding for each media item using
    TF-IDF feature extraction + TruncatedSVD dimensionality reduction.
    The resulting vector is L2-normalized so cosine similarity is equal to dot product.
    Packed into binary BLOB (64 * 4 = 256 bytes).
    """
    print(f"\n[ML Compiler] Generating {dim}-dimensional semantic embeddings for {len(items)} items...")

    corpus = []
    for item in items:
        title = item.get("title", "") or ""
        synopsis = item.get("metaData", "") or ""
        tags = item.get("tags", "") or ""
        quality = item.get("quality", "") or ""
        year = str(item.get("year", "")) or ""

        genres_list = [c.get("name", "") for c in item.get("categories", []) if isinstance(c, dict)]
        genres = " ".join(genres_list)

        text = f"{title} {title} {genres} {genres} {tags} {year} {quality} {synopsis}"
        corpus.append(text)

    # 1. TF-IDF representation
    tfidf = TfidfVectorizer(
        max_features=12000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        stop_words="english",
        dtype=np.float32,
    )
    X = tfidf.fit_transform(corpus)

    # 2. Dimensionality reduction to 64 components. Randomized SVD keeps this
    #    near-linear instead of the O(n^2) exact path.
    actual_dim = min(dim, X.shape[1], len(items) - 1)
    if actual_dim < dim:
        actual_dim = max(2, actual_dim)

    svd = TruncatedSVD(
        n_components=actual_dim,
        random_state=42,
        algorithm="randomized",
        n_iter=5,
    )
    X_reduced = svd.fit_transform(X).astype(np.float32, copy=False)

    # Pad with zeros if fewer than target dimensions
    if actual_dim < dim:
        padded = np.zeros((X_reduced.shape[0], dim), dtype=np.float32)
        padded[:, :actual_dim] = X_reduced
        X_reduced = padded

    # 3. L2 Normalize (Cosine similarity becomes standard dot product)
    X_normalized = normalize(X_reduced, norm="l2")

    # 4. Pack into binary buffers (64 floats * 4 bytes = 256 bytes)
    packed_blobs = [row.tobytes() for row in X_normalized]

    print(f"[ML Compiler] Successfully compiled {len(packed_blobs)} vectors (256 bytes each).")
    return packed_blobs


def compile_sqlite_database(
    db_path: str,
    categories: List[Dict[str, Any]],
    items: List[Dict[str, Any]],
    embeddings: List[bytes],
):
    print(f"\n[Database] Compiling SQLite database at {db_path}...")
    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("PRAGMA journal_mode = WAL;")
    cur.execute("PRAGMA synchronous = NORMAL;")

    # 1. Plugin info
    cur.execute("""
        CREATE TABLE plugin_info (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
    """)
    cur.execute("INSERT INTO plugin_info VALUES ('id', 'bdix.circleftp');")
    cur.execute("INSERT INTO plugin_info VALUES ('name', 'Circle FTP');")
    cur.execute(f"INSERT INTO plugin_info VALUES ('compiled_at', '{int(time.time())}');")
    cur.execute(f"INSERT INTO plugin_info VALUES ('total_items', '{len(items)}');")

    # 2. Categories
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
            (str(cat.get("id")), cat.get("name", ""), str(cat.get("parentId")) if cat.get("parentId") else None, 0),
        )
        for sub in cat.get("subCategory", []):
            cur.execute(
                "INSERT OR REPLACE INTO categories VALUES (?, ?, ?, ?)",
                (str(sub.get("id")), sub.get("name", ""), str(cat.get("id")), 0),
            )

    # 3. Media Items with Embedding BLOB
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
            embedding BLOB NOT NULL
        );
    """)

    # 4. Episodes Table
    cur.execute("""
        CREATE TABLE episodes (
            id TEXT PRIMARY KEY,
            media_id TEXT NOT NULL,
            season_num INTEGER NOT NULL DEFAULT 1,
            season_name TEXT DEFAULT 'Season 1',
            episode_num INTEGER NOT NULL,
            title TEXT NOT NULL,
            stream_url TEXT NOT NULL,
            duration_text TEXT,
            FOREIGN KEY(media_id) REFERENCES media_items(id)
        );
    """)

    # 5. Full-Text Search Virtual Table
    cur.execute("""
        CREATE VIRTUAL TABLE media_fts USING fts5(
            title, synopsis, genres,
            content='media_items',
            content_rowid='rowid'
        );
    """)

    # Insert items
    print("[Database] Inserting media records...")
    for idx, item in enumerate(items):
        item_id = str(item.get("id"))
        title = item.get("title", "") or ""
        synopsis = item.get("metaData", "") or ""
        if is_blocked_media(title, synopsis):
            continue

        m_type = "series" if item.get("type") == "series" else "movie"
        image_sm = item.get("imageSm")
        image_lg = item.get("image")
        poster = f"{BASE_URL}/uploads/{image_sm}" if image_sm else None
        banner = f"{BASE_URL}/uploads/{image_lg}" if image_lg else None

        cats = item.get("categories", [])
        cat_id = str(cats[0].get("id")) if cats else "0"
        sub_cat_id = str(cats[1].get("id")) if len(cats) > 1 else None

        if is_blocked_category_id(cat_id) or is_blocked_category_id(sub_cat_id):
            continue
        if any(is_blocked_category(c.get("name", "")) for c in cats if isinstance(c, dict)):
            continue

        genres = ", ".join([c.get("name", "") for c in cats if isinstance(c, dict)])

        created_epoch = int(time.time())
        created_str = item.get("createdAt")
        if created_str:
            try:
                # Approximate parse
                created_epoch = int(time.mktime(time.strptime(created_str[:10], "%Y-%m-%d")))
            except Exception:
                pass

        cur.execute(
            """
            INSERT OR REPLACE INTO media_items (
                id, title, type, category_id, sub_category_id,
                poster_url, banner_url, year, quality, duration_text,
                synopsis, rating, genres, created_at, extra_json, embedding
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                item_id,
                title,
                m_type,
                cat_id,
                sub_cat_id,
                poster,
                banner,
                str(item.get("year", "") or ""),
                str(item.get("quality", "") or ""),
                str(item.get("watchTime", "") or ""),
                str(item.get("metaData", "") or ""),
                0.0,
                genres,
                created_epoch,
                json.dumps({"tags": item.get("tags")}),
                embeddings[idx],
            ),
        )

        # FTS insert
        cur.execute(
            "INSERT INTO media_fts(rowid, title, synopsis, genres) VALUES (last_insert_rowid(), ?, ?, ?)",
            (title, str(item.get("metaData", "") or ""), genres),
        )

    # 6. Indices
    print("[Database] Building performance indices...")
    cur.execute("CREATE INDEX idx_media_cat ON media_items(category_id, type, created_at DESC);")
    cur.execute("CREATE INDEX idx_media_type ON media_items(type, created_at DESC);")
    cur.execute("CREATE INDEX idx_episodes ON episodes(media_id, season_num, episode_num);")

    conn.commit()
    conn.close()
    db_size = os.path.getsize(db_path)
    print(f"[Database] SQLite database successfully built: {db_size / (1024*1024):.2f} MB")


def compress_database(db_path: str) -> List[str]:
    compressed_files = []
    
    zst_path = f"{db_path}.zst"
    print(f"\n[Compressor] Compressing {db_path} -> {zst_path}...")
    try:
        import zstandard as zstd
        cctx = zstd.ZstdCompressor(level=19)
        with open(db_path, "rb") as ifh, open(zst_path, "wb") as ofh:
            cctx.copy_stream(ifh, ofh)
        compressed_files.append(zst_path)
        print(f"[Compressor] Created {os.path.basename(zst_path)}: {os.path.getsize(zst_path) / (1024*1024):.2f} MB")
    except Exception as e:
        print(f"[Compressor] Zstandard compression warning: {e}")

    gz_path = f"{db_path}.gz"
    print(f"\n[Compressor] Compressing {db_path} -> {gz_path}...")
    try:
        import gzip
        with open(db_path, "rb") as f_in, gzip.open(gz_path, "wb", compresslevel=9) as f_out:
            f_out.writelines(f_in)
        compressed_files.append(gz_path)
        print(f"[Compressor] Created {os.path.basename(gz_path)}: {os.path.getsize(gz_path) / (1024*1024):.2f} MB")
    except Exception as e:
        print(f"[Compressor] Gzip compression warning: {e}")

    return compressed_files


def compute_sha256(file_path: str) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    digest = h.hexdigest().lower()
    sha_path = f"{file_path}.sha256"
    with open(sha_path, "w") as f:
        f.write(f"{digest}  {os.path.basename(file_path)}\n")
    print(f"[Checksum] {os.path.basename(file_path)}: {digest}")
    return digest


def generate_manifest(
    output_dir: str,
    db_filename: str,
    sha256: str,
    total_items: int,
    download_url: str,
):
    manifest = {
        "version": int(time.time()),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "plugin_id": "bdix.circleftp",
        "total_items": total_items,
        "filename": db_filename,
        "sha256": sha256,
        "download_url": download_url,
    }
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[Manifest] Generated manifest at {manifest_path}")


def main():
    parser = argparse.ArgumentParser(description="Fensy CircleFTP Indexer & ML Compiler")
    parser.add_argument("--outdir", default="dist", help="Output directory for compiled database")
    parser.add_argument("--max-pages-per-cat", type=int, default=None, help="Debug limit for pages per category")
    parser.add_argument("--release-url-prefix", default="https://github.com/Mushfiqtaief/fensy-indexer/releases/download/latest", help="CDN URL prefix")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    session = get_active_session()

    # 1. Crawl
    categories = crawl_categories(session)
    all_items = []
    seen_ids = set()

    started = time.time()
    with ThreadPoolExecutor(max_workers=CATEGORY_WORKERS) as pool:
        futures = {
            pool.submit(
                crawl_category_posts,
                session,
                cat.get("id"),
                cat.get("name", "Unknown"),
                args.max_pages_per_cat,
            ): cat
            for cat in categories
        }
        for fut in as_completed(futures):
            cat = futures[fut]
            try:
                posts = fut.result()
            except Exception as e:
                print(f"[Crawler] Category {cat.get('id')} failed: {e}")
                continue
            for p in posts:
                pid = str(p.get("id"))
                if pid not in seen_ids:
                    seen_ids.add(pid)
                    all_items.append(p)

    print(f"\n[Summary] Total unique items collected: {len(all_items)} in {time.time() - started:.1f}s")

    # Drop non-video posts before embedding so software/APK rows never occupy
    # vector space or enter the feed.
    video_items = [
        p for p in all_items
        if not is_blocked_media(p.get("title", "") or "", p.get("metaData", "") or "")
        and not any(
            is_blocked_category(c.get("name", ""))
            for c in (p.get("categories") or [])
            if isinstance(c, dict)
        )
        and not any(
            is_blocked_category_id(c.get("id"))
            for c in (p.get("categories") or [])
            if isinstance(c, dict)
        )
    ]
    dropped = len(all_items) - len(video_items)
    if dropped:
        print(f"[Filter] Dropped {dropped} non-video posts.")
    all_items = video_items

    if not all_items:
        print("[Error] No items scraped. Exiting.")
        sys.exit(1)

    # 2. ML Semantic Vectors
    embeddings = compute_ml_embeddings(all_items, dim=64)

    # 3. SQLite Database
    db_path = os.path.join(args.outdir, "circleftp.db")
    compile_sqlite_database(db_path, categories, all_items, embeddings)

    # 4. Compression & Digest
    compressed_paths = compress_database(db_path)
    primary_compressed = compressed_paths[0] if compressed_paths else db_path
    primary_sha256 = compute_sha256(primary_compressed)

    for cp in compressed_paths:
        compute_sha256(cp)
    compute_sha256(db_path)

    default_sha_file = os.path.join(args.outdir, "circleftp.db.sha256")
    with open(default_sha_file, "w") as f:
        f.write(f"{primary_sha256}  circleftp.db.zst\n")

    # 5. Manifest
    comp_filename = os.path.basename(primary_compressed)
    generate_manifest(
        output_dir=args.outdir,
        db_filename=comp_filename,
        sha256=primary_sha256,
        total_items=len(all_items),
        download_url=f"{args.release_url_prefix}/{comp_filename}",
    )

    print("\n[Fensy Indexer] Complete compilation finished successfully!")


if __name__ == "__main__":
    main()
