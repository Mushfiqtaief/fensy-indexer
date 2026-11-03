#!/usr/bin/env python3
"""
Fensy Universal Media Indexer & Machine Learning Vector Compiler
Crawls BDIX / FTP catalog sources through socks5 / Mihomo proxy,
generates 64-dimensional semantic recommendation vectors, and compiles
a lightweight, compressed SQLite database for 0ms client mounting.
"""

import os
import sys
import time
import json
import sqlite3
import hashlib
import struct
import argparse
from typing import List, Dict, Any, Optional

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
                session.proxies.update({"http": local_proxy, "https": local_proxy})
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
                session.proxies.update({"http": socks_proxy, "https": socks_proxy})
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


def crawl_categories(session: requests.Session) -> List[Dict[str, Any]]:
    print("\n[Crawler] Fetching categories...")
    r = session.get(f"{BASE_URL}/api/categories", timeout=10)
    r.raise_for_status()
    categories = r.json()
    print(f"[Crawler] Found {len(categories)} root categories.")
    return categories


def crawl_category_posts(
    session: requests.Session,
    cat_id: Any,
    cat_name: str,
    max_pages: Optional[int] = None,
) -> List[Dict[str, Any]]:
    items = []
    page = 1
    total_pages = 1

    print(f"\n[Crawler] Crawling category [{cat_id}] '{cat_name}'...")
    while page <= total_pages:
        if max_pages and page > max_pages:
            break
        try:
            url = f"{BASE_URL}/api/posts"
            params = {
                "categoryExact": cat_id,
                "page": page,
                "limit": 50,
                "order": "desc",
            }
            r = session.get(url, params=params, timeout=10)
            if r.status_code != 200:
                print(f"  [Warning] HTTP {r.status_code} on page {page}, retrying...")
                time.sleep(1)
                continue

            data = r.json()
            posts = data.get("posts", [])
            if not posts:
                break

            items.extend(posts)
            pagination = data.get("pagination", {})
            total_pages = pagination.get("pages", 1)
            total_items = pagination.get("total", len(items))

            print(f"  -> Page {page}/{total_pages} ({len(items)}/{total_items} items)", end="\r")
            page += 1
            time.sleep(0.08)
        except Exception as e:
            print(f"\n  [Error] Failed page {page}: {e}. Retrying...")
            time.sleep(1)

    print(f"\n  -> Finished category '{cat_name}': {len(items)} total items.")
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
    )
    X = tfidf.fit_transform(corpus)

    # 2. Dimensionality reduction to 64 components
    actual_dim = min(dim, X.shape[1], len(items) - 1)
    if actual_dim < dim:
        actual_dim = max(2, actual_dim)

    svd = TruncatedSVD(n_components=actual_dim, random_state=42)
    X_reduced = svd.fit_transform(X)

    if actual_dim < dim:
        padded = np.zeros((X_reduced.shape[0], dim), dtype=np.float32)
        padded[:, :actual_dim] = X_reduced
        X_reduced = padded

    # 3. L2 Normalize
    X_normalized = normalize(X_reduced, norm="l2")

    # 4. Pack into binary buffers (64 floats * 4 bytes = 256 bytes)
    packed_blobs = []
    for row in X_normalized:
        blob = struct.pack(f"{dim}f", *row)
        packed_blobs.append(blob)

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
        m_type = "series" if item.get("type") == "series" else "movie"
        image_sm = item.get("imageSm")
        image_lg = item.get("image")
        poster = f"{BASE_URL}/uploads/{image_sm}" if image_sm else None
        banner = f"{BASE_URL}/uploads/{image_lg}" if image_lg else None

        cats = item.get("categories", [])
        cat_id = str(cats[0].get("id")) if cats else "0"
        sub_cat_id = str(cats[1].get("id")) if len(cats) > 1 else None
        genres = ", ".join([c.get("name", "") for c in cats if isinstance(c, dict)])

        created_epoch = int(time.time())
        created_str = item.get("createdAt")
        if created_str:
            try:
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


def compress_database(db_path: str) -> str:
    """Compresses SQLite database to .zst using zstandard."""
    zst_path = f"{db_path}.zst"
    print(f"\n[Compressor] Compressing {db_path} -> {zst_path}...")
    try:
        import zstandard as zstd

        cctx = zstd.ZstdCompressor(level=19)
        with open(db_path, "rb") as ifh, open(zst_path, "wb") as ofh:
            cctx.copy_stream(ifh, ofh)
    except ImportError:
        import gzip
        zst_path = f"{db_path}.gz"
        print(f"[Compressor] zstandard not found, falling back to gzip: {zst_path}...")
        with open(db_path, "rb") as f_in, gzip.open(zst_path, "wb") as f_out:
            f_out.writelines(f_in)

    zst_size = os.path.getsize(zst_path)
    print(f"[Compressor] Compressed package created: {zst_size / (1024*1024):.2f} MB")
    return zst_path


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

    for cat in categories:
        cat_id = cat.get("id")
        cat_name = cat.get("name", "Unknown")
        posts = crawl_category_posts(session, cat_id, cat_name, max_pages=args.max_pages_per_cat)
        for p in posts:
            pid = str(p.get("id"))
            if pid not in seen_ids:
                seen_ids.add(pid)
                all_items.append(p)

    print(f"\n[Summary] Total unique items collected: {len(all_items)}")
    if not all_items:
        print("[Error] No items scraped. Exiting.")
        sys.exit(1)

    # 2. ML Semantic Vectors
    embeddings = compute_ml_embeddings(all_items, dim=64)

    # 3. SQLite Database
    db_path = os.path.join(args.outdir, "circleftp.db")
    compile_sqlite_database(db_path, categories, all_items, embeddings)

    # 4. Compression & Digest
    compressed_path = compress_database(db_path)
    sha256 = compute_sha256(compressed_path)

    # 5. Manifest
    comp_filename = os.path.basename(compressed_path)
    generate_manifest(
        output_dir=args.outdir,
        db_filename=comp_filename,
        sha256=sha256,
        total_items=len(all_items),
        download_url=f"{args.release_url_prefix}/{comp_filename}",
    )

    print("\n[Fensy Indexer] Complete compilation finished successfully!")


if __name__ == "__main__":
    main()
