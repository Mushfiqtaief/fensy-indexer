"""
Lightweight IMDb public dataset matcher (`title.basics.tsv.gz`).
Resolves canonical `imdb_id` (tconst) and standard IMDb `genres` + `year`
offline during CI indexing with zero API keys or rate limits.
"""

import gzip
import os
import re
import tempfile
import time
import unicodedata
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Tuple

IMDB_BASICS_URL = "https://datasets.imdbws.com/title.basics.tsv.gz"
CACHE_MAX_AGE_SECS = 12 * 3600

_YEAR_RE = re.compile(r"\b(19\d\d|20\d\d)\b")
_NOISE_RE = re.compile(
    r"\b(1080p|720p|480p|360p|2160p|4k|uhd|bluray|bdrip|brrip|web-?dl|webrip|hdrip|"
    r"hdtv|dvdrip|remux|x264|x265|hevc|h264|h265|10bit|aac|ddp?5\.1|atmos|esub|msub|"
    r"dual audio|multi audio|hindi|bengali|tamil|telugu|kannada|malayalam|english|"
    r"dubbed|subbed|cam|hdcam|predvd|rip|remastered|extended|directors cut)\b",
    re.IGNORECASE,
)
_SEASON_RE = re.compile(
    r"\b(season\s*\d+|s\d+(-s?\d+)?|all\s*seasons?|complete\s*series|part\s*\d+|vol(ume)?\.?\s*\d+)\b",
    re.IGNORECASE,
)
_BRACKET_RE = re.compile(r"\[(.*?)\]|\((.*?)\)")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")

_VALID_TYPES = {
    "movie": False,
    "tvMovie": False,
    "tvSeries": True,
    "tvMiniSeries": True,
}


def _ascii_lower(title: str) -> str:
    if not title:
        return ""
    norm = unicodedata.normalize("NFKD", title)
    return norm.encode("ascii", "ignore").decode("ascii").lower()


def normalize_imdb_title(title: str) -> str:
    s = _ascii_lower(title).replace("&", " and ")
    return _NON_ALNUM_RE.sub(" ", s).strip()


def normalize_title(title: str) -> str:
    s = _ascii_lower(title)

    def _strip_bracket(m: re.Match) -> str:
        inner = m.group(1) or m.group(2) or ""
        if _NOISE_RE.search(inner) or _YEAR_RE.search(inner):
            return " "
        return m.group(0)

    s = _BRACKET_RE.sub(_strip_bracket, s)
    s = _NOISE_RE.sub(" ", s)
    s = _SEASON_RE.sub(" ", s)
    s = s.replace("&", " and ")
    return _NON_ALNUM_RE.sub(" ", s).strip()


def extract_year(raw_year: Optional[str], title: str) -> Optional[int]:
    for src in (raw_year or "", title or ""):
        m = _YEAR_RE.search(src)
        if m:
            return int(m.group(1))
    return None


def _ensure_dataset_file() -> str:
    cache_path = os.path.join(tempfile.gettempdir(), "fensy_imdb_basics.tsv.gz")
    if os.path.exists(cache_path):
        age = time.time() - os.path.getmtime(cache_path)
        if age < CACHE_MAX_AGE_SECS and os.path.getsize(cache_path) > 50_000_000:
            print(f"[IMDb] Reusing cached dataset ({os.path.getsize(cache_path) // (1024*1024)} MB)")
            return cache_path

    print("[IMDb] Downloading public IMDb title.basics.tsv.gz...")
    req = urllib.request.Request(
        IMDB_BASICS_URL,
        headers={"User-Agent": "FensyIndexer/2.2"},
    )
    tmp_path = cache_path + ".tmp"
    with urllib.request.urlopen(req, timeout=120) as resp, open(tmp_path, "wb") as out:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    os.replace(tmp_path, cache_path)
    print(f"[IMDb] Downloaded {os.path.getsize(cache_path) // (1024*1024)} MB")
    return cache_path


def enrich_with_imdb(
    items: List[Dict[str, Any]],
    get_title: Callable[[Dict[str, Any]], str],
    get_year: Callable[[Dict[str, Any]], Optional[str]],
    is_series_fn: Callable[[Dict[str, Any]], bool],
    on_match: Callable[[Dict[str, Any], str, Optional[str], Optional[int]], None],
) -> int:
    """
    Matches items against IMDb basics dataset and invokes `on_match(item, imdb_id, genres, year)`.
    Returns number of matched items.
    """
    if not items:
        return 0

    needed_titles: set[str] = set()
    parsed_items: List[Tuple[Dict[str, Any], str, Optional[int], bool]] = []
    for item in items:
        t = get_title(item)
        norm = normalize_title(t)
        if len(norm) >= 2:
            needed_titles.add(norm)
            y = extract_year(get_year(item), t)
            s = is_series_fn(item)
            parsed_items.append((item, norm, y, s))

    try:
        gz_path = _ensure_dataset_file()
    except Exception as e:
        print(f"[IMDb] Warning: skipped dataset download ({e})")
        return 0

    # Index: (norm_title, is_series) -> List[(year_int_or_None, tconst, genres_or_None)]
    lookup: Dict[Tuple[str, bool], List[Tuple[Optional[int], str, Optional[str]]]] = {}

    started = time.time()
    try:
        with gzip.open(gz_path, "rt", encoding="utf-8", errors="ignore") as f:
            header = f.readline()  # skip header
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 9:
                    continue
                t_type = parts[1]
                is_series = _VALID_TYPES.get(t_type)
                if is_series is None:
                    continue

                p_norm = normalize_imdb_title(parts[2])
                o_norm = normalize_imdb_title(parts[3]) if parts[3] != parts[2] else ""

                if p_norm not in needed_titles and o_norm not in needed_titles:
                    continue

                tconst = parts[0]
                start_year = int(parts[5]) if parts[5].isdigit() else None
                genres = None if parts[8] == r"\N" else parts[8].replace(",", ", ")
                entry = (start_year, tconst, genres)

                if p_norm in needed_titles:
                    lookup.setdefault((p_norm, is_series), []).append(entry)
                if o_norm and o_norm in needed_titles:
                    lookup.setdefault((o_norm, is_series), []).append(entry)
    except Exception as e:
        print(f"[IMDb] Warning: failed parsing dataset ({e})")
        return 0

    matched = 0
    for item, norm, y, is_series in parsed_items:
        candidates = lookup.get((norm, is_series))
        if not candidates:
            continue

        best = None
        if y is not None:
            exact = [c for c in candidates if c[0] == y]
            if not exact:
                exact = [c for c in candidates if c[0] is not None and abs(c[0] - y) <= 1]
            if exact:
                with_genres = [c for c in exact if c[2] is not None]
                best = with_genres[0] if with_genres else exact[0]
        else:
            # Prefer entries with genres and a valid year
            with_meta = [c for c in candidates if c[0] is not None and c[2] is not None]
            best = with_meta[-1] if with_meta else candidates[0]

        if best is not None:
            on_match(item, best[1], best[2], best[0])
            matched += 1

    print(
        f"[IMDb] Matched {matched}/{len(items)} items in {time.time() - started:.1f}s "
        f"({len(lookup)} title buckets cached)"
    )
    return matched
