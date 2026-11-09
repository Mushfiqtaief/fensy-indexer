"""
Deterministic 64-dimensional float32 feature hashing for catalog recommendations.
100% mathematically and bit-for-bit identical to Rust core/fensy_engine/src/catalog/ml.rs.
Zero external dependencies (uses standard library hashlib, struct, re, math).
"""

import base64
import hashlib
import math
import re
import struct

STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "into", "over", "under", "after",
    "before", "about", "when", "where", "while", "your", "their", "have", "will", "are", "was",
    "were", "been", "being", "has", "had", "but", "not", "you", "all", "any", "can", "her", "his",
    "its", "our", "out", "who", "what", "which", "how", "why",
}

_WORD_SPLIT = re.compile(r'[^a-zA-Z0-9]+')


def _to_f32(val: float) -> float:
    return struct.unpack('<f', struct.pack('<f', val))[0]


_F32_0_7 = _to_f32(0.7)
_F32_0_6 = _to_f32(0.6)


def hash_feature_into(vec: list[float], feature: str, weight: float) -> None:
    digest = hashlib.md5(feature.encode('utf-8')).digest()
    b1 = struct.unpack('<I', digest[0:4])[0] % 64
    s1 = 1.0 if (digest[4] & 1) == 0 else -1.0
    b2 = struct.unpack('<I', digest[8:12])[0] % 64
    s2 = 1.0 if (digest[12] & 1) == 0 else -1.0
    vec[b1] = _to_f32(vec[b1] + _to_f32(s1 * weight))
    vec[b2] = _to_f32(vec[b2] + _to_f32(s2 * _to_f32(weight * _F32_0_7)))


def tokenize_and_accumulate(vec: list[float], text: str, weight: float, bigram_weight: float) -> None:
    if not text:
        return
    tokens = [
        t for t in _WORD_SPLIT.split(text.lower())
        if len(t) >= 2 and t not in STOPWORDS
    ]
    for tok in tokens:
        hash_feature_into(vec, tok, weight)
        if len(tok) >= 5:
            prefix = tok[:4]
            hash_feature_into(vec, f"stem:{prefix}", _to_f32(weight * _F32_0_6))

    for i in range(len(tokens) - 1):
        hash_feature_into(vec, f"bi:{tokens[i]}_{tokens[i+1]}", bigram_weight)


def compute_embedding_vec(
    title: str,
    synopsis: str | None = None,
    genres: str | None = None,
    category: str | None = None,
) -> list[float]:
    vec = [0.0] * 64
    tokenize_and_accumulate(vec, title, 3.0, 2.0)
    if genres:
        tokenize_and_accumulate(vec, genres, 2.5, 2.0)
    if category:
        tokenize_and_accumulate(vec, category, 2.0, 2.0)
    if synopsis:
        tokenize_and_accumulate(vec, synopsis, 1.0, 2.0)

    norm_sq = sum(float(x) * float(x) for x in vec)
    if norm_sq <= 0.0 and title and title.strip():
        hash_feature_into(vec, title.strip().lower(), 1.0)
        norm_sq = sum(float(x) * float(x) for x in vec)

    norm = math.sqrt(norm_sq)
    if norm > 0.0:
        vec = [_to_f32(float(x) / norm) for x in vec]
    return vec


def compute_embedding_bytes(
    title: str,
    synopsis: str | None = None,
    genres: str | None = None,
    category: str | None = None,
) -> bytes:
    """Returns 256-byte binary little-endian float32[64] BLOB for SQLite."""
    vec = compute_embedding_vec(title, synopsis, genres, category)
    return struct.pack('<64f', *vec)


def compute_embedding_b64(
    title: str,
    synopsis: str | None = None,
    genres: str | None = None,
    category: str | None = None,
) -> str:
    raw = compute_embedding_bytes(title, synopsis, genres, category)
    return base64.b64encode(raw).decode('ascii')
