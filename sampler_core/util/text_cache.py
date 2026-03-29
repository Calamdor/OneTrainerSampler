"""Shared text-encoder embedding cache key generation."""
import hashlib


def te_cache_key(pos_prompt: str, neg_prompt: str, te_dtype: str) -> str:
    """Stable 24-char hex key for one (pos, neg, dtype) text-encoder cache entry."""
    data = f"{pos_prompt}\x00{neg_prompt}\x00{te_dtype}"
    return hashlib.sha256(data.encode("utf-8")).hexdigest()[:24]
