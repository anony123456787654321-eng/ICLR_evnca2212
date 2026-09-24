"""Deterministic, uniform root-ID hashing.

Every provenance structure in this project maps a root identifier to a value in
[0, 1).  The map must be:

  * deterministic across processes / machines (no Python ``hash()`` salt),
  * uniform, so that order statistics of the hashes support KMV / Theta
    cardinality and Horvitz-Thompson payload estimators,
  * *coordinated*: the same root observed by two different cells gets the same
    hash, which is what makes hash-based sampling mergeable.
"""
from __future__ import annotations

import hashlib
from typing import Union

_TWO64 = float(1 << 64)


def root_hash64(root_id: Union[str, bytes, int], seed: int = 0) -> int:
    """Stable 64-bit hash of a root identifier."""
    if isinstance(root_id, int):
        payload = root_id.to_bytes(8, "little", signed=True)
    elif isinstance(root_id, str):
        payload = root_id.encode("utf-8")
    else:
        payload = bytes(root_id)
    digest = hashlib.blake2b(
        payload, digest_size=8, key=seed.to_bytes(8, "little")
    ).digest()
    return int.from_bytes(digest, "little")


def root_uniform(root_id: Union[str, bytes, int], seed: int = 0) -> float:
    """Stable hash of a root identifier into [0, 1)."""
    return root_hash64(root_id, seed) / _TWO64


def aux_hashes(root_id: Union[str, bytes, int], n: int, m: int, seed: int = 0):
    """``n`` independent indices in ``[0, m)`` for Bloom / count-min structures."""
    h = root_hash64(root_id, seed)
    h1 = h & 0xFFFFFFFF
    h2 = (h >> 32) | 1  # odd, so the Kirsch-Mitzenmacher scheme cycles fully
    return [((h1 + i * h2) % m) for i in range(n)]
