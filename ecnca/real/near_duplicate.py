"""Near-duplicate detection baselines for the aggregation stage.

The duplicated-hop evaluation compares evidence conservation against exact
canonical deduplication, and a reviewer is right to object that exact matching
is the weakest possible duplicate defence: a re-chunked or reworded copy is not
byte-identical, so canonical dedup never fires. This module supplies the
defences that industrial pipelines actually use, so the comparison is against
the strong form of the alternative rather than the weak one.

Three detectors, each the standard method from its own literature:

* `minhash`  -- 5-gram word shingles, 128 permutations, Jaccard estimate at or
  above `MINHASH_THRESHOLD`. This is the corpus-deduplication setting.
* `simhash`  -- 64-bit weighted feature hash, Hamming distance at or below
  `SIMHASH_HAMMING`. This is the web near-duplicate setting.
* `embed`    -- cosine similarity of a frozen sentence encoder at or above
  `EMBED_THRESHOLD`. This is the semantic retrieval setting.

Every threshold is a frozen literature default, fixed here before any
duplicated condition was scored and never fitted on an evaluation population.
Changing one is a preregistration change, not a hyperparameter choice.

The detectors are deliberately *greedy and order-dependent*: an arrival is
suppressed if it is near-duplicate to some already-accepted arrival. That is how
these methods are deployed, and the order dependence is a property worth
measuring rather than a defect to hide. Evidence conservation, by contrast,
resolves to the same state under any delivery order because its join is on a
semilattice.

None of these detectors is a ledger. They suppress *payloads* that look alike;
they cannot express that two textually unlike arrivals descend from one
observation, nor that a genuine refinement of a root should be permitted through
while still crediting that root once. That distinction is the point of the
comparison.
"""
from __future__ import annotations

import functools
import hashlib
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

# ---------------------------------------------------------------------------
# Frozen thresholds. Literature defaults, not tuned here.
# ---------------------------------------------------------------------------
SHINGLE_SIZE = 5          # word n-gram width, corpus-dedup convention
MINHASH_PERMUTATIONS = 128
MINHASH_THRESHOLD = 0.80  # Jaccard, the standard near-duplicate operating point
SIMHASH_BITS = 64
SIMHASH_HAMMING = 3       # 64-bit web near-duplicate convention
EMBED_THRESHOLD = 0.95    # cosine, frozen before evaluation

DETECTORS = ("minhash", "simhash", "embed")

_MASK64 = (1 << 64) - 1


def _tokens(text: str) -> list[str]:
    return text.lower().split()


def shingles(text: str, size: int = SHINGLE_SIZE) -> set[str]:
    """Word n-gram shingles. Short texts fall back to the whole token string,
    so a passage shorter than the window is still comparable rather than empty."""
    words = _tokens(text)
    if len(words) < size:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i:i + size]) for i in range(len(words) - size + 1)}


def _hash64(data: str, salt: int = 0) -> int:
    digest = hashlib.blake2b(f"{salt}\x00{data}".encode("utf-8"),
                             digest_size=8).digest()
    return int.from_bytes(digest, "big")


def _base_hashes(grams) -> np.ndarray:
    """One 64-bit hash per shingle, as unsigned integers."""
    return np.fromiter((_hash64(g) for g in grams), dtype=np.uint64,
                       count=len(grams))


# The permutations are drawn once from a fixed seed, so a signature depends only
# on the text: two processes, two machines and two runs agree exactly.
PERMUTATION_SEED = 0x5EED0001
_PERM_RNG = np.random.default_rng(PERMUTATION_SEED)
_PERM_A = (_PERM_RNG.integers(1, 1 << 62, size=MINHASH_PERMUTATIONS,
                              dtype=np.uint64) | np.uint64(1))
_PERM_B = _PERM_RNG.integers(0, 1 << 62, size=MINHASH_PERMUTATIONS,
                             dtype=np.uint64)


# ---------------------------------------------------------------------------
# MinHash
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=65536)
def minhash_signature(text: str) -> np.ndarray:
    """Deterministic MinHash signature over `MINHASH_PERMUTATIONS` permutations.

    Each shingle is hashed once to 64 bits, then the standard affine family
    `a*h + b` supplies the permutations, which is the usual construction and
    keeps the cost linear in the number of shingles rather than in their product
    with the number of permutations.

    An empty shingle set yields the all-maximum signature, which has estimated
    Jaccard zero against every other signature, so an empty payload can never be
    suppressed as a near-duplicate of a non-empty one. The returned array is
    read-only because it is memoised and shared between callers.
    """
    grams = sorted(shingles(text))
    if not grams:
        sig = np.full(MINHASH_PERMUTATIONS, _MASK64, dtype=np.uint64)
    else:
        base = _base_hashes(grams)                       # (G,)
        # uint64 arithmetic wraps, which is the modulus this family relies on.
        with np.errstate(over="ignore"):
            permuted = (_PERM_A[:, None] * base[None, :]) + _PERM_B[:, None]
        sig = permuted.min(axis=1)
    sig.flags.writeable = False
    return sig


def minhash_jaccard(a: np.ndarray, b: np.ndarray) -> float:
    """Estimated Jaccard: the fraction of permutations that agree."""
    if a.size == 0 or a.size != b.size:
        return 0.0
    if int(a[0]) == _MASK64 and np.all(a == _MASK64):
        return 0.0
    if int(b[0]) == _MASK64 and np.all(b == _MASK64):
        return 0.0
    return float((a == b).mean())


# ---------------------------------------------------------------------------
# SimHash
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=65536)
def simhash(text: str) -> int:
    """64-bit SimHash: sign of the summed bit votes over the shingle set."""
    grams = sorted(shingles(text))
    if not grams:
        return 0
    base = _base_hashes(grams)
    bit_index = np.arange(SIMHASH_BITS, dtype=np.uint64)
    bits = ((base[:, None] >> bit_index[None, :]) & np.uint64(1))  # (G, 64)
    votes = np.where(bits == 1, 1, -1).sum(axis=0)
    weights = (np.uint64(1) << bit_index)
    return int(weights[votes > 0].sum())


def hamming(a: int, b: int) -> int:
    return int(bin(a ^ b).count("1"))


# ---------------------------------------------------------------------------
# Greedy near-duplicate index
# ---------------------------------------------------------------------------

@dataclass
class NearDuplicateIndex:
    """Greedy first-come near-duplicate suppression.

    `accept(text)` returns True and records the payload when it is not a near
    duplicate of anything already accepted, and False otherwise. Detection is
    over payloads only: the index has no notion of lineage, so it cannot tell a
    second observation of a distinct source from a reworded copy of the first.
    """
    detector: str
    embed: Callable[[str], np.ndarray] | None = None
    minhash_threshold: float = MINHASH_THRESHOLD
    simhash_hamming: int = SIMHASH_HAMMING
    embed_threshold: float = EMBED_THRESHOLD
    _signatures: list = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if self.detector not in DETECTORS:
            raise ValueError(f"unknown detector {self.detector!r}")
        if self.detector == "embed" and self.embed is None:
            raise ValueError(
                "the embed detector needs an `embed` callable; without one it "
                "would silently degrade to accepting everything")

    def _signature(self, text: str):
        if self.detector == "minhash":
            return minhash_signature(text)
        if self.detector == "simhash":
            return simhash(text)
        vec = np.asarray(self.embed(text), dtype=np.float64).ravel()  # type: ignore[misc]
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0 else vec

    def _is_duplicate(self, sig) -> bool:
        for other in self._signatures:
            if self.detector == "minhash":
                if minhash_jaccard(sig, other) >= self.minhash_threshold:
                    return True
            elif self.detector == "simhash":
                if hamming(sig, other) <= self.simhash_hamming:
                    return True
            else:
                if float(sig @ other) >= self.embed_threshold:
                    return True
        return False

    def accept(self, text: str) -> bool:
        sig = self._signature(text)
        if self._is_duplicate(sig):
            return False
        self._signatures.append(sig)
        return True


def config() -> dict:
    """The frozen configuration, for the record written alongside results."""
    return {"shingle_size": SHINGLE_SIZE,
            "minhash_permutations": MINHASH_PERMUTATIONS,
            "minhash_threshold": MINHASH_THRESHOLD,
            "simhash_bits": SIMHASH_BITS,
            "simhash_hamming": SIMHASH_HAMMING,
            "embed_threshold": EMBED_THRESHOLD,
            "note": "frozen literature defaults; not tuned on any evaluation "
                    "population"}
