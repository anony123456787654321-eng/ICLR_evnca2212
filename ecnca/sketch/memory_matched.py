"""Memory-matched provenance baselines.

All of these get the same wire format as the Theta sketch -- ``capacity`` atom
slots plus a small auxiliary structure -- so any difference in the results is a
difference in *retention policy*, not in bandwidth.

  Reservoir : uniform-random retention.  Uncoordinated, so two cells keep
              different subsets of the same root set; the overlap between two
              messages is unobservable beyond the retained atoms, and evicted
              roots are forgotten, so cycles re-count them.
  LRU       : recency retention.  Conservative (no rescaling) and lossy.
  Bloom     : first-k atoms + a Bloom membership filter over every root ever
              seen.  Cardinality is estimated well, but the payload of evicted
              roots is gone and false positives silently delete real evidence.
  CountMin  : as Bloom, with counters instead of bits (detects repeats, but
              hash collisions make novel roots look already-seen).
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np

from ..hashing import aux_hashes
from .base import ProvenanceSketch, version_key


class ReservoirSketch(ProvenanceSketch):
    name = "reservoir"

    def __init__(self, payload_dim: int, capacity: int = 32, rng: Optional[np.random.Generator] = None,
                 hash_seed: int = 0):
        self.payload_dim = int(payload_dim)
        self.capacity = int(capacity)
        self.hash_seed = int(hash_seed)
        self.rng = rng if rng is not None else np.random.default_rng(hash_seed)
        self.entries: Dict[str, Tuple[np.ndarray, float]] = {}
        self.n_seen: float = 0.0        # distinct roots *believed* seen

    def insert(self, root_id, payload, mass, refinement: float = 0.0) -> None:
        cand = (np.asarray(payload, np.float64).copy(), float(mass), float(refinement))
        cur = self.entries.get(root_id)
        if cur is not None:
            if version_key(cand[2], cand[0], cand[1]) > version_key(cur[2], cur[0], cur[1]):
                self.entries[root_id] = cand
            return
        self.n_seen += 1.0
        self.entries[root_id] = cand
        self._subsample()

    def _subsample(self) -> None:
        if len(self.entries) <= self.capacity:
            return
        keys = sorted(self.entries)
        keep = self.rng.choice(len(keys), size=self.capacity, replace=False)
        self.entries = {keys[i]: self.entries[keys[i]] for i in sorted(keep)}

    def merge(self, other: "ReservoirSketch") -> "ReservoirSketch":
        out = ReservoirSketch(self.payload_dim, self.capacity, self.rng, self.hash_seed)
        overlap = len(set(self.entries) & set(other.entries))
        out.n_seen = max(self.n_seen, other.n_seen, self.n_seen + other.n_seen - overlap)
        out.entries = dict(self.entries)
        for root, value in other.entries.items():
            cur = out.entries.get(root)
            if cur is None or version_key(value[2], value[0], value[1]) > version_key(cur[2], cur[0], cur[1]):
                out.entries[root] = value
        out._subsample()
        return out

    def copy(self):
        out = ReservoirSketch(self.payload_dim, self.capacity, self.rng, self.hash_seed)
        out.entries = dict(self.entries)
        out.n_seen = self.n_seen
        return out

    def _scale(self) -> float:
        if not self.entries:
            return 0.0
        return max(1.0, self.n_seen) / len(self.entries)

    def estimate_payload(self):
        if not self.entries:
            return self._zero()
        return np.sum([p for p, _, _ in self.entries.values()], axis=0) * self._scale()

    def estimate_mass(self):
        return float(sum(m for _, m, _ in self.entries.values())) * self._scale()

    def estimate_count(self):
        return float(max(len(self.entries), self.n_seen))

    def estimate_payload_cov(self):
        """Uniform-sample variance.  Valid in a single stream; note that under
        merging ``n_seen`` is itself wrong, which this term cannot capture."""
        k = len(self.entries)
        if k <= 1 or self._scale() <= 1.0:
            return np.zeros((self.payload_dim, self.payload_dim))
        P = np.stack([p for p, _, _ in self.entries.values()])
        n_hat = max(self.estimate_count(), float(k))
        S = np.atleast_2d(np.cov(P, rowvar=False, ddof=1))
        return (n_hat ** 2) * max(0.0, 1.0 - k / n_hat) * S / k

    def n_bytes(self):
        return self.capacity * (8 + 8 * self.payload_dim + 8) + 8

    def signature(self):
        return ("reservoir", tuple(sorted(self.entries)), round(self.n_seen, 9))


class LRUSketch(ProvenanceSketch):
    name = "lru"

    def __init__(self, payload_dim: int, capacity: int = 32, hash_seed: int = 0):
        self.payload_dim = int(payload_dim)
        self.capacity = int(capacity)
        self.hash_seed = int(hash_seed)
        self.entries: Dict[str, Tuple[np.ndarray, float, int]] = {}
        self.clock: int = 0

    def insert(self, root_id, payload, mass, refinement: float = 0.0) -> None:
        cur = self.entries.get(root_id)
        if cur is not None:
            cand = (np.asarray(payload, np.float64).copy(), float(mass), cur[2], float(refinement))
            if version_key(cand[3], cand[0], cand[1]) > version_key(cur[3], cur[0], cur[1]):
                self.entries[root_id] = cand
            return
        self.clock += 1
        self.entries[root_id] = (np.asarray(payload, np.float64).copy(), float(mass),
                                 self.clock, float(refinement))
        self._evict()

    def _evict(self) -> None:
        if len(self.entries) <= self.capacity:
            return
        order = sorted(self.entries.items(), key=lambda kv: -kv[1][2])
        self.entries = dict(order[: self.capacity])

    def merge(self, other: "LRUSketch") -> "LRUSketch":
        out = LRUSketch(self.payload_dim, self.capacity, self.hash_seed)
        out.clock = max(self.clock, other.clock)
        out.entries = dict(self.entries)
        for root, value in other.entries.items():
            cur = out.entries.get(root)
            if cur is None:
                out.clock += 1
                out.entries[root] = (value[0], value[1], out.clock, value[3])
            elif version_key(value[3], value[0], value[1]) > version_key(cur[3], cur[0], cur[1]):
                out.entries[root] = (value[0], value[1], cur[2], value[3])
        out._evict()
        return out

    def copy(self):
        out = LRUSketch(self.payload_dim, self.capacity, self.hash_seed)
        out.entries = dict(self.entries)
        out.clock = self.clock
        return out

    def estimate_payload(self):
        if not self.entries:
            return self._zero()
        return np.sum([p for p, _, _, _ in self.entries.values()], axis=0)

    def estimate_mass(self):
        return float(sum(m for _, m, _, _ in self.entries.values()))

    def estimate_count(self):
        return float(len(self.entries))

    def n_bytes(self):
        return self.capacity * (8 + 8 * self.payload_dim + 8 + 4)

    def signature(self):
        return ("lru", tuple(sorted(self.entries)))


class MembershipSketch(ProvenanceSketch):
    """First-k atoms + an approximate membership structure over all roots."""

    def __init__(self, payload_dim: int, capacity: int = 32, n_bits: int = 2048,
                 n_hashes: int = 3, backend: str = "bloom", cm_width: int = 256,
                 cm_depth: int = 3, hash_seed: int = 0):
        self.payload_dim = int(payload_dim)
        self.capacity = int(capacity)
        self.backend = backend
        self.hash_seed = int(hash_seed)
        self.n_hashes = int(n_hashes)
        self.entries: Dict[str, Tuple[np.ndarray, float, int]] = {}
        self.clock = 0
        if backend == "bloom":
            self.n_bits = int(n_bits)
            self.bits = np.zeros(self.n_bits, dtype=bool)
        elif backend == "countmin":
            self.cm_width, self.cm_depth = int(cm_width), int(cm_depth)
            self.counts = np.zeros((self.cm_depth, self.cm_width), dtype=np.int32)
        else:
            raise ValueError(backend)

    @property
    def name(self):
        return self.backend

    # -- membership backend --------------------------------------------------
    def _seen(self, root_id) -> bool:
        if self.backend == "bloom":
            idx = aux_hashes(root_id, self.n_hashes, self.n_bits, self.hash_seed)
            return bool(self.bits[idx].all())
        idx = aux_hashes(root_id, self.cm_depth, self.cm_width, self.hash_seed)
        return bool(min(self.counts[r, i] for r, i in enumerate(idx)) > 0)

    def _mark(self, root_id) -> None:
        if self.backend == "bloom":
            idx = aux_hashes(root_id, self.n_hashes, self.n_bits, self.hash_seed)
            self.bits[idx] = True
        else:
            idx = aux_hashes(root_id, self.cm_depth, self.cm_width, self.hash_seed)
            for r, i in enumerate(idx):
                self.counts[r, i] += 1

    def _cardinality(self) -> float:
        if self.backend == "bloom":
            x = int(self.bits.sum())
            if x == 0:
                return 0.0
            if x >= self.n_bits:
                return float(self.n_bits)  # saturated
            return -(self.n_bits / self.n_hashes) * math.log(1.0 - x / self.n_bits)
        nz = int((self.counts[0] > 0).sum())
        if nz == 0:
            return 0.0
        if nz >= self.cm_width:
            return float(self.cm_width)
        return -self.cm_width * math.log(1.0 - nz / self.cm_width)

    # -- sketch API ----------------------------------------------------------
    def insert(self, root_id, payload, mass, refinement: float = 0.0) -> None:
        cur = self.entries.get(root_id)
        if cur is not None:
            cand = (np.asarray(payload, np.float64).copy(), float(mass), cur[2], float(refinement))
            if version_key(cand[3], cand[0], cand[1]) > version_key(cur[3], cur[0], cur[1]):
                self.entries[root_id] = cand
            return
        if self._seen(root_id):
            return                              # false positives lose evidence here
        self._mark(root_id)
        if len(self.entries) < self.capacity:
            self.clock += 1
            self.entries[root_id] = (np.asarray(payload, np.float64).copy(), float(mass),
                                     self.clock, float(refinement))

    def _blank(self):
        if self.backend == "bloom":
            out = MembershipSketch(self.payload_dim, self.capacity, self.n_bits, self.n_hashes,
                                   "bloom", hash_seed=self.hash_seed)
        else:
            out = MembershipSketch(self.payload_dim, self.capacity, backend="countmin",
                                   cm_width=self.cm_width, cm_depth=self.cm_depth,
                                   n_hashes=self.n_hashes, hash_seed=self.hash_seed)
        return out

    def merge(self, other: "MembershipSketch") -> "MembershipSketch":
        out = self._blank()
        if self.backend == "bloom":
            out.bits = self.bits | other.bits
        else:
            out.counts = np.maximum(self.counts, other.counts)
        out.entries = dict(self.entries)
        out.clock = max(self.clock, other.clock)
        for root, value in other.entries.items():
            cur = out.entries.get(root)
            if cur is not None:
                if version_key(value[3], value[0], value[1]) > version_key(cur[3], cur[0], cur[1]):
                    out.entries[root] = (value[0], value[1], cur[2], value[3])
                continue
            if len(out.entries) >= self.capacity:
                continue
            out.clock += 1
            out.entries[root] = (value[0], value[1], out.clock, value[3])
        return out

    def copy(self):
        out = self._blank()
        if self.backend == "bloom":
            out.bits = self.bits.copy()
        else:
            out.counts = self.counts.copy()
        out.entries = dict(self.entries)
        out.clock = self.clock
        return out

    def _scale(self) -> float:
        if not self.entries:
            return 0.0
        return max(1.0, self._cardinality()) / len(self.entries)

    def estimate_payload(self):
        if not self.entries:
            return self._zero()
        return np.sum([p for p, _, _, _ in self.entries.values()], axis=0) * self._scale()

    def estimate_mass(self):
        return float(sum(m for _, m, _, _ in self.entries.values())) * self._scale()

    def estimate_count(self):
        return float(max(len(self.entries), self._cardinality()))

    def n_bytes(self):
        atoms = self.capacity * (8 + 8 * self.payload_dim + 8)
        aux = self.n_bits // 8 if self.backend == "bloom" else self.cm_width * self.cm_depth * 4
        return atoms + aux

    def signature(self):
        aux = tuple(np.flatnonzero(self.bits).tolist()) if self.backend == "bloom" \
            else tuple(self.counts.ravel().tolist())
        return (self.backend, tuple(sorted(self.entries)), aux)
