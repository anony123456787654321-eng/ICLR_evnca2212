"""Exact root-ID ledger: the upper bound and the ground truth of the mechanism.

Cost is O(n_roots * d), so this is not deployable -- it exists to (i) certify
that lineage-constrained fusion reproduces the centralised unique-evidence
posterior and (ii) upper-bound every compressed structure.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .base import ProvenanceSketch, _sig_entries, version_key


class ExactLedger(ProvenanceSketch):
    name = "exact"

    def __init__(self, payload_dim: int, capacity: int = -1, refine: bool = True):
        self.payload_dim = int(payload_dim)
        self.capacity = -1  # unbounded
        # refine=False is the ablation in which provenance is ONLY a duplicate
        # filter: the least-refined version of a root wins and every downstream
        # computation over that observation is thrown away.  Deterministic (a
        # min-join), so it stays order-invariant -- it is simply lossy.
        self.refine = bool(refine)
        self.entries: Dict[str, Tuple[np.ndarray, float, float]] = {}

    def insert(self, root_id: str, payload: np.ndarray, mass: float,
               refinement: float = 0.0) -> None:
        cand = (np.asarray(payload, dtype=np.float64).copy(), float(mass), float(refinement))
        cur = self.entries.get(root_id)
        if cur is None or self._wins(cand, cur):
            self.entries[root_id] = cand

    def _wins(self, cand, cur) -> bool:
        a = version_key(cand[2], cand[0], cand[1])
        b = version_key(cur[2], cur[0], cur[1])
        return a > b if self.refine else a < b

    def merge(self, other: "ExactLedger") -> "ExactLedger":
        out = ExactLedger(self.payload_dim, refine=self.refine)
        out.entries = dict(self.entries)
        for root, value in other.entries.items():
            cur = out.entries.get(root)
            if cur is None or out._wins(value, cur):
                out.entries[root] = value
        return out

    def copy(self) -> "ExactLedger":
        out = ExactLedger(self.payload_dim, refine=self.refine)
        out.entries = dict(self.entries)
        return out

    def estimate_payload(self) -> np.ndarray:
        if not self.entries:
            return self._zero()
        return np.sum([p for p, _, _ in self.entries.values()], axis=0)

    def estimate_mass(self) -> float:
        return float(sum(m for _, m, _ in self.entries.values()))

    def estimate_count(self) -> float:
        return float(len(self.entries))

    def n_bytes(self) -> int:
        # 8 bytes for the root hash + payload + mass, per retained root
        return len(self.entries) * (8 + 8 * self.payload_dim + 8)

    def signature(self):
        return ("exact", tuple((r, tuple(np.round(v[0], 12).tolist()), round(v[1], 12),
                                round(v[2], 12)) for r, v in sorted(self.entries.items())))
