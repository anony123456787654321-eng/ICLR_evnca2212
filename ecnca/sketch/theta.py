"""Bottom-k / Theta tuple sketch: the provenance mechanism of EC-NCA.

Each root is mapped to a *coordinated* uniform hash u = h(root_id) in [0, 1).
The sketch keeps the k smallest hashes together with their additive payloads,
plus a threshold ``theta``; the invariant is

    every retained root has u < theta,   and   every root with u < theta that
    the sketch has ever seen is retained.

Consequences that the paper leans on:

  * ``merge`` is a join on a semilattice -> commutative, associative, idempotent,
    so message order, cycles and re-delivery cannot change the fixed point;
  * with at most k distinct roots, theta stays 1.0 and the sketch is *exact*;
  * above k roots, retention is a Bernoulli(theta) sample that is identical in
    every cell (coordination), so the Horvitz-Thompson estimator
    ``sum_retained(payload) / theta`` is unbiased with O(k^-1/2) relative error.

Decoupled budgets
-----------------
The relative variance of the plain HT estimator splits into two terms:

    Var/mean^2  ~  CV(payload)^2 / k_payload   +   1 / k_hash

and for realistic evidence the *cardinality* term dominates -- CV is order 0.2
to 2, so the second term is what sets the error.  A hash slot costs 8 bytes and
a payload slot costs 8*(payload_dim+2), so at a fixed byte budget it is much
cheaper to buy hashes than payloads.  ``hash_capacity`` therefore keeps a wider
KMV set of bare hashes for the cardinality, while ``capacity`` bounds the
payload slots, and the estimate becomes ``n_hat * mean(retained payloads)``.
With ``hash_capacity == capacity`` this reduces *exactly* to the plain HT form.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from ..hashing import root_uniform
from .base import ProvenanceSketch, version_key


class ThetaSketch(ProvenanceSketch):
    name = "theta"

    def __init__(self, payload_dim: int, capacity: int = 32, hash_seed: int = 0,
                 hash_capacity: Optional[int] = None, rescale: bool = True):
        if capacity < 1:
            raise ValueError("capacity (k) must be >= 1")
        self.payload_dim = int(payload_dim)
        self.capacity = int(capacity)
        self.hash_capacity = int(hash_capacity) if hash_capacity else int(capacity)
        if self.hash_capacity < self.capacity:
            raise ValueError("hash_capacity must be >= capacity")
        self.hash_seed = int(hash_seed)
        self.rescale = bool(rescale)
        self.theta: float = 1.0          # payload-slot threshold
        self.theta_h: float = 1.0        # hash-slot threshold (cardinality)
        # root_id -> (u, payload, mass, refinement)
        self.entries: Dict[str, Tuple[float, np.ndarray, float, float]] = {}
        self.hashes: Dict[str, float] = {}

    # ------------------------------------------------------------------ core
    def _trim(self) -> None:
        if len(self.entries) > self.capacity:
            order = sorted(self.entries.items(), key=lambda kv: kv[1][0])
            self.theta = order[self.capacity][1][0]      # (k+1)-th smallest hash
            self.entries = dict(order[: self.capacity])
        if len(self.hashes) > self.hash_capacity:
            order = sorted(self.hashes.items(), key=lambda kv: kv[1])
            self.theta_h = order[self.hash_capacity][1]
            self.hashes = dict(order[: self.hash_capacity])

    def insert(self, root_id: str, payload: np.ndarray, mass: float,
               refinement: float = 0.0) -> None:
        u = root_uniform(root_id, self.hash_seed)
        if u < self.theta_h and root_id not in self.hashes:
            self.hashes[root_id] = u
        if u >= self.theta:
            self._trim()
            return                                        # provably discardable
        cand = (u, np.asarray(payload, dtype=np.float64).copy(), float(mass), float(refinement))
        cur = self.entries.get(root_id)
        if cur is None or version_key(cand[3], cand[1], cand[2]) > version_key(cur[3], cur[1], cur[2]):
            self.entries[root_id] = cand                  # refine in place, still one root
        self._trim()

    def merge(self, other: "ThetaSketch") -> "ThetaSketch":
        if other.hash_seed != self.hash_seed:
            raise ValueError("cannot merge sketches with different hash seeds")
        out = ThetaSketch(self.payload_dim, self.capacity, self.hash_seed, self.hash_capacity,
                          self.rescale)
        out.theta = min(self.theta, other.theta)
        out.theta_h = min(self.theta_h, other.theta_h)
        entries: Dict[str, Tuple[float, np.ndarray, float, float]] = {}
        for src in (self.entries, other.entries):
            for root, value in src.items():
                if value[0] >= out.theta:
                    continue
                cur = entries.get(root)                   # union across roots,
                if cur is None or version_key(value[3], value[1], value[2]) > version_key(cur[3], cur[1], cur[2]):
                    entries[root] = value                 # join within a root
        hashes: Dict[str, float] = {}
        for src in (self.hashes, other.hashes):
            for root, u in src.items():
                if u < out.theta_h:
                    hashes[root] = u
        out.entries, out.hashes = entries, hashes
        out._trim()
        return out

    def copy(self) -> "ThetaSketch":
        out = ThetaSketch(self.payload_dim, self.capacity, self.hash_seed, self.hash_capacity,
                          self.rescale)
        out.theta, out.theta_h = self.theta, self.theta_h
        out.entries, out.hashes = dict(self.entries), dict(self.hashes)
        return out

    # ------------------------------------------------------------ estimators
    @property
    def is_exact(self) -> bool:
        return self.theta >= 1.0

    def estimate_payload(self) -> np.ndarray:
        """Belief readout.

        ``rescale=True``  Horvitz-Thompson: an unbiased estimate of what the
            FULL evidence set would imply.  Right magnitude, but the mean is
            located from k roots, so it must be paired with
            ``estimate_payload_cov``.
        ``rescale=False`` conservative: the exact natural-parameter sum of the
            retained roots.  This is a genuine posterior conditioned on a
            lineage-distinct subsample, so it is calibrated by construction with
            no variance correction needed -- and because the retained set is a
            deterministic function of the root set (coordinated hashing), it is
            still merge-invariant, unlike an LRU or reservoir subsample.
        """
        if not self.entries:
            return self._zero()
        total = np.sum([p for _, p, _, _ in self.entries.values()], axis=0)
        if self.is_exact or not self.rescale:
            return total
        return total * (self.estimate_count() / len(self.entries))

    def estimate_mass(self) -> float:
        if not self.entries:
            return 0.0
        total = float(sum(m for _, _, m, _ in self.entries.values()))
        if self.is_exact:
            return total
        return total * (self.estimate_count() / len(self.entries))

    def estimate_count(self) -> float:
        """Cardinality from the (wider) hash set: n_hat = |hashes| / theta_h."""
        n = float(len(self.hashes))
        return n if self.theta_h >= 1.0 else n / self.theta_h

    def estimate_payload_cov(self) -> np.ndarray:
        """Covariance of the Horvitz-Thompson payload estimate.

        Two independent sources of error, both estimable from what is retained:

          sampling  : the mean over k retained payloads estimates the population
                      mean, with a finite-population correction (1 - k/n_hat);
          cardinality: n_hat itself has relative variance ~ 1 / k_hash, which
                      scales the whole estimate.

            V = n_hat^2 [ (1 - k/n_hat) * S / k  +  mean mean^T / k_hash ]
        """
        k = len(self.entries)
        if k == 0 or self.is_exact or not self.rescale:
            return np.zeros((self.payload_dim, self.payload_dim))
        P = np.stack([p for _, p, _, _ in self.entries.values()])
        n_hat = max(self.estimate_count(), float(k))
        mean = P.mean(axis=0)
        if k > 1:
            S = np.cov(P, rowvar=False, ddof=1)
            S = np.atleast_2d(S)
        else:
            S = np.zeros((self.payload_dim, self.payload_dim))
        fpc = max(0.0, 1.0 - k / n_hat)
        V = (n_hat ** 2) * (fpc * S / k + np.outer(mean, mean) / max(self.hash_capacity, 1))
        return V

    # ------------------------------------------------------------ accounting
    def n_bytes(self) -> int:
        # k_p payload slots (8B hash + payload + 8B mass) + k_h bare hashes + 2 thresholds
        return (self.capacity * (8 + 8 * self.payload_dim + 8)
                + self.hash_capacity * 8 + 16)

    def signature(self):
        items = tuple(
            (root, round(u, 15), tuple(np.round(p, 12).tolist()), round(float(m), 12),
             round(float(rf), 12))
            for root, (u, p, m, rf) in sorted(self.entries.items())
        )
        return ("theta", self.capacity, self.hash_capacity, self.rescale, round(self.theta, 15),
                round(self.theta_h, 15), tuple(sorted(self.hashes)), items)
