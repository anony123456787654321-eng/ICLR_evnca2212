"""Non-provenance aggregation baselines, wrapped in the sketch interface.

These are the "ordinary NCA" style rules.  They see exactly the same atoms and
the same message schedule as EC-NCA; the only thing they lack is lineage, so
they cannot tell a new root from a returning one.

  NaiveSum  : belief = sum of everything received.  Re-circulation on a cycle
              and duplicate roots are both indistinguishable from new evidence.
  MeanPool  : belief = average of everything received.  Immune to duplication
              but cannot accumulate: N distinct sources are as good as one.
  CovInt    : pairwise covariance intersection.  Provably never overconfident
              under unknown correlation, therefore provably under-confident
              when the evidence really is independent.
  InvCovInt : inverse covariance intersection (Noack et al.), the less
              conservative version of the same idea.
"""
from __future__ import annotations

import numpy as np

from ..gaussian import pack, unpack
from ..sketch.base import ProvenanceSketch


class _VectorState(ProvenanceSketch):
    capacity = -1

    def __init__(self, payload_dim: int, dim: int):
        self.payload_dim = int(payload_dim)
        self.dim = int(dim)
        self.vec = np.zeros(self.payload_dim)

    def estimate_mass(self):
        return float(np.trace(unpack(self.estimate_payload(), self.dim)[1]))

    def n_bytes(self):
        return 8 * self.payload_dim

    def signature(self):
        return (self.name, tuple(np.round(self.vec, 12).tolist()))


class NaiveSum(_VectorState):
    name = "naive_sum"

    def insert(self, root_id, payload, mass, refinement=0.0):
        self.vec = self.vec + np.asarray(payload, np.float64)

    def merge(self, other):
        out = NaiveSum(self.payload_dim, self.dim)
        out.vec = self.vec + other.vec
        return out

    def copy(self):
        out = NaiveSum(self.payload_dim, self.dim)
        out.vec = self.vec.copy()
        return out

    def estimate_payload(self):
        return self.vec.copy()

    def estimate_count(self):
        return float("nan")


class MeanPool(_VectorState):
    name = "mean_pool"

    def __init__(self, payload_dim, dim):
        super().__init__(payload_dim, dim)
        self.count = 0.0

    def insert(self, root_id, payload, mass, refinement=0.0):
        self.vec = self.vec + np.asarray(payload, np.float64)
        self.count += 1.0

    def merge(self, other):
        out = MeanPool(self.payload_dim, self.dim)
        out.vec = self.vec + other.vec
        out.count = self.count + other.count
        return out

    def copy(self):
        out = MeanPool(self.payload_dim, self.dim)
        out.vec, out.count = self.vec.copy(), self.count
        return out

    def estimate_payload(self):
        if self.count <= 0:
            return self._zero()
        return self.vec / self.count

    def estimate_count(self):
        return 1.0

    def signature(self):
        return (self.name, tuple(np.round(self.vec, 12).tolist()), round(self.count, 9))


_OMEGA_GRID = np.linspace(0.02, 0.98, 25)


def _omega_star(La, Lb, dim):
    """omega minimising log det of the CI-fused covariance (grid search).

    A grid is used rather than a solver because this runs inside every merge of
    every cell of every step; the objective is smooth and 1-D, and a 25-point
    grid is within 1e-4 nats of the optimum on this problem class.
    """
    eye = 1e-12 * np.eye(dim)
    best_w, best = 0.5, -np.inf
    for w in _OMEGA_GRID:
        sign, ld = np.linalg.slogdet(w * La + (1 - w) * Lb + eye)
        if sign > 0 and ld > best:
            best, best_w = ld, float(w)
    return best_w


class CovInt(_VectorState):
    """Pairwise covariance intersection over the incoming message stream."""

    name = "cov_int"

    def _fuse(self, va, vb):
        ha, La = unpack(va, self.dim)
        hb, Lb = unpack(vb, self.dim)
        if np.allclose(La, 0):
            return vb.copy()
        if np.allclose(Lb, 0):
            return va.copy()
        w = _omega_star(La, Lb, self.dim)
        return pack(w * ha + (1 - w) * hb, w * La + (1 - w) * Lb)

    def insert(self, root_id, payload, mass, refinement=0.0):
        self.vec = self._fuse(self.vec, np.asarray(payload, np.float64))

    def merge(self, other):
        out = CovInt(self.payload_dim, self.dim)
        out.vec = self._fuse(self.vec, other.vec)
        return out

    def copy(self):
        out = CovInt(self.payload_dim, self.dim)
        out.vec = self.vec.copy()
        return out

    def estimate_payload(self):
        return self.vec.copy()

    def estimate_count(self):
        return 1.0


class InvCovInt(CovInt):
    """Inverse covariance intersection (Noack et al. 2017).

    Lambda_f = La + Lb - Lg,  Lg = (w Sa + (1-w) Sb)^-1, with the common
    information mean taken as the w-weighted mean of the two estimates.
    """

    name = "inv_cov_int"

    def _fuse(self, va, vb):
        ha, La = unpack(va, self.dim)
        hb, Lb = unpack(vb, self.dim)
        if np.allclose(La, 0):
            return vb.copy()
        if np.allclose(Lb, 0):
            return va.copy()
        eye = np.eye(self.dim)
        Sa = np.linalg.inv(La + 1e-9 * eye)
        Sb = np.linalg.inv(Lb + 1e-9 * eye)
        w = _omega_star(La, Lb, self.dim)
        Lg = np.linalg.inv(w * Sa + (1 - w) * Sb + 1e-9 * eye)
        xg = w * (Sa @ ha) + (1 - w) * (Sb @ hb)
        Lf = La + Lb - Lg
        ev, V = np.linalg.eigh(0.5 * (Lf + Lf.T))
        Lf = (V * np.clip(ev, 1e-8, None)) @ V.T
        return pack(ha + hb - Lg @ xg, Lf)

    def copy(self):
        out = InvCovInt(self.payload_dim, self.dim)
        out.vec = self.vec.copy()
        return out

    def merge(self, other):
        out = InvCovInt(self.payload_dim, self.dim)
        out.vec = self._fuse(self.vec, other.vec)
        return out
