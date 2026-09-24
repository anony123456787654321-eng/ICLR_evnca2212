"""Closed-form results linking evidence accounting to predictive risk.

Every function here is a theorem from the paper made executable, so a claimed
identity can be checked against simulation rather than trusted. The numbering
matches Appendix~A of the paper.

Three results carry the argument.

`excess_nll` is the master decomposition (Theorem 1). For a Gaussian readout
whose precision is scaled by `lam` and whose mean carries squared Mahalanobis
error `delta`, the expected excess negative log-likelihood against the correct
conditional splits additively:

    excess = (d/2) * phi(lam)  +  lam * delta / 2,        phi(x) = x - 1 - log x

The first term is *precision misallocation* and the second is *mean error*.
Duplication drives the first (lam = m). Compression perturbs it (lam = W_hat/W).
Refinement reduces the second. The terms do not interact beyond the visible
`lam` factor, which is what lets conservation and refinement be argued
separately.

`bottomk_variance` is the exact variance of the plain bottom-k
Horvitz--Thompson total under the implemented (k+1)-st-order-statistic
threshold (Theorem 4).

`split_budget_variance` is the exact variance of the estimator this repository
actually implements (Theorem 5): a narrow payload set of size `k` carried
alongside a wider bare-hash set of size `k_hash` used only for cardinality. It
reduces to `bottomk_variance` when the two budgets coincide, and it is what
makes the allocation rule of Section~6 derivable rather than tuned.

All variance results assume the idealisation stated in Assumption A1: root
hashes are independent and uniform on [0, 1), and payload weights are fixed
given the root set, so that the selection is independent of the values it
selects. `hash_collision_bound` quantifies the only way the first half of that
assumption fails in the implementation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = ["phi", "excess_nll", "duplication_penalty", "duplication_penalty_matrix",
           "bottomk_variance", "split_budget_variance", "relative_variance",
           "compression_excess_nll_bound", "allocate_budget", "Allocation",
           "hash_collision_bound", "PAYLOAD_SLOT_BYTES", "HASH_SLOT_BYTES"]

# Byte costs read off ThetaSketch.n_bytes: a payload slot stores an 8-byte hash,
# `payload_dim` float64s and an 8-byte mass; a hash slot stores only the hash.
HASH_SLOT_BYTES = 8


def PAYLOAD_SLOT_BYTES(payload_dim: int) -> int:
    return 8 * (int(payload_dim) + 2)


# ---------------------------------------------------------------------------
# Theorem 1: the master decomposition
# ---------------------------------------------------------------------------

def phi(x):
    """phi(x) = x - 1 - log x, the Gaussian precision-misallocation penalty.

    Non-negative, zero only at x = 1, convex, and asymptotically quadratic near
    1 with phi(1+e) = e^2/2 + O(e^3). It is the scalar Bregman divergence of
    -log on the precision scale, which is why it appears identically for
    duplication and for compression error.
    """
    x = np.asarray(x, dtype=float)
    if np.any(x <= 0):
        raise ValueError("phi is defined for positive precision multipliers only")
    return x - 1.0 - np.log(x)


def excess_nll(lam, d: int, delta: float = 0.0):
    """Expected excess NLL of a Gaussian readout against the true conditional.

    Reports E_P[-log q] - E_P[-log p] where P = N(mu, Lambda^{-1}) is correct and
    q = N(mu_hat, (lam*Lambda)^{-1}) is the readout, with
    delta = (mu - mu_hat)^T Lambda (mu - mu_hat) the squared Mahalanobis error of
    the mean. Exact, not asymptotic.
    """
    return 0.5 * d * phi(lam) + 0.5 * np.asarray(lam, dtype=float) * float(delta)


def duplication_penalty(m, d: int) -> float:
    """Theorem 2: the price of crediting one observation m times.

    A provenance-free accumulator that receives m copies of every root reports
    precision m*Lambda. With the mean unchanged this costs exactly
    (d/2)(m - 1 - log m) nats, which is also KL(P || Q_m). Evidence conservation
    pays zero: it holds lam = 1 whatever m is.
    """
    return 0.5 * d * phi(m)


def duplication_penalty_matrix(precisions, multiplicities) -> float:
    """Theorem 2, corollary: unequal duplication and matrix-valued precisions.

    `precisions` is a stack of d-by-d PSD source precisions Lambda_r and
    `multiplicities` the count m_r credited to each. The penalty is
    (1/2) sum_i phi(nu_i) over the generalised eigenvalues nu_i of the pair
    (sum_r m_r Lambda_r, sum_r Lambda_r), which collapses to
    duplication_penalty when every m_r equals m.
    """
    precisions = np.asarray(precisions, dtype=float)
    multiplicities = np.asarray(multiplicities, dtype=float)
    if precisions.ndim != 3:
        raise ValueError("precisions must be a stack of square matrices")
    true = precisions.sum(axis=0)
    dup = (multiplicities[:, None, None] * precisions).sum(axis=0)
    # Generalised eigenvalues of (dup, true) via a symmetric whitening, which is
    # numerically better behaved than solving the pencil directly.
    L = np.linalg.cholesky(true)
    Linv = np.linalg.inv(L)
    nu = np.linalg.eigvalsh(Linv @ dup @ Linv.T)
    if np.any(nu <= 0):
        raise ValueError("duplicated precision must stay positive definite")
    return 0.5 * float(np.sum(phi(nu)))


# ---------------------------------------------------------------------------
# Theorems 4 and 5: sketch variance
# ---------------------------------------------------------------------------

def bottomk_variance(weights, k: int) -> float:
    """Theorem 4: exact variance of the plain bottom-k HT total.

        Var = (n - k) / (k - 1) * sum_r w_r^2,      n > k > 1

    for the estimator W_hat = theta^{-1} * sum_{r in S} w_r with S the k roots of
    smallest hash and theta the (k+1)-st smallest hash, which is the threshold
    ThetaSketch._trim actually stores. Below k roots the sketch is exact and the
    variance is zero.
    """
    w = np.asarray(weights, dtype=float)
    n = w.size
    if k < 1:
        raise ValueError("k must be >= 1")
    if n <= k:
        return 0.0
    if k < 2:
        raise ValueError("the (k+1)-st-order-statistic estimator needs k >= 2 "
                         "for a finite variance; E[1/theta] diverges at k = 1")
    return (n - k) / (k - 1) * float(np.sum(w ** 2))


def split_budget_variance(weights, k: int, k_hash: int) -> float:
    """Theorem 5: exact variance of the implemented split-budget estimator.

    The estimator is W_hat = n_hat * mean(retained payloads) with
    n_hat = k_hash / theta_h, which is what ThetaSketch.estimate_mass computes.
    Writing Q = sum_r w_r^2 and W = sum_r w_r, for n > k_hash >= k and
    k_hash >= 2,

        Var = k_hash (n - k) / (k (k_hash - 1)) * Q
              - (k_hash - k) / (k (k_hash - 1)) * W^2.

    Setting k_hash = k recovers Theorem 4 exactly. In the intermediate regime
    k < n <= k_hash the cardinality is known exactly and the estimator is the
    ordinary simple-random-sample total, whose variance is returned instead.

    The derivation rests on the selected set being independent of the order
    statistics of the hashes, which holds because ranks and order statistics of
    an i.i.d. continuous sample are independent (Assumption A1).
    """
    w = np.asarray(weights, dtype=float)
    n = w.size
    k, k_hash = int(k), int(k_hash)
    if k_hash < k:
        raise ValueError("hash budget must be at least the payload budget")
    if n <= k:
        return 0.0
    W = float(w.sum())
    Q = float(np.sum(w ** 2))
    if n <= k_hash:
        # Cardinality exact; W_hat = (n/k) * sum_{r in S} w_r, an SRS total.
        return (n * n) * (1.0 - k / n) * _srs_var(w) / k
    if k_hash < 2:
        raise ValueError("the cardinality estimator needs k_hash >= 2")
    a = k_hash * (n - k) / (k * (k_hash - 1))
    b = (k_hash - k) / (k * (k_hash - 1))
    return a * Q - b * W * W


def _srs_var(w: np.ndarray) -> float:
    """Population variance with the (n-1) divisor used by the SRS total."""
    n = w.size
    if n < 2:
        return 0.0
    return float(np.sum((w - w.mean()) ** 2) / (n - 1))


def relative_variance(weights, k: int, k_hash: int) -> float:
    """Var(W_hat) / W^2, the quantity the risk bound consumes."""
    W = float(np.sum(weights))
    if W == 0.0:
        raise ValueError("relative variance needs a non-zero total weight")
    return split_budget_variance(weights, k, k_hash) / (W * W)


def compression_excess_nll_bound(weights, k: int, k_hash: int, d: int,
                                 delta_prob: float = 0.05,
                                 mean_error: float = 0.0) -> dict:
    """Theorem 6: finite-sample bound on excess NLL from compression alone.

    With probability at least 1 - delta_prob, the compressed readout's excess NLL
    against the exact-ledger readout is at most

        (d/2) * eps^2 / (2 (1 - eps)^2)  +  eps * mean_error / 2,
        eps = sqrt( Var(W_hat) / (W^2 delta_prob) )                (Chebyshev).

    The bound depends on the root count, both capacities, the dispersion of the
    evidence weights and the dimension, and on nothing else. In particular it
    does not depend on how many times any root was delivered, because the ledger
    state is a function of the root SET: that independence is the point of the
    theorem, not a side remark.
    """
    if not 0.0 < delta_prob < 1.0:
        raise ValueError("delta_prob must lie in (0, 1)")
    rel = relative_variance(weights, k, k_hash)
    eps = math.sqrt(rel / delta_prob)
    out = {"relative_variance": rel, "epsilon": eps,
           "failure_probability": delta_prob, "valid": eps < 1.0}
    if eps >= 1.0:
        # Chebyshev is vacuous here; report it rather than a meaningless number.
        out["excess_nll_bound"] = float("inf")
        return out
    out["excess_nll_bound"] = (0.5 * d * eps * eps / (2.0 * (1.0 - eps) ** 2)
                               + 0.5 * eps * float(mean_error))
    return out


# ---------------------------------------------------------------------------
# Theorem 7: budget allocation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Allocation:
    payload_slots: int
    hash_slots: int
    bytes_used: int
    relative_variance: float
    continuous_ratio: float
    boundary: bool          # True when the k_hash >= k constraint is active


def allocate_budget(total_bytes: int, payload_dim: int, weights=None,
                    cv: float | None = None, n: int | None = None,
                    search: int = 6) -> Allocation:
    """Theorem 7: split a fixed byte budget between payload and hash slots.

    Minimises the relative variance subject to
    k * PAYLOAD_SLOT_BYTES + k_hash * HASH_SLOT_BYTES <= total_bytes and
    k_hash >= k >= 1.

    The large-n objective is CV^2/k + 1/k_hash, whose stationary point is

        k_hash / k = sqrt(c_payload / c_hash) / CV = sqrt(payload_dim + 2) / CV,

    so dispersed evidence buys payload slots and uniform evidence buys hashes.
    When CV > sqrt(payload_dim + 2) the unconstrained optimum would want fewer
    hashes than payloads, the k_hash >= k constraint binds, and the answer is the
    plain bottom-k sketch; `boundary` records that.

    The continuous solution is then rounded and a small integer neighbourhood is
    searched against the EXACT variance of Theorem 5, so the returned allocation
    is optimal among integer points near the relaxation rather than merely
    rounded.
    """
    if weights is None and (cv is None or n is None):
        raise ValueError("supply either `weights` or both `cv` and `n`")
    if weights is not None:
        w = np.asarray(weights, dtype=float)
        n = w.size
        mean = float(w.mean())
        if mean == 0.0:
            raise ValueError("evidence weights must have non-zero mean")
        cv = float(w.std(ddof=0) / mean)
    else:
        w = None
    c_p = PAYLOAD_SLOT_BYTES(payload_dim)
    c_h = HASH_SLOT_BYTES
    if total_bytes < c_p + c_h:
        raise ValueError("budget cannot buy even one payload and one hash slot")

    ratio = math.sqrt(c_p / c_h) / cv if cv > 0 else float("inf")
    boundary = ratio < 1.0
    if boundary:
        ratio = 1.0                       # k_hash >= k binds; plain bottom-k
    k_cont = total_bytes / (c_p + c_h * ratio)

    def score(kp: int, kh: int) -> float:
        if kp < 1 or kh < kp or kh < 2 or kp * c_p + kh * c_h > total_bytes:
            return float("inf")
        if w is not None:
            return relative_variance(w, kp, kh)
        # Weights unavailable: score the exact formula on a synthetic population
        # with the same n and CV, which is what the formula depends on.
        synth = _population_with_cv(n, cv)
        return relative_variance(synth, kp, kh)

    best, best_score = None, float("inf")
    centre = max(1, int(round(k_cont)))
    for kp in range(max(1, centre - search), centre + search + 1):
        kh = int((total_bytes - kp * c_p) // c_h)
        if kh < kp:
            kh = kp
        for cand in {kh, kh - 1, kp}:
            if cand < 2:                   # the cardinality estimator needs >= 2
                continue
            s = score(kp, cand)
            if s < best_score:
                best, best_score = (kp, cand), s
    if best is None:
        raise ValueError("no feasible integer allocation within the budget")
    kp, kh = best
    return Allocation(payload_slots=kp, hash_slots=kh,
                      bytes_used=kp * c_p + kh * c_h,
                      relative_variance=best_score,
                      continuous_ratio=ratio, boundary=boundary)


def _population_with_cv(n: int, cv: float) -> np.ndarray:
    """A deterministic weight vector with unit mean and the requested CV."""
    if n < 2:
        return np.ones(1)
    z = np.arange(n, dtype=float)
    z = (z - z.mean()) / z.std()
    w = 1.0 + cv * z
    if w.min() <= 0:                       # keep weights positive
        w = w - w.min() + 1e-6
        w = w / w.mean()
    return w


# ---------------------------------------------------------------------------
# Assumption A1: hashing
# ---------------------------------------------------------------------------

def hash_collision_bound(n_roots: int, bits: int = 64) -> float:
    """Probability that the 64-bit root hashing violates the continuity idealisation.

    The variance theorems assume distinct continuous hashes. The implementation
    uses a 64-bit BLAKE2b digest scaled into [0, 1), so two distinct roots
    collide with probability at most n(n-1)/2 * 2^-bits by a union bound. At the
    scales in this paper this is below 1e-13 and the idealisation is harmless;
    the implementation additionally breaks ties on the root identifier, so a
    collision degrades the estimate rather than the ledger's algebraic
    properties.
    """
    n = int(n_roots)
    return min(1.0, n * (n - 1) / 2.0 * 2.0 ** (-bits))
