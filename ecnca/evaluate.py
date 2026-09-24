"""One place that runs any fusion method on any EC-Gaussian example."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .data import Example
from .fusion.aggregators import CovInt, InvCovInt, MeanPool, NaiveSum
from .fusion.bp import local_payloads, run_gaussian_bp
from .metrics import score_cells
from .schedule import DeliverySchedule, run_node_gossip
from .sketch import (ExactLedger, LRUSketch, MembershipSketch, ReservoirSketch, ThetaSketch,
                     split_budget)

GOSSIP_METHODS = ("ec_exact", "ec_theta", "ec_theta_cons", "ec_theta_split", "reservoir", "lru", "bloom",
                  "countmin", "naive_sum", "mean_pool", "cov_int", "inv_cov_int")
BP_METHODS = ("bp_backtrack", "bp_cavity", "bp_trw")
ALL_METHODS = GOSSIP_METHODS + BP_METHODS


def matched_params(kind: str, payload_dim: int, capacity: int) -> Dict:
    """Parameters that fit a competing structure inside the Theta byte budget.

    Theta spends every byte on atom slots.  Bloom/count-min buy membership for
    roots they can no longer carry a payload for: half the atom slots, the rest
    of the budget in bits/counters.  That trade-off is the Pareto axis of
    Figure 4, not an unfair handicap.
    """
    slot = 8 + 8 * payload_dim + 8
    budget = capacity * slot + 8
    if kind == "bloom":
        k_a = max(1, capacity // 2)
        n_bits = max(64, (budget - k_a * slot) * 8)
        return dict(capacity=k_a, n_bits=int(n_bits), n_hashes=3)
    if kind == "countmin":
        k_a = max(1, capacity // 2)
        cells = max(16, (budget - k_a * slot) // 4)
        depth = 3
        return dict(capacity=k_a, cm_width=int(max(8, cells // depth)), cm_depth=depth)
    return dict(capacity=capacity)


def _make_state(method: str, ex: Example, capacity: int, rng: np.random.Generator,
                hash_seed: int):
    d, pd = ex.spec.dim, ex.spec.payload_dim
    if method == "ec_exact":
        return ExactLedger(pd)
    if method == "ec_exact_filter":
        return ExactLedger(pd, refine=False)
    if method == "ec_theta":
        return ThetaSketch(pd, capacity, hash_seed=hash_seed)
    if method == "ec_theta_cons":
        return ThetaSketch(pd, capacity, hash_seed=hash_seed, rescale=False)
    if method == "ec_theta_split":
        kp, kh = split_budget(pd, capacity)
        return ThetaSketch(pd, kp, hash_seed=hash_seed, hash_capacity=kh)
    if method == "reservoir":
        return ReservoirSketch(pd, capacity, rng=rng, hash_seed=hash_seed)
    if method == "lru":
        return LRUSketch(pd, capacity, hash_seed=hash_seed)
    if method in ("bloom", "countmin"):
        p = matched_params(method, pd, capacity)
        return MembershipSketch(pd, backend=method, hash_seed=hash_seed, **p)
    if method == "naive_sum":
        return NaiveSum(pd, d)
    if method == "mean_pool":
        return MeanPool(pd, d)
    if method == "cov_int":
        return CovInt(pd, d)
    if method == "inv_cov_int":
        return InvCovInt(pd, d)
    raise ValueError(method)


def run_method(method: str, ex: Example, steps: int, rng: np.random.Generator,
               capacity: int = 32, schedule: Optional[DeliverySchedule] = None,
               hash_seed: int = 0, probe_steps: Optional[List[int]] = None):
    """Return (payload sums, payload sampling covariances, bytes/cell, trace)."""
    n_cells = len(ex.adj)
    trace: List[Dict] = []

    if method in BP_METHODS:
        local = local_payloads(n_cells, ex.atoms, ex.spec.payload_dim)
        mode = {"bp_backtrack": "backtrack", "bp_cavity": "cavity", "bp_trw": "trw"}[method]
        if probe_steps:
            for t in sorted(probe_steps):
                b = run_gaussian_bp(ex.adj, local, t, mode=mode, rng=np.random.default_rng(rng.integers(1 << 30)),
                                    schedule=schedule)
                trace.append({"step": t, "payload_sums": b})
        beliefs = run_gaussian_bp(ex.adj, local, steps, mode=mode, rng=rng, schedule=schedule)
        nbytes = 8 * ex.spec.payload_dim * max(1, max(len(v) for v in ex.adj))
        return ([beliefs[i] for i in range(n_cells)], [None] * n_cells, nbytes, trace)

    states = [_make_state(method, ex, capacity, rng, hash_seed) for _ in range(n_cells)]
    probe_set = set(probe_steps or [])

    def probe(t, st):
        if t in probe_set:
            trace.append({"step": t, "payload_sums": [s.estimate_payload() for s in st]})

    states = run_node_gossip(states, ex.adj, steps, ex.injections_by_step(), rng,
                             schedule=schedule, probe=probe if probe_set else None)
    nbytes = int(np.mean([s.n_bytes() for s in states]))
    return ([s.estimate_payload() for s in states],
            [s.estimate_payload_cov() for s in states], nbytes, trace)


def evaluate(method: str, ex: Example, steps: int, rng: np.random.Generator,
             sketch_uncertainty: bool = True, **kw) -> Dict:
    """``sketch_uncertainty=False`` reproduces the naive plug-in belief, which is
    the ablation showing why the estimator-variance term is needed at all."""
    sums, covs, nbytes, _ = run_method(method, ex, steps, rng, **kw)
    out = score_cells(ex.spec, sums, ex.x_true, ex.unique_payload_sum,
                      covs if sketch_uncertainty else None)
    out.update(method=method, bytes_per_cell=nbytes, n_unique=ex.n_unique,
               n_atoms=ex.n_atoms, **ex.meta)
    return out
