"""Gaussian belief propagation baselines: backtracking, cavity, and TRW.

In natural parameters the messages are plain additive vectors, so these are the
exact analytic analogues of sum-aggregation NCAs:

  backtrack : m_{i->j} = belief_i                 (includes m_{j->i})
  cavity    : m_{i->j} = local_i + sum_{l != j} m_{l->i}   (= non-backtracking
              walk aggregation; kills 2-cycles only)
  trw       : cavity with every message scaled by an edge-appearance
              probability rho -- conservative, but rho has to be known and it
              still cannot see a duplicated *root*, which is not a graph loop.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from ..schedule import DeliverySchedule


def local_payloads(n_cells: int, atoms, payload_dim: int) -> np.ndarray:
    """Sum of every atom delivered to each cell -- duplicates included."""
    out = np.zeros((n_cells, payload_dim))
    for a in atoms:
        out[a.cell] += a.payload
    return out


def run_gaussian_bp(
    adj: List[List[int]],
    local: np.ndarray,
    steps: int,
    mode: str = "cavity",
    rho: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
    schedule: Optional[DeliverySchedule] = None,
) -> np.ndarray:
    n, d = local.shape
    schedule = schedule or DeliverySchedule()
    rng = rng or np.random.default_rng(0)
    if rho is None:
        n_edges = sum(len(v) for v in adj) / 2
        rho = float(np.clip((n - 1) / max(n_edges, 1.0), 1e-3, 1.0))
    w = rho if mode == "trw" else 1.0

    msg: Dict[tuple, np.ndarray] = {(i, j): np.zeros(d) for i in range(n) for j in adj[i]}
    for _ in range(steps):
        new = dict(msg)
        for i in range(n):
            if schedule.fire_prob < 1.0 and rng.random() >= schedule.fire_prob:
                continue
            incoming = {l: msg[(l, i)] for l in adj[i]}
            total = local[i] + w * sum(incoming.values()) if incoming else local[i].copy()
            for j in adj[i]:
                if schedule.drop_prob and rng.random() < schedule.drop_prob:
                    continue
                if mode == "backtrack":
                    new[(i, j)] = total
                else:
                    new[(i, j)] = total - w * incoming[j]
        msg = new

    beliefs = np.zeros((n, d))
    for i in range(n):
        beliefs[i] = local[i] + w * sum(msg[(l, i)] for l in adj[i]) if adj[i] else local[i]
    return beliefs
