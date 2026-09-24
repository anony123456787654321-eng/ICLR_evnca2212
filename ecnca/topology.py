"""Communication graphs.

The point of the topology sweep is that evidence conservation must not depend on
the graph: a path has no loops, a cycle has one long loop that cavity/
non-backtracking messaging cannot break, a torus has many, and a complete graph
delivers everything to everyone in one step.
"""
from __future__ import annotations

from typing import List

import numpy as np

Adjacency = List[List[int]]


def _sym(n: int, edges) -> Adjacency:
    adj: Adjacency = [[] for _ in range(n)]
    for a, b in edges:
        if b not in adj[a]:
            adj[a].append(b)
        if a not in adj[b]:
            adj[b].append(a)
    return [sorted(v) for v in adj]


def build(name: str, n_cells: int = 16, rng: np.random.Generator | None = None,
          p_edge: float = 0.25) -> Adjacency:
    rng = rng or np.random.default_rng(0)
    name = name.lower()
    if name == "path":
        return _sym(n_cells, [(i, i + 1) for i in range(n_cells - 1)])
    if name == "cycle":
        return _sym(n_cells, [(i, (i + 1) % n_cells) for i in range(n_cells)])
    if name == "tree":
        return _sym(n_cells, [(i, (i - 1) // 2) for i in range(1, n_cells)])
    if name in ("grid", "torus"):
        side = int(round(np.sqrt(n_cells)))
        if side * side != n_cells:
            raise ValueError(f"{name} needs a square cell count, got {n_cells}")
        edges = []
        for r in range(side):
            for c in range(side):
                i = r * side + c
                if c + 1 < side or name == "torus":
                    edges.append((i, r * side + (c + 1) % side))
                if r + 1 < side or name == "torus":
                    edges.append((i, ((r + 1) % side) * side + c))
        return _sym(n_cells, edges)
    if name in ("random", "erdos", "er"):
        edges = [(i, j) for i in range(n_cells) for j in range(i + 1, n_cells)
                 if rng.random() < p_edge]
        adj = _sym(n_cells, edges)
        for i in range(n_cells - 1):          # force connectivity
            if not adj[i]:
                adj = _sym(n_cells, edges + [(i, (i + 1) % n_cells)])
                edges.append((i, (i + 1) % n_cells))
        return adj
    if name == "complete":
        return _sym(n_cells, [(i, j) for i in range(n_cells) for j in range(i + 1, n_cells)])
    raise ValueError(f"unknown topology: {name}")


def diameter(adj: Adjacency) -> int:
    n = len(adj)
    best = 0
    for s in range(n):
        dist = [-1] * n
        dist[s] = 0
        frontier = [s]
        while frontier:
            nxt = []
            for u in frontier:
                for v in adj[u]:
                    if dist[v] < 0:
                        dist[v] = dist[u] + 1
                        nxt.append(v)
            frontier = nxt
        best = max(best, max(dist))
    return best
