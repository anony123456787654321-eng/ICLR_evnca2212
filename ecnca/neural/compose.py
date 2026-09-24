"""Gate 3C -- compositional message refinement.

Gate 3R could not separate the latest-version join from a first-version filter:
both reached the same endpoint, so the task had no power to distinguish them.
This benchmark is constructed so that a first-version filter is *provably*
unable to reach the target.

Construction
------------
A root carries a vector x.  Node i owns a private operator T_i, available ONLY
at node i.  The terminal output required is

    y = T_{L-1} T_{L-2} ... T_1 T_0 x

The root is delivered along the chain 0 -> 1 -> ... -> L-1 AND directly to the
terminal node.  So the terminal holds two messages with the same root id:

  version 0    the raw x, straight from the source
  version L-1  the chain-refined payload, carrying T_{L-2}...T_0 x

`full` keeps the latest version and applies its own T_{L-1}: reachable.
`filter_only` keeps the first version, i.e. raw x, and owns only T_{L-1}: the
operators T_0..T_{L-2} are not present anywhere in its inputs, so the target is
not a function of what it holds.  That is an information argument, not an
empirical one -- see tests/test_gate3c.py.

Both messages share a root id and therefore an evidence ceiling: the composition
is COMPUTATION, and credits no additional evidence.

Anti-shortcut rules, asserted in tests:
  * a node's features contain its own operator only;
  * the composed target never appears in any feature;
  * the terminal prediction is decoded from the joined message payload alone --
    no persistent cell state carries content between hops.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

_BIG = 1.0e6


def mlp(sizes, act=nn.SiLU):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


@dataclass
class ComposeSpec:
    dim: int = 4
    max_len: int = 8               # allocation cap; actual length varies
    operator: str = "orthogonal"   # orthogonal | contractive
    noise_sigma: float = 0.0       # exact composition by default


def _operator(rng, d, kind):
    """Numerically stable so error cannot explode from path length alone."""
    A = rng.normal(size=(d, d))
    Q, R = np.linalg.qr(A)
    Q = Q * np.sign(np.diag(R))                 # unique, det = +-1, ||Q|| = 1
    if kind == "contractive":
        Q = 0.9 * Q
    return Q.astype(np.float32)


def make_compose_batch(spec: ComposeSpec, batch_size: int, length: int,
                       direct_to_terminal: bool = True, redeliveries: int = 1,
                       cyclic: bool = False, rng: Optional[np.random.Generator] = None,
                       device: str = "cpu") -> Dict[str, torch.Tensor]:
    """`length` nodes in a chain; node i owns T_i; terminal is node length-1."""
    rng = rng or np.random.default_rng(0)
    B, d, L = batch_size, spec.dim, length
    N = spec.max_len

    x = rng.normal(size=(B, d)).astype(np.float32)
    T = np.zeros((B, N, d, d), np.float32)
    node_valid = np.zeros((B, N), np.float32)
    target = np.zeros((B, d), np.float32)
    partial = np.zeros((B, N, d), np.float32)      # ground truth after each hop

    for b in range(B):
        v = x[b].copy()
        for i in range(L):
            T[b, i] = _operator(rng, d, spec.operator)
            node_valid[b, i] = 1.0
            v = T[b, i] @ v
            partial[b, i] = v
        target[b] = v

    # DIRECTED: adj[i, j] = 1 means node i may receive from node j.  An
    # undirected chain lets the raw copy seeded at the terminal flow BACKWARDS
    # and compete with the forward composition -- node L-2 would adopt the raw
    # vector from the terminal and apply its operator to that instead.
    adj = np.zeros((N, N), np.float32)
    for i in range(L - 1):
        adj[i + 1, i] = 1.0                        # i+1 receives from i
    if cyclic and L > 2:
        adj[0, L - 1] = 1.0                        # terminal feeds back to origin

    # who holds the RAW root at step 0
    seed_count = np.zeros((B, N), np.float32)
    seed_count[:, 0] = 1.0
    # Node 0 is the ORIGIN: the root arriving there is a delivery, so node 0
    # contributes its operator.  A raw copy handed straight to the terminal is
    # NOT a delivery through the chain, so it must not trigger that node's
    # operator -- otherwise the terminal would transform the raw vector and then
    # discard the result when the chain message arrives.
    seed_applies = np.zeros((B, N), np.float32)
    seed_applies[:, 0] = 1.0
    if direct_to_terminal:
        seed_count[:, L - 1] += 1.0                # same root, version 0
    for k in range(1, redeliveries):
        seed_count[:, int(rng.integers(0, L))] += 1.0
    seed_cells = (seed_count > 0).astype(np.float32)

    t = lambda a: torch.as_tensor(a, device=device)
    return {"x": t(x), "T": t(T), "node_valid": t(node_valid), "target": t(target),
            "partial": t(partial), "adj": t(adj), "seed_cells": t(seed_cells),
            "seed_count": t(seed_count), "seed_applies": t(seed_applies),
            "length": length, "terminal": L - 1,
            "root_mass": torch.full((B,), float(d), device=device)}


def oracle(batch) -> torch.Tensor:
    """Exact composition, recomputed from the operators (never a stored answer)."""
    x, T = batch["x"], batch["T"]
    L = batch["length"]
    v = x
    for i in range(L):
        v = torch.einsum("bij,bj->bi", T[:, i], v)
    return v


class ComposingNCA(nn.Module):
    """Variants: full | filter_only | no_provenance | plain."""

    def __init__(self, dim=4, z_dim=None, hidden=64, variant="full"):
        super().__init__()
        # The transported payload lives in the problem's vector space.  The
        # previous 64-D opaque code forced a generic MLP to discover both a
        # coordinate system and matrix multiplication from concatenated inputs;
        # unsurprisingly it learned the unconditional mean instead.  Pairwise
        # T_ij * z_j features make the local operator a *learnable linear map*
        # while leaving transport, ordering and provenance as the mechanism
        # under test.
        self.dim, self.z_dim, self.variant = dim, dim, variant
        self.use_provenance = variant in ("full", "filter_only")
        self.keep_latest = variant != "filter_only"
        self.refine = nn.Linear(dim + dim * dim, dim)
        self.register_buffer("tie_proj", torch.randn(dim) * 0.01)

    def transform(self, payload: torch.Tensor, operator: torch.Tensor) -> torch.Tensor:
        """Learn one local transformation from explicit bilinear features.

        A linear layer can represent ``operator @ payload - payload`` exactly:
        its inputs contain ``payload`` and every product ``T_ij * payload_j``.
        This is an inductive bias for composition, not target leakage--the
        composed answer and all downstream operators remain unavailable.
        """
        products = (operator * payload.unsqueeze(-2)).flatten(-2)
        delta = self.refine(torch.cat([payload, products], dim=-1))
        return payload + delta

    def _join(self, Z, V, W, adj, extra=None):
        B, N, _ = Z.shape
        tie = torch.tanh((Z * self.tie_proj).sum(-1))
        key = (V * _BIG + tie).masked_fill(W <= 0, -float("inf"))
        nb = key.unsqueeze(1).expand(B, N, N).masked_fill((adj <= 0).unsqueeze(0), -float("inf"))
        cand = torch.cat([key.unsqueeze(2), nb], dim=2)                    # [B,N,1+N]
        if self.keep_latest:
            pick = cand.argmax(2)
        else:
            pick = cand.masked_fill(torch.isinf(cand), float("inf")).argmin(2)
        Zc = torch.cat([Z.unsqueeze(2), Z.unsqueeze(1).expand(B, N, N, self.z_dim)], dim=2)
        Vc = torch.cat([V.unsqueeze(2), V.unsqueeze(1).expand(B, N, N)], dim=2)
        Wc = torch.cat([W.unsqueeze(2), W.unsqueeze(1).expand(B, N, N)], dim=2)
        idx = pick.unsqueeze(2)
        Zn = Zc.gather(2, idx.unsqueeze(-1).expand(B, N, 1, self.z_dim)).squeeze(2)
        Vn = Vc.gather(2, idx).squeeze(2)
        Wn = torch.maximum(W, Wc.gather(2, idx).squeeze(2))
        adopted = (pick != 0).float() * Wn
        out_extra = None
        if extra is not None:
            Ec = torch.cat([extra.unsqueeze(2),
                            extra.unsqueeze(1).expand(B, N, N, extra.shape[-1])], dim=2)
            out_extra = Ec.gather(2, idx.unsqueeze(-1).expand(
                B, N, 1, extra.shape[-1])).squeeze(2) * Wn.unsqueeze(-1)
        return Zn * Wn.unsqueeze(-1), Vn * Wn, Wn, adopted, out_extra

    def forward(self, batch, steps: Optional[int] = None, exact: bool = False):
        """Two synchronous phases per step: PROPAGATE, then APPLY.

        Ordered application is enforced by version, not by adoption:

            node i applies T_i  iff  it currently holds version i,
            producing version i+1

        so node 0 acts only on the raw payload, node 1 only on T_0 x, and the
        terminal only on T_{L-2}...T_0 x, giving terminal version L.  A raw copy
        handed straight to the terminal stays at version 0 and therefore cannot
        trigger the terminal operator.  Because the join is a max on version,
        versions never decrease, so a cycle or a re-delivery can never let a node
        apply twice.

        `exact=True` additionally carries the real vector and applies the actual
        matrices -- a non-learned control that must reproduce T_{L-1}...T_0 x.
        """
        x, T, adj = batch["x"], batch["T"], batch["adj"]
        W = batch["seed_cells"].clone()
        B, N = W.shape
        L = int(batch["length"])
        steps = steps if steps is not None else 2 * L + 4
        Z = x.unsqueeze(1).expand(B, N, self.dim).contiguous() * W.unsqueeze(-1)
        V = torch.zeros(B, N, device=x.device)
        E = x.unsqueeze(1).expand(B, N, self.dim).contiguous() * W.unsqueeze(-1)
        # credited evidence for THE one root. Provenance variants merge it by
        # max, so however many copies or cycles deliver it the credit is one
        # root's worth. Provenance-free variants have no identity and so
        # accumulate every arrival as separate evidence.
        seed_count = batch.get("seed_count", W)
        credit = W.clone() if self.use_provenance else seed_count.clone()
        order = torch.arange(N, device=x.device).view(1, N)
        applied = torch.zeros(B, N, device=x.device)
        applied_log = []

        for _ in range(steps):
            # ---- phase 1: propagate (no computation happens here) ----------
            if self.use_provenance:
                Z, V, W, _, E = self._join(Z, V, W, adj, extra=E)
                nb = credit.unsqueeze(1).expand(B, N, N).masked_fill(
                    (adj <= 0).unsqueeze(0), 0.0).amax(2)
                credit = torch.maximum(credit, nb) * W          # join: never > 1
            else:
                deg = adj.sum(1).clamp(min=1).view(1, N, 1)
                Z = Z + torch.einsum("ij,bjz->biz", adj, Z) / deg
                E = E + torch.einsum("ij,bjd->bid", adj, E) / deg.squeeze(-1).unsqueeze(-1)
                W = torch.clamp(W + torch.einsum("ij,bj->bi", adj, W), 0, 1)
                credit = (credit + torch.einsum("ij,bj->bi", adj, credit)) * W

            # ---- phase 2: apply, gated on holding exactly version i ---------
            if self.use_provenance:
                can = ((V == order.float()) & (W > 0)
                       & (order < L) & (applied == 0)).float() * batch["node_valid"]
            else:
                can = (W > 0).float() * batch["node_valid"]   # no order, no identity
            Z_new = self.transform(Z, T)
            Z = Z + (Z_new - Z) * can.unsqueeze(-1)
            if exact:
                E = E + (torch.einsum("bnij,bnj->bni", T, E) - E) * can.unsqueeze(-1)
            V = V + can
            applied = torch.clamp(applied + can, 0, 1)
            applied_log.append(can)

        term = int(batch["terminal"])
        ceiling = batch["root_mass"]
        credited = credit[:, term] * ceiling
        out = {"pred": Z[:, term], "Z": Z, "version": V, "W": W,
               "applied_total": torch.stack(applied_log).sum(0),
               "credited_precision": credited,
               "claim_ratio": credited / ceiling.clamp(min=1e-9),
               "credit_map": credit}
        if exact:
            out["exact_payload"] = E[:, term]
        return out
