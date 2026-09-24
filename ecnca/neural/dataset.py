"""Batched EC-Gaussian for the learned model, in TWO views of the same data.

root view         R slots, one per lineage-distinct observation.  Duplicates of
                  a root share its slot, so lineage can be merged by OR/max.
occurrence view   O slots, one per DELIVERED MESSAGE.  A root delivered three
                  times occupies three unrelated slots carrying no shared
                  identity, so duplicates are indistinguishable from separate
                  ordinary messages and can only be merged additively.

Both views carry identical observation content.  The provenance-free ablations
see the occurrence view only: they are not a provenance model with the lineage
join switched off, they simply have no notion that two messages share a source.

Each root declares its own precision matrix Lambda_r = A_r^T A_r / sigma^2,
derivable by a cell from its own observation.  A cell may claim any fraction
rho in [0, 1] of it; NLL disciplines an over-claim, and lineage stops the claim
being counted twice.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch

from ..topology import build


@dataclass
class BatchSpec:
    dim: int = 4
    obs_dim: int = 4
    n_cells: int = 16
    topology: str = "torus"
    prior_precision: float = 1.0
    noise_sigma: float = 1.0
    max_roots: int = 32
    max_occurrences: int = 128


def adjacency(spec: BatchSpec) -> torch.Tensor:
    adj = build(spec.topology, spec.n_cells, np.random.default_rng(0))
    A = torch.zeros(spec.n_cells, spec.n_cells)
    for i, nbrs in enumerate(adj):
        for j in nbrs:
            A[i, j] = 1.0
    return A


def _root_features(A_s, y_s, h_s, Lam_s, mass, tri):
    return np.concatenate([A_s.ravel(), y_s, h_s, Lam_s[tri], [np.log1p(mass)]])


def make_batch(
    spec: BatchSpec,
    batch_size: int,
    n_roots: int,
    copies: int = 1,
    regime: str = "clean",
    rng: Optional[np.random.Generator] = None,
    device: str = "cpu",
    conflict_frac: float = 0.35,
    decoy_scale: float = 2.5,
) -> Dict[str, torch.Tensor]:
    """regimes: clean | duplicate | sybil | spread | conflict | amplified."""
    rng = rng or np.random.default_rng(0)
    B, N, d, m = batch_size, spec.n_cells, spec.dim, spec.obs_dim
    R, O = spec.max_roots, spec.max_occurrences
    sigma, tri = spec.noise_sigma, np.tril_indices(spec.dim)
    feat_dim = m * d + m + d + d * (d + 1) // 2 + 1

    root_feat = np.zeros((B, R, feat_dim), dtype=np.float32)
    root_Lam = np.zeros((B, R, d, d), dtype=np.float32)
    root_mass = np.zeros((B, R), dtype=np.float32)
    root_valid = np.zeros((B, R), dtype=np.float32)
    L0 = np.zeros((B, N, R), dtype=np.float32)

    occ_feat = np.zeros((B, O, feat_dim), dtype=np.float32)
    occ_Lam = np.zeros((B, O, d, d), dtype=np.float32)
    occ_h = np.zeros((B, O, d), dtype=np.float32)
    occ_valid = np.zeros((B, O), dtype=np.float32)
    occ_root = np.full((B, O), -1, dtype=np.int64)      # bookkeeping for METRICS only
    occ_cell = np.full((B, O), -1, dtype=np.int64)
    O0 = np.zeros((B, N, O), dtype=np.float32)

    is_decoy = np.zeros((B, R), dtype=np.float32)
    x_true = rng.normal(0.0, 1.0 / np.sqrt(spec.prior_precision), size=(B, d)).astype(np.float32)
    x_decoy = (x_true + decoy_scale * rng.normal(size=(B, d))).astype(np.float32)
    h_oracle = np.zeros((B, d), dtype=np.float32)
    Lam_oracle = np.zeros((B, d, d), dtype=np.float32)

    for b in range(B):
        r_slot = o_slot = 0
        for s in range(n_roots):
            decoy = regime in ("conflict", "amplified") and rng.random() < conflict_frac
            src = x_decoy[b] if decoy else x_true[b]
            A_s = (rng.normal(0, 1, size=(m, d)) / np.sqrt(d)).astype(np.float32)
            y_s = (A_s @ src + rng.normal(0, sigma, size=m)).astype(np.float32)
            prec = 1.0 / (sigma ** 2)
            Lam_s = (prec * (A_s.T @ A_s)).astype(np.float32)
            h_s = (prec * (A_s.T @ y_s)).astype(np.float32)
            mass_s = float(np.trace(Lam_s))
            feats = _root_features(A_s, y_s, h_s, Lam_s, mass_s, tri).astype(np.float32)

            if regime == "spread":
                stride = max(1, N // max(copies, 1))
                start = int(rng.integers(0, N))
                cells = [(start + c * stride) % N for c in range(min(copies, N))]
            elif regime == "amplified":
                n_c = copies if decoy else 1          # only the minority is amplified
                cells = rng.choice(N, size=min(n_c, N), replace=False)
            elif regime in ("clean", "conflict"):
                cells = rng.choice(N, size=1, replace=False)
            else:
                cells = rng.choice(N, size=min(copies, N), replace=False)

            for c_idx, cell in enumerate(cells):
                sybil = (regime == "sybil" and c_idx > 0)
                if c_idx == 0 or sybil:
                    if r_slot >= R:
                        break
                    r = r_slot
                    r_slot += 1
                    root_feat[b, r], root_Lam[b, r] = feats, Lam_s
                    root_mass[b, r], root_valid[b, r] = mass_s, 1.0
                    is_decoy[b, r] = float(decoy)
                    # the oracle counts LINEAGE-distinct roots; a sybil relabel
                    # really is new lineage as far as any honest system can tell
                    h_oracle[b] += h_s
                    Lam_oracle[b] += Lam_s
                L0[b, cell, r] = 1.0

                if o_slot < O:                        # every delivery is its own slot
                    occ_feat[b, o_slot], occ_Lam[b, o_slot], occ_h[b, o_slot] = feats, Lam_s, h_s
                    occ_valid[b, o_slot] = 1.0
                    occ_root[b, o_slot], occ_cell[b, o_slot] = r, cell
                    O0[b, cell, o_slot] = 1.0
                    o_slot += 1

    t = lambda a: torch.as_tensor(a, device=device)
    return {
        "root_feat": t(root_feat), "root_Lam": t(root_Lam), "root_mass": t(root_mass),
        "root_valid": t(root_valid), "L0": t(L0),
        "occ_feat": t(occ_feat), "occ_Lam": t(occ_Lam), "occ_h": t(occ_h),
        "occ_valid": t(occ_valid),
        "occ_root": t(occ_root), "occ_cell": t(occ_cell), "O0": t(O0),
        "is_decoy": t(is_decoy), "x_true": t(x_true), "x_decoy": t(x_decoy),
        "h_oracle": t(h_oracle), "Lam_oracle": t(Lam_oracle),
        "adj": adjacency(spec).to(device),
        "prior_precision": torch.tensor(float(spec.prior_precision), device=device),
        "n_roots": n_roots, "copies": copies, "regime": regime,
    }


def oracle_posterior(batch: Dict[str, torch.Tensor]):
    """Centralised unique-evidence posterior: the ceiling any method may reach."""
    d = batch["x_true"].shape[-1]
    Lam = batch["Lam_oracle"] + batch["prior_precision"] * torch.eye(d, device=batch["x_true"].device)
    mu = torch.linalg.solve(Lam, batch["h_oracle"].unsqueeze(-1)).squeeze(-1)
    return mu, Lam


def duplicate_batch(batch: Dict[str, torch.Tensor], copies: int,
                    rng: np.random.Generator, spread: bool = False):
    """Re-deliver every root ``copies`` times, changing nothing else.

    Built by extending the existing batch rather than regenerating it, so the
    observations, the latent and the oracle are bit-identical and any change in
    the model's belief is attributable to duplication alone.
    """
    out = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
    L0, O0 = out["L0"], out["O0"]
    B, N, R = L0.shape
    O = O0.shape[-1]
    occ_root, occ_cell = out["occ_root"], out["occ_cell"]
    occ_valid, occ_feat, occ_Lam = out["occ_valid"], out["occ_feat"], out["occ_Lam"]

    for b in range(B):
        free = [o for o in range(O) if occ_valid[b, o] == 0]
        ptr = 0
        for r in range(R):
            if out["root_valid"][b, r] == 0:
                continue
            base_cell = int(L0[b, :, r].argmax())
            for c in range(1, copies):
                if ptr >= len(free):
                    break
                cell = ((base_cell + c * max(1, N // copies)) % N) if spread \
                    else int(rng.integers(0, N))
                L0[b, cell, r] = 1.0
                o = free[ptr]; ptr += 1
                occ_feat[b, o] = out["occ_feat"][b, int((occ_root[b] == r).float().argmax())]
                occ_Lam[b, o] = out["root_Lam"][b, r]
                occ_valid[b, o] = 1.0
                occ_root[b, o], occ_cell[b, o] = r, cell
                O0[b, cell, o] = 1.0
    return out


# ---------------------------------------------------------------------------
# Multimodal ambiguity benchmark
# ---------------------------------------------------------------------------
def make_ambiguous_batch(
    spec: BatchSpec,
    batch_size: int,
    n_hyp: int = 2,
    roots_per_hyp: int = 4,
    resolve_roots: int = 6,
    resolve_step: int = 6,
    copies: int = 1,
    cross_sector_copies: int = 0,
    separation: float = 4.0,
    rng: Optional[np.random.Generator] = None,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """Genuinely ambiguous evidence, with ground-truth hypothesis membership.

    Each example carries ``n_hyp`` well-separated candidate latents, all of them
    plausible from the evidence available early on.  Evidence for hypothesis j is
    injected into a contiguous BLOCK of cells, so different groups of cells
    initially believe different things and the correct sector structure is known
    by construction -- which is what lets recovery be scored with ARI/NMI
    instead of an arbitrary divergence threshold.

    At ``resolve_step`` a burst of independent evidence for the true hypothesis
    arrives across the whole grid.  The losing sector should then collapse,
    cells should switch, and confidence should rise -- because that evidence is
    lineage-distinct, unlike the descendants circulating before it.

    ``cross_sector_copies`` re-delivers roots from one block into the OTHER
    block, so descendants of a single source appear in several sectors.  Under
    evidence conservation that must not raise confidence, however many sectors
    they reach.
    """
    rng = rng or np.random.default_rng(0)
    B, N, d, m = batch_size, spec.n_cells, spec.dim, spec.obs_dim
    R, O = spec.max_roots, spec.max_occurrences
    sigma, tri = spec.noise_sigma, np.tril_indices(spec.dim)
    feat_dim = m * d + m + d + d * (d + 1) // 2 + 1
    side = int(round(np.sqrt(N)))

    root_feat = np.zeros((B, R, feat_dim), dtype=np.float32)
    root_Lam = np.zeros((B, R, d, d), dtype=np.float32)
    root_h = np.zeros((B, R, d), dtype=np.float32)
    root_mass = np.zeros((B, R), dtype=np.float32)
    root_valid = np.zeros((B, R), dtype=np.float32)
    root_hyp = np.full((B, R), -1, dtype=np.int64)
    root_arrival = np.zeros((B, R), dtype=np.int64)
    L0 = np.zeros((B, N, R), dtype=np.float32)

    occ_feat = np.zeros((B, O, feat_dim), dtype=np.float32)
    occ_Lam = np.zeros((B, O, d, d), dtype=np.float32)
    occ_h = np.zeros((B, O, d), dtype=np.float32)
    occ_valid = np.zeros((B, O), dtype=np.float32)
    occ_root = np.full((B, O), -1, dtype=np.int64)
    occ_cell = np.full((B, O), -1, dtype=np.int64)
    occ_arrival = np.zeros((B, O), dtype=np.int64)
    O0 = np.zeros((B, N, O), dtype=np.float32)

    hyp_x = np.zeros((B, n_hyp, d), dtype=np.float32)
    true_hyp = rng.integers(0, n_hyp, size=B).astype(np.int64)
    cell_label = np.zeros((B, N), dtype=np.int64)
    h_oracle = np.zeros((B, d), dtype=np.float32)
    Lam_oracle = np.zeros((B, d, d), dtype=np.float32)

    # cells are split into contiguous column blocks, one per hypothesis
    for i in range(N):
        cell_label[:, i] = min(int((i % side) * n_hyp // side), n_hyp - 1)

    for b in range(B):
        base = rng.normal(size=(d,))
        for j in range(n_hyp):
            direction = rng.normal(size=(d,))
            direction /= np.linalg.norm(direction) + 1e-9
            hyp_x[b, j] = base + separation * direction * (j - (n_hyp - 1) / 2.0)
        r_slot = o_slot = 0

        def emit(x_src, hyp, cells, step):
            nonlocal r_slot, o_slot
            A_s = (rng.normal(0, 1, size=(m, d)) / np.sqrt(d)).astype(np.float32)
            y_s = (A_s @ x_src + rng.normal(0, sigma, size=m)).astype(np.float32)
            prec = 1.0 / (sigma ** 2)
            Lam_s = (prec * (A_s.T @ A_s)).astype(np.float32)
            h_s = (prec * (A_s.T @ y_s)).astype(np.float32)
            mass_s = float(np.trace(Lam_s))
            feats = _root_features(A_s, y_s, h_s, Lam_s, mass_s, tri).astype(np.float32)
            if r_slot >= R:
                return
            r = r_slot; r_slot += 1
            root_feat[b, r], root_Lam[b, r], root_h[b, r] = feats, Lam_s, h_s
            root_mass[b, r], root_valid[b, r] = mass_s, 1.0
            root_hyp[b, r], root_arrival[b, r] = hyp, step
            h_oracle[b] += h_s
            Lam_oracle[b] += Lam_s
            for cell in cells:
                L0[b, cell, r] = 1.0
                if o_slot < O:
                    occ_feat[b, o_slot], occ_Lam[b, o_slot], occ_h[b, o_slot] = feats, Lam_s, h_s
                    occ_valid[b, o_slot] = 1.0
                    occ_root[b, o_slot], occ_cell[b, o_slot] = r, cell
                    occ_arrival[b, o_slot] = step
                    O0[b, cell, o_slot] = 1.0
                    o_slot += 1

        # ---- ambiguous phase: each block gets evidence for its own hypothesis
        for j in range(n_hyp):
            block = np.flatnonzero(cell_label[b] == j)
            for _ in range(roots_per_hyp):
                own = rng.choice(block, size=min(max(1, copies), len(block)), replace=False)
                cells = list(own)
                if cross_sector_copies > 0:
                    other = np.flatnonzero(cell_label[b] != j)
                    cells += list(rng.choice(other, size=min(cross_sector_copies, len(other)),
                                             replace=False))
                emit(hyp_x[b, j], j, cells, 0)

        # ---- resolving phase: independent evidence for the TRUE hypothesis
        for _ in range(resolve_roots):
            cells = rng.choice(N, size=max(1, N // 8), replace=False)
            emit(hyp_x[b, int(true_hyp[b])], int(true_hyp[b]), list(cells), resolve_step)

    x_true = hyp_x[np.arange(B), true_hyp]
    t = lambda a: torch.as_tensor(a, device=device)
    return {
        "root_feat": t(root_feat), "root_Lam": t(root_Lam), "root_mass": t(root_mass),
        "root_valid": t(root_valid), "L0": t(L0), "root_arrival": t(root_arrival),
        "root_h": t(root_h),
        "occ_feat": t(occ_feat), "occ_Lam": t(occ_Lam), "occ_h": t(occ_h),
        "occ_valid": t(occ_valid),
        "occ_root": t(occ_root), "occ_cell": t(occ_cell), "O0": t(O0),
        "occ_arrival": t(occ_arrival), "root_hyp": t(root_hyp),
        "hyp_x": t(hyp_x), "true_hyp": t(true_hyp), "cell_label": t(cell_label),
        "is_decoy": t((root_hyp != true_hyp[:, None]).astype(np.float32)),
        "x_true": t(x_true), "x_decoy": t(hyp_x[:, 0]),
        "h_oracle": t(h_oracle), "Lam_oracle": t(Lam_oracle),
        "adj": adjacency(spec).to(device),
        "prior_precision": torch.tensor(float(spec.prior_precision), device=device),
        "n_roots": int(root_valid[0].sum()), "copies": copies,
        "regime": "ambiguous", "resolve_step": resolve_step,
    }
