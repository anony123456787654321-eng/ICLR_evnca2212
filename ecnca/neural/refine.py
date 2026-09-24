"""Gate 3R -- Learned Lineage Refinement.

The positive claim this module exists to demonstrate:

    neural computation can improve a message derived from one source, without
    that computation being counted as additional independent evidence.

Every message carries an EVOLVING representation:

    (root_id, version, z, rho, q)

``z`` starts from the RAW observation only -- for the Gaussian benchmark the
encoder sees (A, y) and ordinary metadata, never A^T y, A^T A, the oracle
posterior, or the target.  Those appear only in losses and in evaluation.  At
every hop a shared residual network refines it:

    z^(t+1) = z^t + RefineNet(z^t, local_state),   version += 1

The lineage refinement join, for messages sharing a root_id:
  * keep the greatest ``version``, with a deterministic tie-break;
  * update content by REPLACEMENT, never by addition;
  * merge ``rho`` by maximum;
  * cap the root's contribution at its own PSD precision ceiling Lambda_r.

Different root ids accumulate normally.  So a refined descendant may change the
belief arbitrarily while adding exactly zero evidence, and a message going round
a cycle cannot raise its own ceiling.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F_

from ..topology import build

_BIG = 1.0e6            # version dominates the join key; tie-break is fractional


def mlp(sizes, act=nn.SiLU):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


@dataclass
class RefineSpec:
    dim: int = 4
    obs_dim: int = 4
    n_cells: int = 4
    topology: str = "path"
    prior_precision: float = 1.0
    noise_sigma: float = 1.0
    max_roots: int = 8


def make_refine_batch(spec: RefineSpec, batch_size: int, n_roots: int = 1,
                      redeliveries: int = 1, rng: Optional[np.random.Generator] = None,
                      device: str = "cpu") -> Dict[str, torch.Tensor]:
    """Raw observations only.  Sufficient statistics are targets, never inputs."""
    rng = rng or np.random.default_rng(0)
    # Two independent streams.  Delivery placement must NOT consume draws from
    # the observation stream, or changing `redeliveries` would silently change
    # the payloads of later batch elements and the paired comparison would be
    # comparing different examples.
    seeds = rng.integers(1 << 31, size=2)
    obs_rng, place_rng = np.random.default_rng(seeds[0]), np.random.default_rng(seeds[1])
    B, N, d, m = batch_size, spec.n_cells, spec.dim, spec.obs_dim
    R, sigma = spec.max_roots, spec.noise_sigma
    feat_dim = m * d + m + 2                       # vec(A), y, log1p|y|, log1p|A|

    feat = np.zeros((B, R, feat_dim), np.float32)
    Lam = np.zeros((B, R, d, d), np.float32)
    hnat = np.zeros((B, R, d), np.float32)
    valid = np.zeros((B, R), np.float32)
    W0 = np.zeros((B, N, R), np.float32)
    x_true = obs_rng.normal(0, 1 / np.sqrt(spec.prior_precision), size=(B, d)).astype(np.float32)
    h_or = np.zeros((B, d), np.float32)
    L_or = np.zeros((B, d, d), np.float32)

    for b in range(B):
        # Always draw the FULL root budget so the observation stream does not
        # depend on n_roots; only the first n_roots are made valid.  Otherwise
        # adding a second source would silently change the first one, and the
        # new_source axis would not be paired.
        for s in range(R):
            A = (obs_rng.normal(0, 1, size=(m, d)) / np.sqrt(d)).astype(np.float32)
            y = (A @ x_true[b] + obs_rng.normal(0, sigma, size=m)).astype(np.float32)
            if s >= n_roots:
                continue
            prec = 1.0 / sigma ** 2
            Lam[b, s] = prec * (A.T @ A)
            hnat[b, s] = prec * (A.T @ y)
            h_or[b] += hnat[b, s]
            L_or[b] += Lam[b, s]
            feat[b, s] = np.concatenate([A.ravel(), y,
                                         [np.log1p(np.linalg.norm(y)),
                                          np.log1p(np.linalg.norm(A))]])
            valid[b, s] = 1.0
            # the root enters at cell 0 -- on a path that makes A->B->C->D
            # version ordering unambiguous
            W0[b, 0, s] = 1.0
            for k in range(1, redeliveries):
                W0[b, int(place_rng.integers(0, N)), s] = 1.0

    adj = torch.zeros(N, N)
    for i, nb in enumerate(build(spec.topology, N, np.random.default_rng(0))):
        for j in nb:
            adj[i, j] = 1.0
    t = lambda a: torch.as_tensor(a, device=device)
    return {"feat": t(feat), "root_Lam": t(Lam), "root_h": t(hnat), "valid": t(valid),
            "W0": t(W0), "x_true": t(x_true), "h_oracle": t(h_or), "Lam_oracle": t(L_or),
            "adj": adj.to(device),
            "prior_precision": torch.tensor(float(spec.prior_precision), device=device),
            "n_roots": n_roots, "redeliveries": redeliveries}


def oracle_posterior(batch):
    d = batch["x_true"].shape[-1]
    Lam = batch["Lam_oracle"] + batch["prior_precision"] * torch.eye(d, device=batch["x_true"].device)
    return torch.linalg.solve(Lam, batch["h_oracle"].unsqueeze(-1)).squeeze(-1), Lam


class RefiningECNCA(nn.Module):
    """Variants: full | filter_only | no_provenance | plain."""

    def __init__(self, dim=4, obs_dim=4, z_dim=48, hidden=64, variant="full"):
        super().__init__()
        self.dim, self.z_dim, self.variant = dim, z_dim, variant
        self.use_provenance = variant in ("full", "filter_only")
        self.keep_latest = variant != "filter_only"
        feat = obs_dim * dim + obs_dim + 2
        self.enc = mlp([feat, z_dim, z_dim])                     # z^0 from RAW (A, y)
        self.refine = mlp([z_dim + hidden, hidden, z_dim])       # shared residual step
        self.cell = nn.GRUCell(z_dim, hidden)
        self.rho_head = mlp([z_dim + hidden, hidden, 1])
        nn.init.constant_(self.rho_head[-1].bias, -2.0)
        self.dec = mlp([z_dim + hidden, hidden, dim])
        # fixed, non-learned projection: a deterministic tie-break at equal version
        self.register_buffer("tie_proj", torch.randn(z_dim) * 0.01)

    def _join(self, Z, V, rho, W, adj, want_adopted=False):
        """Lineage refinement join across neighbours, per root."""
        B, N, R, _ = Z.shape
        tie = torch.tanh((Z * self.tie_proj).sum(-1))            # in (-1, 1)
        key = V * _BIG + tie                                     # version dominates
        key = key.masked_fill(W <= 0, -float("inf"))
        nb = key.unsqueeze(1).expand(B, N, N, R).masked_fill(
            (adj <= 0).view(1, N, N, 1), -float("inf"))
        cand = torch.cat([key.unsqueeze(2), nb], dim=2)          # [B,N,1+N,R]
        Zc = torch.cat([Z.unsqueeze(2), Z.unsqueeze(1).expand(B, N, N, R, self.z_dim)], dim=2)
        Vc = torch.cat([V.unsqueeze(2), V.unsqueeze(1).expand(B, N, N, R)], dim=2)
        Wc = torch.cat([W.unsqueeze(2), W.unsqueeze(1).expand(B, N, N, R)], dim=2)
        pick = (cand.argmin(2) if not self.keep_latest else cand.argmax(2))
        if not self.keep_latest:                                  # filter_only: keep FIRST
            cand2 = cand.masked_fill(torch.isinf(cand), float("inf"))
            pick = cand2.argmin(2)
        idx = pick.unsqueeze(2)
        Z_new = Zc.gather(2, idx.unsqueeze(-1).expand(B, N, 1, R, self.z_dim)).squeeze(2)
        V_new = Vc.gather(2, idx).squeeze(2)
        W_new = torch.maximum(W, Wc.gather(2, idx).squeeze(2))
        adopted = (pick != 0).float() * W_new       # index 0 is "keep own copy"
        # rho merges by max: a root can never be credited more than once
        rnb = rho.unsqueeze(1).expand(B, N, N, R).masked_fill(
            (adj <= 0).view(1, N, N, 1), 0.0).amax(2)
        res = (Z_new * W_new.unsqueeze(-1), V_new * W_new,
               torch.maximum(rho, rnb) * W_new, W_new)
        return (*res, adopted) if want_adopted else res

    def forward(self, batch, steps: int = 8, max_version: int = 8):
        feat, Lam_s, adj = batch["feat"], batch["root_Lam"], batch["adj"]
        valid, W0 = batch["valid"], batch["W0"]
        B, N, R = W0.shape
        z0 = self.enc(feat) * valid.unsqueeze(-1)                 # [B,R,zdim]
        Z = z0.unsqueeze(1) * W0.unsqueeze(-1)
        V = torch.zeros(B, N, R, device=feat.device)
        rho = torch.zeros(B, N, R, device=feat.device)
        W = W0.clone()
        C = W0.clone()                                            # occurrence count
        h = torch.zeros(B, N, self.cell.hidden_size, device=feat.device)
        preds = []

        for _ in range(steps):
            adopted = None
            if self.use_provenance:
                Z, V, rho, W, adopted = self._join(Z, V, rho, W, adj, want_adopted=True)
            else:
                # every arrival is an unrelated occurrence: content is averaged,
                # evidence ACCUMULATES, so cycles and descendants both inflate
                nbZ = torch.einsum("ij,bjrz->birz", adj, Z)
                deg = adj.sum(1).clamp(min=1).view(1, N, 1, 1)
                Z = torch.where(W.unsqueeze(-1) > 0, Z, torch.zeros_like(Z)) + nbZ / deg
                W = torch.clamp(W + torch.einsum("ij,bjr->bir", adj, W), 0, 1) * valid.unsqueeze(1)
                C = torch.clamp(C + torch.einsum("ij,bjr->bir", adj, C), 0, 16.0) * valid.unsqueeze(1)
                rho = torch.clamp(rho + torch.einsum("ij,bjr->bir", adj, rho), 0, 1) * W

            pooled = (Z * W.unsqueeze(-1)).sum(2)
            h = self.cell(pooled.reshape(B * N, -1), h.reshape(B * N, -1)).view(B, N, -1)

            # Refinement is a CHAIN, not parallel local compute: a cell refines
            # the message when it receives one (A sends, B extracts a feature, C
            # forms a hypothesis...).  If every cell refined its own copy every
            # step there would be no division of labour, and discarding a
            # neighbour's more-refined message would cost nothing -- which is
            # exactly what made filter_only indistinguishable from full.
            fresh = adopted if adopted is not None else W
            if not hasattr(self, "_seeded"):
                pass
            can = (V < max_version).float() * W * torch.clamp(fresh + (V == 0).float() * W, 0, 1)
            dz = self.refine(torch.cat(
                [Z, h.unsqueeze(2).expand(B, N, R, h.shape[-1])], dim=-1))
            Z = Z + dz * can.unsqueeze(-1)                        # residual refinement
            V = V + can
            r_new = torch.sigmoid(self.rho_head(torch.cat(
                [Z, h.unsqueeze(2).expand(B, N, R, h.shape[-1])], dim=-1))).squeeze(-1) * W
            rho = torch.maximum(rho, r_new) if self.use_provenance else torch.clamp(rho + r_new, 0, 1)
            preds.append(self._decode(Z, W, h, rho, Lam_s, C, batch["prior_precision"]))

        out = preds[-1]
        out["preds"] = preds
        out["version"] = V
        out["W"] = W
        return out

    def _decode(self, Z, W, h, rho, Lam_s, C, prior):
        B, N, R, _ = Z.shape
        d = self.dim
        pooled = (Z * W.unsqueeze(-1)).sum(2)
        mu = self.dec(torch.cat([pooled, h], dim=-1))
        weight = rho if self.use_provenance else rho * C          # prov-free double counts
        Lam = (prior * torch.eye(d, device=Z.device)
               + torch.einsum("bnr,brij->bnij", weight, Lam_s))
        return {"mu": mu, "Lam": Lam,
                "M": Lam.diagonal(dim1=-2, dim2=-1).sum(-1) - prior * d}


def anytime_loss(preds, oracle_mu, gamma: float = 1.4):
    """Increasing weight with depth: later refinements must be better."""
    T = len(preds)
    w = torch.tensor([gamma ** t for t in range(T)], device=oracle_mu.device)
    w = w / w.sum()
    total = torch.zeros((), device=oracle_mu.device)
    for t, p in enumerate(preds):
        total = total + w[t] * F_.mse_loss(p["mu"], oracle_mu.unsqueeze(1).expand_as(p["mu"]))
    return total


def train_refine(variant: str, spec: RefineSpec, iters: int = 3000, batch_size: int = 32,
                 steps: int = 8, max_version: int = 8, lr: float = 3e-4, seed: int = 0,
                 device: str = "cpu", out: str = "runs/refine", log_every: int = 250,
                 z_dim: int = 48, hidden: int = 64, refine_weight: float = 10.0,
                 invariance_weight: float = 1.0, anytime_gamma: float = 1.4):
    """Anytime training: decode at every depth, weight later depths more.

    The model must learn progressively better USABLE representations of the same
    raw observation.  The target is not extra Shannon information -- it is
    better extraction of what the observation already carries.
    """
    import json, os, time
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    os.makedirs(out, exist_ok=True)
    model = RefiningECNCA(dim=spec.dim, obs_dim=spec.obs_dim, z_dim=z_dim,
                          hidden=hidden, variant=variant).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters)
    hist, t0 = [], time.time()

    for it in range(1, iters + 1):
        n_roots = int(rng.integers(1, 3))
        k = int(rng.integers(1, 4))
        b = make_refine_batch(spec, batch_size, n_roots=n_roots, redeliveries=1,
                              rng=rng, device=device)
        bd = make_refine_batch(spec, batch_size, n_roots=n_roots, redeliveries=k,
                               rng=np.random.default_rng(int(rng.integers(1 << 30))),
                               device=device)
        for key in ("feat", "root_Lam", "root_h", "valid", "x_true", "h_oracle", "Lam_oracle"):
            bd[key] = b[key]                      # same evidence, more deliveries

        mu_or, _ = oracle_posterior(b)
        out_b = model(b, steps=steps, max_version=max_version)
        refine = anytime_loss(out_b["preds"], mu_or, gamma=anytime_gamma)
        err = (b["x_true"].unsqueeze(1) - out_b["mu"]).unsqueeze(-1)
        quad = (err.transpose(-1, -2) @ out_b["Lam"] @ err).squeeze(-1).squeeze(-1)
        nll = 0.5 * (quad - torch.linalg.slogdet(out_b["Lam"])[1]).mean()
        out_d = model(bd, steps=steps, max_version=max_version)
        inv = (F_.mse_loss(out_d["mu"].mean(1), out_b["mu"].mean(1).detach())
               + F_.mse_loss(out_d["Lam"].mean(1), out_b["Lam"].mean(1).detach()))
        loss = refine_weight * refine + nll + invariance_weight * inv

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()

        if it % log_every == 0 or it == 1:
            gap = float(((out_b["mu"].mean(1) - mu_or) ** 2).sum(-1).sqrt().mean())
            rec = dict(iter=it, refine=float(refine), nll=float(nll), inv=float(inv),
                       gap=gap, secs=round(time.time() - t0, 1))
            hist.append(rec)
            print(f"[{variant}] it {it:5d}  refine {rec['refine']:.4f}  nll {rec['nll']:8.3f}  "
                  f"inv {rec['inv']:.4f}  gap {gap:.4f}  ({rec['secs']:.0f}s)", flush=True)
            tmp = os.path.join(out, "ckpt.pt.tmp")
            torch.save({"model": model.state_dict(), "variant": variant, "iter": it,
                        "history": hist, "spec": spec.__dict__,
                        "architecture": {"z_dim": z_dim, "hidden": hidden},
                        "objective": {"refine_weight": refine_weight,
                                      "invariance_weight": invariance_weight,
                                      "anytime_gamma": anytime_gamma}}, tmp)
            os.replace(tmp, os.path.join(out, "ckpt.pt"))
            json.dump(hist, open(os.path.join(out, "history.json"), "w"), indent=2)
    # Final save stamped with the requested iteration count.  Saving only on log
    # steps left the last recorded iter at e.g. 4998, so `iter >= iters` was
    # never true and a re-run silently RETRAINED instead of loading.
    tmp = os.path.join(out, "ckpt.pt.tmp")
    torch.save({"model": model.state_dict(), "variant": variant, "iter": iters,
                "history": hist, "spec": spec.__dict__,
                "architecture": {"z_dim": z_dim, "hidden": hidden},
                "objective": {"refine_weight": refine_weight,
                              "invariance_weight": invariance_weight,
                              "anytime_gamma": anytime_gamma}}, tmp)
    os.replace(tmp, os.path.join(out, "ckpt.pt"))
    json.dump(hist, open(os.path.join(out, "history.json"), "w"), indent=2)
    return model, hist
