"""Gate 4D integration: an EC-NCA whose sectors are GROWN, not fixed.

Separate from the frozen fixed-K SectorizedECNCA, which is not modified.

  * learned normalized cell embeddings drive the growing controller at EVERY
    rollout step;
  * a per-example active mask over K_max -- K_max is memory capacity only;
  * active sectors gate neighbour affinity, the mixture prediction, and both the
    local and global sector context;
  * each root is attributed across ACTIVE sectors by a masked per-root softmax,
    so attribution varies by root instead of copying cell membership;
  * sum_k a_irk = 1 holds through birth, merge, retirement and reassignment,
    because the softmax is renormalised over whatever the active set currently
    is, and evidence totals are never recomputed from sector state.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F_

from .dynamic_sectors import GrowingSectorController, SectorConfig


def mlp(sizes, act=nn.SiLU):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


def belief_geometry_loss(model, batch):
    """Label-free metric learning from the posterior implied by local evidence."""
    emb = model.initial_sector_embedding(batch)
    return model._geometry_loss(emb, batch["L0"], batch["root_Lam"],
                                batch["root_h"], batch["prior_precision"])


def _similarity_target(belief):
    """Continuous, label-free pair target induced by posterior locations."""
    d2 = torch.cdist(belief, belief).square()
    B, N, _ = d2.shape
    off = ~torch.eye(N, dtype=torch.bool, device=d2.device).unsqueeze(0)
    mask = off.expand(B, -1, -1)
    vals = d2[mask].reshape(B, -1)
    scale = torch.quantile(vals, 0.35, dim=1).clamp(min=1e-4).view(B, 1, 1)
    target = torch.exp(-d2 / scale)
    return target, mask


def _posterior_compatibility_target(belief, precision):
    """Pair similarity after accounting for posterior uncertainty.

    Raw posterior means from two noisy observations can be far apart while
    remaining statistically compatible.  Their squared Mahalanobis separation
    under the sum of their covariances is approximately chi-square(d) under a
    shared latent.  Only separation beyond that noise floor should create
    pressure for different sectors.
    """
    covariance = torch.linalg.inv(precision.detach())
    diff = belief.unsqueeze(2) - belief.unsqueeze(1)                 # [B,N,N,d]
    pair_cov = covariance.unsqueeze(2) + covariance.unsqueeze(1)     # [B,N,N,d,d]
    solved = torch.linalg.solve(pair_cov, diff.unsqueeze(-1)).squeeze(-1)
    mahal = (diff * solved).sum(-1)
    d = belief.shape[-1]
    excess = (mahal - float(d)).clamp(min=0.0)
    target = torch.exp(-excess / max(2.0 * d, 1.0))
    N = belief.shape[1]
    mask = (~torch.eye(N, dtype=torch.bool, device=belief.device)).unsqueeze(0)
    return target, mask.expand(belief.shape[0], -1, -1)


def _embedding_geometry_loss(emb, belief, precision=None):
    target, mask = (_posterior_compatibility_target(belief.detach(), precision)
                    if precision is not None else _similarity_target(belief.detach()))
    predicted = ((torch.einsum("bnd,bmd->bnm", emb, emb) + 1.0) * 0.5).clamp(0, 1)
    return F_.mse_loss(predicted[mask], target[mask])


class DynamicSectorECNCA(nn.Module):
    def __init__(self, dim=4, obs_dim=4, hidden=64, sim_dim=16, root_emb=48,
                 k_max=12, use_provenance=True, use_sectors=True,
                 sector_cfg: Optional[SectorConfig] = None,
                 fire_prob=0.5, edge_dropout=0.1):
        super().__init__()
        self.dim, self.hidden, self.sim_dim = dim, hidden, sim_dim
        self.k_max, self.use_provenance, self.use_sectors = k_max, use_provenance, use_sectors
        self.fire_prob, self.edge_dropout = fire_prob, edge_dropout
        self.cfg = sector_cfg or SectorConfig(dim=sim_dim, k_max=k_max)
        self.ctrl = GrowingSectorController(self.cfg)

        feat = obs_dim * dim + obs_dim + dim + dim * (dim + 1) // 2 + 1
        self.slot_enc = mlp([feat, root_emb, root_emb])
        self.evidence_proj = nn.Linear(root_emb, hidden)
        # Sector geometry sees both the persistent belief and the evidence that
        # differentiates cells *now*.  Clustering h before the GRU update made
        # every rollout start from identical zero vectors and left the dynamic
        # controller permanently at one sector.
        self.to_sim = nn.Linear(2 * hidden, sim_dim, bias=False)
        nn.init.zeros_(self.to_sim.weight)
        anchor = torch.zeros(sim_dim); anchor[0] = 1.0
        self.register_buffer("sector_anchor", anchor)
        self.sector_anchor_scale = 0.25
        # Sector identity has an explicit semantic coordinate system: the
        # current posterior location.  A bounded learned residual lets neural
        # computation move a cell near a boundary without allowing an arbitrary
        # hidden-state rotation to manufacture sectors after consensus.
        self.log_content_residual_scale = nn.Parameter(torch.tensor(0.0))
        self.sector_proj = mlp([hidden, hidden])
        self.sector_score = mlp([1 + hidden, hidden, 1])
        self.msg = mlp([hidden, hidden])
        # Keep the same shape for sector and no-sector ablations.  No-sector
        # variants receive zero contexts; capacity is therefore matched rather
        # than silently removing a large block of GRU parameters.
        self.gru = nn.GRUCell(4 * hidden, hidden)
        self.rho_head = mlp([hidden + root_emb, hidden, 1])
        nn.init.constant_(self.rho_head[-1].bias, -3.0)
        self.attr_head = mlp([hidden + root_emb, hidden, k_max])
        self.mu_head = mlp([hidden + sim_dim, hidden, dim])
        # The analytic posterior is the calibrated starting point; the neural
        # head learns a refinement residual instead of rediscovering Gaussian
        # fusion from scratch.  Zero initialisation makes this exact at step 0.
        nn.init.zeros_(self.mu_head[-1].weight)
        nn.init.zeros_(self.mu_head[-1].bias)

    # ------------------------------------------------------------------ util
    def _neighbour_max(self, X, adj):
        B, N, S = X.shape
        if adj.dim() == 2:
            mask = (adj > 0).view(1, N, N, 1)
        elif adj.dim() == 3:
            mask = (adj > 0).unsqueeze(-1)
        else:
            raise ValueError(f"adjacency must be [N,N] or [B,N,N], got {adj.shape}")
        nb = X.unsqueeze(1).expand(B, N, N, S).masked_fill(~mask, float("-inf")).amax(dim=2)
        return torch.maximum(X, torch.nan_to_num(nb, neginf=0.0))

    def _affinity_neighbours(self, h, adj, q=None):
        if q is None:
            deg = adj.sum(dim=1, keepdim=True).clamp(min=1.0)
            return torch.einsum("ij,bjh->bih", adj / deg, h)
        aff = torch.einsum("bik,bjk->bij", q, q)
        w = adj.unsqueeze(0) * aff
        return torch.einsum("bij,bjh->bih", w / w.sum(-1, keepdim=True).clamp(min=1e-6), h)

    def _sector_embedding(self, h, v, belief):
        """Posterior geometry plus a learned content residual.

        This is the vector-space counterpart of consistent hashing: prototypes
        partition the current semantic space, and a cell is reassigned when its
        vector crosses a Voronoi boundary.  No fixed semantic sector count is
        supplied.  Population centring makes the construction translation
        invariant across independently generated examples.
        """
        semantic = belief - belief.mean(dim=1, keepdim=True)
        base = h.new_zeros(*semantic.shape[:-1], self.sim_dim)
        n_copy = min(self.dim, self.sim_dim)
        base[..., :n_copy] = semantic[..., :n_copy]
        residual = torch.tanh(self.to_sim(torch.cat([h, v], dim=-1)))
        residual = residual - residual.mean(dim=1, keepdim=True)
        content_scale = self.log_content_residual_scale.exp().clamp(0.05, 2.0)
        anchored = (base + content_scale * residual
                    + self.sector_anchor_scale * self.sector_anchor)
        return F_.normalize(anchored, dim=-1, eps=1e-8)

    def _current_belief(self, W, Lam_s, h_s, prior):
        eye = torch.eye(self.dim, device=W.device)
        Lam = prior * eye + torch.einsum("bns,bsij->bnij", W, Lam_s)
        hnat = torch.einsum("bns,bsd->bnd", W, h_s)
        return torch.linalg.solve(Lam, hnat.unsqueeze(-1)).squeeze(-1)

    def initial_sector_embedding(self, batch):
        """Differentiable geometry of the evidence initially held by each cell.

        This is used by the label-free development loss to make vector distance
        reflect local posterior disagreement.  It consumes the same slot
        features and placements as the rollout, never hypothesis labels.
        """
        if self.use_provenance:
            feat, valid, W0 = batch["root_feat"], batch["root_valid"], batch["L0"]
        else:
            feat, valid, W0 = batch["occ_feat"], batch["occ_valid"], batch["O0"]
        e = self.slot_enc(feat) * valid.unsqueeze(-1)
        v = self.evidence_proj(torch.einsum("bns,bse->bne", W0, e))
        h0 = torch.zeros_like(v)
        if self.use_provenance:
            Lam_s, h_s = batch["root_Lam"], batch["root_h"]
        else:
            Lam_s, h_s = batch["occ_Lam"], batch["occ_h"]
        belief = self._current_belief(W0, Lam_s, h_s, batch["prior_precision"])
        return self._sector_embedding(h0, v, belief)

    def _geometry_loss(self, emb, W, Lam_s, h_s, prior):
        """Match sector-vector distance to the posterior each cell can justify.

        The target is recomputed from the current ledger, so the same objective
        rewards separation while cells hold conflicting evidence and rewards
        collapse after their evidence agrees.  It never reads hypothesis or
        sector labels.
        """
        belief = self._current_belief(W, Lam_s, h_s, prior)
        eye = torch.eye(self.dim, device=W.device)
        precision = prior * eye + torch.einsum("bns,bsij->bnij", W, Lam_s)
        return _embedding_geometry_loss(emb, belief, precision)

    def _uncertainty_linkage(self, belief, precision, informative):
        """Link posterior locations that are statistically compatible.

        Under a shared latent, pairwise Mahalanobis separation under the sum of
        posterior covariances is approximately chi-square(d).  The analytic
        three-standard-deviation cutoff avoids adding a held-out-tuned distance
        knob.  Uninformed cells cannot bridge otherwise distinct components.
        """
        covariance = torch.linalg.inv(precision.detach())
        diff = belief.unsqueeze(2) - belief.unsqueeze(1)
        pair_cov = covariance.unsqueeze(2) + covariance.unsqueeze(1)
        solved = torch.linalg.solve(pair_cov, diff.unsqueeze(-1)).squeeze(-1)
        mahal = (diff * solved).sum(-1)
        threshold = float(self.dim) + 3.0 * (2.0 * float(self.dim)) ** 0.5
        linked = mahal <= threshold
        linked = linked & informative.unsqueeze(2) & informative.unsqueeze(1)
        eye = torch.eye(linked.shape[-1], dtype=torch.bool, device=linked.device)
        return linked | eye.unsqueeze(0)

    # ------------------------------------------------------------- forward
    def forward(self, batch, steps: int = 12, instrument: bool = False):
        if self.use_provenance:
            feat, Lam_s = batch["root_feat"], batch["root_Lam"]
            h_s = batch["root_h"]
            valid, W0 = batch["root_valid"], batch["L0"]
            arrival = batch.get("root_arrival")
        else:
            feat, Lam_s = batch["occ_feat"], batch["occ_Lam"]
            h_s = batch["occ_h"]
            valid, W0 = batch["occ_valid"], batch["O0"]
            arrival = batch.get("occ_arrival")
        adj, d, prior = batch["adj"], self.dim, batch["prior_precision"]
        B, N, S = W0.shape
        dev = feat.device

        e = self.slot_enc(feat) * valid.unsqueeze(-1)
        h = torch.zeros(B, N, self.hidden, device=dev)
        rho = torch.zeros(B, N, S, device=dev)
        W = torch.zeros_like(W0)
        states, inst, geometry_losses = None, [], []

        for t in range(steps):
            newly = W0 if arrival is None and t == 0 else (
                W0 * (arrival == t).unsqueeze(1).to(W0.dtype) if arrival is not None
                else torch.zeros_like(W0))
            W = torch.maximum(W, newly) if self.use_provenance else torch.clamp(W + newly, 0, 1)

            # Infer the partition BEFORE messages propagate.  The previous
            # ordering diffused incompatible evidence once while every cell was
            # still in the initial sector, irreversibly erasing the very modes
            # the controller was meant to preserve.
            v_local = self.evidence_proj(torch.einsum("bns,bse->bne", W, e))
            if self.use_sectors:
                belief = self._current_belief(W, Lam_s, h_s, prior)
                emb = self._sector_embedding(h, v_local, belief)
                geometry_losses.append(self._geometry_loss(emb, W, Lam_s, h_s, prior))
                emb_d = emb.detach()
                informative = W.sum(-1) > 0
                density_linked = None
                if self.cfg.uncertainty_birth:
                    eye = torch.eye(self.dim, device=W.device)
                    precision = prior * eye + torch.einsum(
                        "bns,bsij->bnij", W, Lam_s
                    )
                    density_linked = self._uncertainty_linkage(
                        belief, precision, informative
                    )
                if states is None:
                    states = [self.ctrl.init_state(emb_d[b]) for b in range(B)]
                protos, active = [], []
                for b in range(B):
                    births = 0
                    while True:
                        before = states[b].births
                        _, states[b] = self.ctrl.step(
                            emb_d[b], states[b],
                            density_linked=(density_linked[b]
                                            if density_linked is not None else None),
                            eligible=(informative[b]
                                      if self.cfg.uncertainty_birth else None),
                        )
                        if states[b].births == before:
                            break
                        births += states[b].births - before
                        if births >= self.cfg.max_births_per_step:
                            break
                    protos.append(states[b].prototypes)
                    active.append(states[b].active)
                P = torch.stack(protos).detach()
                A = torch.stack(active)
                sim = torch.einsum("bns,bks->bnk", emb, P) / self.cfg.temperature
                q = sim.masked_fill(~A.unsqueeze(1), -float("inf")).softmax(-1)
            else:
                q = P = A = None

            step_adj = adj * (torch.rand_like(adj) > self.edge_dropout).float() \
                if self.training and self.edge_dropout > 0 else adj
            if q is not None:
                hard = q.argmax(-1)
                same_sector = (hard.unsqueeze(2) == hard.unsqueeze(1)).to(step_adj.dtype)
                prop_adj = step_adj.unsqueeze(0) * same_sector
            else:
                prop_adj = step_adj
            fire = (torch.rand(B, N, 1, device=dev) < self.fire_prob).float()
            if self.use_provenance:
                Wn = self._neighbour_max(W, prop_adj) * valid.unsqueeze(1)
                rn = self._neighbour_max(rho, prop_adj) * Wn
            else:
                if prop_adj.dim() == 2:
                    spread_W = torch.einsum("ij,bjs->bis", prop_adj, W)
                    spread_rho = torch.einsum("ij,bjs->bis", prop_adj, rho)
                else:
                    spread_W = torch.einsum("bij,bjs->bis", prop_adj, W)
                    spread_rho = torch.einsum("bij,bjs->bis", prop_adj, rho)
                Wn = torch.clamp(W + spread_W, 0, 1) \
                    * valid.unsqueeze(1)
                rn = torch.clamp(rho + spread_rho, 0, 1) * Wn
            W = fire * Wn + (1 - fire) * W
            rho = fire * rn + (1 - fire) * rho

            v = self.evidence_proj(torch.einsum("bns,bse->bne", W, e))

            if self.use_sectors:
                ctx_g, ctx_l, alpha, M_k = self._sector_context(h, q, rho, Lam_s, A)
                inp = torch.cat([v, self.msg(self._affinity_neighbours(h, adj, q)),
                                 ctx_g.unsqueeze(1).expand(B, N, self.hidden), ctx_l], dim=-1)
            else:
                q = P = A = alpha = M_k = None
                zero_ctx = torch.zeros(B, N, 2 * self.hidden, device=dev)
                inp = torch.cat([v, self.msg(self._affinity_neighbours(h, adj)),
                                 zero_ctx], dim=-1)

            h = self.gru(inp.reshape(B * N, -1), h.reshape(B * N, -1)).view(B, N, self.hidden)
            r_new = torch.sigmoid(self.rho_head(torch.cat(
                [h.unsqueeze(2).expand(B, N, S, self.hidden),
                 e.unsqueeze(1).expand(B, N, S, e.shape[-1])], dim=-1))).squeeze(-1) * W
            rho = torch.maximum(rho, r_new) if self.use_provenance \
                else torch.clamp(rho + r_new, 0, 1)
            if instrument and self.use_sectors:
                inst.append({"step": t, "q": q.detach(),
                             "n_active": float(A.sum(-1).float().mean()),
                             "births": float(sum(s.births for s in states)) / B,
                             "merges": float(sum(s.merges for s in states)) / B,
                             "retirements": float(sum(s.retirements for s in states)) / B})

        final_belief = self._current_belief(W, Lam_s, h_s, prior)
        out = self._readout(h, rho, Lam_s, prior, q, P, A, alpha, e, final_belief)
        out.update(q=q, active=A, sector_mass=M_k, W=W,
                   alpha=alpha,
                   n_active=(A.sum(-1).float() if A is not None else None), rho=rho)
        out["geometry_loss"] = (torch.stack(geometry_losses).mean()
                                if geometry_losses else h.new_zeros(()))
        if instrument:
            out["instrument"] = inst
        return out

    def _sector_context(self, h, q, rho, Lam_s, active):
        denom = q.sum(dim=1, keepdim=True).transpose(1, 2) + 1e-6
        summary = torch.einsum("bnk,bnh->bkh", q, h) / denom
        support = torch.einsum("bnk,bns->bnks", q, rho).amax(dim=1)
        mass = Lam_s.diagonal(dim1=-2, dim2=-1).sum(-1)
        M_k = torch.einsum("bks,bs->bk", support, mass)
        score = self.sector_score(torch.cat(
            [torch.log1p(M_k).unsqueeze(-1), self.sector_proj(summary)], dim=-1)).squeeze(-1)
        score = score.masked_fill(~active, -float("inf"))
        alpha = score.softmax(dim=-1)
        return (torch.einsum("bk,bkh->bh", alpha, summary),
                torch.einsum("bnk,bkh->bnh", q, summary), alpha, M_k)

    def _readout(self, h, rho, Lam_s, prior, q, P, A, alpha, e, belief):
        B, N, _ = h.shape
        S = rho.shape[-1]
        d, eye = self.dim, torch.eye(self.dim, device=h.device)
        Lam_total = prior * eye + torch.einsum("bns,bsij->bnij", rho, Lam_s)
        M = Lam_total.diagonal(dim1=-2, dim2=-1).sum(-1) - prior * d
        if not self.use_sectors or q is None:
            zero = torch.zeros(B, N, self.sim_dim, device=h.device)
            pooled = belief.mean(1, keepdim=True).expand(B, N, d)
            mu = pooled + self.mu_head(torch.cat([h, zero], dim=-1))
            return {"mu": mu, "Lam": Lam_total, "M": M, "mix_mu": mu.unsqueeze(2),
                    "mix_Lam": Lam_total.unsqueeze(2),
                    "mix_w": torch.ones(B, N, 1, device=h.device), "attr": None}
        K = self.k_max
        residual = self.mu_head(torch.cat(
            [h.unsqueeze(2).expand(B, N, K, self.hidden),
             P.unsqueeze(1).expand(B, N, K, self.sim_dim)], dim=-1))
        denom = q.sum(dim=1).unsqueeze(-1).clamp(min=1e-6)
        sector_belief = torch.einsum("bnk,bnd->bkd", q, belief) / denom
        mix_mu = sector_belief.unsqueeze(1) + residual
        # masked PER-ROOT softmax: attribution differs by root, and renormalises
        # over whatever the active set currently is
        logits = self.attr_head(torch.cat(
            [h.unsqueeze(2).expand(B, N, S, self.hidden),
             e.unsqueeze(1).expand(B, N, S, e.shape[-1])], dim=-1))
        logits = logits.masked_fill(~A.view(B, 1, 1, K), -float("inf"))
        attr = logits.softmax(-1)                                  # sum_k = 1
        mix_Lam = prior * eye + torch.einsum("bnsk,bsij->bnkij", rho.unsqueeze(-1) * attr, Lam_s)
        # q is cell membership: it decides communication neighbourhoods.  It is
        # not the posterior probability of a hypothesis.  Using q as the
        # mixture weight made each cell predict only its own sector and erased
        # the system-level multimodal belief.  alpha is the evidence-conditioned
        # population weight and is therefore the correct late-binding readout.
        mix_w = alpha.unsqueeze(1).expand(B, N, K)
        mu = torch.einsum("bnk,bnkd->bnd", mix_w, mix_mu)
        return {"mu": mu, "Lam": Lam_total, "M": M, "mix_mu": mix_mu,
                "mix_Lam": mix_Lam, "mix_w": mix_w, "attr": attr}
