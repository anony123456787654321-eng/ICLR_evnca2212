"""Sectorized Evidence-Conserving NCA.

    new state = learned sector comparison + provenance-aware confidence control

Every message carries m_i = (h_i, P_i, q_i): what the cell thinks, which
external observations influenced it, and its soft membership across sectors.
The point is to separate two things message-passing systems conflate:

  computational innovation   how much a message improves the representation.
                             Unconstrained -- B's abstraction of A's observation
                             is welcome, and may change mu freely.
  evidential independence    how much genuinely new external information backs
                             it.  Bounded by lineage: ten transformations of one
                             observation are not ten observations.

Confidence ceiling
------------------
Confidence is a sum of the roots' OWN precision matrices,

    Lambda_i = Lambda_prior + sum_s rho_is * Lambda_s,     rho in [0, 1]

so it is supported in every direction by real evidence, not merely in total
magnitude.  rho is the learned confidence vector; refinement raises it by max-
join across cells, and a root contributes at most Lambda_s however many times it
arrives.  A scalar trace ceiling would let a cell be arbitrarily confident along
a direction no observation constrains.

Provenance-free ablations
-------------------------
`no_provenance` and `plain` are not this model with the join switched off.  They
consume the OCCURRENCE view, in which every delivered message owns an unrelated
slot carrying no shared identity, and slot weights merge additively.  They
therefore cannot represent the fact that two messages share a source, and a root
delivered r times contributes r * Lambda_r.  Module shapes are identical to the
provenance variants, so the comparison is parameter-matched by construction.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F_


def mlp(sizes, act=nn.SiLU):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


class SectorizedECNCA(nn.Module):
    def __init__(self, dim: int = 4, obs_dim: int = 4, hidden: int = 64,
                 n_sectors: int = 6, sim_dim: int = 32, root_emb: int = 48,
                 use_sectors: bool = True, use_provenance: bool = True,
                 sector_iters: int = 2, fire_prob: float = 0.5,
                 edge_dropout: float = 0.1):
        super().__init__()
        self.dim, self.obs_dim, self.hidden = dim, obs_dim, hidden
        self.K, self.use_sectors, self.use_provenance = n_sectors, use_sectors, use_provenance
        self.sector_iters = sector_iters
        # Without asynchrony every cell holds every root within `diameter` steps
        # and the population collapses to one identical state, leaving the
        # sector layer nothing to compare.
        self.fire_prob, self.edge_dropout = fire_prob, edge_dropout

        feat = obs_dim * dim + obs_dim + dim + dim * (dim + 1) // 2 + 1
        self.slot_enc = mlp([feat, root_emb, root_emb])
        self.evidence_proj = nn.Linear(root_emb, hidden)

        self.to_sim = nn.Linear(hidden, sim_dim, bias=False)
        self.centroid_seed = nn.Parameter(torch.randn(n_sectors, sim_dim) * 0.2)
        self.centroid_update = mlp([sim_dim, sim_dim, sim_dim])
        # Cosine similarity with a learnable temperature.  Raw dot products of
        # an unnormalised h give logits near zero, so the softmax can only ever
        # be flat: measured entropy sat at 96-100% of log K with pairwise JS
        # ~1e-4, i.e. every cell uniformly mixed across every sector.
        self.log_tau = nn.Parameter(torch.tensor(-1.4))       # tau ~ 0.25
        self.sector_proj = mlp([hidden, hidden])
        # influence sees (support, summary) -- deliberately NOT population
        self.sector_score = mlp([1 + hidden, hidden, 1])
        # plasticity: raw_a is passed through softplus so plasticity is
        # MONOTONE increasing in the number of live sectors by construction,
        # rather than only if training happens to land on a positive weight.
        self.plastic_raw_a = nn.Parameter(torch.tensor(0.5))
        self.plastic_bc = nn.Parameter(torch.tensor([0.0, 1.0]))

        self.msg = mlp([hidden, hidden])
        # Two contexts, not one.  The global term is the sector COMPARISON,
        # weighted by evidential support so the largest group cannot dominate.
        # The local term is the summary of the sectors THIS cell belongs to --
        # without it every cell receives an identical context and sectors cannot
        # differentiate anyone's update, which makes multimodality unrepresentable.
        ctx = 2 * hidden + n_sectors if use_sectors else 0
        self.gru = nn.GRUCell(hidden + hidden + ctx, hidden)
        self.rho_head = mlp([hidden + root_emb, hidden, 1])
        nn.init.constant_(self.rho_head[-1].bias, -3.0)   # start humble, earn confidence
        # Sector-conditioned mixture readout: one prediction per sector, so a
        # cell can hold several hypotheses at once instead of averaging them.
        # This is also what makes sectors load-bearing in the LOSS -- a bimodal
        # belief is only expressible by putting the modes in different sectors.
        self.mu_head = mlp([hidden + sim_dim, hidden, dim])
        # Attribution of each root to sectors, summing to one over sectors, so
        # the evidence spent across all modes can never exceed the evidence the
        # cell actually holds.  Duplicating a root across sectors therefore
        # cannot raise total justified confidence.
        self.attr_head = mlp([hidden + root_emb, hidden, n_sectors])
        self.sim_dim = sim_dim

    # ------------------------------------------------------------------ parts
    def _neighbour_max(self, X, adj):
        B, N, S = X.shape
        mask = (adj > 0).view(1, N, N, 1)
        nb = X.unsqueeze(1).expand(B, N, N, S).masked_fill(~mask, float("-inf")).amax(dim=2)
        return torch.maximum(X, torch.nan_to_num(nb, neginf=0.0))

    def _neighbour_add(self, X, adj):
        """Ordinary additive aggregation: no identity, so no deduplication."""
        return torch.clamp(X + torch.einsum("ij,bjs->bis", adj, X), 0.0, 1.0)

    def _sectors(self, h):
        z = F_.normalize(self.to_sim(h), dim=-1)
        seed = self.centroid_seed.unsqueeze(0).expand(h.shape[0], -1, -1)
        c, tau = F_.normalize(seed, dim=-1), self.log_tau.exp().clamp(0.05, 5.0)
        q = None
        for _ in range(self.sector_iters):
            q = (torch.einsum("bns,bks->bnk", z, c) / tau).softmax(dim=-1)
            w = q / (q.sum(dim=1, keepdim=True) + 1e-6)
            # residual on the learned seed: replacing centroids outright is a
            # collapse fixed point at uniform q (every centroid becomes the same
            # population mean, so q stays uniform forever)
            c = F_.normalize(seed + self.centroid_update(
                torch.einsum("bnk,bns->bks", w, z)), dim=-1)
        return q, c

    def _sector_context(self, h, q, rho, slot_mass):
        denom = q.sum(dim=1, keepdim=True).transpose(1, 2) + 1e-6
        summary = torch.einsum("bnk,bnh->bkh", q, h) / denom
        # soft union over member cells: max, so duplicated lineage adds nothing
        support = torch.einsum("bnk,bns->bnks", q, rho).amax(dim=1)
        M_k = torch.einsum("bks,bs->bk", support, slot_mass)
        score = self.sector_score(torch.cat(
            [torch.log1p(M_k).unsqueeze(-1), self.sector_proj(summary)], dim=-1)).squeeze(-1)
        alpha = score.softmax(dim=-1)
        ctx_global = torch.einsum("bk,bkh->bh", alpha, summary)        # comparison
        ctx_local = torch.einsum("bnk,bkh->bnh", q, summary)           # own sectors
        return ctx_global, ctx_local, alpha, M_k

    @staticmethod
    def _effective_sectors(M_k):
        pi = M_k / (M_k.sum(dim=-1, keepdim=True) + 1e-6)
        return (-(pi * torch.log(pi + 1e-9)).sum(dim=-1)).exp()

    # ---------------------------------------------------------------- forward
    def forward(self, batch, steps: int = 12, instrument: bool = False,
                trace: bool = False):
        if self.use_provenance:
            feat, Lam_s = batch["root_feat"], batch["root_Lam"]
            valid, W0 = batch["root_valid"], batch["L0"]
            arrival = batch.get("root_arrival")
        else:
            feat, Lam_s = batch["occ_feat"], batch["occ_Lam"]
            valid, W0 = batch["occ_valid"], batch["O0"]
            arrival = batch.get("occ_arrival")
        W = torch.zeros_like(W0)
        adj, d = batch["adj"], self.dim
        B, N, S = W.shape
        slot_mass = Lam_s.diagonal(dim1=-2, dim2=-1).sum(-1)

        e = self.slot_enc(feat) * valid.unsqueeze(-1)
        h = torch.zeros(B, N, self.hidden, device=feat.device)
        rho = torch.zeros(B, N, S, device=feat.device)
        prev_hard, inst, tr = None, [], []

        for t in range(steps):
            # evidence may arrive part-way through the rollout, which is what
            # lets a later independent observation resolve an ambiguity
            if arrival is None:
                newly = W0 if t == 0 else torch.zeros_like(W0)
            else:
                newly = W0 * (arrival == t).unsqueeze(1).to(W0.dtype)
            W = torch.maximum(W, newly) if self.use_provenance \
                else torch.clamp(W + newly, 0.0, 1.0)

            step_adj = adj
            if self.training and self.edge_dropout > 0:
                step_adj = adj * (torch.rand_like(adj) > self.edge_dropout).float()
            fire = (torch.rand(B, N, 1, device=feat.device) < self.fire_prob).float()

            if self.use_provenance:
                W_new = self._neighbour_max(W, step_adj) * valid.unsqueeze(1)
                rho_new = self._neighbour_max(rho, step_adj) * W_new
            else:
                W_new = self._neighbour_add(W, step_adj) * valid.unsqueeze(1)
                rho_new = self._neighbour_add(rho, step_adj) * W_new
            W = fire * W_new + (1 - fire) * W
            rho = fire * rho_new + (1 - fire) * rho

            v = self.evidence_proj(torch.einsum("bns,bse->bne", W, e))
            if self.use_sectors:
                q, cent = self._sectors(h)
                ctx_g, ctx_l, alpha, M_k = self._sector_context(h, q, rho, slot_mass)
                inp = torch.cat([v, self.msg(self._mean_neighbours(h, adj, q)),
                                 ctx_g.unsqueeze(1).expand(B, N, self.hidden),
                                 ctx_l, q], dim=-1)
            else:
                q = cent = alpha = M_k = None
                inp = torch.cat([v, self.msg(self._mean_neighbours(h, adj))], dim=-1)

            h_new = self.gru(inp.reshape(B * N, -1), h.reshape(B * N, -1)).view(B, N, self.hidden)
            if self.use_sectors:
                k_eff = self._effective_sectors(M_k)
                a = F_.softplus(self.plastic_raw_a)          # positive by construction
                b, c = self.plastic_bc
                g = torch.sigmoid(a * torch.log(k_eff + 1e-6).unsqueeze(-1)
                                  + c * (-(q * torch.log(q + 1e-9)).sum(-1)) + b).unsqueeze(-1)
                h = (1 - g) * h + g * h_new
            else:
                g = k_eff = None
                h = h_new

            new_rho = torch.sigmoid(self.rho_head(torch.cat(
                [h.unsqueeze(2).expand(B, N, S, self.hidden),
                 e.unsqueeze(1).expand(B, N, S, e.shape[-1])], dim=-1))).squeeze(-1) * W
            # distinct-slot merging by max: a root contributes at most Lambda_s
            rho = torch.maximum(rho, new_rho) if self.use_provenance \
                else torch.clamp(rho + new_rho, 0.0, 1.0)

            if trace:
                tr.append({"step": t,
                           "mu": self._readout(h, rho, Lam_s, batch["prior_precision"],
                                               q, cent, e)["mu"],
                           "W": W})
            if instrument and self.use_sectors:
                inst.append(self._instrument(t, q, cent, M_k, k_eff, g, prev_hard))
                prev_hard = q.argmax(-1)

        out = self._readout(h, rho, Lam_s, batch["prior_precision"], q, cent, e)
        out.update(q=q, alpha=alpha, sector_mass=M_k, k_eff=k_eff,
                   plasticity=g.mean() if g is not None else None, W=W, rho=rho)
        if instrument:
            out["instrument"] = inst
        if trace:
            out["trace"] = tr
        return out

    def _mean_neighbours(self, h, adj, q=None):
        """Neighbour aggregation, gated by sector agreement when sectors exist.

        Plain mean aggregation is a diffusion: it pulls every cell toward the
        same belief, which destroys multimodality by construction and leaves the
        sector layer nothing to preserve.  Weighting each neighbour by sector
        affinity sum_k q_ik q_jk gives sectors an actual job -- deciding who a
        cell listens to -- so cells holding competing interpretations stop
        averaging each other away, while cells that agree still refine a shared
        representation.
        """
        if q is None:
            deg = adj.sum(dim=1, keepdim=True).clamp(min=1.0)
            return torch.einsum("ij,bjh->bih", adj / deg, h)
        affinity = torch.einsum("bik,bjk->bij", q, q)          # [B, N, N]
        w = adj.unsqueeze(0) * affinity
        w = w / w.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        return torch.einsum("bij,bjh->bih", w, h)

    def _readout(self, h, rho, Lam_s, prior, q=None, cent=None, e=None):
        """Sector-conditioned mixture, plus the marginal summary.

        Total justified precision is Lambda_prior + sum_r rho_ir Lambda_r --
        computed from the DEDUPLICATED root set and independent of how evidence
        is attributed across sectors, so no arrangement of modes can inflate it.
        Each mode uses only its attributed share.
        """
        B, N, _ = h.shape
        d, eye = self.dim, torch.eye(self.dim, device=h.device)
        Lam_total = prior * eye + torch.einsum("bns,bsij->bnij", rho, Lam_s)
        M = Lam_total.diagonal(dim1=-2, dim2=-1).sum(-1) - prior * d

        if not self.use_sectors or q is None:
            zero = torch.zeros(B, N, self.sim_dim, device=h.device)
            mu = self.mu_head(torch.cat([h, zero], dim=-1))
            return {"mu": mu, "Lam": Lam_total, "M": M,
                    "mix_mu": mu.unsqueeze(2), "mix_Lam": Lam_total.unsqueeze(2),
                    "mix_w": torch.ones(B, N, 1, device=h.device)}

        K = q.shape[-1]
        # one mean per sector, conditioned on that sector's centroid
        h_k = h.unsqueeze(2).expand(B, N, K, self.hidden)
        c_k = cent.unsqueeze(1).expand(B, N, K, self.sim_dim)
        mix_mu = self.mu_head(torch.cat([h_k, c_k], dim=-1))               # [B,N,K,d]

        S = rho.shape[-1]
        attr = self.attr_head(torch.cat(
            [h.unsqueeze(2).expand(B, N, S, self.hidden),
             e.unsqueeze(1).expand(B, N, S, e.shape[-1])], dim=-1)).softmax(-1)  # [B,N,S,K]
        rho_k = rho.unsqueeze(-1) * attr                                    # partitions rho
        mix_Lam = (prior * eye
                   + torch.einsum("bnsk,bsij->bnkij", rho_k, Lam_s))        # [B,N,K,d,d]
        mu = torch.einsum("bnk,bnkd->bnd", q, mix_mu)                       # marginal mean
        return {"mu": mu, "Lam": Lam_total, "M": M, "mix_mu": mix_mu,
                "mix_Lam": mix_Lam, "mix_w": q, "attr": attr}

    # --------------------------------------------------------- instrumentation
    @staticmethod
    def _instrument(step, q, cent, M_k, k_eff, g, prev_hard):
        """Is the sector structure real, or is every cell uniformly mixed?"""
        B, N, K = q.shape
        hard = q.argmax(-1)
        occ = F_.one_hot(hard, K).float().mean(dim=1)                 # [B, K]
        ent = -(q * torch.log(q + 1e-9)).sum(-1)                      # [B, N]
        P, Q = q.unsqueeze(2), q.unsqueeze(1)                         # pairwise
        M = 0.5 * (P + Q)
        js = 0.5 * ((P * (torch.log(P + 1e-9) - torch.log(M + 1e-9))).sum(-1)
                    + (Q * (torch.log(Q + 1e-9) - torch.log(M + 1e-9))).sum(-1))
        off = ~torch.eye(N, dtype=torch.bool, device=q.device)
        cd = torch.cdist(cent, cent)
        koff = ~torch.eye(K, dtype=torch.bool, device=q.device)
        return {
            "step": step,
            "hard": hard.detach(),          # [B, N] -- for the B4 separation test
            "mean_entropy": float(ent.mean()),
            "max_entropy": float(torch.log(torch.tensor(float(K)))),
            "pairwise_js": float(js[:, off].mean()),
            "n_sectors_occupied": float((occ > 0).float().sum(-1).mean()),
            "occupancy_max": float(occ.max(-1).values.mean()),
            "centroid_separation": float(cd[:, koff].mean()),
            "sector_support_gini": float((M_k.max(-1).values
                                          / (M_k.sum(-1) + 1e-6)).mean()),
            "switch_rate": float((hard != prev_hard).float().mean()) if prev_hard is not None else 0.0,
            "k_eff": float(k_eff.mean()), "plasticity": float(g.mean()),
        }


def gaussian_nll(mu, Lam, x_true):
    err = (x_true.unsqueeze(1) - mu).unsqueeze(-1)
    quad = (err.transpose(-1, -2) @ Lam @ err).squeeze(-1).squeeze(-1)
    logdet = torch.linalg.slogdet(Lam)[1]
    import math
    return 0.5 * (quad - logdet + mu.shape[-1] * math.log(2 * math.pi))


def mixture_nll(mix_mu, mix_Lam, mix_w, x_true):
    """-log sum_k w_k N(x; mu_k, Lambda_k^-1), per cell.

    Under ambiguity a single Gaussian must sit between the hypotheses and is
    penalised at whichever one is true; a mixture that keeps them apart is not.
    That is the gradient pressure which makes sectors align with hypotheses
    instead of being an unused side channel.
    """
    import math
    err = (x_true.unsqueeze(1).unsqueeze(2) - mix_mu).unsqueeze(-1)      # [B,N,K,d,1]
    quad = (err.transpose(-1, -2) @ mix_Lam @ err).squeeze(-1).squeeze(-1)
    logdet = torch.linalg.slogdet(mix_Lam)[1]
    d = mix_mu.shape[-1]
    log_comp = -0.5 * (quad - logdet + d * math.log(2 * math.pi))
    return -torch.logsumexp(torch.log(mix_w.clamp(min=1e-9)) + log_comp, dim=-1)


def mode_separation(mix_mu, mix_w, min_w: float = 0.05):
    """Mean pairwise distance between modes carrying non-negligible weight."""
    B, N, K, d = mix_mu.shape
    if K < 2:
        return torch.zeros((), device=mix_mu.device)
    live = (mix_w > min_w).float()
    pair = torch.cdist(mix_mu.reshape(B * N, K, d), mix_mu.reshape(B * N, K, d))
    m = (live.unsqueeze(-1) * live.unsqueeze(-2)).reshape(B * N, K, K)
    off = 1.0 - torch.eye(K, device=mix_mu.device)
    m = m * off
    return (pair * m).sum() / m.sum().clamp(min=1e-6)


def coverage(mu, Lam, x_true, z: float = 1.959963985):
    """Fraction of coordinates inside the marginal 95% interval."""
    Sigma = torch.linalg.inv(Lam)
    sd = Sigma.diagonal(dim1=-2, dim2=-1).clamp(min=1e-12).sqrt()
    return ((x_true.unsqueeze(1) - mu).abs() <= z * sd).float().mean()
