"""A faithful message-passing neural network with optional evidence conservation.

Follows the Gilmer et al. abstraction: a shared message function computed per
edge from the pair of endpoint states, neighbourhood aggregation, a shared node
update, and repeated propagation steps. It is deliberately NOT a cellular
automaton: updates are synchronous and there is no stochastic firing, which is
what distinguishes it from `SectorizedECNCA`.

`mpnn_plain` and `mpnn_ec` share a byte-identical content path. Both drive the
content state from the occurrence stream, which is what an ordinary MPNN sees.
They differ only in how credited precision is accumulated:

* `mpnn_ec`   propagates a root ledger by a max join, so a root is credited once
              however many occurrences of it arrive;
* `mpnn_plain` propagates occurrences additively, so every arrival adds credit.

Neither variant is tuned separately; they are the same module under one flag.

`credit_form` is independent of the variant. `matrix`, the default, credits
each root with its own precision matrix, so support accumulates only along the
directions that root's observation constrains. `scalar` credits the same total
information isotropically, `tr(Lambda_r) / d` in every direction, which is what
any source count or set of source identifiers implies once it is turned into
confidence. The two forms share every parameter and differ only there.
`scalar_fit` is the scalar form with one learnable global scale, so a scalar
model can learn how confident to be, though not in which direction.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .model import mlp

VARIANTS = ("mpnn_ec", "mpnn_plain")
CREDIT_FORMS = ("matrix", "scalar", "scalar_fit")


class MPNNEvidence(nn.Module):
    def __init__(self, dim: int = 4, obs_dim: int = 4, hidden: int = 64,
                 variant: str = "mpnn_ec", message_hidden: int | None = 158,
                 credit_form: str = "matrix"):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(variant)
        if credit_form not in CREDIT_FORMS:
            raise ValueError(credit_form)
        self.credit_form = credit_form
        if credit_form == "scalar_fit":
            self.log_scale = nn.Parameter(torch.zeros(()))
        self.variant = variant
        self.use_provenance = variant == "mpnn_ec"
        self.dim, self.hidden = dim, hidden
        # Identical slot-feature layout to SectorizedECNCA, so both backbones
        # read the same observation encoding.
        feat_dim = obs_dim * dim + obs_dim + dim + dim * (dim + 1) // 2 + 1
        message_hidden = message_hidden or hidden  # 158 matches SectorizedECNCA to +0.04%
        self.slot_enc = mlp([feat_dim, hidden, hidden])
        self.evidence_proj = nn.Linear(hidden, hidden)
        # Shared edge message function M(h_v, h_w) and shared node update U.
        self.message = mlp([2 * hidden, message_hidden, hidden])
        self.gru = nn.GRUCell(hidden + hidden, hidden)
        self.mu_head = mlp([hidden, hidden, dim])

    @staticmethod
    def _neighbour_max(X, adj):
        """Max join: a root already held is not re-credited by a neighbour."""
        B, N, S = X.shape
        mask = (adj > 0).view(1, N, N, 1)
        nb = X.unsqueeze(1).expand(B, N, N, S).masked_fill(~mask, float("-inf")).amax(2)
        return torch.maximum(X, torch.nan_to_num(nb, neginf=0.0))

    @staticmethod
    def _neighbour_add(X, adj):
        """Additive aggregation: no identity, so no deduplication."""
        return torch.clamp(X + torch.einsum("ij,bjs->bis", adj, X), 0.0, 1.0)

    def _edge_messages(self, h, adj):
        """Per-edge message from the endpoint pair, summed over neighbours.

        This is the step that makes it an MPNN rather than a mean-field update:
        the message depends on both endpoints, not on a pooled neighbourhood.
        """
        B, N, H = h.shape
        src = h.unsqueeze(1).expand(B, N, N, H)      # h_w, the sender
        dst = h.unsqueeze(2).expand(B, N, N, H)      # h_v, the receiver
        m = self.message(torch.cat([dst, src], dim=-1))
        return (m * adj.view(1, N, N, 1)).sum(2)

    def forward(self, batch, steps: int = 12, **_):
        # Content is driven by the occurrence stream in BOTH variants, so the
        # two differ in accounting alone.
        occ_feat, occ_valid, O0 = batch["occ_feat"], batch["occ_valid"], batch["O0"]
        occ_arrival = batch.get("occ_arrival")
        adj = batch["adj"]
        B, N, O = O0.shape

        # Credit stream: roots for EC, occurrences for plain.
        if self.use_provenance:
            Lam_s, valid, C0 = batch["root_Lam"], batch["root_valid"], batch["L0"]
            arrival = batch.get("root_arrival")
        else:
            Lam_s, valid, C0 = batch["occ_Lam"], occ_valid, O0
            arrival = occ_arrival

        e = self.slot_enc(occ_feat) * occ_valid.unsqueeze(-1)
        h = torch.zeros(B, N, self.hidden, device=occ_feat.device)
        content = torch.zeros_like(O0)
        credit = torch.zeros_like(C0)

        for t in range(steps):
            def deliver(state, initial, arr):
                newly = initial if arr is None or t > 0 else initial
                if arr is not None:
                    newly = initial * (arr == t).unsqueeze(1).to(initial.dtype)
                elif t > 0:
                    newly = torch.zeros_like(initial)
                return torch.clamp(state + newly, 0.0, 1.0)

            content = deliver(content, O0, occ_arrival)
            content = self._neighbour_add(content, adj) * occ_valid.unsqueeze(1)

            credit = torch.maximum(credit, C0) if self.use_provenance \
                else deliver(credit, C0, arrival)
            if self.use_provenance:
                if arrival is not None:
                    credit = torch.maximum(
                        credit, C0 * (arrival == t).unsqueeze(1).to(C0.dtype))
                credit = self._neighbour_max(credit, adj) * valid.unsqueeze(1)
            else:
                credit = self._neighbour_add(credit, adj) * valid.unsqueeze(1)

            v = self.evidence_proj(torch.einsum("bns,bse->bne", content, e))
            inp = torch.cat([v, self._edge_messages(h, adj)], dim=-1)
            h = self.gru(inp.reshape(B * N, -1), h.reshape(B * N, -1)).view(B, N, self.hidden)

        prior = float(batch.get("prior", 1.0))
        eye = torch.eye(self.dim, device=h.device)
        if self.credit_form in ("scalar", "scalar_fit"):
            # Same trace as the directional form, spread evenly over directions.
            iso = Lam_s.diagonal(dim1=-2, dim2=-1).sum(-1) / self.dim
            if self.credit_form == "scalar_fit":
                iso = iso * self.log_scale.exp()
            Lam_s = iso[..., None, None] * eye
        Lam_total = prior * eye + torch.einsum("bns,bsij->bnij", credit, Lam_s)
        M = Lam_total.diagonal(dim1=-2, dim2=-1).sum(-1) - prior * self.dim
        mu = self.mu_head(h)
        return {"mu": mu, "Lam": Lam_total, "M": M,
                "mix_mu": mu.unsqueeze(2), "mix_Lam": Lam_total.unsqueeze(2),
                "mix_w": torch.ones(B, N, 1, device=h.device),
                "credited": credit.sum(-1)}


def build(variant: str, **kw):
    return MPNNEvidence(variant=variant, **kw)


class NonBacktrackingMPNN(nn.Module):
    """Learned non-backtracking message passing.

    This is NOT NBA-GNN. It is a home-built MPNN that excludes the immediate
    reverse message, following the non-backtracking idea that Park et al. (2024)
    formalise; that work is cited as motivation, not as the implementation. The
    name is `nonbacktracking_mpnn` everywhere for exactly this reason.

    States live on directed edges. The message on edge v->w is built from the
    incoming edge messages u->v for every u in N(v) EXCLUDING w, so a message
    cannot immediately turn around. Longer cycles are preserved.

    Its purpose in this paper is a negative one. Non-backtracking propagation
    addresses short cyclic feedback; it does not identify two occurrences
    descended from one external root, so it still inflates under duplication.
    Credit is therefore occurrence-based, like any provenance-free method.
    """

    def __init__(self, dim: int = 4, obs_dim: int = 4, hidden: int = 64,
                 message_hidden: int | None = 158):
        super().__init__()
        self.variant = "nonbacktracking_mpnn"
        self.use_provenance = False
        self.dim, self.hidden = dim, hidden
        feat_dim = obs_dim * dim + obs_dim + dim + dim * (dim + 1) // 2 + 1
        message_hidden = message_hidden or hidden
        self.slot_enc = mlp([feat_dim, hidden, hidden])
        self.evidence_proj = nn.Linear(hidden, hidden)
        self.message = mlp([2 * hidden, message_hidden, hidden])
        self.gru = nn.GRUCell(hidden + hidden, hidden)
        self.mu_head = mlp([hidden, hidden, dim])

    def forward(self, batch, steps: int = 12, **_):
        occ_feat, occ_valid, O0 = batch["occ_feat"], batch["occ_valid"], batch["O0"]
        occ_arrival = batch.get("occ_arrival")
        Lam_s, adj = batch["occ_Lam"], batch["adj"]
        B, N, O = O0.shape
        H = self.hidden

        e = self.slot_enc(occ_feat) * occ_valid.unsqueeze(-1)
        h = torch.zeros(B, N, H, device=occ_feat.device)
        edge = torch.zeros(B, N, N, H, device=occ_feat.device)   # edge[b, v, w] = m_{v->w}
        content = torch.zeros_like(O0)
        mask = (adj > 0).float().view(1, N, N, 1)
        # Exclude the immediate reverse: the message v->w may not consume w->v.
        no_reverse = mask * (1.0 - torch.eye(N, device=adj.device).view(1, N, N, 1))

        for t in range(steps):
            newly = O0 if occ_arrival is None and t == 0 else (
                O0 * (occ_arrival == t).unsqueeze(1).to(O0.dtype)
                if occ_arrival is not None else torch.zeros_like(O0))
            content = torch.clamp(content + newly, 0.0, 1.0)
            content = MPNNEvidence._neighbour_add(content, adj) * occ_valid.unsqueeze(1)

            v_in = self.evidence_proj(torch.einsum("bns,bse->bne", content, e))
            # incoming[b, v, w] = sum over u in N(v)\{w} of edge[b, u, v]
            incoming_all = (edge * mask).sum(1)                      # [B, N, H] over u
            incoming = incoming_all.unsqueeze(2) - edge.transpose(1, 2)
            incoming = incoming * no_reverse.transpose(1, 2)
            src = (h + v_in).unsqueeze(2).expand(B, N, N, H)
            edge = self.message(torch.cat([src, incoming], dim=-1)) * mask

            agg = (edge * mask).sum(1)                               # into each node
            h = self.gru(torch.cat([v_in, agg], dim=-1).reshape(B * N, -1),
                         h.reshape(B * N, -1)).view(B, N, H)

        prior = float(batch.get("prior", 1.0))
        eye = torch.eye(self.dim, device=h.device)
        credit = content * occ_valid.unsqueeze(1)
        Lam_total = prior * eye + torch.einsum("bns,bsij->bnij", credit, Lam_s)
        M = Lam_total.diagonal(dim1=-2, dim2=-1).sum(-1) - prior * self.dim
        mu = self.mu_head(h)
        return {"mu": mu, "Lam": Lam_total, "M": M,
                "mix_mu": mu.unsqueeze(2), "mix_Lam": Lam_total.unsqueeze(2),
                "mix_w": torch.ones(B, N, 1, device=h.device),
                "credited": credit.sum(-1)}
