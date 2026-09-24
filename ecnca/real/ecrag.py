"""EC-RAG: evidence-conserving aggregation over retrieved passages.

    claim -> retriever -> passages -> FROZEN encoder -> passage graph with
    lineage -> EC-NCA aggregation -> verdict / evidence-selection heads

Each message carries content, lineage, identity, retrieval relevance, credited
evidence and a transformed hidden state.  Two quantities are kept strictly
apart:

  content state    free to change under neural computation
  evidence credit  bounded by the lineage-distinct documents actually received

Verdict logits are produced from content, then SCALED by a monotone function of
credited evidence.  So repetition may reshape the representation but cannot
sharpen the distribution: with one document delivered sixteen times the credit
is one document's worth, and the temperature reflects that.

Variants
  ec_exact        exact root ledger, credit merged by max
  ec_theta        compressed Theta ledger at fixed capacity
  filter_only     lineage known, but the FIRST version of a root is kept
  no_provenance   identical parameters, no lineage: credit accumulates per
                  occurrence, so duplicates inflate confidence
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..hashing import root_uniform

VARIANTS = ("ec_exact", "ec_theta", "filter_only", "no_provenance")
LABELS = ["Supported", "Refuted", "Not Enough Evidence",
          "Conflicting Evidence/Cherrypicking"]


def mlp(sizes, act=nn.SiLU):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


def theta_keep_mask(root_ids: Sequence[Sequence[str]], capacity: int,
                    hash_seed: int = 0) -> torch.Tensor:
    """Bottom-k retention over DISTINCT roots, per example.

    Coordinated hashing: whether a root is retained depends only on the root id,
    so two cells that never met keep the same subset -- which is what makes the
    ledger mergeable.
    """
    B = len(root_ids)
    S = max((len(r) for r in root_ids), default=0)
    keep = torch.zeros(B, S)
    for b, roots in enumerate(root_ids):
        uniq = sorted({r for r in roots if r})
        if len(uniq) <= capacity:
            kept = set(uniq)
        else:
            kept = set(sorted(uniq, key=lambda r: root_uniform(r, hash_seed))[:capacity])
        for i, r in enumerate(roots):
            keep[b, i] = 1.0 if r in kept else 0.0
    return keep


class ECRag(nn.Module):
    def __init__(self, emb_dim: int = 64, hidden: int = 128, n_labels: int = 4,
                 variant: str = "ec_exact", theta_capacity: int = 8,
                 steps: int = 4, dropout: float = 0.1):
        super().__init__()
        self.variant, self.steps = variant, steps
        self.theta_capacity = theta_capacity
        self.use_provenance = variant in ("ec_exact", "ec_theta", "filter_only")
        self.keep_latest = variant != "filter_only"
        self.enc_claim = mlp([emb_dim, hidden, hidden])
        # +2: retrieval relevance and claim-passage similarity
        self.enc_passage = mlp([emb_dim + 2, hidden, hidden])
        self.msg = mlp([hidden * 2, hidden, hidden])
        self.gru = nn.GRUCell(hidden * 2, hidden)
        self.drop = nn.Dropout(dropout)
        self.rho_head = mlp([hidden, hidden, 1])
        nn.init.constant_(self.rho_head[-1].bias, -1.0)      # start humble
        self.pool = mlp([hidden * 2, hidden, hidden])
        self.verdict = mlp([hidden, hidden, n_labels])
        self.evidence_head = mlp([hidden * 2, hidden, 1])
        # temperature is driven by credited evidence, never by passage count
        self.temp = nn.Parameter(torch.tensor([1.0, 0.5]))

    # ------------------------------------------------------------------ parts
    def _first_index_of_root(self, root_idx: torch.Tensor, mask: torch.Tensor):
        """Index of the first passage carrying each root (filter_only)."""
        B, N = root_idx.shape
        first = torch.zeros(B, N, device=root_idx.device)
        for b in range(B):
            seen = {}
            for i in range(N):
                if mask[b, i] <= 0:
                    continue
                r = int(root_idx[b, i])
                if r not in seen:
                    seen[r] = i
                    first[b, i] = 1.0
        return first

    def forward(self, batch: Dict[str, torch.Tensor],
                collect: bool = False) -> Dict[str, torch.Tensor]:
        claim = batch["claim_emb"]                      # [B, E]
        pas = batch["passage_emb"]                      # [B, N, E]
        mask = batch["passage_mask"]                    # [B, N]
        root_idx = batch["root_index"].long()           # [B, N] -> distinct-root slot
        root_mask = batch["root_mask"]                  # [B, R] valid root slots
        rel = batch.get("retrieval_score", torch.zeros_like(mask))
        B, N, _ = pas.shape
        R = root_mask.shape[1]

        c = self.enc_claim(claim)                                   # [B, H]
        sim = F.cosine_similarity(pas, claim.unsqueeze(1), dim=-1)   # [B, N]
        h = self.enc_passage(torch.cat([pas, rel.unsqueeze(-1),
                                        sim.unsqueeze(-1)], dim=-1)) * mask.unsqueeze(-1)

        # ---- message passing among passages of one claim -------------------
        for _ in range(self.steps):
            deg = mask.sum(-1, keepdim=True).clamp(min=1.0)
            nb = (h * mask.unsqueeze(-1)).sum(1, keepdim=True) / deg.unsqueeze(-1)
            inp = torch.cat([self.msg(torch.cat([h, nb.expand_as(h)], dim=-1)),
                             c.unsqueeze(1).expand(B, N, c.shape[-1])], dim=-1)
            h = self.gru(inp.reshape(B * N, -1), h.reshape(B * N, -1)).view(B, N, -1)
            h = self.drop(h) * mask.unsqueeze(-1)

        # ---- credited evidence: per PASSAGE, then reduced per ROOT ----------
        rho = torch.sigmoid(self.rho_head(h)).squeeze(-1) * mask       # [B, N]
        if self.variant == "ec_theta":
            rho = rho * batch["theta_keep"]
        if self.variant == "filter_only":
            rho = rho * self._first_index_of_root(root_idx, mask)

        onehot = F.one_hot(root_idx.clamp(min=0), R).float() * mask.unsqueeze(-1)
        if self.use_provenance:
            # a root is credited at most once, however many passages carry it
            per_root = (rho.unsqueeze(-1) * onehot).amax(dim=1)         # [B, R]
        else:
            # no lineage: every occurrence adds
            per_root = (rho.unsqueeze(-1) * onehot).sum(dim=1)
        per_root = per_root * root_mask
        credited = per_root.sum(-1)                                     # [B]
        ceiling = root_mask.sum(-1).clamp(min=1e-6)                     # 1.0 per root

        # ---- readout -------------------------------------------------------
        w = (rho * mask).unsqueeze(-1)
        pooled = (h * w).sum(1) / w.sum(1).clamp(min=1e-6)
        state = self.pool(torch.cat([pooled, c], dim=-1))
        logits = self.verdict(state)
        # sharpness is a monotone function of CREDITED EVIDENCE only
        scale = F.softplus(self.temp[0]) + F.softplus(self.temp[1]) * credited
        ev_logits = self.evidence_head(
            torch.cat([h, state.unsqueeze(1).expand(B, N, state.shape[-1])], dim=-1)
        ).squeeze(-1).masked_fill(mask <= 0, -1e4)

        out = {"logits": logits * scale.unsqueeze(-1), "raw_logits": logits,
               "credited_evidence": credited, "ceiling": ceiling,
               "claim_ratio": credited / ceiling, "per_root": per_root,
               "rho": rho, "evidence_logits": ev_logits, "scale": scale,
               "state": state}
        if collect:
            out["h"] = h
        return out
