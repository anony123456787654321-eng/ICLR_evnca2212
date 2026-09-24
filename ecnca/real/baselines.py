"""Matched baselines that see identical frozen embeddings.

Only EC vs no_provenance needs exact parameter matching, but every algorithmic
baseline receives the same cached vectors, so differences are attributable to
aggregation alone.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ecrag import ECRag, mlp
from .set_transformer import VARIANTS as SET_TRANSFORMER_VARIANTS
from .set_transformer import SetTransformerRag

# Aggregation-only baselines. They share ECRag's encoder/heads so the comparison
# isolates HOW evidence is combined, not how it is represented.
AGGREGATORS = ("mean_pool", "sum_pool", "canonical_dedup", "deepsets")


class AggregationBaseline(nn.Module):
    """mean / sum pooling, canonical-URL dedup, and DeepSets over the same input."""

    def __init__(self, emb_dim: int = 768, hidden: int = 128, n_labels: int = 4,
                 mode: str = "mean_pool", dropout: float = 0.1):
        super().__init__()
        assert mode in AGGREGATORS, mode
        self.mode = mode
        self.enc_claim = mlp([emb_dim, hidden, hidden])
        self.enc_passage = mlp([emb_dim + 2, hidden, hidden])
        self.phi = mlp([hidden, hidden, hidden])          # DeepSets element map
        self.drop = nn.Dropout(dropout)
        self.pool = mlp([hidden * 2, hidden, hidden])
        self.verdict = mlp([hidden, hidden, n_labels])
        self.evidence_head = mlp([hidden * 2, hidden, 1])

    def forward(self, batch: Dict[str, torch.Tensor], collect: bool = False):
        claim, pas, mask = batch["claim_emb"], batch["passage_emb"], batch["passage_mask"]
        rel = batch.get("retrieval_score", torch.zeros_like(mask))
        B, N, _ = pas.shape
        c = self.enc_claim(claim)
        sim = F.cosine_similarity(pas, claim.unsqueeze(1), dim=-1)
        h = self.drop(self.enc_passage(torch.cat(
            [pas, rel.unsqueeze(-1), sim.unsqueeze(-1)], dim=-1))) * mask.unsqueeze(-1)

        m = mask
        if self.mode == "canonical_dedup":
            # keep ONE passage per canonical document root: the strongest
            # non-learned duplicate defence in the RAG literature
            m = mask * batch["first_of_root"]
        if self.mode in ("mean_pool", "canonical_dedup"):
            pooled = (h * m.unsqueeze(-1)).sum(1) / m.sum(1, keepdim=True).clamp(min=1e-6)
        elif self.mode == "sum_pool":
            pooled = (h * m.unsqueeze(-1)).sum(1)
        else:                                              # deepsets
            pooled = (self.phi(h) * m.unsqueeze(-1)).sum(1)
        state = self.pool(torch.cat([pooled, c], dim=-1))
        logits = self.verdict(state)
        ev = self.evidence_head(torch.cat(
            [h, state.unsqueeze(1).expand(B, N, state.shape[-1])], dim=-1)
        ).squeeze(-1).masked_fill(mask <= 0, -1e4)
        # these baselines have NO evidence ledger: credit is passage count,
        # which is exactly the quantity EC refuses to use
        credited = m.sum(-1)
        ceiling = batch["root_mask"].sum(-1).clamp(min=1e-6)
        return {"logits": logits, "raw_logits": logits, "credited_evidence": credited,
                "ceiling": ceiling, "claim_ratio": credited / ceiling,
                "evidence_logits": ev, "scale": torch.ones(B, device=pas.device),
                "state": state}


def build(name: str, emb_dim: int, hidden: int, **kw):
    if name in AGGREGATORS:
        return AggregationBaseline(emb_dim=emb_dim, hidden=hidden, mode=name)
    if name in SET_TRANSFORMER_VARIANTS:
        return SetTransformerRag(emb_dim=emb_dim, hidden=hidden, variant=name)
    return ECRag(emb_dim=emb_dim, hidden=hidden, variant=name, **kw)
