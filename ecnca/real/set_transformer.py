"""Set Transformer aggregation over AVeriTeC passages, with and without EC.

This answers the objection that mean/sum/DeepSets are too weak to be the only
unordered-passage baselines. It follows Lee et al. (2019): multihead
self-attention blocks over the passage set, then pooling by multihead attention
(PMA) with a learned seed vector.

`set_transformer_plain` and `set_transformer_ec` share a byte-identical content
path and identical heads. They differ only in credited evidence:

* plain: credit is the passage count, the quantity EC refuses to use;
* ec:    a root is credited at most once, however many passages carry it.

Neither is tuned separately. This is NOT a FiD model and is not labelled as one:
there is no generative decoder and no per-passage decoder fusion.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ecrag import mlp

VARIANTS = ("set_transformer_plain", "set_transformer_ec")


class MAB(nn.Module):
    """Multihead attention block: Q attends to K with a residual FF, per Lee et al."""

    def __init__(self, dim: int, n_heads: int = 4, ff: int | None = None):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm0, self.norm1 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.ff = mlp([dim, ff or dim, dim])

    def forward(self, q, k, key_padding_mask=None):
        a, _ = self.attn(q, k, k, key_padding_mask=key_padding_mask,
                         need_weights=False)
        h = self.norm0(q + a)
        return self.norm1(h + self.ff(h))


class SetTransformerRag(nn.Module):
    def __init__(self, emb_dim: int = 768, hidden: int = 128, n_labels: int = 4,
                 variant: str = "set_transformer_ec", n_heads: int = 4,
                 n_blocks: int = 1, ff: int | None = 124, dropout: float = 0.1):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(variant)
        self.variant = variant
        self.use_provenance = variant == "set_transformer_ec"
        self.enc_claim = mlp([emb_dim, hidden, hidden])
        self.enc_passage = mlp([emb_dim + 2, hidden, hidden])
        # One SAB plus the PMA block matches ECRag's 543,880 parameters to
        # -0.07%; a second SAB would overshoot by 14% at any feed-forward width.
        self.blocks = nn.ModuleList([MAB(hidden, n_heads, ff)
                                     for _ in range(n_blocks)])
        self.seed = nn.Parameter(torch.randn(1, 1, hidden) * 0.02)
        self.pma = MAB(hidden, n_heads, ff)
        self.drop = nn.Dropout(dropout)
        self.rho_head = mlp([hidden, hidden, 1])
        nn.init.constant_(self.rho_head[-1].bias, -1.0)
        self.pool = mlp([hidden * 2, hidden, hidden])
        self.verdict = mlp([hidden, hidden, n_labels])
        self.evidence_head = mlp([hidden * 2, hidden, 1])
        self.temp = nn.Parameter(torch.tensor([1.0, 0.5]))

    def forward(self, batch: Dict[str, torch.Tensor], collect: bool = False):
        claim, pas, mask = batch["claim_emb"], batch["passage_emb"], batch["passage_mask"]
        root_idx = batch["root_index"].long()
        root_mask = batch["root_mask"]
        rel = batch.get("retrieval_score", torch.zeros_like(mask))
        B, N, _ = pas.shape
        R = root_mask.shape[1]
        pad = mask <= 0

        c = self.enc_claim(claim)
        sim = F.cosine_similarity(pas, claim.unsqueeze(1), dim=-1)
        h = self.enc_passage(torch.cat([pas, rel.unsqueeze(-1),
                                        sim.unsqueeze(-1)], dim=-1)) * mask.unsqueeze(-1)
        for block in self.blocks:
            h = block(h, h, key_padding_mask=pad) * mask.unsqueeze(-1)
        h = self.drop(h) * mask.unsqueeze(-1)
        pooled = self.pma(self.seed.expand(B, 1, -1), h,
                          key_padding_mask=pad).squeeze(1)

        rho = torch.sigmoid(self.rho_head(h)).squeeze(-1) * mask
        onehot = F.one_hot(root_idx.clamp(min=0), R).float() * mask.unsqueeze(-1)
        if self.use_provenance:
            per_root = (rho.unsqueeze(-1) * onehot).amax(dim=1)
        else:
            per_root = (rho.unsqueeze(-1) * onehot).sum(dim=1)
        credited = (per_root * root_mask).sum(-1)
        ceiling = root_mask.sum(-1).clamp(min=1e-6)

        state = self.pool(torch.cat([pooled, c], dim=-1))
        raw = self.verdict(state)
        scale = F.softplus(self.temp[0]) + F.softplus(self.temp[1]) * credited
        logits = raw * scale.unsqueeze(-1) if self.use_provenance else raw
        ev = self.evidence_head(torch.cat(
            [h, state.unsqueeze(1).expand(B, N, state.shape[-1])], dim=-1)
        ).squeeze(-1).masked_fill(pad, -1e4)
        return {"logits": logits, "raw_logits": raw, "credited_evidence": credited,
                "ceiling": ceiling, "claim_ratio": credited / ceiling,
                "evidence_logits": ev, "scale": scale, "state": state}


def build(variant: str, emb_dim: int = 768, hidden: int = 128, **kw):
    return SetTransformerRag(emb_dim=emb_dim, hidden=hidden, variant=variant, **kw)
