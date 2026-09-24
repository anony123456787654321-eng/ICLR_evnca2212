"""Order-aware Transformer transport baseline for the MuSiQue audit.

This exists to answer one objection: that evidence conservation is a trick that
only works for a recurrent cell. The content path here is a Transformer encoder
with causal attention over the ordered hops; the ledger is bolted on unchanged.

Parameter matching. The recurrent model spends 345,600 parameters on the shared
question/paragraph/answer projections and 148,224 on its GRUCell, totalling
493,824. Matching that budget leaves 148,224 for the sequence mixer, which at
d_model=128 admits exactly one encoder layer (dim_feedforward=320, a standard
2.5x ratio). Two layers would force dim_feedforward below d_model, which is
degenerate. Depth one is therefore a consequence of fair matching, not a choice,
and it is reported as such.

Only `credited_evidence` depends on the variant. `transformer_ec` and
`transformer_plain` run a byte-identical content path, so any difference in task
metrics between them would be a bug; tests assert they are identical.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

VARIANTS = ("transformer_ec", "transformer_filter_only", "transformer_plain")
MAX_HOPS = 4


class TransformerTransport(nn.Module):
    """Ordered Transformer transport with an optional lineage ledger."""

    def __init__(self, emb_dim: int = 768, hidden: int = 128,
                 variant: str = "transformer_ec", n_heads: int = 4,
                 ff_dim: int = 320, n_layers: int = 1, max_hops: int = MAX_HOPS):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(variant)
        self.variant = variant
        self.hidden = hidden
        # Identical projections to the recurrent model, so the controlled
        # difference between the two backbones is the sequence mixer alone.
        self.start = nn.Parameter(torch.zeros(hidden))
        self.question = nn.Sequential(nn.Linear(emb_dim, hidden), nn.SiLU(),
                                      nn.Linear(hidden, hidden))
        self.paragraph = nn.Sequential(nn.Linear(emb_dim, hidden), nn.SiLU(),
                                       nn.Linear(hidden, hidden))
        self.position = nn.Parameter(torch.zeros(max_hops, hidden))
        nn.init.normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=n_heads, dim_feedforward=ff_dim,
            dropout=0.0, activation="gelu", batch_first=True,
            norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers,
                                             enable_nested_tensor=False)
        self.answer = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(),
                                    nn.Linear(hidden, emb_dim))

    @staticmethod
    def _causal_mask(n: int, device) -> torch.Tensor:
        """A step may attend to itself and to earlier steps only.

        Boolean, to match the boolean padding mask; mixing a float mask with a
        boolean one is deprecated and silently reinterprets one of them.
        """
        return torch.triu(torch.ones(n, n, dtype=torch.bool, device=device), 1)

    def forward(self, batch, collect: bool = False):
        question, paragraph, mask = (batch["question"], batch["paragraph"],
                                     batch["mask"])
        batch_size, n_steps, _ = question.shape
        device = question.device

        # Additive fusion of the two projections keeps d_model at `hidden`
        # without spending parameters on a reduction, so the budget goes to the
        # sequence mixer that is actually under test.
        #
        # Positions are indexed by HOP, not by arrival order. A duplicated
        # arrival of hop j reuses hop j's embedding, so a stream longer than the
        # chain needs no new positional parameters and no retraining.
        hop_index = batch.get("hop_index")
        if hop_index is None:
            position = self.position[:n_steps].unsqueeze(0)
        else:
            position = self.position[hop_index.clamp(max=self.position.shape[0] - 1)]
        tokens = (self.question(question) + self.paragraph(paragraph)
                  + position + self.start.view(1, 1, -1))
        # Padding is masked as well as future steps, so a short chain cannot
        # read a padded slot and a long chain cannot read ahead.
        states = self.encoder(tokens,
                              mask=self._causal_mask(n_steps, device),
                              src_key_padding_mask=(mask <= 0))
        states = torch.nan_to_num(states) * mask.unsqueeze(-1)

        index = (batch["n_hops"] - 1).clamp(min=0)
        rows = torch.arange(batch_size, device=device)
        # `n_hops` counts arrivals, so this is the last delivered event.
        terminal = states[rows, 0] if self.variant == "transformer_filter_only" \
            else states[rows, index]
        terminal_prediction = F.normalize(self.answer(terminal), dim=-1)
        local_prediction = F.normalize(self.answer(states), dim=-1)

        has_evidence = (mask.sum(-1) > 0).float()
        credited = has_evidence if self.variant in (
            "transformer_ec", "transformer_filter_only") else mask.sum(-1)
        out = {"prediction": terminal_prediction,
               "local_prediction": local_prediction,
               "credited_evidence": credited, "ceiling": has_evidence,
               "claim_ratio": credited / has_evidence.clamp(min=1.0)}
        if collect:
            out["versions"] = states
        return out

    @staticmethod
    def ledger_readout(versions: torch.Tensor, variant: str,
                       delivered_versions: list[int]):
        """Resolve re-delivery and cycles without another transformation."""
        if not delivered_versions:
            raise ValueError("empty delivery")
        selected = min(delivered_versions) \
            if variant == "transformer_filter_only" else max(delivered_versions)
        payload = versions[:, selected]
        credit = 1.0 if variant in ("transformer_ec",
                                    "transformer_filter_only") \
            else float(len(delivered_versions))
        return payload, credit


def build(variant: str, emb_dim: int = 768, hidden: int = 128, **kw):
    return TransformerTransport(emb_dim=emb_dim, hidden=hidden,
                                variant=variant, **kw)
