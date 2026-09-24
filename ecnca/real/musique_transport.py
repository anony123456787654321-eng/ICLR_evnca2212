"""Matched message-transport model for the preregistered MuSiQue audit."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .musique import MuSiQueRecord


VARIANTS = ("full", "filter_only", "no_provenance", "plain")


def make_transport_batch(records: list[MuSiQueRecord], cache, device="cpu"):
    batch_size = len(records)
    steps = max(r.n_hops for r in records)
    dim = cache.dim
    questions = torch.zeros(batch_size, steps, dim, device=device)
    paragraphs = torch.zeros_like(questions)
    answers = torch.zeros_like(questions)
    mask = torch.zeros(batch_size, steps, device=device)
    for row, record in enumerate(records):
        n = record.n_hops
        questions[row, :n] = torch.as_tensor(
            cache.encode([s.question for s in record.steps]), device=device)
        paragraphs[row, :n] = torch.as_tensor(
            cache.encode([s.paragraph_title + ". " + s.paragraph_text
                          for s in record.steps]), device=device)
        answers[row, :n] = torch.as_tensor(
            cache.encode([s.answer for s in record.steps]), device=device)
        mask[row, :n] = 1.0
    return {"question": questions, "paragraph": paragraphs, "answer": answers,
            "mask": mask, "n_hops": mask.sum(-1).long(),
            "record_ids": [r.record_id for r in records]}


class MuSiQueTransport(nn.Module):
    """One root whose payload may improve while its evidence credit stays one."""

    def __init__(self, emb_dim=768, hidden=128, variant="full"):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(variant)
        self.variant = variant
        self.start = nn.Parameter(torch.zeros(hidden))
        self.question = nn.Sequential(nn.Linear(emb_dim, hidden), nn.SiLU(),
                                      nn.Linear(hidden, hidden))
        self.paragraph = nn.Sequential(nn.Linear(emb_dim, hidden), nn.SiLU(),
                                       nn.Linear(hidden, hidden))
        self.transition = nn.GRUCell(2 * hidden, hidden)
        self.answer = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(),
                                    nn.Linear(hidden, emb_dim))

    def forward(self, batch, collect=False):
        question, paragraph, mask = (batch["question"], batch["paragraph"],
                                     batch["mask"])
        batch_size, n_steps, _ = question.shape
        state = self.start.unsqueeze(0).expand(batch_size, -1)
        first = None
        versions, predictions = [], []
        for step in range(n_steps):
            local = torch.cat([self.question(question[:, step]),
                               self.paragraph(paragraph[:, step])], -1)
            incoming = first if self.variant == "filter_only" and first is not None else state
            proposal = self.transition(local, incoming)
            active = mask[:, step:step + 1]
            proposal = active * proposal + (1 - active) * incoming
            if first is None:
                first = proposal
            if self.variant != "filter_only":
                state = proposal
            versions.append(proposal)
            predictions.append(F.normalize(self.answer(proposal), dim=-1))
        terminal = first if self.variant == "filter_only" else state
        terminal_prediction = F.normalize(self.answer(terminal), dim=-1)
        local_prediction = torch.stack(predictions, 1)
        has_evidence = (mask.sum(-1) > 0).float()
        credited = has_evidence if self.variant in ("full", "filter_only") \
            else mask.sum(-1)
        out = {"prediction": terminal_prediction,
               "local_prediction": local_prediction,
               "credited_evidence": credited, "ceiling": has_evidence,
               "claim_ratio": credited / has_evidence.clamp(min=1.0)}
        if collect:
            out["versions"] = torch.stack(versions, 1)
        return out

    @staticmethod
    def ledger_readout(versions: torch.Tensor, variant: str,
                       delivered_versions: list[int]):
        """Resolve re-delivery/cycles without invoking another transformation."""
        if not delivered_versions:
            raise ValueError("empty delivery")
        selected = min(delivered_versions) if variant == "filter_only" \
            else max(delivered_versions)
        payload = versions[:, selected]
        credit = 1.0 if variant in ("full", "filter_only") \
            else float(len(delivered_versions))
        return payload, credit


def cosine_loss(output, batch, local_weight=0.25):
    mask = batch["mask"]
    local = 1 - (output["local_prediction"] * batch["answer"]).sum(-1)
    local = (local * mask).sum() / mask.sum().clamp(min=1)
    final_index = batch["n_hops"] - 1
    target = batch["answer"][torch.arange(len(final_index), device=mask.device),
                             final_index]
    terminal = (1 - (output["prediction"] * target).sum(-1)).mean()
    return terminal + local_weight * local, {"terminal": terminal, "local": local}


@torch.no_grad()
def retrieval_metrics(prediction, target, candidate, target_index):
    scores = prediction @ candidate.T
    order = scores.argsort(-1, descending=True)
    ranks = (order == target_index.unsqueeze(1)).nonzero()[:, 1] + 1
    return {"top1": float((ranks == 1).float().mean()),
            "mrr": float((1.0 / ranks.float()).mean()),
            "cosine_error": float((1 - (prediction * target).sum(-1)).mean())}
