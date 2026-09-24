"""Probabilistic readout for MuSiQue transport, fitted on clean data only.

The transport task is embedding retrieval, so it has no native probability. This
module defines one, and defines it *before* any duplicated-hop result is
inspected, so that evidence conservation cannot appear successful merely because
confidence was defined tautologically from the ledger.

Two readouts are reported side by side.

`content` uses candidate cosine similarities alone:
    p = softmax(cos(prediction, candidates) / T)
`evidence` additionally lets credited evidence sharpen the distribution:
    p = softmax(cos(prediction, candidates) * (a + b * credit_ratio) / T)
with b >= 0 enforced, so the dependence on evidence is positive and monotone.

Every parameter is fitted by minimising NLL on CLEAN validation examples and
then frozen. The duplicated-hop evaluation never refits them. The two readouts
share one functional form across variants, so a difference between variants is a
difference in what they credit, not in how they are scored.

PREREGISTERED: ECE uses 15 equal-width bins on [0, 1] over the max probability.
This binning is fixed here and is not adjusted after seeing results.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn.functional as F

ECE_BINS = 15
READOUTS = ("content", "evidence")


@dataclass
class Calibration:
    """Frozen readout parameters. `b` is stored pre-softplus."""
    log_temperature: float = 0.0
    a: float = 1.0
    raw_b: float = -3.0

    @property
    def temperature(self) -> float:
        return float(np.exp(self.log_temperature))

    @property
    def b(self) -> float:
        return float(np.log1p(np.exp(self.raw_b)))

    def as_dict(self) -> dict:
        d = asdict(self)
        d.update(temperature=self.temperature, b=self.b)
        return d


def logits(similarity: torch.Tensor, credit_ratio: torch.Tensor,
           cal: Calibration, readout: str) -> torch.Tensor:
    """Scores over the candidate pool. `similarity` is [N, C], credit is [N]."""
    if readout not in READOUTS:
        raise ValueError(readout)
    scale = torch.as_tensor(1.0 / cal.temperature, dtype=similarity.dtype,
                            device=similarity.device)
    if readout == "content":
        return similarity * scale
    gain = cal.a + cal.b * credit_ratio.unsqueeze(-1).to(similarity.dtype)
    return similarity * gain * scale


def fit(similarity: torch.Tensor, target_index: torch.Tensor,
        credit_ratio: torch.Tensor, readout: str,
        iters: int = 400, lr: float = 0.05) -> Calibration:
    """Fit the readout by NLL on clean validation data."""
    log_t = torch.zeros(1, requires_grad=True)
    a = torch.ones(1, requires_grad=True)
    raw_b = torch.full((1,), -3.0, requires_grad=True)
    params = [log_t] if readout == "content" else [log_t, a, raw_b]
    opt = torch.optim.Adam(params, lr=lr)
    sim = similarity.detach().float()
    credit = credit_ratio.detach().float()
    for _ in range(iters):
        opt.zero_grad(set_to_none=True)
        scale = torch.exp(-log_t)
        if readout == "content":
            z = sim * scale
        else:
            # softplus keeps the evidence coefficient non-negative, so evidence
            # can only sharpen, never blunt.
            z = sim * (a + F.softplus(raw_b) * credit.unsqueeze(-1)) * scale
        loss = F.cross_entropy(z, target_index)
        loss.backward()
        opt.step()
    return Calibration(log_temperature=float(log_t.detach()),
                       a=float(a.detach()), raw_b=float(raw_b.detach()))


def expected_calibration_error(confidence: np.ndarray, correct: np.ndarray,
                               bins: int = ECE_BINS) -> float:
    """Equal-width binning on [0, 1]; the rule is preregistered above."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    total, n = 0.0, len(confidence)
    if n == 0:
        return float("nan")
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (confidence > lo) & (confidence <= hi) if lo > 0 else \
            (confidence >= lo) & (confidence <= hi)
        if not m.any():
            continue
        total += m.sum() / n * abs(correct[m].mean() - confidence[m].mean())
    return float(total)


@torch.no_grad()
def score(similarity: torch.Tensor, target_index: torch.Tensor,
          credit_ratio: torch.Tensor, cal: Calibration, readout: str) -> dict:
    """Every probabilistic metric the paper reports, from one readout."""
    z = logits(similarity, credit_ratio, cal, readout)
    p = z.softmax(-1)
    n, n_candidates = p.shape
    rows = torch.arange(n, device=p.device)
    p_true = p[rows, target_index].clamp_min(1e-12)
    top = p.argmax(-1)
    correct = (top == target_index).float()
    conf = p.max(-1).values
    onehot = F.one_hot(target_index, n_candidates).to(p.dtype)
    order = z.argsort(-1, descending=True)
    ranks = (order == target_index.unsqueeze(1)).float().argmax(-1) + 1
    entropy = -(p.clamp_min(1e-12).log() * p).sum(-1)
    return {"nll": float(-p_true.log().mean()),
            "brier": float(((p - onehot) ** 2).sum(-1).mean()),
            "ece": expected_calibration_error(conf.cpu().numpy(),
                                              correct.cpu().numpy()),
            "accuracy": float(correct.mean()),
            "mean_max_confidence": float(conf.mean()),
            "predictive_entropy": float(entropy.mean()),
            "mrr": float((1.0 / ranks.float()).mean()),
            "n": int(n), "n_candidates": int(n_candidates)}
