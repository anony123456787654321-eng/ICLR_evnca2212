"""Baseline credit gates with the SAME content/credit separation.

The learned mechanism must beat these, specifically in the harmful non-exact
region (transform strengths 0.0005-0.01), where a byte-different repeat still
carries 85-99% of the exact-repeat harm. Beating a whole-message damping
baseline is not sufficient, and beating deterministic equality only on exact
repeats proves nothing: equality already solves that case perfectly and for
free.

Every gate here scales ONLY the evidential channels, exactly as the learned
mechanism does, so the comparison isolates the decision rule rather than the
content/credit split.

  EqualityGate     byte-identical to a stored signature -> no credit.
                   Free, exact, and blind to any perturbation at all.
  CosineGate       cosine above a threshold -> no credit. The threshold is
                   fitted on the CALIBRATION split only, never on development
                   or test. Its difficulty is visible in the measurement: a
                   strength-0.001 repeat sits at cosine 0.99999982 while
                   genuinely new evidence sits near 0.45, so the threshold has
                   to be extreme to catch the harmful case.
  LastSignatureGate learned, but compares only the immediately previous
                   signature. Cannot recognise A -> B -> A.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class GateResult:
    credit: torch.Tensor        # (B,1,H,W) in [0,1]
    matched: torch.Tensor       # (B,1,H,W) bool: recognised as already credited
    detail: dict


class EqualityGate(nn.Module):
    """Deterministic byte equality. No parameters, no training."""

    def __init__(self, tolerance: float = 0.0):
        super().__init__()
        self.tolerance = tolerance

    def forward(self, signature: torch.Tensor,
                memory: torch.Tensor) -> GateResult:
        """memory is (B,M,D,H,W) of stored signatures."""
        diff = (memory - signature.unsqueeze(1)).abs().amax(dim=2)  # (B,M,H,W)
        hit = (diff <= self.tolerance).any(dim=1, keepdim=True)
        return GateResult(
            credit=(~hit).to(signature.dtype),
            matched=hit,
            detail={"rule": "byte-equality", "tolerance": self.tolerance},
        )


class CosineGate(nn.Module):
    """Cosine-similarity threshold, fitted on calibration data only."""

    def __init__(self, threshold: float = 0.999):
        super().__init__()
        self.register_buffer("threshold", torch.tensor(float(threshold)))

    @torch.no_grad()
    def fit(self, same_origin: torch.Tensor, new_origin: torch.Tensor) -> float:
        """Choose the threshold maximising separation on CALIBRATION pairs.

        Takes cosines already computed on calibration data. The midpoint
        between the new-origin maximum and the same-origin minimum is used when
        the two are separable; otherwise the equal-error point.
        """
        lo, hi = float(new_origin.max()), float(same_origin.min())
        self.threshold.fill_((lo + hi) / 2 if hi > lo else
                             float(torch.cat([same_origin, new_origin]).median()))
        return float(self.threshold)

    def forward(self, signature: torch.Tensor,
                memory: torch.Tensor) -> GateResult:
        cos = F.cosine_similarity(memory, signature.unsqueeze(1), dim=2)
        hit = (cos >= self.threshold).any(dim=1, keepdim=True)
        return GateResult(
            credit=(~hit).to(signature.dtype),
            matched=hit,
            detail={"rule": "cosine-threshold",
                    "threshold": float(self.threshold),
                    "fitted_on": "calibration split only"},
        )


class LastSignatureGate(nn.Module):
    """Learned, but sees only the previous signature.

    Included to isolate what bounded multi-origin memory buys: this gate cannot
    recognise A -> B -> A, because B overwrote A.
    """

    def __init__(self, signature_dim: int, hidden: int = 32):
        super().__init__()
        d = signature_dim
        self.net = nn.Sequential(
            nn.Conv2d(4 * d + 2, hidden, 1), nn.ReLU(),
            nn.Conv2d(hidden, 1, 1),
        )

    def forward(self, signature: torch.Tensor,
                memory: torch.Tensor) -> GateResult:
        prev = memory[:, -1]                     # only the most recent slot
        diff = signature - prev
        mag = torch.linalg.vector_norm(diff, dim=1, keepdim=True)
        cos = F.cosine_similarity(signature, prev, dim=1).unsqueeze(1)
        feats = torch.cat([signature, prev, diff, signature * prev, mag, cos], 1)
        credit = torch.sigmoid(self.net(feats))
        return GateResult(
            credit=credit, matched=credit < 0.5,
            detail={"rule": "learned-last-signature",
                    "limitation": "cannot recognise A -> B -> A"},
        )
