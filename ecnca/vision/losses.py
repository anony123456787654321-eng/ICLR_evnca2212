"""Losses used by the deterministic vision trainer."""

import torch
import torch.nn.functional as F


def spatial_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean class loss over pixels, using the non-spatial NLL kernel.

    Some CUDA/PyTorch combinations reject spatial NLL under strict
    determinism. Flattening pixels into examples preserves the objective and
    avoids that spatial kernel. Classes remain the final dimension.
    """
    if logits.ndim != 4 or targets.shape != (logits.shape[0], *logits.shape[2:]):
        raise ValueError("expected logits [B,C,H,W] and targets [B,H,W]")
    rows = logits.permute(0, 2, 3, 1).reshape(-1, logits.shape[1])
    return F.cross_entropy(rows, targets.reshape(-1))
