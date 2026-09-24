"""Frozen endpoint evaluations for Phase C Targets 1 and 2.

Every variant is measured on IDENTICAL frozen examples with IDENTICAL fire and
noise schedules, keyed on the split's stable example indices, so a difference
between variants is attributable to the model and not to which cells fired.

Training loss is never used to rank variants: `reference`, `wider` and `decay`
optimise a task term alone, `sectors` and `both` add a mixture term, and the
origin-bearing variants add provenance and conservation. Those totals are not
comparable, and a lower one does not mean a better task fit.

Target 1 (dynamic sectors), primary CELL accuracy:
    complete / progressive / progressive_reset / ambiguity pairs /
    geometry-scaled procedural drawings.

Target 2 (evidence-origin memory), primary CELL accuracy:
    fresh / withhold / state_hold / exact repeat /
    recursive near-repeats at 0.0005, 0.001, 0.005, 0.01 /
    transformed same-origin / genuinely new SAME-IMAGE evidence.

The last of those was missing: "new evidence" had been another image's message,
which can carry a different class and so measured "wrong label injected" rather
than "new evidence arrived". ``same_image_reveal`` restores a previously
occluded region of the SAME digit instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from .interventions import transform_message

# Frozen: declared before any variant was trained.
NEAR_REPEAT_STRENGTHS = (0.0005, 0.001, 0.005, 0.01)
GEOMETRY_SCALES = (0.5, 1.0, 1.5)
CONSENSUS_LEVEL = 0.90
CONFIDENT_LEVEL = 0.95


@dataclass
class CellMetrics:
    """Per-image measurements. Cells within an image are not independent."""
    cell_accuracy: np.ndarray
    digit_correct: np.ndarray
    agreement: np.ndarray
    confidently_wrong: np.ndarray
    consensus_step: np.ndarray          # -1 == never reached, censored
    credited_evidence: np.ndarray | None = None
    sector_occupancy: np.ndarray | None = None
    sector_churn: np.ndarray | None = None

    def summary(self) -> dict:
        reached = self.consensus_step[self.consensus_step >= 0]
        out = {
            "cell_accuracy": float(self.cell_accuracy.mean()),
            "cell_accuracy_sem": float(
                self.cell_accuracy.std(ddof=1) / np.sqrt(self.cell_accuracy.size)
            ) if self.cell_accuracy.size > 1 else None,
            "digit_accuracy": float(self.digit_correct.mean()),
            "agreement": float(self.agreement.mean()),
            "disagreement": float(1.0 - self.agreement.mean()),
            "confidently_wrong": float(self.confidently_wrong.mean()),
            "consensus_censored_fraction": float(
                (self.consensus_step < 0).mean()
            ),
            "consensus_step_median_among_reached": (
                float(np.median(reached)) if reached.size else None
            ),
            "n_images": int(self.cell_accuracy.size),
        }
        for name, arr in (("credited_evidence", self.credited_evidence),
                          ("sector_occupancy", self.sector_occupancy),
                          ("sector_churn", self.sector_churn)):
            if arr is not None:
                out[name] = float(arr.mean())
        return out


def _step_scores(model, x, labels, alive):
    a2 = alive[:, 0]
    na = a2.flatten(1).sum(1).clamp(min=1)
    logits = model.classify(x)
    cp = logits.argmax(1)
    ok = ((cp == labels[:, None, None]) & a2).flatten(1).sum(1)
    cell = ok.float() / na.float()
    votes = torch.zeros(x.shape[0], logits.shape[1], device=x.device)
    for c in range(logits.shape[1]):
        votes[:, c] = ((cp == c) & a2).flatten(1).sum(1).to(votes.dtype)
    top = votes.max(1).values
    agree = top / na.to(top.dtype)
    pred = votes.argmax(1)
    return cell, pred, agree


@torch.no_grad()
def rollout_metrics(model, x, labels, alive, schedule, steps, *,
                    message_fn=None, hold_window=None) -> CellMetrics:
    """Run `steps` updates, collecting per-image metrics and consensus timing.

    ``message_fn(t, x, model, cached) -> (message_or_None, cached)`` lets an
    intervention control what a cell receives at step ``t``. Returning None
    means "no update this step" (state hold).
    """
    n = x.shape[0]
    dev = x.device
    reached = torch.full((n,), -1, dtype=torch.long, device=dev)
    hold = torch.zeros(n, dtype=torch.long, device=dev)
    cached = None
    for t in range(steps):
        fire = schedule.fire[t]
        noise = None if schedule.noise is None else schedule.noise[t]
        if message_fn is None:
            x = model(x, fire=fire, noise=noise)
        else:
            msg, cached = message_fn(t, x, model, cached)
            if msg is not None:
                x = model.apply_message(x, msg, fire=fire, noise=noise)
        cell, pred, agree = _step_scores(model, x, labels, alive)
        above = agree >= CONSENSUS_LEVEL
        hold = torch.where(above, hold + 1, torch.zeros_like(hold))
        newly = (reached < 0) & (hold >= 20)
        reached = torch.where(newly, torch.full_like(reached, t + 1), reached)

    cell, pred, agree = _step_scores(model, x, labels, alive)
    correct = pred == labels
    credited = None
    if getattr(model, "origin_memory", None) is not None:
        _, _, credit = model._unpack_origin(x)
        credited = credit.flatten(1).mean(1).cpu().numpy()
    occ = churn = None
    diag = getattr(model, "last_diagnostics", None) or {}
    if "active_sectors" in diag:
        occ = np.full(n, float(diag["active_sectors"]))
        churn = np.full(n, float(diag.get("switching", 0.0)))

    return CellMetrics(
        cell_accuracy=cell.cpu().numpy(),
        digit_correct=correct.float().cpu().numpy(),
        agreement=agree.cpu().numpy(),
        confidently_wrong=((~correct) & (agree >= CONFIDENT_LEVEL))
            .float().cpu().numpy(),
        consensus_step=reached.cpu().numpy(),
        credited_evidence=credited,
        sector_occupancy=occ,
        sector_churn=churn,
    )


# --- Target 2 interventions ---------------------------------------------
def make_message_fn(arm: str, *, boundary: int, window: int,
                    strength: float = 0.0, reveal_state=None):
    """One intervention, as a message function over a matched schedule."""

    def fn(t, x, model, cached):
        inside = boundary <= t < boundary + window
        if not inside:
            return model.message(x), cached

        if arm == "fresh":
            return model.message(x), cached
        if arm == "withhold":
            base = cached if cached is not None else model.message(x)
            return torch.zeros_like(base), base
        if arm == "state_hold":
            return None, cached
        if arm == "exact_repeat":
            if cached is None:
                cached = model.message(x)
            return cached, cached
        if arm == "near_repeat":
            # RECURSIVE: each delivery re-transforms the previous one, so the
            # drift compounds over the window as it would under repeated
            # legitimate propagation. Not a single small perturbation.
            if cached is None:
                cached = model.message(x)
            else:
                cached = transform_message(cached, strength=strength)
            return cached, cached
        if arm == "transformed_same_origin":
            if cached is None:
                cached = model.message(x)
            return transform_message(cached, strength=0.05), cached
        if arm == "same_image_reveal":
            # Genuinely NEW evidence about the SAME image and class: a
            # previously occluded region is restored into the receiving state
            # at the boundary, then ordinary fresh messages continue.
            if cached is None and reveal_state is not None:
                cached = True
                return None, cached          # signal: adopt below
            return model.message(x), cached
        raise ValueError(f"unknown arm {arm!r}")

    return fn


@torch.no_grad()
def same_image_reveal(model, occluded_images, full_images, labels, alive,
                      schedule, *, boundary, steps):
    """New evidence about the SAME image: restore an occluded region.

    Delivering a message perceived elsewhere is not enough -- the cells it
    describes are still dead in the receiver, so their residual is masked away.
    The restored pixels must enter the RECEIVING state.
    """
    dev = occluded_images.device
    x = model.initialize(occluded_images)
    for t in range(boundary):
        x = model(x, fire=schedule.fire[t],
                  noise=None if schedule.noise is None else schedule.noise[t])
    # Adopt the additional observation into the receiver itself.
    full = full_images if full_images.dim() == 4 else full_images.unsqueeze(1)
    x = torch.cat([full, x[:, 1:]], dim=1)

    class _Tail:
        fire = schedule.fire[boundary:]
        noise = None if schedule.noise is None else schedule.noise[boundary:]

    return rollout_metrics(model, x, labels, alive, _Tail, steps - boundary)
