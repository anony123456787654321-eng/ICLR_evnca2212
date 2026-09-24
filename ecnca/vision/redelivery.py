"""E3: separating new computation from exact message redelivery.

The Phase 1 test compared a rollout against a *reset-RNG* re-run of the same
updates. That is not redelivery: resetting the generator and stepping again
still recomputes perception from the current grid, so the cell receives genuinely
new neighbourhood information. It measured extra computation, not duplication.

A valid redelivery arm must hold the computation budget fixed and hand a cell a
message it has already integrated, carrying no new information. The
perception/message factorisation in ``ReferenceCA`` makes that expressible:

    m_t = model.message(x_t)          # neighbourhood information at step t
    x_{t+1} = model.apply_message(x_t, m_t, ...)

Two arms over the same number of updates T:

* ``fresh``     -- m recomputed every step: T steps of new information.
* ``redeliver`` -- m captured at step ``cache_at`` and re-applied for the
  following ``repeat`` steps. Same T updates, same fire/noise schedule, but
  those steps convey **no** new neighbourhood information.

The distinction the plan requires:
  (a) legitimate new computation or transformed information -> ``fresh``
  (b) exact repeated delivery containing no new information -> ``redeliver``

Ground-truth equality of input support does **not** require intermediate neural
states to stay identical, so divergence between the arms is expected and is not
by itself a defect. What E3 measures is whether a variant's *predictions* are
degraded by duplicate delivery.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from .stochastic import UpdateSchedule


@dataclass
class RedeliveryResult:
    arm: str
    steps: int
    cache_at: int | None
    repeat: int
    cell_accuracy: list[float] = field(default_factory=list)
    digit_accuracy: list[float] = field(default_factory=list)
    final_state: torch.Tensor | None = None
    messages_recomputed: int = 0

    def summary(self) -> dict:
        return {
            "arm": self.arm,
            "steps": self.steps,
            "cache_at": self.cache_at,
            "repeat": self.repeat,
            "messages_recomputed": self.messages_recomputed,
            "cell_accuracy_final": self.cell_accuracy[-1] if self.cell_accuracy else None,
            "digit_accuracy_final": self.digit_accuracy[-1] if self.digit_accuracy else None,
            "cell_accuracy_mean": float(np.mean(self.cell_accuracy)) if self.cell_accuracy else None,
        }


def _metrics(model, x, alive, labels) -> tuple[float, float]:
    logits = model.classify(x)
    a2 = alive[:, 0]
    na = a2.flatten(1).sum(1)
    pred = logits.argmax(1)
    ok = ((pred == labels[:, None, None]) & a2).flatten(1).sum(1)
    nonempty = na > 0
    frac = torch.where(nonempty, ok.float() / na.clamp(min=1).float(),
                       torch.zeros_like(ok, dtype=torch.float32))
    onehot = torch.zeros_like(logits).scatter_(1, pred.unsqueeze(1), 1.0)
    votes = (onehot * alive.to(logits.dtype)).flatten(2).sum(2)
    digit_ok = (votes.argmax(1) == labels) & nonempty
    n = int(nonempty.sum())
    return (
        float(frac[nonempty].sum()) / max(n, 1),
        float(digit_ok.sum()) / max(n, 1),
    )


@torch.no_grad()
def run_arm(
    model,
    images: torch.Tensor,
    labels: torch.Tensor,
    schedule: UpdateSchedule,
    *,
    arm: str,
    cache_at: int | None = None,
    repeat: int = 0,
) -> RedeliveryResult:
    """Run one E3 arm at a FIXED update budget of ``schedule.steps``."""
    if arm not in ("fresh", "redeliver"):
        raise ValueError(f"unknown arm {arm!r}")
    if arm == "redeliver":
        if cache_at is None or repeat <= 0:
            raise ValueError("redeliver requires cache_at and repeat > 0")
        if cache_at + repeat > schedule.steps:
            raise ValueError(
                f"cache_at({cache_at}) + repeat({repeat}) exceeds budget "
                f"{schedule.steps}"
            )

    model.eval()
    device = next(model.parameters()).device
    images, labels = images.to(device), labels.to(device)
    sched = schedule.to(device)

    x = model.initialize(images)
    alive = model.living_mask(x)
    res = RedeliveryResult(arm=arm, steps=sched.steps, cache_at=cache_at, repeat=repeat)

    cached: torch.Tensor | None = None
    for t in range(sched.steps):
        fire, noise = sched.step(t)
        replaying = (
            arm == "redeliver" and cache_at is not None
            and cache_at <= t < cache_at + repeat
        )
        if replaying:
            if cached is None:
                # First replayed step: capture the message the cell would have
                # received here, then keep re-delivering exactly that one.
                cached = model.message(x)
                res.messages_recomputed += 1
            m = cached
        else:
            m = model.message(x)
            res.messages_recomputed += 1
        x = model.apply_message(x, m, fire=fire, noise=noise)
        c, d = _metrics(model, x, alive, labels)
        res.cell_accuracy.append(c)
        res.digit_accuracy.append(d)

    res.final_state = x
    return res


@torch.no_grad()
def compare(
    model,
    images: torch.Tensor,
    labels: torch.Tensor,
    schedule: UpdateSchedule,
    *,
    cache_at: int,
    repeat: int,
) -> dict:
    """Paired E3 comparison at an identical update budget and schedule."""
    fresh = run_arm(model, images, labels, schedule, arm="fresh")
    redel = run_arm(
        model, images, labels, schedule, arm="redeliver",
        cache_at=cache_at, repeat=repeat,
    )
    drift = float(
        (fresh.final_state[:, 1:] - redel.final_state[:, 1:]).abs().mean()
    )
    return {
        "budget_steps": schedule.steps,
        "schedule_seed": schedule.seed,
        "cache_at": cache_at,
        "repeat": repeat,
        "fresh": fresh.summary(),
        "redeliver": redel.summary(),
        "state_drift_mean_abs": drift,
        "cell_accuracy_delta": (
            redel.cell_accuracy[-1] - fresh.cell_accuracy[-1]
        ),
        "digit_accuracy_delta": (
            redel.digit_accuracy[-1] - fresh.digit_accuracy[-1]
        ),
        "messages_saved": fresh.messages_recomputed - redel.messages_recomputed,
        "note": (
            "Both arms take the same number of updates with the same fire/noise "
            "schedule. The redeliver arm replaces `repeat` steps of new "
            "neighbourhood information with exact re-delivery of one cached "
            "message, so any difference is attributable to duplicate delivery "
            "rather than to extra or reduced computation."
        ),
    }
