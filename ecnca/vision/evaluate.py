"""Common evaluator for E1-E5, shared by every variant.

Protocol decisions recorded here (see `results/mnist_restart/PROTOCOL.md`):

* **Per-image averaging is primary.** For each image we average over its own
  live cells, then average across images. Cells within an image are not
  independent trials, so the source's batch-pooled figure would weight large
  digits more heavily. The pooled statistic is still reported as
  ``pooled_cell_accuracy`` for comparability with the article.
* **Mutation pairing is deterministic and model-independent.** Pairs come from
  ``mutation_pairs`` seeded by ``PAIR_SEED``; no model output participates in
  pair selection, so no filter can select examples a model gets wrong.
* **Alive cells are defined by the immutable grey channel** at the *current*
  input, which after mutation is the new digit's mask.
* **Empty images** (no cell above threshold) contribute no cells; they are
  counted in ``empty_images`` and excluded from per-image means because the
  mean over zero cells is undefined. They are not silently dropped.
* **All-digit evaluation is primary.** The 8/0 slice is computed
  unconditionally from the same run, never selected after inspecting errors.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from .reference import EVAL_STEPS, LIVING_THRESHOLD, OUTPUT_CHANNELS, mutate
from .stochastic import (
    UpdateSchedule, make_schedule, reference_identical_schedule,
    schedule_for_ids,
)

PAIR_SEED = 20260911


# --- pairing -------------------------------------------------------------
def mutation_pairs(
    labels: np.ndarray, *, seed: int = PAIR_SEED, require_change: bool = True
) -> np.ndarray:
    """Deterministic (source_index, target_index) pairs over a whole split.

    Every image is used exactly once as a source. The target is a fixed random
    permutation, repaired so that no pair has the same digit on both sides
    (a "mutation" to the same class would not test adaptation). Repair swaps
    targets between offending positions, so the mapping stays a permutation and
    every image is also used exactly once as a target.
    """
    n = int(labels.shape[0])
    rng = np.random.default_rng(seed)
    target = rng.permutation(n)
    if require_change:
        for _ in range(64):
            bad = np.flatnonzero(labels[target] == labels)
            if bad.size == 0:
                break
            # Rotate the offending positions' targets among themselves; with
            # >1 class present this converges in a few passes.
            roll = rng.permutation(bad.size)
            target[bad] = target[bad][roll]
        bad = np.flatnonzero(labels[target] == labels)
        if bad.size:
            # Deterministic fallback: swap each remaining offender with any
            # position whose digits differ both ways.
            for i in bad:
                for j in range(n):
                    if labels[target[j]] != labels[i] and labels[target[i]] != labels[j]:
                        target[i], target[j] = target[j], target[i]
                        break
        bad = np.flatnonzero(labels[target] == labels)
        if bad.size:
            raise RuntimeError(f"{bad.size} same-digit pairs could not be repaired")
    return np.stack([np.arange(n), target], axis=1)


# --- metrics -------------------------------------------------------------
@dataclass
class StepStats:
    """Per-step accumulators, summed over batches then finalized."""
    per_image_sum: float = 0.0
    per_image_count: int = 0
    pooled_correct: float = 0.0
    pooled_total: float = 0.0
    agreement_sum: float = 0.0
    agreement_count: int = 0
    digit_correct: float = 0.0
    digit_count: int = 0

    def merge(self, other: "StepStats") -> None:
        self.per_image_sum += other.per_image_sum
        self.per_image_count += other.per_image_count
        self.pooled_correct += other.pooled_correct
        self.pooled_total += other.pooled_total
        self.agreement_sum += other.agreement_sum
        self.agreement_count += other.agreement_count
        self.digit_correct += other.digit_correct
        self.digit_count += other.digit_count

    def finalize(self) -> dict:
        return {
            "cell_accuracy": self.per_image_sum / max(self.per_image_count, 1),
            "pooled_cell_accuracy": self.pooled_correct / max(self.pooled_total, 1.0),
            "total_agreement": self.agreement_sum / max(self.agreement_count, 1),
            "digit_accuracy": self.digit_correct / max(self.digit_count, 1),
        }


def _step_metrics(
    logits: torch.Tensor, alive: torch.Tensor, labels: torch.Tensor
) -> StepStats:
    """logits (B,10,H,W); alive (B,1,H,W) bool; labels (B,)."""
    alive2 = alive[:, 0]
    n_alive = alive2.flatten(1).sum(1)
    pred = logits.argmax(1)
    correct = (pred == labels[:, None, None]) & alive2

    acc = torch.float32 if logits.device.type == "mps" else torch.float64
    per_img = correct.flatten(1).sum(1).to(acc)
    nonempty = n_alive > 0
    frac = torch.zeros_like(per_img)
    frac[nonempty] = per_img[nonempty] / n_alive[nonempty].to(acc)

    # Majority (digit-level) readout over alive cells only.
    onehot = torch.zeros_like(logits).scatter_(1, pred.unsqueeze(1), 1.0)
    votes = (onehot * alive.to(logits.dtype)).flatten(2).sum(2)  # (B,10)
    digit_pred = votes.argmax(1)
    digit_ok = (digit_pred == labels) & nonempty

    # Total agreement: every alive cell predicts the same class.
    top = votes.max(1).values
    agree = (top == n_alive.to(votes.dtype)) & nonempty

    return StepStats(
        per_image_sum=float(frac[nonempty].sum()),
        per_image_count=int(nonempty.sum()),
        pooled_correct=float(per_img.sum()),
        pooled_total=float(n_alive.sum()),
        agreement_sum=float(agree.sum()),
        agreement_count=int(nonempty.sum()),
        digit_correct=float(digit_ok.sum()),
        digit_count=int(nonempty.sum()),
    )


@dataclass
class RolloutResult:
    pre: list[dict] = field(default_factory=list)
    post: list[dict] = field(default_factory=list)
    empty_images: int = 0
    n_images: int = 0
    per_image_post_mean: np.ndarray | None = None  # (N,) primary-score inputs
    per_image_pre_final: np.ndarray | None = None
    post_recovery_step: np.ndarray | None = None

    def primary_score(self) -> float:
        """E2 primary: mean over pairs of the mean correct-alive-cell fraction
        over post-mutation updates 1..200."""
        if self.per_image_post_mean is None:
            raise RuntimeError("no post-mutation trajectory recorded")
        return float(np.mean(self.per_image_post_mean))


@torch.no_grad()
def rollout(
    model,
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    steps: int = EVAL_STEPS,
    mutate_to: tuple[torch.Tensor, torch.Tensor] | None = None,
    generator: torch.Generator | None = None,
    seed: int | None = None,
    example_ids=None,
    noise_channels: int | None = None,
    pre_schedule: UpdateSchedule | None = None,
    post_schedule: UpdateSchedule | None = None,
    threshold: float = LIVING_THRESHOLD,
    recovery_level: float = 0.9,
    recovery_hold: int = 20,
    track_per_image: bool = True,
) -> RolloutResult:
    """Run `steps` updates, optionally mutate, then run `steps` more.

    Determinism. Pass ``seed`` (or explicit schedules) to make the rollout
    reproducible and to give matched variants identical update masks. A
    ``torch.Generator`` is NOT sufficient on CUDA: a CPU generator cannot drive
    a CUDA tensor, so ``generator`` was silently ignored there, which made
    Phase 1 evaluation irreproducible on GPU. ``seed`` builds an
    ``UpdateSchedule`` on the CPU and moves it, so the same seed gives the same
    masks on CPU, MPS and CUDA.

    Returns per-step aggregates plus the per-image post-mutation means that the
    primary score and the paired statistics are computed from.
    """
    model.eval()
    device = next(model.parameters()).device
    images = images.to(device)
    labels = labels.to(device)

    n = int(images.shape[0])
    if seed is not None:
        # `total_channels`, not `channels`: the reference-identical builder
        # draws the reference's own fire mask and noise FIRST and takes
        # auxiliary channels from a separate stream, so a variant's extra
        # channels cannot perturb the reference's randomness.
        common = dict(
            steps=steps, total_channels=noise_channels or model.channel_n,
            fire_rate=model.config.fire_rate,
            add_noise=model.config.add_noise,
            noise_std=model.config.noise_std,
        )
        if example_ids is None:
            # Positional fallback. Prefer example_ids: a position-keyed stream
            # changes when --eval-batch changes, which changes the protocol.
            example_ids = list(range(n))
        if pre_schedule is None:
            pre_schedule = reference_identical_schedule(
                example_ids, eval_seed=seed, stage="pre", **common
            )
        if post_schedule is None and mutate_to is not None:
            post_schedule = reference_identical_schedule(
                example_ids, eval_seed=seed, stage="post", **common
            )
    if pre_schedule is not None:
        pre_schedule = pre_schedule.to(device)
        if pre_schedule.steps < steps:
            raise ValueError(
                f"pre_schedule has {pre_schedule.steps} steps, need {steps}"
            )
    if post_schedule is not None:
        post_schedule = post_schedule.to(device)

    x = model.initialize(images)
    alive = model.living_mask(x)
    res = RolloutResult(n_images=n)
    res.empty_images = int((alive.flatten(1).sum(1) == 0).sum())

    # MPS has no float64. Accumulate in the widest dtype the device supports
    # and widen to float64 only once the values reach the host.
    acc = torch.float32 if device.type == "mps" else torch.float64

    for t in range(steps):
        if pre_schedule is not None:
            fire, noise = pre_schedule.step(t)
            x = model(x, fire=fire, noise=noise)
        else:
            x = model(x, generator=generator)
        st = _step_metrics(model.classify(x), alive, labels)
        res.pre.append(st.finalize())
    logits = model.classify(x)
    a2 = alive[:, 0]
    na = a2.flatten(1).sum(1)
    ok = ((logits.argmax(1) == labels[:, None, None]) & a2).flatten(1).sum(1)
    pre_frac = torch.where(
        na > 0, ok.to(acc) / na.clamp(min=1).to(acc), torch.zeros(n, dtype=acc, device=device)
    )
    res.per_image_pre_final = pre_frac.cpu().numpy().astype(np.float64)

    if mutate_to is None:
        return res

    new_images, new_labels = mutate_to
    new_images = new_images.to(device)
    new_labels = new_labels.to(device)
    x = mutate(x, new_images, threshold=threshold)
    alive = model.living_mask(x)
    labels = new_labels

    running = torch.zeros(n, dtype=acc, device=device)
    recovery = torch.full((n,), -1, dtype=torch.long, device=device)
    hold = torch.zeros(n, dtype=torch.long, device=device)
    a2 = alive[:, 0]
    na = a2.flatten(1).sum(1)
    nonempty = na > 0

    for step in range(steps):
        if post_schedule is not None:
            fire, noise = post_schedule.step(step)
            x = model(x, fire=fire, noise=noise)
        else:
            x = model(x, generator=generator)
        logits = model.classify(x)
        st = _step_metrics(logits, alive, labels)
        res.post.append(st.finalize())
        if track_per_image:
            ok = ((logits.argmax(1) == labels[:, None, None]) & a2).flatten(1).sum(1)
            frac = torch.where(
                nonempty,
                ok.to(acc) / na.clamp(min=1).to(acc),
                torch.zeros(n, dtype=acc, device=device),
            )
            running += frac
            # Recovery: first step at >= level held for `hold` consecutive steps.
            above = frac >= recovery_level
            hold = torch.where(above, hold + 1, torch.zeros_like(hold))
            newly = (recovery < 0) & (hold >= recovery_hold)
            recovery = torch.where(
                newly,
                torch.full_like(recovery, step + 1 - recovery_hold + 1),
                recovery,
            )

    if track_per_image:
        res.per_image_post_mean = (running / steps).cpu().numpy().astype(np.float64)
        # -1 marks censored (never recovered); callers must not average only
        # successful recoveries.
        res.post_recovery_step = recovery.cpu().numpy()
    return res


def confusion_matrix(
    model, images: torch.Tensor, labels: torch.Tensor, *, steps: int = EVAL_STEPS,
    generator: torch.Generator | None = None, batch_size: int = 250,
    seed: int | None = None, example_ids=None,
) -> np.ndarray:
    """Digit-level confusion over a split after `steps` updates (E1 diagnostic).

    Pass ``seed`` for a reproducible, device-correct result. Each batch uses a
    schedule derived from ``seed`` and the batch's start index, so the matrix
    does not depend on ``batch_size``.
    """
    device = next(model.parameters()).device
    cm = np.zeros((10, 10), dtype=np.int64)
    model.eval()
    with torch.no_grad():
        for s in range(0, images.shape[0], batch_size):
            xb = images[s:s + batch_size].to(device)
            yb = labels[s:s + batch_size].to(device)
            x = model.initialize(xb)
            alive = model.living_mask(x)
            sched = None
            if seed is not None:
                ids = (
                    list(range(s, s + int(xb.shape[0])))
                    if example_ids is None
                    else [int(v) for v in example_ids[s:s + int(xb.shape[0])]]
                )
                sched = reference_identical_schedule(
                    ids, steps=steps, total_channels=model.channel_n,
                    fire_rate=model.config.fire_rate, eval_seed=seed,
                    stage="clean", add_noise=model.config.add_noise,
                    noise_std=model.config.noise_std, device=device,
                )
            for t in range(steps):
                if sched is not None:
                    fire, noise = sched.step(t)
                    x = model(x, fire=fire, noise=noise)
                else:
                    x = model(x, generator=generator)
            logits = model.classify(x)
            pred = logits.argmax(1)
            onehot = torch.zeros_like(logits).scatter_(1, pred.unsqueeze(1), 1.0)
            votes = (onehot * alive.to(logits.dtype)).flatten(2).sum(2)
            dp = votes.argmax(1).cpu().numpy()
            for t, p in zip(yb.cpu().numpy(), dp):
                cm[t, p] += 1
    return cm


def connected_components(image: np.ndarray, threshold: float = LIVING_THRESHOLD) -> int:
    """Count 8-connected live components (diagnostic, not a filter)."""
    alive = image > threshold
    seen = np.zeros_like(alive, dtype=bool)
    count = 0
    h, w = alive.shape
    for i in range(h):
        for j in range(w):
            if alive[i, j] and not seen[i, j]:
                count += 1
                stack = [(i, j)]
                seen[i, j] = True
                while stack:
                    a, b = stack.pop()
                    for da in (-1, 0, 1):
                        for db in (-1, 0, 1):
                            na_, nb = a + da, b + db
                            if 0 <= na_ < h and 0 <= nb < w and alive[na_, nb] and not seen[na_, nb]:
                                seen[na_, nb] = True
                                stack.append((na_, nb))
    return count
