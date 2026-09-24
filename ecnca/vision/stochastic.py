"""Deterministic, device-correct stochastic sources for CA rollouts.

Why this exists. ``torch.rand(..., device="cuda", generator=g)`` raises unless
``g`` is a CUDA generator, and a CPU generator silently cannot drive a CUDA
tensor. Phase 1 passed a CPU ``torch.Generator`` into evaluation on a CUDA
device; the call fell back to the global RNG, so repeated evaluation of the
same checkpoint was **not** reproducible. Two 500-image evaluations of the same
100k checkpoint differed (8->0 rate 0.0000 vs 0.0062), which is how the defect
surfaced.

The fix has two parts:

1. ``UpdateSchedule`` precomputes every fire mask and noise tensor for a whole
   rollout from a single integer seed. Replaying a rollout replays the exact
   same stochastic decisions, on any device, bit-identically.
2. Because the schedule is an explicit object, **matched variants can be given
   the identical schedule**. Paired comparisons then differ only by the model,
   not by which cells happened to fire.

A schedule is generated on the CPU and moved to the target device, so the same
seed yields the same masks on CPU, MPS and CUDA. That costs one host-to-device
copy per rollout and buys cross-device reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


def seeded_generator(seed: int) -> torch.Generator:
    """A CPU generator. Always CPU: see the module docstring."""
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    return g


@dataclass
class UpdateSchedule:
    """Precomputed per-step fire masks and residual noise for one rollout.

    ``fire[t]`` has shape (B,1,H,W) and ``noise[t]`` shape (B,C,H,W), where C is
    the number of mutable channels. Index ``t`` is the update index, 0-based.
    """

    fire: torch.Tensor    # (T,B,1,H,W) bool
    noise: torch.Tensor | None  # (T,B,C,H,W) float, or None when noise is off
    seed: int
    fire_rate: float

    @property
    def steps(self) -> int:
        return int(self.fire.shape[0])

    def to(self, device: torch.device | str) -> "UpdateSchedule":
        return UpdateSchedule(
            fire=self.fire.to(device),
            noise=None if self.noise is None else self.noise.to(device),
            seed=self.seed,
            fire_rate=self.fire_rate,
        )

    def step(self, t: int) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self.fire[t], (None if self.noise is None else self.noise[t])

    def slice_batch(self, sl: slice) -> "UpdateSchedule":
        """Restrict to a sub-batch, keeping the same stochastic decisions."""
        return UpdateSchedule(
            fire=self.fire[:, sl],
            noise=None if self.noise is None else self.noise[:, sl],
            seed=self.seed,
            fire_rate=self.fire_rate,
        )


def make_schedule(
    *,
    steps: int,
    batch: int,
    height: int = 28,
    width: int = 28,
    channels: int,
    fire_rate: float,
    seed: int,
    add_noise: bool,
    noise_std: float,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> UpdateSchedule:
    """Build a reproducible schedule from one integer seed.

    Generated on the CPU then moved, so a given seed produces identical masks
    on every device.
    """
    g = seeded_generator(seed)
    fire = (
        torch.rand(steps, batch, 1, height, width, generator=g, dtype=dtype)
        <= fire_rate
    )
    noise = None
    if add_noise:
        noise = torch.empty(steps, batch, channels, height, width, dtype=dtype)
        noise.normal_(0.0, noise_std, generator=g)
    sched = UpdateSchedule(fire=fire, noise=noise, seed=int(seed), fire_rate=float(fire_rate))
    return sched.to(device) if str(device) != "cpu" else sched


# --- per-example schedules ----------------------------------------------
#
# Why seed+batch_offset is not enough. A schedule keyed by `seed + s` (the
# batch's start index) gives example i a different stochastic stream depending
# on which batch it lands in, so changing --eval-batch changes every example's
# fire mask. The Phase 1 confusion matrix happened to be batch-size invariant
# only because each batch's schedule was regenerated from a start-index seed
# and the examples stayed in the same order; any re-batching that moves an
# example across a boundary breaks it.
#
# The fix is to key the stream on a STABLE EXAMPLE ID rather than a position.
# Example i then receives the same fire mask and noise wherever it sits in a
# batch, so batch size becomes a performance knob instead of a protocol change.
#
# Caveat, stated because it bounds what the tests can assert: identical random
# inputs do NOT imply bit-identical model outputs across devices or batch
# shapes. cuDNN/MPS kernel selection and reduction order vary with tensor
# shape, so float arithmetic differs in the last bits. Batch-size invariance is
# therefore asserted for the *stochastic inputs* exactly, and for *predictions
# and metrics* within a declared tolerance.

_ID_SALT = 0x9E3779B97F4A7C15  # golden-ratio odd constant, for stream mixing


def example_seed(example_id: int, *, stage: str, eval_seed: int) -> int:
    """A stable 63-bit stream seed for (example, rollout stage, eval seed).

    ``stage`` names the rollout phase ("pre", "post", "redeliver", ...) so the
    same example gets independent streams before and after a mutation.
    """
    h = (int(eval_seed) & 0xFFFFFFFFFFFFFFFF)
    h ^= (int(example_id) + 1) * _ID_SALT
    h &= 0xFFFFFFFFFFFFFFFF
    for ch in stage.encode():
        h = (h * 0x100000001B3) ^ ch
        h &= 0xFFFFFFFFFFFFFFFF
    # splitmix64 finalizer, so nearby ids give well-separated streams
    h ^= h >> 30
    h = (h * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    h ^= h >> 27
    h = (h * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    h ^= h >> 31
    return h & 0x7FFFFFFFFFFFFFFF


def schedule_for_ids(
    example_ids,
    *,
    steps: int,
    channels: int,
    fire_rate: float,
    eval_seed: int,
    stage: str,
    add_noise: bool,
    noise_std: float,
    height: int = 28,
    width: int = 28,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
    noise_channels: int | None = None,
) -> UpdateSchedule:
    """Build a schedule whose per-example streams depend only on example IDs.

    ``fire`` is generated per example and is therefore **identical across
    variants** regardless of how many channels each variant carries: the fire
    mask is (1,H,W) per cell, not per channel.

    ``noise_channels`` supports variants with extra state. Noise for the first
    ``channels`` channels is drawn from the same stream as the reference, so a
    wider variant's *shared* channels receive bit-identical noise; the extra
    channels draw from a separate, later stream. A variant therefore cannot
    gain or lose by perturbing the reference's noise sequence.
    """
    ids = [int(i) for i in example_ids]
    n = len(ids)
    total_ch = channels if noise_channels is None else int(noise_channels)
    if total_ch < channels:
        raise ValueError(
            f"noise_channels ({total_ch}) must be >= channels ({channels})"
        )

    fire = torch.empty(steps, n, 1, height, width, dtype=torch.bool)
    noise = (
        torch.empty(steps, n, total_ch, height, width, dtype=dtype)
        if add_noise else None
    )
    for k, ex in enumerate(ids):
        g = seeded_generator(example_seed(ex, stage=stage, eval_seed=eval_seed))
        fire[:, k, 0] = (
            torch.rand(steps, height, width, generator=g, dtype=dtype) <= fire_rate
        )
        if noise is not None:
            # Shared channels first, from the reference's stream position.
            shared = torch.empty(steps, channels, height, width, dtype=dtype)
            shared.normal_(0.0, noise_std, generator=g)
            noise[:, k, :channels] = shared
            if total_ch > channels:
                extra = torch.empty(
                    steps, total_ch - channels, height, width, dtype=dtype
                )
                extra.normal_(0.0, noise_std, generator=g)
                noise[:, k, channels:] = extra

    sched = UpdateSchedule(
        fire=fire, noise=noise, seed=int(eval_seed), fire_rate=float(fire_rate)
    )
    sched.example_ids = ids  # type: ignore[attr-defined]
    sched.stage = stage      # type: ignore[attr-defined]
    return sched.to(device) if str(device) != "cpu" else sched


# --- reference-identical schedules ---------------------------------------
#
# THE BUG THIS FIXES. A variant carrying auxiliary channels draws a bigger
# noise tensor from the same generator, which advances the RNG stream further,
# so the FIRE MASK drawn next differs and different cells update. Measured: the
# reference draws 59,584 noise values and `origin` 175,616, and the first fire
# value differs (0.4076 vs 0.5644) -- the variants diverged from the frozen
# reference at step 0 under stochastic schedules even with every mechanism gate
# exactly closed.
#
# The reference's own randomness must therefore be drawn FIRST and identically
# for every variant, with auxiliary randomness taken from a separate stream.

REFERENCE_MUTABLE_CHANNELS = 19


def reference_identical_schedule(
    example_ids,
    *,
    steps: int,
    total_channels: int,
    fire_rate: float,
    eval_seed: int,
    stage: str,
    add_noise: bool,
    noise_std: float,
    height: int = 28,
    width: int = 28,
    reference_channels: int = REFERENCE_MUTABLE_CHANNELS,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> UpdateSchedule:
    """A schedule whose reference portion is identical for every variant.

    The fire mask and the first ``reference_channels`` of noise are drawn from
    the reference stream, in the reference's own order, so a variant with any
    number of auxiliary channels receives exactly what the frozen model would.
    Auxiliary channels draw from an independent stream keyed on the same
    example, so they are reproducible without perturbing the reference.
    """
    ids = [int(i) for i in example_ids]
    n = len(ids)
    if total_channels < reference_channels:
        raise ValueError(
            f"total_channels ({total_channels}) < reference_channels "
            f"({reference_channels})"
        )

    fire = torch.empty(steps, n, 1, height, width, dtype=torch.bool)
    noise = (torch.empty(steps, n, total_channels, height, width, dtype=dtype)
             if add_noise else None)

    for k, ex in enumerate(ids):
        # Reference stream: fire mask and the reference's own channels, in the
        # order the frozen model draws them.
        g = seeded_generator(example_seed(ex, stage=stage, eval_seed=eval_seed))
        fire[:, k, 0] = (
            torch.rand(steps, height, width, generator=g, dtype=dtype)
            <= fire_rate
        )
        if noise is not None:
            ref_noise = torch.empty(steps, reference_channels, height, width,
                                    dtype=dtype)
            ref_noise.normal_(0.0, noise_std, generator=g)
            noise[:, k, :reference_channels] = ref_noise
            extra = total_channels - reference_channels
            if extra > 0:
                # Independent stream: cannot perturb the reference portion.
                g2 = seeded_generator(
                    example_seed(ex, stage=f"{stage}-aux", eval_seed=eval_seed))
                aux = torch.empty(steps, extra, height, width, dtype=dtype)
                aux.normal_(0.0, noise_std, generator=g2)
                noise[:, k, reference_channels:] = aux

    sched = UpdateSchedule(fire=fire, noise=noise, seed=int(eval_seed),
                           fire_rate=float(fire_rate))
    sched.example_ids = ids          # type: ignore[attr-defined]
    sched.stage = stage              # type: ignore[attr-defined]
    sched.reference_channels = reference_channels  # type: ignore[attr-defined]
    return sched.to(device) if str(device) != "cpu" else sched
