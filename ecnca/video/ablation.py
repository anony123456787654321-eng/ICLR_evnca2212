"""Does the task actually exercise memory?

Before comparing methods, establish that historical information matters on the
predefined event population. If XMem scores the same with long-term memory
disabled, then either the task does not need it, the ablation did not take
effect, or the evaluation is underpowered -- and those three have different
consequences. Reporting a method gain on a task that does not need memory
would be meaningless.

A configuration flag is NOT evidence that memory was disabled. These helpers
instrument the actual memory READS, so "disabled" means observed reads changed.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np


CONFIGURATIONS = (
    "normal",             # published configuration
    "no_long_term_read",  # long-term excluded from READOUT; bounding kept
    "recent_only",        # controlled recent-memory-only
    "disrupted_content",  # memory content disrupted where technically valid
)


# ReadTrace and reads_actually_changed lived here and read counters that do
# not exist on XMem (`long_mem_reads`, `work_mem_reads`,
# `distinct_entries_touched`) off an attribute that does not exist either
# (`network.memory`). getattr returned a fabricated zero for every
# configuration, so the comparison was 0 against 0 and every ablation was
# declared ineffective regardless of what it did. Real instrumentation now
# lives in ecnca/video/instrument.py, which wraps the live
# InferenceCore.memory.match_memory and reports an UNMEASURED state rather
# than a zero when the interface is unreachable.

# UNITS. j_and_f returns FRACTIONS in [0,1]; every threshold here is in J&F
# POINTS (0.02 -> 2 points). Mixing the two made a real 2-point drop compare
# as 0.02 against a 1.0 threshold and read as no degradation at all.
def as_points(delta_fraction: float) -> float:
    """Convert a J&F difference expressed as a fraction into points."""
    return float(delta_fraction) * 100.0


def interpret_null(delta_jf_points: float, *, n_events: int,
                   ci_halfwidth_points: float, ablation_effective: bool,
                   min_detectable: float = 3.0) -> dict:
    """Distinguish the three reasons a memory ablation might show nothing.

    Collapsing them would let "our task does not need long-term memory" be
    reported as "video memory is unnecessary", which the evidence cannot
    support.
    """
    if not ablation_effective:
        kind = "ineffective_ablation"
        text = ("The ablation did not change memory reads, so this says "
                "nothing about whether the task needs memory.")
    elif ci_halfwidth_points > min_detectable:
        kind = "underpowered"
        text = (f"The interval (+/-{ci_halfwidth_points:.2f} J&F points) is "
                f"wider than the {min_detectable:.1f}-point effect we would "
                f"care about over {n_events} events, so a null is not "
                f"evidence of absence.")
    elif abs(delta_jf_points) < 1.0:
        kind = "recent_memory_sufficient"
        text = ("Removing long-term memory cost less than 1 J&F point with a "
                "tight interval: on THIS event population recent memory "
                "suffices. This is a statement about this population, not "
                "about video memory in general.")
    else:
        kind = "memory_matters"
        text = (f"Removing memory cost {delta_jf_points:.2f} J&F points, so "
                f"the event population does exercise historical information.")
    return {"interpretation": kind, "explanation": text,
            "delta_jf_points": delta_jf_points, "n_events": n_events,
            "ci_halfwidth_points": ci_halfwidth_points,
            "units": "J&F points (a 0.80 -> 0.78 change is 2.00 points)"}


def paired_difference(normal: dict, ablated: dict, *, boots: int = 5000,
                      seed: int = 0) -> dict:
    """Paired per-video difference between two configurations, in POINTS.

    The question is whether removing memory HURTS, and every video is scored
    under both configurations, so the videos pair exactly. Pairing cancels the
    between-video variation that otherwise dominates.

    An earlier version computed the interval from the spread of the NORMAL
    scores alone. On three DAVIS events that gave +/-19.15 points -- driven by
    bmx-bumps scoring 0.70 against dog-gooses 0.43, a difference between
    videos that says nothing about the ablation. The paired interval on the
    same data is +/-6.79.
    """
    keys = sorted(set(normal) & set(ablated))
    if not keys:
        return {"n_videos": 0, "mean_difference_points": None,
                "ci95_points": None, "ci_halfwidth_points": float("inf"),
                "note": "no video was scored under both configurations"}
    d = np.array([float(normal[k]) - float(ablated[k]) for k in keys]) * 100.0
    mean = float(d.mean())
    if len(d) < 2:
        return {"n_videos": len(d), "mean_difference_points": mean,
                "ci95_points": None, "ci_halfwidth_points": float("inf"),
                "per_video_points": {k: float(v) for k, v in zip(keys, d)},
                "note": ("a single paired video supports no interval; the "
                         "result is not evidence either way")}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(boots, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {
        "n_videos": len(d),
        "mean_difference_points": mean,
        "ci95_points": [float(lo), float(hi)],
        "ci_halfwidth_points": float(max(hi - mean, mean - lo)),
        "excludes_zero": bool(lo > 0 or hi < 0),
        "per_video_points": {k: float(v) for k, v in zip(keys, d)},
        "unit": "video (paired); positive means the ablation HURT",
    }


def gate(per_config: dict, *, min_degradation_points: float = 1.0) -> dict:
    """The engineering gate: does the task exercise memory at all?

    Deliberately NOT a per-video filter. Requiring every video to degrade, or
    keeping only those that do, would select the population on the outcome --
    exactly what the event extractor is built to prevent.
    """
    normal = per_config.get("normal", {}).get("mean_jf")
    # The corrected control excludes long-term from the READOUT while keeping
    # bounded management; the old "no_long_term" name meant enable_long_term=
    # False, which also removed the bound.
    ablated = per_config.get("no_long_term_read", {}).get("mean_jf")
    if normal is None or ablated is None:
        return {"passed": False, "reason": "a required configuration is missing"}
    # mean_jf values are FRACTIONS; the threshold is in POINTS.
    delta_points = as_points(float(normal) - float(ablated))
    return {
        "passed": delta_points >= min_degradation_points,
        "delta_jf_points": delta_points,
        "min_degradation_points": min_degradation_points,
        "units": "J&F points",
        "reason": (
            f"removing long-term memory cost {delta_points:.2f} J&F points"
            if delta_points >= min_degradation_points else
            f"removing long-term memory cost only {delta_points:.2f} J&F "
            f"points; the event population may not exercise it (see "
            f"interpret_null)"
        ),
    }
