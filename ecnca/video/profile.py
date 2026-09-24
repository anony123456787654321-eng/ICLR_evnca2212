"""Measure the ACTUAL training step, and refuse to guess.

The MNIST campaign profiled a workload the training never performed: a plain
28x28 static classifier stood in for a 56x56 recurrent drawing history, and
the concurrency it authorised then exhausted the device. The same mistake here
would be worse -- video frames are two orders of magnitude larger.

So this profiles the real thing: the real backbone, the real feature grid, the
real losses, forward AND backward, the longest allowed training segment, the
intended bank capacity, and the intended precision.

A missing memory reading is reported as MISSING. It never becomes "100%
headroom" -- on the last run a device that could not report memory produced a
100% headroom verdict, which is exactly backwards.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class MemoryReading:
    allocated_mib: float | None = None
    reserved_mib: float | None = None
    device_free_mib: float | None = None
    device_total_mib: float | None = None
    measurable: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def read_memory(device) -> MemoryReading:
    import torch

    if getattr(device, "type", str(device)) != "cuda":
        return MemoryReading(measurable=False)
    try:
        free, total = torch.cuda.mem_get_info()
    except Exception:
        return MemoryReading(
            allocated_mib=round(torch.cuda.memory_allocated() / 2**20, 1),
            reserved_mib=round(torch.cuda.memory_reserved() / 2**20, 1),
            measurable=False,
        )
    return MemoryReading(
        allocated_mib=round(torch.cuda.memory_allocated() / 2**20, 1),
        reserved_mib=round(torch.cuda.memory_reserved() / 2**20, 1),
        device_free_mib=round(free / 2**20, 1),
        device_total_mib=round(total / 2**20, 1),
        measurable=True,
    )


def headroom(peak: MemoryReading) -> dict:
    """Headroom, or an explicit refusal to state one.

    Zero or missing measurements must not read as abundant memory.
    """
    if not peak.measurable or not peak.device_total_mib:
        return {
            "headroom_frac": None,
            "measurable": False,
            "verdict": (
                "MEMORY NOT MEASURABLE. No headroom is claimed and "
                "concurrency stays at one process. A missing reading is not "
                "evidence of free memory."
            ),
        }
    used = peak.device_total_mib - (peak.device_free_mib or 0.0)
    if used <= 0:
        return {
            "headroom_frac": None,
            "measurable": False,
            "verdict": (
                "device reported zero usage while training was resident, "
                "which cannot be right; treating memory as UNMEASURED"
            ),
        }
    frac = 1.0 - used / peak.device_total_mib
    return {"headroom_frac": round(frac, 4), "measurable": True,
            "used_mib": round(used, 1),
            "verdict": f"{frac:.1%} headroom measured"}


def estimate_campaign(seconds_per_iteration: float, *, iterations: int,
                      arms: int, cap_hours: float) -> dict:
    """A measured compute estimate, enforced against the frozen cap."""
    total_h = seconds_per_iteration * iterations * arms / 3600.0
    return {
        "seconds_per_iteration": round(seconds_per_iteration, 4),
        "iterations_per_arm": iterations,
        "trainable_arms": arms,
        "estimated_training_hours": round(total_h, 2),
        "cap_hours": cap_hours,
        "within_cap": total_h <= cap_hours,
        "max_iterations_within_cap": int(
            cap_hours * 3600.0 / max(seconds_per_iteration * arms, 1e-9)),
        "note": (
            "training GPU time only. Download, caching, audit, profiling and "
            "evaluation are reported separately."
        ),
    }


class Budget:
    """Enforces the cap across the whole campaign.

    Reaching it saves an incomplete result and says so. Silently shortening
    training and marking it complete is the failure this exists to prevent.
    """

    def __init__(self, cap_hours: float):
        self.cap_seconds = cap_hours * 3600.0
        self.spent = 0.0
        self._t0: float | None = None

    def start(self) -> None:
        self._t0 = time.perf_counter()

    def tick(self) -> float:
        if self._t0 is None:
            return self.spent
        now = time.perf_counter()
        self.spent += now - self._t0
        self._t0 = now
        return self.spent

    @property
    def exhausted(self) -> bool:
        return self.tick() >= self.cap_seconds

    def report(self) -> dict:
        return {
            "training_seconds_spent": round(self.tick(), 1),
            "cap_seconds": self.cap_seconds,
            "exhausted": self.exhausted,
            "status": ("INCOMPLETE -- the training cap was reached; this is "
                       "a partial result and is reported as such"
                       if self.exhausted else "within budget"),
        }
