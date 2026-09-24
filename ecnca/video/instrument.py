"""Observe what XMem's memory ACTUALLY does, on the real object.

The previous attempt read counters off `xmem.network.memory`. That attribute
does not exist, and neither do the counters: `getattr(..., 0)` returned
fabricated zeros for every configuration, so `reads_actually_changed` compared
0 against 0, concluded the ablation was ineffective, and would have reported
an uninterpretable null no matter what the ablation really did.

The memory lives on `InferenceCore.memory` (inference_core.py:27), and the
read happens in `MemoryManager.match_memory`. This wraps that method on the
live object and records what each call actually touched.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass
class MemoryObservation:
    """What the memory did, measured -- never defaulted."""

    configuration: str
    calls: int = 0
    long_term_engaged_calls: int = 0
    long_term_elements: int = 0        # max observed
    working_elements: int = 0          # max observed
    readout_checksum: float = 0.0      # sum of readout means, per call
    available: bool = False            # was the memory object reachable?
    detail: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class MemoryProbe:
    """Wraps `InferenceCore.memory.match_memory` for the duration of a run.

    Reports `available=False` rather than zeros when the interface has moved,
    so a missing measurement can never be mistaken for a measured zero.
    """

    def __init__(self, processor, configuration: str):
        self.processor = processor
        self.obs = MemoryObservation(configuration=configuration)
        self._original = None

    def __enter__(self) -> "MemoryProbe":
        mem = getattr(self.processor, "memory", None)
        original = getattr(mem, "match_memory", None)
        if mem is None or original is None:
            self.obs.available = False
            self.obs.detail = (
                "InferenceCore.memory.match_memory was not reachable; NO read "
                "statistics were collected. This is an unmeasured state, not "
                "a measurement of zero."
            )
            return self
        self._original = original
        self.obs.available = True

        def probed(query_key, selection):
            out = original(query_key, selection)
            o = self.obs
            o.calls += 1
            lt = getattr(mem, "long_mem", None)
            if lt is not None and getattr(mem, "enable_long_term", False):
                try:
                    if lt.engaged():
                        o.long_term_engaged_calls += 1
                        o.long_term_elements = max(o.long_term_elements,
                                                   int(lt.size))
                except Exception:
                    pass
            wm = getattr(mem, "work_mem", None)
            if wm is not None:
                try:
                    o.working_elements = max(o.working_elements, int(wm.size))
                except Exception:
                    pass
            try:
                o.readout_checksum += float(out.mean())
            except Exception:
                pass
            return out

        mem.match_memory = probed
        return self

    def __exit__(self, *exc) -> None:
        if self._original is not None:
            self.processor.memory.match_memory = self._original
        return None


def reads_actually_changed(normal: MemoryObservation,
                           ablated: MemoryObservation) -> dict:
    """Did the ablation reach the memory path?

    Refuses to answer when either side was not measured. An unmeasured
    configuration cannot establish that an ablation was ineffective.
    """
    if not normal.available or not ablated.available:
        return {
            "configuration": ablated.configuration,
            "ablation_effective": None,
            "measured": False,
            "verdict": (
                "NOT MEASURED -- the memory interface was not reachable, so "
                "whether the ablation took effect is unknown. This is not "
                "evidence that it was ineffective."
            ),
        }
    same_engaged = (normal.long_term_engaged_calls
                    == ablated.long_term_engaged_calls)
    same_lt = normal.long_term_elements == ablated.long_term_elements
    # The readout itself is the strongest signal: identical reads mean the
    # ablation changed nothing the model actually consumed.
    same_readout = abs(normal.readout_checksum
                       - ablated.readout_checksum) < 1e-9
    effective = not (same_engaged and same_lt and same_readout)
    return {
        "configuration": ablated.configuration,
        "measured": True,
        "calls": [normal.calls, ablated.calls],
        "long_term_engaged_calls": [normal.long_term_engaged_calls,
                                    ablated.long_term_engaged_calls],
        "long_term_elements": [normal.long_term_elements,
                               ablated.long_term_elements],
        "readout_changed": not same_readout,
        "ablation_effective": effective,
        "verdict": (
            "reads changed; the ablation reached the memory path"
            if effective else
            "READS UNCHANGED -- the configuration did not alter what the "
            "model read, so any null from it is uninterpretable"
        ),
    }
