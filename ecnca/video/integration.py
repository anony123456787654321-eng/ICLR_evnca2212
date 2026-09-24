"""The seam between XMem and our controller.

Everything here is written so that a missing XMem install fails LOUDLY at the
integration gate rather than silently degrading into a different experiment.
A stub that quietly replaced the baseline would produce numbers that look like
a comparison and are not one.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path


# XMem's published inference configuration. Every stage loads THIS dict, so a
# baseline and a method arm cannot silently run under different memory
# settings. Departures from the published values are recorded in the
# integration report rather than made silently.
XMEM_CONFIG = {
    "key_dim": 64,
    "value_dim": 512,
    "hidden_dim": 64,
    "enable_long_term": True,
    "enable_long_term_count_usage": True,
    "max_mid_term_frames": 10,
    "min_mid_term_frames": 5,
    "num_prototypes": 128,
    "max_long_term_elements": 10000,
    "top_k": 30,
    "mem_every": 5,
    "deep_update_every": -1,
    "single_object": False,
}


class XMemUnavailable(RuntimeError):
    """The baseline is not importable. No comparison is possible."""


@dataclass
class XMemHandles:
    """What we need from XMem, resolved once and checked."""

    module: object
    network: object
    checkpoint: str
    config: dict


def load_xmem(repo: Path, checkpoint: Path, *, device="cuda",
              config: dict | None = None) -> XMemHandles:
    """Import the official implementation and load published weights.

    Raises rather than falling back: a silent fallback would let the study
    report a comparison against something other than the published baseline.
    """
    repo = Path(repo)
    if not repo.exists():
        raise XMemUnavailable(
            f"no XMem checkout at {repo}. The launcher acquires it; it is not "
            f"vendored here, and there is no substitute baseline."
        )
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        mod = importlib.import_module("model.network")
    except Exception as exc:                       # pragma: no cover - env
        raise XMemUnavailable(
            f"XMem is present at {repo} but not importable ({exc}). Check the "
            f"pinned revision and its dependencies."
        ) from exc
    if not Path(checkpoint).exists():
        raise XMemUnavailable(f"no XMem weights at {checkpoint}")

    import torch

    # XMem loads its checkpoint without map_location, and the published v1.0
    # weights were saved on CUDA, so constructing it on a CPU-only machine
    # raises. Loading the state dict ourselves keeps the study runnable on
    # both, and the weights are identical either way.
    cfg = {**XMEM_CONFIG, **(config or {})}
    net = mod.XMem(cfg).eval()
    sd = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd)
    net.load_weights(sd, init_as_zero_if_needed=True) if hasattr(
        net, "load_weights") else net.load_state_dict(sd)
    net = net.to(device)
    for p in net.parameters():
        p.requires_grad_(False)
    return XMemHandles(module=mod, network=net, checkpoint=str(checkpoint),
                       config=dict(cfg))


# --------------------------------------------------------------------------
# The seam
# --------------------------------------------------------------------------
class ControlledReadout:
    """Applies our controller to XMem's memory readout.

    The contract is narrow on purpose: in and out are the same feature grid,
    so every arm is interchangeable and `arm="xmem"` is EXACTLY the published
    path with nothing of ours in it.
    """

    def __init__(self, controller=None):
        self.controller = controller
        self.state = None
        self._frame = 0

    def reset(self) -> None:
        self.state = None
        self._frame = 0

    def __call__(self, readout, *, obs_id: int | None = None,
                 revise: bool = False):
        if self.controller is None:
            return readout                     # the untouched baseline
        if self.state is None:
            self.state = self.controller.initial_state(readout)
        out, self.state = self.controller(
            readout, self.state,
            obs_id=self._frame if obs_id is None else obs_id, revise=revise)
        self._frame += 1
        return out


def assert_baseline_is_untouched(readout) -> dict:
    """arm="xmem" must return its input unchanged.

    If the seam perturbs the readout even slightly, the "baseline" in our
    tables is not XMem and the whole comparison is void.
    """
    import torch

    seam = ControlledReadout(None)
    out = seam(readout)
    identical = bool(torch.equal(out, readout))
    return {
        "baseline_untouched": identical,
        "max_abs_difference": 0.0 if identical
        else float((out - readout).abs().max()),
        "meaning": (
            "the published path is bit-identical with no controller attached"
            if identical else
            "THE SEAM PERTURBS THE BASELINE -- the comparison is invalid"
        ),
    }


def persistent_memory_bytes(xmem_state: dict | None,
                            controller_state: dict | None) -> dict:
    """Account persistent memory in BYTES, not slot counts.

    Adding a controller must not quietly add free storage, so the budget
    covers XMem's stores AND our bank, hypotheses, support and metadata.
    """
    import torch

    def walk(obj) -> int:
        if obj is None:
            return 0
        if torch.is_tensor(obj):
            return obj.numel() * obj.element_size()
        if isinstance(obj, dict):
            return sum(walk(v) for v in obj.values())
        if isinstance(obj, (list, tuple)):
            return sum(walk(v) for v in obj)
        return 0

    a, b = walk(xmem_state), walk(controller_state)
    return {
        "xmem_bytes": a,
        "controller_bytes": b,
        "total_bytes": a + b,
        "total_mib": round((a + b) / 2**20, 3),
    }


def matched_budget_config(baseline_bytes: int, controller_bytes: int,
                          *, xmem_working_capacity: int,
                          bytes_per_working_frame: int) -> dict:
    """Shrink XMem's working memory so totals match the unmodified baseline.

    Without this the controller arm simply has more memory, and any gain is
    explained by storage rather than by mechanism.
    """
    if bytes_per_working_frame <= 0:
        return {"matched": False,
                "reason": "per-frame cost unknown; cannot match budgets"}
    drop = -(-controller_bytes // bytes_per_working_frame)   # ceil
    new_cap = max(1, xmem_working_capacity - int(drop))
    return {
        "matched": True,
        "original_working_capacity": xmem_working_capacity,
        "reduced_working_capacity": new_cap,
        "frames_surrendered": int(drop),
        "controller_bytes": controller_bytes,
        "note": (
            "the unmodified policy is ALSO run at this reduced capacity, so a "
            "separation cannot come from having given ourselves more room"
        ),
    }
