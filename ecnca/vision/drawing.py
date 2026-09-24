"""Instrumented drawing / replay for the published interactive demo.

Audited against the demo's own source, pinned in ``ecnca/vision/SOURCES.json``
(``demo.js``, sha256 ``d070bdd5…2dab``, retrieved 2026-09-11). The behaviours
reproduced here, with the source lines they come from:

* Grid is **56x56** (``const D = 28 * 2``), 20 channels, all zero at start.
* An MNIST sample is **zero-padded**, not scaled: ``padding = (56 - 28)/2 = 14``
  and ``tf.pad(digit, [[0,0],[14,14],[14,14],[0,19]])``. So the digit occupies
  the central 28x28 and is 28 cells from no boundary.
* The **brush is not the mutation protocol.** Drawing is
  ``state.assign(state.mul(mask).add(stroke_pad))`` with ``mask = 1 - stroke``
  applied to **all 20 channels**, where ``stroke`` is the antialiased canvas
  alpha in [0,1]. A bright stroke pixel therefore *attenuates that cell's
  hidden state toward zero*, and a faint antialiased edge fades it partially.
  Contrast ``switcheroo()`` (the digit-swap path), which masks by the binary
  ``> 0.1`` living rule. The brush uses a soft mask; the mutation uses a hard
  one. Conflating them would misattribute any drawing-history effect.
* Erasing (shift or the eraser button) is ``state.mul(mask)`` with no add, and
  uses a 5x larger radius.
* Updates **continue while drawing**: ``stepsPerFrame`` is driven by a speed
  slider, so an unknown number of CA steps elapse between strokes. Our replay
  records that number explicitly instead of leaving it implicit.
* ``firingChance = 0.5``; noise is ``randomNormal([1,h,w,ch-1], 0., 0.02)`` --
  i.e. inference in the demo is noisy, matching training.
* Readout is ``argMax`` over the 10 class channels concatenated with a constant
  0.01 "background" plane, mapped through a colour lookup. **No numeric vote is
  displayed**, which is itself a candidate explanation for a perceived failure:
  a colour map cannot show that a cell is nearly tied.

Nothing here reads the MNIST test split.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from .reference import LIVING_THRESHOLD, OUTPUT_CHANNELS
from .stochastic import UpdateSchedule, schedule_for_ids

GRID = 56               # const D = 28 * 2
MNIST_PAD = (GRID - 28) // 2   # 14
DEFAULT_RADIUS = 2.0    # drawRadius default
ERASER_SCALE = 5.0      # shift/eraser multiplies the radius by 5
FIRING_CHANCE = 0.5
BACKGROUND_LEVEL = 0.01  # the constant plane argMax competes against


# --- stroke capture ------------------------------------------------------
@dataclass
class Stroke:
    """One brush segment, in grid coordinates."""
    t_ms: float
    x0: float
    y0: float
    x1: float
    y1: float
    radius: float = DEFAULT_RADIUS
    erase: bool = False
    steps_after: int = 0   # CA updates simulated after this stroke

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DrawingRecord:
    """A complete, replayable drawing session."""
    strokes: list[Stroke] = field(default_factory=list)
    grid: int = GRID
    label: int | None = None
    label_source: str = "unlabelled"
    provenance: str = "procedural-diagnostic"
    model_hash: str | None = None
    resets: list[int] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "grid": self.grid,
            "label": self.label,
            "label_source": self.label_source,
            "provenance": self.provenance,
            "model_hash": self.model_hash,
            "resets": list(self.resets),
            "notes": self.notes,
            "n_strokes": len(self.strokes),
            "total_steps_during_drawing": sum(s.steps_after for s in self.strokes),
            "strokes": [s.to_dict() for s in self.strokes],
        }

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @staticmethod
    def load(path: Path) -> "DrawingRecord":
        d = json.loads(Path(path).read_text())
        rec = DrawingRecord(
            grid=d.get("grid", GRID), label=d.get("label"),
            label_source=d.get("label_source", "unlabelled"),
            provenance=d.get("provenance", "unknown"),
            model_hash=d.get("model_hash"), resets=d.get("resets", []),
            notes=d.get("notes", ""),
        )
        rec.strokes = [Stroke(**s) for s in d["strokes"]]
        return rec


# --- rasterisation -------------------------------------------------------
def rasterise_segment(
    grid: int, x0: float, y0: float, x1: float, y1: float, radius: float
) -> np.ndarray:
    """Antialiased capsule (thick line segment), matching a canvas stroke.

    The demo draws onto an HTML canvas with ``lineWidth`` and round caps, then
    reads back the alpha channel. We reproduce the alpha analytically as a
    smooth coverage falloff over one cell, which is what canvas antialiasing
    produces at these radii.
    """
    yy, xx = np.mgrid[0:grid, 0:grid].astype(np.float64)
    dx, dy = x1 - x0, y1 - y0
    length2 = dx * dx + dy * dy
    if length2 < 1e-12:
        dist = np.hypot(xx - x0, yy - y0)
    else:
        t = ((xx - x0) * dx + (yy - y0) * dy) / length2
        t = np.clip(t, 0.0, 1.0)
        dist = np.hypot(xx - (x0 + t * dx), yy - (y0 + t * dy))
    # Coverage: 1 inside, 0 beyond one cell outside, smooth in between.
    alpha = np.clip(radius + 0.5 - dist, 0.0, 1.0)
    return alpha.astype(np.float32)


def apply_stroke(
    state: torch.Tensor, alpha: np.ndarray, *, erase: bool
) -> torch.Tensor:
    """The demo's brush, exactly.

    ``mask = 1 - stroke`` multiplies **every** channel, so a bright stroke
    attenuates the cell's hidden state; the stroke intensity is then added to
    channel 0 only (unless erasing).
    """
    a = torch.as_tensor(alpha, dtype=state.dtype, device=state.device)
    a = a.unsqueeze(0).unsqueeze(0)               # (1,1,H,W)
    mask = 1.0 - a
    out = state * mask                             # attenuates ALL channels
    if not erase:
        add = torch.zeros_like(state)
        add[:, 0:1] = a
        out = out + add
    return out.clamp(min=0.0) if erase else out


# --- state construction --------------------------------------------------
def blank_state(channels: int = 20, grid: int = GRID,
                device: torch.device | str = "cpu") -> torch.Tensor:
    return torch.zeros(1, channels, grid, grid, device=device)


def pad_mnist(image: np.ndarray, grid: int = GRID) -> np.ndarray:
    """Zero-pad a 28x28 MNIST image into the centre of the grid (no scaling)."""
    if image.shape != (28, 28):
        raise ValueError(f"expected a 28x28 image, got {image.shape}")
    pad = (grid - 28) // 2
    out = np.zeros((grid, grid), dtype=np.float32)
    out[pad:pad + 28, pad:pad + 28] = image
    return out


def state_from_raster(
    raster: np.ndarray, channels: int = 20,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Arm A: a completed image with zero mutable state."""
    g = raster.shape[0]
    s = blank_state(channels, g, device)
    s[0, 0] = torch.as_tensor(raster, dtype=s.dtype, device=device)
    return s


def reset_mutable(state: torch.Tensor) -> torch.Tensor:
    """Arm C: clear mutable channels, preserve the image (demo `reset` keeps
    the grey plane only in the sense that it reloads it; here we preserve the
    raster the user actually drew)."""
    out = state.clone()
    out[:, 1:] = 0.0
    return out


# --- replay --------------------------------------------------------------
def render(record: DrawingRecord) -> np.ndarray:
    """The final grayscale raster a record produces, with no CA updates.

    Applied with the same brush arithmetic as the live path, so the raster is
    identical to what arm B ends up with in channel 0.
    """
    s = blank_state(1, record.grid)
    for st in record.strokes:
        a = rasterise_segment(record.grid, st.x0, st.y0, st.x1, st.y1, st.radius)
        s = apply_stroke(s, a, erase=st.erase)
    return s[0, 0].numpy()


@torch.no_grad()
def replay(
    model,
    record: DrawingRecord,
    *,
    arm: str,
    post_steps: int,
    schedule: UpdateSchedule | None = None,
    eval_seed: int = 0,
    example_id: int = 0,
    device: torch.device | str = "cpu",
) -> dict:
    """Run one of the four paired arms on the SAME final raster.

    A ``complete``     -- final raster, zero mutable state, then evolve.
    B ``progressive``  -- draw stroke by stroke with updates in between.
    C ``progressive_reset`` -- as B, then clear mutable state, keeping the image.
    D ``paused``       -- draw with NO updates, then evolve.

    ``post_steps`` is matched across arms. Pre-completion updates are counted
    separately in the result, because B does strictly more computation.
    """
    if arm not in ("complete", "progressive", "progressive_reset", "paused"):
        raise ValueError(f"unknown arm {arm!r}")
    device = torch.device(device)
    model.eval()
    ch = model.channel_n + 1
    grid = record.grid

    if schedule is None:
        schedule = schedule_for_ids(
            [example_id], steps=post_steps, channels=model.channel_n,
            fire_rate=model.config.fire_rate, eval_seed=eval_seed,
            stage=f"draw-post", add_noise=model.config.add_noise,
            noise_std=model.config.noise_std, height=grid, width=grid,
        )
    schedule = schedule.to(device)

    # Which arms simulate the drawing session, and which clear the learned
    # state afterwards. `progressive_reset` must live through the SAME history
    # as `progressive` -- same strokes, same draw-live schedule, same
    # pre-completion updates -- and only then have its mutable channels
    # cleared. Skipping the history would make it a relabelled `complete` and
    # destroy the contrast it exists to provide.
    draws_live = arm in ("progressive", "progressive_reset")
    clears_after = arm == "progressive_reset"

    pre_steps = 0
    if arm == "complete":
        state = state_from_raster(render(record), ch, device)
    else:
        state = blank_state(ch, grid, device)
        # A separate stream for the during-drawing updates, so the matched
        # post-completion schedule is untouched by how long drawing took.
        # Both live arms draw from this one stream, keyed identically.
        live = None
        total_live = sum(s.steps_after for s in record.strokes)
        if draws_live and total_live:
            live = schedule_for_ids(
                [example_id], steps=total_live, channels=model.channel_n,
                fire_rate=model.config.fire_rate, eval_seed=eval_seed,
                stage="draw-live", add_noise=model.config.add_noise,
                noise_std=model.config.noise_std, height=grid, width=grid,
            ).to(device)
        cursor = 0
        for st in record.strokes:
            a = rasterise_segment(grid, st.x0, st.y0, st.x1, st.y1, st.radius)
            state = apply_stroke(state, a, erase=st.erase)
            if draws_live:
                for _ in range(st.steps_after):
                    fire, noise = live.step(cursor)
                    state = model(state, fire=fire, noise=noise)
                    cursor += 1
                    pre_steps += 1
        if clears_after:
            # Preserve the raster the drawing session actually produced --
            # which, because updates ran between strokes, is not necessarily
            # render(record): the brush mask attenuates channel 0 of state the
            # CA has since modified. Keeping the session's own channel 0 is
            # what makes this "erase learned history", not "redraw cleanly".
            state = reset_mutable(state)

    final_raster = state[0, 0].detach().cpu().numpy().copy()

    # Post-completion evolution, matched across arms.
    alive = model.living_mask(state)
    traj = []
    for t in range(post_steps):
        fire, noise = schedule.step(t)
        state = model(state, fire=fire, noise=noise)
        traj.append(_readout(model, state, alive))

    return {
        "arm": arm,
        "pre_completion_updates": pre_steps,
        "post_completion_updates": post_steps,
        "total_updates": pre_steps + post_steps,
        "schedule_seed": schedule.seed,
        "final_raster": final_raster,
        "trajectory": traj,
        "alive_cells": int(alive.sum()),
        "final_state": state,
    }


def _readout(model, state: torch.Tensor, alive: torch.Tensor) -> dict:
    """Numeric class votes -- the thing the demo's colour map cannot show."""
    logits = model.classify(state)
    a2 = alive[:, 0]
    n = int(a2.sum())
    cell_pred = logits.argmax(1)
    votes = np.zeros(OUTPUT_CHANNELS, dtype=np.int64)
    if n:
        sel = cell_pred[a2]
        for c in range(OUTPUT_CHANNELS):
            votes[c] = int((sel == c).sum())
    # The demo's argMax includes a constant background plane; a cell whose top
    # class logit falls below it renders as background rather than a class.
    with_bg = torch.cat(
        [logits, torch.full_like(logits[:, :1], BACKGROUND_LEVEL)], dim=1
    )
    bg_cells = int(((with_bg.argmax(1) == OUTPUT_CHANNELS) & a2).sum())
    srt = logits.sort(dim=1, descending=True).values
    margin = ((srt[:, 0] - srt[:, 1]) * a2.to(logits.dtype)).sum() / max(n, 1)
    top = int(votes.max()) if n else 0
    return {
        "votes": votes.tolist(),
        "predicted": int(votes.argmax()) if n else -1,
        "agreement": top / max(n, 1),
        "disagreement": 1.0 - top / max(n, 1),
        "background_cells": bg_cells,
        "mean_top2_margin": float(margin),
        "alive": n,
    }
