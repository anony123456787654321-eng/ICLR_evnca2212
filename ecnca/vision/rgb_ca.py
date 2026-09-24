"""A CA for RGB images, and the interventions the headroom gate measures.

Design decisions the plan requires, and why:

**Cell-activity rule.** MNIST's rule is "alive iff grey > 0.1". Carrying that to
CIFAR would turn every dark pixel into a non-computing cell -- a black cat would
be mostly dead grid. So **every cell is always active** on RGB: the grid
communicates everywhere. ``ACTIVITY_RULE = "all-cells-active"`` is recorded in
every report so no reader assumes the MNIST threshold.

**Unobserved is not black.** Progressive reveal needs "not yet seen" to differ
from "seen, and dark". The input therefore carries an explicit **observation
mask** channel: 1 where revealed, 0 where not. Unrevealed pixels also hold 0 in
the colour channels, but the mask distinguishes the two cases, and a test
asserts that a black revealed patch and an unrevealed patch produce different
states.

**Communication topology is identical across variants**: a single trainable
3x3 convolution, the same stencil as the MNIST reference. Only channel counts
differ between variants, never the neighbourhood.

**Readout.** Every cell predicts the global image label from its own 10 logit
channels; the image prediction is the majority vote over cells (all cells, since
all are active). Reported alongside mean-logit pooling so the readout choice is
visible rather than implicit.

State layout (RGB):
    0..2    immutable observed colour (R,G,B)
    3       immutable observation mask (1 = revealed)
    4..k    hidden communication channels
    last 10 class logits
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

OUTPUT_CHANNELS = 10
ACTIVITY_RULE = "all-cells-active"
INPUT_CHANNELS = 4  # R,G,B + observation mask


@dataclass(frozen=True)
class RGBConfig:
    hidden_channels: int = 22          # so mutable = 22 + 10 = 32
    perceive_features: int = 96
    mlp_features: int = 96
    fire_rate: float = 0.5
    add_noise: bool = True
    noise_std: float = 0.02

    @property
    def channel_n(self) -> int:
        """Mutable channels: hidden + logits."""
        return self.hidden_channels + OUTPUT_CHANNELS

    @property
    def total_channels(self) -> int:
        return INPUT_CHANNELS + self.channel_n

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(channel_n=self.channel_n, total_channels=self.total_channels,
                 activity_rule=ACTIVITY_RULE, input_channels=INPUT_CHANNELS)
        return d


class RGBCA(nn.Module):
    """Strictly local RGB cellular automaton. 3x3 perception only."""

    def __init__(self, config: RGBConfig | None = None):
        super().__init__()
        cfg = config or RGBConfig()
        self.config = cfg
        self.channel_n = cfg.channel_n
        self.perceive = nn.Conv2d(cfg.total_channels, cfg.perceive_features,
                                  kernel_size=3, padding=1)
        self.hidden = nn.Conv2d(cfg.perceive_features, cfg.mlp_features, 1)
        self.out = nn.Conv2d(cfg.mlp_features, cfg.channel_n, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    # -- state ------------------------------------------------------------
    def initialize(self, images: torch.Tensor,
                   mask: torch.Tensor | None = None) -> torch.Tensor:
        """(B,H,W,3) or (B,3,H,W) in [0,1] -> (B,C,H,W) state.

        ``mask`` is (B,1,H,W) with 1 where a pixel has been observed. Default
        is fully observed.
        """
        if images.dim() != 4:
            raise ValueError(f"expected 4-D images, got {tuple(images.shape)}")
        if images.shape[-1] == 3:
            images = images.permute(0, 3, 1, 2).contiguous()
        b, _, h, w = images.shape
        if mask is None:
            mask = torch.ones(b, 1, h, w, device=images.device, dtype=images.dtype)
        rest = torch.zeros(b, self.channel_n, h, w,
                           device=images.device, dtype=images.dtype)
        # Colour is zeroed where unobserved; the mask is what says "unobserved"
        # rather than "observed black".
        return torch.cat([images * mask, mask, rest], dim=1)

    def classify(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, -OUTPUT_CHANNELS:]

    def activity_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Every cell is active. Returns (B,1,H,W) of ones.

        Deliberately not the MNIST intensity threshold: that would make dark
        image content non-computing.
        """
        return torch.ones_like(x[:, :1], dtype=torch.bool)

    def observation_mask(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, 3:4]

    # -- message interface (same factorisation as the MNIST reference) ----
    def message(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.perceive(x))

    def apply_message(self, x: torch.Tensor, m: torch.Tensor, *,
                      fire: torch.Tensor | None = None,
                      noise: torch.Tensor | None = None,
                      generator: torch.Generator | None = None) -> torch.Tensor:
        obs, state = x[:, :INPUT_CHANNELS], x[:, INPUT_CHANNELS:]
        ds = self.out(F.relu(self.hidden(m)))
        if self.config.add_noise:
            if noise is None:
                noise = torch.empty_like(ds)
                noise.normal_(0.0, self.config.noise_std, generator=generator)
            ds = ds + noise
        if fire is None:
            fire = torch.rand(x[:, :1].shape, device=x.device, dtype=x.dtype,
                              generator=generator) <= self.config.fire_rate
        ds = ds * fire.to(ds.dtype)   # no activity gate: all cells compute
        return torch.cat([obs, state + ds], dim=1)

    def forward(self, x: torch.Tensor, *, fire: torch.Tensor | None = None,
                noise: torch.Tensor | None = None,
                generator: torch.Generator | None = None) -> torch.Tensor:
        return self.apply_message(x, self.message(x), fire=fire, noise=noise,
                                  generator=generator)


# --- readout -------------------------------------------------------------
def readout(model, x: torch.Tensor) -> dict:
    """Per-cell logits -> image prediction. Both poolings reported."""
    logits = model.classify(x)
    b = logits.shape[0]
    cell_pred = logits.argmax(1)
    votes = torch.zeros(b, OUTPUT_CHANNELS, device=logits.device)
    for c in range(OUTPUT_CHANNELS):
        votes[:, c] = (cell_pred == c).flatten(1).sum(1).to(votes.dtype)
    n = cell_pred.flatten(1).shape[1]
    top = votes.max(1).values
    srt = logits.sort(dim=1, descending=True).values
    return {
        "majority": votes.argmax(1),
        "mean_logit": logits.mean(dim=(2, 3)).argmax(1),
        "votes": votes,
        "agreement": top / n,
        "disagreement": 1.0 - top / n,
        "cell_top2_margin": (srt[:, 0] - srt[:, 1]).mean(dim=(1, 2)),
    }


# --- interventions -------------------------------------------------------
def patch_mask(h: int, w: int, boxes, device=None, dtype=torch.float32) -> torch.Tensor:
    """(1,1,H,W) mask with 1 inside each (r0,c0,r1,c1) box."""
    m = torch.zeros(1, 1, h, w, device=device, dtype=dtype)
    for r0, c0, r1, c1 in boxes:
        m[0, 0, r0:r1, c0:c1] = 1.0
    return m


def informativeness(images: np.ndarray, boxes) -> np.ndarray:
    """A label-free proxy for how much a patch could carry: its colour variance.

    Used only to CHOOSE which patch to reveal, never as a metric. A patch of
    flat background has near-zero variance and should not count as
    "genuinely informative" -- that distinction is what separates the
    new-information arm from the repeat arm.
    """
    out = []
    for img in images:
        vals = []
        for r0, c0, r1, c1 in boxes:
            p = img[r0:r1, c0:c1]
            vals.append(float(p.var()))
        out.append(vals)
    return np.asarray(out)


def reveal(x: torch.Tensor, images: torch.Tensor, add: torch.Tensor) -> torch.Tensor:
    """Reveal additional pixels, preserving learned state elsewhere.

    ``add`` is (B,1,H,W) or (1,1,H,W) with 1 where new pixels become observed.
    Newly revealed cells receive their true colour and mask=1; the hidden state
    of every cell is left untouched, so this is an *information* intervention,
    not a state reset.
    """
    if images.shape[-1] == 3:
        images = images.permute(0, 3, 1, 2).contiguous()
    mask = x[:, 3:4]
    new_mask = torch.clamp(mask + add, max=1.0)
    colour = images * new_mask
    return torch.cat([colour, new_mask, x[:, INPUT_CHANNELS:]], dim=1)
