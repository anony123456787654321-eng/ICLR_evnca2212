"""Faithful PyTorch port of the Distill self-classifying MNIST CA.

Source pinned in ``ecnca/vision/SOURCES.json``: the authors' notebook
``notebooks/mnist_ca.ipynb`` from google-research/self-organising-systems at
commit ``be4424851c6795d850cfad0977a797053670f7e9``.

This is the **20-channel classifier**, not the 16-channel growth model:
one immutable grey/pixel channel + 19 mutable channels (9 hidden for
communication, 10 digit logits).

Adaptation notes (differences from the source, all deliberate):

* Framework: TensorFlow 1.x/2.x Keras -> PyTorch. Layer shapes, activations,
  initializers, update rule, noise, fire rate, loss, gradient normalization,
  optimizer, and schedule are matched.
* ``Conv2D(..., padding="SAME")`` on a stride-1 3x3 kernel is exactly
  ``nn.Conv2d(..., padding=1)`` with zero padding, which is what TF uses here.
* The source's evaluation accuracy pools all live cells across the batch. We
  additionally report a per-image mean (see ``ecnca/vision/evaluate.py``);
  the pooled figure is retained for comparability with the article.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- constants, verbatim from the pinned notebook ------------------------
CHANNEL_N = 19          # mutable channels; excludes the grey channel
HIDDEN_CHANNELS = 9     # general-purpose communication channels
OUTPUT_CHANNELS = 10    # digit logits, the last 10 mutable channels
PERCEIVE_FEATURES = 80  # Conv2D(80, 3, relu, padding=SAME)
HIDDEN_FEATURES = 80    # Conv2D(80, 1, relu)
CELL_FIRE_RATE = 0.5
LIVING_THRESHOLD = 0.1  # "alive if normalized grey value is larger than 0.1"
RESIDUAL_NOISE_STD = 0.02  # 2e-2 on the residual updates
BATCH_SIZE = 16
POOL_SIZE = BATCH_SIZE * 10   # 160
TRAIN_STEPS_PER_ITER = 20     # iter_n inside train_step
LEARNING_RATE = 1e-3
LR_BOUNDARIES = (30_000, 70_000)
LR_VALUES = (1e-3, 1e-4, 1e-5)
TOTAL_ITERATIONS = 100_000
EVAL_STEPS = 200              # 200 before mutation, 200 after

assert HIDDEN_CHANNELS + OUTPUT_CHANNELS == CHANNEL_N


@dataclass(frozen=True)
class ReferenceConfig:
    channel_n: int = CHANNEL_N
    fire_rate: float = CELL_FIRE_RATE
    add_noise: bool = True
    living_threshold: float = LIVING_THRESHOLD
    noise_std: float = RESIDUAL_NOISE_STD
    perceive_features: int = PERCEIVE_FEATURES
    hidden_features: int = HIDDEN_FEATURES

    def to_dict(self) -> dict:
        return asdict(self)


class ReferenceCA(nn.Module):
    """The reference cellular automaton.

    State layout, channel index -> meaning:
        0        immutable grey (pixel intensity), never written
        1..9     hidden communication channels
        10..19   digit logits for classes 0..9
    """

    def __init__(self, config: ReferenceConfig | None = None):
        super().__init__()
        cfg = config or ReferenceConfig()
        self.config = cfg
        self.channel_n = cfg.channel_n

        # Perception is a single *trainable* 3x3 convolution over the full
        # state including the grey channel -- the article's "fully trainable
        # 3x3 kernels", replacing Growing-CA's fixed Sobel filters.
        self.perceive = nn.Conv2d(
            cfg.channel_n + 1, cfg.perceive_features, kernel_size=3, padding=1
        )
        self.hidden = nn.Conv2d(cfg.perceive_features, cfg.hidden_features, 1)
        self.out = nn.Conv2d(cfg.hidden_features, cfg.channel_n, 1)
        # Zero-initialized final layer: the CA starts as a no-op, which is
        # what makes the residual formulation trainable.
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    # -- state helpers ----------------------------------------------------
    def initialize(self, images: torch.Tensor) -> torch.Tensor:
        """(B,28,28) or (B,1,28,28) grey in [0,1] -> (B,20,28,28) state."""
        if images.dim() == 3:
            images = images.unsqueeze(1)
        if images.dim() != 4 or images.shape[1] != 1:
            raise ValueError(f"expected (B,1,H,W) grey image, got {tuple(images.shape)}")
        state = torch.zeros(
            images.shape[0], self.channel_n, images.shape[2], images.shape[3],
            device=images.device, dtype=images.dtype,
        )
        return torch.cat([images, state], dim=1)

    def classify(self, x: torch.Tensor) -> torch.Tensor:
        """Last 10 channels are the per-cell digit logits. (B,10,H,W)."""
        return x[:, -OUTPUT_CHANNELS:]

    def living_mask(self, x: torch.Tensor) -> torch.Tensor:
        """(B,1,H,W) bool. Alive iff the immutable grey channel > threshold."""
        return x[:, :1] > self.config.living_threshold

    # -- perception / message interface -----------------------------------
    #
    # An update is factored into two stages so that a neighbourhood message can
    # be captured and redelivered verbatim:
    #
    #   message(x) -> m      the 3x3 perception output: everything a cell learns
    #                        about its neighbourhood this step
    #   apply(x, m) -> x'    the purely per-cell update computed from m
    #
    # ``forward`` is exactly ``apply(x, message(x))``. Because ``apply`` reads
    # the grid only at the cell itself, redelivering a cached ``m`` reproduces
    # the information content of an earlier step with no new neighbourhood
    # information -- which is what E3's redelivery arm requires. See
    # ``ecnca/vision/redelivery.py``.

    def message(self, x: torch.Tensor) -> torch.Tensor:
        """The neighbourhood message: trainable 3x3 perception. (B,80,H,W)."""
        if x.shape[1] != self.channel_n + 1:
            raise ValueError(
                f"expected {self.channel_n + 1} channels, got {x.shape[1]}"
            )
        return F.relu(self.perceive(x))

    def apply_message(
        self,
        x: torch.Tensor,
        m: torch.Tensor,
        *,
        fire: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        fire_rate: float | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Per-cell update from a (possibly cached) message.

        Every operation here is 1x1: a cell reads only its own column of ``m``,
        its own grey value, and its own fire/noise draw. No spatial mixing.
        """
        grey, state = x[:, :1], x[:, 1:]
        ds = self.out(F.relu(self.hidden(m)))

        if self.config.add_noise:
            if noise is None:
                noise = torch.empty_like(ds)
                noise.normal_(0.0, self.config.noise_std, generator=generator)
            ds = ds + noise

        if fire is None:
            rate = self.config.fire_rate if fire_rate is None else fire_rate
            fire = torch.rand(
                grey.shape, device=x.device, dtype=x.dtype, generator=generator
            ) <= rate
        residual_mask = fire & self.living_mask(x)
        ds = ds * residual_mask.to(ds.dtype)

        # The grey channel is concatenated through unchanged: immutable input.
        return torch.cat([grey, state + ds], dim=1)

    # -- dynamics ---------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        *,
        fire_rate: float | None = None,
        manual_noise: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        fire: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """One stochastic CA update. Strictly local: 3x3 perception only.

        ``fire``/``noise`` accept precomputed draws from an
        ``ecnca.vision.stochastic.UpdateSchedule``, which is how evaluation is
        made deterministic and how matched variants are given identical update
        masks. ``manual_noise`` is retained as an alias for ``noise`` for
        compatibility with the source notebook's signature.
        """
        if noise is None:
            noise = manual_noise
        return self.apply_message(
            x,
            self.message(x),
            fire=fire,
            noise=noise,
            fire_rate=fire_rate,
            generator=generator,
        )


def make_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    return torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


def lr_at(iteration: int) -> float:
    """Piecewise-constant decay: 1e-3, then 1e-4 at 30k, 1e-5 at 70k."""
    lo, hi = LR_BOUNDARIES
    if iteration < lo:
        return LR_VALUES[0]
    if iteration < hi:
        return LR_VALUES[1]
    return LR_VALUES[2]


def target_pictures(labels: torch.Tensor) -> torch.Tensor:
    """(B,) int labels -> (B,10,1,1) one-hot targets, broadcast over cells.

    The source builds a full (B,28,28,10) one-hot "label picture"; the value is
    constant across cells, so we keep the broadcastable form and let the loss
    expand it. Numerically identical, and it avoids materialising 7840 copies.
    """
    return F.one_hot(labels, OUTPUT_CHANNELS).to(torch.float32)[:, :, None, None]


def individual_l2_loss(
    model: ReferenceCA, x: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """Per-sample stabilized loss: sum((one_hot - logits)^2) / 2.

    Matches the source exactly, including summing over *all* cells (dead cells
    included). Dead cells receive no gradient because their residual is masked
    to zero, so this is a reduction choice, not a leak: see
    ``tests/test_mnist_reference.py::test_dead_cells_receive_no_gradient``.
    """
    diff = target_pictures(labels).to(x.device) - model.classify(x)
    return (diff ** 2).sum(dim=(1, 2, 3)) / 2


def batch_l2_loss(
    model: ReferenceCA, x: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    return individual_l2_loss(model, x, labels).mean()


def batch_ce_loss(
    model: ReferenceCA, x: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """The *unstable* configuration from the article, kept for comparison."""
    logits = model.classify(x)
    tgt = labels[:, None, None].expand(-1, logits.shape[2], logits.shape[3])
    return F.cross_entropy(logits, tgt)


def normalize_gradients_(model: nn.Module, eps: float = 1e-8) -> None:
    """Per-tensor gradient normalization: g <- g / (||g|| + eps).

    From the source training loop. This is part of the stabilized recipe and
    materially changes the trajectory; omitting it is not the reference.
    """
    for p in model.parameters():
        if p.grad is not None:
            p.grad.div_(p.grad.norm() + eps)


def mutate(
    x: torch.Tensor, new_images: torch.Tensor, threshold: float = LIVING_THRESHOLD
) -> torch.Tensor:
    """Replace the digit, preserving eligible cell state.

    The source protocol: "erase all cell states that are not present in both
    digits and bring alive the cells that were not present in the original
    digit". Concretely the new grey channel replaces the old, and every mutable
    channel is multiplied by the *new* image's living mask -- so cells alive
    only in the old digit are zeroed, cells alive only in the new digit start
    from zero state, and cells alive in both carry their state across.
    """
    if new_images.dim() == 3:
        new_images = new_images.unsqueeze(1)
    mask = (new_images > threshold).to(x.dtype)
    return torch.cat([new_images, x[:, 1:] * mask], dim=1)
