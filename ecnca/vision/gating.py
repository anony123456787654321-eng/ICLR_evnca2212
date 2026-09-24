"""Structurally exact-zero trainable gates.

A sigmoid gate is never exactly zero. ``sigmoid(-12)`` is 6.1e-6, and a CA is
recurrent: on the trained 100k checkpoint that residual compounds until the
variant's predictions diverge from the reference, so no variant could pass a
strict equivalence audit before training. Pushing the bias further only moves
the step count at which the same thing happens -- 9.4e-14 at -30 is still not
zero.

``clamp(0, 1)`` on a zero-initialised raw parameter is exactly zero in the
forward pass and carries gradient 1.0 at zero (PyTorch's subgradient at the
lower bound), so the gate is a true no-op that can still open.

The one hazard is that a negative raw is DEAD: ``clamp`` has gradient 0 below
the bound, so a gate pushed negative never recovers. ``ExactZeroGate`` clamps
the raw parameter back to >= 0 after each optimiser step, which
``clamp_raw_(model)`` performs; a test asserts the gate survives a gradient
step that would otherwise bury it.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ExactZeroGate(nn.Module):
    """A gate whose initial output is exactly 0.0 and whose gradient is not.

    ``shape=()`` gives a scalar gate; a shape gives a per-channel or spatial
    map. ``scale`` bounds the open value, so a fully open gate contributes
    ``scale`` rather than 1.0 where that is wanted.
    """

    def __init__(self, shape: tuple[int, ...] = (), scale: float = 1.0):
        super().__init__()
        self.raw = nn.Parameter(torch.zeros(shape))
        self.scale = float(scale)

    def forward(self) -> torch.Tensor:
        # Exactly zero at raw=0, gradient 1.0 at zero, bounded above at 1.
        return self.raw.clamp(0.0, 1.0) * self.scale

    @torch.no_grad()
    def clamp_raw_(self) -> None:
        """Keep raw >= 0 so the gate can never be pushed into the dead region."""
        self.raw.clamp_(min=0.0)

    def is_closed(self) -> bool:
        return bool(torch.all(self.forward() == 0.0))

    def extra_repr(self) -> str:
        return f"shape={tuple(self.raw.shape)}, scale={self.scale}"


@torch.no_grad()
def clamp_gates_(model: nn.Module) -> int:
    """Clamp every ExactZeroGate's raw parameter back into the live region.

    Call after ``optimizer.step()``. Returns how many gates were clamped.
    """
    n = 0
    for m in model.modules():
        if isinstance(m, ExactZeroGate):
            m.clamp_raw_()
            n += 1
    return n


def all_gates_closed(model: nn.Module) -> bool:
    gates = [m for m in model.modules() if isinstance(m, ExactZeroGate)]
    return all(g.is_closed() for g in gates) if gates else True


class ResidualAdapter(nn.Module):
    """Parameter-matched added capacity that leaves the backbone bit-identical.

    Widening the reference's convolutions changes their shapes, which changes
    which cuDNN kernel runs and therefore the floating-point execution order.
    On a recurrent CA that roundoff amplifies, so a "wider" baseline built by
    growing tensors could not be bit-exact with the frozen backbone even when
    the added weights were silent.

    This keeps the reference convolutions untouched and adds a PARALLEL branch
    whose contribution passes through an exact-zero gate. Before training the
    variant is the reference, bit for bit; the adapter's parameters all receive
    gradient as soon as the gate leaves zero.
    """

    def __init__(self, in_channels: int, out_channels: int, hidden: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, out_channels, 1),
        )
        self.gate = ExactZeroGate()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gate() * self.body(x)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def adapter_hidden_for(target_parameters: int, in_channels: int,
                       out_channels: int) -> int:
    """Smallest hidden width whose adapter reaches ``target_parameters``.

    Used to match a mechanism variant's parameter count without touching the
    backbone's shapes.
    """
    best, best_err = 1, float("inf")
    for h in range(1, 1024):
        n = ResidualAdapter(in_channels, out_channels, h).parameter_count()
        err = abs(n - target_parameters)
        if err < best_err:
            best, best_err = h, err
        if n > target_parameters * 1.5:
            break
    return best
