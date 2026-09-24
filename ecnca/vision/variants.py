"""The six comparison variants, capacity-matched on real functional counts.

Matching rule (from the plan): count parameters, per-cell prototypes, masks,
persistent context dimensions and inference heads. Dummy unused parameters do
not constitute a match, and residual differences are exposed rather than hidden
in a single total.

Every variant shares the reference's communication topology exactly -- one
trainable 3x3 perception, then 1x1 layers -- so only channel counts and the
extra mechanism differ, never the neighbourhood.

One layout rule follows from that. Sector channels are slot-indexed, and slot
indices are private to a cell: the slot a birth claims depends on that cell's
own occupancy history, so one cell's "slot 0" and its neighbour's "slot 0" are
not the same interpretation. They must therefore NOT be fed to the ordinary
perception convolution, whose weights are indexed by channel and so by slot.
Doing that made a pure slot permutation -- identical unordered prototype set --
move the next-step logits by 14.1 (measured). Perception reads the
permutation-free part of the state; the sector channels reach a neighbour only
through ``SectorMessage``, which pools them as a set.

    reference   the frozen Distill model, unchanged
    sectors     dynamic interpretation sectors only
    origin      evidence-origin memory only
    both        sectors + origin
    wider       parameter-matched plain NCA (no mechanism)
    decay       reset/history-decay baseline: the cheap way to forget
"""

from __future__ import annotations

import dataclasses
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .reference import (
    HIDDEN_CHANNELS, LIVING_THRESHOLD, OUTPUT_CHANNELS, ReferenceCA,
    ReferenceConfig,
)
from .gates import CosineGate, EqualityGate, LastSignatureGate
from .gating import ExactZeroGate, ResidualAdapter, adapter_hidden_for
from .origin import BoundedOriginMemory, MemoryConfig, OriginConfig, OriginMemory
from .sectors import (
    DynamicSectors, SectorConfig, SectorMessage, SectorState, mixture_readout,
)

VARIANTS = (
    "reference", "sectors", "origin", "both", "wider", "decay",
    # Baselines with the SAME content/credit separation, so a comparison
    # isolates the decision rule rather than the split. The learned mechanism
    # must beat these specifically in the harmful non-exact region, where a
    # byte-different repeat still carries most of the exact-repeat harm.
    "equality_gate",     # deterministic byte equality; free, exact, blind
    "cosine_gate",       # cosine threshold, fitted on CALIBRATION only
    "last_signature",    # learned, but sees only the previous signature
)
GATE_VARIANTS = ("equality_gate", "cosine_gate", "last_signature")

# Bias that makes a sigmoid gate start effectively closed. sigmoid(-12) is
# about 6e-6, so a transplanted variant reproduces the frozen reference to
# well inside the 1e-4 no-op tolerance while the gate remains trainable.
NOOP_GATE_BIAS = -12.0


@dataclass(frozen=True)
class VariantConfig:
    variant: str = "reference"
    hidden_channels: int = HIDDEN_CHANNELS
    perceive_features: int = 80
    mlp_features: int = 80
    fire_rate: float = 0.5
    add_noise: bool = True
    noise_std: float = 0.02
    living_threshold: float = LIVING_THRESHOLD
    sector: SectorConfig = SectorConfig()
    origin: OriginConfig = OriginConfig()
    memory: MemoryConfig = MemoryConfig()
    origin_dim: int = 8          # learned origin signature width
    cosine_threshold: float = 0.999   # cosine_gate; refit on calibration
    # Deterministic controls need their rule ACTIVE during evaluation. Staged
    # with the influence gate at 0 they reproduce the reference exactly, which
    # is the opposite of a baseline's job. `control_enabled` opens the gate to
    # full strength for a fixed-rule variant; the disabled setting is retained
    # so the equivalence audit can still check reference reproduction.
    control_enabled: bool = True
    adapter_hidden: int = 0           # 0 -> derive from the target count
    adapter_target_parameters: int = 30_000
    decay_rate: float = 0.9      # `decay` baseline only

    @property
    def sector_channels(self) -> int:
        return self.sector.channels if self.variant in ("sectors", "both") else 0

    @property
    def uses_origin_memory(self) -> bool:
        """Variants carrying a per-cell origin memory of any kind."""
        return self.variant in ("origin", "both") + GATE_VARIANTS

    @property
    def origin_channels(self) -> int:
        """Bounded memory slots, their validity flags, and one credit scalar.

        Every origin-bearing variant -- learned or baseline gate -- carries the
        SAME state, so a comparison isolates the decision rule.
        """
        if not self.uses_origin_memory:
            return 0
        m = self.memory
        return m.slots * (m.signature_dim + 1) + 1

    @property
    def reference_channel_n(self) -> int:
        """Mutable channels the REFERENCE computation owns: hidden + logits.

        Perception reads exactly these plus the grey channel, so the reference
        convolution keeps its original (80, 20, 3, 3) shape in every variant.
        Mechanism state lives OUTSIDE this and enters only through the control
        interface -- previously `origin` perception read 57 channels instead of
        20, which put the memory inside the reference computation even though
        the zero-padded transplant made it numerically exact.
        """
        return self.hidden_channels + OUTPUT_CHANNELS

    @property
    def channel_n(self) -> int:
        """Total mutable channels carried in the state tensor."""
        return (self.hidden_channels + OUTPUT_CHANNELS
                + self.sector_channels + self.origin_channels)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["sector"] = self.sector.to_dict()
        d.update(channel_n=self.channel_n,
                 sector_channels=self.sector_channels,
                 origin_channels=self.origin_channels)
        return d


def capacity_report(model: nn.Module, cfg: VariantConfig) -> dict:
    """The fields every variant must report, so matching is checkable."""
    total = sum(p.numel() for p in model.parameters())
    used = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "variant": cfg.variant,
        "total_parameters": total,
        "trainable_parameters": used,
        "state_channels_per_cell": cfg.channel_n + 1,
        "hidden_channels": cfg.hidden_channels,
        "per_cell_prototypes": (
            cfg.sector.n_slots if cfg.variant in ("sectors", "both") else 0
        ),
        "prototype_dim": (
            cfg.sector.proto_dim if cfg.variant in ("sectors", "both") else 0
        ),
        "persistent_context_dims": cfg.origin_channels,
        "masks_per_cell": 0,
        "inference_heads": 1,
        "receptive_field_per_step": "3x3 (Chebyshev radius 1)",
        "state_bytes_per_cell_fp32": (cfg.channel_n + 1) * 4,
    }


class VariantCA(nn.Module):
    """One CA with an optional mechanism. Topology identical across variants."""

    def __init__(self, config: VariantConfig | None = None):
        super().__init__()
        cfg = config or VariantConfig()
        if cfg.variant not in VARIANTS:
            raise ValueError(f"unknown variant {cfg.variant!r}")
        self.config = cfg
        self.channel_n = cfg.channel_n

        # Perception reads the permutation-free prefix only. For every variant
        # without sectors that is the whole state, so the reference topology is
        # untouched; for the sector variants the slot-indexed channels are
        # withheld and delivered by SectorMessage instead.
        # The reference perception reads the grey channel plus the reference's
        # OWN mutable channels and nothing else, so its tensor keeps the
        # original (80, 20, 3, 3) shape in every variant. This previously
        # excluded only sector channels, leaving origin memory inside the
        # reference computation at 57 input channels -- numerically exact via
        # the zero-padded transplant, but the wrong architecture: mechanism
        # state must enter through the control interface, not perception.
        self.perceive_channels = cfg.reference_channel_n + 1
        self.perceive = nn.Conv2d(self.perceive_channels, cfg.perceive_features,
                                  kernel_size=3, padding=1)
        self.hidden = nn.Conv2d(cfg.perceive_features, cfg.mlp_features, 1)
        # Write only the channels the residual update OWNS: hidden content and
        # logits. Sector and origin channels are maintained by their own
        # modules, which overwrite whatever the residual put there -- so those
        # output rows received exactly zero gradient (measured 0.0 against
        # 2.2e-2 for the logit rows). They were dead parameters that diluted
        # the shared trunk and inflated the count `wider` was matched against.
        self.residual_channels = cfg.hidden_channels + OUTPUT_CHANNELS

        # `wider` keeps the reference's convolution SHAPES untouched and adds a
        # parallel residual adapter behind an exact-zero gate. Growing the
        # reference tensors changes their shapes, which changes which cuDNN
        # kernel runs and therefore the floating-point execution order; on a
        # recurrent CA that roundoff amplifies, so a grown backbone could not
        # be bit-identical to the frozen one even with silent added weights.
        self.adapter = None
        if cfg.variant == "wider":
            hidden = cfg.adapter_hidden or adapter_hidden_for(
                cfg.adapter_target_parameters,
                cfg.channel_n + 1, self.residual_channels)
            self.adapter = ResidualAdapter(
                cfg.channel_n + 1, self.residual_channels, hidden)
        self.out = nn.Conv2d(cfg.mlp_features, self.residual_channels, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

        if cfg.variant in ("sectors", "both"):
            self.sectors = DynamicSectors(cfg.hidden_channels, cfg.sector)
            self.sector_message = SectorMessage(cfg.hidden_channels, cfg.sector)
            # How strongly the occupancy-weighted mixture over a cell's own
            # hypotheses speaks in the final logits. Learned and zero-init, so
            # the variant starts as the reference plus a silent mechanism and
            # has to earn the contribution -- the same discipline the
            # zero-initialised residual head follows.
            # A sigmoid gate is never exactly zero: sigmoid(-12) is 6.1e-6,
            # and on a recurrent CA that residual compounds until predictions
            # diverge from the frozen reference, so no variant could pass a
            # strict pre-training equivalence audit. clamp(0,1) on a
            # zero-initialised raw parameter is EXACTLY zero and still carries
            # gradient 1.0 at zero.
            self.mixture_gate = nn.Conv2d(cfg.hidden_channels, 1, 1)
            nn.init.zeros_(self.mixture_gate.weight)
            nn.init.zeros_(self.mixture_gate.bias)
            self.mixture_strength = ExactZeroGate()
        else:
            self.sectors = None
            self.sector_message = None
            self.mixture_gate = None
        if cfg.uses_origin_memory:
            # Bounded multi-origin memory: recognition compares against the
            # WHOLE retained set, not only the previous signature, so A -> B ->
            # A is recognisable. Credit is a max over deliveries, making it
            # structurally idempotent and bounded rather than approximately so.
            self.origin_memory = BoundedOriginMemory(
                cfg.perceive_features,
                MemoryConfig(slots=cfg.memory.slots,
                             signature_dim=cfg.memory.signature_dim,
                             hidden=cfg.memory.hidden,
                             credit_ceiling=cfg.memory.credit_ceiling,
                             write_threshold=cfg.memory.write_threshold,
                             decay=cfg.memory.decay),
            )
            # Baseline gates replace the LEARNED novelty with a fixed rule,
            # while keeping the identical memory state and content/credit
            # split. `origin`/`both` use the learned path (gate is None).
            # A baseline gate REPLACES the learned comparison, so the memory's
            # pair scorer is unused in those variants and correctly receives no
            # gradient. Freezing it makes that explicit -- an unfrozen unused
            # module reads as a dead-parameter bug, and it would also inflate
            # the trainable count the capacity match is computed from.
            if cfg.variant == "equality_gate":
                self.credit_gate = EqualityGate()
            elif cfg.variant == "cosine_gate":
                self.credit_gate = CosineGate(cfg.cosine_threshold)
            elif cfg.variant == "last_signature":
                self.credit_gate = LastSignatureGate(cfg.memory.signature_dim)
            else:
                self.credit_gate = None
            # Exactly zero at initialisation, for the same reason: with
            # sigmoid(-12) the logit update was scaled by 1 - 6.1e-6 rather
            # than by 1, and the CA amplified the difference.
            self.credit_strength = ExactZeroGate()
            if self.credit_gate is not None and cfg.control_enabled:
                # A fixed rule has nothing to learn, so its influence is set
                # open rather than trained. An ENABLED control may legitimately
                # differ from the reference; only a DISABLED one must match.
                with torch.no_grad():
                    self.credit_strength.raw.fill_(1.0)
            if self.credit_gate is not None:
                for prm in self.origin_memory.pair.parameters():
                    prm.requires_grad_(False)
                for prm in self.origin_memory.provenance.parameters():
                    prm.requires_grad_(False)
        else:
            self.origin_memory = None
            self.credit_gate = None

    # -- layout -----------------------------------------------------------
    @property
    def _hidden_slice(self):
        return slice(1, 1 + self.config.hidden_channels)

    @property
    def _logit_slice(self):
        s = 1 + self.config.hidden_channels
        return slice(s, s + OUTPUT_CHANNELS)

    @property
    def _origin_slice(self):
        s = 1 + self.config.hidden_channels + OUTPUT_CHANNELS
        return slice(s, s + self.config.origin_channels)

    @property
    def _sector_slice(self):
        # LAST, deliberately. Everything before this point is what the
        # perception convolution is allowed to read; sector channels are
        # slot-indexed and must reach a neighbour only through SectorMessage.
        s = (1 + self.config.hidden_channels + OUTPUT_CHANNELS
             + self.config.origin_channels)
        return slice(s, s + self.config.sector_channels)

    @property
    def _perceive_slice(self):
        """The permutation-free prefix of the state: grey, hidden, logits,
        origin. A convolution over these is well defined because none of them
        is indexed by a per-cell slot number."""
        return slice(0, self.perceive_channels)

    # -- origin state packing ---------------------------------------------
    def _unpack_origin(self, x: torch.Tensor):
        """State channels -> (memory, valid, credit).

        Layout: all slot signatures, then all validity flags, then one credit
        scalar. Slot ORDER carries no meaning -- the memory's matching reduces
        symmetrically over slots -- so this is a storage convention only.
        """
        cfg = self.config.memory
        raw = x[:, self._origin_slice]
        b, _, h, w = raw.shape
        k, d = cfg.slots, cfg.signature_dim
        memory = raw[:, : k * d].view(b, k, d, h, w)
        valid = raw[:, k * d : k * d + k]
        credit = raw[:, k * d + k :]
        return memory, valid, credit

    def _pack_origin(self, memory: torch.Tensor, valid: torch.Tensor,
                     credit: torch.Tensor) -> torch.Tensor:
        b, k, d, h, w = memory.shape
        return torch.cat([memory.reshape(b, k * d, h, w), valid, credit], dim=1)

    def initialize(self, images: torch.Tensor) -> torch.Tensor:
        if images.dim() == 3:
            images = images.unsqueeze(1)
        state = torch.zeros(images.shape[0], self.channel_n,
                            images.shape[2], images.shape[3],
                            device=images.device, dtype=images.dtype)
        return torch.cat([images, state], dim=1)

    def classify(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, self._logit_slice]

    def living_mask(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :1] > self.config.living_threshold

    # -- message interface (same factorisation as the reference) ----------
    def message(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.perceive(x[:, self._perceive_slice]))

    def sector_neighbourhood(self, x: torch.Tensor
                             ) -> tuple[torch.Tensor | None, dict]:
        """The permutation-invariant read over neighbouring prototype sets."""
        if self.sector_message is None:
            return None, {}
        return self.sector_message(x[:, self._hidden_slice],
                                   x[:, self._sector_slice])

    def apply_message(self, x: torch.Tensor, m: torch.Tensor, *,
                      sector_message: torch.Tensor | None = None,
                      sector_diagnostics: dict | None = None,
                      fire: torch.Tensor | None = None,
                      noise: torch.Tensor | None = None,
                      generator: torch.Generator | None = None) -> torch.Tensor:
        cfg = self.config
        grey, state = x[:, :1], x[:, 1:]
        if self.sector_message is not None and sector_message is None:
            sector_message, sector_diagnostics = self.sector_neighbourhood(x)
        # ds covers hidden + logits only; mechanism channels pass through and
        # are then written by their own modules below.
        ds = self.out(F.relu(self.hidden(m)))
        if self.adapter is not None:
            # Parallel branch; exactly zero until its gate opens, so the
            # backbone's own computation is untouched bit for bit.
            ds = ds + self.adapter(x)
        tail = state.shape[1] - self.residual_channels
        if tail:
            ds = torch.cat([ds, torch.zeros_like(state[:, self.residual_channels:])], 1)

        if cfg.add_noise:
            if noise is None:
                noise = torch.empty_like(ds)
                noise.normal_(0.0, cfg.noise_std, generator=generator)
            ds = ds + noise

        if fire is None:
            fire = torch.rand(grey.shape, device=x.device, dtype=x.dtype,
                              generator=generator) <= cfg.fire_rate
        residual = fire & self.living_mask(x)

        # --- evidence-origin credit ---------------------------------------
        diagnostics: dict = {}
        if self.origin_memory is not None:
            mem_in, valid_in, credit_in = self._unpack_origin(x)
            sig, learned_novelty, provenance, mem_out, valid_out, credit_out = \
                self.origin_memory(m, mem_in, valid_in, credit_in)

            if self.credit_gate is None:
                # Learned path: novelty comes from the bounded memory's own
                # comparison against the whole retained set.
                novelty = learned_novelty
                gate_detail = {"rule": "learned-bounded-memory"}
            else:
                # Baseline path: a fixed rule replaces the learned decision,
                # over the IDENTICAL memory state, so the comparison isolates
                # the rule rather than the content/credit split.
                gr = self.credit_gate(sig, mem_in)
                novelty = gr.credit
                gate_detail = gr.detail
                # Credit must follow the gate's decision, not the learned one.
                credit_out = torch.maximum(
                    credit_in, novelty * cfg.memory.credit_ceiling
                ).clamp(max=cfg.memory.credit_ceiling)

            # Content transformation is unrestricted; only the EVIDENTIAL
            # contribution is scaled. A repeat may still refine what the cell
            # computes -- it simply earns no fresh credit.
            #
            # The scale is blended from 1.0 toward `novelty` through a gate
            # that starts closed. An untrained novelty head outputs 0.5, which
            # would HALVE the reference's own logit update and make a
            # transplanted variant deviate by up to 7.7 before training. At
            # `credit_gate_bias` the blend starts at ~0, so the variant begins
            # as the frozen reference exactly and learns to apply credit.
            lo = cfg.hidden_channels
            hi = lo + OUTPUT_CHANNELS
            blend = self.credit_strength()
            scale = torch.ones_like(ds)
            scale[:, lo:hi] = 1.0 + blend * (novelty - 1.0)
            diagnostics["credit_blend"] = blend.detach()
            ds = ds * scale
            diagnostics["novelty"] = novelty.mean()
            diagnostics["credit"] = credit_out.mean()
            diagnostics["memory_slots_valid"] = valid_out.mean()
            self.last_provenance_logits = provenance
            self.last_gate_detail = gate_detail
            self._pending_origin = (sig, mem_out, valid_out, credit_out)

        ds = ds * residual.to(ds.dtype)
        new_state = state + ds

        # --- decay baseline ------------------------------------------------
        if cfg.variant == "decay":
            lo = cfg.hidden_channels
            new_state = torch.cat([
                new_state[:, :lo] * cfg.decay_rate, new_state[:, lo:],
            ], dim=1)

        out = torch.cat([grey, new_state], dim=1)

        # --- origin bookkeeping -------------------------------------------
        if self.origin_memory is not None:
            _, mem_out, valid_out, credit_out = self._pending_origin
            # A dead cell holds no origin state, matching every other channel.
            keep = residual.to(credit_out.dtype)
            packed = self._pack_origin(
                mem_out, valid_out, credit_out * keep
            )
            # The trailing slice matters: sector channels now sit AFTER the
            # origin block, and dropping it silently truncated the state.
            out = torch.cat([
                out[:, : self._origin_slice.start],
                packed,
                out[:, self._origin_slice.stop:],
            ], dim=1)

        # --- sector bookkeeping -------------------------------------------
        if self.sectors is not None:
            # The sector module itself is strictly per-cell. Everything a cell
            # learns about its neighbours' interpretations arrives in
            # `sector_message`, which is one 3x3 hop and pools the neighbours'
            # slots as an unordered set. See the notes in sectors.SectorMessage.
            raw, sd = self.sectors.step(
                out[:, self._hidden_slice], out[:, self._sector_slice],
                alive_mask=self.living_mask(out)[:, 0],
                message=sector_message,
            )
            out = torch.cat([
                out[:, : self._sector_slice.start], raw,
                out[:, self._sector_slice.stop:],
            ], dim=1)
            diagnostics.update(sd)
            if sector_diagnostics:
                diagnostics.update(sector_diagnostics)

            # --- sector-conditioned prediction -----------------------------
            # The final prediction is the occupancy-weighted mixture over the
            # cell's live hypotheses, added to the residual logits through a
            # learned gate. A single argmax slot would discard exactly the
            # competing reading the mechanism exists to keep, so the readout
            # is the mixture, not the winner.
            mix, weights = mixture_readout(raw, self.config.sector)
            # Exactly zero at initialisation: the spatial map is multiplied by
            # a clamped scalar that starts at 0.0, so the mixture contributes
            # nothing until the gate is trained open.
            gate = self.mixture_strength() * torch.sigmoid(
                self.mixture_gate(out[:, self._hidden_slice]))
            gate = gate * self.living_mask(out).to(gate.dtype)
            lo, hi = self._logit_slice.start, self._logit_slice.stop
            out = torch.cat([
                out[:, :lo], out[:, lo:hi] + gate * mix, out[:, hi:],
            ], dim=1)
            diagnostics["mixture_gate"] = gate.mean()
            self.last_mixture_weights = weights

        self.last_diagnostics = diagnostics
        return out

    def forward(self, x: torch.Tensor, *, fire=None, noise=None,
                generator=None) -> torch.Tensor:
        sm, sd = self.sector_neighbourhood(x)
        return self.apply_message(x, self.message(x), sector_message=sm,
                                  sector_diagnostics=sd,
                                  fire=fire, noise=noise, generator=generator)


def match_capacity(target_parameters: int, base: VariantConfig,
                   tolerance: float = 0.02) -> VariantConfig:
    """Parameter-match the plain NCA to a mechanism variant.

    The capacity is added as a PARALLEL residual adapter behind an exact-zero
    gate; the reference's convolution shapes are left untouched.

    An earlier version searched `perceive_features` instead, which shrank the
    backbone (80 -> 32 features) so the frozen reference could not be
    transplanted into it at all: `wider` started from random weights and
    failed the no-op equivalence check at 0.8384 argmax agreement. That is not
    a controlled comparison -- the baseline must be the SAME trained network
    plus inert extra capacity, so any difference is attributable to the
    mechanism rather than to a different backbone.
    """
    cfg = VariantConfig(
        variant="wider", hidden_channels=base.hidden_channels,
        perceive_features=base.perceive_features,
        mlp_features=base.mlp_features,
        fire_rate=base.fire_rate, add_noise=base.add_noise,
        noise_std=base.noise_std, living_threshold=base.living_threshold,
    )
    # The adapter carries whatever the mechanism variant has above the
    # reference backbone.
    plain = sum(p.numel() for p in
                VariantCA(VariantConfig(variant="reference")).parameters())
    excess = max(target_parameters - plain, 0)
    if excess:
        cfg = dataclasses.replace(cfg, adapter_hidden=adapter_hidden_for(
            excess, cfg.channel_n + 1,
            cfg.hidden_channels + OUTPUT_CHANNELS))
    return cfg
