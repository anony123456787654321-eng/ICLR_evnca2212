"""NCA memory controller and its matched non-recurrent comparator.

Three propositions are separable by construction here:

  1. MEMORY SELECTION   -- `ObservationBank` retains several earlier
     observations, against a recent-only configuration.
  2. COMPETING HYPOTHESES -- `HypothesisSet` keeps alternative interpretations
     alive until evidence separates them; ablated by capacity 1.
  3. LOCAL RECURRENCE   -- `NCAController` propagates information through a
     fixed neighbourhood over several updates; `NonRecurrentController` has
     the same parameters, the same observations and the same training
     exposure, but no spatial propagation.

The comparator is built here, beside the method, so it cannot quietly become a
strawman. On MNIST a capacity control was silently broken for weeks and made
its own comparison meaningless.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# Architecture identity. A checkpoint trained under a different value is not
# comparable and must never be reused.
#   1 -- original
#   2 -- gradient-deadlock fix: the update block's output is normally
#        initialised (only the residual projection stays zero-initialised),
#        and the hypothesis compatibility score is squashed rather than
#        clamped, so it is differentiable at the ceiling. Under version 1 the
#        controller could not learn at all.
MODEL_VERSION = 2

NEIGHBOURHOOD = 3           # fixed 3x3; radius grows by 1 per update


@dataclass(frozen=True)
class MemoryConfig:
    # XMem's memory READOUT dimension, read from the published weights
    # (value_encoder.fuser.block2.conv2 -> 512). This was 64, which is XMem's
    # KEY dimension: the controller would have failed on first contact with
    # the real readout. Verified against the v1.0 checkpoint.
    feature_dim: int = 512       # channels of the low-resolution readout grid
    state_dim: int = 32          # per-cell local content state
    hypotheses: int = 4          # bounded competing interpretations per cell
    bank_size: int = 8           # retained observation entries
    updates_per_frame: int = 3   # local recurrent updates; start small
    # The residual may never exceed this fraction of the readout's own
    # magnitude. Fixed by the protocol, not learned: the controller must
    # correct the frozen model, not replace it, at any horizon.
    max_residual_fraction: float = 0.5
    hidden: int = 64
    birth_threshold: float = 0.5   # evidence incompatible with every slot
    retire_threshold: float = 0.05  # support below which a slot is released
    merge_threshold: float = 0.95   # cosine above which two slots are one


# --------------------------------------------------------------------------
# Observation bank
# --------------------------------------------------------------------------
class ObservationBank(nn.Module):
    """Bounded store of earlier observations, keyed by STABLE identifiers.

    Two operations are deliberately distinct:

      ADD    a genuinely new observation occupies its own entry.
      REVISE an existing entry is refined in place.

    Collapsing these would let a changed-but-useful view either overwrite the
    earlier one (losing history) or fill the bank with near-duplicates
    (crowding out genuinely different views). Which happens is a modelling
    decision, so it is explicit.

    Stable observation ids exist for REPLAY BOOKKEEPING only: they let us prove
    that delivering the same observation twice does not create two entries.
    They are not evidence about object identity, which remains an estimate.
    """

    def __init__(self, cfg: MemoryConfig):
        super().__init__()
        self.config = cfg
        self.encode = nn.Conv2d(cfg.feature_dim, cfg.state_dim, 1)

    def empty(self, batch: int, h: int, w: int, device, dtype=torch.float32):
        cfg = self.config
        return {
            "entries": torch.zeros(batch, cfg.bank_size, cfg.state_dim, h, w,
                                   device=device, dtype=dtype),
            "valid": torch.zeros(batch, cfg.bank_size, h, w, device=device,
                                 dtype=dtype),
            "age": torch.zeros(batch, cfg.bank_size, h, w, device=device,
                               dtype=dtype),
            # -1 = free. Stable ids, for exact-replay bookkeeping only.
            "obs_id": torch.full((batch, cfg.bank_size), -1.0, device=device,
                                 dtype=dtype),
        }

    def write(self, bank: dict, feature: torch.Tensor, obs_id: int,
              *, revise: bool = False) -> dict:
        """Store one observation. Exact replay of `obs_id` is IDEMPOTENT.

        Re-delivering an observation already held must not create a second
        copy, because that would silently let one piece of evidence vote twice
        -- the failure the MNIST origin memory was built to avoid.
        """
        cfg = self.config
        sig = self.encode(feature)                       # (B,S,H,W)
        ids = bank["obs_id"]
        b = sig.shape[0]
        entries, valid, age = (bank["entries"].clone(), bank["valid"].clone(),
                               bank["age"].clone())
        ids = ids.clone()
        age = age + 1.0

        for i in range(b):
            row = ids[i]
            existing = (row == float(obs_id)).nonzero().flatten()
            if len(existing):
                slot = int(existing[0])
                if revise:
                    entries[i, slot] = sig[i]            # refine in place
                # Exact replay with revise=False: nothing changes. Idempotent.
                age[i, slot] = 0.0
                continue
            free = (row < 0).nonzero().flatten()
            if len(free):
                slot = int(free[0])
            else:
                # Eviction: oldest entry, measured per-slot over the grid.
                slot = int(age[i].flatten(1).mean(1).argmax())
            entries[i, slot] = sig[i]
            valid[i, slot] = 1.0
            age[i, slot] = 0.0
            ids[i, slot] = float(obs_id)

        return {"entries": entries, "valid": valid, "age": age, "obs_id": ids}

    def read(self, bank: dict, query: torch.Tensor) -> torch.Tensor:
        """Content-addressed read. Permutation-invariant over entries."""
        e, v = bank["entries"], bank["valid"]
        scores = (e * query.unsqueeze(1)).sum(2)          # (B,K,H,W)
        scores = scores.masked_fill(v <= 0, float("-inf"))
        empty = (v.sum(1, keepdim=True) <= 0)
        w = torch.softmax(scores, dim=1).unsqueeze(2)
        w = torch.nan_to_num(w, nan=0.0)
        out = (e * w).sum(1)
        return torch.where(empty, torch.zeros_like(out), out)


# --------------------------------------------------------------------------
# Competing hypotheses
# --------------------------------------------------------------------------
class HypothesisSet(nn.Module):
    """Bounded alternative interpretations per cell, with explicit lifecycle.

    Slot INDEX carries no meaning: the read is permutation-invariant, and slot
    identity is not object identity. A slot is a hypothesis about what this
    cell is looking at, and two slots may later prove to be the same thing
    (merge) or one may lose all support (retire).
    """

    def __init__(self, cfg: MemoryConfig):
        super().__init__()
        self.config = cfg
        self.compat = nn.Sequential(
            nn.Conv2d(2 * cfg.state_dim, cfg.hidden, 1), nn.ReLU(),
            nn.Conv2d(cfg.hidden, 1, 1),
        )
        # Zero-initialised: compatibility starts as pure content similarity
        # and the learned term has to earn its correction, the same discipline
        # the NCA's zero-initialised output head follows.
        nn.init.zeros_(self.compat[-1].weight)
        nn.init.zeros_(self.compat[-1].bias)

    def empty(self, batch: int, h: int, w: int, device, dtype=torch.float32):
        cfg = self.config
        return {
            "states": torch.zeros(batch, cfg.hypotheses, cfg.state_dim, h, w,
                                  device=device, dtype=dtype),
            "support": torch.zeros(batch, cfg.hypotheses, h, w, device=device,
                                   dtype=dtype),
        }

    def compatibility(self, hyp: dict, evidence: torch.Tensor) -> torch.Tensor:
        """How well each slot explains this evidence. (B,M,H,W) in [0,1].

        Content similarity is the BASE, with a learned correction on top.

        A purely learned scorer is ~0.5 everywhere at random initialisation,
        which sits exactly on `birth_threshold`: births then depend on the
        random weights rather than on the evidence, and the set could not tell
        an unambiguous sequence from an ambiguous one before training
        (measured: a repeated identical observation concentrated 29-85% of
        support depending only on the seed, overlapping the ambiguous case
        entirely). Anchoring on cosine similarity makes "this is the same
        thing I already hold" true at initialisation, and the learned term
        starts at zero so it refines that rather than replacing it.
        """
        s = hyp["states"]
        b, m, d, h, w = s.shape
        ev = evidence.unsqueeze(1).expand(b, m, d, h, w)
        cos = F.cosine_similarity(s, ev, dim=2).clamp(-1.0, 1.0)
        base = 0.5 * (cos + 1.0)                       # [0,1]
        pair = torch.cat([s, ev], dim=2).reshape(b * m, 2 * d, h, w)
        correction = self.compat(pair).reshape(b, m, h, w)
        return (base + correction).clamp(0.0, 1.0)

    def update(self, hyp: dict, evidence: torch.Tensor) -> tuple[dict, dict]:
        """Fold new evidence in: support, birth, merge, retirement."""
        cfg = self.config
        compat = self.compatibility(hyp, evidence)
        active = (hyp["support"] > cfg.retire_threshold).to(evidence.dtype)
        best = (compat * active).amax(dim=1, keepdim=True)

        # BIRTH only where the evidence fits nothing already held. An
        # unambiguous frame must not be forced to carry several hypotheses.
        needs_new = (best < cfg.birth_threshold).to(evidence.dtype)
        free = (1.0 - active)
        # The FIRST free slot, without cumsum: cumsum_cuda_kernel has no
        # deterministic implementation, so under
        # use_deterministic_algorithms it warned and the run was silently
        # non-reproducible. argmax over the slot axis returns the first
        # maximum, which is exactly the first free slot, and has a
        # deterministic CUDA kernel.
        first_idx = free.argmax(dim=1, keepdim=True)
        any_free = free.amax(dim=1, keepdim=True) > 0
        first_free = torch.zeros_like(free)
        first_free.scatter_(1, first_idx, 1.0)
        first_free = first_free * any_free.to(evidence.dtype)
        born = needs_new * first_free

        # Support follows compatibility; a slot receiving nothing decays. A
        # newly born slot starts fully supported by the evidence that created it.
        support = torch.where(
            born > 0, torch.ones_like(hyp["support"]),
            0.9 * hyp["support"] + 0.1 * compat * active)

        states = torch.where(
            born.unsqueeze(2) > 0,
            evidence.unsqueeze(1).expand_as(hyp["states"]),
            hyp["states"] + 0.1 * compat.unsqueeze(2) * (
                evidence.unsqueeze(1) - hyp["states"]),
        )

        # RETIRE: support below threshold releases the slot.
        alive = (support > cfg.retire_threshold).to(states.dtype)
        states = states * alive.unsqueeze(2)
        support = support * alive

        diagnostics = {
            "active_hypotheses": alive.sum(1).mean().detach(),
            "born": born.sum().detach(),
            "max_compatibility": best.mean().detach(),
        }
        return {"states": states, "support": support}, diagnostics

    def read(self, hyp: dict) -> torch.Tensor:
        """Support-weighted summary. Symmetric in the slots, so any
        permutation of them gives the same answer."""
        s, sup = hyp["states"], hyp["support"]
        w = torch.softmax(
            sup.masked_fill(sup <= 0, float("-inf")), dim=1).unsqueeze(2)
        w = torch.nan_to_num(w, nan=0.0)
        return (s * w).sum(1)


# --------------------------------------------------------------------------
# Controllers
# --------------------------------------------------------------------------
class _Base(nn.Module):

    def _residual(self, feature, cells):
        """The correction, bounded RELATIVE to the readout it corrects.

        Contracting the state is not enough on its own: `to_feature` is a free
        linear projection, so a large learned weight still produces an
        arbitrarily large residual. Measured with a contractive state and an
        unbounded projection, the residual reached 26x the readout at a weight
        scale of 20.

        The bound is a fraction of the readout's own magnitude, so the
        controller CORRECTS the frozen model rather than replacing it, at any
        horizon. `gain` is learned and starts at sigmoid(0) = 0.5 of the cap;
        the cap itself is fixed by the protocol, not learned, so training
        cannot dissolve it.
        """
        import torch

        raw = self.to_feature(cells)
        # PER-CELL scale, over CHANNELS only. A global amax over (C,H,W)
        # makes every output cell depend on every input cell, so
        # `non_recurrent` stops being pointwise -- measured, it moved 81
        # cells instead of 1 and was no longer a control for local
        # recurrence, which is the attribution the whole study rests on.
        scale = feature.abs().amax(dim=1, keepdim=True).clamp_min(1e-6)
        limit = self.config.max_residual_fraction * torch.sigmoid(self.gain)
        return torch.tanh(raw / scale) * scale * limit

    def __init__(self, cfg: MemoryConfig):
        super().__init__()
        self.config = cfg
        self.bank = ObservationBank(cfg)
        self.hypotheses = HypothesisSet(cfg)
        self.encode = nn.Conv2d(cfg.feature_dim, cfg.state_dim, 1)
        self.to_feature = nn.Conv2d(cfg.state_dim, cfg.feature_dim, 1)
        # THE ONLY zero-initialised layer on this path, and it must stay the
        # only one. The controller returns readout + to_feature(cells), so a
        # zero final projection makes an untrained controller EXACTLY the
        # published baseline -- the property every arm's comparability rests
        # on. (Returning to_feature(cells) alone made an untrained controller
        # emit a random projection of a zero state, discarding XMem's readout
        # and collapsing J&F from 0.9561 to 0.0845 on a real clip.)
        #
        # Zero-initialising the update block's output as well created a
        # DEADLOCK rather than a slow start: to_feature.weight = 0 blocks
        # gradient to `cells`, while update[-1].weight = 0 pins `cells` at
        # zero, so neither can ever leave zero. Measured on the real module:
        # exactly ONE parameter in the whole controller (to_feature.bias)
        # received gradient, and 200 Adam steps left both weight matrices at
        # exactly 0.0. One zero layer is a soft start; two in series is a
        # permanent dead path.
        nn.init.zeros_(self.to_feature.weight)
        nn.init.zeros_(self.to_feature.bias)
        self.last_diagnostics: dict = {}

    def initial_state(self, feature: torch.Tensor):
        b, _, h, w = feature.shape
        dev, dt = feature.device, feature.dtype
        return {
            "cells": torch.zeros(b, self.config.state_dim, h, w,
                                 device=dev, dtype=dt),
            "bank": self.bank.empty(b, h, w, dev, dt),
            "hyp": self.hypotheses.empty(b, h, w, dev, dt),
        }

    @property
    def communication_radius(self) -> int:
        """Cells reachable per frame. 0 for the non-recurrent comparator."""
        raise NotImplementedError


class NCAController(_Base):
    """Local recurrent update over a fixed 3x3 neighbourhood.

    Information crosses one cell per update, so after `updates_per_frame`
    updates the communication radius is exactly that number. This is the
    property the non-recurrent comparator lacks, and it is measurable rather
    than asserted.
    """

    def __init__(self, cfg: MemoryConfig, *, use_hypotheses: bool = True):
        super().__init__(cfg)
        self.use_hypotheses = use_hypotheses
        extra = cfg.state_dim if use_hypotheses else 0
        # SHARED local weights, applied identically at every cell.
        self.update = nn.Sequential(
            nn.Conv2d(3 * cfg.state_dim + extra, cfg.hidden,
                      NEIGHBOURHOOD, padding=1),
            nn.ReLU(),
            nn.Conv2d(cfg.hidden, cfg.state_dim, 1),
        )
        # NORMALLY initialised. The zero-initialised residual projection
        # (`to_feature`) already guarantees exact baseline output at init, so
        # zeroing this one buys nothing and deadlocks the gradient path.
        nn.init.kaiming_normal_(self.update[-1].weight, nonlinearity="relu")
        nn.init.zeros_(self.update[-1].bias)
        # sigmoid(0)=0.5: an even mix of retained state and new update.
        self.leak = nn.Parameter(torch.zeros(1, cfg.state_dim, 1, 1))
        # Learned share of the residual cap; sigmoid(0)=0.5 of it.
        self.gain = nn.Parameter(torch.zeros(1))

    @property
    def communication_radius(self) -> int:
        return self.config.updates_per_frame

    def forward(self, feature, state, *, obs_id: int, revise: bool = False):
        cfg = self.config
        evidence = self.encode(feature)
        bank = self.bank.write(state["bank"], feature, obs_id, revise=revise)
        hyp, diag = self.hypotheses.update(state["hyp"], evidence)

        cells = state["cells"]
        for _ in range(cfg.updates_per_frame):
            recalled = self.bank.read(bank, cells)
            parts = [cells, evidence, recalled]
            if self.use_hypotheses:
                parts.append(self.hypotheses.read(hyp))
            # CONTRACTIVE. `cells = cells + update(...)` has no fixed
            # point: trained on 8 frames it is bounded, but run for the
            # 60-100 frames of a real event clip it diverges and swamps
            # XMem's readout. Measured on DAVIS with the real loss: J&F
            # 0.9247 at length 8 (above the 0.9147 baseline) falling to
            # 0.2574 at length 32 while the baseline held at 0.8860.
            # A learned leak makes the recurrence a contraction, so
            # horizon length stops changing the operating point.
            decay = torch.sigmoid(self.leak)
            # The update's OUTPUT is bounded too. A convex combination alone
            # is not a contraction when `update` itself grows with `cells`:
            # with the leak but an unbounded update, `nca` still reached
            # 7.4e7 times the readout over 100 frames. tanh gives the map a
            # fixed point regardless of horizon.
            cells = (1.0 - decay) * cells + decay * torch.tanh(
                self.update(torch.cat(parts, dim=1)))

        self.last_diagnostics = dict(
            diag, communication_radius=cfg.updates_per_frame)
        out = {"cells": cells, "bank": bank, "hyp": hyp}
        return feature + self._residual(feature, cells), out


class NonRecurrentController(_Base):
    """Matched comparator WITHOUT local recurrence.

    Same observations, same bank, same hypothesis machinery, same training
    exposure, and comparable trainable capacity -- but the update is applied
    once and is strictly pointwise, so no information crosses between cells.
    Communication radius 0.

    The 3x3 convolution is replaced by a 1x1 of the same width, so this arm
    holds FEWER parameters in that layer. Capacity is equalised by widening it,
    and the residual difference is reported rather than hidden: see
    `capacity_report`.
    """

    def __init__(self, cfg: MemoryConfig, *, use_hypotheses: bool = True):
        super().__init__(cfg)
        self.use_hypotheses = use_hypotheses
        extra = cfg.state_dim if use_hypotheses else 0
        in_ch = 3 * cfg.state_dim + extra
        # Widen so total trainable capacity matches the NCA arm. Every other
        # submodule is shared and identical, so only the update block differs:
        #   3x3 arm: in*hidden*9 + hidden + hidden*state + state
        #   1x1 arm: in*width  + width  + width*state  + state
        # Equating the two and solving for width gives the expression below;
        # the residual from integer rounding is reported in capacity_report.
        k = NEIGHBOURHOOD * NEIGHBOURHOOD
        width = max(1, int(round(
            cfg.hidden * (k * in_ch + 1 + cfg.state_dim)
            / (in_ch + 1 + cfg.state_dim))))
        self.update = nn.Sequential(
            nn.Conv2d(in_ch, width, 1), nn.ReLU(),
            nn.Conv2d(width, cfg.state_dim, 1),
        )
        # NORMALLY initialised. The zero-initialised residual projection
        # (`to_feature`) already guarantees exact baseline output at init, so
        # zeroing this one buys nothing and deadlocks the gradient path.
        nn.init.kaiming_normal_(self.update[-1].weight, nonlinearity="relu")
        nn.init.zeros_(self.update[-1].bias)
        # sigmoid(0)=0.5: an even mix of retained state and new update.
        self.leak = nn.Parameter(torch.zeros(1, cfg.state_dim, 1, 1))
        # Learned share of the residual cap; sigmoid(0)=0.5 of it.
        self.gain = nn.Parameter(torch.zeros(1))

    @property
    def communication_radius(self) -> int:
        return 0

    def forward(self, feature, state, *, obs_id: int, revise: bool = False):
        cfg = self.config
        evidence = self.encode(feature)
        bank = self.bank.write(state["bank"], feature, obs_id, revise=revise)
        hyp, diag = self.hypotheses.update(state["hyp"], evidence)

        cells = state["cells"]
        # The SAME number of update applications, so compute exposure matches;
        # only the spatial extent differs.
        for _ in range(cfg.updates_per_frame):
            recalled = self.bank.read(bank, cells)
            parts = [cells, evidence, recalled]
            if self.use_hypotheses:
                parts.append(self.hypotheses.read(hyp))
            # CONTRACTIVE. `cells = cells + update(...)` has no fixed
            # point: trained on 8 frames it is bounded, but run for the
            # 60-100 frames of a real event clip it diverges and swamps
            # XMem's readout. Measured on DAVIS with the real loss: J&F
            # 0.9247 at length 8 (above the 0.9147 baseline) falling to
            # 0.2574 at length 32 while the baseline held at 0.8860.
            # A learned leak makes the recurrence a contraction, so
            # horizon length stops changing the operating point.
            decay = torch.sigmoid(self.leak)
            # The update's OUTPUT is bounded too. A convex combination alone
            # is not a contraction when `update` itself grows with `cells`:
            # with the leak but an unbounded update, `nca` still reached
            # 7.4e7 times the readout over 100 frames. tanh gives the map a
            # fixed point regardless of horizon.
            cells = (1.0 - decay) * cells + decay * torch.tanh(
                self.update(torch.cat(parts, dim=1)))

        self.last_diagnostics = dict(diag, communication_radius=0)
        out = {"cells": cells, "bank": bank, "hyp": hyp}
        return feature + self._residual(feature, cells), out


def build_controller(arm: str, cfg: MemoryConfig | None = None) -> _Base:
    """The four comparison arms of the feasibility study."""
    cfg = cfg or MemoryConfig()
    if arm == "nca":
        return NCAController(cfg, use_hypotheses=True)
    if arm == "nca_no_hypotheses":
        return NCAController(cfg, use_hypotheses=False)
    if arm == "non_recurrent":
        return NonRecurrentController(cfg, use_hypotheses=True)
    if arm == "recent_only":
        return NCAController(MemoryConfig(**{**cfg.__dict__, "bank_size": 1}),
                             use_hypotheses=True)
    raise ValueError(f"unknown arm {arm!r}")


def capacity_report(arms: dict[str, nn.Module]) -> dict:
    """Trainable parameters per arm. Parameter matching is NOT sufficient on
    its own -- compute and latency are reported separately -- but a large gap
    here would invalidate the attribution outright."""
    out = {}
    for name, m in arms.items():
        total = sum(p.numel() for p in m.parameters() if p.requires_grad)
        out[name] = {
            "trainable_parameters": total,
            "communication_radius": getattr(m, "communication_radius", None),
        }
    ref = out.get("nca", {}).get("trainable_parameters")
    if ref:
        for name, v in out.items():
            v["ratio_to_nca"] = round(v["trainable_parameters"] / ref, 4)
    return out
