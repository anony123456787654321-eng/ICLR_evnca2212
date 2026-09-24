"""Dynamic interpretation sectors for a strictly local NCA.

The target, measured on the frozen reference: progressive drawing loses 11.5
points (0.4450 -> 0.3300) because surviving early hidden state contaminates the
final interpretation, and it does so while becoming *more* confidently wrong
(bad-consensus 0.170 -> 0.215). Sectors exist to preserve competing
interpretations through that period instead of letting one early reading spread.

**Not a fixed-k classifier.** Sectors are regions of the cell-state vector
space. Their number adapts: a cell whose state is far from every prototype it
can see starts a new one (birth); prototypes that drift together merge; a
prototype nothing occupies retires. The analogy is consistent hashing expanding
and contracting with the represented structure, not a fixed partition.

**Locality is the hard constraint.** The previous study's
``GrowingSectorController`` reduces over the whole population -- ``.mean(0)``,
``.sum(dim=0)`` in ``maybe_birth``, ``maybe_density_birth``, ``maybe_retire``
and ``update_prototypes``. None of that is usable here. Every operation below
reads a 3x3 neighbourhood only, and a test asserts one-step locality against
the same radius bound as the reference.

**Slot indices are private to a cell.** A slot is storage, not a name. The
previous revision put prototype channels into the ordinary state and let the
3x3 ``perceive`` convolution read them, which silently assumed that one cell's
"slot 0" and its neighbour's "slot 0" denote the same interpretation. They do
not: the slot a birth happens to claim depends on that cell's own occupancy
history. Permuting a cell's slots while preserving the identical unordered
prototype set moved the next-step logits by 14.1 (measured). Neighbouring cells
now consume prototype *sets* through :class:`SectorMessage` instead, and the
raw slot-indexed channels never reach a convolution.

Representation. Each cell carries ``n_slots`` entries of (prototype vector of
dimension ``proto_dim``, scalar occupancy, belief vector of ``belief_dim``
class logits), all held in ordinary state channels. Slots are soft: a cell's
membership is a softmax over distances, so there is no hard assignment to
differentiate through, and slot *identity* is never assumed to be shared
between neighbours -- prototypes are matched by representation, which is what
makes "merge" meaningful across cells.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# The nine radius-one offsets, including the cell itself. Perception reads
# exactly this neighbourhood, and so does the sector message.
_OFFSETS = tuple((dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1))


@dataclass(frozen=True)
class SectorConfig:
    n_slots: int = 4             # storage capacity, NOT the number in use
    proto_dim: int = 8
    belief_dim: int = 10         # per-slot hypothesis: one logit per digit
    temperature: float = 0.5     # softmax sharpness over distances
    birth_distance: float = 1.2  # far from every visible prototype -> new one
    merge_distance: float = 0.35  # prototypes this close collapse together
    retire_occupancy: float = 0.05  # unoccupied prototypes fade
    occupancy_decay: float = 0.9
    prototype_lr: float = 0.25   # how fast a prototype tracks its members
    belief_lr: float = 0.5       # how fast a hypothesis tracks its evidence
    attn_dim: int = 8            # query/key width for the neighbour message
    message_dim: int = 12        # width of the aggregated neighbour message

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def slot_channels(self) -> int:
        """State channels one slot occupies: prototype, occupancy, belief."""
        return self.proto_dim + 1 + self.belief_dim

    @property
    def channels(self) -> int:
        """State channels consumed: prototypes, occupancies and beliefs."""
        return self.n_slots * self.slot_channels


class SectorState:
    """View onto the sector channels inside a cell-state tensor.

    Layout is blocked by quantity, not by slot: all prototypes, then all
    occupancies, then all beliefs. Nothing outside this class and
    :class:`SectorMessage` may depend on that order, and nothing at all may
    depend on the order of slots *within* a block.
    """

    def __init__(self, raw: torch.Tensor, config: SectorConfig):
        self.config = config
        b, _, h, w = raw.shape
        k, d, e = config.n_slots, config.proto_dim, config.belief_dim
        self.protos = raw[:, : k * d].view(b, k, d, h, w)
        self.occ = raw[:, k * d : k * d + k].view(b, k, 1, h, w)
        self.belief = raw[:, k * d + k :].view(b, k, e, h, w)

    @staticmethod
    def pack(protos: torch.Tensor, occ: torch.Tensor,
             belief: torch.Tensor) -> torch.Tensor:
        b, k, d, h, w = protos.shape
        e = belief.shape[2]
        return torch.cat([protos.reshape(b, k * d, h, w),
                          occ.reshape(b, k, h, w),
                          belief.reshape(b, k * e, h, w)], dim=1)

    @staticmethod
    def permute_slots(raw: torch.Tensor, config: SectorConfig,
                      perm) -> torch.Tensor:
        """Reorder a cell's slots. The unordered set of entries is unchanged.

        Test helper and documentation in one: it is the exact symmetry every
        consumer of sector channels has to respect.
        """
        sec = SectorState(raw, config)
        idx = torch.as_tensor(perm, dtype=torch.long, device=raw.device)
        return SectorState.pack(sec.protos[:, idx], sec.occ[:, idx],
                                sec.belief[:, idx])


def mixture_readout(raw: torch.Tensor, config: SectorConfig
                    ) -> tuple[torch.Tensor, torch.Tensor]:
    """Occupancy-weighted mixture over a cell's own hypotheses.

    Returns ``(logits, weights)`` of shapes (B,belief_dim,H,W) and (B,K,1,H,W).
    A cell's prediction is not one slot's belief but the mixture of every live
    slot's belief, weighted by how much evidence that slot holds. Two
    interpretations that both carry occupancy therefore both remain visible in
    the readout, which is the whole point of keeping them.

    Symmetric in the slot axis: the weights are a function of the occupancies
    alone and the sum is over that axis, so permuting slots cannot move it.
    """
    sec = SectorState(raw, config)
    live = sec.occ > config.retire_occupancy
    w = sec.occ * live.to(sec.occ.dtype)
    total = w.sum(dim=1, keepdim=True)
    # A cell with nothing live yet abstains rather than dividing by zero.
    w = torch.where(total > 1e-6, w / total.clamp(min=1e-6), torch.zeros_like(w))
    return (w * sec.belief).sum(dim=1), w


def _neighbour_mean(x: torch.Tensor) -> torch.Tensor:
    """3x3 mean over neighbours. Strictly local: radius 1, like perception."""
    c = x.shape[1]
    k = torch.full((c, 1, 3, 3), 1.0 / 9.0, device=x.device, dtype=x.dtype)
    return F.conv2d(x, k, padding=1, groups=c)


def _neighbour_max(x: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(x, kernel_size=3, stride=1, padding=1)


def _shift(x: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    """Translate by one cell with zero padding: the value at (dy,dx) from here.

    Zero padding matches the reference's ``padding=1`` convolution, so a border
    cell sees the same absent-neighbour convention through both paths.
    """
    x = F.pad(x, (1, 1, 1, 1))
    h, w = x.shape[-2] - 2, x.shape[-1] - 2
    return x[..., 1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]


class SectorMessage(nn.Module):
    """Permutation-invariant neighbourhood read over prototype sets.

    **Mechanism: content-query attention over (prototype, occupancy, belief)
    entries, pooled jointly across the nine radius-one offsets.**

    Chosen over DeepSets sum-pooling and over explicit similarity matching for
    one reason each. DeepSets would aggregate every visible prototype with a
    weight that does not depend on what the reading cell currently holds, so a
    cell could not preferentially read the neighbouring interpretation that
    competes with its own -- and preserving *competing* interpretations is the
    mechanism's entire purpose. Explicit similarity matching (Hungarian, or
    greedy nearest-prototype pairing) does produce a correspondence, but the
    assignment is discrete: it is not differentiable, and near a tie it is not
    even continuous, so the merge signal it feeds would jump. Attention keeps
    the soft, differentiable correspondence that the rest of the module already
    relies on, and the query is built from the reading cell's own content, so
    "which neighbouring interpretation matters to me" is learned rather than
    fixed.

    Invariance argument. Scores and values are computed per entry by 1x1
    projections, the softmax is taken over the flattened (offset, slot) axis,
    and the output is the weighted sum over that axis. Softmax-normalised
    weighted summation is a symmetric function of the multiset of
    (score, value) pairs, so permuting the slot axis -- at any offset,
    independently per offset -- leaves the result bitwise-equal up to
    floating-point summation order. No projection is ever indexed by slot.

    Locality. Every tensor is projected pointwise *before* being shifted, and
    the only spatial operation is a single one-cell shift. Effective radius is
    exactly 1, the same bound perception obeys.
    """

    def __init__(self, state_dim: int, config: SectorConfig):
        super().__init__()
        self.config = config
        cfg = config
        entry_dim = cfg.proto_dim + 1 + cfg.belief_dim
        # Query comes from the reading cell's own content; keys and values from
        # each visible entry. All 1x1, so they commute with the shift.
        self.query = nn.Conv2d(state_dim, cfg.attn_dim, 1)
        self.key = nn.Conv2d(entry_dim, cfg.attn_dim, 1)
        self.value = nn.Conv2d(entry_dim, cfg.message_dim, 1)

    def forward(self, hidden: torch.Tensor, raw: torch.Tensor
                ) -> tuple[torch.Tensor, dict]:
        """(B,state_dim,H,W), (B,channels,H,W) -> (B,message_dim,H,W)."""
        cfg = self.config
        sec = SectorState(raw, cfg)
        b, k, _, h, w = sec.occ.shape

        # Per-entry features, computed once at each cell before any shifting.
        entry = torch.cat([sec.protos, sec.occ, sec.belief], dim=2)
        entry = entry.reshape(b * k, -1, h, w)
        keys = self.key(entry).view(b, k, cfg.attn_dim, h, w)
        vals = self.value(entry).view(b, k, cfg.message_dim, h, w)
        live = (sec.occ > cfg.retire_occupancy)

        q = self.query(hidden).unsqueeze(1)                  # (B,1,A,H,W)
        scale = float(cfg.attn_dim) ** 0.5

        scores, values, alive = [], [], []
        for dy, dx in _OFFSETS:
            sk = _shift(keys.reshape(b, k * cfg.attn_dim, h, w), dy, dx)
            sv = _shift(vals.reshape(b, k * cfg.message_dim, h, w), dy, dx)
            sl = _shift(live.to(q.dtype).reshape(b, k, h, w), dy, dx)
            sk = sk.view(b, k, cfg.attn_dim, h, w)
            sv = sv.view(b, k, cfg.message_dim, h, w)
            scores.append((q * sk).sum(dim=2, keepdim=True) / scale)
            values.append(sv)
            alive.append(sl.view(b, k, 1, h, w) > 0.5)

        # Flatten (offset, slot) into ONE axis and softmax over it. Slot order
        # is destroyed here, which is precisely the guarantee being bought.
        score = torch.cat(scores, dim=1)                     # (B,9K,1,H,W)
        value = torch.cat(values, dim=1)                     # (B,9K,M,H,W)
        mask = torch.cat(alive, dim=1)                       # (B,9K,1,H,W)
        score = score.masked_fill(~mask, float("-inf"))
        none = (~mask).all(dim=1, keepdim=True)
        score = torch.where(none.expand_as(score), torch.zeros_like(score),
                            score)
        attn = torch.softmax(score, dim=1)
        # A cell that can see no live interpretation emits nothing rather than
        # a uniform average of dead slots.
        attn = attn * (~none).to(attn.dtype)
        message = (attn * value).sum(dim=1)                  # (B,M,H,W)

        diagnostics = {
            "message_entropy": (
                -(attn.clamp(min=1e-9) * attn.clamp(min=1e-9).log())
                .sum(dim=1).mean()
            ),
            "message_norm": message.abs().mean(),
        }
        return message, diagnostics


class DynamicSectors(nn.Module):
    """Local, adaptive sector bookkeeping over cell state.

    One call performs, in order: soft assignment, prototype tracking, belief
    tracking, occupancy update, birth, merge, retirement. Every step uses only
    the cell and its 3x3 neighbourhood.
    """

    def __init__(self, state_dim: int, config: SectorConfig | None = None):
        super().__init__()
        self.config = config or SectorConfig()
        cfg = self.config
        # Projects hidden content to the space sectors are defined in. Learned,
        # so "interpretation" is what the task makes distinguishable rather
        # than raw channel values.
        self.project = nn.Conv2d(state_dim, cfg.proto_dim, 1)
        # The hypothesis a cell's current content supports, before it is routed
        # to a slot. Reads the aggregated neighbour message too, so a slot's
        # belief is informed by the interpretations around it.
        self.evidence = nn.Conv2d(state_dim + cfg.message_dim, cfg.belief_dim, 1)
        # NOT zero-initialised. A zero-init output layer is the right idiom for
        # the CA's residual head -- it makes the automaton start as a no-op --
        # but `evidence` sits UPSTREAM of the whole sector mechanism. Zeroing it
        # makes `ev` identically zero, so no gradient can reach the beliefs or
        # the attention module that feeds them: measured grad(ev -> message)
        # exactly 0.0, and the entire SectorMessage query/key/value stack was
        # dead at 1, 3, 10 and 20 steps. The "starts silent" property is
        # provided instead by the zero-initialised `mixture_gate` in
        # VariantCA, which gates the mixture's contribution to the logits while
        # leaving this path differentiable.
        nn.init.normal_(self.evidence.weight, std=0.05)
        nn.init.zeros_(self.evidence.bias)

    # -- assignment -------------------------------------------------------
    def assign(self, z: torch.Tensor, sec: SectorState) -> torch.Tensor:
        """Soft membership of each cell in each of its own slots. (B,K,1,H,W).

        Retired slots (occupancy at or below the retirement floor) are excluded
        so a dead prototype cannot attract members.
        """
        d = torch.linalg.vector_norm(sec.protos - z.unsqueeze(1), dim=2, keepdim=True)
        logits = -d / self.config.temperature
        alive = sec.occ > self.config.retire_occupancy
        logits = logits.masked_fill(~alive, float("-inf"))
        # A cell with no live slot yet falls back to uniform rather than NaN.
        none_alive = (~alive).all(dim=1, keepdim=True)
        logits = torch.where(none_alive.expand_as(logits),
                             torch.zeros_like(logits), logits)
        return torch.softmax(logits, dim=1)

    # -- lifecycle --------------------------------------------------------
    def step(self, hidden: torch.Tensor, raw: torch.Tensor,
             alive_mask: torch.Tensor | None = None,
             message: torch.Tensor | None = None
             ) -> tuple[torch.Tensor, dict]:
        """One sector update. Returns (new raw sector channels, diagnostics).

        ``message`` is the permutation-invariant neighbourhood read produced by
        :class:`SectorMessage`. It is optional so the module can be exercised
        per-cell in isolation; when absent the beliefs see content only.
        """
        cfg = self.config
        sec = SectorState(raw, cfg)
        z = self.project(hidden)                      # (B,D,H,W)

        member = self.assign(z, sec)                  # (B,K,1,H,W)

        # Prototype tracking: each prototype moves toward the members it holds.
        #
        # NO neighbour term here, deliberately. Spreading is the job of the
        # perception convolution and of SectorMessage, each of which is a
        # single one-cell hop. Adding a second 3x3 hop inside this module would
        # compose with those to give an effective radius of 2 per step
        # (measured: the sector channels reached distance 2 while hidden and
        # logits stayed at 1). A sector variant would then access information
        # faster than the reference, and any gain would be attributable to
        # reach rather than to the mechanism. This module is per-cell.
        target = z.unsqueeze(1)
        protos = sec.protos + cfg.prototype_lr * member * (target - sec.protos)
        b, k, d, h, w = protos.shape

        # --- hypotheses ----------------------------------------------------
        # Each slot carries a belief, not just a historical prototype. The
        # evidence a cell currently has flows into the slots that hold it, in
        # proportion to membership, so a slot the cell has stopped occupying
        # retains the reading it was formed under instead of being overwritten
        # by whatever arrives next. That retention is what lets two temporal
        # interpretations coexist.
        if message is None:
            message = hidden.new_zeros(b, cfg.message_dim, h, w)
        ev = self.evidence(torch.cat([hidden, message], dim=1)).unsqueeze(1)
        belief = sec.belief + cfg.belief_lr * member * (ev - sec.belief)

        # Occupancy: decayed mass of current membership. Per-cell, as above.
        occ = cfg.occupancy_decay * sec.occ + (1 - cfg.occupancy_decay) * member

        # --- birth: a cell whose content has moved away from what its
        # prototypes encode starts a new interpretation ---------------------
        #
        # Distance is measured against the PRE-update prototypes, not the ones
        # just fitted. Fitting first and measuring after makes birth
        # unreachable: a prototype that has already tracked the cell's current
        # state is never far from it, and the active count collapses to exactly
        # 1 everywhere (measured). The meaningful signal is temporal -- content
        # arriving that the existing interpretations do not explain, which is
        # precisely the competing-interpretation case during drawing.
        dist = torch.linalg.vector_norm(
            sec.protos - z.unsqueeze(1), dim=2, keepdim=True
        )
        live = sec.occ > cfg.retire_occupancy
        far = torch.where(live, dist, torch.full_like(dist, float("inf")))
        nearest = far.min(dim=1, keepdim=True).values
        needs_birth = (nearest > cfg.birth_distance) | (~live).all(dim=1, keepdim=True)
        # Claim the emptiest slot. The choice reads occupancy only -- a
        # property of the slot's CONTENT -- so relabelling the slots relabels
        # the winner with them. Exact ties fall back to the lowest index, which
        # is the one place slot order is still visible; the entries involved
        # are then identical, so the resulting multiset does not depend on it.
        emptiest = occ.argmin(dim=1, keepdim=True)
        claim = torch.zeros_like(occ).scatter_(1, emptiest, 1.0) * needs_birth
        protos = torch.where(claim.expand_as(protos) > 0,
                             z.unsqueeze(1).expand_as(protos), protos)
        # A newborn slot starts from the evidence that caused the birth, so the
        # new interpretation is a real competing hypothesis immediately rather
        # than a zero vector that the mixture would read as "no opinion".
        belief = torch.where(claim.expand_as(belief) > 0,
                             ev.expand_as(belief), belief)
        occ = torch.where(claim > 0, torch.full_like(occ, 0.5), occ)

        # --- merge: prototypes that have drifted together collapse ---------
        #
        # Vectorised over the K(K-1)/2 pairs instead of a Python double loop.
        # The loop cost 6.1x the reference's time for only 2.3x its parameters
        # -- overhead charged to the mechanism's budget rather than to its
        # capacity. The survivor is the pair member with the HIGHER occupancy,
        # not the lower index: index order is private to a cell, so choosing by
        # it made the whole step non-equivariant and the next state disagreed
        # after a slot permutation. Occupancy is content, so relabelling the
        # slots relabels the survivor with them. The survivor takes the
        # occupancy-weighted mean of both prototypes and beliefs -- evidence is
        # conserved through the merge -- and a slot claimed by birth this step
        # is exempt for one step so a newborn interpretation is not destroyed
        # before it can compete.
        live = occ > cfg.retire_occupancy
        newborn = claim > 0
        eligible = live & ~newborn                      # (B,K,1,H,W)

        iu, ju = torch.triu_indices(k, k, offset=1, device=protos.device)
        pd = torch.linalg.vector_norm(
            protos[:, iu] - protos[:, ju], dim=2, keepdim=True
        )                                               # (B,P,1,H,W)
        pair_close = (
            (pd < cfg.merge_distance) & eligible[:, iu] & eligible[:, ju]
        )
        # Each slot folds into at most one partner per step. Built functionally
        # with torch.where rather than in-place slice writes -- the in-place
        # version broke autograd ("one of the variables needed for gradient
        # computation has been modified"), which the unit tests missed because
        # they never backpropagate. The smoke stage caught it.
        merged_flag = torch.zeros_like(occ)
        if bool(pair_close.any()):
            absorbed = torch.zeros_like(occ, dtype=torch.bool)
            for pidx in range(iu.shape[0]):
                i, j = int(iu[pidx]), int(ju[pidx])
                close = (pair_close[:, pidx:pidx + 1]
                         & ~absorbed[:, j:j + 1] & ~absorbed[:, i:i + 1])
                if not bool(close.any()):
                    continue
                wi, wj = occ[:, i:i + 1], occ[:, j:j + 1]
                tot = (wi + wj).clamp(min=1e-6)
                blend = (wi * protos[:, i:i + 1]
                         + wj * protos[:, j:j + 1]) / tot
                bblend = (wi * belief[:, i:i + 1]
                          + wj * belief[:, j:j + 1]) / tot
                # Occupancy decides which slot survives; ties keep i, and the
                # two entries are then interchangeable anyway.
                keep_i_wins = (wi >= wj) & close
                keep_j_wins = (wj > wi) & close

                sel_i = torch.zeros_like(occ, dtype=torch.bool)
                sel_i[:, i:i + 1] = True
                sel_j = torch.zeros_like(occ, dtype=torch.bool)
                sel_j[:, j:j + 1] = True
                write = ((sel_i & keep_i_wins) | (sel_j & keep_j_wins))
                protos = torch.where(write.expand_as(protos),
                                     blend.expand_as(protos), protos)
                belief = torch.where(write.expand_as(belief),
                                     bblend.expand_as(belief), belief)

                total = wi + wj
                new_i = torch.where(keep_i_wins, total,
                                    torch.where(keep_j_wins,
                                                torch.zeros_like(wi), wi))
                new_j = torch.where(keep_j_wins, total,
                                    torch.where(keep_i_wins,
                                                torch.zeros_like(wj), wj))
                occ = torch.cat([
                    occ[:, :i], new_i, occ[:, i + 1:j], new_j, occ[:, j + 1:],
                ], dim=1)
                # Whichever slot lost is the absorbed one.
                lost = (sel_i & keep_j_wins) | (sel_j & keep_i_wins)
                absorbed = absorbed | lost
                merged_flag = merged_flag + close.to(merged_flag.dtype)
        merged = merged_flag

        # --- retirement: unoccupied prototypes fade out --------------------
        retired = (occ <= cfg.retire_occupancy).to(occ.dtype)
        occ = occ * (1.0 - retired) + occ * retired * cfg.occupancy_decay

        if alive_mask is not None:
            # protos is (B,K,D,H,W) and occ is (B,K,1,H,W); the mask arrives as
            # (B,H,W), so it needs two leading singleton dims, not one.
            m = alive_mask.to(protos.dtype)[:, None, None]
            protos = protos * m
            occ = occ * m
            belief = belief * m

        new_raw = SectorState.pack(protos, occ, belief)
        mix, weights = mixture_readout(new_raw, cfg)
        # Switching: how much membership mass the update moved between slots,
        # as total variation between the assignment before the step and the
        # assignment the new prototypes induce for the same content. A cell
        # that has changed its mind reads high; a settled cell reads zero.
        # Slot-order-free: it compares two distributions over the SAME cell's
        # slots and reduces over that axis.
        after = self.assign(z, SectorState(new_raw, cfg))
        switching = 0.5 * (after - member).abs().sum(dim=1).mean()

        diagnostics = {
            "active_sectors": (occ > cfg.retire_occupancy).to(occ.dtype)
                              .sum(dim=1).mean(),
            "births": needs_birth.to(occ.dtype).mean(),
            "merges": merged.mean(),
            "retired": retired.mean(),
            "switching": switching,
            "membership_entropy": (
                -(member.clamp(min=1e-9) * member.clamp(min=1e-9).log())
                .sum(dim=1).mean()
            ),
            "mixture_entropy": (
                -(weights.clamp(min=1e-9) * weights.clamp(min=1e-9).log())
                .sum(dim=1).mean()
            ),
            "mixture_max_weight": weights.amax(dim=1).mean(),
            "mixture_logit_range": (mix.amax(dim=1) - mix.amin(dim=1)).mean(),
        }
        return new_raw, diagnostics
