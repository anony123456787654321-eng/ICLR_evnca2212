"""Gate 4D -- Evidence-Conserving Growing Sectors.

The frozen Gate 4 model fixes K = 6.  Here the ACTIVE sector count is inferred:
`k_max` is only a memory allocation, never a semantic commitment.

Separation of concerns, kept strict:

  content     the evolving belief vector decides sector MEMBERSHIP
  provenance  the root ledger decides evidence CREDIT

A message may move between sectors as its representation changes.  A new sector
is a new interpretation, never new evidence.  One root may be attributed across
several sectors, but its total credited evidence is capped:

    a_irk >= 0,   sum_{k in K_t} a_irk = 1   =>   sum_k rho_ir a_irk = rho_ir

and that identity must survive sector birth, reassignment, merge and retirement.
It holds by construction here: attribution is a softmax over the ACTIVE set, so
it renormalises to 1 whenever the active set changes, and the evidence total is
never recomputed from sector state.

Controller: a growing Voronoi / DP-means scheme.  Births come from PERSISTENT
geometric disagreement, never from message multiplicity -- duplicating a message
moves no cell in belief space, so it cannot create a sector.  Adding a prototype
mostly reassigns nearby cells, the vector-space analogue of consistent hashing's
bounded remapping.  Proposal order, tie-breaks and sector identities are
deterministic, so asynchronous delivery cannot produce arbitrary partitions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F_


@dataclass
class SectorConfig:
    dim: int = 16
    k_max: int = 12                 # memory capacity only, NOT the sector count
    # FROZEN on a development split (analysis/gate4d_threshold_dev.py, dev seed
    # 90001, disjoint from every evaluation seed).  Chosen as the configuration
    # satisfying all five dev scenarios with the fewest sector operations.
    birth_distance: float = 0.30    # cosine distance beyond which a cell dissents
    merge_distance: float = 0.15    # prototypes this close are candidates to merge
    retire_support: float = 0.02    # fraction of mass below which a sector dies
    patience: int = 3               # steps a condition must persist (hysteresis)
    temperature: float = 0.25
    max_births_per_step: int = 1    # deterministic, one proposal at a time
    proto_momentum: float = 0.5     # EMA on the centroid update, damps churn
    birth_support_fraction: float = 0.10  # a coherent group can birth immediately
    # A distance-to-one-centroid test misses two nearby but individually tight
    # modes: their midpoint can lie within `birth_distance` of both.  The
    # density fallback finds disconnected local components using a scale set by
    # the median nearest-neighbour distance.  Six times that local scale stayed
    # connected for the wide unimodal development control while separating
    # coherent groups; unlike a global threshold it adapts to representation
    # scale.  A component still needs population support and centroid separation
    # above the merge threshold.
    density_birth: bool = True
    density_scale: float = 6.0
    # Kept for the preregistered ablation.  Distance from a single centroid is
    # not the default because a noisy unimodal cloud can repeatedly birth and
    # re-merge an outlier group, while two nearby modes can both be close to
    # their midpoint.  Density-supported birth fixes both failure directions.
    distance_birth: bool = False
    # Controller centroids use hard DP-means assignments.  Soft q remains the
    # differentiable routing signal consumed by the NCA, but using it to move
    # detached prototypes pulls nearby sectors toward their shared midpoint and
    # creates a birth/merge oscillation.
    hard_prototype_update: bool = True
    initial_sectors: int = 1
    dynamic_ops: bool = True
    # Optional NCA-side linkage can say which cell beliefs remain statistically
    # compatible after accounting for posterior uncertainty.  The generic
    # controller defaults to vector-density linkage so frozen gates reproduce.
    uncertainty_birth: bool = False


@dataclass
class SectorState:
    prototypes: torch.Tensor        # [k_max, dim], unit norm
    active: torch.Tensor            # [k_max] bool
    sector_id: torch.Tensor         # [k_max] long -- stable identities
    dissent_age: torch.Tensor       # legacy scalar slot, retained for state compat
    proposal_age: Dict[int, int] = field(default_factory=dict)  # cell -> streak
    pending_prototype: Optional[torch.Tensor] = None
    pending_age: int = 0
    merge_age: Dict[Tuple[int, int], int] = field(default_factory=dict)
    retire_age: torch.Tensor = None
    next_id: int = 1
    births: int = 0
    merges: int = 0
    retirements: int = 0

    @property
    def n_active(self) -> int:
        return int(self.active.sum())


class GrowingSectorController:
    def __init__(self, cfg: SectorConfig, device: str = "cpu"):
        self.cfg = cfg
        self.device = device

    # ------------------------------------------------------------------ init
    def init_state(self, Z: torch.Tensor) -> SectorState:
        """Start with ONE active sector at a deterministic population medoid.

        Dynamic embeddings are population-centred, so their mean can be the
        zero vector precisely when two strong opinions oppose each other.  A
        zero prototype is equidistant from everyone and caused repeated births.
        The cosine medoid is an actual belief vector and is permutation-stable
        up to equivalent ties.
        """
        cfg = self.cfg
        protos = torch.zeros(cfg.k_max, cfg.dim, device=Z.device)
        Zn = F_.normalize(Z.detach(), dim=-1, eps=1e-8)
        centrality = (Zn @ Zn.T).mean(dim=1)
        best = torch.nonzero((centrality - centrality.max()).abs() < 1e-9).flatten().min()
        protos[0] = Zn[int(best)]
        active = torch.zeros(cfg.k_max, dtype=torch.bool, device=Z.device)
        active[0] = True
        sid = torch.zeros(cfg.k_max, dtype=torch.long, device=Z.device)
        sid[0] = 0
        # Fixed-K ablation: deterministic farthest-first initialisation.  The
        # dynamic method keeps the default of one and earns every later sector.
        n_initial = min(max(int(cfg.initial_sectors), 1), cfg.k_max, len(Zn))
        for slot in range(1, n_initial):
            sim = Zn @ protos[:slot].T
            distance = 1.0 - sim.max(-1).values
            candidate = torch.nonzero((distance - distance.max()).abs() < 1e-9).flatten().min()
            protos[slot] = Zn[int(candidate)]
            active[slot] = True
            sid[slot] = slot
        return SectorState(prototypes=protos, active=active, sector_id=sid,
                           dissent_age=torch.zeros(cfg.k_max, device=Z.device),
                           retire_age=torch.zeros(cfg.k_max, device=Z.device),
                           next_id=n_initial)

    def init_grouped_state(self, Z: torch.Tensor, group: torch.Tensor) -> SectorState:
        """Seed one viewpoint per provenance group, without counting messages.

        Group identity determines only the initial partition. Subsequent
        vector-space merge, birth, retirement and migration are unchanged.
        Repeating messages within a root leaves every prototype identical.
        """
        unique = torch.unique(group, sorted=True)
        if len(unique) > self.cfg.k_max:
            raise ValueError("provenance groups exceed sector memory capacity")
        prototypes = torch.zeros(
            self.cfg.k_max, self.cfg.dim, dtype=Z.dtype, device=Z.device
        )
        active = torch.zeros(self.cfg.k_max, dtype=torch.bool, device=Z.device)
        sector_id = torch.zeros(self.cfg.k_max, dtype=torch.long, device=Z.device)
        for slot, value in enumerate(unique):
            members = group == value
            prototypes[slot] = F_.normalize(
                Z[members].detach().mean(0), dim=-1, eps=1e-8
            )
            active[slot] = True
            sector_id[slot] = slot
        return SectorState(
            prototypes=prototypes,
            active=active,
            sector_id=sector_id,
            dissent_age=torch.zeros(self.cfg.k_max, device=Z.device),
            retire_age=torch.zeros(self.cfg.k_max, device=Z.device),
            next_id=len(unique),
        )

    # ------------------------------------------------------------ assignment
    def assign(self, Z: torch.Tensor, st: SectorState) -> torch.Tensor:
        """Soft membership over ACTIVE prototypes only. Rows sum to exactly 1."""
        Zn = F_.normalize(Z, dim=-1, eps=1e-8)
        sim = Zn @ st.prototypes.T                              # cosine, [N, k_max]
        sim = sim.masked_fill(~st.active.unsqueeze(0), -float("inf"))
        return (sim / self.cfg.temperature).softmax(dim=-1)

    @staticmethod
    def attribution(q: torch.Tensor, n_roots: int) -> torch.Tensor:
        """a[i, r, k] with sum_k a = 1 for every (cell, root).

        Attribution follows membership, so it renormalises automatically when
        the active set changes -- no evidence is created or destroyed by a
        sector operation.
        """
        return q.unsqueeze(1).expand(q.shape[0], n_roots, q.shape[1]).contiguous()

    @staticmethod
    def credited_evidence(rho: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """sum_k rho_ir a_irk -- must equal rho_ir identically."""
        return (rho.unsqueeze(-1) * a).sum(-1)

    # ------------------------------------------------------- prototype update
    def update_prototypes(self, Z: torch.Tensor, q: torch.Tensor, st: SectorState,
                          eligible: Optional[torch.Tensor] = None):
        """Move each active prototype to the membership-weighted mean of its cells.

        Without this the controller is not DP-means at all: the initial
        prototype stays pinned at the JOINT mean of every cluster, so cells stay
        beyond the birth threshold and it births indefinitely, while converging
        clusters never bring their prototypes together so merges never fire.
        """
        Zn = F_.normalize(Z.detach(), dim=-1, eps=1e-8)
        if self.cfg.hard_prototype_update:
            hard = q.argmax(-1)
            w = F_.one_hot(hard, self.cfg.k_max).to(Zn.dtype)
            w = w * st.active.unsqueeze(0).to(Zn.dtype)
        else:
            w = q * st.active.unsqueeze(0).float()
        if eligible is not None:
            w = w * eligible.unsqueeze(-1).to(w.dtype)
        w = w.detach()
        mass = w.sum(dim=0)                                       # [k_max]
        mean = torch.einsum("nk,nd->kd", w, Zn)
        m = self.cfg.proto_momentum
        for k in torch.nonzero(st.active).flatten().tolist():
            if float(mass[k]) <= 1e-8:
                continue
            tgt = F_.normalize(mean[k] / mass[k], dim=-1, eps=1e-8)
            st.prototypes[k] = F_.normalize(m * st.prototypes[k] + (1 - m) * tgt,
                                            dim=-1, eps=1e-8)

    # ---------------------------------------------------------------- births
    def _propose_birth(self, Z, q, st) -> Optional[int]:
        """The most persistently dissenting cell, by deterministic order."""
        Zn = F_.normalize(Z, dim=-1, eps=1e-8)
        sim = Zn @ st.prototypes.T
        sim = sim.masked_fill(~st.active.unsqueeze(0), -float("inf"))
        dist = 1.0 - sim.max(dim=-1).values                     # cosine distance
        dissenting = dist > self.cfg.birth_distance
        if not bool(dissenting.any()):
            return None
        idx = int(torch.nonzero(dissenting).flatten()[
            int(torch.argmax(dist[dissenting]))]) if False else None
        # deterministic: greatest distance, ties broken by lowest cell index
        cand = torch.nonzero(dissenting).flatten()
        best = cand[int(torch.argmax(dist[cand]))]
        ties = cand[(dist[cand] - dist[best]).abs() < 1e-9]
        return int(ties.min())

    def maybe_birth(self, Z, q, st) -> bool:
        """Persistence is tracked by a GEOMETRIC proposal, not cell index.

        A scalar counter lets unrelated outliers accumulate into a birth, while
        a cell-index counter misses a coherent moving group when a different
        member is most distant on successive asynchronous steps.  We retain an
        EMA proposal direction and advance patience only when the new candidate
        lies in the same vector-space neighbourhood.
        """
        cand = self._propose_birth(Z, q, st)
        if cand is None:
            st.proposal_age.clear()
            st.pending_prototype = None
            st.pending_age = 0
            return False
        zc = F_.normalize(Z[cand].detach(), dim=-1, eps=1e-8)
        Zn = F_.normalize(Z.detach(), dim=-1, eps=1e-8)
        near = (1.0 - Zn @ zc) <= self.cfg.birth_distance
        min_support = max(2, int(np.ceil(self.cfg.birth_support_fraction * len(Z))))
        group_supported = int(near.sum()) >= min_support
        if st.pending_prototype is None:
            st.pending_prototype = F_.normalize(Zn[near].mean(0), dim=-1, eps=1e-8) \
                if group_supported else zc.clone()
            st.pending_age = self.cfg.patience if group_supported else 1
        else:
            distance = float(1.0 - st.pending_prototype @ zc)
            # Birth proposals are allowed to drift within the same prospective
            # Voronoi region; the tighter merge threshold was too strict for
            # evolving neural embeddings and reset the streak every step.
            if distance <= self.cfg.birth_distance:
                st.pending_prototype = F_.normalize(
                    0.5 * st.pending_prototype + 0.5 * zc, dim=-1, eps=1e-8)
                st.pending_age += 1
            else:
                st.pending_prototype, st.pending_age = zc.clone(), 1
        if st.pending_age < self.cfg.patience:
            return False
        free = torch.nonzero(~st.active).flatten()
        if len(free) == 0:
            return False                                         # capacity, not semantics
        slot = int(free.min())
        st.prototypes[slot] = st.pending_prototype
        st.active[slot] = True
        st.sector_id[slot] = st.next_id
        st.next_id += 1
        st.births += 1
        st.proposal_age.clear()
        st.pending_prototype = None
        st.pending_age = 0
        return True

    def _density_components(self, X: torch.Tensor,
                            linked: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        """Connected components at an adaptive, local geometric scale."""
        X = F_.normalize(X.detach(), dim=-1, eps=1e-8)
        n = len(X)
        if n < 2:
            return [torch.arange(n, device=X.device)]
        if linked is None:
            dist = (1.0 - X @ X.T).clamp(min=0.0)
            nearest = (dist + torch.eye(n, device=X.device) * 9.0).min(-1).values
            # The floor only avoids a zero graph for bit-identical points.  It is
            # tighter than the merge criterion and cannot join distinct sectors by
            # itself.
            eps = max(1e-3, float(nearest.median()) * self.cfg.density_scale)
            linked = dist <= eps
        else:
            if linked.shape != (n, n):
                raise ValueError(f"linked must be [{n},{n}], got {tuple(linked.shape)}")
            linked = linked.to(dtype=torch.bool, device=X.device)
            linked = linked | linked.T | torch.eye(
                n, dtype=torch.bool, device=X.device
            )
        unseen = set(range(n))
        components = []
        while unseen:
            start = min(unseen)
            unseen.remove(start)
            stack, members = [start], []
            while stack:
                current = stack.pop()
                members.append(current)
                neighbours = torch.nonzero(linked[current]).flatten().tolist()
                for neighbour in neighbours:
                    if neighbour in unseen:
                        unseen.remove(neighbour)
                        stack.append(neighbour)
            components.append(torch.tensor(sorted(members), device=X.device))
        return components

    def maybe_density_birth(self, Z, q, st,
                            linked: Optional[torch.Tensor] = None,
                            eligible: Optional[torch.Tensor] = None) -> bool:
        """Split a sector containing multiple coherent density components.

        DP-means' distance threshold is retained as the fast path.  This
        fallback addresses its midpoint failure without consulting labels or a
        predetermined sector count.  Proposals are deterministic: greatest
        centroid separation first, then the lowest source-cell index.
        """
        if not self.cfg.density_birth:
            return False
        free = torch.nonzero(~st.active).flatten()
        if len(free) == 0:
            return False
        hard = q.argmax(-1)
        min_support = max(2, int(np.ceil(self.cfg.birth_support_fraction * len(Z))))
        proposals = []
        Zn = F_.normalize(Z.detach(), dim=-1, eps=1e-8)
        for slot in torch.nonzero(st.active).flatten().tolist():
            member_mask = hard == slot
            if eligible is not None:
                member_mask = member_mask & eligible
            member_idx = torch.nonzero(member_mask).flatten()
            if len(member_idx) < 2 * min_support:
                continue
            local_linked = (linked[member_idx][:, member_idx]
                            if linked is not None else None)
            components = self._density_components(Zn[member_idx], local_linked)
            supported = [c for c in components if len(c) >= min_support]
            if len(supported) < 2:
                continue
            for component in supported:
                global_idx = member_idx[component]
                other_mask = torch.ones(len(member_idx), dtype=torch.bool, device=Z.device)
                other_mask[component] = False
                other_idx = member_idx[other_mask]
                if len(other_idx) < min_support:
                    continue
                candidate = F_.normalize(Zn[global_idx].mean(0), dim=-1, eps=1e-8)
                remainder = F_.normalize(Zn[other_idx].mean(0), dim=-1, eps=1e-8)
                separation = float(1.0 - candidate @ remainder)
                if separation <= self.cfg.merge_distance:
                    continue
                # Preserve the established sector identity: the component
                # farthest from the parent prototype receives the new slot,
                # while the component already represented by the parent stays
                # put.  Choosing only by component index caused a label swap
                # and remapped every unrelated cell despite an unchanged
                # partition.
                from_parent = float(1.0 - candidate @ st.prototypes[slot])
                proposals.append((-from_parent, -separation,
                                  int(global_idx.min()), candidate))
        if not proposals:
            return False
        proposals.sort(key=lambda item: (item[0], item[1], item[2]))
        candidate = proposals[0][3]
        slot = int(free.min())
        st.prototypes[slot] = candidate
        st.active[slot] = True
        st.sector_id[slot] = st.next_id
        st.next_id += 1
        st.births += 1
        st.pending_prototype = None
        st.pending_age = 0
        return True

    # ---------------------------------------------------------------- merges
    def maybe_merge(self, Z, q, st) -> bool:
        act = torch.nonzero(st.active).flatten().tolist()
        if len(act) < 2:
            st.merge_age.clear()
            return False
        best = None
        for ai in range(len(act)):
            for bj in range(ai + 1, len(act)):
                i, j = act[ai], act[bj]
                d = float(1.0 - (st.prototypes[i] @ st.prototypes[j]))
                if d < self.cfg.merge_distance and (best is None or d < best[0]):
                    best = (d, i, j)
        if best is None:
            st.merge_age.clear()
            return False
        _, i, j = best
        key = (min(i, j), max(i, j))
        st.merge_age[key] = st.merge_age.get(key, 0) + 1
        if st.merge_age[key] < self.cfg.patience:
            return False
        w_i = float(q[:, i].sum())
        w_j = float(q[:, j].sum())
        keep, drop = (i, j) if (w_i, -i) >= (w_j, -j) else (j, i)   # deterministic
        merged = st.prototypes[keep] * max(w_i, w_j) + st.prototypes[drop] * min(w_i, w_j)
        st.prototypes[keep] = F_.normalize(merged, dim=-1, eps=1e-8)
        st.active[drop] = False
        st.prototypes[drop] = 0.0
        st.merges += 1
        st.merge_age.clear()
        return True

    # -------------------------------------------------------------- retiring
    def maybe_retire(self, q, st) -> bool:
        act = torch.nonzero(st.active).flatten()
        if len(act) < 2:
            return False
        if self.cfg.hard_prototype_update:
            support = F_.one_hot(q.argmax(-1), self.cfg.k_max).float().mean(0)
        else:
            support = q.sum(dim=0) / max(float(q.sum()), 1e-9)
        changed = False
        for k in act.tolist():
            if float(support[k]) < self.cfg.retire_support:
                st.retire_age[k] += 1
                if int(st.retire_age[k]) >= self.cfg.patience and int(st.active.sum()) > 1:
                    st.active[k] = False
                    st.prototypes[k] = 0.0
                    st.retirements += 1
                    changed = True
            else:
                st.retire_age[k] = 0
        return changed

    # ------------------------------------------------------------------ step
    def step(self, Z: torch.Tensor, st: SectorState, allow_ops: bool = True,
             density_linked: Optional[torch.Tensor] = None,
             eligible: Optional[torch.Tensor] = None):
        """One controller step. Deterministic order: assign, birth, merge, retire."""
        allow_ops = allow_ops and self.cfg.dynamic_ops
        q = self.assign(Z, st)
        self.update_prototypes(Z, q, st, eligible)  # DP-means centroid step
        q = self.assign(Z, st)
        if allow_ops:
            born = self.maybe_density_birth(
                Z, q, st, linked=density_linked, eligible=eligible
            )
            if not born and self.cfg.distance_birth:
                born = self.maybe_birth(Z, q, st)
            # A newborn slot did not exist when q was computed.  Passing that
            # stale q to retirement assigns the new sector zero support and can
            # delete it immediately (always when patience=1).
            if born:
                q = self.assign(Z, st)
                self.update_prototypes(Z, q, st, eligible)
                q = self.assign(Z, st)
            merged = self.maybe_merge(Z, q, st)
            if merged:
                q = self.assign(Z, st)
            self.maybe_retire(q, st)
            q = self.assign(Z, st)                # reassign after any change
            self.update_prototypes(Z, q, st, eligible)
            q = self.assign(Z, st)
        return q, st

    def run(self, Z_seq: List[torch.Tensor], st: Optional[SectorState] = None):
        st = st or self.init_state(Z_seq[0])
        qs = []
        for Z in Z_seq:
            q, st = self.step(Z, st)
            qs.append(q)
        return qs, st


def remapping_fraction(q_before: torch.Tensor, q_after: torch.Tensor) -> float:
    """Fraction of cells whose hard assignment changed across a sector op.

    Both arguments are the SOFT membership matrices [N, k_max]; the caller
    restricts `q_after` to the originally-present cells.  Passing already-hardened
    single-column tensors makes every argmax 0 and the function silently reports
    0.0 -- which is what an earlier version of the bounded-remapping test did.
    """
    if q_before.shape[0] != q_after.shape[0]:
        raise ValueError(f"remapping compares the same cells: got "
                         f"{q_before.shape[0]} before and {q_after.shape[0]} after")
    if q_before.dim() != 2 or q_before.shape[1] < 2:
        raise ValueError("remapping_fraction expects soft memberships [N, K>=2]")
    return float((q_before.argmax(-1) != q_after.argmax(-1)).float().mean())


def partition_of(q: torch.Tensor) -> frozenset:
    """Label-invariant partition: the set of co-membership groups."""
    hard = q.argmax(-1)
    groups = {}
    for i, k in enumerate(hard.tolist()):
        groups.setdefault(k, []).append(i)
    return frozenset(frozenset(v) for v in groups.values())
