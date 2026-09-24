"""EC-Gaussian benchmark generator with exact lineage ground truth.

An *example* is: a graph of cells, a set of unique root observations, and a
schedule of atom injections in which some atoms are descendants (copies) of a
root that is already present.  Because we build the lineage ourselves, the
unique-evidence posterior is known exactly, which is what makes this a
benchmark rather than a demo.

Regimes
-------
clean        each unique root injected once
duplicate    each root injected r times (exact copies, same root_id)
replacement  r x more *distinct* roots instead of r copies -- the control that
             shows a method is not merely ignoring extra atoms
conflicting  a fraction of roots are generated from a decoy latent and then
             duplicated r times (repetition of wrong evidence)
sybil        each copy is relabelled with a fresh fake root_id: the documented
             failure mode of any lineage-trusting method
corrupt      a fraction of copies lose their root_id and get a random one
transform    the A->B->C->D regime.  Descendant j of a root carries a PARTIAL
             extraction of that root's sufficient statistic -- natural
             parameters scaled by f_j = 1 - exp(-lam (j+1)) -- so every
             descendant is a valid but less-informed readout of the SAME
             observation, converging to it as computation proceeds.  A message
             therefore carries new computation and zero new evidence, which is
             the case that separates the three required behaviours:
               1. more transformations must improve accuracy,
               2. without raising confidence past that one root's information,
               3. while a genuinely new root raises both.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from .gaussian import GaussianSpec, make_observation
from .topology import build


@dataclass
class Atom:
    root_id: str
    payload: np.ndarray
    mass: float
    cell: int
    step: int
    is_descendant: bool = False
    refinement: float = 1.0


@dataclass
class Example:
    spec: GaussianSpec
    x_true: np.ndarray
    adj: List[List[int]]
    atoms: List[Atom]
    unique_payload_sum: np.ndarray      # sum over lineage-distinct roots ONLY
    n_unique: int
    n_atoms: int
    meta: Dict = field(default_factory=dict)

    def injections_by_step(self) -> Dict[int, List[Atom]]:
        out: Dict[int, List[Atom]] = {}
        for a in self.atoms:
            out.setdefault(a.step, []).append(a)
        return out


def generate_example(
    spec: GaussianSpec,
    n_unique: int,
    rng: np.random.Generator,
    topology: str = "grid",
    n_cells: int = 16,
    regime: str = "clean",
    duplication: int = 1,
    inject_window: int = 1,
    conflict_fraction: float = 0.25,
    corrupt_fraction: float = 0.5,
    decoy_scale: float = 3.0,
    refine_rate: float = 0.7,
) -> Example:
    adj = build(topology, n_cells, rng)
    x_true = rng.normal(0.0, 1.0 / np.sqrt(spec.prior_precision), size=spec.dim)
    x_decoy = x_true + decoy_scale * rng.normal(size=spec.dim)

    n_roots = n_unique * duplication if regime == "replacement" else n_unique
    roots: List[Tuple[str, np.ndarray, float]] = []
    for s in range(n_roots):
        conflicting = regime == "conflicting" and rng.random() < conflict_fraction
        src = x_decoy if conflicting else x_true
        payload, mass = make_observation(spec, src, rng)
        roots.append((f"root::{s:06d}", payload, mass))

    # lineage-distinct evidence = one payload per unique root
    unique_sum = np.sum([p for _, p, _ in roots], axis=0) if roots else np.zeros(spec.payload_dim)

    copies = 1 if regime in ("clean", "replacement") else max(1, duplication)
    atoms: List[Atom] = []
    fake = 0
    for root_id, payload, mass in roots:
        for c in range(copies):
            rid = root_id
            descendant = c > 0
            frac = 1.0
            if regime == "transform":
                # the c-th computation over this observation recovers f_c of it
                frac = 1.0 - np.exp(-refine_rate * (c + 1))
            if descendant and regime == "sybil":
                rid, fake = f"sybil::{fake:06d}", fake + 1
            elif descendant and regime == "corrupt" and rng.random() < corrupt_fraction:
                rid, fake = f"lost::{fake:06d}", fake + 1
            atoms.append(Atom(rid, frac * payload, frac * mass,
                              cell=int(rng.integers(0, n_cells)),
                              step=int(rng.integers(0, max(1, inject_window))),
                              is_descendant=descendant, refinement=frac))
    rng.shuffle(atoms)
    return Example(spec, x_true, adj, atoms, unique_sum, n_roots, len(atoms),
                   meta=dict(topology=topology, n_cells=n_cells, regime=regime,
                             duplication=duplication, copies=copies))
