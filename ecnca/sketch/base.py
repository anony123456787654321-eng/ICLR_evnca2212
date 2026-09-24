"""Common interface for provenance structures.

A provenance structure is the *evidence ledger* half of a cell's state.  It is
deliberately separated from the learned workspace: the workspace may keep
computing forever, but the quantities returned by ``estimate_payload`` and
``estimate_mass`` can only grow when a lineage-distinct root arrives.

All structures share one message format, which is what makes the memory-matched
comparison fair:

    (retained atoms, up to ``capacity``) + (auxiliary bits) + (scalar corrections)

They differ only in (a) *which* atoms survive eviction, (b) whether an evicted
root can be re-counted later, and (c) how the discarded mass is estimated back.
"""
from __future__ import annotations

import abc
import hashlib
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


def version_key(refinement: float, payload: np.ndarray, mass: float = 0.0):
    """Total order on the versions of ONE root.

    A message can carry a new *computation* without carrying new *evidence*.
    When two messages descend from the same root, the ledger keeps the more
    refined one instead of the first one to arrive -- a max-register join.  The
    payload hash breaks ties so the order is total, which is what keeps the join
    commutative and associative (and therefore order- and cycle-invariant).

    Crucially this changes only WHICH version of a root is held, never HOW MANY
    roots are counted: refinement moves the belief, lineage bounds the
    confidence.

    Order: (refinement, -mass, payload hash).  Refinement is therefore the ONLY
    channel through which a message can raise the information credited to a
    root.  Two messages claiming equal refinement are equally well-informed
    about the same observation, so the ledger keeps the one claiming LESS
    information -- a re-delivered message can move the belief but can never
    inflate confidence, whatever payload it carries.  The payload hash then makes
    the order total, which is what keeps the join commutative and associative.
    """
    digest = hashlib.blake2b(np.asarray(payload, np.float64).tobytes(), digest_size=8).digest()
    return (float(refinement), -float(mass), int.from_bytes(digest, "little"))


@dataclass(frozen=True)
class EvidenceAtom:
    """One lineage-distinct observation, as it travels through the grid."""

    root_id: str
    payload: np.ndarray          # additive natural-parameter contribution
    mass: float                  # scalar information weight (e.g. trace of Lambda)
    refinement: float = 0.0      # how much of the root's information was extracted
    example_id: int = -1
    cell_id: int = -1
    arrival_step: int = -1


class ProvenanceSketch(abc.ABC):
    """Mergeable summary of the set of roots a cell has evidence from."""

    capacity: int
    payload_dim: int

    # --- mutation -----------------------------------------------------------
    @abc.abstractmethod
    def insert(self, root_id: str, payload: np.ndarray, mass: float,
               refinement: float = 0.0) -> None:
        """Absorb a single atom.

        Idempotent in ``root_id`` for *counting*; within a root, the higher
        ``refinement`` wins, so a better computation over the same observation
        replaces a worse one without adding evidence.
        """

    @abc.abstractmethod
    def merge(self, other: "ProvenanceSketch") -> "ProvenanceSketch":
        """Return a NEW sketch equal to the union of ``self`` and ``other``."""

    @abc.abstractmethod
    def copy(self) -> "ProvenanceSketch":
        ...

    # --- estimation ---------------------------------------------------------
    @abc.abstractmethod
    def estimate_payload(self) -> np.ndarray:
        """Unbiased estimate of the sum of payloads over all distinct roots."""

    @abc.abstractmethod
    def estimate_mass(self) -> float:
        ...

    @abc.abstractmethod
    def estimate_count(self) -> float:
        """Estimate of the number of distinct roots absorbed."""

    def estimate_payload_cov(self) -> np.ndarray:
        """Sampling covariance of ``estimate_payload``.

        An unbiased estimate of the payload SUM is not enough to report a
        calibrated belief: the cell would then state the confidence of n roots
        around a mean computed from k of them.  This returns the estimator's own
        variance so the reported covariance can absorb it.

        The default is zero, which is correct for an exact ledger and *honest*
        for structures whose retention is deterministic (LRU, Bloom, count-min):
        those do not rescale, so there is no rescaling variance -- but equally,
        no unbiased variance estimate exists for a non-random sample, so we do
        not invent one.
        """
        return np.zeros((self.payload_dim, self.payload_dim))

    # --- accounting ---------------------------------------------------------
    @abc.abstractmethod
    def n_bytes(self) -> int:
        """Bytes of cell state, used for the memory-matched Pareto frontier."""

    def message_bytes(self) -> int:
        """Bytes on the wire per edge-step (identical to state here)."""
        return self.n_bytes()

    # --- helpers ------------------------------------------------------------
    def _zero(self) -> np.ndarray:
        return np.zeros(self.payload_dim, dtype=np.float64)

    def __eq__(self, other: object) -> bool:  # structural equality, for ACI tests
        if not isinstance(other, ProvenanceSketch):
            return NotImplemented
        return self.signature() == other.signature()

    @abc.abstractmethod
    def signature(self):
        """Hashable canonical form; two sketches are equal iff signatures match."""


def _sig_entries(entries: Dict[str, tuple]) -> tuple:
    out = []
    for root, (payload, mass) in sorted(entries.items()):
        out.append((root, tuple(np.round(np.asarray(payload), 12).tolist()), round(float(mass), 12)))
    return tuple(out)
