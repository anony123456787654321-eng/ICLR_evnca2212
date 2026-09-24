from .base import EvidenceAtom, ProvenanceSketch
from .exact import ExactLedger
from .memory_matched import LRUSketch, MembershipSketch, ReservoirSketch
from .theta import ThetaSketch

__all__ = [
    "EvidenceAtom", "ProvenanceSketch", "ExactLedger", "ThetaSketch",
    "ReservoirSketch", "LRUSketch", "MembershipSketch", "make_sketch",
    "split_budget", "theta_budget_bytes",
]


def theta_budget_bytes(payload_dim: int, capacity: int) -> int:
    """Bytes used by the plain (k, k) Theta sketch -- the byte budget to match."""
    return capacity * (8 + 8 * payload_dim + 8) + capacity * 8 + 16


def split_budget(payload_dim: int, capacity: int, payload_fraction: float = 0.75):
    """Split the plain-Theta byte budget into (k_payload, k_hash).

    Relative variance is ~ CV^2 / k_payload + 1 / k_hash, and a hash slot is
    8 bytes against 8*(payload_dim+2) for a payload slot, so trading payload
    slots for hashes is cheap wherever the cardinality term dominates.
    """
    slot = 8 + 8 * payload_dim + 8
    budget = theta_budget_bytes(payload_dim, capacity)
    kp = max(1, int(round(payload_fraction * capacity)))
    kh = max(kp, (budget - 16 - kp * slot) // 8)
    return kp, kh


def make_sketch(kind: str, payload_dim: int, capacity: int = 32, **kw):
    """Factory used by configs so a sweep can name a structure with a string."""
    kind = kind.lower()
    if kind in ("exact", "ledger", "exact_ledger"):
        return ExactLedger(payload_dim)
    if kind in ("theta", "bottomk", "ec"):
        return ThetaSketch(payload_dim, capacity, hash_seed=kw.get("hash_seed", 0),
                           hash_capacity=kw.get("hash_capacity"))
    if kind in ("theta_cons", "ec_theta_cons"):
        return ThetaSketch(payload_dim, capacity, hash_seed=kw.get("hash_seed", 0), rescale=False)
    if kind in ("theta_split", "ec_theta_split"):
        kp, kh = split_budget(payload_dim, capacity, kw.get("payload_fraction", 0.75))
        return ThetaSketch(payload_dim, kp, hash_seed=kw.get("hash_seed", 0), hash_capacity=kh)
    if kind == "reservoir":
        return ReservoirSketch(payload_dim, capacity, rng=kw.get("rng"),
                               hash_seed=kw.get("hash_seed", 0))
    if kind == "lru":
        return LRUSketch(payload_dim, capacity, hash_seed=kw.get("hash_seed", 0))
    if kind == "bloom":
        return MembershipSketch(payload_dim, capacity, n_bits=kw.get("n_bits", 2048),
                                n_hashes=kw.get("n_hashes", 3), backend="bloom",
                                hash_seed=kw.get("hash_seed", 0))
    if kind in ("countmin", "cm"):
        return MembershipSketch(payload_dim, capacity, backend="countmin",
                                cm_width=kw.get("cm_width", 256), cm_depth=kw.get("cm_depth", 3),
                                hash_seed=kw.get("hash_seed", 0))
    raise ValueError(f"unknown sketch kind: {kind}")
