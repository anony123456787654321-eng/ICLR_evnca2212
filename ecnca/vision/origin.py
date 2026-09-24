"""Evidence-origin memory: learned provenance beside ordinary content.

The measured problem this must solve, from the seven-arm audit on the frozen
reference at 50% duplication:

    fresh                    cell 0.9492
    cached_duplicate         cell 0.3141   <- exact repeats are catastrophic
    transformed_same_origin  cell 0.9421   <- same origin, transformed: harmless
    exact_duplicate_filter   cell 0.4695   <- damping the whole message: 24% recovered

So the damage is **exact-repeat degeneracy**, not same-origin information as
such. A mechanism that suppressed same-origin credit in general would discard
something costing 0.7 points while missing the thing costing 63.5.

Why a similarity threshold cannot do it. Measured on an untrained head, the
origin-signature cosine against a previous message is 1.0000 for an exact
repeat, 0.9996 for a transformed same-origin message, and 0.9459 for a new
origin. The two cases that differ by 63 accuracy points are 0.0004 apart in
cosine -- so any fixed threshold either suppresses both or neither. The
distinction has to be **learned**, which is what ``OriginMemory`` below does.

Design, against the plan's requirements:

* content and credit are separate. ``novelty`` scales only the evidential
  channels (the logits); hidden content passes through unscaled, so repeated
  information may still refine.
* the deployed decision runs through the learned representation. Ground-truth
  provenance supervises the auxiliary head during training and audits it
  afterwards, but never enters the forward pass.
* three-way, not binary: repeat / transformed-same-origin / new-origin, because
  the middle case must be treated as safe rather than suppressed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Provenance classes used ONLY as auxiliary training targets and for auditing.
PROVENANCE = ("exact_repeat", "transformed_same_origin", "new_origin")
EXACT_REPEAT, TRANSFORMED, NEW_ORIGIN = range(3)


@dataclass(frozen=True)
class OriginConfig:
    signature_dim: int = 8
    hidden: int = 32
    credit_floor: float = 0.0    # credit granted to a detected exact repeat
    credit_ceiling: float = 1.0
    aux_weight: float = 0.3      # weight of the provenance auxiliary loss

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def channels(self) -> int:
        """Signature plus one accumulated-credit scalar."""
        return self.signature_dim + 1


class OriginMemory(nn.Module):
    """Learned origin signature, novelty estimate, and provenance auditing.

    Inputs are the current message and the cell's stored signature. Outputs are
    a new signature, a scalar novelty in [0,1] that scales evidential credit,
    and provenance logits used only for the auxiliary loss and the audit.
    """

    def __init__(self, message_channels: int, config: OriginConfig | None = None):
        super().__init__()
        cfg = config or OriginConfig()
        self.config = cfg
        d = cfg.signature_dim

        self.signature = nn.Conv2d(message_channels, d, 1)
        # The comparison head sees the pair plus explicit difference features,
        # because an exact repeat and a transformed repeat differ by 0.0004 in
        # cosine -- a raw concatenation makes that nearly invisible, while the
        # elementwise difference and its magnitude make it linearly available.
        self.compare = nn.Sequential(
            nn.Conv2d(4 * d + 2, cfg.hidden, 1), nn.ReLU(),
            nn.Conv2d(cfg.hidden, cfg.hidden, 1), nn.ReLU(),
        )
        self.novelty = nn.Conv2d(cfg.hidden, 1, 1)
        self.provenance = nn.Conv2d(cfg.hidden, len(PROVENANCE), 1)

    def encode(self, message: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.signature(message))

    def _features(self, sig: torch.Tensor, prev: torch.Tensor) -> torch.Tensor:
        diff = sig - prev
        prod = sig * prev
        # Difference magnitude and cosine, computed explicitly: an exact repeat
        # has diff exactly zero, which is the one cue that cleanly separates it
        # from a transformed repeat.
        mag = torch.linalg.vector_norm(diff, dim=1, keepdim=True)
        cos = F.cosine_similarity(sig, prev, dim=1).unsqueeze(1)
        return torch.cat([sig, prev, diff, prod, mag, cos], dim=1)

    def forward(self, message: torch.Tensor, prev_signature: torch.Tensor):
        """Returns (signature, novelty, provenance_logits)."""
        sig = self.encode(message)
        h = self.compare(self._features(sig, prev_signature))
        cfg = self.config
        novelty = torch.sigmoid(self.novelty(h))
        novelty = cfg.credit_floor + (cfg.credit_ceiling - cfg.credit_floor) * novelty
        return sig, novelty, self.provenance(h)


def provenance_targets(
    message: torch.Tensor, previous: torch.Tensor, *, tol: float = 1e-6
) -> torch.Tensor:
    """Ground-truth provenance for the auxiliary loss. TRAINING ONLY.

    Derived from the messages themselves, never from a supplied identifier:
    exact equality is checkable, and 'same origin' is established by
    construction in the intervention matrix. Shape (B,H,W) of class indices.
    """
    if message.shape != previous.shape:
        raise ValueError("message and previous must have the same shape")
    same = (message - previous).abs().amax(dim=1) <= tol
    return torch.where(
        same,
        torch.full(same.shape, EXACT_REPEAT, device=message.device, dtype=torch.long),
        torch.full(same.shape, NEW_ORIGIN, device=message.device, dtype=torch.long),
    )


def auxiliary_loss(provenance_logits: torch.Tensor, targets: torch.Tensor,
                   weight: float = 1.0) -> torch.Tensor:
    """Cross-entropy on the provenance head. Supervises, never gates."""
    return weight * F.cross_entropy(provenance_logits, targets)


def audit_separation(memory: OriginMemory, message: torch.Tensor,
                     transformed: torch.Tensor, new: torch.Tensor) -> dict:
    """Does the learned novelty separate the three cases the audit measured?

    This is the check that decides whether the mechanism can work at all: the
    exact repeat must receive markedly less credit than the transformed
    same-origin message, which the reference showed is nearly harmless.
    """
    with torch.no_grad():
        sig = memory.encode(message)
        out = {}
        for name, other in (("exact_repeat", message),
                            ("transformed_same_origin", transformed),
                            ("new_origin", new)):
            s2, nov, prov = memory(other, sig)
            out[name] = {
                "novelty": float(nov.mean()),
                "signature_cosine": float(
                    F.cosine_similarity(s2, sig, dim=1).mean()
                ),
                "predicted_provenance": int(prov.mean(dim=(0, 2, 3)).argmax()),
            }
    out["repeat_vs_transformed_gap"] = (
        out["transformed_same_origin"]["novelty"] - out["exact_repeat"]["novelty"]
    )
    out["separates"] = out["repeat_vs_transformed_gap"] > 0.2
    return out


# --- bounded multi-origin memory -----------------------------------------
#
# The previous component compared only against the immediately preceding
# signature, so it could not recognise A -> B -> A: B overwrote A. A genuine
# origin memory retains a bounded SET of previously credited signatures.
#
# Three properties this must have, each tested:
#   * permutation invariance of the memory aggregation -- slot order carries no
#     meaning, so reordering the retained set must not change the decision;
#   * bounded credit -- a defined ceiling, not an unbounded running sum. The
#     earlier `credit = credit + novelty` reached 14,021 after 100 steps;
#   * idempotence -- re-delivering an already-credited origin must not increase
#     total credit, however many times it arrives.

@dataclass(frozen=True)
class MemoryConfig:
    slots: int = 4               # bounded storage per cell
    signature_dim: int = 8
    hidden: int = 32
    credit_ceiling: float = 1.0  # total credit a cell can ever hold
    # Novelty above which a signature is stored. Deliberately BELOW 0.5: an
    # untrained sigmoid outputs exactly 0.5, and with a strict `>` test at 0.5
    # nothing was ever written, so the memory stayed permanently empty and the
    # pair scorer received no gradient at all. With a valid slot present the
    # same weights get healthy gradient (measured 2.6e+02), so this threshold
    # was the whole difference between a live module and a dead one.
    write_threshold: float = 0.3
    decay: float = 0.999         # slow forgetting, so stale origins age out


class BoundedOriginMemory(nn.Module):
    """A small per-cell set of credited origin signatures.

    Recognition compares an incoming signature against the WHOLE retained set,
    aggregated permutation-invariantly (max over slots of a shared pairwise
    scorer -- a DeepSets-style symmetric reduction). Credit is a bounded
    quantity with an explicit ceiling, not accumulated logits.
    """

    def __init__(self, message_channels: int, config: MemoryConfig | None = None):
        super().__init__()
        cfg = config or MemoryConfig()
        self.config = cfg
        d = cfg.signature_dim
        self.signature = nn.Conv2d(message_channels, d, 1)
        # Scores ONE (incoming, stored) pair. Applied to every slot with shared
        # weights and reduced symmetrically, so the aggregation cannot depend
        # on slot order.
        self.pair = nn.Sequential(
            nn.Conv2d(4 * d + 2, cfg.hidden, 1), nn.ReLU(),
            nn.Conv2d(cfg.hidden, 1, 1),
        )
        self.provenance = nn.Sequential(
            nn.Conv2d(cfg.hidden, cfg.hidden, 1), nn.ReLU(),
            nn.Conv2d(cfg.hidden, len(PROVENANCE), 1),
        )
        self._hidden_cache: torch.Tensor | None = None

    @property
    def channels(self) -> int:
        """State channels: the slot signatures, their validity, and credit."""
        return self.config.slots * (self.config.signature_dim + 1) + 1

    def encode(self, message: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.signature(message))

    def _pair_features(self, sig: torch.Tensor, stored: torch.Tensor):
        diff = sig - stored
        mag = torch.linalg.vector_norm(diff, dim=1, keepdim=True)
        cos = F.cosine_similarity(sig, stored, dim=1).unsqueeze(1)
        return torch.cat([sig, stored, diff, sig * stored, mag, cos], dim=1)

    def match(self, sig: torch.Tensor, memory: torch.Tensor,
              valid: torch.Tensor):
        """Compare against EVERY retained signature. (B,M,D,H,W) memory.

        Returns (familiarity, hidden) where familiarity is in [0,1] and high
        means "this origin has already been credited".
        """
        b, m, d, h, w = memory.shape
        scores, hiddens = [], []
        for i in range(m):
            feats = self._pair_features(sig, memory[:, i])
            hid = self.pair[1](self.pair[0](feats))     # conv -> relu
            scores.append(self.pair[2](hid))
            hiddens.append(hid)
        s = torch.stack(scores, dim=1)                  # (B,M,1,H,W)
        # Invalid slots must not win the max.
        s = s.masked_fill(~valid.unsqueeze(2).bool(), float("-inf"))
        empty = (~valid.bool()).all(dim=1, keepdim=True).unsqueeze(2)
        s = torch.where(empty.expand_as(s), torch.zeros_like(s), s)
        best = s.amax(dim=1)                            # symmetric reduction
        idx = s.argmax(dim=1)
        # Select the winning slot's hidden features as a one-hot masked sum
        # rather than gather(). They are mathematically identical here --
        # argmax gives exactly one index per position over a disjoint slot
        # dimension -- and verified bit-identical in both forward and
        # gradient. gather's backward dispatches to scatter_add_, which has no
        # deterministic CUDA implementation, so under
        # torch.use_deterministic_algorithms(True) this gradient path would
        # either raise or silently vary between runs. The scatter_ below
        # builds a constant mask outside the autograd graph, so no
        # nondeterministic op remains in the backward pass.
        stacked = torch.stack(hiddens, dim=1)
        onehot = torch.zeros_like(s)
        onehot.scatter_(1, idx.unsqueeze(1), 1.0)
        hid = (stacked * onehot).sum(dim=1)
        return torch.sigmoid(best), hid

    def forward(self, message: torch.Tensor, memory: torch.Tensor,
                valid: torch.Tensor, credit: torch.Tensor):
        """One step. Returns (signature, novelty, provenance, memory, valid, credit)."""
        cfg = self.config
        sig = self.encode(message)
        familiarity, hid = self.match(sig, memory, valid)
        novelty = 1.0 - familiarity
        prov = self.provenance(hid)

        # Credit is BOUNDED **and STRUCTURALLY IDEMPOTENT**.
        #
        # The earlier scheme closed a fraction of the remaining headroom on
        # every delivery, so repeated small novelty still crept to the ceiling
        # (measured 0.4995 -> 0.9980 over 21 identical deliveries). Boundedness
        # alone is not the requirement: re-delivering an already-credited
        # origin must add NOTHING, and that must hold for an untrained model
        # too, not only once the provenance head has learned.
        #
        # So credit is a MAX over deliveries, not an accumulation:
        #
        #     credit <- max(credit, novelty)
        #
        # A familiar origin has novelty at or below what it already
        # contributed, so the max leaves credit untouched however many times it
        # arrives -- idempotent by construction. A genuinely new origin has
        # higher novelty and raises credit. The ceiling is then automatic,
        # because novelty is a sigmoid in [0, 1].
        #
        # `decay` is applied only when nothing new arrived, so a stale origin
        # ages out without penalising a cell that is still receiving evidence.
        raw_credit = torch.maximum(credit, novelty * cfg.credit_ceiling)
        stale = (novelty <= credit).to(credit.dtype)
        credit = raw_credit * (stale * cfg.decay + (1.0 - stale))
        credit = credit.clamp(max=cfg.credit_ceiling)

        # Write genuinely new origins only. A RECOGNISED repeat must not shift
        # the ring: an unconditional shift evicted the oldest legitimate origin
        # every time a familiar message arrived, which is exactly the memory
        # the bounded set exists to keep.
        # An empty memory must always accept its first observation, whatever
        # the threshold: otherwise recognition can never begin.
        empty = (valid.sum(dim=1, keepdim=True) == 0)
        write_b = (novelty > cfg.write_threshold) | empty        # (B,1,H,W) bool
        write = write_b.to(sig.dtype)
        w5 = write_b.unsqueeze(1)                                # (B,1,1,H,W)
        shifted_mem = torch.cat([sig.unsqueeze(1), memory[:, :-1]], dim=1)
        shifted_val = torch.cat([write, valid[:, :-1]], dim=1)
        memory = torch.where(w5.expand_as(memory), shifted_mem, memory)
        valid = torch.where(write_b.expand_as(valid), shifted_val, valid)
        return sig, novelty, prov, memory, valid, credit


@torch.no_grad()
def idempotence_error(memory: "BoundedOriginMemory", message: torch.Tensor,
                      *, repeats: int = 20) -> dict:
    """How much credit leaks when one origin is delivered repeatedly?

    Ideal is zero growth after the first delivery. This measures the gap rather
    than asserting the property, because idempotence is trained (via the
    provenance auxiliary), not structural.
    """
    cfg = memory.config
    b, _, h, w = message.shape
    dev = message.device
    mem = torch.zeros(b, cfg.slots, cfg.signature_dim, h, w, device=dev)
    valid = torch.zeros(b, cfg.slots, h, w, device=dev)
    credit = torch.zeros(b, 1, h, w, device=dev)

    _, _, _, mem, valid, credit = memory(message, mem, valid, credit)
    after_first = float(credit.mean())
    novelties = []
    for _ in range(repeats):
        _, nov, _, mem, valid, credit = memory(message, mem, valid, credit)
        novelties.append(float(nov.mean()))
    after_repeats = float(credit.mean())
    return {
        "credit_after_first": after_first,
        "credit_after_repeats": after_repeats,
        "leak": after_repeats - after_first,
        "mean_novelty_on_repeats": float(np.mean(novelties)) if novelties else None,
        "bounded": after_repeats <= cfg.credit_ceiling + 1e-6,
        "ideal": 0.0,
        "note": (
            "Boundedness is structural; idempotence is trained. A nonzero leak "
            "on an untrained model is expected -- report it, do not hide it."
        ),
    }


@torch.no_grad()
def recognises_cycle(memory: "BoundedOriginMemory", a: torch.Tensor,
                     b: torch.Tensor) -> dict:
    """A -> B -> A: is the second A recognised as already credited?

    A last-signature gate cannot do this, because B overwrote A. This is the
    property bounded multi-origin memory exists to provide.
    """
    cfg = memory.config
    n, _, h, w = a.shape
    dev = a.device
    mem = torch.zeros(n, cfg.slots, cfg.signature_dim, h, w, device=dev)
    valid = torch.zeros(n, cfg.slots, h, w, device=dev)
    credit = torch.zeros(n, 1, h, w, device=dev)

    _, nov_a1, _, mem, valid, credit = memory(a, mem, valid, credit)
    _, nov_b, _, mem, valid, credit = memory(b, mem, valid, credit)
    _, nov_a2, _, mem, valid, credit = memory(a, mem, valid, credit)
    # Is A's signature actually still in the retained set? That is the
    # STRUCTURAL property bounded memory provides; whether the scorer then
    # ranks it highest is a TRAINED one. Separating them matters: an untrained
    # pair scorer gives B a higher score than A, so the max-reduction picks the
    # wrong slot and novelty reads high even though A was never forgotten
    # (measured margin about -0.058 across five seeds).
    cos = F.cosine_similarity(
        mem, memory.encode(a).unsqueeze(1), dim=2
    ).amax(dim=1)
    retained = float(cos.max()) > 0.99
    return {
        "novelty_first_A": float(nov_a1.mean()),
        "novelty_B": float(nov_b.mean()),
        "novelty_second_A": float(nov_a2.mean()),
        "signature_retained": retained,
        "best_cosine_to_A": float(cos.max()),
        "recognised": float(nov_a2.mean()) < float(nov_a1.mean()),
        "margin": float(nov_a1.mean() - nov_a2.mean()),
        "note": (
            "`signature_retained` is structural: bounded multi-origin memory "
            "keeps A across the intervening B, which a last-signature gate "
            "cannot. `recognised` is trained: it additionally requires the "
            "pair scorer to rank the retained A above the alternatives, which "
            "the provenance auxiliary supplies."
        ),
    }
