"""Duplicated-hop interventions on MuSiQue transport chains.

The question this answers is whether evidence conservation is merely free or
actually protective. A clean-trained transport model is evaluated out of
distribution on chains whose supporting hops are re-delivered, re-chunked,
paraphrased, reordered or circulated. Nothing here trains anything.

Two objects do the work.

`DeliveryEvent` is one arrival: a hop index, a question/paragraph pair, a
lineage `root_id` and a `version`. Lineage identity is the identity of the
supporting paragraph, so an exact copy, an overlapping chunk and a paraphrase of
the same paragraph all share one `root_id`. Only the `version` differs, and only
a strictly higher version may replace an incumbent.

`resolve_stream` applies the ledger at delivery time. Under evidence
conservation an arrival that does not beat the incumbent under the version key
is dropped before it reaches the content path, which is what makes exact
redelivery a no-op rather than a second application of a private
transformation. Provenance-free accounting has no ledger, so every arrival is
processed and every arrival adds credit.

Duplicated arrivals of hop j carry hop j's positional index, not a new one, so
an expanded stream needs no new positional parameters and no retraining.
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import torch

from .musique import MuSiQueRecord
from .near_duplicate import DETECTORS, NearDuplicateIndex

INTERVENTIONS = ("exact", "overlap", "paraphrase", "reorder", "cycle")
# `rechunk` replaces the `overlap` construction for new runs. It is kept out of
# INTERVENTIONS so that the frozen sweeps, whose reports enumerate exactly the
# five names above, are neither extended nor re-labelled.
# `fragments` is the same re-chunking with the paragraph itself withheld: only
# distinct windows cut from it arrive, so no complete rendering is available.
EXTRA_INTERVENTIONS = ("rechunk", "fragments")
MULTIPLICITIES = (1, 2, 4, 8, 16)
SCOPES = ("one_hop", "all_hops")


def _hash(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest()


@dataclass(frozen=True)
class DeliveryEvent:
    """One arrival at the transport chain."""
    hop: int
    question: str
    paragraph: str
    root_id: str
    version: int

    @property
    def payload_hash(self) -> str:
        return _hash(self.paragraph)


def base_stream(record: MuSiQueRecord) -> list[DeliveryEvent]:
    """The undisturbed chain: one arrival per hop, version 0."""
    events = []
    for hop, step in enumerate(record.steps):
        paragraph = step.paragraph_title + ". " + step.paragraph_text
        events.append(DeliveryEvent(
            hop=hop, question=step.question, paragraph=paragraph,
            # Lineage identity is the supporting paragraph, not the arrival.
            root_id=f"{record.record_id}:p{step.paragraph_idx}",
            version=0))
    return events


def _target_hops(record: MuSiQueRecord, scope: str) -> list[int]:
    if scope == "all_hops":
        return list(range(record.n_hops))
    if scope == "one_hop":
        # The terminal hop is the one the readout depends on most directly.
        return [record.n_hops - 1]
    raise ValueError(scope)


def partial_windows(text: str, count: int, overlap: float = 0.5) -> list[str]:
    """The construction behind the frozen `overlap` sweeps, kept verbatim.

    It yields at most three distinct windows of a paragraph, each a partial
    span, and never the whole paragraph, so from fourfold multiplicity onward
    every arm sees the same three windows. `overlapping_views` replaces it for
    new runs; this copy exists only so the frozen outputs stay reproducible.
    """
    words = text.split()
    if count <= 1 or len(words) < 4:
        return [text] * max(count, 1)
    width = max(4, int(len(words) * 0.7))
    stride = max(1, int(width * (1.0 - overlap)))
    views, start = [], 0
    while len(views) < count:
        span = words[start:start + width]
        if len(span) < 4:
            span = words[-width:]
        views.append(" ".join(span))
        start += stride
        if start + 4 > len(words):
            start = 0
    return views[:count]


def _spread(last: int) -> list[int]:
    """0..last in an order that halves the largest gap first."""
    order, seen = [], set()

    def add(x):
        if x not in seen:
            seen.add(x)
            order.append(x)
    add(0)
    add(last)
    intervals = [(0, last)]
    while intervals:
        nxt = []
        for lo, hi in intervals:
            if hi - lo > 1:
                mid = (lo + hi) // 2
                add(mid)
                nxt += [(lo, mid), (mid, hi)]
        intervals = nxt
    return order


def overlapping_views(text: str, count: int, width: float = 0.7) -> list[str]:
    """`count` distinct renderings of one paragraph, the paragraph first.

    The first view is the paragraph itself, byte for byte, so the original
    delivery is part of every re-chunked stream. The rest are distinct
    contiguous windows cut from it, widths nearest `width` of the paragraph
    first and offsets spread across it, so no view adds a fact the paragraph
    does not contain. The order does not depend on `count`, so the views at a
    lower multiplicity are a prefix of those at a higher one. A paragraph too
    short to yield `count` distinct spans repeats its views, which does not
    arise on MuSiQue, whose shortest supporting paragraph has 21 words.
    """
    if count <= 1:
        return [text]
    words = text.split()
    n = len(words)
    views, seen = [text], {text}
    nominal = max(1, round(width * n))
    widths = sorted(range(min(4, max(n - 1, 1)), n),
                    key=lambda w: (abs(w - nominal), -w))
    for w in widths:
        for start in _spread(n - w):
            view = " ".join(words[start:start + w])
            if view not in seen:
                seen.add(view)
                views.append(view)
                if len(views) == count:
                    return views
    return [views[i % len(views)] for i in range(count)]


def window_version(paragraph: str, view: str, version: int) -> int:
    """Refinement level of a view cut from a paragraph held at `version`.

    A contiguous window carries a subset of its paragraph's content, so it is a
    less complete rendering of the same observation and ranks below it, by the
    number of words it omits. The paragraph itself keeps its version. Under the
    max-register join a window can therefore never displace the whole
    paragraph, while credit, which counts identifiers, is unaffected.
    """
    return version - (len(paragraph.split()) - len(view.split()))


def build_stream(record: MuSiQueRecord, intervention: str, multiplicity: int,
                 scope: str = "one_hop", seed: int = 0,
                 paraphrases: dict[str, list[str]] | None = None
                 ) -> list[DeliveryEvent]:
    """Expand the clean chain under one intervention at one multiplicity."""
    if intervention not in INTERVENTIONS + EXTRA_INTERVENTIONS:
        raise ValueError(intervention)
    base = base_stream(record)
    if multiplicity <= 1:
        return list(base)
    targets = set(_target_hops(record, scope))
    rng = random.Random((seed, record.record_id, intervention, multiplicity,
                         scope).__hash__() & 0xFFFFFFFF)

    expanded: list[DeliveryEvent] = []
    for event in base:
        if event.hop not in targets:
            expanded.append(event)
            continue
        if intervention in ("exact", "reorder", "cycle"):
            # Byte-identical copies: same payload, same root, same version.
            expanded.extend([event] * multiplicity)
        elif intervention == "rechunk":
            # The paragraph once in full, then distinct windows cut from it,
            # all under its root. Each window ranks below the paragraph by the
            # words it omits, so the ledger holds the whole paragraph.
            views = overlapping_views(event.paragraph, multiplicity)
            expanded.extend(
                DeliveryEvent(event.hop, event.question, view, event.root_id,
                              window_version(event.paragraph, view,
                                             event.version))
                for view in views)
        elif intervention == "fragments":
            # The windows of `rechunk` without the paragraph they are cut from,
            # so every arm, the ledger included, can hold only a fragment.
            views = overlapping_views(event.paragraph, multiplicity + 1)[1:]
            expanded.extend(
                DeliveryEvent(event.hop, event.question, view, event.root_id,
                              window_version(event.paragraph, view,
                                             event.version))
                for view in views)
        elif intervention == "overlap":
            views = partial_windows(event.paragraph, multiplicity)
            # Same root and the SAME refinement level. An overlapping chunk is
            # an alternative rendering of one paragraph, not an improvement on
            # it, so treating it as a later version would let re-chunking pass
            # for refinement. The ledger therefore keeps one view by the hash
            # tie-break, deterministically and independently of arrival order.
            expanded.extend(
                DeliveryEvent(event.hop, event.question, view, event.root_id, 0)
                for view in views)
        elif intervention == "paraphrase":
            pool = (paraphrases or {}).get(event.root_id) or []
            if not pool:
                # No audited paraphrase available: fall back to exact copies and
                # let the caller's eligibility funnel record the shortfall.
                expanded.extend([event] * multiplicity)
            else:
                chosen = [event.paragraph] + [pool[i % len(pool)]
                                              for i in range(multiplicity - 1)]
                # A paraphrase is an alternative rendering, not a refinement, so
                # it shares the root AND the version with the original.
                expanded.extend(
                    DeliveryEvent(event.hop, event.question, text,
                                  event.root_id, 0)
                    for text in chosen)

    if intervention == "reorder":
        # Same multiset of arrivals, different order. Hop order is preserved
        # between hops; only arrivals within a hop are permuted, so the chain
        # itself is not scrambled.
        by_hop: dict[int, list[DeliveryEvent]] = {}
        for e in expanded:
            by_hop.setdefault(e.hop, []).append(e)
        expanded = []
        for hop in sorted(by_hop):
            group = by_hop[hop][:]
            rng.shuffle(group)
            expanded.extend(group)
    elif intervention == "cycle":
        # Circulate the whole chain: the same evidence returns by another path.
        rounds = max(1, multiplicity // max(len(base), 1))
        expanded = list(base) * max(rounds, 1)
        if multiplicity > 1 and len(expanded) == len(base):
            expanded = list(base) * 2
    return expanded


def version_key(event: DeliveryEvent) -> tuple:
    """kappa = (refinement level, payload hash). Higher wins."""
    return (event.version, event.payload_hash)


# Fusion rules, as opposed to ledgers and duplicate detectors. Neither
# suppresses an arrival; both admit the whole stream and differ only in how much
# source-level credit the fused estimate implies. They are accounting controls
# on one content path, in the same sense as `plain` and `no_provenance`, so any
# difference between them is the accounting rule and nothing else.
FUSION_MODES = ("dempster", "covariance_intersection")

# Degree of support carried by one delivery, for the Dempster arm. Frozen at the
# maximally non-committal value before any evaluation was run and not tuned:
# 0.5 is even odds for a simple support function. Dempster credit saturates at
# 1 / DEMPSTER_MASS, so this constant sets the ceiling of that arm and is
# reported rather than fitted.
DEMPSTER_MASS = 0.5


def dempster_credit(count: int, mass: float = DEMPSTER_MASS) -> float:
    """Credit implied by combining `count` identical simple support functions.

    Dempster's rule of combination is commutative and associative but NOT
    idempotent, so combining one source with itself sharpens the belief instead
    of leaving it unchanged. For a simple support function carrying mass `m` on
    a proposition and `1 - m` on the frame, combining `count` copies gives
    belief `1 - (1 - m) ** count`. Expressed in units of a single delivery's
    belief that is `(1 - (1 - m) ** count) / m`, which is `1.0` at one delivery
    and rises to a ceiling of `1 / m`.

    The resulting curve is the reason this arm is distinct from `none`: credit
    inflates under repetition as a provenance-free method does, but saturates
    rather than growing without bound.
    """
    if count <= 0:
        return 0.0
    return (1.0 - (1.0 - mass) ** count) / mass


# `minhash`, `simhash` and `embed` are near-duplicate defences rather than
# ledgers: they suppress payloads that look alike. They are included so the
# comparison is against the strong form of deduplication and not only its
# weakest, exact form.
LEDGER_MODES = (("max", "first", "canonical", "none")
                + DETECTORS + FUSION_MODES)
DEDUP_MODES = ("canonical",) + DETECTORS

# variant name -> ledger mode, for the non-learned defences. Each of these runs
# the provenance-free content path under a different duplicate detector, so any
# difference between them is the detector and nothing else.
DEDUP_VARIANTS = {"canonical_dedup": "canonical",
                  "minhash_dedup": "minhash",
                  "simhash_dedup": "simhash",
                  "embed_dedup": "embed"}

# variant name -> fusion mode. These run the same provenance-free content path
# as the dedup variants, so they are comparable to them and to `plain` by
# construction, and differ only in the credit the fused estimate implies.
FUSION_VARIANTS = {"dempster_fusion": "dempster",
                   "covariance_intersection": "covariance_intersection"}


def ledger_mode(variant: str) -> str:
    """Which ledger a variant applies at delivery time."""
    if "filter_only" in variant:
        return "first"
    if variant in DEDUP_VARIANTS:
        return DEDUP_VARIANTS[variant]
    if variant in FUSION_VARIANTS:
        return FUSION_VARIANTS[variant]
    if variant in ("full", "transformer_ec", "ec_exact") or variant.endswith("_ec"):
        return "max"
    return "none"


def resolve_stream(events: Sequence[DeliveryEvent], mode: str,
                   embed: Callable[[str], "np.ndarray"] | None = None
                   ) -> tuple[list[DeliveryEvent], dict[str, float], int]:
    """Apply the ledger at delivery time.

    `max` is the evidence-conserving max-register join: an arrival reaches the
    content path only if it strictly beats the incumbent version of its root, so
    exact redelivery is a no-op and a refinement is not.
    `first` is the ablation that keeps only the first version of each root, so
    it also conserves credit but discards every later computation.
    `canonical` is exact deduplication: byte-identical arrivals collapse, but it
    has no notion of version, so a re-chunked or reworded copy of the same
    paragraph is a fresh unit of evidence.
    `none` has no ledger: every arrival is processed and every arrival adds
    credit, which is the provenance-free behaviour the paper objects to.
    `minhash`, `simhash` and `embed` are near-duplicate defences at the frozen
    thresholds of `near_duplicate.py`: an arrival is suppressed when it looks
    like something already accepted. They are greedy and therefore order
    dependent, and they detect similarity of payload rather than identity of
    lineage, so two textually unlike renderings of one source both pass.
    `dempster` and `covariance_intersection` are fusion rules rather than
    ledgers. Neither suppresses anything; both admit the whole stream and
    differ only in the credit the fused estimate implies. Dempster's rule is
    not idempotent, so repetition sharpens belief and credit rises to a
    ceiling of `1 / DEMPSTER_MASS`. Covariance intersection fuses under convex
    weights, so repetition cannot raise a root above 1.0, but the same
    constraint divides credit across lineage-distinct roots and under-credits
    them.
    """
    if mode not in LEDGER_MODES:
        raise ValueError(mode)
    if mode == "none":
        credit: dict[str, float] = {}
        for e in events:
            credit[e.root_id] = credit.get(e.root_id, 0.0) + 1.0
        return list(events), credit, 0

    if mode == "canonical":
        seen: set[tuple[str, str]] = set()
        kept, suppressed, credit = [], 0, {}
        for e in events:
            key = (e.root_id, e.payload_hash)
            if key in seen:
                suppressed += 1
                continue
            seen.add(key)
            kept.append(e)
            credit[e.root_id] = credit.get(e.root_id, 0.0) + 1.0
        return kept, credit, suppressed

    if mode in DETECTORS:
        # Near-duplicate suppression is global over the stream, exactly as a
        # corpus or retrieval pipeline would apply it: the detector has no
        # access to root identity, so it cannot scope itself per source.
        index = NearDuplicateIndex(mode, embed=embed)
        kept, suppressed, credit = [], 0, {}
        for e in events:
            if not index.accept(e.paragraph):
                suppressed += 1
                continue
            kept.append(e)
            credit[e.root_id] = credit.get(e.root_id, 0.0) + 1.0
        return kept, credit, suppressed

    if mode in FUSION_MODES:
        counts: dict[str, int] = {}
        for e in events:
            counts[e.root_id] = counts.get(e.root_id, 0) + 1
        if mode == "dempster":
            credit = {root: dempster_credit(n) for root, n in counts.items()}
        else:
            # Covariance intersection fuses with convex weights that sum to one,
            # so the fused precision never exceeds the largest input precision.
            # That makes it duplicate-safe for free: repeating one source cannot
            # raise its credit above 1.0. It pays for that with the other error,
            # because the same constraint splits the budget across genuinely
            # lineage-distinct roots, so R distinct roots each receive 1 / R of
            # the credit they are due. The rule cannot tell the two cases apart
            # because it assumes the correlation is unknown rather than reading
            # it off the lineage.
            total = float(sum(counts.values())) or 1.0
            credit = {root: n / total for root, n in counts.items()}
        return list(events), credit, 0

    ledger: dict[str, tuple] = {}
    kept, suppressed = [], 0
    for e in events:
        key = version_key(e)
        incumbent = ledger.get(e.root_id)
        if incumbent is None:
            ledger[e.root_id] = key
            kept.append(e)
        elif mode == "max" and key > incumbent:
            ledger[e.root_id] = key
            kept.append(e)
        else:
            suppressed += 1
    return kept, {root: 1.0 for root in ledger}, suppressed


def stream_stats(events: Sequence[DeliveryEvent]) -> dict:
    return {"n_events": len(events),
            "n_roots": len({e.root_id for e in events}),
            "n_payloads": len({e.payload_hash for e in events}),
            "hops": sorted({e.hop for e in events})}


def make_delivery_batch(streams: Sequence[Sequence[DeliveryEvent]],
                        records: Sequence[MuSiQueRecord], cache,
                        device: str = "cpu") -> dict:
    """Tensorise a batch of resolved delivery streams.

    The step axis is the arrival axis, which may be longer than the chain. Each
    arrival keeps its hop index so a duplicated arrival reuses that hop's
    positional embedding rather than needing a new one.
    """
    batch_size = len(streams)
    width = max(len(s) for s in streams)
    dim = cache.dim
    question = torch.zeros(batch_size, width, dim, device=device)
    paragraph = torch.zeros_like(question)
    answer = torch.zeros_like(question)
    mask = torch.zeros(batch_size, width, device=device)
    hop_index = torch.zeros(batch_size, width, dtype=torch.long, device=device)
    for row, (stream, record) in enumerate(zip(streams, records)):
        if not stream:
            continue
        question[row, :len(stream)] = torch.as_tensor(
            cache.encode([e.question for e in stream]), device=device)
        paragraph[row, :len(stream)] = torch.as_tensor(
            cache.encode([e.paragraph for e in stream]), device=device)
        answer[row, :len(stream)] = torch.as_tensor(
            cache.encode([record.steps[e.hop].answer for e in stream]),
            device=device)
        mask[row, :len(stream)] = 1.0
        hop_index[row, :len(stream)] = torch.tensor([e.hop for e in stream],
                                                    device=device)
    return {"question": question, "paragraph": paragraph, "answer": answer,
            "mask": mask, "hop_index": hop_index,
            "n_hops": mask.sum(-1).long(),
            "chain_hops": torch.tensor([r.n_hops for r in records],
                                       device=device),
            "record_ids": [r.record_id for r in records]}
