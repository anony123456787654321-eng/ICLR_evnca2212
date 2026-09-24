"""Evidence-conserving evidence selection: turn repetition into room.

The confidence experiment could only rescale an answer distribution, and
positive scaling cannot change which answer wins. This module changes what the
model READS, under a fixed budget, which is where an accuracy claim can come
from.

The principle:

    Repeated evidence should neither occupy extra influence nor waste the space
    needed for new information. Useful refinements should survive.

A *source* is a lineage root (a canonical document). A source's memory holds the
distinct content delivered for it, not one passage and not one copy per
delivery:

  * the same passage delivered again adds nothing;
  * a different passage from that source may refine the memory;
  * another source opens another memory.

Source identity is an accounting unit. It is not a claim that two sources are
statistically independent.

Selection then fills a token budget from the candidate pool, preferring
relevance and penalising content that repeats what is already selected. Slots
freed by collapsing repeats are refilled from the remaining candidates rather
than left empty, which is the mechanism under test.

`SELECTORS` are the comparison set. Every one of them sees the same candidates
and the same budget, so a difference is the selection rule and nothing else.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence

import hashlib

import numpy as np


def _content_hash(text: str) -> str:
    """Local content hash for synthesised burst passages.

    Matches the shape of ecnca.real.lineage.content_hash without importing it,
    so this module stays usable on synthetic candidates in tests.
    """
    return hashlib.blake2b(" ".join(text.lower().split()).encode(),
                           digest_size=12).hexdigest()


SELECTORS = ("topk", "canonical_norefill", "canonical_refill", "mmr",
             "source_balanced", "conserving")

# The three retrieval conditions under test. `same_source_burst` is the primary
# one: it is what separates lineage-aware selection from exact deduplication,
# because the burst passages have DIFFERENT content hashes and one shared root,
# so no content-identity rule can collapse them.
# `same_source_burst` is the PRIMARY redundancy condition and uses only
# genuinely available distinct passages from one source, up to the available
# count. `same_source_marker` is a separately named FORMATTING-VARIATION
# diagnostic that pads a short burst with marked continuations; it manufactures
# distinct content hashes and must never stand in for the primary condition.
CONDITIONS = ("natural", "exact_repeat", "same_source_burst")
DIAGNOSTIC_CONDITIONS = ("same_source_marker",)
ALL_CONDITIONS = CONDITIONS + DIAGNOSTIC_CONDITIONS


@dataclass(frozen=True)
class Candidate:
    """One retrieved passage with its lineage and retrieval rank."""
    text: str
    root_id: str
    content_hash: str
    rank: int
    score: float = 0.0
    question: str = ""

    @property
    def display(self) -> str:
        q, t = self.question.strip(), self.text.strip()
        if not q:
            return t
        return f"{q if q.endswith('?') else q + '?'} {t}"


def apply_condition(cands: Sequence[Candidate], condition: str,
                    burst: int = 8) -> List[Candidate]:
    """Build a condition's delivery stream, retaining the FULL pool for refill.

    natural            the store's ranking, untouched.
    exact_repeat       `burst` exact copies of the top-ranked passage inserted
                       into the top ten. Every copy shares the content hash, so
                       exact deduplication collapses them completely. This is
                       the condition where canonical dedup should already win,
                       and it exists to show conservation matches rather than
                       beats it, plus to test prompt invariance.
    same_source_burst  up to `burst` DISTINCT-content passages drawn from the
                       single canonical source that contributes most to the
                       pool, inserted into the top ten. Different hashes, one
                       root: exact deduplication cannot collapse them and only
                       lineage can. This is the primary condition.

    In both perturbed conditions the original pool is preserved after the
    inserted block, so any selector may refill from it.
    """
    if condition not in ALL_CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}")
    order = sorted(cands, key=lambda c: c.rank)
    if condition == "natural" or not order:
        return list(order)

    if condition == "exact_repeat":
        head = order[0]
        inserted = [Candidate(head.text, head.root_id, head.content_hash,
                              rank=0, score=head.score, question=head.question)
                    for _ in range(burst)]
    else:
        by_root: Dict[str, List[Candidate]] = {}
        for c in order:
            by_root.setdefault(c.root_id, []).append(c)

        def distinct_count(root):
            return len({c.content_hash for c in by_root[root]})

        root = max(by_root, key=lambda r: (distinct_count(r), -by_root[r][0].rank))
        seen, pool = set(), []
        for c in by_root[root]:
            if c.content_hash in seen:
                continue
            seen.add(c.content_hash)
            pool.append(c)
        if len(pool) < 2:
            # Not enough genuinely distinct content from any one source.
            return list(order)

        if condition == "same_source_burst":
            # PRIMARY: use only the distinct passages that genuinely exist, up
            # to the requested size. Padding a short burst with marked
            # continuations would manufacture content hashes and turn a
            # redundancy condition into a formatting-variation condition.
            inserted = [Candidate(c.text, c.root_id, c.content_hash, rank=i,
                                  score=c.score, question=c.question)
                        for i, c in enumerate(pool[:burst])]
        else:
            # same_source_marker: DIAGNOSTIC ONLY. Pads to `burst` with marked
            # continuations, so the burst size is guaranteed but the extra
            # hashes are synthetic.
            inserted = []
            for i in range(burst):
                if i < len(pool):
                    c = pool[i]
                    inserted.append(Candidate(c.text, c.root_id, c.content_hash,
                                              rank=i, score=c.score,
                                              question=c.question))
                else:
                    base = pool[i % len(pool)]
                    text = f"{base.text} (continued, part {i + 1})"
                    inserted.append(Candidate(text, base.root_id,
                                              _content_hash(text), rank=i,
                                              score=base.score,
                                              question=base.question))

    # Inserted block occupies the head of the ranking; the untouched pool
    # follows so refill still has everything it had before.
    out = []
    for i, c in enumerate(inserted):
        out.append(Candidate(c.text, c.root_id, c.content_hash, rank=i,
                             score=c.score, question=c.question))
    offset = len(out)
    for c in order:
        out.append(Candidate(c.text, c.root_id, c.content_hash,
                             rank=offset + c.rank, score=c.score,
                             question=c.question))
    return out


def condition_report(original: Sequence[Candidate],
                     stream: Sequence[Candidate], condition: str,
                     k: int = 10, burst: int = 8) -> dict:
    """What a condition actually did, so a degenerate one cannot pass silently.

    `burst_achieved` is the number of inserted items and `burst_eligible` says
    whether the source had enough genuinely distinct content to reach the
    requested size. Reporting both is required for the primary condition, whose
    burst is capped by availability rather than padded.
    """
    head = sorted(stream, key=lambda c: c.rank)[:k]
    order = sorted(original, key=lambda c: c.rank)
    by_root: Dict[str, set] = {}
    for c in order:
        by_root.setdefault(c.root_id, set()).add(c.content_hash)
    best = max((len(v) for v in by_root.values()), default=0)
    achieved = max(0, len(stream) - len(original))
    return {"condition": condition,
            "n_stream": len(stream), "n_original": len(original),
            "head_roots": len({c.root_id for c in head}),
            "head_hashes": len({c.content_hash for c in head}),
            "head_passages": len(head),
            "burst_requested": burst,
            "burst_achieved": achieved,
            "max_distinct_from_one_source": best,
            "burst_eligible": bool(best >= burst),
            "degenerate": len(stream) == len(original) and condition != "natural"}


def token_cost(cand: Candidate, chars_per_token: float = 4.0) -> float:
    """Budget cost of including a candidate.

    A character proxy keeps selection independent of any particular tokeniser so
    every selector is scored on identical arithmetic; the driver re-checks the
    true tokenised length of the final prompt.
    """
    return len(cand.display) / chars_per_token


@dataclass
class SourceMemory:
    """The distinct content credited to one lineage root.

    `versions` holds content hashes in arrival order. An exact repeat is a no-op,
    which is what makes redelivery leave the canonical state unchanged. A new
    distinct passage from the same source is a refinement and is retained, which
    is what distinguishes this from keeping only the first passage.
    """
    root_id: str
    versions: List[str] = field(default_factory=list)
    passages: List[Candidate] = field(default_factory=list)

    def deliver(self, cand: Candidate) -> bool:
        """True if this delivery added new content to the source's memory."""
        if cand.content_hash in self.versions:
            return False
        self.versions.append(cand.content_hash)
        self.passages.append(cand)
        return True

    @property
    def credit(self) -> float:
        """One unit per source, however many passages it carries."""
        return 1.0


class SourceLedger:
    """Source-aware memory over a delivery stream.

    Credited evidence is the number of sources with any content, so it is
    invariant to how many times anything was delivered. The retained passages
    are the union of distinct content per source, so refinements survive.
    """

    def __init__(self) -> None:
        self.memories: Dict[str, SourceMemory] = {}
        self.n_deliveries = 0
        self.n_novel = 0

    def deliver(self, cand: Candidate) -> bool:
        self.n_deliveries += 1
        mem = self.memories.setdefault(cand.root_id, SourceMemory(cand.root_id))
        novel = mem.deliver(cand)
        self.n_novel += int(novel)
        return novel

    def deliver_all(self, cands: Sequence[Candidate]) -> "SourceLedger":
        for c in cands:
            self.deliver(c)
        return self

    @property
    def credited_evidence(self) -> float:
        return float(len(self.memories))

    @property
    def retained(self) -> List[Candidate]:
        """Distinct content across sources, in retrieval-rank order."""
        out = [p for m in self.memories.values() for p in m.passages]
        return sorted(out, key=lambda c: c.rank)

    def state_signature(self) -> tuple:
        """Canonical state. Equal signatures mean equal information content."""
        return tuple(sorted((r, tuple(m.versions))
                            for r, m in self.memories.items()))


# ---------------------------------------------------------------------------
# selectors
# ---------------------------------------------------------------------------

def _fill(order: Sequence[Candidate], budget: float,
          skip: Callable[[Candidate, List[Candidate]], bool] | None = None,
          cost_fn: Callable[[Candidate], float] | None = None
          ) -> List[Candidate]:
    cost = cost_fn or token_cost
    chosen: List[Candidate] = []
    used = 0.0
    for c in order:
        if skip is not None and skip(c, chosen):
            continue
        item = cost(c)
        if used + item > budget:
            continue
        chosen.append(c)
        used += item
    return chosen


def select_topk(cands: Sequence[Candidate], budget: float, k: int = 10,
                cost_fn=None, **kw) -> List[Candidate]:
    """Ordinary retrieval: the first k by rank, duplicates included."""
    return _fill(sorted(cands, key=lambda c: c.rank)[:k], budget, cost_fn=cost_fn)


def select_canonical_norefill(cands: Sequence[Candidate], budget: float,
                              k: int = 10, cost_fn=None, **kw) -> List[Candidate]:
    """Exact dedup inside the top k, freed slots left EMPTY.

    The weak form of deduplication. Included because beating it would not
    establish anything: the interesting comparison is against dedup that
    refills.
    """
    top = sorted(cands, key=lambda c: c.rank)[:k]
    seen: set = set()
    keep = []
    for c in top:
        if c.content_hash in seen:
            continue
        seen.add(c.content_hash)
        keep.append(c)
    return _fill(keep, budget, cost_fn=cost_fn)


def select_canonical_refill(cands: Sequence[Candidate], budget: float,
                            k: int = 10, cost_fn=None, **kw) -> List[Candidate]:
    """Exact dedup, then refill the freed budget from the remaining pool.

    The strong deduplication baseline. Any gain our method shows must be over
    THIS, not over dedup that wastes the space it frees.
    """
    order = sorted(cands, key=lambda c: c.rank)
    seen: set = set()

    def skip(c, _chosen):
        if c.content_hash in seen:
            return True
        seen.add(c.content_hash)
        return False

    return _fill(order, budget, skip, cost_fn=cost_fn)


def select_mmr(cands: Sequence[Candidate], budget: float, k: int = 10,
               embed: Callable[[Sequence[str]], np.ndarray] | None = None,
               lam: float = 0.7, cost_fn=None, **kw) -> List[Candidate]:
    """Relevance/diversity selection under the same budget.

    A content-similarity baseline with no notion of lineage. It penalises
    passages that look like what is already chosen, which is the natural
    alternative to counting sources.
    """
    order = sorted(cands, key=lambda c: c.rank)
    cost = cost_fn or token_cost
    if embed is None:
        return select_canonical_refill(cands, budget, k, cost_fn=cost_fn)
    vecs = np.asarray(embed([c.display for c in order]), dtype=float)
    vecs /= np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-9)
    rel = np.array([1.0 / (1.0 + c.rank) for c in order])
    chosen: List[int] = []
    used = 0.0
    while True:
        best, best_val = None, -np.inf
        for i, c in enumerate(order):
            if i in chosen or used + cost(c) > budget:
                continue
            div = 0.0 if not chosen else float(np.max(vecs[chosen] @ vecs[i]))
            val = lam * rel[i] - (1.0 - lam) * div
            if val > best_val:
                best, best_val = i, val
        if best is None:
            break
        chosen.append(best)
        used += cost(order[best])
    return [order[i] for i in sorted(chosen)]


def select_source_balanced(cands: Sequence[Candidate], budget: float,
                           k: int = 10, cost_fn=None, **kw) -> List[Candidate]:
    """One passage per source, round-robin by rank.

    Source-aware but with NO refinement: it never takes a second passage from a
    source, so it cannot recover a later paragraph that corrects an earlier one.
    This isolates how much of any gain comes from retaining refinements rather
    than from balancing sources.
    """
    order = sorted(cands, key=lambda c: c.rank)
    by_root: Dict[str, List[Candidate]] = {}
    for c in order:
        by_root.setdefault(c.root_id, []).append(c)
    cost = cost_fn or token_cost
    roots = sorted(by_root, key=lambda r: by_root[r][0].rank)
    chosen, used = [], 0.0
    for r in roots:
        c = by_root[r][0]
        if used + cost(c) <= budget:
            chosen.append(c)
            used += cost(c)
    return sorted(chosen, key=lambda c: c.rank)


def select_conserving(cands: Sequence[Candidate], budget: float, k: int = 10,
                      embed: Callable[[Sequence[str]], np.ndarray] | None = None,
                      per_source_cap: int = 3,
                      novelty_threshold: float = 0.97,
                      cost_fn: Callable[[Candidate], float] | None = None,
                      **kw) -> List[Candidate]:
    """The hybrid: source-aware memory, retained refinements, budgeted refill.

    Deliver the whole candidate pool into a source ledger, which collapses exact
    repeats without discarding a source's other passages. Then fill the budget
    in rank order, admitting a further passage from an already-represented source
    only when it is not near-identical to what that source already contributed,
    and capping any one source so a single document cannot monopolise the
    budget. Slots freed by collapsing repeats are spent on new sources.

    The three ingredients are separable and each has its own ablation in
    `SELECTORS`: refill (`canonical_refill`), source balance
    (`source_balanced`), and content diversity (`mmr`).
    """
    cost = cost_fn or token_cost
    # Exact-content repetition is removed GLOBALLY, not merely within a source,
    # so an identical passage republished under two identifiers occupies one
    # slot.
    ledger = SourceLedger().deliver_all(sorted(cands, key=lambda c: c.rank))
    retained, seen_global = [], set()
    for c in ledger.retained:
        if c.content_hash in seen_global:
            continue
        seen_global.add(c.content_hash)
        retained.append(c)
    vecs = None
    if embed is not None and retained:
        vecs = np.asarray(embed([c.display for c in retained]), dtype=float)
        vecs /= np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-9)
    index = {id(c): i for i, c in enumerate(retained)}
    per_source: Dict[str, int] = {}
    chosen_idx: List[int] = []
    chosen, used = [], 0.0
    for c in retained:
        item_cost = cost(c)
        if used + item_cost > budget:
            continue
        if per_source.get(c.root_id, 0) >= per_source_cap:
            continue
        if chosen_idx and vecs is not None:
            i = index[id(c)]
            # Similarity is penalised against EVERYTHING already selected, not
            # only against the same source, so a near-duplicate republished
            # elsewhere is also excluded.
            if float(np.max(vecs[chosen_idx] @ vecs[i])) >= novelty_threshold:
                continue
        chosen.append(c)
        chosen_idx.append(index[id(c)])
        used += item_cost
        per_source[c.root_id] = per_source.get(c.root_id, 0) + 1
    return chosen


_SELECTORS: Dict[str, Callable[..., List[Candidate]]] = {
    "topk": select_topk,
    "canonical_norefill": select_canonical_norefill,
    "canonical_refill": select_canonical_refill,
    "mmr": select_mmr,
    "source_balanced": select_source_balanced,
    "conserving": select_conserving,
}


def select(name: str, cands: Sequence[Candidate], budget: float, **kw
           ) -> List[Candidate]:
    if name not in _SELECTORS:
        raise ValueError(f"unknown selector {name!r}")
    return _SELECTORS[name](cands, budget, **kw)


def information_use(chosen: Sequence[Candidate]) -> dict:
    """How much distinct, non-repeated evidence reached the prompt."""
    return {"n_passages": len(chosen),
            "n_sources": len({c.root_id for c in chosen}),
            "n_distinct_text": len({c.content_hash for c in chosen}),
            "n_exact_repeats": len(chosen) - len({c.content_hash for c in chosen}),
            "tokens": float(sum(token_cost(c) for c in chosen)),
            "mean_rank": float(np.mean([c.rank for c in chosen])) if chosen else 0.0}
