"""Outcome-blind retrieval-quality slices for AVeriTeC.

The thresholds are frozen in ``TIER1_PREREGISTRATION.md``.  This module is
deliberately independent of torch and model outputs so slice membership cannot
silently become an outcome-selected analysis.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Sequence, Set

from .averitec import ClaimRecord, Passage


_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but",
    "by", "for", "from", "had", "has", "have", "he", "her", "hers", "him",
    "his", "i", "in", "into", "is", "it", "its", "of", "on", "or", "our",
    "she", "that", "the", "their", "them", "there", "they", "this", "to",
    "was", "we", "were", "what", "when", "where", "which", "who", "will",
    "with", "would", "you", "your",
})


def content_tokens(text: str) -> Set[str]:
    return {token for token in _TOKEN.findall((text or "").lower())
            if token not in _STOP}


def strong_content_match(retrieved: Passage, gold: Passage,
                         min_shared: int = 8,
                         min_containment: float = 0.45) -> bool:
    a, b = content_tokens(retrieved.text), content_tokens(gold.text)
    if not a or not b:
        return False
    shared = len(a & b)
    containment = shared / min(len(a), len(b))
    return shared >= min_shared and containment >= min_containment


@dataclass(frozen=True)
class QualityFlags:
    exact_root: bool
    strong_content: bool

    @property
    def root_or_content(self) -> bool:
        return self.exact_root or self.strong_content


def quality_flags(record: ClaimRecord) -> QualityFlags:
    gold = list(record.gold_passages)
    gold_roots = {p.root_id for p in gold if p.root_id}
    exact = any(p.root_id in gold_roots for p in record.passages if p.root_id)
    content = any(strong_content_match(r, g)
                  for r in record.passages for g in gold)
    return QualityFlags(exact_root=exact, strong_content=content)


def select_quality_slice(records: Sequence[ClaimRecord], name: str) -> list[int]:
    if name not in ("exact_root", "strong_content", "root_or_content"):
        raise ValueError(f"unknown quality slice {name!r}")
    return [i for i, record in enumerate(records)
            if bool(getattr(quality_flags(record), name))]

