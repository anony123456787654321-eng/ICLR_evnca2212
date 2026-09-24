"""Paired real-evidence interventions on AVeriTeC.

The templated new-root probe used in Gate B is WITHDRAWN from primary
reporting: it restated the claim, so it added lineage but no information, and
its accuracy component was meaningless.  This module replaces it with a paired
intervention over REAL held-out gold roots.

`informative` is defined BEFORE results are read (see `INFORMATIVE_DEF`).
"""
from __future__ import annotations

import copy
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .averitec import ClaimRecord

# Frozen before any result is inspected.  A second root counts as informative
# when it contributes text the first root does not already contain: at least
# `MIN_NEW_WORDS` content words absent from condition A, and a shingle overlap
# with A below `MAX_OVERLAP`.  This is a property of the EVIDENCE, never of the
# model's output or of the label.
INFORMATIVE_DEF = {"min_new_words": 8, "max_shingle_overlap": 0.5, "shingle_n": 3}


def _shell(rec: ClaimRecord) -> ClaimRecord:
    return ClaimRecord(claim_id=rec.claim_id, claim=rec.claim, label=rec.label,
                       split=rec.split, justification=rec.justification,
                       record_id=rec.record_id, claim_text_hash=rec.claim_text_hash,
                       raw=rec.raw)


def _by_root(rec: ClaimRecord) -> Dict[str, List]:
    out: Dict[str, List] = {}
    for p in rec.passages:
        if p.root_id:
            out.setdefault(p.root_id, []).append(p)
    return out


def is_informative(a_texts: Sequence[str], b_texts: Sequence[str]) -> bool:
    from .lineage import jaccard, normalise_text, shingles
    n = INFORMATIVE_DEF["shingle_n"]
    a_words = set(" ".join(normalise_text(t) for t in a_texts).split())
    b_words = set(" ".join(normalise_text(t) for t in b_texts).split())
    new_words = len(b_words - a_words)
    ov = jaccard(shingles(" ".join(a_texts), n), shingles(" ".join(b_texts), n))
    return (new_words >= INFORMATIVE_DEF["min_new_words"]
            and ov <= INFORMATIVE_DEF["max_shingle_overlap"])


def one_vs_two_real_roots(records: Sequence[ClaimRecord]
                          ) -> Tuple[List[ClaimRecord], List[ClaimRecord], List[bool]]:
    """Condition A: the first real root. Condition B: A plus a held-out second.

    Same claim, same model, same first root -- so any change is attributable to
    the added lineage and not to a different or easier example.
    """
    A: List[ClaimRecord] = []
    B: List[ClaimRecord] = []
    informative: List[bool] = []
    for rec in records:
        roots = _by_root(rec)
        if len(roots) < 2:
            continue
        names = sorted(roots)                     # deterministic
        first, second = names[0], names[1]
        a = _shell(rec); a.passages = [copy.deepcopy(p) for p in roots[first]]
        b = _shell(rec)
        b.passages = ([copy.deepcopy(p) for p in roots[first]]
                      + [copy.deepcopy(p) for p in roots[second]])
        A.append(a); B.append(b)
        informative.append(is_informative([p.text for p in roots[first]],
                                          [p.text for p in roots[second]]))
    return A, B, informative
