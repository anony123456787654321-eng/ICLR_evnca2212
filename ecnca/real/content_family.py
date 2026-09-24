"""Automatic content-family lineage for mirrored and relabelled passages."""
from __future__ import annotations

import copy
import hashlib
import re
from collections import defaultdict
from dataclasses import replace
from typing import Iterable, Sequence

from .averitec import ClaimRecord, Passage


_WORD = re.compile(r"[a-z0-9]+")
_SPACE = re.compile(r"\s+")
_SYNONYMS = {
    "said": "stated", "says": "states", "reported": "described",
    "increase": "rise", "increased": "rose", "decrease": "decline",
    "decreased": "declined", "large": "substantial", "small": "minor",
    "people": "persons", "government": "administration", "claim": "assertion",
    "false": "incorrect", "true": "correct", "showed": "demonstrated",
}


def _tokens(text: str) -> list[str]:
    return _WORD.findall((text or "").lower())


def _char_ngrams(text: str, n: int = 5) -> set[str]:
    normal = _SPACE.sub(" ", " ".join(_tokens(text))).strip()
    if len(normal) <= n:
        return {normal} if normal else set()
    return {normal[i:i + n] for i in range(len(normal) - n + 1)}


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a or b else 0.0


def surface_similarity(a: str, b: str) -> float:
    """Symmetric near-copy score in [0,1]."""
    ta, tb = set(_tokens(a)), set(_tokens(b))
    if not ta or not tb:
        return 0.0
    shared = len(ta & tb)
    containment = shared / min(len(ta), len(tb))
    length_ratio = min(len(ta), len(tb)) / max(len(ta), len(tb))
    token_score = 0.75 * containment + 0.25 * length_ratio
    return float(max(token_score, _jaccard(_char_ngrams(a), _char_ngrams(b))))


def deterministic_perturbations(text: str) -> dict[str, str]:
    tokens = _tokens(text)
    if len(tokens) < 20:
        return {}
    format_change = "  ".join(token.upper() if i % 3 == 0 else token
                               for i, token in enumerate(tokens)) + "!!!"
    deletion = " ".join(token for i, token in enumerate(tokens) if i % 5 != 0)
    width = max(5, len(tokens) // 4)
    blocks = [tokens[i:i + width] for i in range(0, len(tokens), width)]
    reorder = " ".join(token for block in reversed(blocks) for token in block)
    synonyms = " ".join(_SYNONYMS.get(token, token) for token in tokens)
    return {"format": format_change, "deletion20": deletion,
            "block_reorder": reorder, "synonyms": synonyms}


def choose_threshold(positives: Sequence[tuple[str, str]],
                     negatives: Sequence[tuple[str, str]],
                     max_false_link_rate: float = 0.005) -> tuple[float, dict]:
    pos = [surface_similarity(a, b) for a, b in positives]
    neg = [surface_similarity(a, b) for a, b in negatives]
    eligible = []
    for integer in range(50, 101):
        threshold = integer / 100
        fpr = sum(score >= threshold for score in neg) / max(len(neg), 1)
        recall = sum(score >= threshold for score in pos) / max(len(pos), 1)
        if fpr <= max_false_link_rate:
            eligible.append((recall, threshold, fpr))
    if not eligible:
        raise ValueError("no threshold satisfies the false-link budget")
    recall, threshold, fpr = max(eligible, key=lambda x: (x[0], x[1]))
    return threshold, {"train_positive_recall": recall,
                       "train_false_link_rate": fpr,
                       "n_positive_pairs": len(pos), "n_negative_pairs": len(neg)}


def training_pairs(records: Sequence[ClaimRecord]):
    positives, negatives = [], []
    for record in records:
        passages = record.passages
        for i, first in enumerate(passages):
            for second in passages[i + 1:]:
                if first.root_id == second.root_id:
                    continue
                pair = (first.text, second.text)
                if first.content_hash == second.content_hash:
                    positives.append(pair)
                else:
                    negatives.append(pair)
        for passage in passages:
            positives.extend((passage.text, changed)
                             for changed in deterministic_perturbations(passage.text).values())
    return positives, negatives


def family_components(passages: Sequence[Passage], threshold: float) -> list[list[int]]:
    parent = list(range(len(passages)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        a, b = find(i), find(j)
        if a != b:
            parent[max(a, b)] = min(a, b)

    for i, first in enumerate(passages):
        for j in range(i + 1, len(passages)):
            second = passages[j]
            if first.root_id == second.root_id or \
                    surface_similarity(first.text, second.text) >= threshold:
                union(i, j)
    groups = defaultdict(list)
    for i in range(len(passages)):
        groups[find(i)].append(i)
    return list(groups.values())


def apply_content_families(records: Sequence[ClaimRecord],
                           threshold: float) -> list[ClaimRecord]:
    out = copy.deepcopy(list(records))
    for record in out:
        for component in family_components(record.passages, threshold):
            roots = sorted({record.passages[i].root_id for i in component})
            if len(roots) <= 1:
                continue
            digest = hashlib.blake2b("\x1f".join(roots).encode(), digest_size=12).hexdigest()
            family = f"family::{digest}"
            for i in component:
                passage = record.passages[i]
                passage.provenance_note = (passage.provenance_note +
                                           f"; family_of={passage.root_id}").strip("; ")
                passage.root_id = family
                passage.document_id = family
    return out

