"""Paired causal interventions on real evidence.

Every intervention returns a record for the SAME claim, so a change in the
model's output cannot be explained by a different example.  Each perturbation
records how it was produced, in `provenance_note`, so the transformation
manifest is recoverable from the data itself.

The interventions that ADD passages derived from an existing document keep that
document's `root_id`: they are computation, not evidence.  Only
`add_lineage_distinct_source` introduces a new root, and `sybil_split` /
`corrupt_lineage` deliberately break the identity to measure the documented
failure modes.
"""
from __future__ import annotations

import copy
import hashlib
import random
import re
from dataclasses import replace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .averitec import ClaimRecord, Passage
from .lineage import content_hash, jaccard, shingles


def _clone(rec: ClaimRecord) -> ClaimRecord:
    return ClaimRecord(claim_id=rec.claim_id, claim=rec.claim, label=rec.label,
                       split=rec.split, justification=rec.justification,
                       passages=[copy.deepcopy(p) for p in rec.passages],
                       raw=rec.raw)


def _suffix(p: Passage, tag: str, i: int) -> str:
    return f"{p.passage_id}::{tag}{i}"


# ------------------------------------------------------------------ identity
def clean(rec: ClaimRecord) -> ClaimRecord:
    return _clone(rec)


# ------------------------------------------------------- same-root duplication
def exact_redelivery(rec: ClaimRecord, factor: int = 2) -> ClaimRecord:
    """Deliver every passage `factor` times, byte-identical."""
    out = _clone(rec)
    extra: List[Passage] = []
    for p in out.passages:
        for i in range(1, factor):
            q = copy.deepcopy(p)
            q.passage_id = _suffix(p, "dup", i)
            q.provenance_note = f"exact redelivery {i} of {p.passage_id}"
            extra.append(q)
    out.passages.extend(extra)
    return out


def overlapping_chunks(rec: ClaimRecord, n_chunks: int = 3,
                       overlap: float = 0.5) -> ClaimRecord:
    """Split each passage into overlapping windows of the SAME document."""
    out = _clone(rec)
    chunks: List[Passage] = []
    for p in out.passages:
        words = p.text.split()
        if len(words) < 8:
            continue
        size = max(4, int(len(words) / max(1, n_chunks - (n_chunks - 1) * overlap)))
        step = max(1, int(size * (1 - overlap)))
        for i in range(n_chunks):
            s = i * step
            piece = " ".join(words[s:s + size])
            if not piece:
                continue
            q = copy.deepcopy(p)
            q.passage_id, q.text = _suffix(p, "chunk", i), piece
            q.content_hash = content_hash(piece)
            q.provenance_note = f"overlapping chunk {i} of {p.passage_id}"
            chunks.append(q)                      # root_id unchanged: same document
    out.passages.extend(chunks)
    return out


def same_root_paraphrase(rec: ClaimRecord, paraphraser: Callable[[str], str],
                         factor: int = 1, min_similarity: float = 0.35,
                         max_similarity: float = 0.98) -> Tuple[ClaimRecord, List[Dict]]:
    """Add paraphrases that keep the ORIGINAL root id.

    Returns the record and a validation manifest.  A paraphrase that is nearly
    identical to the source is a redelivery, not a paraphrase; one that shares
    almost nothing may have changed the meaning.  Both are recorded and flagged
    rather than silently accepted.
    """
    out = _clone(rec)
    manifest: List[Dict] = []
    added: List[Passage] = []
    for p in out.passages:
        for i in range(factor):
            text = paraphraser(p.text)
            sim = jaccard(shingles(p.text, 3), shingles(text, 3))
            ok = min_similarity <= sim <= max_similarity
            manifest.append({"source": p.passage_id, "root_id": p.root_id,
                             "similarity": sim, "accepted": ok,
                             "method": getattr(paraphraser, "__name__", "callable")})
            if not ok:
                continue
            q = copy.deepcopy(p)
            q.passage_id, q.text = _suffix(p, "para", i), text
            q.content_hash = content_hash(text)
            q.provenance_note = f"paraphrase of {p.passage_id} (same root)"
            added.append(q)                       # root_id unchanged
    out.passages.extend(added)
    return out, manifest


def summary_chain(rec: ClaimRecord, summariser: Callable[[str], str],
                  depth: int = 2) -> ClaimRecord:
    """Summaries, then summaries of those summaries -- all one root."""
    out = _clone(rec)
    added: List[Passage] = []
    for p in out.passages:
        text = p.text
        for d in range(depth):
            text = summariser(text)
            if not text.strip():
                break
            q = copy.deepcopy(p)
            q.passage_id, q.text = _suffix(p, f"sum{d}", d), text
            q.content_hash = content_hash(text)
            q.provenance_note = f"summary depth {d + 1} of {p.passage_id}"
            added.append(q)
    out.passages.extend(added)
    return out


# --------------------------------------------------------- new lineage / noise
def add_lineage_distinct_source(rec: ClaimRecord, text: str, root_id: str,
                                url: str = "", informative: bool = True) -> ClaimRecord:
    """The ONLY intervention that legitimately increases credited evidence."""
    out = _clone(rec)
    out.passages.append(Passage(
        passage_id=f"{rec.claim_id}::newroot::{root_id[-8:]}", text=text,
        root_id=root_id, document_id=root_id, content_hash=content_hash(text),
        source_url=url, is_gold=False,
        provenance_note=f"lineage-distinct source ({'informative' if informative else 'contradicting'})"))
    return out


def irrelevant_documents(rec: ClaimRecord, texts: Sequence[str]) -> ClaimRecord:
    out = _clone(rec)
    for i, t in enumerate(texts):
        rid = f"doc::irrelevant{hashlib.blake2b(t.encode(), digest_size=6).hexdigest()}"
        out.passages.append(Passage(
            passage_id=f"{rec.claim_id}::irrel{i}", text=t, root_id=rid,
            document_id=rid, content_hash=content_hash(t), is_gold=False,
            retrieval_score=0.0, provenance_note="irrelevant retrieved document"))
    return out


# ------------------------------------------------- documented failure modes
def corrupt_lineage(rec: ClaimRecord, fraction: float = 0.5,
                    seed: int = 0) -> ClaimRecord:
    """Destroy a fraction of root ids. A documented limitation, not a hidden case."""
    out = _clone(rec)
    rng = random.Random(seed)
    for p in out.passages:
        if rng.random() < fraction:
            p.root_id = ""
            p.provenance_note = "lineage corrupted (root id destroyed)"
    return out


def sybil_split(rec: ClaimRecord, factor: int = 4, seed: int = 0) -> ClaimRecord:
    """One source relabelled under many false roots. Defeats lineage BY DESIGN."""
    out = _clone(rec)
    added: List[Passage] = []
    for p in out.passages:
        for i in range(1, factor):
            q = copy.deepcopy(p)
            q.passage_id = _suffix(p, "sybil", i)
            h = hashlib.blake2b(f"{p.passage_id}{i}".encode(), digest_size=12).hexdigest()
            q.root_id = q.document_id = f"doc::sybil{h}"
            q.provenance_note = f"sybil relabel {i} of {p.passage_id}"
            added.append(q)
    out.passages.extend(added)
    return out


def mirror_pages(rec: ClaimRecord, factor: int = 2) -> ClaimRecord:
    """Identical text served from DIFFERENT urls: correlated but not detectable
    by document root alone.  This is what content-family lineage is for."""
    out = _clone(rec)
    added: List[Passage] = []
    for p in out.passages:
        for i in range(1, factor):
            q = copy.deepcopy(p)
            q.passage_id = _suffix(p, "mirror", i)
            q.source_url = f"https://mirror{i}.example/{p.content_hash}"
            h = hashlib.blake2b(q.source_url.encode(), digest_size=12).hexdigest()
            q.root_id = q.document_id = f"doc::mirror{h}"
            q.provenance_note = f"mirror copy {i} of {p.passage_id}"
            added.append(q)
    out.passages.extend(added)
    return out


INTERVENTIONS: Dict[str, Callable] = {
    "clean": clean,
    "redeliver_2x": lambda r: exact_redelivery(r, 2),
    "redeliver_4x": lambda r: exact_redelivery(r, 4),
    "redeliver_8x": lambda r: exact_redelivery(r, 8),
    "redeliver_16x": lambda r: exact_redelivery(r, 16),
    "overlapping_chunks": overlapping_chunks,
    "corrupt_lineage": corrupt_lineage,
    "sybil_split": sybil_split,
    "mirror_pages": mirror_pages,
}
