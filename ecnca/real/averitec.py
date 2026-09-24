"""AVeriTeC loading into a stable record schema.

Every available field is preserved: dropping one early makes an audit
impossible later.  The loader is tolerant of field-name variation across
releases and records what it could NOT find rather than silently defaulting.
"""
from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import hashlib

from .lineage import (content_family_ids, content_hash, document_root_id,
                      normalise_text)

# Missing-lineage policy. PRIMARY is conservative: passages in one claim with no
# usable source identity share ONE claim-local unknown root. Giving each its own
# root would manufacture evidence out of missing metadata, which is exactly the
# error this paper is about.
MISSING_POLICIES = ("claim_local_unknown", "exclude", "separate_roots")

LABELS = ["Supported", "Refuted", "Not Enough Evidence",
          "Conflicting Evidence/Cherrypicking"]
_ALIASES = {
    "claim_id": ("claim_id", "id", "claimId"),
    "claim": ("claim", "text", "claim_text"),
    "label": ("label", "verdict", "gold_label"),
    "justification": ("justification", "gold_justification"),
    "questions": ("questions", "qa_pairs", "evidence"),
    "question": ("question", "q"),
    "answers": ("answers", "answer", "a"),
    "answer": ("answer", "text", "answer_text"),
    "answer_type": ("answer_type", "type"),
    "source_url": ("source_url", "url", "original_url"),
    "cached_source_url": ("cached_source_url", "cached_url", "archive_url"),
    "source_medium": ("source_medium", "medium"),
    "store_id": ("store_id", "knowledge_store_id", "doc_id", "document_id"),
}


def _get(d: Dict, key: str, default=None):
    for name in _ALIASES.get(key, (key,)):
        if isinstance(d, dict) and name in d and d[name] not in (None, ""):
            return d[name]
    return default


@dataclass
class Passage:
    """One evidence passage, carrying its lineage."""
    passage_id: str
    text: str
    root_id: str                     # PRIMARY lineage: canonical document
    document_id: str
    content_hash: str
    question: str = ""
    answer_type: str = ""
    source_url: str = ""
    cached_source_url: str = ""
    source_medium: str = ""
    store_id: str = ""
    is_gold: bool = True
    retrieval_score: float = 0.0
    content_family_id: str = ""      # sensitivity only
    provenance_note: str = ""        # e.g. "paraphrase of <id>"


def record_id(item: Dict[str, Any]) -> str:
    """Stable identity from claim text PLUS distinguishing metadata.

    Positional ids break under file reordering; claim text alone would merge two
    genuinely different records that happen to share wording (the same sentence
    said by different speakers on different dates is a different claim).
    """
    parts = [normalise_text(_get(item, "claim", "") or "")]
    for key in ("claim_date", "speaker", "reporting_source",
                "original_claim_url", "fact_checking_article"):
        parts.append(normalise_text(str(_get(item, key, "") or "")))
    digest = hashlib.blake2b("\x1f".join(parts).encode("utf-8"),
                             digest_size=12).hexdigest()
    return f"rec::{digest}"


def claim_text_hash(item_or_text) -> str:
    """Hash of the normalised CLAIM TEXT alone -- for leakage detection only."""
    text = item_or_text if isinstance(item_or_text, str) else \
        (_get(item_or_text, "claim", "") or "")
    return hashlib.blake2b(normalise_text(text).encode("utf-8"),
                           digest_size=12).hexdigest()


@dataclass
class ClaimRecord:
    claim_id: str
    claim: str
    label: str
    split: str
    passages: List[Passage] = field(default_factory=list)
    justification: str = ""
    record_id: str = ""
    claim_text_hash: str = ""
    is_blind: bool = False          # labels/evidence intentionally withheld
    # gold evidence retained alongside a retrieved record, for RECALL only --
    # never fed to the model, or a retrieval miss would look like a success
    gold_passages: List[Passage] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def root_ids(self) -> List[str]:
        return sorted({p.root_id for p in self.passages if p.root_id})

    @property
    def n_roots(self) -> int:
        return len(self.root_ids)

    def to_json(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("raw", None)
        return d


def load_split(path: str, split: str, add_content_family: bool = False,
               family_threshold: float = 0.80,
               missing_policy: str = "claim_local_unknown") -> List[ClaimRecord]:
    """Load one official AVeriTeC split file into ClaimRecords."""
    with open(path) as fh:
        data = json.load(fh)
    if isinstance(data, dict):
        data = data.get("data", data.get("claims", list(data.values())))
    out: List[ClaimRecord] = []
    for i, item in enumerate(data):
        rid = record_id(item)
        cid = str(_get(item, "claim_id", rid))      # official id when present
        rec = ClaimRecord(claim_id=cid, claim=_get(item, "claim", "") or "",
                          label=_get(item, "label", "") or "", split=split,
                          justification=_get(item, "justification", "") or "",
                          record_id=rid, claim_text_hash=claim_text_hash(item),
                          raw=item)
        for qi, q in enumerate(_get(item, "questions", []) or []):
            qtext = _get(q, "question", "") or ""
            answers = _get(q, "answers", []) or []
            if isinstance(answers, dict):
                answers = [answers]
            for ai, ans in enumerate(answers):
                if not isinstance(ans, dict):
                    ans = {"answer": str(ans)}
                url = _get(ans, "source_url", "") or ""
                cached = _get(ans, "cached_source_url", "") or ""
                sid = _get(ans, "store_id", "") or ""
                root = document_root_id(url=url, cached_url=cached, store_id=sid)
                text = _get(ans, "answer", "") or ""
                rec.passages.append(Passage(
                    passage_id=f"{cid}::q{qi}::a{ai}", text=text, root_id=root,
                    document_id=root, content_hash=content_hash(text),
                    question=qtext, answer_type=_get(ans, "answer_type", "") or "",
                    source_url=url, cached_source_url=cached,
                    source_medium=_get(ans, "source_medium", "") or "",
                    store_id=str(sid), is_gold=True))
        # A record with no label AND no evidence is a blind-test record; that is
        # intentional, not a completeness failure.
        rec.is_blind = (not rec.label) and (not rec.passages)
        if missing_policy == "claim_local_unknown":
            for pp in rec.passages:
                if not pp.root_id:
                    pp.root_id = pp.document_id = f"unknown::{rid}"
                    pp.provenance_note = "no usable source identity (claim-local unknown root)"
        elif missing_policy == "exclude":
            rec.passages = [pp for pp in rec.passages if pp.root_id]
        elif missing_policy == "separate_roots":
            for k, pp in enumerate(rec.passages):
                if not pp.root_id:
                    pp.root_id = pp.document_id = f"unknown::{rid}::{k}"
        if add_content_family and rec.passages:
            fams = content_family_ids([p.text for p in rec.passages],
                                      [p.root_id for p in rec.passages],
                                      threshold=family_threshold)
            for p, f in zip(rec.passages, fams):
                p.content_family_id = f
        out.append(rec)
    return out


def audit(records: Sequence[ClaimRecord]) -> Dict[str, Any]:  # noqa: C901
    """Everything Gate A needs, computed from the loaded records."""
    n = len(records)
    labels = Counter(r.label for r in records)
    n_roots = [r.n_roots for r in records]
    passages = [p for r in records for p in r.passages]
    missing_root = sum(1 for p in passages if not p.root_id)
    by_hash: Dict[str, set] = {}
    for p in passages:
        by_hash.setdefault(p.content_hash, set()).add(p.root_id)
    cross_url_dupes = sum(1 for roots in by_hash.values() if len(roots) > 1)
    root_to_docs: Dict[str, set] = {}
    for p in passages:
        if p.root_id:
            root_to_docs.setdefault(p.root_id, set()).add(
                p.cached_source_url or p.source_url)
    blind = [r for r in records if r.is_blind]
    unknown_roots = sum(1 for p in passages if p.root_id.startswith("unknown::"))
    return {
        "n_claims": n,
        "n_blind_records": len(blind),
        "is_blind_split": bool(n and len(blind) == n),
        "passages_with_unknown_root": unknown_roots,
        "n_passages": len(passages),
        "label_distribution": dict(labels),
        "claims_with_multiple_roots": int(sum(1 for k in n_roots if k >= 2)),
        "fraction_multi_root": (sum(1 for k in n_roots if k >= 2) / n) if n else 0.0,
        "mean_roots_per_claim": (sum(n_roots) / n) if n else 0.0,
        "passages_missing_root": missing_root,
        "fraction_missing_root": (missing_root / len(passages)) if passages else 0.0,
        "identical_text_across_different_roots": cross_url_dupes,
        "distinct_roots": len(root_to_docs),
        "empty_text_passages": sum(1 for p in passages if not p.text.strip()),
    }


def check_no_split_leakage(splits: Dict[str, Sequence[ClaimRecord]]) -> Dict[str, Any]:
    """Claim ids and claim texts must not repeat across splits."""
    ids = {k: {r.record_id for r in v} for k, v in splits.items()}
    # normalised text hash, matching the lineage module -- a weaker
    # lower()/strip() comparison UNDERCOUNTS overlap (it missed half of it here)
    texts = {k: {r.claim_text_hash for r in v if r.claim} for k, v in splits.items()}
    overlaps = {}
    names = sorted(splits)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            overlaps[f"{a}|{b}"] = {
                "shared_claim_ids": sorted(ids[a] & ids[b])[:20],
                "n_shared_ids": len(ids[a] & ids[b]),
                "n_shared_texts": len(texts[a] & texts[b]),
            }
    return {"overlaps": overlaps,
            "clean": all(v["n_shared_ids"] == 0 and v["n_shared_texts"] == 0
                         for v in overlaps.values())}


# ---------------------------------------------------------------------------
# Cross-split overlap: classify, then apply a frozen policy
# ---------------------------------------------------------------------------
def _meta(rec: ClaimRecord, key: str) -> str:
    return normalise_text(str(rec.raw.get(key) or "")) if isinstance(rec.raw, dict) else ""


def classify_cross_split(a: ClaimRecord, b: ClaimRecord) -> str:
    """Why do these two records share claim text?

    exact_duplicate      same claim url, or same fact-checking article AND date
    annotation_variant   same fact-checking article, differing surface metadata
                         -- one underlying claim annotated more than once
    same_text_diff_event same wording, different speaker or source and no shared
                         fact-checking article -- a different claim
    uncertain            anything else; kept for manual review
    """
    url_a, url_b = _meta(a, "original_claim_url"), _meta(b, "original_claim_url")
    fca_a, fca_b = _meta(a, "fact_checking_article"), _meta(b, "fact_checking_article")
    date_a, date_b = _meta(a, "claim_date"), _meta(b, "claim_date")
    spk_a, spk_b = _meta(a, "speaker"), _meta(b, "speaker")
    if url_a and url_a == url_b:
        return "exact_duplicate"
    if fca_a and fca_a == fca_b:
        return "exact_duplicate" if date_a == date_b else "annotation_variant"
    if spk_a and spk_b and spk_a != spk_b:
        return "same_text_diff_event"
    if date_a and date_b and date_a != date_b:
        return "same_text_diff_event"
    return "uncertain"


def cross_split_report(splits: Dict[str, Sequence[ClaimRecord]],
                       min_words: int = 5) -> Dict[str, Any]:
    """Every cross-split claim-text match, classified.

    `min_words` guards against removing training data because a short generic
    string matched.
    """
    by_hash: Dict[str, Dict[str, List[ClaimRecord]]] = {}
    for name, recs in splits.items():
        for r in recs:
            if len(normalise_text(r.claim).split()) < min_words:
                continue
            by_hash.setdefault(r.claim_text_hash, {}).setdefault(name, []).append(r)
    matches, counts = [], Counter()
    for h, per in by_hash.items():
        names = sorted(per)
        if len(names) < 2:
            continue
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                for ra in per[names[i]]:
                    for rb in per[names[j]]:
                        kind = classify_cross_split(ra, rb)
                        counts[kind] += 1
                        matches.append({
                            "claim_text_hash": h, "kind": kind,
                            "split_a": names[i], "record_a": ra.record_id,
                            "label_a": ra.label, "date_a": ra.raw.get("claim_date"),
                            "speaker_a": ra.raw.get("speaker"),
                            "split_b": names[j], "record_b": rb.record_id,
                            "label_b": rb.label, "date_b": rb.raw.get("claim_date"),
                            "speaker_b": rb.raw.get("speaker"),
                            "labels_agree": (ra.label == rb.label) if (ra.label and rb.label) else None,
                            "claim": ra.claim[:160]})
    return {"n_matches": len(matches), "by_kind": dict(counts), "matches": matches}


def leakage_safe_training_view(train: Sequence[ClaimRecord],
                               report: Dict[str, Any],
                               remove_kinds: Sequence[str] = ("exact_duplicate",
                                                              "annotation_variant"),
                               ) -> Dict[str, Any]:
    """Frozen policy: the official dev split is NEVER modified.

    A leakage-safe TRAINING view drops train records that duplicate a dev record
    under the listed kinds. `same_text_diff_event` is retained: identical wording
    from a different speaker on a different date is a different claim, and
    removing it would discard legitimate data.
    """
    drop = set()
    for m in report["matches"]:
        if m["kind"] not in remove_kinds:
            continue
        if m["split_a"] == "train" and m["split_b"] == "dev":
            drop.add(m["record_a"])
        elif m["split_b"] == "train" and m["split_a"] == "dev":
            drop.add(m["record_b"])
    kept = [r for r in train if r.record_id not in drop]
    return {"n_removed": len(drop), "removed_record_ids": sorted(drop),
            "n_kept": len(kept), "view": kept,
            "policy": {"remove_kinds": list(remove_kinds),
                       "dev_split_modified": False,
                       "test_labels_used": False}}
