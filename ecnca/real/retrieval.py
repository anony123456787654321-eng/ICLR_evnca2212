"""Official precomputed AVeriTeC retrieval, loaded into ClaimRecords.

Two official artefacts, both JSONL:

  dev_top_k_sentences.json   {claim_id, claim, top_100: [{sentence, url}]}
  dev_top_3_rerank_qa.json   {claim_id, claim, evidence: [{question, answer, url}]}

`claim_id` is the POSITIONAL index into dev.json, which carries no id field of
its own, so alignment is by index and is asserted rather than assumed.

Passages from one document are canonicalised to ONE root.  Overlapping
retrieved sentences from the same page stay separate content messages sharing a
single evidence lineage -- which is the whole point: more text from one source
is more computation, not more evidence.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence

from .averitec import ClaimRecord, Passage
from .lineage import content_hash, document_root_id

SOURCES = ("top_k_sentences", "rerank_qa")


class MalformedRetrievalError(ValueError):
    pass


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path) as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise MalformedRetrievalError(
                    f"{os.path.basename(path)} line {i + 1}: {exc}") from exc
    if not rows:
        raise MalformedRetrievalError(f"{path} contains no records")
    return rows


def load_retrieval(path: str, source: str) -> Dict[int, List[Dict[str, Any]]]:
    """claim index -> ranked list of {text, url, rank, question}."""
    if source not in SOURCES:
        raise ValueError(f"unknown source {source!r}; expected one of {SOURCES}")
    out: Dict[int, List[Dict[str, Any]]] = {}
    for row in _read_jsonl(path):
        if "claim_id" not in row:
            raise MalformedRetrievalError("record without claim_id")
        cid = int(row["claim_id"])
        items: List[Dict[str, Any]] = []
        if source == "top_k_sentences":
            for rank, e in enumerate(row.get("top_100") or []):
                items.append({"text": e.get("sentence", "") or "",
                              "url": e.get("url", "") or "", "rank": rank,
                              "question": ""})
        else:
            for rank, e in enumerate(row.get("evidence") or []):
                items.append({"text": e.get("answer", "") or "",
                              "url": e.get("url", "") or "", "rank": rank,
                              "question": e.get("question", "") or ""})
        out[cid] = items
    return out


def gold_roots(rec: ClaimRecord) -> set:
    return {p.root_id for p in rec.passages if p.root_id}


def attach_retrieved(dev_records: Sequence[ClaimRecord],
                     retrieval: Dict[int, List[Dict[str, Any]]],
                     top_k: int = 10, source: str = "top_k_sentences",
                     require_alignment: bool = True) -> List[ClaimRecord]:
    """Return NEW records whose passages are the retrieved ones.

    The gold passages are kept on the record (as `gold_passages`) so retrieval
    recall can be measured, but they are NOT fed to the model: mixing them in
    would make a retrieval miss look like an aggregation success.
    """
    out: List[ClaimRecord] = []
    for idx, rec in enumerate(dev_records):
        items = retrieval.get(idx)
        if items is None:
            if require_alignment:
                raise MalformedRetrievalError(
                    f"no retrieval for dev index {idx}; alignment is positional")
            items = []
        gold = gold_roots(rec)
        new = ClaimRecord(claim_id=rec.claim_id, claim=rec.claim, label=rec.label,
                          split=rec.split, justification=rec.justification,
                          record_id=rec.record_id, claim_text_hash=rec.claim_text_hash,
                          is_blind=rec.is_blind, raw=rec.raw)
        for it in items[:top_k]:
            root = document_root_id(url=it["url"])
            text = it["text"]
            new.passages.append(Passage(
                passage_id=f"{rec.record_id}::ret{it['rank']}", text=text,
                root_id=root or f"unknown::{rec.record_id}",
                document_id=root or f"unknown::{rec.record_id}",
                content_hash=content_hash(text), question=it.get("question", ""),
                source_url=it["url"], is_gold=False,
                # rank -> score, so rank 0 is the strongest and ties are impossible
                retrieval_score=1.0 / (1.0 + it["rank"]),
                provenance_note=f"retrieved rank {it['rank']} ({source})"))
        # gold-evidence match indicator, per retrieved passage
        for p in new.passages:
            p.is_gold = p.root_id in gold
        new.gold_passages = list(rec.passages)          # kept for recall only
        out.append(new)
    return out


def retrieval_recall(records: Sequence[ClaimRecord]) -> Dict[str, float]:
    """How much of the GOLD lineage did retrieval actually surface?

    Reported separately from accuracy so a retrieval miss is never scored as an
    aggregation failure.
    """
    doc_hits = doc_tot = 0
    claims_any = claims_all = 0
    for r in records:
        gold = {p.root_id for p in getattr(r, "gold_passages", []) if p.root_id}
        got = {p.root_id for p in r.passages if p.root_id}
        if not gold:
            continue
        hit = gold & got
        doc_hits += len(hit); doc_tot += len(gold)
        claims_any += int(bool(hit))
        claims_all += int(gold <= got)
    n = sum(1 for r in records if {p.root_id for p in getattr(r, "gold_passages", []) if p.root_id})
    return {"root_recall": doc_hits / max(doc_tot, 1),
            "claims_with_any_gold_root": claims_any / max(n, 1),
            "claims_with_all_gold_roots": claims_all / max(n, 1),
            "n_scored": n}
