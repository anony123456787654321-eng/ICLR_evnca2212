"""Preregistered semantic-paraphrase Sybil audit on frozen AVeriTeC models."""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.content_family_audit import CacheOnlyEncoder, predict, score_block, t95  # noqa: E402
from ecnca.real.averitec import ClaimRecord, Passage, load_split  # noqa: E402
from ecnca.real.baselines import build  # noqa: E402
from ecnca.real.content_family import (apply_content_families, surface_similarity,
                                       training_pairs)  # noqa: E402
from ecnca.real.encode import EmbeddingCache, SentenceTransformerEncoder  # noqa: E402
from ecnca.real.lineage import content_hash, normalise_text  # noqa: E402


BGE_REVISION = "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"
NLI_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
NLI_REVISION = "6f5cf0a2b59cabb106aca4c287eed12e357e90eb"


def cosine(a, b):
    return float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))


def choose_semantic_threshold(positives, negatives, cache,
                              max_false_link_rate=0.005):
    texts = sorted({x for pair in positives + negatives for x in pair})
    vectors = cache.encode(texts)
    by_text = dict(zip(texts, vectors))
    pos = [cosine(by_text[a], by_text[b]) for a, b in positives]
    neg = [cosine(by_text[a], by_text[b]) for a, b in negatives]
    candidates = sorted(set(pos + neg), reverse=True)
    eligible = []
    for threshold in candidates:
        fpr = sum(x >= threshold for x in neg) / max(len(neg), 1)
        if fpr <= max_false_link_rate:
            recall = sum(x >= threshold for x in pos) / max(len(pos), 1)
            eligible.append((recall, threshold, fpr))
    if not eligible:
        raise ValueError("no semantic threshold satisfies the false-link budget")
    recall, threshold, fpr = max(eligible, key=lambda x: (x[0], x[1]))
    return float(threshold), {"train_positive_recall": float(recall),
                              "train_false_link_rate": float(fpr),
                              "n_positive_pairs": len(pos),
                              "n_negative_pairs": len(neg)}


def load_candidates(path):
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


@torch.inference_mode()
def bidirectional_entailment(rows, device="cpu", batch_size=32):
    tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL, revision=NLI_REVISION)
    model = AutoModelForSequenceClassification.from_pretrained(
        NLI_MODEL, revision=NLI_REVISION).to(device).eval()
    pairs, locations = [], []
    for row_index, row in enumerate(rows):
        for candidate_index, candidate in enumerate(row["candidates"]):
            pairs.extend([(row["source_text"], candidate["text"]),
                          (candidate["text"], row["source_text"])])
            locations.extend([(row_index, candidate_index, "source_to_candidate"),
                              (row_index, candidate_index, "candidate_to_source")])
    outputs = {}
    for start in range(0, len(pairs), batch_size):
        block = pairs[start:start + batch_size]
        encoded = tokenizer([x[0] for x in block], [x[1] for x in block],
                            padding=True, truncation=True, max_length=512,
                            return_tensors="pt").to(device)
        probability = model(**encoded).logits.softmax(-1).cpu().numpy()
        for location, probs in zip(locations[start:start + batch_size], probability):
            label_id = int(probs.argmax())
            outputs[location] = {"label": model.config.id2label[label_id].lower(),
                                 "probabilities": {model.config.id2label[i].lower(): float(p)
                                                   for i, p in enumerate(probs)}}
    return outputs


def retain_candidates(rows, cache, surface_threshold=0.87, nli=None):
    all_text = [r["source_text"] for r in rows]
    all_text += [c["text"] for r in rows for c in r["candidates"]]
    vectors = cache.encode(all_text)
    by_text = dict(zip(all_text, vectors))
    reports = []
    for row in rows:
        source = row["source_text"]
        source_words = len(normalise_text(source).split())
        seen, retained, candidates = set(), [], []
        for candidate in row["candidates"]:
            text = candidate["text"]
            words = len(normalise_text(text).split())
            sim = cosine(by_text[source], by_text[text]) if text else 0.0
            surf = surface_similarity(source, text) if text else 0.0
            norm = normalise_text(text)
            reasons = []
            if sim < 0.80: reasons.append("bge_below_080")
            if surf >= surface_threshold: reasons.append("surface_family_would_merge")
            if not (0.5 * source_words <= words <= 1.5 * source_words):
                reasons.append("length_outside_range")
            if not norm or norm in seen: reasons.append("empty_or_duplicate")
            if nli is not None:
                index = len(candidates)
                forward = nli[(len(reports), index, "source_to_candidate")]
                reverse = nli[(len(reports), index, "candidate_to_source")]
                if forward["label"] != "entailment" or reverse["label"] != "entailment":
                    reasons.append("not_bidirectional_entailment")
            if not reasons:
                retained.append(text); seen.add(norm)
            candidates.append({**candidate, "bge_cosine": sim,
                               "surface_similarity": surf,
                               "nli": ({"source_to_candidate": forward,
                                        "candidate_to_source": reverse}
                                       if nli is not None else None),
                               "retained": not reasons, "reasons": reasons})
        reports.append({**row, "candidates": candidates,
                        "retained": retained, "eligible": len(retained) >= 4})
    return reports


def make_attack_records(reports, multiplicity, defense, semantic_threshold):
    out = []
    for report in reports:
        if not report["eligible"]:
            continue
        texts = report["retained"][:multiplicity]
        retained_meta = {c["text"]: c for c in report["candidates"] if c["retained"]}
        passages = []
        for i, text in enumerate(texts):
            root = f"sybil::{report['record_id']}::{i}"
            if defense == "semantic" and \
                    retained_meta[text]["bge_cosine"] >= semantic_threshold:
                root = f"semantic::{report['record_id']}"
            passages.append(Passage(
                passage_id=f"{report['record_id']}::semantic_sybil::{i}", text=text,
                root_id=root, document_id=root, content_hash=content_hash(text),
                source_url=f"https://semantic-sybil-{i}.invalid/story",
                retrieval_score=1.0, provenance_note="paraphrase of one source",
                is_gold=False))
        record = ClaimRecord(
            claim_id=report["record_id"], claim=report["claim"],
            label=report["label"], split="dev", passages=passages,
            record_id=report["record_id"])
        out.append(record)
    if defense == "surface":
        out = apply_content_families(out, 0.87)
    # Canonical URL and declared URL lineage are identical here because the
    # adversary deliberately assigns each paraphrase a different URL.
    return out


def evaluate(reports, args, semantic_threshold):
    cache = EmbeddingCache(args.cache, CacheOnlyEncoder())
    rows = []
    for seed in [int(x) for x in args.seeds.split(",")]:
        checkpoint = torch.load(Path(args.run) / f"ec_exact_s{seed}" / "ckpt.pt",
                                map_location=args.device, weights_only=False)
        model = build("ec_exact", emb_dim=768, hidden=128).to(args.device)
        model.load_state_dict(checkpoint["model"])
        baseline_probs = None
        baseline_credit = None
        for defense in ("url", "canonical", "surface", "semantic"):
            for multiplicity in (1, 2, 4, 8):
                records = make_attack_records(reports, multiplicity, defense,
                                              semantic_threshold)
                probs, labels, credit = predict(model, records, cache)
                score = score_block(probs, labels)
                if defense == "url" and multiplicity == 1:
                    baseline_probs, baseline_credit = probs, credit
                assert baseline_probs is not None and baseline_credit is not None
                rows.append({"seed": seed, "defense": defense,
                             "multiplicity": multiplicity, "n": len(records),
                             "mean_credited": float(credit.mean()),
                             "credit_ratio_to_1x": float(
                                 credit.mean() / max(baseline_credit.mean(), 1e-12)),
                             "nll": score["nll"], "accuracy": score["accuracy"],
                             "prediction_flip_rate": float(
                                 (probs.argmax(1) != baseline_probs.argmax(1)).mean()),
                             "mean_probability_l1": float(
                                 np.abs(probs - baseline_probs).sum(1).mean())})
    return rows


def summarise(rows):
    summary = []
    for defense in ("url", "canonical", "surface", "semantic"):
        for multiplicity in (1, 2, 4, 8):
            block = [r for r in rows if r["defense"] == defense and
                     r["multiplicity"] == multiplicity]
            summary.append({"defense": defense, "multiplicity": multiplicity,
                            "n_claims": block[0]["n"],
                            "credit_ratio_mean": float(np.mean(
                                [r["credit_ratio_to_1x"] for r in block])),
                            "credit_ratio_ci95": t95(
                                [r["credit_ratio_to_1x"] for r in block]),
                            "nll_mean": float(np.mean([r["nll"] for r in block])),
                            "flip_rate_mean": float(np.mean(
                                [r["prediction_flip_rate"] for r in block]))})
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="results/semantic_sybil/candidates.jsonl")
    ap.add_argument("--data-dir", default="data/averitec/data")
    ap.add_argument("--run", default="results/gate_c")
    ap.add_argument("--cache", default="results/semantic_sybil/bge_cache.npz")
    ap.add_argument("--out", default="results/semantic_sybil")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    encoder = SentenceTransformerEncoder("BAAI/bge-base-en-v1.5", BGE_REVISION,
                                         device=args.device, batch_size=64)
    semantic_cache = EmbeddingCache(args.cache, encoder)
    train = load_split(os.path.join(args.data_dir, "train.json"), "train")
    positives, negatives = training_pairs(train)
    semantic_threshold, fit = choose_semantic_threshold(
        positives, negatives, semantic_cache)
    candidate_rows = load_candidates(args.candidates)
    nli = bidirectional_entailment(candidate_rows, args.device)
    reports = retain_candidates(candidate_rows, semantic_cache, nli=nli)
    # Frozen model batching also needs the claim embeddings. Populate them with
    # the same pinned encoder before switching to the cache-only evaluator.
    semantic_cache.encode([report["claim"] for report in reports])
    semantic_cache.save()
    eligible = sum(r["eligible"] for r in reports)
    if eligible == 0:
        raise RuntimeError("no claims retain four semantic paraphrases")
    rows = evaluate(reports, args, semantic_threshold)
    report = {"semantic_threshold": semantic_threshold, "train_fit": fit,
              "surface_threshold": 0.87, "population": len(reports),
              "eligible_population": eligible,
              "eligibility_rate": eligible / len(reports),
              "per_candidate": reports, "per_seed": rows,
              "summary": summarise(rows),
              "scope": "single-source semantic paraphrase Sybil attack"}
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "per_candidate"},
                     indent=2))


if __name__ == "__main__":
    main()
