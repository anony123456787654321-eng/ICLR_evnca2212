"""Preregistered content-family lineage audit on frozen AVeriTeC models."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ecnca.real.averitec import ClaimRecord, Passage, load_split  # noqa: E402
from ecnca.real.baselines import build  # noqa: E402
from ecnca.real.batching import make_batch  # noqa: E402
from ecnca.real.content_family import (apply_content_families, choose_threshold,
                                       deterministic_perturbations,
                                       surface_similarity, training_pairs)  # noqa: E402
from ecnca.real.encode import EmbeddingCache  # noqa: E402
from ecnca.real.lineage import content_hash  # noqa: E402
from ecnca.real.retrieval import attach_retrieved, load_retrieval  # noqa: E402


BGE_REVISION = "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"


class CacheOnlyEncoder:
    name = (f"BAAI/bge-base-en-v1.5@{BGE_REVISION}"
            "|len512|norm|instr=False")
    dim = 768

    def encode(self, texts):  # pragma: no cover
        raise RuntimeError(f"frozen BGE cache lacks {len(texts)} requested texts")


@torch.no_grad()
def predict(model, records, cache, bs=32, max_passages=256):
    """Evaluate a frozen model without importing the Gate-C CLI module."""
    model.eval()
    device = next(model.parameters()).device
    probabilities, labels, credit = [], [], []
    for start in range(0, len(records), bs):
        batch = make_batch(records[start:start + bs], cache,
                           max_passages=max_passages, device=device)
        output = model(batch)
        probabilities.append(F.softmax(output["logits"], -1).cpu().numpy())
        labels.append(batch["labels"].cpu().numpy())
        credit.append(output["credited_evidence"].cpu().numpy())
    return (np.concatenate(probabilities), np.concatenate(labels),
            np.concatenate(credit))


def score_block(probabilities, labels):
    """Metrics required by the preregistered content-family audit."""
    keep = labels >= 0
    probabilities, labels = probabilities[keep], labels[keep]
    if len(labels) == 0:
        return {"accuracy": float("nan"), "nll": float("nan"), "n": 0}
    predicted = probabilities.argmax(1)
    chosen = np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1.0)
    return {"accuracy": float((predicted == labels).mean()),
            "nll": float(-np.log(chosen).mean()), "n": int(len(labels))}


def t95(values):
    values = np.asarray(values, dtype=np.float64)
    sem = float(stats.sem(values))
    if sem == 0:
        return [float(values.mean()), float(values.mean())]
    radius = float(stats.t.ppf(0.975, len(values) - 1) * sem)
    return [float(values.mean() - radius), float(values.mean() + radius)]


def dev_pair_audit(records, threshold):
    exact, negatives = [], []
    perturb = {name: [] for name in ("format", "deletion20", "block_reorder", "synonyms")}
    for record in records:
        for first, second in combinations(record.passages, 2):
            if first.root_id == second.root_id:
                continue
            if first.content_hash == second.content_hash:
                exact.append(surface_similarity(first.text, second.text))
            else:
                negatives.append(surface_similarity(first.text, second.text))
        for passage in record.passages:
            for name, changed in deterministic_perturbations(passage.text).items():
                perturb[name].append(surface_similarity(passage.text, changed))
    recall = lambda values: sum(x >= threshold for x in values) / max(len(values), 1)
    return {
        "real_exact_mirror_recall": recall(exact),
        "n_real_exact_pairs": len(exact),
        "false_link_rate": recall(negatives),
        "n_conservative_negative_pairs": len(negatives),
        "perturbation_recall": {name: recall(values) for name, values in perturb.items()},
        "n_perturbations": {name: len(values) for name, values in perturb.items()},
        "overall_perturbation_recall": recall([x for values in perturb.values()
                                                for x in values]),
    }


def sybilize(records, aliases=8, limit=200):
    out = []
    for record in records:
        if not record.passages:
            continue
        base = record.passages[0]
        clone = copy.deepcopy(record)
        clone.passages = []
        for index in range(aliases):
            root = f"sybil::{record.record_id}::{index}"
            clone.passages.append(Passage(
                passage_id=f"{record.record_id}::sybil{index}", text=base.text,
                root_id=root, document_id=root, content_hash=content_hash(base.text),
                question=base.question, source_url=f"https://alias{index}.invalid/story",
                retrieval_score=base.retrieval_score,
                provenance_note="adversarial root relabel", is_gold=base.is_gold))
        out.append(clone)
        if len(out) >= limit:
            break
    return out


def root_count(records):
    return sum(len({p.root_id for p in record.passages}) for record in records)


def evaluate_models(records, family_records, sybil, sybil_family, args):
    cache = EmbeddingCache(args.cache, CacheOnlyEncoder())
    seed_rows = []
    for seed in [int(x) for x in args.seeds.split(",")]:
        checkpoint = torch.load(Path(args.run) / f"ec_exact_s{seed}" / "ckpt.pt",
                                map_location=args.device, weights_only=False)
        model = build("ec_exact", emb_dim=768, hidden=128).to(args.device)
        model.load_state_dict(checkpoint["model"])
        p0, y0, c0 = predict(model, records, cache)
        pf, yf, cf = predict(model, family_records, cache)
        ps, ys, cs = predict(model, sybil, cache)
        psf, ysf, csf = predict(model, sybil_family, cache)
        clean, family = score_block(p0, y0), score_block(pf, yf)
        attack, defended = score_block(ps, ys), score_block(psf, ysf)
        seed_rows.append({
            "seed": seed,
            "clean_nll": clean["nll"], "family_nll": family["nll"],
            "nll_change": family["nll"] - clean["nll"],
            "clean_accuracy": clean["accuracy"], "family_accuracy": family["accuracy"],
            "accuracy_change": family["accuracy"] - clean["accuracy"],
            "clean_credited": float(c0.mean()), "family_credited": float(cf.mean()),
            "sybil_true_source_ratio": float(cs.mean()),
            "defended_true_source_ratio": float(csf.mean()),
            "sybil_nll": attack["nll"], "defended_nll": defended["nll"],
        })
    changes = [r["nll_change"] for r in seed_rows]
    defended = [r["defended_true_source_ratio"] for r in seed_rows]
    return {
        "per_seed": seed_rows,
        "nll_change_mean": float(np.mean(changes)), "nll_change_ci95": t95(changes),
        "accuracy_change_mean": float(np.mean([r["accuracy_change"] for r in seed_rows])),
        "sybil_true_source_ratio_mean": float(np.mean(
            [r["sybil_true_source_ratio"] for r in seed_rows])),
        "defended_true_source_ratio_mean": float(np.mean(defended)),
        "defended_true_source_ratio_ci95": t95(defended),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/averitec/data")
    ap.add_argument("--store-dir", default="data/averitec/data_store")
    ap.add_argument("--run", default="results/gate_c")
    ap.add_argument("--cache", default="results/gate_b/bge_cache.npz")
    ap.add_argument("--out", default="results/content_family")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    train = load_split(os.path.join(args.data_dir, "train.json"), "train")
    dev = load_split(os.path.join(args.data_dir, "dev.json"), "dev")
    positives, negatives = training_pairs(train)
    threshold, fit = choose_threshold(positives, negatives)
    pair_audit = dev_pair_audit(dev, threshold)

    retrieval = load_retrieval(
        os.path.join(args.store_dir, "dev_top_k_sentences.json"), "top_k_sentences")
    retrieved = attach_retrieved(dev, retrieval, top_k=10)
    families = apply_content_families(retrieved, threshold)
    sybil = sybilize(retrieved)
    sybil_family = apply_content_families(sybil, threshold)
    models = evaluate_models(retrieved, families, sybil, sybil_family, args)

    roots_before, roots_after = root_count(retrieved), root_count(families)
    sybil_roots_before, sybil_roots_after = root_count(sybil), root_count(sybil_family)
    criteria = {
        "real_mirror_recall_ge_095": pair_audit["real_exact_mirror_recall"] >= 0.95,
        "perturbation_recall_ge_080": pair_audit["overall_perturbation_recall"] >= 0.80,
        "dev_false_link_rate_le_001": pair_audit["false_link_rate"] <= 0.01,
        "defended_true_source_ratio_le_105":
            models["defended_true_source_ratio_ci95"][1] <= 1.05,
        "clean_nll_degradation_le_002": models["nll_change_ci95"][1] <= 0.02,
    }
    report = {
        "threshold": threshold, "train_fit": fit, "dev_pair_audit": pair_audit,
        "retrieved_roots_before": roots_before, "retrieved_roots_after": roots_after,
        "retrieved_root_merge_fraction": (roots_before - roots_after) / roots_before,
        "sybil_examples": len(sybil), "sybil_roots_before": sybil_roots_before,
        "sybil_roots_after": sybil_roots_after,
        "model_evaluation": models, "criteria": criteria,
        "pass": all(criteria.values()),
        "scope": "surface near-copy families; not semantic source equivalence",
    }
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
