"""Directional against scalar credit on real retrieval data.

Pre-registered in docs/DIRECTIONAL_PREREGISTRATION.md (experiment A3, with its
dated amendment), committed before this script produced any output.

Every answerable MuSiQue development record carries twenty paragraphs, a few of
them annotated as supporting and the rest real distractors chosen to resemble
the question. The top five paragraphs by similarity to the question form the
retrieved set, and the set is sufficient when it holds every supporting
paragraph. Whether retrieved evidence suffices is the question a system has to
answer before it commits to an answer.

Each retrieved source contributes rank-one credit along its embedding. Support
along a requirement direction is the credit that direction receives. The
directional score keeps the weakest direction, because a multi-hop answer fails
if any hop is unsupported however many sources back the others. The scalar
score averages the same support over directions, which is what a source count
implies. The two use identical inputs.

Only the label reads the support annotations. Requirement directions come from
the decomposition text with every reference to an earlier answer removed, so no
intermediate answer reaches a score.

This script writes per-record scores and labels. Areas under the curve and
their intervals are computed by analysis/directional_analysis.py, which applies
the pre-registered rules.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
K = 5
COPIES = 4


def requirement_text(step: str) -> str:
    """Decomposition step with every reference to an earlier answer removed."""
    text = re.sub(r"#\d+", " ", step).replace(">>", " ")
    return re.sub(r"\s+", " ", text).strip()


def load(path):
    records = []
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("answerable", True):
                records.append(r)
    return records


def support(dirs, emb):
    """Credited support along each requirement direction, one row per hop."""
    return ((dirs @ emb.T) ** 2).sum(1)


def scores(dirs, q, emb):
    s = support(dirs, emb)
    cos = emb @ q
    # Decomposition-aware baselines added by amendment 2: the best single
    # source per requirement, with no credit accumulated across sources.
    best = (dirs @ emb.T).max(1)
    return {"D": float(s.min()), "S": float(s.mean()),
            "sum_cos": float(cos.sum()), "max_cos": float(cos.max()),
            "best_min": float(best.min()), "mean_best": float(best.mean())}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dev",
                    default="data/raw/musique/data/musique_ans_v1.0_dev.jsonl")
    ap.add_argument("--encoder", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--revision",
                    default="a5beb1e3e68b9ab74eb54cfd186867f64f240e1a")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--cache", default="results/directional_musique/embeddings.npz")
    ap.add_argument("--out", default="results/directional_musique/per_record.json")
    a = ap.parse_args()
    os.chdir(Path(__file__).resolve().parent.parent)

    records = load(a.dev)
    passages, queries = [], []
    for r in records:
        passages += [p["title"] + ". " + p["paragraph_text"]
                     for p in r["paragraphs"]]
        queries.append(r["question"])
        queries += [requirement_text(d["question"])
                    for d in r["question_decomposition"]]

    cache = Path(a.cache)
    if cache.is_file():
        z = np.load(cache, allow_pickle=True)
        P, Q = z["P"], z["Q"]
    else:
        from sentence_transformers import SentenceTransformer
        try:
            model = SentenceTransformer(a.encoder, revision=a.revision,
                                        device=a.device)
        except TypeError:
            # Older sentence-transformers take no revision argument; the output
            # then records that the revision was not applied.
            model = SentenceTransformer(a.encoder, device=a.device)
            a.revision = "unpinned (library ignores revision)"
        enc = lambda xs: model.encode(xs, batch_size=64, convert_to_numpy=True,
                                      normalize_embeddings=True,
                                      show_progress_bar=False)
        P = enc(passages)
        Q = enc([QUERY_INSTRUCTION + t if t else QUERY_INSTRUCTION
                 for t in queries])
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, P=P, Q=Q)
    print(f"[a3] {len(records)} records, {len(passages)} passages, "
          f"{len(queries)} queries")

    rows, pi, qi, empty_steps = [], 0, 0, 0
    for r in records:
        n = len(r["paragraphs"])
        emb = P[pi:pi + n]; pi += n
        q = Q[qi]; qi += 1
        steps = r["question_decomposition"]
        texts = [requirement_text(d["question"]) for d in steps]
        dirs = Q[qi:qi + len(steps)]; qi += len(steps)
        keep = [i for i, t in enumerate(texts) if t]
        empty_steps += len(texts) - len(keep)
        # A step emptied by removing its reference falls back to the question.
        dirs = dirs[keep] if keep else q[None, :]

        ranked = np.argsort(-(emb @ q), kind="stable")
        top = ranked[:K]
        supporting = {i for i, p in enumerate(r["paragraphs"])
                      if p["is_supporting"]}
        sufficient = supporting <= set(top.tolist())

        clean = scores(dirs, q, emb[top])
        # Copies of the top-ranked paragraph carry its lineage. Lineage-aware
        # scoring counts that source once, so its scores equal the clean ones;
        # provenance-free scoring counts every arrival.
        arrivals = np.concatenate([top, np.repeat(top[:1], COPIES)])
        dup_none = scores(dirs, q, emb[arrivals])
        dup_lin = scores(dirs, q, emb[np.unique(arrivals)])
        rows.append({"id": r["id"], "n_hops": len(steps),
                     "n_supporting": len(supporting),
                     "sufficient": bool(sufficient),
                     "clean": clean,
                     "dup_lineage": dup_lin, "dup_none": dup_none})

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"experiment": "A3", "preregistration":
         "docs/DIRECTIONAL_PREREGISTRATION.md",
         "encoder": a.encoder, "revision": a.revision, "k": K,
         "copies": COPIES, "n_records": len(rows),
         "empty_requirement_steps": empty_steps, "rows": rows}, indent=1)
        + "\n")
    print(f"[a3] wrote {out}; empty requirement steps {empty_steps}")


if __name__ == "__main__":
    main()
