"""Duplicated-hop robustness of clean-trained MuSiQue transport models.

Out-of-distribution evaluation only. Every model is a checkpoint trained on
clean chains; nothing here is trained or fine-tuned, and no duplication is used
for training. Calibration is fitted on clean validation examples and frozen
before any duplicated condition is scored.

Results are written per (variant, seed) so a run resumes by skipping completed
seeds, and the merge stage fails loudly if any cell is missing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ecnca.real.encode import (EmbeddingCache, HashEncoder,
                               SentenceTransformerEncoder)  # noqa: E402
from ecnca.real.musique import load_musique, split_linear  # noqa: E402
from ecnca.real.musique_calibration import (READOUTS, Calibration, fit,
                                            score)  # noqa: E402
from ecnca.real.musique_duplication import (DEDUP_VARIANTS, FUSION_VARIANTS,
                                            INTERVENTIONS,
                                            MULTIPLICITIES, SCOPES,
                                            base_stream, build_stream,
                                            ledger_mode, make_delivery_batch,
                                            resolve_stream, stream_stats)  # noqa: E402
from ecnca.real.near_duplicate import DETECTORS  # noqa: E402
from ecnca.real.near_duplicate import config as near_duplicate_config  # noqa: E402
from ecnca.real.musique_transport import MuSiQueTransport  # noqa: E402
from ecnca.real.transformer_transport import TransformerTransport  # noqa: E402

BGE_REVISION = "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"
RECURRENT = ("full", "filter_only", "no_provenance", "plain")
TRANSFORMER = ("transformer_ec", "transformer_filter_only", "transformer_plain")
# The dedup variants are not trained models: each is the provenance-free content
# path under a different duplicate detector, so any difference between them is
# the detector alone. `canonical_dedup` is exact matching; the other three are
# the near-duplicate defences that industrial pipelines actually deploy, at the
# frozen thresholds of ecnca/real/near_duplicate.py.
DERIVED = {name: "plain" for name in DEDUP_VARIANTS}
# The fusion arms are accounting controls on the same provenance-free content
# path, so they reuse the `plain` checkpoint rather than training their own.
DERIVED.update({name: "plain" for name in FUSION_VARIANTS})
DEFAULT_VARIANTS = ("full", "filter_only", "plain", "canonical_dedup",
                    "minhash_dedup", "simhash_dedup", "embed_dedup",
                    "dempster_fusion", "covariance_intersection",
                    "transformer_ec", "transformer_filter_only",
                    "transformer_plain")


def stable_validation(record_id: str) -> bool:
    return int(hashlib.blake2b(record_id.encode(), digest_size=2).hexdigest(),
               16) % 20 == 0


def populations(train_path, dev_path, hops):
    train = split_linear(load_musique(train_path))
    dev = split_linear(load_musique(dev_path))
    validation = [r for r in train if r.n_hops in (2, 3) and stable_validation(r.record_id)]
    return validation, [r for r in dev if r.n_hops in hops]


def build_model(variant, emb_dim, hidden):
    base = DERIVED.get(variant, variant)
    if base in TRANSFORMER:
        return TransformerTransport(emb_dim=emb_dim, hidden=hidden, variant=base)
    return MuSiQueTransport(emb_dim, hidden, base)


def checkpoint_path(run_dir, variant, seed):
    return Path(run_dir) / f"{DERIVED.get(variant, variant)}_s{seed}" / "ckpt.pt"


@torch.no_grad()
def forward_streams(model, streams, records, cache, device, batch_size):
    """Run one resolved delivery stream per record; return terminal predictions."""
    out = []
    for start in range(0, len(records), batch_size):
        chunk = slice(start, start + batch_size)
        batch = make_delivery_batch(streams[chunk], records[chunk], cache, device)
        out.append(model(batch)["prediction"].cpu())
    return torch.cat(out)


def candidate_pool(records, cache, device):
    """Fixed pool of unique gold terminal answers. Built once from the clean
    population and reused by every intervention, so no intervention can change
    the label space it is scored against."""
    terminal = torch.as_tensor(
        cache.encode([r.steps[-1].answer for r in records]), device=device).cpu()
    keys, pool, index = {}, [], []
    for vec in terminal:
        key = vec.numpy().tobytes()
        if key not in keys:
            keys[key] = len(pool)
            pool.append(vec)
        index.append(keys[key])
    return torch.stack(pool), torch.tensor(index), terminal


def condition_streams(records, intervention, multiplicity, scope, seed,
                      mode, paraphrases, embed=None):
    streams, credits, suppressed, events, covered = [], [], 0, 0, 0
    for record in records:
        raw = build_stream(record, intervention, multiplicity, scope, seed,
                           paraphrases)
        if intervention == "paraphrase":
            roots = {e.root_id for e in raw}
            covered += int(any((paraphrases or {}).get(r) for r in roots))
        kept, credit, sup = resolve_stream(raw, mode, embed=embed)
        streams.append(kept)
        credits.append(sum(credit.values()))
        suppressed += sup
        events += len(raw)
    return (streams, np.array(credits, dtype=float), suppressed, events,
            covered)


def evaluate_variant(variant, seed, args, validation, dev, cache, device):
    """Evaluate one (variant, seed). `args.hops` is recorded in the cell so a
    merged report can never be mistaken for a different hop group."""
    model = build_model(variant, cache.dim, args.hidden).to(device)
    path = checkpoint_path(args.checkpoints, variant, seed)
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    mode = ledger_mode(variant)
    # Only the embedding detector needs an encoder. It reuses the same frozen
    # BGE cache the content path uses, so the detector sees exactly the
    # representation the model sees and gets no private information.
    embed = ((lambda text: cache.encode([text])[0]) if mode == "embed"
             else None)

    # ---- clean pass on the evaluation population -------------------------
    pool, target_index, terminal = candidate_pool(dev, cache, device)
    clean_streams = [resolve_stream(base_stream(r), mode, embed=embed)[0]
                     for r in dev]
    clean_pred = forward_streams(model, clean_streams, dev, cache, device,
                                 args.batch_size)
    ceiling = np.array([len({e.root_id for e in base_stream(r)}) for r in dev],
                       dtype=float)
    clean_credit = np.array(
        [sum(resolve_stream(base_stream(r), mode, embed=embed)[1].values())
         for r in dev], dtype=float)

    # ---- calibration on CLEAN VALIDATION only, then frozen ---------------
    val_pool, val_index, _ = candidate_pool(validation, cache, device)
    val_streams = [resolve_stream(base_stream(r), mode, embed=embed)[0]
                   for r in validation]
    val_pred = forward_streams(model, val_streams, validation, cache, device,
                               args.batch_size)
    val_credit = torch.ones(len(validation))
    val_sim = val_pred @ val_pool.T
    calibration = {r: fit(val_sim, val_index, val_credit, r,
                          iters=args.calibration_iters) for r in READOUTS}

    rows = []
    for intervention in args.interventions:
        for scope in args.scopes:
            for multiplicity in args.multiplicities:
                if multiplicity == 1 and (intervention != args.interventions[0]
                                          or scope != args.scopes[0]):
                    continue  # 1x is the same condition for every intervention
                streams, credit, suppressed, events, covered = condition_streams(
                    dev, intervention, multiplicity, scope, args.intervention_seed,
                    mode, args.paraphrases, embed=embed)
                pred = forward_streams(model, streams, dev, cache, device,
                                       args.batch_size)
                sim = pred @ pool.T
                ratio = torch.as_tensor(credit / np.maximum(ceiling, 1e-9)).float()
                drift = float((pred - clean_pred).norm(dim=-1).mean())
                cell = {"variant": variant, "seed": seed,
                        "intervention": intervention, "scope": scope,
                        # For paraphrase: how many records had an audited pool.
                        # Zero means the condition degenerated to exact copies
                        # and must not be read as a paraphrase result.
                        "paraphrase_covered": covered,
                        "multiplicity": multiplicity, "ledger": mode,
                        "n_events": events, "n_suppressed": suppressed,
                        "credited_over_ceiling": float(ratio.mean()),
                        "prediction_drift": drift,
                        "cosine_error": float(
                            1 - (pred * terminal).sum(-1).mean())}
                for readout in READOUTS:
                    cell[readout] = score(sim, target_index, ratio,
                                          calibration[readout], readout)
                rows.append(cell)
    return {"variant": variant, "seed": seed, "ledger": mode,
            "hops": list(args.hop_group),
            "near_duplicate_config": (near_duplicate_config()
                                      if mode in DETECTORS else None),
            "params": sum(p.numel() for p in model.parameters()),
            "checkpoint": str(path),
            "calibration": {r: calibration[r].as_dict() for r in READOUTS},
            "n_dev": len(dev), "n_validation": len(validation),
            "n_candidates": int(pool.shape[0]), "rows": rows}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    root = "data/raw/musique/data"
    ap.add_argument("--train", default=f"{root}/musique_ans_v1.0_train.jsonl")
    ap.add_argument("--dev", default=f"{root}/musique_ans_v1.0_dev.jsonl")
    ap.add_argument("--checkpoints", default="results/architecture/transport")
    ap.add_argument("--out", default="results/musique_dup/eval")
    ap.add_argument("--cache", default="results/musique/bge_cache.npz")
    ap.add_argument("--paraphrase-file", default="")
    ap.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--hops", default="4")
    ap.add_argument("--interventions", default=",".join(INTERVENTIONS))
    ap.add_argument("--scopes", default=",".join(SCOPES))
    ap.add_argument("--multiplicities", default=",".join(map(str, MULTIPLICITIES)))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--validation-limit", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--encoder", choices=("hash", "bge"), default="bge")
    ap.add_argument("--emb-dim", type=int, default=64)
    ap.add_argument("--encode-batch", type=int, default=128)
    ap.add_argument("--calibration-iters", type=int, default=400)
    ap.add_argument("--intervention-seed", type=int, default=17)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    args.interventions = args.interventions.split(",")
    args.scopes = args.scopes.split(",")
    args.multiplicities = [int(x) for x in args.multiplicities.split(",")]
    hops = tuple(int(x) for x in args.hops.split(","))
    args.hop_group = hops

    encoder = (HashEncoder(args.emb_dim) if args.encoder == "hash"
               else SentenceTransformerEncoder("BAAI/bge-base-en-v1.5",
                                               BGE_REVISION, device=args.device,
                                               batch_size=args.encode_batch))
    cache = EmbeddingCache(args.cache, encoder)
    validation, dev = populations(args.train, args.dev, hops)
    if args.limit:
        dev = dev[:args.limit]
    validation = validation[:args.validation_limit]
    args.paraphrases = (json.loads(Path(args.paraphrase_file).read_text())["pool"]
                        if args.paraphrase_file else None)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    print(f"[dup] dev={len(dev)} validation={len(validation)} hops={hops} "
          f"device={args.device}", flush=True)
    for variant in args.variants.split(","):
        for seed in [int(s) for s in args.seeds.split(",")]:
            cell = out / f"{variant}_s{seed}.json"
            if cell.exists() and not args.force:
                print(f"[dup] skip {variant} s{seed} (already complete)", flush=True)
                continue
            start = time.time()
            result = evaluate_variant(variant, seed, args, validation, dev,
                                      cache, args.device)
            result["eval_seconds"] = time.time() - start
            if args.device.startswith("cuda"):
                result["peak_memory_mb"] = torch.cuda.max_memory_allocated() / 2**20
            cell.write_text(json.dumps(result, indent=2) + "\n")
            print(f"[dup] {variant} s{seed}: {len(result['rows'])} conditions "
                  f"in {result['eval_seconds']:.1f}s", flush=True)
    cache.save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
