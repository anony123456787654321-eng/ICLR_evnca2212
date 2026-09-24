"""Train/evaluate the preregistered MuSiQue message-transport benchmark."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ecnca.real.encode import (EmbeddingCache, HashEncoder,
                               SentenceTransformerEncoder)  # noqa: E402
from ecnca.real.musique import load_musique, split_linear  # noqa: E402
from ecnca.real.musique_transport import (VARIANTS, MuSiQueTransport, cosine_loss,
                                          make_transport_batch,
                                          retrieval_metrics)  # noqa: E402
from ecnca.real.transformer_transport import (VARIANTS as TRANSFORMER_VARIANTS,
                                              TransformerTransport)  # noqa: E402

ALL_VARIANTS = VARIANTS + TRANSFORMER_VARIANTS


def build_model(variant, emb_dim, hidden):
    """Recurrent and Transformer backbones behind one name space.

    Both arms are trained by the identical loop below: same records, same cache,
    same optimizer, iterations, batch size and seeds. Neither is tuned
    separately, which is what makes the head-to-head fair.
    """
    if variant in TRANSFORMER_VARIANTS:
        return TransformerTransport(emb_dim=emb_dim, hidden=hidden, variant=variant)
    return MuSiQueTransport(emb_dim, hidden, variant)


def is_filter(variant):
    return variant in ("filter_only", "transformer_filter_only")

BGE_REVISION = "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"


def stable_validation(record_id):
    return int(hashlib.blake2b(record_id.encode(), digest_size=2).hexdigest(), 16) % 20 == 0


def populations(train_path, dev_path):
    train = split_linear(load_musique(train_path))
    dev = split_linear(load_musique(dev_path))
    fit = [r for r in train if r.n_hops in (2, 3) and not stable_validation(r.record_id)]
    validation = [r for r in train if r.n_hops in (2, 3) and stable_validation(r.record_id)]
    by_hops = {h: [r for r in dev if r.n_hops == h] for h in (2, 3, 4)}
    return fit, validation, by_hops


def encoder_for(args):
    if args.encoder == "hash":
        return HashEncoder(args.emb_dim)
    return SentenceTransformerEncoder("BAAI/bge-base-en-v1.5", BGE_REVISION,
                                      device=args.device, batch_size=args.encode_batch)


def sample(records, size, rng):
    return [records[rng.randrange(len(records))] for _ in range(size)]


@torch.no_grad()
def evaluate(model, records, cache, device, batch_size=64):
    model.eval()
    predictions, targets = [], []
    for start in range(0, len(records), batch_size):
        batch = make_transport_batch(records[start:start + batch_size], cache, device)
        output = model(batch)
        index = batch["n_hops"] - 1
        target = batch["answer"][torch.arange(len(index), device=device), index]
        predictions.append(output["prediction"].cpu())
        targets.append(target.cpu())
    prediction, target = torch.cat(predictions), torch.cat(targets)
    # The fixed candidate pool is the unique gold terminal answer embedding in
    # this evaluation population. Nearest-candidate top1 is exact-answer
    # retrieval, not open-ended QA generation.
    keys, candidate, target_index = {}, [], []
    for vector in target:
        key = vector.numpy().tobytes()
        if key not in keys:
            keys[key] = len(candidate); candidate.append(vector)
        target_index.append(keys[key])
    candidate = torch.stack(candidate)
    metrics = retrieval_metrics(prediction, target, candidate,
                                torch.tensor(target_index))
    # Chance normalisation makes hop groups with different pool sizes
    # comparable: 0 is chance, 1 is perfect.
    k = len(candidate)
    chance_top1 = 1.0 / k
    chance_mrr = sum(1.0 / r for r in range(1, k + 1)) / k
    metrics["chance_top1"] = chance_top1
    metrics["chance_mrr"] = chance_mrr
    metrics["norm_top1"] = (metrics["top1"] - chance_top1) / (1.0 - chance_top1)
    metrics["norm_mrr"] = (metrics["mrr"] - chance_mrr) / (1.0 - chance_mrr)
    return {**metrics, "n": len(records), "n_candidates": k}


def structural_probe(model, record, cache, device):
    model.eval()
    batch = make_transport_batch([record], cache, device)
    with torch.no_grad():
        output = model(batch, collect=True)
        versions = output["versions"]
        exact, credit_16 = model.ledger_readout(
            versions, model.variant, [record.n_hops - 1] * 16)
        cycle, cycle_credit = model.ledger_readout(
            versions, model.variant, list(range(record.n_hops)) * 4)
        latest = versions[:, record.n_hops - 1]
        drift = float((exact - latest).norm())
        cycle_drift = float((cycle - (versions[:, 0] if is_filter(model.variant)
                                      else latest)).norm())
        # Credited evidence against the ceiling across the redelivery sweep.
        multiplicity = {}
        for m in (1, 2, 4, 8, 16):
            _, credit = model.ledger_readout(
                versions, model.variant, [record.n_hops - 1] * m)
            multiplicity[str(m)] = credit
    return {"exact_redelivery_payload_drift": drift,
            "cycle_payload_drift": cycle_drift,
            "credit_by_multiplicity": multiplicity,
            "redelivery_credit_ratio": credit_16,
            "cycle_credit_ratio": cycle_credit}


def main():
    ap = argparse.ArgumentParser()
    root = "data/raw/musique/data"
    ap.add_argument("--train", default=f"{root}/musique_ans_v1.0_train.jsonl")
    ap.add_argument("--dev", default=f"{root}/musique_ans_v1.0_dev.jsonl")
    ap.add_argument("--out", default="results/musique/smoke")
    ap.add_argument("--cache", default="results/musique/bge_cache.npz")
    ap.add_argument("--variants", default=",".join(VARIANTS),
                    help="any of: " + ",".join(ALL_VARIANTS))
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--emb-dim", type=int, default=64)
    ap.add_argument("--encoder", choices=("hash", "bge"), default="hash")
    ap.add_argument("--encode-batch", type=int, default=128)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    fit, validation, dev = populations(args.train, args.dev)
    encoder = encoder_for(args)
    cache = EmbeddingCache(args.cache, encoder)
    emb_dim = encoder.dim
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in [int(x) for x in args.seeds.split(",")]:
        for variant in args.variants.split(","):
            torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
            rng = random.Random(seed)
            model = build_model(variant, emb_dim, args.hidden).to(args.device)
            if args.device.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats()
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
            start_time = time.time()
            for iteration in range(1, args.iters + 1):
                model.train(); optimizer.zero_grad(set_to_none=True)
                batch = make_transport_batch(sample(fit, args.batch_size, rng),
                                             cache, args.device)
                output = model(batch)
                loss, parts = cosine_loss(output, batch)
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if iteration == 1 or iteration % max(args.iters // 5, 1) == 0:
                    print(f"[{variant} s{seed}] {iteration}/{args.iters} "
                          f"loss={loss.item():.4f}", flush=True)
            metrics = {str(h): evaluate(model, records, cache, args.device)
                       for h, records in dev.items()}
            probe = structural_probe(model, dev[4][0], cache, args.device)
            peak_mb = (torch.cuda.max_memory_allocated() / 2**20
                       if args.device.startswith("cuda") else None)
            row = {"variant": variant, "seed": seed,
                   "backbone": ("transformer" if variant in TRANSFORMER_VARIANTS
                                else "recurrent"),
                   "params": sum(p.numel() for p in model.parameters()),
                   "train_seconds": time.time() - start_time,
                   "peak_memory_mb": peak_mb,
                   "validation": evaluate(model, validation[:512], cache, args.device),
                   "dev_by_hops": metrics, "probe": probe}
            rows.append(row)
            run = out / f"{variant}_s{seed}"; run.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "args": vars(args)}, run / "ckpt.pt")
            (run / "result.json").write_text(json.dumps(row, indent=2) + "\n")
    cache.save()
    report = {"dataset": "MuSiQue-Answerable v1.0",
              "fit_records": len(fit), "validation_records": len(validation),
              "dev_counts": {str(k): len(v) for k, v in dev.items()}, "rows": rows}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
