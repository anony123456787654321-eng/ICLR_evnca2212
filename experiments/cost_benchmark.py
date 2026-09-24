"""What does evidence conservation cost, head to head against its alternatives?

The paper claims conservation is free on clean data in task terms. It does not
yet answer the separate question a practitioner asks first: what does the ledger
cost to run, relative to no defence at all and relative to the deduplication
methods it is competing with. This measures that.

Two costs are separated, because they behave differently.

`ledger` is the cost of resolving a delivery stream: pure accounting, no model.
This is where the near-duplicate detectors are expensive, because each arrival
is compared against everything already accepted, while the evidence-conserving
join is a dictionary lookup per arrival.

`forward` is the cost of the content path on whatever survives resolution. This
is where conservation *saves*, because suppressed arrivals are never encoded and
never enter the sequence.

Reporting only one of the two would be misleading in opposite directions, so
both are reported per condition together with the arrival counts that explain
them. Nothing here is trained, and no frozen result is read or overwritten.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ecnca.real.encode import (EmbeddingCache, HashEncoder,  # noqa: E402
                               SentenceTransformerEncoder)
from ecnca.real.musique import load_musique, split_linear  # noqa: E402
from ecnca.real.musique_duplication import (DEDUP_VARIANTS,  # noqa: E402
                                            base_stream, build_stream,
                                            ledger_mode, make_delivery_batch,
                                            resolve_stream)
from ecnca.real.near_duplicate import DETECTORS  # noqa: E402
from ecnca.real.near_duplicate import config as near_duplicate_config  # noqa: E402
from ecnca.real.musique_transport import MuSiQueTransport  # noqa: E402

BGE_REVISION = "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"

# One representative variant per defence, all on the same content path, so the
# forward column is comparable and only the resolution differs.
DEFENCES = ("full", "filter_only", "plain", "canonical_dedup",
            "minhash_dedup", "simhash_dedup", "embed_dedup")


def _percentile(values, q):
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=float), q))


def ledger_cost(records, defence, intervention, multiplicity, scope,
                intervention_seed, embed, repeats):
    """Wall-clock of stream resolution, plus the arrivals it admits.

    The stream is built once and resolved `repeats` times, so construction cost
    is excluded and only resolution is timed. The minimum over repeats is
    reported alongside the mean because it is the measurement least polluted by
    unrelated scheduling.
    """
    mode = ledger_mode(defence)
    streams = [build_stream(r, intervention, multiplicity, scope,
                            intervention_seed) for r in records]
    arrivals = sum(len(s) for s in streams)

    per_repeat, resolved = [], None
    for _ in range(repeats):
        start = time.perf_counter()
        resolved = [resolve_stream(s, mode, embed=embed) for s in streams]
        per_repeat.append(time.perf_counter() - start)
    admitted = sum(len(k) for k, _, _ in resolved)
    credited = sum(sum(c.values()) for _, c, _ in resolved)
    suppressed = sum(sup for _, _, sup in resolved)
    return {"arrivals": arrivals, "admitted": admitted,
            "suppressed": suppressed, "credited_evidence": credited,
            "ledger_seconds_mean": statistics.fmean(per_repeat),
            "ledger_seconds_min": min(per_repeat),
            "ledger_us_per_arrival": 1e6 * min(per_repeat) / max(arrivals, 1),
            "resolved": [k for k, _, _ in resolved]}


@torch.no_grad()
def forward_cost(model, streams, records, cache, device, batch_size, repeats):
    """Wall-clock and peak memory of the content path on the admitted stream."""
    if device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        for begin in range(0, len(records), batch_size):
            chunk = slice(begin, begin + batch_size)
            batch = make_delivery_batch(streams[chunk], records[chunk], cache,
                                        device)
            model(batch)["prediction"]
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        timings.append(time.perf_counter() - start)
    out = {"forward_seconds_mean": statistics.fmean(timings),
           "forward_seconds_min": min(timings)}
    if device.startswith("cuda"):
        out["forward_peak_memory_mb"] = (
            torch.cuda.max_memory_allocated() / 2 ** 20)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    root = "data/raw/musique/data"
    ap.add_argument("--dev", default=f"{root}/musique_ans_v1.0_dev.jsonl")
    ap.add_argument("--checkpoints", default="results/architecture/transport")
    ap.add_argument("--out", default="results/cost/report.json")
    ap.add_argument("--cache", default="results/musique/bge_cache.npz")
    ap.add_argument("--defences", default=",".join(DEFENCES))
    ap.add_argument("--interventions", default="exact,overlap")
    ap.add_argument("--multiplicities", default="1,4,16")
    ap.add_argument("--scope", default="one_hop")
    ap.add_argument("--hops", default="4")
    ap.add_argument("--seed", type=int, default=0,
                    help="which trained checkpoint supplies the content path")
    ap.add_argument("--intervention-seed", type=int, default=17)
    ap.add_argument("--limit", type=int, default=256)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--encoder", choices=("hash", "bge"), default="bge")
    ap.add_argument("--emb-dim", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-model", action="store_true",
                    help="ledger costs only; skips every checkpoint")
    a = ap.parse_args()

    hops = tuple(int(x) for x in a.hops.split(","))
    encoder = (HashEncoder(a.emb_dim) if a.encoder == "hash"
               else SentenceTransformerEncoder("BAAI/bge-base-en-v1.5",
                                               BGE_REVISION, device=a.device))
    cache = EmbeddingCache(a.cache, encoder)
    dev = [r for r in split_linear(load_musique(a.dev)) if r.n_hops in hops]
    if a.limit:
        dev = dev[:a.limit]
    print(f"[cost] {len(dev)} records, hops={hops}, device={a.device}",
          flush=True)

    model = None
    if not a.no_model:
        # One content path for every defence. The dedup variants and `plain`
        # share weights by construction, and `full`/`filter_only` differ only in
        # resolution, so a difference in the forward column is stream length and
        # nothing else.
        model = MuSiQueTransport(cache.dim, a.hidden, "plain").to(a.device)
        ckpt = Path(a.checkpoints) / f"plain_s{a.seed}" / "ckpt.pt"
        if ckpt.is_file():
            model.load_state_dict(torch.load(ckpt, map_location=a.device,
                                             weights_only=False)["model"])
            print(f"[cost] content path: {ckpt}", flush=True)
        else:
            print(f"[cost] WARNING: {ckpt} absent; timing an untrained content "
                  f"path. Timings are valid, task metrics are not reported "
                  f"here, so this changes nothing measured.", flush=True)
        model.eval()

    rows = []
    for defence in a.defences.split(","):
        mode = ledger_mode(defence)
        embed = ((lambda text: cache.encode([text])[0]) if mode == "embed"
                 else None)
        for intervention in a.interventions.split(","):
            for multiplicity in [int(x) for x in a.multiplicities.split(",")]:
                cost = ledger_cost(dev, defence, intervention, multiplicity,
                                   a.scope, a.intervention_seed, embed,
                                   a.repeats)
                streams = cost.pop("resolved")
                row = {"defence": defence, "ledger": mode,
                       "intervention": intervention,
                       "multiplicity": multiplicity, "scope": a.scope,
                       "n_records": len(dev), **cost}
                if model is not None:
                    row.update(forward_cost(model, streams, dev, cache,
                                            a.device, a.batch_size, a.repeats))
                    row["total_seconds_min"] = (row["ledger_seconds_min"]
                                                + row["forward_seconds_min"])
                rows.append(row)
                print(f"[cost] {defence:16s} {intervention:9s} x{multiplicity:<3d}"
                      f" arrivals={row['arrivals']:6d} admitted={row['admitted']:6d}"
                      f" ledger={row['ledger_seconds_min']:8.4f}s"
                      + (f" forward={row['forward_seconds_min']:8.4f}s"
                         if model is not None else ""), flush=True)

    report = {"rows": rows, "n_records": len(dev), "hops": list(hops),
              "repeats": a.repeats, "device": a.device,
              "encoder": a.encoder, "emb_dim": cache.dim,
              "content_path": "plain (shared by every defence)",
              "near_duplicate_config": near_duplicate_config(),
              "torch": torch.__version__,
              "note": "ledger and forward costs are reported separately because "
                      "conservation pays in the first and saves in the second"}
    if a.device.startswith("cuda"):
        report["device_name"] = torch.cuda.get_device_name()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    cache.save()
    print(f"[cost] wrote {out} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
