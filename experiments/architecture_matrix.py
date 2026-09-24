"""Head-to-head architecture matrix: does evidence conservation transfer?

The question is whether EC is a general evidence-accounting constraint or an
NCA-specific trick. Each backbone is run in an ordinary-accounting form and an
EC form that share a byte-identical content path, so the only difference is how
credit accumulates.

Backbones on the synthetic graph benchmark:

    mpnn_plain / mpnn_ec        Gilmer-style message passing
    nonbacktracking_mpnn        excludes the immediate reverse message
    plain / full                the existing recurrent NCA reference

Every variant sees the same batches, the same optimizer, the same iteration
count and the same seeds. Nothing is tuned per variant.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ecnca.neural.dataset import BatchSpec, duplicate_batch, make_batch  # noqa: E402
from ecnca.neural.model import SectorizedECNCA, coverage, gaussian_nll  # noqa: E402
from ecnca.neural.mpnn import MPNNEvidence, NonBacktrackingMPNN  # noqa: E402

BACKBONES = {
    "mpnn_ec": ("mpnn", lambda spec, h: MPNNEvidence(
        dim=spec.dim, obs_dim=spec.obs_dim, hidden=h, variant="mpnn_ec")),
    "mpnn_plain": ("mpnn", lambda spec, h: MPNNEvidence(
        dim=spec.dim, obs_dim=spec.obs_dim, hidden=h, variant="mpnn_plain")),
    "nonbacktracking_mpnn": ("nonbacktracking", lambda spec, h: NonBacktrackingMPNN(
        dim=spec.dim, obs_dim=spec.obs_dim, hidden=h)),
    "nca_ec": ("nca", lambda spec, h: SectorizedECNCA(
        dim=spec.dim, obs_dim=spec.obs_dim, hidden=h, use_sectors=False,
        use_provenance=True)),
    "nca_plain": ("nca", lambda spec, h: SectorizedECNCA(
        dim=spec.dim, obs_dim=spec.obs_dim, hidden=h, use_sectors=False,
        use_provenance=False)),
}
EC_FORM = {"mpnn_plain": "mpnn_ec", "nca_plain": "nca_ec"}
MULTIPLICITIES = (1, 2, 4, 8, 16)


def evaluate(model, batch, steps):
    """gaussian_nll and coverage take per-cell mu/Lam and broadcast x_true."""
    model.eval()
    with torch.no_grad():
        out = model(batch, steps=steps)
    mu, Lam, truth = out["mu"], out["Lam"], batch["x_true"]
    return {"rmse": float(((mu - truth.unsqueeze(1)) ** 2).mean().sqrt()),
            "nll": float(gaussian_nll(mu, Lam, truth).mean()),
            "coverage": float(coverage(mu, Lam, truth)),
            "mean_precision": float(Lam.diagonal(dim1=-2, dim2=-1).sum(-1).mean())}


def conservation_sweep(model, base, steps, rng_seed=1):
    """Credited evidence mass against redelivery multiplicity.

    `M` is trace of the credited precision above the prior, which every backbone
    reports, so the sweep compares the same quantity across architectures.
    """
    model.eval()
    ratios, first = {}, None
    with torch.no_grad():
        for m in MULTIPLICITIES:
            batch = base if m == 1 else duplicate_batch(
                base, m, np.random.default_rng(rng_seed))
            credited = float(model(batch, steps=steps)["M"].max())
            first = credited if first is None else first
            ratios[str(m)] = credited / max(first, 1e-9)
    return ratios


def regimes(spec, batch_size, n_roots, seed, device="cpu"):
    """Clean, duplicated, cyclic and reordered-delivery conditions."""
    out = {}
    out["clean"] = make_batch(spec, batch_size, n_roots,
                              rng=np.random.default_rng(seed), device=device)
    cyc = BatchSpec(**{**spec.__dict__, "topology": "cycle"})
    out["cycle"] = make_batch(cyc, batch_size, n_roots,
                              rng=np.random.default_rng(seed), device=device)
    out["duplicate"] = duplicate_batch(out["clean"], 4,
                                       np.random.default_rng(seed + 100))
    # Reordered/delayed delivery: same evidence, permuted arrival times.
    reordered = {k: (v.clone() if torch.is_tensor(v) else v)
                 for k, v in out["clean"].items()}
    for key in ("root_arrival", "occ_arrival"):
        if key in reordered and torch.is_tensor(reordered[key]):
            # Index tensors must live with the batch.  A CPU permutation works
            # in the default smoke test but fails once the driver is moved to
            # CUDA, which is exactly the environment of the confirmatory run.
            generator = torch.Generator(device=reordered[key].device).manual_seed(seed)
            perm = torch.randperm(reordered[key].shape[-1], generator=generator,
                                  device=reordered[key].device)
            reordered[key] = reordered[key][..., perm]
    out["reordered"] = reordered
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/architecture/graph")
    ap.add_argument("--variants", default=",".join(BACKBONES))
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--cells", type=int, default=8)
    ap.add_argument("--roots", type=int, default=3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--topology", default="path")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    spec = BatchSpec(n_cells=args.cells, dim=4, obs_dim=4, max_roots=8,
                     max_occurrences=96, topology=args.topology)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in [int(x) for x in args.seeds.split(",")]:
        # One evaluation set per seed, shared by every variant.
        held = regimes(spec, args.batch_size, args.roots, seed + 9000,
                       device=args.device)
        for name in args.variants.split(","):
            family, factory = BACKBONES[name]
            torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
            model = factory(spec, args.hidden).to(args.device)
            opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
            start = time.time()
            # Identical batch stream per seed: the data rng is keyed on the seed
            # and the iteration, never on the variant.
            for it in range(1, args.iters + 1):
                model.train(); opt.zero_grad(set_to_none=True)
                batch = make_batch(spec, args.batch_size, args.roots,
                                   rng=np.random.default_rng(seed * 100003 + it),
                                   device=args.device)
                o = model(batch, steps=args.steps)
                loss = gaussian_nll(o["mu"], o["Lam"], batch["x_true"]).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                if it == 1 or it % max(args.iters // 4, 1) == 0:
                    print(f"[{name} s{seed}] {it}/{args.iters} loss={loss.item():.4f}",
                          flush=True)
            row = {"variant": name, "family": family, "seed": seed,
                   "params": sum(p.numel() for p in model.parameters()),
                   "train_seconds": time.time() - start,
                   "task": {k: evaluate(model, b, args.steps) for k, b in held.items()},
                   "credit_by_multiplicity": conservation_sweep(
                       model, held["clean"], args.steps)}
            rows.append(row)
            print(f"[{name} s{seed}] credit sweep {row['credit_by_multiplicity']}",
                  flush=True)

    report = {"benchmark": "synthetic evidence graph",
              "spec": {"cells": args.cells, "roots": args.roots,
                       "topology": args.topology, "steps": args.steps,
                       "iters": args.iters, "batch_size": args.batch_size},
              "rows": rows}
    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"variants": sorted({r['variant'] for r in rows}),
                      "seeds": sorted({r['seed'] for r in rows}),
                      "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
