"""Directional against scalar credit in a learned message-passing model.

Pre-registered in docs/DIRECTIONAL_PREREGISTRATION.md (experiment A2), which
was committed before this script produced any result.

The model, task and training loop are those of experiments/architecture_matrix
.py, so the only new ingredient is the credit form. Every arm shares one
content path. `ec` and `plain` differ in whether a root is credited once or per
arrival; `matrix` and `scalar` differ in whether that credit keeps the
direction of the root's observation or is spread evenly with the same trace.

Observation rank is the experimental axis. With rank-one observations some
latent directions are never observed, and a scalar credit form claims support
along them anyway. With full-rank observations there is little direction to
lose, so the gap between the forms should shrink.
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
from ecnca.neural.model import coverage, gaussian_nll  # noqa: E402
from ecnca.neural.mpnn import MPNNEvidence  # noqa: E402

ARMS = {
    "ec_matrix": ("mpnn_ec", "matrix"),
    "ec_scalar": ("mpnn_ec", "scalar"),
    "plain_matrix": ("mpnn_plain", "matrix"),
    "plain_scalar": ("mpnn_plain", "scalar"),
    # Added by amendment 2 of the pre-registration, before any output.
    "ec_scalar_fit": ("mpnn_ec", "scalar_fit"),
}
COPIES = 16


def evaluate(model, batch, steps):
    model.eval()
    with torch.no_grad():
        out = model(batch, steps=steps)
    mu, Lam, truth = out["mu"], out["Lam"], batch["x_true"]
    return {"nll": float(gaussian_nll(mu, Lam, truth).mean()),
            "coverage": float(coverage(mu, Lam, truth)),
            "rmse": float(((mu - truth.unsqueeze(1)) ** 2).mean().sqrt())}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results/directional_synthetic")
    ap.add_argument("--obs-dims", default="1,2,4")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--cells", type=int, default=8)
    ap.add_argument("--roots", type=int, default=3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--topology", default="path")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--scale-lr", type=float, default=None,
                    help="separate learning rate for the learned scale of the "
                         "scalar_fit arm; a robustness check, not registered")
    args = ap.parse_args()
    os.chdir(Path(__file__).resolve().parent.parent)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for obs_dim in [int(x) for x in args.obs_dims.split(",")]:
        spec = BatchSpec(n_cells=args.cells, dim=4, obs_dim=obs_dim,
                         max_roots=8, max_occurrences=96,
                         topology=args.topology)
        dest = out_dir / f"obs{obs_dim}.json"
        rows = []
        for seed in [int(x) for x in args.seeds.split(",")]:
            # One held-out set per seed and rank, shared by every arm.
            clean = make_batch(spec, args.batch_size, args.roots,
                               rng=np.random.default_rng(seed + 9000),
                               device=args.device)
            held = {"clean": clean,
                    "duplicate": duplicate_batch(
                        clean, COPIES, np.random.default_rng(seed + 9100))}
            for arm in args.arms.split(","):
                variant, form = ARMS[arm]
                torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
                model = MPNNEvidence(dim=spec.dim, obs_dim=spec.obs_dim,
                                     hidden=args.hidden, variant=variant,
                                     credit_form=form).to(args.device)
                scale = [p for n, p in model.named_parameters()
                         if n == "log_scale"]
                rest = [p for n, p in model.named_parameters()
                        if n != "log_scale"]
                groups = [{"params": rest}]
                if scale:
                    groups.append({"params": scale,
                                   "lr": args.scale_lr or 3e-4})
                opt = torch.optim.AdamW(groups, lr=3e-4)
                start = time.time()
                for it in range(1, args.iters + 1):
                    model.train(); opt.zero_grad(set_to_none=True)
                    batch = make_batch(spec, args.batch_size, args.roots,
                                       rng=np.random.default_rng(
                                           seed * 100003 + it),
                                       device=args.device)
                    o = model(batch, steps=args.steps)
                    loss = gaussian_nll(o["mu"], o["Lam"],
                                        batch["x_true"]).mean()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                row = {"arm": arm, "variant": variant, "credit_form": form,
                       "obs_dim": obs_dim, "seed": seed,
                       "train_seconds": time.time() - start,
                       "learned_scale": (float(model.log_scale.exp())
                                         if hasattr(model, "log_scale")
                                         else None),
                       "scale_lr": args.scale_lr,
                       "task": {k: evaluate(model, b, args.steps)
                                for k, b in held.items()}}
                rows.append(row)
                print(f"[dir] obs{obs_dim} {arm:>12} s{seed} "
                      f"clean={row['task']['clean']['nll']:.4f} "
                      f"dup={row['task']['duplicate']['nll']:.4f} "
                      f"({row['train_seconds']:.0f}s)", flush=True)
        dest.write_text(json.dumps(
            {"experiment": "A2", "preregistration":
             "docs/DIRECTIONAL_PREREGISTRATION.md",
             "spec": {"dim": 4, "obs_dim": obs_dim, "cells": args.cells,
                      "roots": args.roots, "steps": args.steps,
                      "iters": args.iters, "copies": COPIES},
             "rows": rows}, indent=2) + "\n")
        print(f"[dir] wrote {dest}", flush=True)


if __name__ == "__main__":
    main()
