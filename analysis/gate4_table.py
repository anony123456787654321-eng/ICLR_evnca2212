"""Final Gate 4 table, measured post-hoc from the frozen checkpoints.

Nothing here retrains or re-tunes: it loads each variant's saved checkpoint and
measures.  The per-step correct-mode probability is obtained by rolling the
model out to each step count and reading the mixture at that point, which needs
no change to the model.

Reported exactly as agreed:
  confirmed   evidence conservation; the mixture's evidence-partition bound
  preliminary multimodal prediction; post-resolution collapse
  open        sector recovery; meaningful switching

USEFUL REFINEMENT IS NOT TESTED HERE.  The `redelivery` probe re-delivers a
byte-identical payload -- duplicate_batch never touches root_feat -- so it can
only test duplicate invariance.  Learned refinement is Gate 3R.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ecnca.neural.dataset import (BatchSpec, duplicate_batch,  # noqa: E402
                                  make_ambiguous_batch)
from ecnca.neural.model import (coverage, mixture_nll, mode_separation)  # noqa: E402
from ecnca.neural.sector_metrics import score_sectors  # noqa: E402
from ecnca.neural.train import TrainConfig, build_model, pick_device  # noqa: E402

VARIANTS = ["full", "no_sectors", "no_provenance", "plain"]
PROBE_STEPS = [2, 3, 5, 7, 9, 11]


PAIRED_KEYS = ("root_feat", "root_Lam", "root_mass", "root_valid", "x_true",
               "h_oracle", "Lam_oracle", "cell_label")


def _assert_paired(a, b, tag):
    """Every evidence tensor must be bit-identical across paired conditions."""
    for k in PAIRED_KEYS:
        if not torch.equal(a[k], b[k]):
            raise AssertionError(f"paired probe '{tag}' altered {k}")


@torch.no_grad()
def measure(model, spec, cfg, device, seed, copies, cross, n_batch=4, bs=12):
    """Every condition is derived from ONE base batch by re-delivery.

    Regenerating with different `copies` consumes the RNG differently and yields
    entirely different examples, so the comparison would not be paired at all.
    """
    rows = []
    for bi in range(n_batch):
        rng = np.random.default_rng(70_000 + 137 * seed + bi)
        b = make_ambiguous_batch(spec, bs, n_hyp=cfg.n_hyp, roots_per_hyp=3,
                                 resolve_roots=4, resolve_step=cfg.resolve_step,
                                 copies=1, cross_sector_copies=0,
                                 rng=rng, device=device)
        if copies > 1 or cross > 0:
            base = b
            b = duplicate_batch(b, max(copies, cross + 1),
                                np.random.default_rng(4242 + bi), spread=bool(cross))
            _assert_paired(base, b, f"copies={copies},cross={cross}")
        ceiling = b["root_mass"].sum(-1).clamp(min=1e-6)
        for st in PROBE_STEPS:
            o = model(b, steps=st, instrument=True)
            dist = (o["mix_mu"] - b["x_true"][:, None, None, :]).pow(2).sum(-1)
            w_star = o["mix_w"].gather(-1, dist.argmin(-1, keepdim=True)).squeeze(-1)
            ari = nmi = float("nan")
            if o.get("instrument"):
                ari, nmi = score_sectors(o["instrument"][-1]["hard"], b["cell_label"])
            mu = o["mu"].mean(1)
            rows.append(dict(
                step=st, copies=copies, cross=cross,
                rmse=float(((mu - b["x_true"]) ** 2).sum(-1).sqrt().mean()),
                nll=float(mixture_nll(o["mix_mu"], o["mix_Lam"], o["mix_w"],
                                      b["x_true"]).mean()),
                coverage=float(coverage(o["mu"], o["Lam"], b["x_true"])),
                claim_ratio=float((o["M"].mean(1) / ceiling).mean()),
                ari=ari, nmi=nmi,
                n_live_modes=float((o["mix_w"] > 0.05).float().sum(-1).mean()),
                mode_separation=float(mode_separation(o["mix_mu"], o["mix_w"])),
                w_correct=float(w_star.mean()),
                k_eff=float(o["k_eff"].mean()) if o.get("k_eff") is not None else float("nan"),
            ))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser("gate4_table")
    ap.add_argument("--run", default="results/gate4_variants")
    ap.add_argument("--out", default="results/gate4_final_table.csv")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    device = pick_device(a.device)
    out_rows = []

    for variant in VARIANTS:
        d = os.path.join(a.run, f"{variant}_s{a.seed}")
        ck_path = os.path.join(d, "ckpt.pt")
        if not os.path.exists(ck_path):
            print(f"[table] {variant}: no checkpoint, skipping")
            continue
        ck = torch.load(ck_path, map_location=device, weights_only=False)
        cfg = TrainConfig(**ck["config"])
        spec = BatchSpec(n_cells=cfg.n_cells, topology=cfg.topology,
                         max_roots=16, max_occurrences=48)
        model = build_model(cfg, spec).to(device)
        model.load_state_dict(ck["model"])
        model.eval()
        n_par = sum(p.numel() for p in model.parameters())
        hist = json.load(open(os.path.join(d, "history.json"))) \
            if os.path.exists(os.path.join(d, "history.json")) else []
        train_secs = hist[-1]["secs"] if hist else float("nan")

        base = measure(model, spec, cfg, device, a.seed, 1, 0)
        tr = measure(model, spec, cfg, device, a.seed, 4, 0)
        xd = measure(model, spec, cfg, device, a.seed, 8, 6)
        res = cfg.resolve_step
        pre, post = base[base.step < res], base[base.step >= res]
        fin = base[base.step == max(PROBE_STEPS)]

        out_rows.append(dict(
            variant=variant, n_params=n_par, train_secs=round(train_secs, 1),
            rmse=fin.rmse.mean(), nll=fin.nll.mean(), coverage=fin.coverage.mean(),
            ari_pre=pre.ari.max(), nmi_pre=pre.nmi.max(),
            n_live_modes=pre.n_live_modes.mean(), mode_separation=pre.mode_separation.mean(),
            keff_pre=pre.k_eff.mean(), keff_post=post.k_eff.mean(),
            w_correct_pre=pre.w_correct.mean(), w_correct_post=post.w_correct.mean(),
            redelivery_rmse=tr[tr.step == max(PROBE_STEPS)].rmse.mean(),
            redelivery_claim=tr[tr.step == max(PROBE_STEPS)].claim_ratio.mean(),
            base_claim=fin.claim_ratio.mean(),
            cross_dup_claim=xd[xd.step == max(PROBE_STEPS)].claim_ratio.mean(),
        ))
        print(f"[table] {variant} done ({n_par} params, {train_secs:.0f}s train)")

    df = pd.DataFrame(out_rows)
    df.to_csv(a.out, index=False)
    pd.set_option("display.width", 250, "display.max_columns", 60)
    print()
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
