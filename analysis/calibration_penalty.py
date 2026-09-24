"""The calibration penalty that over-counted precision imposes.

Conservation is usually argued as an invariance. That argument invites the
question of what invariance buys against the truth, since a rule can be
perfectly invariant and perfectly wrong. This script states and checks the
quantitative answer.

Model a single underlying observation of precision Lambda delivered N times.
A provenance-free readout credits each arrival, so it reports precision
N * Lambda while the correct posterior carries Lambda. For two Gaussians with a
common mean and a precision ratio of N, the Kullback-Leibler divergence from the
correct posterior to the reported one is

    KL(N) = 0.5 * (N - 1 - ln N),

which is zero at N = 1 and grows without bound. Because the expected excess
negative log-likelihood of a wrong predictive distribution equals exactly that
divergence, the formula predicts a quantity this repository already measured
before the prediction existed.

The prediction carries no free parameter. It applies wherever the
provenance-free readout inflates precision linearly in the arrival count, which
the credit column verifies directly rather than assuming.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MULTS = (1, 2, 4, 8, 16)


def kl_penalty(n: float) -> float:
    """Excess negative log-likelihood, in nats, from crediting n copies once."""
    if n <= 0:
        raise ValueError("arrival count must be positive")
    return 0.5 * (n - 1.0 - math.log(n))


def credit_curves(report: str):
    agg = defaultdict(lambda: defaultdict(list))
    for r in json.loads(Path(report).read_text())["rows"]:
        for m, v in r["credit_by_multiplicity"].items():
            agg[r["variant"]][int(m)].append(v)
    return {v: {m: st.mean(xs) for m, xs in by.items()} for v, by in agg.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graph", default="results/architecture/graph/report.json")
    ap.add_argument("--measured-gap", type=float, default=6.144,
                    help="ArchDupHarmGap, the measured excess NLL in nats")
    ap.add_argument("--measured-lo", type=float, default=3.501)
    ap.add_argument("--measured-hi", type=float, default=8.787)
    ap.add_argument("--tol", type=float, default=1e-9,
                    help="tolerance for calling the credit curve exactly linear")
    ap.add_argument("--out", default="results/calibration_penalty.json")
    ap.add_argument("--figure", default="paper/figures/fig_calibration_penalty.pdf")
    a = ap.parse_args()
    os.chdir(Path(__file__).resolve().parent.parent)

    curves = credit_curves(a.graph)
    linear = {}
    for v, by in curves.items():
        linear[v] = all(abs(by.get(m, float("nan")) - m) <= a.tol for m in MULTS)

    pred = {m: kl_penalty(m) for m in MULTS}
    at16 = pred[16]
    inside = a.measured_lo <= at16 <= a.measured_hi
    rel = abs(at16 - a.measured_gap) / a.measured_gap

    out = {
        "formula": "KL(N) = 0.5 * (N - 1 - ln N), nats",
        "free_parameters": 0,
        "predicted_by_multiplicity": pred,
        "credit_curves": {v: {str(m): by.get(m) for m in MULTS}
                          for v, by in curves.items()},
        "premise_exactly_linear": linear,
        "measured_excess_nll_at_16": a.measured_gap,
        "measured_interval": [a.measured_lo, a.measured_hi],
        "predicted_at_16": at16,
        "relative_error": rel,
        "prediction_inside_measured_interval": bool(inside),
    }
    dest = Path(a.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2) + "\n")

    print("premise check, provenance-free credit must be linear in arrivals")
    for v in sorted(curves):
        if v.endswith("_plain") or "nonbacktracking" in v:
            ys = " ".join(f"{curves[v].get(m, float('nan')):.4f}" for m in MULTS)
            print(f"  {v:>22} {ys}   exactly linear: {linear[v]}")
    print("\npredicted excess negative log-likelihood, nats")
    for m in MULTS:
        print(f"  N={m:<3} {pred[m]:.4f}")
    print(f"\npredicted at N=16          {at16:.4f}")
    print(f"measured ArchDupHarmGap    {a.measured_gap:.4f}  "
          f"[{a.measured_lo:.3f}, {a.measured_hi:.3f}]")
    print(f"relative error             {rel*100:.2f}%")
    print(f"inside measured interval   {inside}")
    if a.figure:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_theme(style="whitegrid", context="paper")
        fig, ax = plt.subplots(figsize=(5.2, 3.2))
        grid = [1 + i * 0.05 for i in range(0, 301)]
        ax.plot(grid, [kl_penalty(x) for x in grid], linewidth=2.0,
                label=r"predicted $\frac{1}{2}(N-1-\ln N)$")
        ax.plot(MULTS, [pred[m] for m in MULTS], "o", markersize=5,
                label="predicted at tested multiplicities")
        ax.errorbar([16], [a.measured_gap],
                    yerr=[[a.measured_gap - a.measured_lo],
                          [a.measured_hi - a.measured_gap]],
                    fmt="s", capsize=4, markersize=7, color="crimson",
                    label="measured excess NLL, five seeds")
        ax.axhline(0.0, color="0.45", linewidth=0.9, zorder=0)
        ax.set_xscale("log", base=2)
        ax.set_xticks(MULTS); ax.set_xticklabels([str(m) for m in MULTS])
        ax.set_xlabel("Deliveries of one underlying observation")
        ax.set_ylabel("Excess negative log-likelihood (nats)")
        ax.set_title("Calibration penalty of over-counted precision")
        ax.legend(fontsize=7, loc="upper left", frameon=True)
        fig.tight_layout()
        out_fig = Path(a.figure); out_fig.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_fig, bbox_inches="tight"); plt.close(fig)
        print(f"-> {out_fig}")
    print(f"\n-> {dest}")


if __name__ == "__main__":
    main()
