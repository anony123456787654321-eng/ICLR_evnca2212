"""Figures generated from the verified measurements used in the manuscript.

Both panels are drawn from committed reports rather than from any hand-entered
value, so a caption can describe exactly what the reader sees.

Figure 1 plots credited evidence against delivery multiplicity for every
backbone in the synthetic graph matrix. It is the clearest statement of the
central invariant, because a conserving arm should trace a flat line while a
provenance-free arm should trace the identity.

Figure 2 plots the two-hop MuSiQue comparison under overlapping re-chunking.
It shows where the ledger's accuracy advantage over surface deduplication
begins, which is the same multiplicity at which the credit curves separate.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

MULTS = (1, 2, 4, 8, 16)
LABEL = {"nca_ec": "Cellular automaton, conserving",
         "nca_plain": "Cellular automaton, provenance-free",
         "mpnn_ec": "Message passing, conserving",
         "mpnn_plain": "Message passing, provenance-free",
         "nonbacktracking_mpnn": "Non-backtracking message passing"}
ORDER = ("mpnn_ec", "nca_ec", "nonbacktracking_mpnn", "nca_plain", "mpnn_plain")


def backbone_panel(ax, report):
    agg = defaultdict(lambda: defaultdict(list))
    for r in json.loads(Path(report).read_text())["rows"]:
        for m, v in r["credit_by_multiplicity"].items():
            agg[r["variant"]][int(m)].append(v)
    for v in ORDER:
        if v not in agg:
            continue
        ys = [st.mean(agg[v][m]) for m in MULTS]
        conserving = v.endswith("_ec")
        ax.plot(MULTS, ys, marker="o", linewidth=2.0 if conserving else 1.6,
                linestyle="-" if conserving else "--", label=LABEL[v])
    ax.axhline(1.0, color="0.45", linewidth=0.9, zorder=0)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xticks(MULTS)
    ax.set_xticklabels([str(m) for m in MULTS])
    ax.set_xlabel("Delivery multiplicity")
    ax.set_ylabel("Credited evidence")
    ax.set_title("Conservation across backbones")
    ax.legend(fontsize=7, loc="upper left", frameon=True)


def musique_panel(ax, findings_path):
    d = json.loads(Path(findings_path).read_text())["findings"]
    def cell(pair, m, metric):
        for f in d:
            if (f["pair"], f["intervention"], f["scope"], f["multiplicity"],
                    f["metric"]) == (pair, "overlap", "all_hops", m, metric):
                return f
        return None
    ms, led, ded, lo, hi = [], [], [], [], []
    for m in (2, 4, 8, 16):
        f = cell("full:canonical_dedup", m, "accuracy")
        if f is None:
            continue
        ms.append(m); led.append(f["a_mean"]); ded.append(f["b_mean"])
        lo.append(f["ci_low"]); hi.append(f["ci_high"])
    ax.plot(ms, led, marker="o", linewidth=2.0, label="Lineage ledger")
    ax.plot(ms, ded, marker="s", linewidth=1.6, linestyle="--",
            label="Surface deduplication")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ms); ax.set_xticklabels([str(m) for m in ms])
    ax.set_xlabel("Duplication multiplicity")
    ax.set_ylabel("Top-1 accuracy")
    ax.set_title("Two-hop MuSiQue under re-chunking")
    ax.legend(fontsize=7, loc="upper right", frameon=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graph", default="results/architecture/graph/report.json")
    ap.add_argument("--fusion", default="results/paired_fusion_h2.json")
    ap.add_argument("--out", default="paper/figures")
    a = ap.parse_args()
    os.chdir(Path(__file__).resolve().parent.parent)
    sns.set_theme(style="whitegrid", context="paper")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    backbone_panel(ax, a.graph)
    fig.tight_layout()
    p1 = out / "fig_conservation_backbones.pdf"
    fig.savefig(p1, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {p1}")

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    musique_panel(ax, a.fusion)
    fig.tight_layout()
    p2 = out / "fig_musique_dose_response.pdf"
    fig.savefig(p2, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {p2}")


if __name__ == "__main__":
    main()
