"""Evaluate the directional-credit experiments against their pre-registration.

docs/DIRECTIONAL_PREREGISTRATION.md, with its two dated amendments, fixes the
predictions and the decision rules. This script applies them mechanically to
the committed outputs of experiments/directional_synthetic.py (A2) and
experiments/directional_musique.py (A3), and writes the verdict, the table
bodies, the figure and the macros the manuscript quotes. Nothing is typed by
hand.
"""
from __future__ import annotations

import json
import math
import os
import random
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
GEN = ROOT / "paper" / "generated"
FIG = ROOT / "paper" / "figures"
N_BOOT = 10000
T975 = {4: 2.776}  # two-sided 95% t quantile, five seeds

A2_ARMS = ("ec_matrix", "ec_scalar_fit", "ec_scalar", "plain_matrix",
           "plain_scalar")
A2_LABEL = {"ec_matrix": r"\textbf{Lineage, directional (this work)}",
            "ec_scalar_fit": "Lineage, scalar with learned scale",
            "ec_scalar": "Lineage, scalar",
            "plain_matrix": "No lineage, directional",
            "plain_scalar": "No lineage, scalar"}
A3_SCORES = ("D", "S", "best_min", "mean_best", "sum_cos", "max_cos", "hops")
# Rows are named by the two choices of Section 3.3: how sources combine within
# a requirement, and how requirements combine. Weakest-requirement rows are the
# directional scores and mean-requirement rows the scalar ones.
A3_LABEL = {"D": r"Weakest requirement, sum over sources (registered primary)",
            "S": "Mean requirement, sum over sources",
            "best_min": "Weakest requirement, best single source",
            "mean_best": "Mean requirement, best single source",
            "sum_cos": "Sum of retrieval scores",
            "max_cos": "Maximum retrieval score",
            "hops": "Hop count alone"}


# ---------------------------------------------------------------- A2 -------
def load_a2():
    rows = []
    base = ROOT / "results" / "directional_synthetic"
    # One file per arm and observation rank, written by parallel jobs.
    for p in sorted(base.glob("*/obs*.json")):
        rows += json.loads(p.read_text())["rows"]
    table = defaultdict(dict)
    for r in rows:
        table[(r["arm"], r["obs_dim"])][r["seed"]] = r["task"]
    return table


def paired_t(a, b):
    """Mean of a - b over shared seeds with a two-sided 95% t interval."""
    seeds = sorted(set(a) & set(b))
    d = [a[s] - b[s] for s in seeds]
    m = st.mean(d)
    half = T975[len(d) - 1] * st.stdev(d) / math.sqrt(len(d))
    return {"diff": m, "lo": m - half, "hi": m + half, "n": len(d)}


def nll(table, arm, obs, cond):
    return {s: v[cond]["nll"] for s, v in table[(arm, obs)].items()}


def a2(table):
    out = {"S1": paired_t(nll(table, "ec_scalar_fit", 1, "clean"),
                          nll(table, "ec_matrix", 1, "clean")),
           "S1_fixed_scale": paired_t(nll(table, "ec_scalar", 1, "clean"),
                                      nll(table, "ec_matrix", 1, "clean")),
           "S2": paired_t(nll(table, "plain_matrix", 1, "duplicate"),
                          nll(table, "ec_matrix", 1, "duplicate"))}
    gaps = {o: paired_t(nll(table, "ec_scalar_fit", o, "clean"),
                        nll(table, "ec_matrix", o, "clean"))
            for o in (1, 2, 4)}
    out["gap_by_rank"] = gaps
    out["S1"]["holds"] = out["S1"]["lo"] > 0
    out["S2"]["holds"] = out["S2"]["lo"] > 0
    out["S3"] = {"gap_rank1": gaps[1]["diff"], "gap_full": gaps[4]["diff"],
                 "holds": gaps[4]["diff"] < gaps[1]["diff"]}
    return out


# ---------------------------------------------------------------- A3 -------
def auroc(score, label):
    """Mann-Whitney AUROC with average ranks for ties."""
    score = np.asarray(score, float); label = np.asarray(label, bool)
    n1, n0 = label.sum(), (~label).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score)); s = score[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def stratified(score, label, strata):
    """Size-weighted mean of within-stratum AUROCs, skipping one-class strata."""
    total, acc = 0, 0.0
    for g in np.unique(strata):
        m = strata == g
        a = auroc(score[m], label[m])
        if not math.isnan(a):
            acc += a * m.sum(); total += m.sum()
    return acc / total


def a3_arrays(doc, cond):
    rows = doc["rows"]
    lab = np.array([r["sufficient"] for r in rows])
    hops = np.array([r["n_hops"] for r in rows])
    sc = {k: np.array([r[cond][k] for r in rows]) for k in A3_SCORES
          if k != "hops"}
    sc["hops"] = -hops.astype(float)
    return sc, lab, hops


def boot_diff(a, b, lab, hops, strat, rng):
    idx_by = {g: np.flatnonzero(hops == g) for g in np.unique(hops)}
    f = (lambda s, l, h: stratified(s, l, h)) if strat else \
        (lambda s, l, h: auroc(s, l))
    point = f(a, lab, hops) - f(b, lab, hops)
    diffs = []
    for _ in range(N_BOOT):
        if strat:
            idx = np.concatenate([rng.choice(ix, len(ix)) for ix in
                                  idx_by.values()])
        else:
            idx = rng.integers(0, len(lab), len(lab))
        diffs.append(f(a[idx], lab[idx], hops[idx])
                     - f(b[idx], lab[idx], hops[idx]))
    diffs = np.sort(diffs)
    return {"diff": point, "lo": float(diffs[int(0.025 * N_BOOT)]),
            "hi": float(diffs[int(0.975 * N_BOOT) - 1])}


def a3(doc):
    rng = np.random.default_rng(0)
    clean, lab, hops = a3_arrays(doc, "clean")
    dup_lin, _, _ = a3_arrays(doc, "dup_lineage")
    dup_none, _, _ = a3_arrays(doc, "dup_none")
    out = {"n": int(len(lab)), "sufficient_rate": float(lab.mean()),
           "hop_counts": {int(g): int((hops == g).sum())
                          for g in np.unique(hops)},
           # A stratum with one class has no AUROC and is left out of the
           # stratified mean; recorded so the weighting can be checked.
           "sufficient_by_hops": {int(g): int(lab[hops == g].sum())
                                  for g in np.unique(hops)}}
    out["auroc"] = {}
    for name, sc in (("clean", clean), ("dup_lineage", dup_lin),
                     ("dup_none", dup_none)):
        out["auroc"][name] = {k: {"pooled": auroc(v, lab),
                                  "stratified": stratified(v, lab, hops)}
                              for k, v in sc.items()}
    r1 = boot_diff(clean["D"], clean["S"], lab, hops, True, rng)
    r1["holds"] = r1["lo"] > 0
    out["R1"] = r1
    out["R1_pooled"] = boot_diff(clean["D"], clean["S"], lab, hops, False, rng)
    r2 = boot_diff(dup_lin["D"], dup_none["S"], lab, hops, False, rng)
    r2["holds"] = r2["lo"] > 0
    out["R2"] = r2
    out["R2_stratified"] = boot_diff(dup_lin["D"], dup_none["S"], lab, hops,
                                     True, rng)
    # R2 changes lineage and aggregation together. These two contrasts
    # separate them: lineage alone, holding the aggregation fixed.
    out["lineage_effect_D"] = boot_diff(dup_lin["D"], dup_none["D"], lab, hops,
                                        False, rng)
    out["lineage_effect_S"] = boot_diff(dup_lin["S"], dup_none["S"], lab, hops,
                                        False, rng)
    out["D_vs_best_min"] = boot_diff(clean["D"], clean["best_min"], lab, hops,
                                     True, rng)
    out["D_vs_sum_cos"] = boot_diff(clean["D"], clean["sum_cos"], lab, hops,
                                    True, rng)
    # Exploratory, not registered, and drawn after every registered contrast
    # so that none of their intervals moves. With each requirement's best
    # single source in place of summed support, does the weakest requirement
    # still beat the mean over requirements?
    out["best_min_vs_mean_best"] = boot_diff(clean["best_min"],
                                             clean["mean_best"], lab, hops,
                                             True, rng)
    # The best-source scores use the cosine, not its square. The two order
    # records identically when every best cosine is positive, which makes the
    # best-source score rank-one directional credit of one source.
    out["best_source_floor"] = float(clean["best_min"].min())
    return out


# ------------------------------------------------------------ outputs -----
def f4(x):
    return f"${x:.4f}$"


def iv(x):
    return f"[{x['lo']:+.4f}, {x['hi']:+.4f}]"


def write_body(path, lines):
    body = "\n".join(lines).rstrip()
    if body.endswith("\\\\"):
        body = body[:-2].rstrip()
    path.write_text(body + "\n")


def main():
    os.chdir(ROOT)
    verdict = {"preregistration": "docs/DIRECTIONAL_PREREGISTRATION.md"}
    mac = {}
    table = load_a2()
    if table:
        v = a2(table)
        verdict["A2"] = v
        lines = ["% generated by analysis/directional_analysis.py"]
        for arm in A2_ARMS:
            cells = []
            for o in (1, 2, 4):
                xs = [t["clean"]["nll"] for t in table[(arm, o)].values()]
                cells.append(f4(st.mean(xs)) if xs else "n/a")
            xs = [t["duplicate"]["nll"] for t in table[(arm, 1)].values()]
            cells.append(f4(st.mean(xs)) if xs else "n/a")
            xs = [t["clean"]["coverage"] for t in table[(arm, 1)].values()]
            cells.append(f4(st.mean(xs)) if xs else "n/a")
            lines.append(" & ".join([A2_LABEL[arm]] + cells) + r" \\")
        write_body(GEN / "tab_directional_synthetic.tex", lines)
        mac.update({
            "DirSynSOne": f"{v['S1']['diff']:+.4f}", "DirSynSOneCI": iv(v["S1"]),
            "DirSynSOneFixed": f"{v['S1_fixed_scale']['diff']:+.4f}",
            "DirSynSOneFixedCI": iv(v["S1_fixed_scale"]),
            "DirSynSTwo": f"{v['S2']['diff']:+.4f}", "DirSynSTwoCI": iv(v["S2"]),
            "DirSynGapOne": f"{v['gap_by_rank'][1]['diff']:+.4f}",
            "DirSynGapTwo": f"{v['gap_by_rank'][2]['diff']:+.4f}",
            "DirSynGapFour": f"{v['gap_by_rank'][4]['diff']:+.4f}",
            "DirSynGapFourCI": iv(v["gap_by_rank"][4])})
    doc_path = ROOT / "results" / "directional_musique" / "per_record.json"
    if doc_path.is_file():
        doc = json.loads(doc_path.read_text())
        v = a3(doc)
        verdict["A3"] = v
        lines = ["% generated by analysis/directional_analysis.py"]
        for k in A3_SCORES:
            c = v["auroc"]["clean"][k]
            dn = v["auroc"]["dup_none"][k]
            # Lineage-aware scores with copies equal the clean ones by
            # construction, so only the per-arrival column is new.
            lines.append(" & ".join([A3_LABEL[k], f4(c["pooled"]),
                                     f4(c["stratified"]), f4(dn["pooled"])])
                         + r" \\")
        write_body(GEN / "tab_directional_musique.tex", lines)
        a = v["auroc"]
        mac.update({
            "DirReN": f"{v['n']:,}".replace(",", "{,}"),
            "DirReSufficient": f"{v['sufficient_rate']:.3f}",
            "DirReROne": f"{v['R1']['diff']:+.4f}", "DirReROneCI": iv(v["R1"]),
            "DirReROnePooled": f"{v['R1_pooled']['diff']:+.4f}",
            "DirReROnePooledCI": iv(v["R1_pooled"]),
            "DirReRTwo": f"{v['R2']['diff']:+.4f}", "DirReRTwoCI": iv(v["R2"]),
            "DirReDStrat": f"{a['clean']['D']['stratified']:.4f}",
            "DirReSStrat": f"{a['clean']['S']['stratified']:.4f}",
            "DirReHopsPooled": f"{a['clean']['hops']['pooled']:.4f}",
            "DirReDvsBestMin": f"{v['D_vs_best_min']['diff']:+.4f}",
            "DirReDvsBestMinCI": iv(v["D_vs_best_min"]),
            "DirReDvsSumCos": f"{v['D_vs_sum_cos']['diff']:+.4f}",
            "DirReDvsSumCosCI": iv(v["D_vs_sum_cos"]),
            "DirReDupNoneS": f"{a['dup_none']['S']['pooled']:.4f}",
            "DirReDupLinD": f"{a['dup_lineage']['D']['pooled']:.4f}",
            "DirReDupNoneD": f"{a['dup_none']['D']['pooled']:.4f}",
            "DirReLinEffD": f"{v['lineage_effect_D']['diff']:+.4f}",
            "DirReLinEffDCI": iv(v["lineage_effect_D"]),
            "DirReLinEffS": f"{v['lineage_effect_S']['diff']:+.4f}",
            "DirReLinEffSCI": iv(v["lineage_effect_S"]),
            "DirReDPooled": f"{a['clean']['D']['pooled']:.4f}",
            "DirReSPooled": f"{a['clean']['S']['pooled']:.4f}",
            "DirReSumCosPooled": f"{a['clean']['sum_cos']['pooled']:.4f}",
            "DirReBestMinStrat": f"{a['clean']['best_min']['stratified']:.4f}",
            "DirReBestMinPooled": f"{a['clean']['best_min']['pooled']:.4f}",
            "DirReMeanBestStrat": f"{a['clean']['mean_best']['stratified']:.4f}",
            "DirReSumCosStrat": f"{a['clean']['sum_cos']['stratified']:.4f}",
            "DirReMaxCosStrat": f"{a['clean']['max_cos']['stratified']:.4f}",
            "DirReBestVsMean": f"{v['best_min_vs_mean_best']['diff']:+.4f}",
            "DirReBestVsMeanCI": iv(v["best_min_vs_mean_best"]),
            "DirReBestFloor": f"{v['best_source_floor']:.4f}",
            "DirReHopsStrat": f"{a['clean']['hops']['stratified']:.4f}"})
    out = ROOT / "results" / "directional_verdict.json"
    out.write_text(json.dumps(verdict, indent=2, default=float) + "\n")
    lines = ["% generated by analysis/directional_analysis.py"]
    lines += [f"\\newcommand{{\\{k}}}{{{x}}}" for k, x in sorted(mac.items())]
    (GEN / "directional_macros.tex").write_text("\n".join(lines) + "\n")
    print(json.dumps({k: (x if k != "A3" else {kk: x[kk] for kk in
                                                ("n", "sufficient_rate",
                                                 "hop_counts", "R1",
                                                 "R1_pooled", "R2",
                                                 "D_vs_best_min",
                                                 "D_vs_sum_cos")})
                      for k, x in verdict.items()}, indent=1, default=float))


if __name__ == "__main__":
    main()
