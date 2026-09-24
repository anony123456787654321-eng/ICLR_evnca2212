"""Exact paired intervals from the per-seed duplication evaluations.

The aggregated reports carry only a mean and a standard deviation per
condition, so `analysis/duplication_accuracy_gain.py` has to bound the paired
difference standard deviation rather than compute it. Where the per-seed files
survive, the paired difference is available exactly and no bound is needed.

The contrast this script exists for is `full` against `filter_only`. Both hold
credited evidence at exactly 1.000 under every duplication regime, so they are
equally conserving; they differ only in that `filter_only` keeps the first
version of each root and discards every later one. The difference therefore
measures REFINEMENT with conservation held fixed, which is the combination the
paper claims rather than either half alone. A reading of `filter_only` as a
weak baseline misses that it already satisfies the conservation property.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUT = Path("results/paired_from_per_seed.json")

# t_{0.975, n-1}
T_CRIT = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 10: 2.262, 15: 2.145,
          20: 2.093}


def load(run_dir: str, variant: str, seeds):
    """condition -> seed -> row, or None when a seed file is absent."""
    out: dict = {}
    for s in seeds:
        p = Path(run_dir) / f"{variant}_s{s}.json"
        if not p.is_file():
            return None
        for r in json.loads(p.read_text())["rows"]:
            key = (r["intervention"], r["scope"], r["multiplicity"])
            out.setdefault(key, {})[s] = r
    return out


def paired(a_rows, b_rows, key, seeds, metric, readout):
    if key not in a_rows or key not in b_rows:
        return None
    try:
        xa = [a_rows[key][s][readout][metric] for s in seeds]
        xb = [b_rows[key][s][readout][metric] for s in seeds]
    except KeyError:
        return None
    diff = [x - y for x, y in zip(xa, xb)]
    mean = st.mean(diff)
    sd = st.stdev(diff) if len(set(diff)) > 1 else 0.0
    n = len(seeds)
    if n not in T_CRIT:
        raise KeyError(f"no critical value tabulated for n={n}")
    se = sd / math.sqrt(n)
    lo, hi = mean - T_CRIT[n] * se, mean + T_CRIT[n] * se
    return {"intervention": key[0], "scope": key[1], "multiplicity": key[2],
            "metric": metric, "readout": readout, "n_seeds": n,
            "a_mean": st.mean(xa), "b_mean": st.mean(xb),
            "diff": mean, "diff_sd": sd, "ci_low": lo, "ci_high": hi,
            "excludes_zero": bool(lo > 0.0 or hi < 0.0),
            "a_credit": sorted({a_rows[key][s]["credited_over_ceiling"]
                                for s in seeds}),
            "b_credit": sorted({b_rows[key][s]["credited_over_ceiling"]
                                for s in seeds})}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="results/musique_dup/eval")
    ap.add_argument("--pairs", default="full:filter_only,full:plain,"
                                       "full:canonical_dedup")
    ap.add_argument("--metrics", default="accuracy,nll,mrr")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--readout", default="content")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    os.chdir(Path(__file__).resolve().parent.parent)

    seeds = [int(s) for s in a.seeds.split(",") if s]
    metrics = [s for s in a.metrics.split(",") if s]
    cache, findings, missing = {}, [], []

    for pair in a.pairs.split(","):
        left, right = pair.split(":")
        for v in (left, right):
            if v not in cache:
                cache[v] = load(a.run, v, seeds)
        if cache[left] is None or cache[right] is None:
            missing.append(pair)
            continue
        keys = sorted(set(cache[left]) & set(cache[right]), key=str)
        for key in keys:
            for metric in metrics:
                row = paired(cache[left], cache[right], key, seeds, metric,
                             a.readout)
                if row is None:
                    continue
                row["pair"] = pair
                findings.append(row)

    out = {"run": a.run, "seeds": seeds, "readout": a.readout,
           "interval": "exact paired t interval over seeds",
           "missing_pairs": missing,
           "n_comparisons": len(findings),
           "n_excluding_zero": sum(f["excludes_zero"] for f in findings),
           "findings": findings}
    dest = Path(a.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(f"{len(findings)} paired comparisons -> {dest}")
    print(f"intervals excluding zero: {out['n_excluding_zero']}")
    if missing:
        print("missing per-seed files for: " + ", ".join(missing))


if __name__ == "__main__":
    main()
