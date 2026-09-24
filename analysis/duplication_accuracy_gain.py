"""Task-quality dominance of the ledger over deployed deduplication.

The duplicated-hop reports already contain a task-quality contrast that the
paper reports only as invariance.  Under OVERLAPPING re-chunking the surface
defences lose the source identity -- a shifted chunk boundary changes every
hash -- while the ledger keeps it.  That costs the surface defences accuracy,
not merely credit, and this script measures how much.

Interval construction
---------------------
The committed reports carry a per-condition mean and standard deviation over a
COMMON set of seeds, not the per-seed values, so the paired difference standard
deviation cannot be recovered exactly.  It is however bounded:

    sd(a - b) <= sd(a) + sd(b),

with equality only under perfect negative correlation across seeds.  Using
sd(a) + sd(b) therefore yields an interval at least as wide as the true paired
interval.  An exclusion of zero under this bound is valid and conservative; a
failure to exclude zero is NOT evidence of absence, because the true interval
may be much narrower.  Re-running the evaluation with per-seed output would
tighten every interval here and is the cheapest available improvement.

Lower is better for nll, ece, brier, prediction_drift and credited evidence;
higher is better for accuracy and mrr.  `improved` is signed accordingly, so a
True always means the ledger won.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

OUT = Path("results/duplication_accuracy_gain.json")

RIVALS = ["canonical_dedup", "minhash_dedup", "simhash_dedup", "embed_dedup",
          "plain"]
LOWER_IS_BETTER = {"content_nll", "content_ece", "content_brier",
                   "prediction_drift", "credited_over_ceiling"}
METRICS = ["content_accuracy", "content_nll", "content_mrr", "content_ece",
           "content_brier", "credited_over_ceiling", "prediction_drift"]

# t_{0.975, n-1}
T_CRIT = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 10: 2.262, 20: 2.093}


def t_crit(n_seeds: int) -> float:
    if n_seeds in T_CRIT:
        return T_CRIT[n_seeds]
    raise KeyError(f"no critical value tabulated for n_seeds={n_seeds}; "
                   "add it rather than interpolating")


def conservative_ci(a, sa, b, sb, n_seeds):
    """Worst-case-pairing interval on (a - b).  See module docstring."""
    diff = a - b
    se = (sa + sb) / math.sqrt(n_seeds)
    t = t_crit(n_seeds)
    return diff, diff - t * se, diff + t * se


def index(rows):
    return {(r["variant"], r["intervention"], r["scope"], r["multiplicity"]): r
            for r in rows}


def compare(idx, interv, scope, mult, metric, rival):
    ec = idx.get(("full", interv, scope, mult))
    rv = idx.get((rival, interv, scope, mult))
    if ec is None or rv is None:
        return None
    n = int(ec["n_seeds"])
    if int(rv["n_seeds"]) != n:
        raise ValueError(f"seed-count mismatch for {rival} at {interv}/{mult}: "
                         f"{ec['n_seeds']} vs {rv['n_seeds']}; the bound "
                         "assumes a common seed set")
    a, sa = ec[metric], ec.get(metric + "_sd", 0.0)
    b, sb = rv[metric], rv.get(metric + "_sd", 0.0)
    diff, lo, hi = conservative_ci(a, sa, b, sb, n)
    excludes = lo > 0.0 or hi < 0.0
    better = diff < 0 if metric in LOWER_IS_BETTER else diff > 0
    return {"hops": None, "intervention": interv, "scope": scope,
            "multiplicity": mult, "metric": metric, "rival": rival,
            "n_seeds": n,
            "ec": a, "ec_sd": sa, "rival_value": b, "rival_sd": sb,
            "diff": diff, "ci_low": lo, "ci_high": hi,
            "excludes_zero": bool(excludes),
            "improved": bool(excludes and better)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reports", default="results/musique_dup_iclr/h2,"
                                         "results/musique_dup_iclr/h3,"
                                         "results/musique_dup_iclr/h4")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--interventions", default="exact,overlap,paraphrase,"
                                               "cycle,reorder")
    ap.add_argument("--strict", action="store_true",
                    help="fail if any named report is missing")
    a = ap.parse_args()
    os.chdir(Path(__file__).resolve().parent.parent)

    interventions = [s for s in a.interventions.split(",") if s]
    findings, missing = [], []

    for rep_dir in a.reports.split(","):
        path = Path(rep_dir) / "report.json"
        if not path.is_file():
            missing.append(str(path))
            continue
        doc = json.loads(path.read_text())
        hops = doc["hops"][0] if doc.get("hops") else None
        idx = index(doc["rows"])
        mults = sorted({r["multiplicity"] for r in doc["rows"]})
        scopes = sorted({r["scope"] for r in doc["rows"]})
        for scope in scopes:
            for interv in interventions:
                for mult in mults:
                    for metric in METRICS:
                        for rival in RIVALS:
                            row = compare(idx, interv, scope, mult, metric,
                                          rival)
                            if row is None:
                                continue
                            row["hops"] = hops
                            findings.append(row)

    if missing and a.strict:
        raise SystemExit("missing reports: " + ", ".join(missing))

    wins = [f for f in findings if f["improved"]]
    losses = [f for f in findings
              if f["excludes_zero"] and not f["improved"]]
    out = {
        "description": "conservative worst-case-pairing intervals on the "
                       "ledger minus each rival; see module docstring",
        "interval": "sd(a-b) bounded by sd(a)+sd(b); conservative",
        "reports": a.reports.split(","),
        "missing_reports": missing,
        "n_comparisons": len(findings),
        "n_ledger_wins": len(wins),
        "n_ledger_losses": len(losses),
        "findings": findings,
    }
    dest = Path(a.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(f"{len(findings)} comparisons -> {dest}")
    print(f"ledger wins (interval excludes zero): {len(wins)}")
    print(f"ledger losses (interval excludes zero): {len(losses)}")


if __name__ == "__main__":
    main()
