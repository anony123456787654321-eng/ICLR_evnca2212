"""Component ablation of the evidence-conserving ledger, from per-seed files.

`analysis/paired_from_per_seed.py` answers a single contrast: is `full` better
than `filter_only`. That is the sharpest contrast, but on its own it does not
establish that the mechanism has no redundant parts. A reviewer is entitled to
ask which of the moving parts is doing the work, and to suspect that one of
them carries the result while the rest are decoration. This script exists to
answer that question from data that already exists, with no new training.

The ledger is not a monolith; it is three decisions applied at delivery time,
and each committed variant removes exactly one of them:

  full            ledger `max`        the whole mechanism
  filter_only     ledger `first`      REFINEMENT removed: still keyed by root,
                                      still conserving, but the first version
                                      of a root wins and later, better versions
                                      are discarded
  plain           ledger `none`       LINEAGE removed: no root identity at all,
                                      so every arrival is credited
  canonical_dedup ledger `canonical`  VERSION ORDERING removed: byte identity
                                      collapses exact copies, but with no
                                      notion of version a re-chunked or reworded
                                      copy of the same source is fresh evidence

Reading the accuracy column alone is a trap, which is why credit is reported in
the same row. `filter_only` holds credited evidence at exactly 1.000 under every
duplication regime, so it is already conserving; the `full` minus `filter_only`
difference therefore isolates refinement with conservation held fixed, and is
the only one of the three that is a clean single-component ablation. `plain` and
`canonical_dedup` do not conserve, so their deltas bundle an accuracy change
with a credit failure that grows with multiplicity. Those two rows are evidence
that lineage and version ordering are load-bearing for CONSERVATION first and
accuracy second, and the credit column is what makes that visible rather than
letting a small accuracy gap understate a total accounting failure.

The fourth component of the mechanism, the bottom-k sketch, is deliberately
ABSENT from this table. `ecnca/real/musique_duplication.py` has no sketch ledger
mode -- its `LEDGER_MODES` are the four above plus the near-duplicate detectors
and the fusion rules -- so no sketch arm was ever evaluated on this harness and
none can be recovered from committed MuSiQue files. Sketch-versus-exact evidence
exists elsewhere in the repository (gate C on AVeriTeC, and the synthetic B4
compressed benchmark), but in different metric spaces that do not share this
table's rows, so it is cited separately rather than fabricated into a fifth row.

Intervals are the exact paired t interval over seeds. Because the per-seed files
survive, the paired difference standard deviation is computed rather than
bounded, exactly as in `paired_from_per_seed.py`.
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

OUT = Path("results/component_ablation.json")

# t_{0.975, n-1}
T_CRIT = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 10: 2.262, 15: 2.145,
          20: 2.093}

# variant -> (expected ledger mode, component of the mechanism it removes).
# The ledger mode is checked against the `ledger` field of every per-seed file
# so that a mislabelled or regenerated run is caught here rather than being
# silently reported as an ablation of something it is not.
COMPONENTS = {
    "full": ("max", "none (full mechanism)"),
    "filter_only": ("first", "learned refinement"),
    "plain": ("none", "lineage identity"),
    "canonical_dedup": ("canonical", "version ordering"),
}

BASELINE = "full"
METRICS = ("accuracy", "nll", "mrr")


def load(run_dir: str, variant: str, seeds):
    """condition -> seed -> row, or None when any seed file is absent."""
    out: dict = {}
    expected = COMPONENTS[variant][0] if variant in COMPONENTS else None
    for s in seeds:
        p = Path(run_dir) / f"{variant}_s{s}.json"
        if not p.is_file():
            return None
        doc = json.loads(p.read_text())
        found = doc.get("ledger")
        if expected is not None and found != expected:
            raise ValueError(
                f"{p}: variant {variant!r} should run ledger {expected!r} "
                f"but the file records {found!r}; the component mapping in "
                f"COMPONENTS no longer matches the committed data")
        for r in doc["rows"]:
            key = (r["intervention"], r["scope"], r["multiplicity"])
            out.setdefault(key, {})[s] = r
    return out


def means(rows, key, seeds, readout):
    """Per-variant means over seeds for one condition, or None if incomplete."""
    if key not in rows:
        return None
    try:
        per = [rows[key][s] for s in seeds]
    except KeyError:
        return None
    out = {f"{m}_mean": st.mean(r[readout][m] for r in per) for m in METRICS}
    out["credited_over_ceiling_mean"] = st.mean(
        r["credited_over_ceiling"] for r in per)
    out["credited_over_ceiling_values"] = sorted(
        {r["credited_over_ceiling"] for r in per})
    return out


def paired(a_rows, b_rows, key, seeds, metric, readout):
    """Exact paired difference a - b over seeds, with a paired t interval."""
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
    return {"metric": metric, "n_seeds": n, "t_crit": T_CRIT[n],
            "full_mean": st.mean(xa), "variant_mean": st.mean(xb),
            "diff": mean, "diff_sd": sd, "ci_low": lo, "ci_high": hi,
            "excludes_zero": bool(lo > 0.0 or hi < 0.0)}


def fmt(x, width=8, places=4):
    return "n/a".rjust(width) if x is None else f"{x:{width}.{places}f}"


def print_tables(cells, variants, ablated):
    """Compact per-metric tables: variant means, then deltas against full."""
    head_v = "".join(f"{v[:9]:>10s}" for v in variants)
    head_d = "".join(f"{('d_' + v)[:10]:>11s}" for v in ablated)
    for metric in METRICS:
        print()
        print(f"== {metric} (mean over seeds; d_* = variant minus {BASELINE}, "
              f"* = paired 95% interval excludes 0) ==")
        print(f"{'regime':<11s}{'scope':<10s}{'m':>3s} |{head_v} |{head_d}")
        for c in cells:
            vals = "".join(
                fmt((c["variants"].get(v) or {}).get(f"{metric}_mean"), 10)
                for v in variants)
            ds = ""
            for v in ablated:
                d = (c["deltas"].get(v) or {}).get(metric)
                if d is None:
                    ds += f"{'n/a':>11s}"
                else:
                    ds += f"{d['diff']:>10.4f}{'*' if d['excludes_zero'] else ' '}"
            print(f"{c['intervention']:<11s}{c['scope']:<10s}"
                  f"{c['multiplicity']:>3d} |{vals} |{ds}")

    print()
    print("== credited_over_ceiling (mean over seeds; 1.000 = conserving) ==")
    print(f"{'regime':<11s}{'scope':<10s}{'m':>3s} |{head_v}")
    for c in cells:
        vals = "".join(
            fmt((c["variants"].get(v) or {}).get("credited_over_ceiling_mean"),
                10, 3) for v in variants)
        print(f"{c['intervention']:<11s}{c['scope']:<10s}"
              f"{c['multiplicity']:>3d} |{vals}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="results/musique_dup/eval")
    ap.add_argument("--variants",
                    default="full,filter_only,plain,canonical_dedup")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--readout", default="content")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    os.chdir(Path(__file__).resolve().parent.parent)

    seeds = [int(s) for s in a.seeds.split(",") if s]
    variants = [v for v in a.variants.split(",") if v]
    if BASELINE not in variants:
        raise SystemExit(f"--variants must include the baseline {BASELINE!r}")
    ablated = [v for v in variants if v != BASELINE]

    loaded, missing = {}, []
    for v in variants:
        rows = load(a.run, v, seeds)
        if rows is None:
            missing.append(v)
            continue
        loaded[v] = rows
    if BASELINE not in loaded:
        raise SystemExit(f"baseline {BASELINE!r} has no per-seed files "
                         f"under {a.run}")

    keys = sorted(set(loaded[BASELINE]), key=lambda k: (k[0], k[1], k[2]))
    cells, n_deltas, n_excl = [], 0, 0
    for key in keys:
        cell = {"intervention": key[0], "scope": key[1],
                "multiplicity": key[2], "variants": {}, "deltas": {}}
        for v in variants:
            if v not in loaded:
                continue
            m = means(loaded[v], key, seeds, a.readout)
            if m is not None:
                cell["variants"][v] = m
        for v in ablated:
            if v not in loaded:
                continue
            d = {}
            for metric in METRICS:
                row = paired(loaded[BASELINE], loaded[v], key, seeds, metric,
                             a.readout)
                if row is None:
                    continue
                # Reported as variant minus full, so a negative accuracy delta
                # means the ablation hurt: removing that component costs.
                # `+ 0.0` normalises the negative zero that negating an exactly
                # zero difference would otherwise leave in the table and JSON.
                row = dict(row)
                row["diff"] = -row["diff"] + 0.0
                row["ci_low"], row["ci_high"] = (-row["ci_high"] + 0.0,
                                                 -row["ci_low"] + 0.0)
                d[metric] = row
                n_deltas += 1
                n_excl += bool(row["excludes_zero"])
            if d:
                cell["deltas"][v] = d
        cells.append(cell)

    out = {
        "run": a.run,
        "seeds": seeds,
        "readout": a.readout,
        "baseline": BASELINE,
        "interval": "exact paired t interval over seeds",
        "t_crit": T_CRIT[len(seeds)],
        "delta_sign_convention": f"variant minus {BASELINE}",
        "components": {v: {"ledger": COMPONENTS[v][0],
                           "component_removed": COMPONENTS[v][1]}
                       for v in variants if v in COMPONENTS},
        "untested_components": {
            "bottom-k sketch": (
                "no sketch ledger mode exists in "
                "ecnca/real/musique_duplication.py, so no sketch arm was "
                "evaluated on this harness and none can be built from "
                "committed MuSiQue per-seed files without new compute")},
        "missing_variants": missing,
        "n_cells": len(cells),
        "n_deltas": n_deltas,
        "n_excluding_zero": n_excl,
        "cells": cells,
    }
    dest = Path(a.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2) + "\n")

    present = [v for v in variants if v in loaded]
    print_tables(cells, present, [v for v in ablated if v in loaded])
    print()
    print(f"{len(cells)} conditions x {len(present)} variants "
          f"over {len(seeds)} seeds -> {dest}")
    print(f"paired deltas: {n_deltas}, intervals excluding zero: {n_excl}")
    print("bottom-k sketch: NOT ablated -- no sketch ledger mode on this "
          "harness (see docstring)")
    if missing:
        print("missing per-seed files for: " + ", ".join(missing))


if __name__ == "__main__":
    main()
