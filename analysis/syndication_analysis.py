"""Evaluate the syndication experiment against its pre-registration.

docs/SYNDICATION_PREREGISTRATION.md fixes the predictions, the endpoints and
the decision rules before any reader output existed. This script applies those
rules mechanically, so the verdict cannot be chosen after the numbers are seen.

Records are the resampling unit. The reader decodes greedily and every context
is a deterministic function of the record, so there is no seed to resample.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ARMS = ("full", "canonical_dedup", "minhash_dedup", "simhash_dedup", "plain")
METRICS = ("gold_recall", "misled", "true_final_in_context",
           "n_context_tokens")


def load(run_dir):
    data = {}
    for arm in ARMS:
        p = Path(run_dir) / f"{arm}_s0.json"
        if not p.is_file():
            continue
        for r in json.loads(p.read_text())["rows"]:
            key = (r["intervention"], r["multiplicity"])
            data.setdefault(arm, {})[key] = {x["record_id"]: x
                                             for x in r["per_record"]}
    return data


def paired(a, b, metric, n_boot, rng):
    ids = [i for i in a if i in b]
    if len(ids) != len(a) or len(ids) != len(b):
        raise SystemExit("record sets differ; cannot pair")
    d = [a[i][metric] - b[i][metric] for i in ids]
    n = len(d)
    point = sum(d) / n
    boots = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n
                   for _ in range(n_boot))
    lo, hi = boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]
    return {"diff": point, "ci_low": lo, "ci_high": hi, "n": n,
            "excludes_zero": bool(lo > 0 or hi < 0)}


def mean(cells, metric):
    vals = [c[metric] for c in cells.values()]
    return sum(vals) / len(vals)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--last", default="results/musique_syndication/last")
    ap.add_argument("--first", default="results/musique_syndication/first")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--rng-seed", type=int, default=0)
    ap.add_argument("--out", default="results/musique_syndication/verdict.json")
    a = ap.parse_args()
    os.chdir(Path(__file__).resolve().parent.parent)
    rng = random.Random(a.rng_seed)

    runs = {"last": load(a.last), "first": load(a.first)}
    out = {"preregistration": "docs/SYNDICATION_PREREGISTRATION.md",
           "dose_response": {}, "contrasts": {}, "predictions": {}}

    for pos, data in runs.items():
        if not data:
            print(f"[verdict] no output found for position={pos}")
            continue
        print(f"\n=== position = {pos} ===")
        print(f"{'arm':>16} {'style':>11} {'m':>3} {'gold':>7} {'misled':>7} "
              f"{'true kept':>9} {'tokens':>7}")
        for arm in ARMS:
            for (style, m), cells in sorted(data.get(arm, {}).items()):
                row = {k: mean(cells, k) for k in METRICS}
                out["dose_response"][f"{pos}/{arm}/{style}/{m}"] = row
                print(f"{arm:>16} {style:>11} {m:>3} "
                      f"{row['gold_recall']:>7.4f} {row['misled']:>7.4f} "
                      f"{row['true_final_in_context']:>9.2f} "
                      f"{row['n_context_tokens']:>7.0f}")
        for style in ("exact", "boilerplate"):
            for m in (1, 16):
                for rival in ARMS[1:]:
                    if (style, m) not in data.get(rival, {}):
                        continue
                    for metric in ("gold_recall", "misled"):
                        c = paired(data["full"][(style, m)],
                                   data[rival][(style, m)], metric,
                                   a.n_boot, rng)
                        out["contrasts"][
                            f"{pos}/{style}/{m}/full-{rival}/{metric}"] = c

    last, first = runs["last"], runs["first"]

    def within(arm, style, metric):
        return paired(last[arm][(style, 16)], last[arm][(style, 1)], metric,
                      a.n_boot, rng)

    preds = {}
    if last:
        p1 = within("plain", "exact", "gold_recall")
        preds["P1"] = {"claim": "plain gold falls from m=1 to m=16, exact, last",
                       **p1, "holds": bool(p1["excludes_zero"]
                                           and p1["diff"] < 0)}
        p2 = paired(last["full"][("exact", 16)], last["plain"][("exact", 16)],
                    "gold_recall", a.n_boot, rng)
        preds["P2"] = {"claim": "full minus plain gold at m=16, exact, last",
                       **p2, "holds": bool(p2["excludes_zero"]
                                           and p2["diff"] > 0)}
        p3 = within("plain", "exact", "misled")
        preds["P3"] = {"claim": "plain misled rises from m=1 to m=16",
                       **p3, "holds": bool(p3["diff"] > 0)}
        p4 = paired(last["canonical_dedup"][("boilerplate", 16)],
                    last["plain"][("boilerplate", 16)], "gold_recall",
                    a.n_boot, rng)
        preds["P4"] = {"claim": "canonical close to plain, boilerplate m=16",
                       **p4, "holds": not p4["excludes_zero"]}
    if first:
        p5 = paired(first["full"][("exact", 1)],
                    first["minhash_dedup"][("exact", 1)], "gold_recall",
                    a.n_boot, rng)
        preds["P5"] = {"claim": "full minus minhash gold at m=1, exact, first",
                       **p5, "holds": bool(p5["excludes_zero"]
                                           and p5["diff"] > 0)}
    out["predictions"] = preds

    print("\n=== pre-registered predictions ===")
    for k, v in preds.items():
        print(f"  {k} {'HOLDS' if v['holds'] else 'FAILS':>5}  {v['claim']}: "
              f"{v['diff']:+.4f} [{v['ci_low']:+.4f}, {v['ci_high']:+.4f}]")
    if "P1" in preds:
        if not preds["P1"]["holds"]:
            print("\nDecision: P1 fails. The reader was not moved by syndicated "
                  "repetition; no reader-protection claim is made and P2 is "
                  "not interpreted.")
        elif preds["P2"]["holds"]:
            print("\nDecision: P1 and P2 hold. The protection result is "
                  "reported with its interval.")
        else:
            print("\nDecision: P1 holds and P2 fails. Report both.")

    dest = Path(a.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\n-> {dest}")


if __name__ == "__main__":
    main()
