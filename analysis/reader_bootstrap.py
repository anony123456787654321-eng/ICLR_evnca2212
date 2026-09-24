"""Record-level bootstrap intervals for the reader experiment.

The exact and overlap interventions carry no seeded randomness (see
ecnca.real.musique_duplication.build_stream and analysis/reader_summary.py,
which verifies this against the sweep rather than assuming it), so repeating a
run under different seed values reproduces it exactly. The unit of genuine
variation in this experiment is which of the 120 records is drawn, not which
seed is passed, so that is the axis resampled here.

Each condition needs one deterministic run with --per-record. It does not need
repetition across seeds, since repeating buys nothing where nothing is random.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

METRICS = ("token_f1", "answer_recall", "exact_match", "n_context_tokens")


def load_rows(run_dir: str, variant: str, seed: int):
    p = Path(run_dir) / f"{variant}_s{seed}.json"
    doc = json.loads(p.read_text())
    out = {}
    for r in doc["rows"]:
        if "per_record" not in r:
            raise SystemExit(
                f"{p} has no per-record scores; rerun with --per-record")
        out[(r["intervention"], r["multiplicity"])] = r["per_record"]
    return out


def bootstrap_diff(a_scores, b_scores, metric, n_boot, rng):
    """Percentile bootstrap for the paired difference, records as the unit.

    a_scores and b_scores must be aligned by record, which the caller
    guarantees by indexing both from the same record_id order.
    """
    diffs = [a[metric] - b[metric] for a, b in zip(a_scores, b_scores)]
    n = len(diffs)
    point = sum(diffs) / n
    boots = []
    for _ in range(n_boot):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        boots.append(sum(sample) / n)
    boots.sort()
    lo = boots[int(0.025 * n_boot)]
    hi = boots[int(0.975 * n_boot) - 1]
    return point, lo, hi


def align(a_rows, b_rows):
    a_by_id = {r["record_id"]: r for r in a_rows}
    b_by_id = {r["record_id"]: r for r in b_rows}
    common = [rid for rid in a_by_id if rid in b_by_id]
    if len(common) != len(a_rows) or len(common) != len(b_rows):
        raise SystemExit("record sets differ between arms; cannot pair")
    return ([a_by_id[rid] for rid in common], [b_by_id[rid] for rid in common])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="results/musique_reader/eval")
    ap.add_argument("--baseline", default="full")
    ap.add_argument("--rivals", default="plain,canonical_dedup,filter_only")
    ap.add_argument("--seed", type=int, default=0,
                    help="which seed's file to read; any file works, since "
                         "exact/overlap are deterministic and produce the "
                         "same per-record scores under every seed value")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--rng-seed", type=int, default=0,
                    help="seed for the bootstrap resampling itself, distinct "
                         "from the experiment seed above")
    ap.add_argument("--out", default="results/musique_reader/bootstrap.json")
    a = ap.parse_args()
    os.chdir(Path(__file__).resolve().parent.parent)

    rng = random.Random(a.rng_seed)
    base = load_rows(a.run, a.baseline, a.seed)
    rivals = {v: load_rows(a.run, v, a.seed) for v in a.rivals.split(",") if v}

    findings = []
    for key in sorted(base):
        for rival, rows in rivals.items():
            if key not in rows:
                continue
            a_scores, b_scores = align(base[key], rows[key])
            for metric in METRICS:
                point, lo, hi = bootstrap_diff(a_scores, b_scores, metric,
                                               a.n_boot, rng)
                findings.append({
                    "intervention": key[0], "multiplicity": key[1],
                    "metric": metric, "baseline": a.baseline, "rival": rival,
                    "n_records": len(a_scores), "diff": point,
                    "ci_low": lo, "ci_high": hi,
                    "excludes_zero": bool(lo > 0 or hi < 0)})

    print(f"{'interv':>8} {'m':>3} {'rival':>16} {'metric':>16} {'diff':>9} "
          f"{'bootstrap 95% CI':>22} {'excl0':>5}")
    for f in findings:
        print(f"{f['intervention']:>8} x{f['multiplicity']:<2} "
              f"{f['rival']:>16} {f['metric']:>16} {f['diff']:>+9.4f} "
              f"[{f['ci_low']:>+8.4f},{f['ci_high']:>+8.4f}] "
              f"{'YES' if f['excludes_zero'] else '':>5}")

    out = {"baseline": a.baseline, "n_records": len(next(iter(base.values()))),
          "n_boot": a.n_boot, "note": "records are the resampling unit; "
          "seeds are not, because exact and overlap are deterministic",
          "findings": findings}
    dest = Path(a.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\n-> {dest}")


if __name__ == "__main__":
    main()
