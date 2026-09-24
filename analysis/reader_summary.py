"""Summarise the reader sweep, and check whether its seeds are independent.

The delivery-stream construction in ecnca.real.musique_duplication seeds a
random generator only for the reorder and paraphrase interventions. The exact
and overlap interventions never touch it: exact copies a byte-identical
payload, and overlap computes a fixed function of the paragraph text. Combined
with ground-truth passages and greedy reader decoding, a sweep restricted to
exact and overlap is therefore fully deterministic, and repeating it under
different seed values must reproduce the same output exactly.

This script verifies that rather than assuming it, because a sweep that quietly
failed to vary would look identical to one that succeeded for the wrong reason.
It then reports the comparison the sweep was run to make.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SEED_VARYING = {"reorder", "paraphrase"}


def load(run_dir: str, variant: str, seeds):
    out = {}
    for s in seeds:
        p = Path(run_dir) / f"{variant}_s{s}.json"
        if not p.is_file():
            continue
        for r in json.loads(p.read_text())["rows"]:
            key = (r["intervention"], r["multiplicity"])
            out.setdefault(key, {})[s] = r
    return out


def check_seed_independence(by_variant, interventions):
    """Report whether distinct seed files actually differ, per intervention."""
    findings = []
    for v, by_key in by_variant.items():
        for key, by_seed in by_key.items():
            interv = key[0]
            seeds = sorted(by_seed)
            if len(seeds) < 2:
                continue
            rows = [by_seed[s] for s in seeds]
            fields = ("exact_match", "answer_recall", "token_f1",
                      "n_context_tokens", "n_kept")
            identical = all(
                all(rows[0][f] == r[f] for f in fields) for r in rows[1:])
            findings.append({"variant": v, "intervention": interv,
                             "multiplicity": key[1], "seeds": seeds,
                             "identical_across_seeds": identical,
                             "expected_to_vary": interv in SEED_VARYING})
    return findings


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="results/musique_reader/eval")
    ap.add_argument("--variants",
                    default="full,filter_only,plain,canonical_dedup")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out", default="results/musique_reader/summary.json")
    a = ap.parse_args()
    os.chdir(Path(__file__).resolve().parent.parent)

    seeds = [int(s) for s in a.seeds.split(",") if s]
    variants = [v for v in a.variants.split(",") if v]
    by_variant = {v: load(a.run, v, seeds) for v in variants}

    checks = check_seed_independence(by_variant, a.variants)
    surprising = [c for c in checks
                 if c["identical_across_seeds"] != (not c["expected_to_vary"])]

    print("seed-independence check (exact and overlap carry no seeded "
          "randomness by construction; a match here is expected, not noise)")
    for c in checks:
        tag = "identical" if c["identical_across_seeds"] else "varies"
        flag = "" if not surprising or c not in surprising else "  <- UNEXPECTED"
        print(f"  {c['variant']:>16} {c['intervention']:>10} "
              f"x{c['multiplicity']:<3} seeds={c['seeds']}  {tag}{flag}")
    if surprising:
        print(f"\n{len(surprising)} cells behaved unexpectedly; investigate "
              "before trusting any interval built from these seeds.")
    else:
        print("\nAll cells behave exactly as the code predicts: deterministic "
              "for exact/overlap. Point estimates below are single "
              "deterministic runs over 120 records, not independent replicates. "
              "No paired interval over these seed files would be valid.")

    print(f"\n{'variant':>16} {'interv':>8} {'m':>3} {'F1':>8} {'recall':>8} "
          f"{'ctx tok':>9} {'kept':>6} {'credit':>8}")
    rows = []
    for v in variants:
        for key, by_seed in sorted(by_variant[v].items()):
            r = by_seed[sorted(by_seed)[0]]
            rows.append(r)
            print(f"{v:>16} {key[0]:>8} x{key[1]:<3} {r['token_f1']:>8.4f} "
                  f"{r['answer_recall']:>8.4f} {r['n_context_tokens']:>9.1f} "
                  f"{r['n_kept']:>6.2f} {r['credited_over_ceiling']:>8.2f}")

    out = {"seed_independence_checks": checks,
           "n_unexpected": len(surprising),
           "rows": rows}
    dest = Path(a.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\n-> {dest}")


if __name__ == "__main__":
    main()
