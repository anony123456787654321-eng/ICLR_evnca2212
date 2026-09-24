"""Which property each accounting rule gives up, and the credit that follows.

Four properties are asked of an accounting rule.

  A1  permutation invariance     the result does not depend on arrival order
  A2  idempotence within a root  replaying a held version is a no-op
  A3  additivity across roots    lineage-distinct payloads accumulate
  A4  bounded state              the store does not grow with delivery count

For the ledger, the provenance-free control, Dempster's rule and covariance
intersection, the credit at a given multiplicity follows from the rule itself.
The implementation computes credit by applying that rule, so agreement between
the closed form and the reported value checks the implementation. It is not an
empirical test and the manuscript does not present it as one. Surface
deduplication and non-backtracking message passing admit no closed form,
because what they credit depends on how the duplicates were constructed, so
their values are measurements.

The empirical parameter-free prediction in the manuscript is the calibration
penalty of analysis/calibration_penalty.py, which is compared against an
independently trained model.

Sources are the committed reports. Nothing here re-runs a model.
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

# Two-hop chains carry two lineage-distinct roots, which is what covariance
# intersection must divide its convex weight budget across.
ROOTS_TWO_HOP = 2
# Frozen before evaluation in ecnca/real/musique_duplication.py.
DEMPSTER_MASS = 0.5
MULT = 16

AXIOMS = ("A1", "A2", "A3", "A4")

# arm -> (satisfied axioms, the axiom whose violation sets its credit, why)
# LaTeX citation for each rule, so the comparison table carries its sources.
CITE = {
    "no defence": "",
    "surface deduplication": "\\citep{lee2022deduplicating,broder1997resemblance,charikar2002simhash}",
    "Dempster's rule": "\\citep{shafer1976dempster}",
    "covariance intersection": "\\citep{julier1997ci}",
    "non-backtracking": "\\citep{park2024nba}",
    "lineage ledger": "",
}

PROFILE = {
    "lineage ledger": (
        ("A1", "A2", "A3", "A4"), None,
        "max-register join within a root, additive union across roots"),
    "no defence": (
        ("A1", "A3"), "A2",
        "every arrival is credited, so credit tracks the delivery count"),
    "surface deduplication": (
        ("A1", "A3", "A4"), "A2",
        "idempotent only for byte-identical arrivals, so re-chunking defeats it"),
    "Dempster's rule": (
        ("A1", "A3", "A4"), "A2",
        "combination is not idempotent, so repetition sharpens belief"),
    "covariance intersection": (
        ("A1", "A2", "A4"), "A3",
        "convex weights sum to one, so distinct roots share one budget"),
    "non-backtracking": (
        ("A1", "A3"), "A2",
        "removes the reverse message, which is cyclic feedback and not duplication"),
}


def predictions():
    """Closed-form credited evidence implied by each rule, no fitted constant."""
    return {
        "no defence": (float(MULT),
                       "credits every arrival, so credit equals N"),
        "Dempster's rule": (1.0 / DEMPSTER_MASS,
                            "belief 1-(1-m)^N in units of one delivery "
                            "saturates at 1/m"),
        "covariance intersection": (1.0 / ROOTS_TWO_HOP,
                                    "convex weights summing to one divide "
                                    "across R roots, giving 1/R"),
        "lineage ledger": (1.0,
                           "a root contributes once however often delivered"),
    }


def measured_fusion(path):
    d = json.loads(Path(path).read_text())["findings"]
    out = {}
    key = ("overlap", "all_hops", MULT)
    for f in d:
        if (f["intervention"], f["scope"], f["multiplicity"]) != key:
            continue
        out.setdefault(f["pair"].split(":")[1], f["b_credit"][0])
        out.setdefault("full", f["a_credit"][0])
    return out


def measured_rechunk(path):
    """Credit at sixteenfold under the `rechunk` sweep, averaged over seeds.

    That sweep delivers the paragraph in full and then distinct windows cut
    from it, so it replaces the frozen partial-window sweep when present.
    """
    agg = defaultdict(list)
    for f in sorted(Path(path).glob("*_s*.json")):
        d = json.loads(f.read_text())
        for r in d["rows"]:
            if (r["intervention"], r["scope"], r["multiplicity"]) == \
                    ("rechunk", "all_hops", MULT):
                agg[d["variant"]].append(r["credited_over_ceiling"])
    return {v: st.mean(xs) for v, xs in agg.items()}


def measured_graph(path):
    agg = defaultdict(list)
    for r in json.loads(Path(path).read_text())["rows"]:
        agg[r["variant"]].append(r["credit_by_multiplicity"][str(MULT)])
    return {v: st.mean(xs) for v, xs in agg.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fusion", default="results/paired_fusion_h2.json")
    ap.add_argument("--rechunk", default="results/musique_rechunk_h2/eval")
    ap.add_argument("--graph", default="results/architecture/graph/report.json")
    ap.add_argument("--out", default="results/axiom_predictions.json")
    ap.add_argument("--tex", default="paper/generated/tab_axioms.tex")
    ap.add_argument("--tol", type=float, default=1e-3)
    a = ap.parse_args()
    os.chdir(Path(__file__).resolve().parent.parent)

    fus, gra = measured_fusion(a.fusion), measured_graph(a.graph)
    re = measured_rechunk(a.rechunk) if Path(a.rechunk).is_dir() else {}
    need = ("full", "plain", "canonical_dedup", "dempster_fusion",
            "covariance_intersection")
    if all(v in re for v in need):
        fus = re
        print(f"credits from the rechunk sweep, {a.rechunk}")
    obs = {
        "no defence": fus.get("plain"),
        "Dempster's rule": fus.get("dempster_fusion"),
        "covariance intersection": fus.get("covariance_intersection"),
        "lineage ledger": fus.get("full"),
        "surface deduplication": fus.get("canonical_dedup"),
        # Every detector, when the sweep measured them all; the row then shows
        # the range, since the detectors credit re-chunked windows differently.
        "non-backtracking": gra.get("nonbacktracking_mpnn"),
    }

    detectors = [fus.get(v) for v in ("canonical_dedup", "minhash_dedup",
                                        "simhash_dedup", "embed_dedup")]
    surface_range = ((min(detectors), max(detectors))
                     if all(x is not None for x in detectors) else None)

    rows, checks = [], []
    for arm, (sat, broken, why) in PROFILE.items():
        pred = predictions().get(arm)
        m = obs.get(arm)
        entry = {"arm": arm, "satisfies": list(sat),
                 "violates": broken, "reason": why, "measured": m}
        if pred is not None and m is not None:
            p, derivation = pred
            entry.update({"predicted": p, "derivation": derivation,
                          "abs_error": abs(p - m),
                          "agrees": bool(abs(p - m) <= a.tol)})
            checks.append(entry)
        rows.append(entry)

    tex = ["% generated by analysis/axiom_predictions.py"]
    # Rows whose value follows from the rule's definition or algebra come
    # first, then rows whose value can only be measured.
    order = ("lineage ledger", "no defence", "Dempster's rule",
             "covariance intersection", "surface deduplication",
             "non-backtracking")
    origin = {"lineage ledger": "definition", "no defence": "definition",
              "Dempster's rule": "algebra, $1/m$",
              "covariance intersection": "algebra, $1/R$",
              "surface deduplication": "measured",
              "non-backtracking": "measured"}
    by_arm = {r["arm"]: r for r in rows}
    for arm in order:
        r = by_arm[arm]
        marks = " & ".join("\\checkmark" if x in r["satisfies"] else "--"
                           for x in AXIOMS)
        val = f"${r['measured']:.3f}$" if r["measured"] is not None else "n/a"
        name = arm[0].upper() + arm[1:]
        name += (" " + CITE[arm]) if CITE.get(arm) else ""
        if arm == "surface deduplication" and surface_range \
                and surface_range[0] != surface_range[1]:
            r["measured_range"] = list(surface_range)
            val = f"${surface_range[0]:.3f}$ to ${surface_range[1]:.3f}$"
            name = ("Surface deduplication \\citep{lee2022deduplicating,"
                    "broder1997resemblance,charikar2002simhash,xiao2023bge}")
        if arm == "lineage ledger":
            name = "\\textbf{Lineage ledger (this work)}"
            val = f"$\\mathbf{{{r['measured']:.3f}}}$"
        tex.append(f"{name} & {marks} & {val} & {origin[arm]} \\\\")
    body = "\n".join(tex).rstrip()
    if body.endswith("\\\\"):
        body = body[:-2].rstrip()
    Path(a.tex).parent.mkdir(parents=True, exist_ok=True)
    Path(a.tex).write_text(body + "\n")

    agree = sum(c["agrees"] for c in checks)
    out = {"axioms": {"A1": "permutation invariance",
                      "A2": "idempotence within a root",
                      "A3": "additivity across lineage-distinct roots",
                      "A4": "bounded state"},
           "multiplicity": MULT, "roots_two_hop": ROOTS_TWO_HOP,
           "dempster_mass": DEMPSTER_MASS,
           "rows": rows, "n_predictions": len(checks), "n_agreeing": agree}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2) + "\n")

    print(f"{'arm':>24} {'A1 A2 A3 A4':>12} {'violates':>9} "
          f"{'predicted':>10} {'measured':>10} {'agrees':>7}")
    for r in rows:
        marks = " ".join(" y" if x in r["satisfies"] else " -" for x in AXIOMS)
        pr = f"{r['predicted']:.4f}" if "predicted" in r else "-"
        me = f"{r['measured']:.4f}" if r["measured"] is not None else "-"
        ag = ("yes" if r.get("agrees") else "no") if "predicted" in r else "-"
        print(f"{r['arm']:>24} {marks:>12} {str(r['violates']):>9} "
              f"{pr:>10} {me:>10} {ag:>7}")
    print(f"\nclosed forms reproduced by the implementation: "
          f"{agree} of {len(checks)}")
    print(f"-> {a.out}\n-> {a.tex}")


if __name__ == "__main__":
    main()
