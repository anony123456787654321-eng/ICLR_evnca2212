"""Two axes of evidence accounting, and why no baseline gets both right.

Credit accounting depends on root identity and arrival multiplicity, not on a
trained checkpoint, so this runs without one. Two measurements:

* REPETITION. One source delivered many times. A method passes when credited
  evidence stays at the lineage-distinct ceiling.
* DISTINCTNESS. R lineage-distinct roots, matched on message count against R
  descendants of a single root. A method passes when it credits the distinct
  configuration strictly higher. This is the contest of Table gate1-full,
  extended to the fusion rules.

The point of reporting both is that the two failures are not the same failure.
Surface deduplication and Dempster's rule inflate under repetition. Covariance
intersection does not inflate, because convex weights cannot exceed the largest
input, but it pays for that by dividing credit across lineage-distinct roots
which it cannot tell apart from repetitions. Only a lineage ledger passes both.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ecnca.real.musique import MuSiQueRecord, MuSiQueStep
from ecnca.real.musique_duplication import (MULTIPLICITIES, base_stream,
                                            build_stream, resolve_stream)

OUT = Path("results/fusion_taxonomy.json")

# name -> (mode, family), in the order the table should read.
# The embed detector requires a real embedding callable and refuses to run
# without one, which is correct: without it the detector would accept
# everything and be scored as a defence that is not defending. Its behaviour is
# already measured against real text in the duplicated-hop grid, so this
# structural comparison names it and records that it was not run here.
NEEDS_EMBEDDER = {"embed"}

ARMS = [
    ("plain", "none", "no defence"),
    ("canonical_dedup", "canonical", "surface deduplication"),
    ("minhash_dedup", "minhash", "surface deduplication"),
    ("simhash_dedup", "simhash", "surface deduplication"),
    ("embed_dedup", "embed", "surface deduplication"),
    ("dempster_fusion", "dempster", "evidence fusion"),
    ("covariance_intersection", "covariance_intersection", "evidence fusion"),
    ("full", "max", "lineage ledger (ours)"),
]


def synthetic(n_hops: int = 4, words: int = 40) -> MuSiQueRecord:
    """A structurally faithful stand-in when the corpus is not restored.

    Credit accounting reads root identity and multiplicity. The detectors also
    read text, so the paragraphs are distinct and long enough for shingling.
    """
    steps = tuple(MuSiQueStep(
        question=f"question {i}", answer=f"answer{i}",
        paragraph_title=f"Title {i}",
        paragraph_text=" ".join(f"w{i}x{j}" for j in range(words)),
        paragraph_idx=i, dependencies=() if i == 0 else (i,))
        for i in range(n_hops))
    return MuSiQueRecord(record_id="rec", question="Q",
                         answer=f"answer{n_hops - 1}", answer_aliases=(),
                         steps=steps, is_linear=True, linear_failure="")


def repetition(record, interventions, mults):
    ceiling = len({e.root_id for e in base_stream(record)})
    rows = []
    for name, mode, family in ARMS:
        if mode in NEEDS_EMBEDDER:
            continue
        for interv in interventions:
            for m in mults:
                events = build_stream(record, interv, m, "one_hop", seed=0)
                credit = sum(resolve_stream(events, mode)[1].values())
                rows.append({"arm": name, "family": family,
                             "intervention": interv, "multiplicity": m,
                             "ceiling": ceiling, "credit": credit,
                             "ratio": credit / ceiling,
                             "conserves": abs(credit - ceiling) < 1e-9})
    return rows


def distinctness(n):
    """R lineage-distinct roots against R descendants of one, matched on count.

    The two configurations deliver the SAME number of messages and differ only
    in how many distinct sources produced them:

      distinct  R roots, one delivery each    -> R messages, R roots
      repeated  one root, R deliveries        -> R messages, 1 root

    Matching on message count is what makes this a test of evidence accounting
    rather than of message volume. A method passes when it credits the distinct
    configuration strictly higher, because R independent observations are worth
    more than R deliveries of one.
    """
    distinct = base_stream(synthetic(n_hops=n))
    repeated = build_stream(synthetic(n_hops=1), "exact", n, "one_hop", seed=0)
    assert len(distinct) == len(repeated) == n, (len(distinct), len(repeated))
    rows = []
    for name, mode, family in ARMS:
        if mode in NEEDS_EMBEDDER:
            continue
        d = sum(resolve_stream(distinct, mode)[1].values())
        r = sum(resolve_stream(repeated, mode)[1].values())
        rows.append({"arm": name, "family": family, "n": n,
                     "credit_distinct": d, "credit_repeated": r,
                     "prefers_distinct": bool(d > r + 1e-9)})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--hops", type=int, default=4)
    ap.add_argument("--interventions", default="exact,overlap,cycle")
    ap.add_argument("--multiplicities",
                    default=",".join(map(str, MULTIPLICITIES)))
    a = ap.parse_args()
    os.chdir(Path(__file__).resolve().parent.parent)

    record = synthetic(n_hops=a.hops)
    interventions = [s for s in a.interventions.split(",") if s]
    mults = [int(s) for s in a.multiplicities.split(",") if s]

    rep = repetition(record, interventions, mults)
    dst = distinctness(max(mults))

    def at(name, interv, m):
        for r in rep:
            if r["arm"] == name and r["intervention"] == interv \
                    and r["multiplicity"] == m:
                return r["credit"]
        return None

    lo, hi = min(mults), max(mults)
    passes = {}
    for name, mode, family in ARMS:
        if mode in NEEDS_EMBEDDER:
            passes[name] = {"family": family, "not_run": True,
                            "reason": "needs a real embedding callable; "
                                      "measured in the duplicated-hop grid"}
            continue
        holds = all(r["conserves"] for r in rep if r["arm"] == name)
        prefers = all(r["prefers_distinct"] for r in dst if r["arm"] == name)
        # Conservation can fail in two opposite directions, and the direction
        # is the interesting part: a method that inflates is over-counting
        # repetitions, while one that deflates is under-counting real sources.
        # Reporting only "conserves" would file both under one verdict.
        inflates = any(at(name, i, hi) > at(name, i, lo) + 1e-9
                       for i in interventions)
        deflates = any(at(name, i, hi) < at(name, i, lo) - 1e-9
                       for i in interventions)
        passes[name] = {"family": family,
                        "conserves_under_repetition": holds,
                        "inflates_under_repetition": inflates,
                        "deflates_under_repetition": deflates,
                        "credits_distinct_roots": prefers,
                        "passes_both": holds and prefers}

    out = {"source": "synthetic structural record",
           "not_run": sorted(NEEDS_EMBEDDER),
           "hops": a.hops, "repetition": rep, "distinctness": dst,
           "verdict": passes}
    dest = Path(a.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2) + "\n")

    print(f"{'arm':>24} {'family':>22} {'inflates':>9} {'deflates':>9} "
          f"{'conserves':>10} {'credits distinct':>17} {'both':>6}")
    for name, _, _ in ARMS:
        v = passes[name]
        if v.get("not_run"):
            print(f"{name:>24} {v['family']:>22} {'-':>9} {'-':>9} "
                  f"{'not run here':>10} {'not run here':>17} {'-':>6}")
            continue
        print(f"{name:>24} {v['family']:>22} "
              f"{str(v['inflates_under_repetition']):>9} "
              f"{str(v['deflates_under_repetition']):>9} "
              f"{str(v['conserves_under_repetition']):>10} "
              f"{str(v['credits_distinct_roots']):>17} "
              f"{str(v['passes_both']):>6}")
    print(f"\n-> {dest}")


if __name__ == "__main__":
    main()
