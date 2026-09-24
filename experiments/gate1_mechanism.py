"""Gate 1 -- does exact lineage remove a failure that no baseline can remove?

The experiment is deliberately decomposed so the claim is narrow and honest:

  A. tree + clean       : cavity BP is *exact* here.  Baselines are not straw.
  B. tree + duplicated  : no loop exists, yet every non-lineage method inflates.
                          => duplication is a distinct failure from cycles.
  C. cycle/torus + clean: the classical loop failure, for completeness.
  D. schedule sweep     : EC's fixed point must not move under async / delay /
                          reorder / repeat delivery.
  E. contests           : 2 lineage-distinct roots vs 100 descendants of one.

Outputs
  mechanism.csv    one row per (method, topology, regime, schedule, seed)
  trace.csv        confidence in nats vs iteration, for Figure 1
  duplication.csv  confidence vs duplication multiplier r, for Figures 2-3
  gate1_report.md  the five Gate-1 checkboxes, decided from the numbers
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _common import ResumableCSV, base_parser, start_run, summarise, write_json, write_rows  # noqa: E402
from ecnca.data import generate_example  # noqa: E402
from ecnca.evaluate import evaluate, run_method  # noqa: E402
from ecnca.gaussian import GaussianSpec  # noqa: E402
from ecnca.metrics import confidence, evidence_mass, score_cells  # noqa: E402
from ecnca.schedule import DeliverySchedule  # noqa: E402
from ecnca.topology import build, diameter  # noqa: E402

METHODS = ["ec_exact", "naive_sum", "mean_pool", "cov_int", "inv_cov_int",
           "bp_backtrack", "bp_cavity", "bp_trw"]
TOPOLOGIES = ["tree", "path", "cycle", "grid", "torus", "random", "complete"]
REGIMES = [("clean", 1), ("duplicate", 4), ("duplicate", 16), ("duplicate", 100),
           ("replacement", 4), ("conflicting", 8), ("sybil", 8), ("corrupt", 8)]
SCHEDULES = ["sync", "async", "lossy", "delayed", "repeating", "adversarial"]
METRICS = ["rmse", "rmse_vs_oracle", "nll", "coverage95", "prec_trace_ratio",
           "prec_rel_error", "confidence", "confidence_oracle", "consensus_spread"]


def main() -> None:
    ap = base_parser("gate1")
    args = ap.parse_args()
    if args.dry_run:
        args.seeds, args.steps = 1, 8
    out = start_run("gate1", args)
    spec = GaussianSpec(dim=args.dim)
    topologies = TOPOLOGIES[:2] if args.dry_run else TOPOLOGIES
    regimes = REGIMES[:2] if args.dry_run else REGIMES
    schedules = SCHEDULES[:1] if args.dry_run else SCHEDULES
    methods = METHODS[:3] if args.dry_run else METHODS

    # ---------------------------------------------------------------- A-D
    csv_out = ResumableCSV(os.path.join(out, "mechanism.csv"),
                           ["method", "topology", "regime", "duplication", "schedule", "seed"],
                           resume=args.resume == "auto")
    rows = []
    for seed in range(args.seeds):
        for topo in topologies:
            for regime, dup in regimes:
                ex = generate_example(spec, args.n_unique, np.random.default_rng(1000 + seed),
                                      topology=topo, n_cells=args.n_cells,
                                      regime=regime, duplication=dup)
                for sched in schedules:
                    for method in methods:
                        row = dict(method=method, topology=topo, regime=regime,
                                   duplication=dup, schedule=sched, seed=seed)
                        if csv_out.is_done(row):
                            continue
                        res = evaluate(method, ex, args.steps, np.random.default_rng(seed),
                                       capacity=args.capacity,
                                       schedule=DeliverySchedule.named(sched))
                        row.update({k: res[k] for k in METRICS})
                        csv_out.write(row)
                        rows.append(row)
        print(f"[gate1] seed {seed + 1}/{args.seeds} done", flush=True)

    # ------------------------------------------------- Figure 1: trace vs step
    probe = sorted(set(list(range(0, args.steps, 2)) + [args.steps - 1]))
    trace_rows = []
    for seed in range(args.seeds):
        for topo in (["tree", "torus"] if not args.dry_run else ["tree"]):
            ex = generate_example(spec, args.n_unique, np.random.default_rng(2000 + seed),
                                  topology=topo, n_cells=args.n_cells, regime="clean")
            for method in methods:
                _, _, _, tr = run_method(method, ex, args.steps, np.random.default_rng(seed),
                                      capacity=args.capacity, probe_steps=probe)
                for entry in tr:
                    conf = [confidence(spec, p) for p in entry["payload_sums"]]
                    trace_rows.append(dict(seed=seed, topology=topo, method=method,
                                           step=entry["step"], confidence=float(np.mean(conf)),
                                           confidence_oracle=confidence(spec, ex.unique_payload_sum)))
    write_rows(os.path.join(out, "trace.csv"), trace_rows)

    # ------------------------------- Figures 2-3: duplication multiplier sweep
    dup_rows = []
    multipliers = [1, 2, 4, 8, 16, 32, 100]
    for seed in range(args.seeds):
        for r in (multipliers[:2] if args.dry_run else multipliers):
            ex_dup = generate_example(spec, args.n_unique, np.random.default_rng(3000 + seed),
                                      topology="torus", n_cells=args.n_cells,
                                      regime="duplicate", duplication=r)
            ex_new = generate_example(spec, args.n_unique, np.random.default_rng(3000 + seed),
                                      topology="torus", n_cells=args.n_cells,
                                      regime="replacement", duplication=r)
            for method in methods:
                for tag, ex in (("duplicate", ex_dup), ("replacement", ex_new)):
                    res = evaluate(method, ex, args.steps, np.random.default_rng(seed),
                                   capacity=args.capacity)
                    dup_rows.append(dict(seed=seed, method=method, multiplier=r, arm=tag,
                                         **{k: res[k] for k in METRICS}))
    write_rows(os.path.join(out, "duplication.csv"), dup_rows)

    # ----------------------------------- Contest: 2 distinct roots vs 100 copies
    # Full-rank observations (obs_dim = dim) so that a single root CAN span the
    # latent space.  Otherwise log-det prefers two roots on rank grounds alone
    # and the contest measures linear algebra rather than evidence conservation.
    contest = []
    cspec = GaussianSpec(dim=args.dim, obs_dim=args.dim)
    for seed in range(args.seeds * 4):
        two = generate_example(cspec, 2, np.random.default_rng(4000 + seed), topology="torus",
                               n_cells=args.n_cells, regime="clean")
        one = generate_example(cspec, 1, np.random.default_rng(4000 + seed), topology="torus",
                               n_cells=args.n_cells, regime="duplicate", duplication=100)
        for method in methods:
            a, ca_cov, _, _ = run_method(method, two, args.steps, np.random.default_rng(seed),
                                         capacity=args.capacity)
            b, cb_cov, _, _ = run_method(method, one, args.steps, np.random.default_rng(seed),
                                         capacity=args.capacity)
            ca = float(np.mean([confidence(cspec, p, c) for p, c in zip(a, ca_cov)]))
            cb = float(np.mean([confidence(cspec, p, c) for p, c in zip(b, cb_cov)]))
            ma = float(np.mean([evidence_mass(cspec, p, c) for p, c in zip(a, ca_cov)]))
            mb = float(np.mean([evidence_mass(cspec, p, c) for p, c in zip(b, cb_cov)]))
            contest.append(dict(seed=seed, method=method, conf_two_distinct=ca,
                                conf_hundred_copies=cb, mass_two_distinct=ma,
                                mass_hundred_copies=mb, two_wins=int(ca > cb),
                                two_wins_mass=int(ma > mb)))
    write_rows(os.path.join(out, "contest.csv"), contest)

    # ------------------------------- Convergence: EC is a FIXED-POINT guarantee
    # Exactness is a property of the fixed point, not of any particular step
    # budget: a cell cannot know about evidence that has not reached it yet.
    # This measures how many steps that takes as a function of graph diameter
    # and delivery schedule, which is also the halting curve.
    conv_rows = []
    step_grid = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    for seed in range(args.seeds):
        for topo in topologies:
            diam = diameter(build(topo, args.n_cells, np.random.default_rng(0)))
            ex = generate_example(spec, args.n_unique, np.random.default_rng(8000 + seed),
                                  topology=topo, n_cells=args.n_cells,
                                  regime="duplicate", duplication=8)
            for sched in schedules:
                for st in (step_grid[:3] if args.dry_run else step_grid):
                    sums, covs, _, _ = run_method("ec_exact", ex, st,
                                                  np.random.default_rng(seed),
                                                  schedule=DeliverySchedule.named(sched))
                    sc = score_cells(spec, sums, ex.x_true, ex.unique_payload_sum, covs)
                    conv_rows.append(dict(seed=seed, topology=topo, diameter=diam,
                                          schedule=sched, steps=st,
                                          prec_rel_error=sc["prec_rel_error"],
                                          converged=int(sc["prec_rel_error"] < 1e-9)))
    write_rows(os.path.join(out, "convergence.csv"), conv_rows)

    # ------------------------------------------------------------- gate report
    write_rows(os.path.join(out, "mechanism_summary.csv"),
               summarise(rows or _read(os.path.join(out, "mechanism.csv")),
                         ["method", "topology", "regime", "duplication", "schedule"], METRICS))
    write_json(os.path.join(out, "gate1_report.json"),
               _gate_report(_read(os.path.join(out, "mechanism.csv")), dup_rows, contest,
                            conv_rows))
    print(f"[gate1] wrote {out}")


def _convergence_check(conv_rows):
    """Every (topology, schedule) must reach machine-exactness at some budget."""
    if not conv_rows:
        return {"pass": False, "reason": "convergence.csv not produced"}
    keys = {(r["topology"], r["schedule"]) for r in conv_rows}
    failed = [k for k in keys
              if not any(r["converged"] for r in conv_rows
                         if (r["topology"], r["schedule"]) == k)]
    return {"pass": not failed, "never_converged": sorted(failed)}


def _steps_to_converge(conv_rows):
    """Smallest step budget at which every seed hits machine-exactness."""
    if not conv_rows:
        return {}
    out = {}
    keys = sorted({(r["topology"], r["schedule"]) for r in conv_rows})
    for topo, sched in keys:
        sub = [r for r in conv_rows if r["topology"] == topo and r["schedule"] == sched]
        budgets = sorted({r["steps"] for r in sub})
        hit = None
        for b in budgets:
            at_b = [r["converged"] for r in sub if r["steps"] == b]
            if at_b and all(at_b):
                hit = b
                break
        out[f"{topo}/{sched}"] = {"diameter": sub[0]["diameter"], "steps": hit}
    return out


def _read(path):
    import csv as _csv
    with open(path) as fh:
        return list(_csv.DictReader(fh))


def _gate_report(rows, dup_rows, contest, conv_rows=None):
    def sel(**kw):
        return [r for r in rows if all(str(r[k]) == str(v) for k, v in kw.items())]

    def mean(rs, key):
        vals = [float(r[key]) for r in rs if r.get(key) not in ("", None)]
        vals = [v for v in vals if np.isfinite(v)]
        return float(np.mean(vals)) if vals else float("nan")

    naive_cycle = mean(sel(method="naive_sum", topology="cycle", regime="clean"), "prec_trace_ratio")
    cavity_cycle = mean(sel(method="bp_cavity", topology="cycle", regime="clean"), "prec_trace_ratio")
    cavity_tree_clean = mean(sel(method="bp_cavity", topology="tree", regime="clean"), "prec_trace_ratio")
    cavity_tree_dup = mean(sel(method="bp_cavity", topology="tree", regime="duplicate",
                               duplication=16), "prec_trace_ratio")
    # The theorem is about the fixed point.  Under synchronous delivery every
    # topology here reaches it inside the step budget, so that is the setting in
    # which exactness is a pass/fail claim; the stochastic schedules are scored
    # by whether they converge given enough steps (see convergence.csv).
    ec_all = mean([r for r in rows if r["method"] == "ec_exact"
                   and r["regime"] in ("clean", "duplicate")
                   and r["schedule"] == "sync"], "prec_rel_error")
    ec_dup100 = mean(sel(method="ec_exact", regime="duplicate", duplication=100), "prec_trace_ratio")
    ec_sched = {s: mean(sel(method="ec_exact", schedule=s, regime="duplicate", duplication=16),
                        "prec_trace_ratio") for s in SCHEDULES}
    two_wins = {m: float(np.mean([c["two_wins"] for c in contest if c["method"] == m]))
                for m in {c["method"] for c in contest}} if contest else {}
    two_wins_mass = {m: float(np.mean([c["two_wins_mass"] for c in contest if c["method"] == m]))
                     for m in {c["method"] for c in contest}} if contest else {}
    return {
        "checks": {
            "naive_overconfident_on_cycles": {"trace_ratio": naive_cycle, "pass": naive_cycle > 2.0},
            "cavity_fails_on_long_loops": {"trace_ratio": cavity_cycle, "pass": cavity_cycle > 2.0},
            "cavity_exact_on_clean_tree": {"trace_ratio": cavity_tree_clean,
                                           "pass": abs(cavity_tree_clean - 1.0) < 0.05},
            "cavity_fails_on_duplicated_tree": {"trace_ratio": cavity_tree_dup,
                                                "pass": cavity_tree_dup > 2.0},
            "ec_matches_centralised": {"prec_rel_error": ec_all, "pass": ec_all < 1e-8},
            "ec_100_copies_add_zero": {"trace_ratio": ec_dup100,
                                       "pass": abs(ec_dup100 - 1.0) < 1e-8},
            "ec_converges_under_every_schedule": _convergence_check(conv_rows),
        "ec_invariant_to_schedule": {"per_schedule": ec_sched,
                                         "pass": max(abs(v - 1.0) for k, v in ec_sched.items()
                                                     if k not in ("lossy", "adversarial")) < 1e-8},
        },
        "two_distinct_beats_hundred_copies": two_wins,
        "two_distinct_beats_hundred_copies_by_trace": two_wins_mass,
        "steps_to_converge": _steps_to_converge(conv_rows),
    }


if __name__ == "__main__":
    main()
