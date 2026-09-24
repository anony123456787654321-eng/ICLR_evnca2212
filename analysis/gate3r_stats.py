"""Five-seed statistical report for Gate 3R, from the raw per-seed CSV.

Every quantity is computed per seed and then aggregated, so the confidence
intervals are over seeds rather than over evaluation batches.  The paired
difference `full_gain - filter_gain` is computed WITHIN seed: the two variants
are trained independently and start from different gaps, so the relative gain is
the comparable quantity, not the endpoint.

Replication passes only if all six criteria hold.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

DEPTHS = [0, 1, 2, 4, 8]
# Drift is scored RELATIVE, and the two drifts have DIFFERENT UNITS, so they are
# normalised by different baselines:
#
#   prediction drift : ||mu_k - mu_1||, a distance in prediction space.
#                      Normalised by the baseline gap-to-oracle of the same
#                      axis at setting 1 -- also a prediction-space distance.
#   precision drift  : |trace_k - trace_1|, in precision units.
#                      Normalised by the baseline precision trace at setting 1.
#
# Dividing the prediction drift by a precision trace (as an earlier version did)
# compares quantities with different units and its magnitude is meaningless.
#
# The EXACT invariant is separate: credited evidence may never exceed the root's
# ceiling (`every_full_seed_bounded`).  The residual relative drift comes from
# `rho` being learned from (z, h), where h depends on when a cell received the
# message -- arrival-time sensitivity, not double counting.
REL_TOL = 0.02
DRIFT_NORMALISATION = {
    "prediction": "baseline gap-to-oracle at setting 1 of the same axis",
    "precision": "baseline precision trace at setting 1 of the same axis",
}


def _t_ci(x, alpha=0.05):
    """Two-sided t interval for the mean; falls back gracefully for n < 2."""
    x = np.asarray([v for v in x if np.isfinite(v)], dtype=float)
    n = len(x)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    m = float(x.mean())
    if n < 2:
        return m, float("nan"), float("nan")
    from scipy import stats
    se = x.std(ddof=1) / np.sqrt(n)
    h = float(stats.t.ppf(1 - alpha / 2, n - 1) * se)
    return m, m - h, m + h


def _boot_ci(x, alpha=0.05, n_boot=20000, seed=0):
    x = np.asarray([v for v in x if np.isfinite(v)], dtype=float)
    if len(x) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    b = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(b, 100 * alpha / 2)), float(np.percentile(b, 100 * (1 - alpha / 2)))


def _curve(df, variant, seed, axis, metric):
    sub = df[(df.variant == variant) & (df.seed == seed) & (df.axis == axis)]
    xs = sorted(sub.setting.unique())
    return xs, [float(sub[sub.setting == x][metric].mean()) for x in xs]


def per_seed(df, variant, seed):
    """All Gate 3R quantities for one (variant, seed)."""
    d, gap = _curve(df, variant, seed, "refinement", "gap_to_oracle")
    _, pr = _curve(df, variant, seed, "refinement", "precision_ratio")
    if not gap or gap[0] <= 0:
        return None
    gain = (gap[0] - gap[-1]) / gap[0]
    # normalised area under the refinement-gap curve: 1.0 = no improvement at
    # any depth, lower is better.  Trapezoid over the depth index, so unequal
    # depth spacing does not weight the tail more heavily.
    g = np.asarray(gap) / gap[0]
    trapz = getattr(np, "trapezoid", None) or np.trapz   # numpy 2 renamed it
    auc = float(trapz(g, dx=1.0) / (len(g) - 1))
    out = dict(variant=variant, seed=seed, gain=gain, gap_auc=auc,
               max_precision_ratio=float(np.max(pr)) if pr else float("nan"))
    for axis, tag in (("pure_redelivery", "pure"), ("parallel_refinement", "par")):
        st, pd_ = _curve(df, variant, seed, axis, "pred_drift")
        _, cd_ = _curve(df, variant, seed, axis, "prec_drift")
        _, pr_ = _curve(df, variant, seed, axis, "precision_ratio")
        _, gp_ = _curve(df, variant, seed, axis, "gap_to_oracle")
        _, tr_ = _curve(df, variant, seed, axis, "trace")
        out[f"{tag}_pred_drift"] = float(np.max(pd_)) if pd_ else float("nan")
        out[f"{tag}_prec_drift"] = float(np.max(cd_)) if cd_ else float("nan")
        out[f"{tag}_max_precision_ratio"] = float(np.max(pr_)) if pr_ else float("nan")
        # different units -> different baselines (see DRIFT_NORMALISATION)
        pred_scale = float(gp_[0]) if gp_ else 1.0        # prediction-space
        prec_scale = float(tr_[0]) if tr_ else 1.0        # precision-space
        out[f"{tag}_pred_scale"] = pred_scale
        out[f"{tag}_prec_scale"] = prec_scale
        out[f"{tag}_pred_drift_rel"] = out[f"{tag}_pred_drift"] / max(pred_scale, 1e-9)
        out[f"{tag}_prec_drift_rel"] = out[f"{tag}_prec_drift"] / max(prec_scale, 1e-9)
    nk, nr = _curve(df, variant, seed, "new_source", "rmse")
    _, nt = _curve(df, variant, seed, "new_source", "trace")
    if len(nr) >= 2 and nr[0] > 0:
        out["new_source_rmse_improvement"] = (nr[0] - nr[-1]) / nr[0]
        out["new_source_precision_increase"] = (nt[-1] - nt[0]) / max(nt[0], 1e-9)
    return out


def main():
    ap = argparse.ArgumentParser("gate3r_stats")
    ap.add_argument("--csv", default="results/gate3r_path2/gate3r_metrics.csv")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    df = pd.read_csv(a.csv)
    seeds = sorted(df.seed.unique())
    rows = [r for v in sorted(df.variant.unique()) for s in seeds
            if (r := per_seed(df, v, s)) is not None]
    ps = pd.DataFrame(rows)

    def col(variant, key):
        x = ps[ps.variant == variant][key]
        return x.to_numpy(dtype=float) if len(x) else np.array([])

    full_gain, filt_gain = col("full", "gain"), col("filter_only", "gain")
    n = min(len(full_gain), len(filt_gain))
    paired = full_gain[:n] - filt_gain[:n]          # within-seed difference

    fm, flo, fhi = _t_ci(full_gain)
    dm, dlo, dhi = _t_ci(paired)
    blo, bhi = _boot_ci(paired)
    prov_free = np.concatenate([col("no_provenance", "max_precision_ratio"),
                                col("plain", "max_precision_ratio")]) \
        if len(col("no_provenance", "max_precision_ratio")) else np.array([])

    crit = {
        "full_gain_mean_ge_25pct": bool(np.isfinite(fm) and fm >= 0.25),
        "paired_diff_ci_above_zero": bool(np.isfinite(dlo) and dlo > 0),
        "every_full_seed_bounded": bool(len(full_gain) > 0
                                        and np.all(col("full", "max_precision_ratio") < 1.05)),
        "pure_redelivery_within_tolerance": bool(
            np.all(col("full", "pure_pred_drift_rel") < REL_TOL)
            and np.all(col("full", "pure_prec_drift_rel") < REL_TOL)),
        "new_source_helps": bool(np.all(col("full", "new_source_rmse_improvement") > 0)
                                 and np.all(col("full", "new_source_precision_increase") > 0)),
        "provenance_free_exceeds_ceiling": bool(len(prov_free) > 0 and np.all(prov_free > 1.05)),
    }
    def _ms(x):
        x = np.asarray([v for v in x if np.isfinite(v)], dtype=float)
        return {"mean": float(x.mean()) if len(x) else float("nan"),
                "std": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
                "per_seed": x.tolist()}

    report = {
        "n_seeds": len(seeds),
        "drift_normalisation": DRIFT_NORMALISATION,
        "summary_mean_std": {v: {k: _ms(col(v, k)) for k in
                                 ("gain", "gap_auc", "max_precision_ratio",
                                  "pure_pred_drift_rel", "pure_prec_drift_rel",
                                  "par_max_precision_ratio",
                                  "new_source_rmse_improvement",
                                  "new_source_precision_increase")}
                             for v in sorted(df.variant.unique())},
        "full_gain": {"mean": fm, "ci95": [flo, fhi], "per_seed": full_gain.tolist()},
        "filter_only_gain": {"mean": float(np.mean(filt_gain)) if len(filt_gain) else float("nan"),
                             "per_seed": filt_gain.tolist()},
        "paired_full_minus_filter": {"mean": dm, "ci95_t": [dlo, dhi],
                                     "ci95_bootstrap": [blo, bhi],
                                     "per_seed": paired.tolist()},
        "gap_auc_normalised": {v: col(v, "gap_auc").tolist()
                               for v in sorted(df.variant.unique())},
        "max_precision_ratio": {v: col(v, "max_precision_ratio").tolist()
                                for v in sorted(df.variant.unique())},
        "pure_redelivery_drift_full": {
            "pred_absolute": col("full", "pure_pred_drift").tolist(),
            "prec_absolute": col("full", "pure_prec_drift").tolist(),
            "pred_relative": col("full", "pure_pred_drift_rel").tolist(),
            "prec_relative": col("full", "pure_prec_drift_rel").tolist(),
            "pred_scale_used": col("full", "pure_pred_scale").tolist(),
            "prec_scale_used": col("full", "pure_prec_scale").tolist()},
        "parallel_refinement_full": {"pred_drift": col("full", "par_pred_drift").tolist(),
                                     "max_precision_ratio": col("full", "par_max_precision_ratio").tolist()},
        "new_source_full": {"rmse_improvement": col("full", "new_source_rmse_improvement").tolist(),
                            "precision_increase": col("full", "new_source_precision_increase").tolist()},
        "provenance_free_max_precision_ratio": prov_free.tolist(),
        "criteria": crit,
        "REPLICATION_PASS": all(crit.values()),
    }
    out = a.out or os.path.join(os.path.dirname(a.csv), "gate3r_stats.json")
    json.dump(report, open(out, "w"), indent=2, default=float)
    ps.to_csv(os.path.join(os.path.dirname(a.csv), "gate3r_per_seed.csv"), index=False)

    print(f"seeds: {len(seeds)}   (CIs over seeds; n<2 gives NaN by design)")
    print(f"full gain            mean {fm:+.4f}  95% CI [{flo:+.4f}, {fhi:+.4f}]")
    print(f"filter_only gain     mean {report['filter_only_gain']['mean']:+.4f}")
    print(f"paired full-filter   mean {dm:+.4f}  t95 [{dlo:+.4f}, {dhi:+.4f}]  "
          f"boot95 [{blo:+.4f}, {bhi:+.4f}]")
    print(f"gap AUC (full)       {np.round(col('full','gap_auc'), 4).tolist()}")
    print(f"max prec/ceiling     full {np.round(col('full','max_precision_ratio'), 4).tolist()}  "
          f"prov-free {np.round(prov_free, 2).tolist()}")
    print(f"pure redeliv drift   pred_rel {np.round(col('full','pure_pred_drift_rel'), 5).tolist()} "
          f"(/gap {np.round(col('full','pure_pred_scale'), 3).tolist()})  "
          f"prec_rel {np.round(col('full','pure_prec_drift_rel'), 6).tolist()} "
          f"(/trace {np.round(col('full','pure_prec_scale'), 2).tolist()})")
    print(f"parallel refinement  pred drift {np.round(col('full','par_pred_drift'), 4).tolist()}  "
          f"max ratio {np.round(col('full','par_max_precision_ratio'), 4).tolist()}")
    print(f"new source           rmse +{np.round(col('full','new_source_rmse_improvement'), 4).tolist()}  "
          f"prec +{np.round(col('full','new_source_precision_increase'), 4).tolist()}")
    print()
    for k, v in crit.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print(f"\nREPLICATION_PASS: {report['REPLICATION_PASS']}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
