"""Gate 3R -- Learned Lineage Refinement.

    neural computation can improve a message derived from one source, without
    that computation being counted as additional independent evidence.

Four axes, kept strictly separate:

  refinement          sequential depth: the learned payload changes along a
                      chain, root identity fixed              <-- the claim
  pure_redelivery     max_version=0, so NO computation happens: an identical
                      payload is delivered repeatedly.  Prediction AND precision
                      must both be invariant -- this is the clean duplicate test.
  parallel_refinement max_version=8 with repeated delivery, so several copies
                      refine concurrently.  The prediction MAY change (that is
                      computation) but precision must stay under the ceiling.
  new_source          a genuinely lineage-distinct observation arrives

Paired evaluation: each raw example is generated ONCE and evaluated at
refinement depths 0, 1, 2, 4, 8 with the same latent, observation, noise, root
identity, graph and delivery schedule.  Every evidence tensor is asserted
bit-identical across the paired conditions.

Both posterior-mean error and gap to the exact centralised posterior are
reported: RMSE against the latent alone has an irreducible noise floor and
cannot show refinement.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _common import write_json, write_rows  # noqa: E402
from ecnca.neural.refine import (RefineSpec, RefiningECNCA, make_refine_batch,  # noqa: E402
                                 oracle_posterior, train_refine)

VARIANTS = ["full", "filter_only", "no_provenance", "plain"]
DEPTHS = [0, 1, 2, 4, 8]
PAIRED_KEYS = ("feat", "root_Lam", "root_h", "valid", "W0", "x_true",
               "h_oracle", "Lam_oracle", "adj")


def _snapshot(b):
    return {k: b[k].clone() for k in PAIRED_KEYS}


def _assert_unchanged(b, snap, tag):
    for k in PAIRED_KEYS:
        if not torch.equal(b[k], snap[k]):
            raise AssertionError(f"paired condition '{tag}' altered {k}")


@torch.no_grad()
def score(model, b, steps, max_version):
    mu_or, _ = oracle_posterior(b)
    o = model(b, steps=steps, max_version=max_version)
    mu = o["mu"].mean(1)
    ceiling = b["root_Lam"].diagonal(dim1=-2, dim2=-1).sum(-1).sum(-1).clamp(min=1e-6)
    return dict(
        gap_to_oracle=float(((mu - mu_or) ** 2).sum(-1).sqrt().mean()),
        rmse=float(((mu - b["x_true"]) ** 2).sum(-1).sqrt().mean()),
        precision_ratio=float((o["M"].max(dim=1).values / ceiling).mean()),
        max_version=float(o["version"].max()),
        trace=float(o["Lam"].diagonal(dim1=-2, dim2=-1).sum(-1).max()),
    )


@torch.no_grad()
def probe(model, spec, seed, steps, device, n_batch=6, bs=24):
    rows = []
    for bi in range(n_batch):
        rng = np.random.default_rng(80_000 + 131 * seed + bi)
        # ---- ONE raw example set, evaluated at every refinement depth --------
        base = make_refine_batch(spec, bs, n_roots=1, redeliveries=1, rng=rng, device=device)
        snap = _snapshot(base)
        for v in DEPTHS:
            rows.append(dict(axis="refinement", setting=v,
                             **score(model, base, steps, v)))
            _assert_unchanged(base, snap, f"refinement/{v}")
        # ---- pure_redelivery: no computation at all, so nothing may move -----
        # ---- parallel_refinement: copies refine concurrently ------------------
        for axis, mv in (("pure_redelivery", 0), ("parallel_refinement", max(DEPTHS))):
            ref_mu = ref_tr = None
            for k in (1, 2, 4, 8):
                b = make_refine_batch(spec, bs, n_roots=1, redeliveries=k,
                                      rng=np.random.default_rng(80_000 + 131 * seed + bi),
                                      device=device)
                assert torch.equal(b["feat"], base["feat"]), "redelivery changed the payload"
                sc = score(model, b, steps, mv)
                o = model(b, steps=steps, max_version=mv)
                mu_k = o["mu"].mean(1)
                tr_k = o["Lam"].diagonal(dim1=-2, dim2=-1).sum(-1).max(dim=1).values
                if k == 1:
                    ref_mu, ref_tr = mu_k, tr_k
                sc["pred_drift"] = float(((mu_k - ref_mu) ** 2).sum(-1).sqrt().mean())
                sc["prec_drift"] = float((tr_k - ref_tr).abs().mean())
                rows.append(dict(axis=axis, setting=k, **sc))
        # ---- new_source: a genuinely lineage-distinct observation -------------
        for n in (1, 2):
            b = make_refine_batch(spec, bs, n_roots=n, redeliveries=1,
                                  rng=np.random.default_rng(80_000 + 131 * seed + bi),
                                  device=device)
            assert torch.equal(b["feat"][:, 0], base["feat"][:, 0]), "first root changed"
            rows.append(dict(axis="new_source", setting=n, **score(model, b, steps, max(DEPTHS))))
    agg = []
    for axis in ("refinement", "pure_redelivery", "parallel_refinement", "new_source"):
        for st in sorted({r["setting"] for r in rows if r["axis"] == axis}):
            sub = [r for r in rows if r["axis"] == axis and r["setting"] == st]
            agg.append({"axis": axis, "setting": st,
                        **{k: float(np.mean([r[k] for r in sub]))
                           for k in sub[0] if k not in ("axis", "setting")}})
    return agg


def main():
    import argparse
    ap = argparse.ArgumentParser("gate3r")
    ap.add_argument("--out", default="results/gate3r")
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--cells", type=int, default=4)
    ap.add_argument("--topology", default="path")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    spec = RefineSpec(n_cells=args.cells, topology=args.topology)
    print(f"[gate3r] {args.topology} n={args.cells} depths={DEPTHS}")

    rows = []
    for variant in args.variants.split(","):
        for seed in range(args.seeds):
            d = os.path.join(args.out, f"{variant}_s{seed}")
            ck = os.path.join(d, "ckpt.pt")
            if os.path.exists(ck) and torch.load(ck, map_location="cpu",
                                                 weights_only=False)["iter"] >= args.iters:
                model = RefiningECNCA(variant=variant).to(args.device)
                model.load_state_dict(torch.load(ck, map_location=args.device,
                                                 weights_only=False)["model"])
                print(f"[gate3r] {variant} s{seed}: loaded")
            else:
                model, _ = train_refine(variant, spec, iters=args.iters,
                                        batch_size=args.batch_size, steps=args.steps,
                                        seed=seed, device=args.device, out=d,
                                        log_every=max(100, args.iters // 6))
            model.eval()
            for r in probe(model, spec, seed, args.steps, args.device):
                rows.append(dict(variant=variant, seed=seed,
                                 topology=args.topology, **r))

    write_rows(os.path.join(args.out, "gate3r_metrics.csv"), rows)
    write_json(os.path.join(args.out, "gate3r_report.json"), _report(rows))
    print(f"[gate3r] wrote {args.out}")


def _report(rows):
    out = {}
    for v in sorted({r["variant"] for r in rows}):
        def curve(axis, key):
            sub = [r for r in rows if r["variant"] == v and r["axis"] == axis]
            xs = sorted({r["setting"] for r in sub})
            return xs, [float(np.mean([r[key] for r in sub if r["setting"] == x])) for x in xs]

        dv, gap = curve("refinement", "gap_to_oracle")
        _, pr = curve("refinement", "precision_ratio")
        _, rm = curve("refinement", "rmse")
        rk, r_pr = curve("pure_redelivery", "precision_ratio")
        _, r_gap = curve("pure_redelivery", "gap_to_oracle")
        _, pure_pd = curve("pure_redelivery", "pred_drift")
        _, pure_cd = curve("pure_redelivery", "prec_drift")
        pk, par_pr = curve("parallel_refinement", "precision_ratio")
        _, par_pd = curve("parallel_refinement", "pred_drift")
        # gap-to-oracle is the right metric for REFINEMENT (same evidence,
        # better extraction) but the wrong one for NEW_SOURCE, where adding a
        # root moves the oracle itself.  There we score error against the latent
        # and absolute precision, not the ratio to a ceiling that just grew.
        nk, n_gap = curve("new_source", "gap_to_oracle")
        _, n_rmse = curve("new_source", "rmse")
        _, n_pr = curve("new_source", "precision_ratio")
        _, n_tr = curve("new_source", "trace")

        gain = (gap[0] - gap[-1]) / gap[0] if gap and gap[0] > 0 else 0.0
        bounded = max(pr) <= 1.05 if pr else False
        # pure redelivery must not move EITHER quantity; parallel refinement may
        # move the prediction but never the ceiling
        # Prediction and precision drift have DIFFERENT UNITS and take
        # different baselines, matching analysis/gate3r_stats.py exactly:
        #   prediction drift / baseline gap-to-oracle  (prediction space)
        #   precision  drift / baseline precision trace (precision space)
        # Dividing both by the trace -- as this report previously did -- is a
        # unit error that makes the prediction criterion pass spuriously.
        _, pure_tr = curve("pure_redelivery", "trace")
        _, pure_gp = curve("pure_redelivery", "gap_to_oracle")
        pred_scale = float(pure_gp[0]) if pure_gp else 1.0
        prec_scale = float(pure_tr[0]) if pure_tr else 1.0
        redeliv_flat = bool(pure_pd
                            and max(pure_pd) / max(pred_scale, 1e-9) < 0.02
                            and max(pure_cd) / max(prec_scale, 1e-9) < 0.02)
        parallel_bounded = bool(par_pr and max(par_pr) <= 1.05)
        new_helps = (n_rmse[-1] < n_rmse[0] * 0.98 and n_tr[-1] > n_tr[0] * 1.02) if n_rmse else False
        out[v] = {
            "refinement_gap": dict(zip(map(str, dv), [round(g, 4) for g in gap])),
            "refinement_rmse": dict(zip(map(str, dv), [round(g, 4) for g in rm])),
            "refinement_precision_ratio": dict(zip(map(str, dv), [round(g, 4) for g in pr])),
            "pure_redelivery_precision_ratio": dict(zip(map(str, rk), [round(g, 4) for g in r_pr])),
            "pure_redelivery_gap": dict(zip(map(str, rk), [round(g, 4) for g in r_gap])),
            "pure_redelivery_pred_drift": dict(zip(map(str, rk), [round(g, 6) for g in pure_pd])),
            "pure_redelivery_prec_drift": dict(zip(map(str, rk), [round(g, 6) for g in pure_cd])),
            "pure_redelivery_pred_drift_rel": round(max(pure_pd) / max(pred_scale, 1e-9), 6)
            if pure_pd else float("nan"),
            "pure_redelivery_prec_drift_rel": round(max(pure_cd) / max(prec_scale, 1e-9), 8)
            if pure_cd else float("nan"),
            "drift_normalisation": {
                "prediction": "baseline gap-to-oracle at setting 1 of the same axis",
                "precision": "baseline precision trace at setting 1 of the same axis"},
            "parallel_refinement_precision_ratio": dict(zip(map(str, pk), [round(g, 4) for g in par_pr])),
            "parallel_refinement_pred_drift": dict(zip(map(str, pk), [round(g, 4) for g in par_pd])),
            "new_source_rmse": dict(zip(map(str, nk), [round(g, 4) for g in n_rmse])),
            "new_source_trace": dict(zip(map(str, nk), [round(g, 4) for g in n_tr])),
            "new_source_gap_secondary": dict(zip(map(str, nk), [round(g, 4) for g in n_gap])),
            "gain_v0_to_final": round(gain, 4),
            "confidence_bounded": bounded,
            "pure_redelivery_invariant": redeliv_flat,
            "parallel_refinement_bounded": parallel_bounded,
            "new_source_helps": new_helps,
        }
    f = out.get("full", {})
    fo = out.get("filter_only", {})
    out["SIGNATURE"] = {
        "full_gain_ge_25pct": bool(f.get("gain_v0_to_final", 0) >= 0.25),
        "full_confidence_bounded": bool(f.get("confidence_bounded")),
        "filter_only_bounded_but_less_gain": bool(
            fo.get("confidence_bounded") and
            fo.get("gain_v0_to_final", 1) < f.get("gain_v0_to_final", 0) * 0.75),
        "provenance_free_inflates": bool(
            not out.get("no_provenance", {}).get("confidence_bounded", True)),
        "pure_redelivery_invariant_for_full": bool(f.get("pure_redelivery_invariant")),
        "parallel_refinement_bounded_for_full": bool(f.get("parallel_refinement_bounded")),
        "new_source_helps_full": bool(f.get("new_source_helps")),
    }
    out["SIGNATURE"]["PASS"] = all(out["SIGNATURE"].values())
    return out


if __name__ == "__main__":
    main()
