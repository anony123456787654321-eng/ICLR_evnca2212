"""Gate 4 -- multimodal ambiguity: the regime sectors exist for.

The Gaussian Gate 3 task has ONE unimodal posterior, so collapsing to a single
sector there is correct behaviour, not a failure -- it is the provenance and
unimodal control.  This benchmark is where sectors have work to do: several
genuinely plausible hypotheses, different groups of cells initially believing
different ones, and later independent evidence resolving it.

Because the generator knows which hypothesis each cell's evidence supports,
sector recovery is scored with chance-corrected ARI / NMI against those labels
rather than against an arbitrary divergence threshold.

Behaviours measured
  P1  clean unimodal data -> one effective sector, lower plasticity
  P2  ambiguous data -> sectors emerge that MATCH the true hypotheses (ARI/NMI)
  P3  cells switch sectors when resolving evidence arrives
  P4  REDELIVERY of an identical payload does not multiply confidence.  This
      probe cannot test useful refinement -- that is Gate 3R.
  P5  independent resolving evidence raises justified confidence and collapses
      the losing sector
  P6  duplicating one source across several sectors does not raise confidence
  P7  no_provenance fails P6; no_sectors is worse at P2 / P5
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _common import write_json, write_rows  # noqa: E402
from ecnca.neural.dataset import (BatchSpec, duplicate_batch,  # noqa: E402
                                  make_ambiguous_batch, make_batch)
from ecnca.neural.model import (coverage, gaussian_nll,  # noqa: E402
                                mixture_nll, mode_separation)
from ecnca.neural.sector_metrics import score_sectors  # noqa: E402
from ecnca.neural.train import (TrainConfig, build_model, is_finished,  # noqa: E402
                                match_hidden, pick_device, train)

VARIANTS = ["full", "no_sectors", "no_provenance", "plain"]


@torch.no_grad()
def probe_ambiguous(model, spec, cfg, device, seed, copies=1, cross=0,
                    n_batch=4, bs=12, resolve_step=6, on_train_set=False):
    """Per-step sector recovery and confidence, across the resolution event.

    ``on_train_set`` reproduces the pinned examples the positive control was
    trained on.  A capacity check that evaluates on FRESH examples measures
    generalisation from 16 memorised ones, which is not the question being
    asked -- and scores every sector metric on inputs the model has no chance on.
    """
    per_step, finals = [], []
    if on_train_set:
        # the batch size is part of the RNG stream, so it must match training or
        # the "same" seed yields entirely different examples
        bs = cfg.batch_size
    for bi in range(n_batch):
        rng = (np.random.default_rng(cfg.seed * 7919 + bi % max(cfg.fixed_examples, 1))
               if on_train_set else np.random.default_rng(70_000 + 137 * seed + bi))
        b = make_ambiguous_batch(spec, bs, n_hyp=cfg.n_hyp, roots_per_hyp=3,
                                 resolve_roots=4, resolve_step=resolve_step,
                                 copies=1, cross_sector_copies=0,
                                 rng=rng, device=device)
        # extend the SAME batch rather than regenerating, so the redelivery and
        # cross-sector probes differ from base only in re-delivery
        if copies > 1 or cross > 0:
            b = duplicate_batch(b, max(copies, cross + 1),
                                np.random.default_rng(4242 + bi), spread=bool(cross))
        out = model(b, steps=cfg.steps, instrument=True)
        ceiling = b["root_mass"].sum(-1).clamp(min=1e-6)
        mu = out["mu"].mean(1)
        # which mode is right, and how much weight does it carry?
        dist = (out["mix_mu"] - b["x_true"][:, None, None, :]).pow(2).sum(-1)
        k_star = dist.argmin(-1, keepdim=True)
        w_correct = out["mix_w"].gather(-1, k_star).squeeze(-1)
        finals.append(dict(
            mix_nll=float(mixture_nll(out["mix_mu"], out["mix_Lam"],
                                      out["mix_w"], b["x_true"]).mean()),
            mode_separation=float(mode_separation(out["mix_mu"], out["mix_w"])),
            weight_on_correct_mode=float(w_correct.mean()),
            n_live_modes=float((out["mix_w"] > 0.05).float().sum(-1).mean()),
            rmse=float(((mu - b["x_true"]) ** 2).sum(-1).sqrt().mean()),
            nll=float(gaussian_nll(out["mu"], out["Lam"], b["x_true"]).mean()),
            coverage=float(coverage(out["mu"], out["Lam"], b["x_true"])),
            claim_ratio=float((out["M"].mean(1) / ceiling).mean()),
            logdet=float(torch.linalg.slogdet(out["Lam"])[1].mean()),
        ))
        for e in (out.get("instrument") or []):
            ari, nmi = score_sectors(e["hard"], b["cell_label"])
            per_step.append(dict(step=e["step"], ari=ari, nmi=nmi,
                                 k_eff=e["k_eff"], plasticity=e["plasticity"],
                                 switch_rate=e["switch_rate"],
                                 n_sectors_occupied=e["n_sectors_occupied"],
                                 mean_entropy=e["mean_entropy"]))
    agg_final = {k: float(np.mean([f[k] for f in finals])) for k in finals[0]}
    steps = sorted({r["step"] for r in per_step})
    agg_step = [{"step": s, **{k: float(np.mean([r[k] for r in per_step if r["step"] == s]))
                               for k in per_step[0] if k != "step"}} for s in steps]
    return agg_final, agg_step


@torch.no_grad()
def probe_unimodal(model, spec, cfg, device, seed, n_batch=4, bs=12, on_train_set=False):
    rows = []
    if on_train_set:
        bs = cfg.batch_size
    for bi in range(n_batch):
        rng = (np.random.default_rng(cfg.seed * 7919 + bi % max(cfg.fixed_examples, 1))
               if on_train_set else np.random.default_rng(60_000 + 137 * seed + bi))
        b = make_batch(spec, bs, n_roots=8, copies=1, regime="clean", rng=rng, device=device)
        out = model(b, steps=cfg.steps, instrument=True)
        for e in (out.get("instrument") or []):
            rows.append({k: e[k] for k in ("step", "k_eff", "plasticity",
                                           "n_sectors_occupied", "mean_entropy")})
    steps = sorted({r["step"] for r in rows})
    return [{"step": s, **{k: float(np.mean([r[k] for r in rows if r["step"] == s]))
                           for k in rows[0] if k != "step"}} for s in steps]


def main():
    import argparse
    ap = argparse.ArgumentParser("gate4")
    ap.add_argument("--out", default="results/gate4")
    ap.add_argument("--iters", type=int, default=6000)
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--cells", type=int, default=64)
    # A torus wraps, so on a small grid every cell is 1-2 hops from the other
    # block and both blocks hold the same evidence within ~4 steps -- there is
    # then no window in which cells can hold competing views, and the sector
    # claim is untestable rather than false.  A non-wrapping grid keeps interior
    # cells informationally distinct through the ambiguous phase.
    ap.add_argument("--topology", default="grid")
    ap.add_argument("--sectors", type=int, default=4)
    ap.add_argument("--hyp", type=int, default=2)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--resolve-step", type=int, default=6)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--w-local", type=float, default=0.0,
                    help="local Bayesian consistency weight (post-freeze addition)")
    ap.add_argument("--positive-control", action="store_true",
                    help="pin a small example set and overfit it: a CAPACITY check, "
                         "not a generalisation result")
    ap.add_argument("--resume", choices=["auto", "off"], default="auto")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    # the ambiguity task uses ~10 roots and <=48 deliveries, so the default
    # slot budget would just inflate the [B,N,N,S] message tensor
    spec = BatchSpec(n_cells=args.cells, topology=args.topology,
                     max_roots=16, max_occurrences=48)
    device = pick_device(args.device)
    fixed = 16 if args.positive_control else 0
    variants = ["full"] if args.positive_control else args.variants.split(",")

    base_cfg = TrainConfig(variant="full", task="ambiguous", n_cells=args.cells,
                           topology=args.topology,
                           n_sectors=args.sectors, n_hyp=args.hyp, steps=args.steps)
    target = sum(p.numel() for p in build_model(base_cfg, spec).parameters())
    print(f"[gate4] device={device} cells={args.cells} sectors={args.sectors} "
          f"hyp={args.hyp} param-target={target} "
          f"{'POSITIVE CONTROL (overfit)' if args.positive_control else ''}")

    finals, steps_rows, uni_rows = [], [], []
    for variant in variants:
        for seed in range(args.seeds):
            cfg = TrainConfig(variant=variant, task="ambiguous", iters=args.iters,
                              batch_size=args.batch_size, steps=args.steps,
                              n_cells=args.cells, n_sectors=args.sectors, n_hyp=args.hyp,
                              topology=args.topology,
                              resolve_step=args.resolve_step, fixed_examples=fixed,
                              w_local=args.w_local,
                              seed=seed, device=device, resume=args.resume,
                              out=os.path.join(args.out, f"{variant}_s{seed}"))
            cfg.hidden = match_hidden(cfg, spec, target)
            if is_finished(cfg):
                print(f"[gate4] {variant} s{seed}: already finished, loading", flush=True)
                model = build_model(cfg, spec).to(device)
                model.load_state_dict(torch.load(os.path.join(cfg.out, "ckpt.pt"),
                                                 map_location=device,
                                                 weights_only=False)["model"])
            else:
                model, _ = train(cfg, spec, log_every=max(50, args.iters // 8))
            model.eval()
            n_par = sum(p.numel() for p in model.parameters())
            print(f"[gate4] {variant} s{seed}: {n_par} params (hidden={cfg.hidden})")

            for copies, cross, tag in ((1, 0, "base"), (4, 0, "redelivery"),
                                       (4, 3, "cross_sector_dup"), (8, 6, "cross_sector_dup8")):
                fin, per = probe_ambiguous(model, spec, cfg, device, seed, copies=copies,
                                           cross=cross, resolve_step=args.resolve_step,
                                           on_train_set=bool(fixed))
                finals.append(dict(variant=variant, seed=seed, n_params=n_par, probe=tag,
                                   copies=copies, cross=cross, **fin))
                if cfg.variant in ("full", "no_provenance"):
                    for r in per:
                        steps_rows.append(dict(variant=variant, seed=seed, probe=tag, **r))
            if cfg.variant in ("full", "no_provenance"):
                for r in probe_unimodal(model, spec, cfg, device, seed,
                                        on_train_set=bool(fixed)):
                    uni_rows.append(dict(variant=variant, seed=seed, **r))

    write_rows(os.path.join(args.out, "gate4_final.csv"), finals)
    write_rows(os.path.join(args.out, "gate4_per_step.csv"), steps_rows)
    write_rows(os.path.join(args.out, "gate4_unimodal.csv"), uni_rows)
    write_json(os.path.join(args.out, "gate4_report.json"),
               _report(finals, steps_rows, uni_rows, args.resolve_step))
    print(f"[gate4] wrote {args.out}")


def _report(finals, steps_rows, uni_rows, resolve_step):
    out = {}
    for variant in sorted({r["variant"] for r in finals}):
        def fin(tag, key):
            v = [r[key] for r in finals if r["variant"] == variant and r["probe"] == tag]
            return float(np.mean(v)) if v else float("nan")

        def step_curve(tag, key):
            sub = [r for r in steps_rows if r["variant"] == variant and r["probe"] == tag]
            xs = sorted({r["step"] for r in sub})
            return xs, [float(np.mean([r[key] for r in sub if r["step"] == x])) for x in xs]

        st, ari = step_curve("base", "ari")
        _, nmi = step_curve("base", "nmi")
        _, keff = step_curve("base", "k_eff")
        _, switch = step_curve("base", "switch_rate")
        pre = [i for i, s in enumerate(st) if 0 < s < resolve_step]
        post = [i for i, s in enumerate(st) if s >= resolve_step]
        uni = [r for r in uni_rows if r["variant"] == variant]
        uni_keff = float(np.mean([r["k_eff"] for r in uni[-4:]])) if uni else float("nan")
        uni_plast = float(np.mean([r["plasticity"] for r in uni[-4:]])) if uni else float("nan")

        ari_pre = float(np.max([ari[i] for i in pre])) if pre and ari else float("nan")
        keff_pre = float(np.mean([keff[i] for i in pre])) if pre and keff else float("nan")
        keff_post = float(np.mean([keff[i] for i in post])) if post and keff else float("nan")
        switch_at = float(np.max([switch[i] for i in post[:2]])) if post and switch else float("nan")

        c0, c4, cx, cx8 = (fin("base", "claim_ratio"), fin("redelivery", "claim_ratio"),
                           fin("cross_sector_dup", "claim_ratio"),
                           fin("cross_sector_dup8", "claim_ratio"))
        r0, r4 = fin("base", "rmse"), fin("redelivery", "rmse")

        # --- claim labels, tightened -------------------------------------
        # P4 asks whether refinement is USEFUL, not merely non-inflating: RMSE
        # must actually improve.  P3 is meaningless unless the sectors being
        # switched between correspond to hypotheses, so it is conditioned on P2.
        # P6 is a STRUCTURAL guarantee for provenance variants (duplicates
        # cannot add roots), so it is only evidence in contrast to no_provenance.
        p2_ok = bool(ari_pre > 0.2)
        out[variant] = {
            "P1_unimodal_collapses_to_one_sector": {
                "k_eff": uni_keff, "plasticity": uni_plast,
                "pass": bool(np.isfinite(uni_keff) and uni_keff < keff_pre)},
            "P2_sectors_match_hypotheses": {
                "ari_pre_resolution": ari_pre,
                "nmi_pre_resolution": float(np.max([nmi[i] for i in pre])) if pre and nmi else float("nan"),
                "pass": bool(ari_pre > 0.2)},
            "P3_meaningful_switching": {
                "switch_rate_at_resolution": switch_at,
                "requires_P2": p2_ok,
                "pass": bool(switch_at > 0.05 and p2_ok),
                "note": ("switching among near-uniform sector probabilities is not "
                         "hypothesis switching; tentative until ARI clears 0.2")},
            "P4_redelivery_bounded": {
                "rmse_1x": r0, "rmse_redelivered": r4, "claim_1x": c0, "claim_redelivered": c4,
                "pass": bool(c4 <= c0 * 1.10),
                "note": ("REDELIVERY only: duplicate_batch copies payloads byte-for-byte "
                         "and never touches root_feat, so this probe cannot test useful "
                         "refinement.  Learned refinement is Gate 3R.")},
            "P5_resolution_lifts_confidence_and_collapses_loser": {
                "k_eff_pre": keff_pre, "k_eff_post": keff_post,
                "pass": bool(np.isfinite(keff_post) and keff_post < keff_pre)},
            "P6_cross_sector_duplication_adds_nothing": {
                "claim_base": c0, "claim_cross3": cx, "claim_cross6": cx8,
                "kind": ("structural for provenance variants (duplicates cannot add "
                         "roots); evidential only in contrast to no_provenance"),
                "pass": bool(cx8 <= c0 * 1.10)},
            "final_rmse": r0, "final_nll": fin("base", "nll"),
            "mixture": {
                "mix_nll": fin("base", "mix_nll"),
                "mode_separation": fin("base", "mode_separation"),
                "weight_on_correct_mode": fin("base", "weight_on_correct_mode"),
                "n_live_modes": fin("base", "n_live_modes")},
        }
    return out


if __name__ == "__main__":
    main()
