"""Training loop for the Sectorized EC-NCA.

Objectives
  mean       MSE of each cell's posterior mean against the centralised
             unique-evidence posterior mean.  Without this the run collapses:
             the gradient on mu is scaled by Lambda, so a humble model gets a
             vanishing signal on its mean, never improves it, and therefore
             never finds it profitable to claim any confidence -- it settles on
             the prior.  Supervising the mean directly breaks that deadlock
             while leaving confidence to be EARNED through the NLL.
  nll        Gaussian NLL of the latent under each cell's reported belief.
  consensus  cells must agree -- an NCA whose cells disagree has no answer.
  invariance the belief on E and on E + duplicates(E) must match.  This is a
             *loss*, not the mechanism: the mechanism is the lineage join, and
             this term only measures whether the learned parts respect it.

Device is auto-detected (cuda -> mps -> cpu), DDP is available from the same
config, and checkpoints are atomic so a run can resume.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F_

from .dataset import (BatchSpec, duplicate_batch, make_ambiguous_batch,
                      make_batch, oracle_posterior)
from .model import SectorizedECNCA, coverage, gaussian_nll, mixture_nll


def pick_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class TrainConfig:
    variant: str = "full"          # full | no_sectors | no_provenance | plain
    hidden: int = 64
    n_sectors: int = 6
    steps: int = 12
    eval_steps: int = 12
    iters: int = 3000
    batch_size: int = 24
    lr: float = 3e-4
    max_roots_train: int = 16
    max_copies_train: int = 4
    w_consensus: float = 0.1
    w_invariance: float = 1.0
    w_mean: float = 5.0
    w_local: float = 0.0        # off by default; see RESULTS.md (post-freeze addition)
    # sigma=0.3 was chosen for the ANALYTIC refinement experiment, where a sharp
    # target is what makes partial extraction visible.  For the learned model it
    # puts the oracle posterior std at ~0.04, far below what 12 NCA steps can
    # resolve, so no claim the model makes can ever be justified and rho fights
    # mu instead of calibrating against it.
    noise_sigma: float = 1.0
    seed: int = 0
    device: str = "auto"
    resume: str = "auto"
    dry_run: bool = False
    n_cells: int = 64
    topology: str = "torus"
    task: str = "unimodal"          # unimodal | ambiguous
    n_hyp: int = 2
    resolve_step: int = 6
    fixed_examples: int = 0         # >0 pins a small set, for the positive control
    out: str = "runs/secnca"


VARIANTS = {
    "full":           dict(use_sectors=True,  use_provenance=True),
    "no_sectors":     dict(use_sectors=False, use_provenance=True),
    "no_provenance":  dict(use_sectors=True,  use_provenance=False),
    "plain":          dict(use_sectors=False, use_provenance=False),
}


def build_model(cfg: TrainConfig, spec: BatchSpec) -> SectorizedECNCA:
    return SectorizedECNCA(dim=spec.dim, obs_dim=spec.obs_dim, hidden=cfg.hidden,
                           n_sectors=cfg.n_sectors, **VARIANTS[cfg.variant])


def match_hidden(cfg: TrainConfig, spec: BatchSpec, target: int, lo: int = 32,
                 hi: int = 256) -> int:
    """Smallest hidden width whose parameter count reaches ``target``.

    Ablations lose whole modules, so comparing them at equal ``hidden`` would
    compare different capacities.  This matches on parameters instead.
    """
    best = lo
    for h in range(lo, hi + 1, 2):
        trial = TrainConfig(**{**asdict(cfg), "hidden": h})
        n = sum(p.numel() for p in build_model(trial, spec).parameters())
        best = h
        if n >= target:
            break
    return best


def make_train_pair(cfg, spec, rng, device, local_bs, n_roots, copies, it):
    """Base batch plus a duplicated twin built from it, so the invariance term
    isolates duplication: same observations, same latent, same oracle."""
    if cfg.fixed_examples:
        # positive control: pin the example set so the model can overfit it
        rng = np.random.default_rng(cfg.seed * 7919 + (it % cfg.fixed_examples))
    if cfg.task == "ambiguous":
        base = make_ambiguous_batch(spec, local_bs, n_hyp=cfg.n_hyp, roots_per_hyp=3,
                                    resolve_roots=4, resolve_step=cfg.resolve_step,
                                    copies=1, cross_sector_copies=0, rng=rng, device=device)
    else:
        base_regime = "conflict" if rng.random() < 0.4 else "clean"
        base = make_batch(spec, local_bs, n_roots, copies=1, regime=base_regime,
                          rng=rng, device=device)
    dup = duplicate_batch(base, copies, rng, spread=bool(rng.random() < 0.5))
    return base, dup


def local_posterior(batch, W):
    """Posterior implied by the evidence a cell ACTUALLY holds. [B, N, d]

    Not a sector label and not the answer: it is what a Bayesian cell should
    believe given its own observations, before anything contradicts them.  Under
    ambiguity that pulls cells holding different evidence to different beliefs,
    which is the divergence a sector layer exists to organise.  Without it the
    only signal is the final answer, so cells never diverge and sectors have
    nothing to represent.
    """
    d = batch["x_true"].shape[-1]
    eye = torch.eye(d, device=W.device)
    Lam = batch["prior_precision"] * eye + torch.einsum("bnr,brij->bnij", W, batch["root_Lam"])
    h = torch.einsum("bnr,brd->bnd", W, batch["root_h"])
    return torch.linalg.solve(Lam, h.unsqueeze(-1)).squeeze(-1)


def losses(model, batch, dup_batch, cfg) -> Dict[str, torch.Tensor]:
    want_trace = cfg.task == "ambiguous" and cfg.w_local > 0
    out = model(batch, steps=cfg.steps, trace=want_trace)
    if cfg.task == "ambiguous":
        # a single Gaussian must sit between the hypotheses and is penalised at
        # whichever is true; a mixture that keeps them apart is not
        nll = mixture_nll(out["mix_mu"], out["mix_Lam"], out["mix_w"],
                          batch["x_true"]).mean()
    else:
        nll = gaussian_nll(out["mu"], out["Lam"], batch["x_true"]).mean()
    cov = coverage(out["mu"], out["Lam"], batch["x_true"])
    if cfg.task == "ambiguous":
        # Supervise only the FINAL answer.  Pointing every cell at the truth
        # from the start would erase the ambiguity the sectors exist to carry,
        # so nothing about the sector structure is supervised: it has to earn
        # its place by making the final answer better.
        target = batch["x_true"].unsqueeze(1).expand_as(out["mu"])
    else:
        target = oracle_posterior(batch)[0].unsqueeze(1).expand_as(out["mu"])
    mean_loss = F_.mse_loss(out["mu"], target)
    local = torch.zeros((), device=out["mu"].device)
    if want_trace and "root_h" in batch:
        pre = [e for e in out["trace"] if 0 < e["step"] < cfg.resolve_step]
        for e in pre:
            local = local + F_.mse_loss(e["mu"], local_posterior(batch, e["W"]).detach())
        local = local / max(len(pre), 1)
    consensus = out["mu"].var(dim=1).mean()

    dup = model(dup_batch, steps=cfg.steps)
    # belief must not move when the SAME roots are delivered more times
    inv = (F_.mse_loss(dup["mu"].mean(1), out["mu"].mean(1).detach())
           + F_.mse_loss(dup["Lam"].mean(1), out["Lam"].mean(1).detach()))
    total = (nll + cfg.w_mean * mean_loss + cfg.w_local * local
             + cfg.w_consensus * consensus + cfg.w_invariance * inv)
    return {"total": total, "nll": nll.detach(), "mean": mean_loss.detach(),
            "consensus": consensus.detach(), "cov": cov.detach(),
            "local": local.detach(),
            "invariance": inv.detach(), "mass": out["M"].mean().detach(),
            "rel_mass": (out["M"].mean()
                         / batch["root_mass"].sum(-1).mean().clamp(min=1e-6)).detach()}


def is_finished(cfg: TrainConfig) -> bool:
    """A run counts as finished when its checkpoint records the full iteration."""
    path = os.path.join(cfg.out, "ckpt.pt")
    if not os.path.exists(path):
        return False
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        return int(ck.get("iter", 0)) >= cfg.iters
    except Exception:
        return False


def maybe_resume(model, opt, cfg):
    """Return the iteration to resume from (0 if starting fresh)."""
    path = os.path.join(cfg.out, "ckpt.pt")
    if cfg.resume != "auto" or not os.path.exists(path):
        return 0, []
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
        print(f"[{cfg.variant}] resumed at iter {ck.get('iter', 0)}", flush=True)
        return int(ck.get("iter", 0)), ck.get("history", [])
    except Exception as exc:
        print(f"[{cfg.variant}] could not resume ({exc}); starting fresh", flush=True)
        return 0, []


def setup_ddp():
    """Initialise DDP when launched under torchrun; otherwise single process."""
    ws = int(os.environ.get("WORLD_SIZE", "1"))
    if ws <= 1:
        return 0, 1, False
    import torch.distributed as dist
    rank = int(os.environ["RANK"])
    local = int(os.environ.get("LOCAL_RANK", rank))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
    return rank, ws, True


def train(cfg: TrainConfig, spec: Optional[BatchSpec] = None, log_every: int = 100):
    spec = spec or BatchSpec()
    spec = BatchSpec(**{**spec.__dict__, "noise_sigma": cfg.noise_sigma,
                        "n_cells": cfg.n_cells, "topology": cfg.topology})
    if cfg.task == "ambiguous":
        spec = BatchSpec(**{**spec.__dict__, "max_roots": 16, "max_occurrences": 48})
    device = pick_device(cfg.device)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    rank, world, ddp = setup_ddp()
    if ddp:
        device = f"cuda:{int(os.environ.get('LOCAL_RANK', rank))}"
    # global batch is held FIXED as GPUs are added, rather than silently
    # multiplied by the world size
    local_bs = max(1, cfg.batch_size // world)
    torch.manual_seed(cfg.seed * 1000 + rank)      # deterministic per process
    model = build_model(cfg, spec).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    start_it, history = maybe_resume(model, opt, cfg)
    if ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[int(os.environ.get("LOCAL_RANK", rank))]
                    if torch.cuda.is_available() else None)
    if cfg.dry_run:
        cfg = TrainConfig(**{**asdict(cfg), "iters": 2, "dry_run": True})
    warm = max(1, int(0.05 * cfg.iters))
    _ = local_bs
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt,
        [torch.optim.lr_scheduler.LinearLR(opt, 0.05, 1.0, total_iters=warm),
         torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, cfg.iters - warm))],
        milestones=[warm])
    rng = np.random.default_rng(cfg.seed)
    os.makedirs(cfg.out, exist_ok=True)

    t0 = time.time()
    for it in range(start_it + 1, cfg.iters + 1):
        n_roots = int(rng.integers(1, cfg.max_roots_train + 1))
        copies = int(rng.integers(2, cfg.max_copies_train + 1))
        # a quarter of batches carry a competing hypothesis; with a single
        # hypothesis every cell converges to the same state and the sector
        # layer has nothing to compare, so it would never learn to.
        # a share of batches carry a competing hypothesis; with a single
        # hypothesis every cell converges to the same state and the sector layer
        # has nothing to compare, so it would never learn to.
        base, dup = make_train_pair(cfg, spec, rng, device, local_bs, n_roots, copies, it)

        out = losses(model, base, dup, cfg)
        opt.zero_grad(set_to_none=True)
        out["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if it % log_every == 0 or it == 1:
            rec = {k: float(v.detach() if torch.is_tensor(v) else v) for k, v in out.items()}
            rec.update(iter=it, secs=round(time.time() - t0, 1))
            rec["rank"] = rank
            history.append(rec)
            with open(os.path.join(cfg.out, "metrics.jsonl"), "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(f"[{cfg.variant}] it {it:5d}  nll {rec['nll']:9.3f}  "
                  f"mu {rec['mean']:7.4f}  "
                  f"inv {rec['invariance']:8.4f}  cons {rec['consensus']:7.4f}  "
                  f"loc {rec['local']:6.3f}  claim {rec['rel_mass']:5.2f}  "
                  f"({rec['secs']:.0f}s)", flush=True)
            if rank == 0:
                save(model, cfg, history, it)
    if rank == 0:
        save(model, cfg, history, cfg.iters)
    return (model.module if hasattr(model, "module") else model), history


def save(model, cfg: TrainConfig, history, it: int = 0):
    """Atomic: write to a temp path and rename, so a kill cannot corrupt it."""
    core = model.module if hasattr(model, "module") else model
    tmp = os.path.join(cfg.out, "ckpt.pt.tmp")
    torch.save({"model": core.state_dict(), "config": asdict(cfg),
                "iter": it, "history": history}, tmp)
    os.replace(tmp, os.path.join(cfg.out, "ckpt.pt"))
    with open(os.path.join(cfg.out, "history.json"), "w") as fh:
        json.dump(history, fh, indent=2)
