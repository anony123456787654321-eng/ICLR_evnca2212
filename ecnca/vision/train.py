"""Reference training loop: the stabilized recipe from the pinned notebook.

Recipe, all from the source:
  * sample pool of 160 (10 x batch 16), persistent across iterations
  * each iteration: replace the first quarter with fresh initial states, and
    (MUTATE_POOL) replace the last quarter with *mutated* states that keep the
    old cell state under the new digit's mask
  * 20 CA steps per iteration, gradient through all 20
  * L2 loss to one-hot targets, summed over cells, halved, batch-averaged
  * residual noise N(0, 0.02) during training
  * per-tensor gradient normalization g <- g/(||g||+1e-8)
  * Adam, piecewise-constant LR 1e-3 -> 1e-4 @30k -> 1e-5 @70k
  * 100,000 iterations
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from . import reference as ref
from .reference import ReferenceCA, ReferenceConfig


@dataclass
class TrainConfig:
    iterations: int = ref.TOTAL_ITERATIONS
    batch_size: int = ref.BATCH_SIZE
    pool_size: int = ref.POOL_SIZE
    steps_per_iter: int = ref.TRAIN_STEPS_PER_ITER
    use_pattern_pool: bool = True
    mutate_pool: bool = True
    loss_type: str = "l2"          # "l2" = stabilized, "ce" = the unstable one
    add_noise: bool = True
    seed: int = 0
    log_every: int = 100
    eval_every: int = 5_000
    ckpt_every: int = 5_000
    device: str = "auto"

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        return d


def resolve_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class SamplePool:
    """Persistent pool of CA states, matching the source's SamplePool usage."""

    def __init__(self, states: torch.Tensor, labels: torch.Tensor):
        self.states = states
        self.labels = labels

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.integers(0, self.states.shape[0], size=n)

    def commit(self, idx: np.ndarray, states: torch.Tensor, labels: torch.Tensor) -> None:
        self.states[idx] = states.detach()
        self.labels[idx] = labels


@dataclass
class TrainState:
    iteration: int = 0
    loss_log: list[float] = field(default_factory=list)
    eval_log: list[dict] = field(default_factory=list)


def train(
    images: np.ndarray,
    labels: np.ndarray,
    config: TrainConfig,
    *,
    out_dir: Path,
    resume: bool = True,
    dev: tuple[np.ndarray, np.ndarray] | None = None,
    eval_images: int = 500,
    on_log=None,
    max_seconds: float | None = None,
) -> dict:
    """Train the reference. Returns a summary dict; writes checkpoints."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(config.device)

    torch.manual_seed(config.seed)
    model = ReferenceCA(ReferenceConfig(add_noise=config.add_noise)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=ref.LEARNING_RATE)
    state = TrainState()
    rng = np.random.default_rng(config.seed)

    imgs_t = torch.from_numpy(images)
    labels_t = torch.from_numpy(labels)

    ckpt_path = out_dir / "reference_last.pt"
    if resume and ckpt_path.exists():
        blob = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(blob["model"])
        opt.load_state_dict(blob["optimizer"])
        state.iteration = blob["iteration"]
        state.loss_log = blob["loss_log"]
        state.eval_log = blob.get("eval_log", [])
        rng = np.random.default_rng()
        rng.bit_generator.state = blob["numpy_rng"]
        torch.set_rng_state(torch.tensor(blob["torch_rng"], dtype=torch.uint8))
        pool = SamplePool(
            blob["pool_states"].to(device), blob["pool_labels"].to(device)
        )
    else:
        start = rng.integers(0, images.shape[0], size=config.pool_size)
        pool = SamplePool(
            model.initialize(imgs_t[start].to(device)).detach(),
            labels_t[start].to(device),
        )

    loss_fn = ref.batch_l2_loss if config.loss_type == "l2" else ref.batch_ce_loss
    q = config.batch_size // 4
    t0 = time.perf_counter()
    timings: list[float] = []

    while state.iteration < config.iterations:
        it_t0 = time.perf_counter()
        for gp in opt.param_groups:
            gp["lr"] = ref.lr_at(state.iteration)

        if config.use_pattern_pool:
            idx = pool.sample(config.batch_size, rng)
            x0 = pool.states[idx].clone()
            y0 = pool.labels[idx].clone()

            fresh = rng.integers(0, images.shape[0], size=q)
            x0[:q] = model.initialize(imgs_t[fresh].to(device))
            y0[:q] = labels_t[fresh].to(device)

            other = rng.integers(0, images.shape[0], size=q)
            new_x = imgs_t[other].to(device)
            new_y = labels_t[other].to(device)
            if config.mutate_pool:
                x0[-q:] = ref.mutate(x0[-q:], new_x)
            else:
                x0[-q:] = model.initialize(new_x)
            y0[-q:] = new_y
        else:
            idx = None
            b = rng.integers(0, images.shape[0], size=config.batch_size)
            x0 = model.initialize(imgs_t[b].to(device))
            y0 = labels_t[b].to(device)

        model.train()
        x = x0
        for _ in range(config.steps_per_iter):
            x = model(x)
        loss = loss_fn(model, x, y0)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        ref.normalize_gradients_(model)
        opt.step()

        if config.use_pattern_pool and idx is not None:
            pool.commit(idx, x, y0)

        state.iteration += 1
        state.loss_log.append(float(loss.detach()))
        timings.append(time.perf_counter() - it_t0)

        if state.iteration % config.log_every == 0:
            recent = float(np.mean(state.loss_log[-config.log_every:]))
            msg = {
                "iteration": state.iteration,
                "loss": recent,
                "lr": ref.lr_at(state.iteration),
                "sec_per_iter": float(np.mean(timings[-config.log_every:])),
                "elapsed_s": time.perf_counter() - t0,
            }
            if on_log:
                on_log(msg)

        if dev is not None and state.iteration % config.eval_every == 0:
            from .evaluate import rollout
            di, dl = dev
            r = rollout(
                model,
                torch.from_numpy(di[:eval_images]),
                torch.from_numpy(dl[:eval_images]),
                steps=ref.EVAL_STEPS,
                generator=None,
            )
            entry = {
                "iteration": state.iteration,
                "dev_cell_accuracy_step200": r.pre[-1]["cell_accuracy"],
                "dev_digit_accuracy_step200": r.pre[-1]["digit_accuracy"],
                "dev_total_agreement_step200": r.pre[-1]["total_agreement"],
                "n_images": int(min(eval_images, len(dl))),
            }
            state.eval_log.append(entry)
            if on_log:
                on_log(entry)

        if state.iteration % config.ckpt_every == 0 or state.iteration == config.iterations:
            _save(ckpt_path, model, opt, state, pool, rng, config)

        if max_seconds is not None and (time.perf_counter() - t0) >= max_seconds:
            _save(ckpt_path, model, opt, state, pool, rng, config)
            break

    _save(ckpt_path, model, opt, state, pool, rng, config)
    return {
        "iterations_completed": state.iteration,
        "final_loss_mean_last_100": float(np.mean(state.loss_log[-100:])) if state.loss_log else None,
        "sec_per_iter_median": float(np.median(timings)) if timings else None,
        "elapsed_s": time.perf_counter() - t0,
        "device": str(device),
        "eval_log": state.eval_log,
        "checkpoint": str(ckpt_path),
    }


def _save(path: Path, model, opt, state: TrainState, pool: SamplePool, rng, config: TrainConfig) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "iteration": state.iteration,
            "loss_log": state.loss_log,
            "eval_log": state.eval_log,
            "pool_states": pool.states.cpu(),
            "pool_labels": pool.labels.cpu(),
            "numpy_rng": rng.bit_generator.state,
            "torch_rng": torch.get_rng_state().tolist(),
            "config": config.to_dict(),
            "reference_config": model.config.to_dict(),
        },
        tmp,
    )
    tmp.replace(path)


def load_checkpoint(path: Path, device: str = "auto") -> tuple[ReferenceCA, dict]:
    dev = resolve_device(device)
    blob = torch.load(Path(path), map_location=dev, weights_only=False)
    cfg = ReferenceConfig(**blob["reference_config"])
    model = ReferenceCA(cfg).to(dev)
    model.load_state_dict(blob["model"])
    model.eval()
    return model, blob
