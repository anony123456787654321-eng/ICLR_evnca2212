"""Seven-arm paired intervention matrix for message-level causal attribution.

The 63.8-point collapse measured on the frozen reference (0.9520 fresh ->
0.3141 with half the rollout receiving one cached message) is **not yet
attributable to duplication**. Three explanations are confounded in that
single arm:

  (a) repeated stale content is actively harmful;
  (b) fresh information is absent (starvation);
  (c) local dynamics keep running on stale input.

Separating them needs arms that vary one factor at a time from an identical
starting state, example, seed, fire mask and noise schedule.

The seven arms, all sharing that starting point:

  fresh                   recompute and deliver the current message
  cached_duplicate        re-deliver the exact message cached at the boundary
  withhold                deliver no incoming neighbourhood information, while
                          the explicitly defined local dynamics continue
  state_hold              perform no state update during the interval
  exact_duplicate_filter  detect the byte-identical repeat and damp it.
                          **This is a DAMPED-MESSAGE BASELINE, not a
                          separation of content from evidential credit.** It
                          scales the whole message by ``credit_scale``, so
                          content and credit fall together. True separation
                          needs the distinct content and origin channels of
                          Target 2, which do not exist yet. Do not report this
                          arm as duplicate-credit rejection.
  transformed_same_origin numerically different messages derived from the same
                          originating observation by legitimate propagation.
                          A constructed smoothing: it stress-tests whether an
                          exact-tensor filter can be evaded, and does NOT show
                          that the reference recognises common origin.
  new_origin              an equally sized message carrying genuinely new
                          evidence about the SAME image and class (a
                          previously occluded region restored). Injecting a
                          different image's message would measure "wrong label
                          injected" instead.

Primary causal contrasts:

  cached_duplicate vs withhold          -> is stale content worse than nothing?
  cached_duplicate vs state_hold        -> do continuing dynamics cause it?
  cached_duplicate vs exact_dup_filter  -> does rejecting the repeat recover it?
  transformed_same_origin vs filter     -> does an exact filter miss transforms?
  transformed_same_origin vs new_origin -> is same-origin distinguishable?

**What "withhold" means, defined explicitly.** A cell receives a zero message
vector, so it gets no neighbourhood information, but its own update MLP still
runs and its own noise and fire draw still apply. It is starvation of incoming
evidence, not suspension of the cell. ``state_hold`` is the other case: the
state is not written at all.

**Unavoidable differences, documented.** ``state_hold`` performs fewer state
writes than the other arms by construction -- that is the intervention. Its
message-recomputation count is reported separately so the comparison is not
read as compute-matched. Every other arm performs exactly ``steps`` updates
with the same fire/noise schedule.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

ARMS = (
    "fresh",
    "cached_duplicate",
    "withhold",
    "state_hold",
    "exact_duplicate_filter",
    "transformed_same_origin",
    "new_origin",
)

PRIMARY_CONTRASTS = (
    ("cached_duplicate", "withhold"),
    ("cached_duplicate", "state_hold"),
    ("cached_duplicate", "exact_duplicate_filter"),
    ("transformed_same_origin", "exact_duplicate_filter"),
    ("transformed_same_origin", "new_origin"),
)


@dataclass
class ArmResult:
    arm: str
    steps: int
    boundary: int
    window: int
    state_writes: int = 0
    messages_recomputed: int = 0
    messages_delivered: int = 0
    duplicates_rejected: int = 0
    trajectory: list[dict] = field(default_factory=list)
    final_state: torch.Tensor | None = None

    def summary(self) -> dict:
        return {
            "arm": self.arm,
            "steps": self.steps,
            "boundary": self.boundary,
            "window": self.window,
            "state_writes": self.state_writes,
            "messages_recomputed": self.messages_recomputed,
            "messages_delivered": self.messages_delivered,
            "duplicates_rejected": self.duplicates_rejected,
            "final": self.trajectory[-1] if self.trajectory else None,
        }


def transform_message(m: torch.Tensor, *, strength: float = 0.15,
                      generator: torch.Generator | None = None) -> torch.Tensor:
    """A legitimate propagation/transformation of the SAME origin.

    Numerically different from ``m`` -- so an exact-tensor filter cannot catch
    it -- while carrying no new observation: the transform is a fixed local
    smoothing plus a small scaling, both functions of ``m`` alone. Nothing
    outside ``m`` enters, which is what makes "same origin" true by
    construction rather than by assertion.
    """
    k = torch.tensor([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]],
                     device=m.device, dtype=m.dtype) / 16.0
    k = k.expand(m.shape[1], 1, 3, 3)
    smoothed = torch.nn.functional.conv2d(m, k, padding=1, groups=m.shape[1])
    return (1.0 - strength) * m + strength * smoothed


@torch.no_grad()
def run_arm(
    model,
    initial_state: torch.Tensor,
    labels: torch.Tensor,
    *,
    arm: str,
    steps: int,
    boundary: int,
    window: int,
    fire: torch.Tensor,
    noise: torch.Tensor | None,
    metric_fn,
    new_origin_state: torch.Tensor | None = None,
    activity: torch.Tensor | None = None,
    credit_scale: float = 0.0,
) -> ArmResult:
    """Run one arm from a shared initial state with a shared schedule.

    ``fire`` is (steps,B,1,H,W); ``noise`` is (steps,B,C,H,W) or None. Both are
    supplied by the caller so every arm receives identical stochastic inputs.
    ``metric_fn(model, state) -> dict`` computes the per-step readout.

    ``credit_scale`` is how much evidential weight ``exact_duplicate_filter``
    grants a detected repeat. 0.0 suppresses it entirely -- which makes the arm
    identical to ``withhold`` and is therefore a degenerate filter, kept only
    as a reference point. A value in (0, 1) implements the intended semantics:
    repeated content may still refine, but carries reduced fresh credit.
    """
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    if boundary + window > steps:
        raise ValueError(
            f"boundary({boundary}) + window({window}) exceeds steps({steps})"
        )
    if arm == "new_origin" and new_origin_state is None:
        raise ValueError("new_origin requires new_origin_state")

    x = initial_state.clone()
    res = ArmResult(arm=arm, steps=steps, boundary=boundary, window=window)
    cached: torch.Tensor | None = None
    zero_msg: torch.Tensor | None = None

    for t in range(steps):
        f = fire[t]
        nz = None if noise is None else noise[t]
        inside = boundary <= t < boundary + window

        if not inside:
            m = model.message(x)
            res.messages_recomputed += 1
            res.messages_delivered += 1
            x = model.apply_message(x, m, fire=f, noise=nz)
            res.state_writes += 1
        else:
            if arm == "fresh":
                m = model.message(x)
                res.messages_recomputed += 1
                res.messages_delivered += 1
                x = model.apply_message(x, m, fire=f, noise=nz)
                res.state_writes += 1

            elif arm == "cached_duplicate":
                if cached is None:
                    cached = model.message(x)
                    res.messages_recomputed += 1
                res.messages_delivered += 1
                x = model.apply_message(x, cached, fire=f, noise=nz)
                res.state_writes += 1

            elif arm == "withhold":
                # No incoming neighbourhood information; the cell's own
                # dynamics (MLP, noise, fire) still run. The zero message is
                # allocated from a cached shape rather than by calling
                # model.message, so the arm does not secretly recompute.
                if zero_msg is None:
                    zero_msg = torch.zeros_like(model.message(x))
                    res.messages_recomputed += 1  # the one shape probe
                x = model.apply_message(x, zero_msg, fire=f, noise=nz)
                res.state_writes += 1

            elif arm == "state_hold":
                # No state update at all. Fewer writes than every other arm by
                # construction; reported, not hidden.
                pass

            elif arm == "exact_duplicate_filter":
                if cached is None:
                    cached = model.message(x)
                    res.messages_recomputed += 1
                    res.messages_delivered += 1
                    x = model.apply_message(x, cached, fire=f, noise=nz)
                    res.state_writes += 1
                else:
                    # The message is byte-identical to one already credited.
                    #
                    # A filter that answers this by delivering a ZERO message
                    # is not a duplicate filter -- it is starvation, and it
                    # collapses onto `withhold` by construction (verified: both
                    # reached 0.0500 in the smoke run). The distinction the
                    # method rests on is that repeated information may still
                    # refine content but must not receive fresh evidential
                    # credit. So the content is delivered and the credit is
                    # withheld, implemented as a damped contribution: the
                    # message still informs the update, scaled by
                    # `credit_scale`, instead of being suppressed outright.
                    res.duplicates_rejected += 1
                    x = model.apply_message(
                        x, cached * credit_scale, fire=f, noise=nz
                    )
                    res.state_writes += 1
                    res.messages_delivered += 1

            elif arm == "transformed_same_origin":
                if cached is None:
                    cached = model.message(x)
                    res.messages_recomputed += 1
                # Numerically different each step, same origin throughout.
                cached = transform_message(cached)
                res.messages_delivered += 1
                x = model.apply_message(x, cached, fire=f, noise=nz)
                res.state_writes += 1

            elif arm == "new_origin":
                # Genuinely new evidence must enter the RECEIVING state, not
                # merely be perceived elsewhere. Delivering a message computed
                # from `new_origin_state` while `x` keeps its occluded input
                # leaves the new evidence unable to take hold: the cells the
                # message describes are still dead in `x`, so their residual
                # is masked away. On the first in-window step we therefore
                # adopt the additional observation into `x` itself, then
                # continue with ordinary fresh recomputation -- which is what
                # "new information arrived" means.
                if cached is None:
                    cached = torch.zeros(())      # marks "already adopted"
                    x = torch.cat(
                        [new_origin_state[:, :1], x[:, 1:]], dim=1
                    )
                m = model.message(x)
                res.messages_recomputed += 1
                res.messages_delivered += 1
                x = model.apply_message(x, m, fire=f, noise=nz)
                res.state_writes += 1

        res.trajectory.append(metric_fn(model, x))

    res.final_state = x
    return res


def contrast_table(results: dict[str, ArmResult], key: str = "accuracy") -> dict:
    """Primary causal contrasts with an explicit interpretation for each."""
    def val(arm: str):
        r = results.get(arm)
        if r is None or not r.trajectory:
            return None
        return r.trajectory[-1].get(key)

    interp = {
        ("cached_duplicate", "withhold"): (
            "If cached_duplicate is WORSE than withhold, repeated stale "
            "content is actively harmful beyond mere absence of fresh "
            "information. If they are equal, the loss is starvation, not "
            "duplication."
        ),
        ("cached_duplicate", "state_hold"): (
            "If state_hold is much better, the damage comes from continuing to "
            "integrate stale input rather than from the missing information."
        ),
        ("cached_duplicate", "exact_duplicate_filter"): (
            "How much of the loss a byte-identical filter recovers. Note a "
            "filter with credit_scale=0 collapses onto `withhold`: suppressing "
            "a message entirely IS starvation. Report the credit_scale used."
        ),
        ("transformed_same_origin", "exact_duplicate_filter"): (
            "An exact filter cannot see a transformed repeat. A gap here sizes "
            "the problem that a learned origin signature must solve."
        ),
        ("transformed_same_origin", "new_origin"): (
            "If these differ, same-origin and new-origin messages are "
            "behaviourally distinguishable, so origin-aware credit has "
            "something to exploit."
        ),
    }
    out = {}
    for a, b in PRIMARY_CONTRASTS:
        va, vb = val(a), val(b)
        out[f"{a}__vs__{b}"] = {
            a: va, b: vb,
            "delta": None if va is None or vb is None else va - vb,
            "interpretation": interp[(a, b)],
        }
    return out
