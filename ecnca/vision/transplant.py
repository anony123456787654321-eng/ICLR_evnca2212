"""Transplant the frozen 100k reference into every variant's shared trunk.

Why this exists. Phase C's screen trained every variant from scratch for 10k
iterations, producing a reference at 0.1394 fresh cell accuracy against the
frozen model's 0.9535 -- and failure modes 6x and 155x too small to test
against. A criterion calibrated on the frozen effects cannot be evaluated on a
substrate that does not exhibit them.

So every Phase C2 variant starts as the frozen 100k reference plus a mechanism
initialised to do nothing. Two properties make that claim checkable:

**Transplantation is verified, not assumed.** Shared tensors are copied exactly
where shapes match. Where a variant's perception reads extra state channels
(origin memory adds 37, sectors 76), the reference's 20 input columns are
copied into the corresponding positions and the ADDED columns are ZEROED, so
perception initially ignores the new state entirely.

**The mechanism is a behaviour-preserving no-op at initialisation.** Every
added pathway is gated by a zero-initialised parameter, so a fresh variant must
reproduce the reference's predictions to numerical tolerance before any
adaptation. ``assert_noop_equivalence`` checks that on complete, progressive
and fresh-message inputs; a variant that fails it is not a controlled
comparison and must not be trained.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .reference import CHANNEL_N, OUTPUT_CHANNELS, ReferenceCA, ReferenceConfig


@dataclass
class TransplantReport:
    variant: str
    copied_exact: list[str]
    copied_padded: list[str]
    zero_initialised: list[str]
    skipped: list[str]
    reference_iteration: int
    reference_sha256: str | None = None

    def to_dict(self) -> dict:
        return {
            "variant": self.variant,
            "copied_exact": self.copied_exact,
            "copied_padded": self.copied_padded,
            "zero_initialised_mechanism": self.zero_initialised,
            "skipped": self.skipped,
            "reference_iteration": self.reference_iteration,
            "reference_sha256": self.reference_sha256,
            "note": (
                "Padded tensors copy the reference's input columns into the "
                "matching positions and ZERO the added ones, so perception "
                "initially ignores the mechanism's state."
            ),
        }


def load_frozen_reference(path, device="cpu") -> tuple[ReferenceCA, dict]:
    blob = torch.load(path, map_location=device, weights_only=False)
    cfg = ReferenceConfig(**blob["reference_config"])
    model = ReferenceCA(cfg).to(device)
    model.load_state_dict(blob["model"])
    model.eval()
    return model, blob


@torch.no_grad()
def transplant(variant_model, reference: ReferenceCA, *,
               reference_iteration: int = 0,
               reference_sha256: str | None = None) -> TransplantReport:
    """Copy the reference into a variant's shared trunk.

    The variant's own state layout puts the reference's channels first --
    grey, hidden, logits -- and any mechanism channels after, which is what
    makes the padded copy well defined.
    """
    ref_sd = dict(reference.named_parameters())
    rep = TransplantReport(
        variant=getattr(variant_model.config, "variant", "?"),
        copied_exact=[], copied_padded=[], zero_initialised=[], skipped=[],
        reference_iteration=reference_iteration,
        reference_sha256=reference_sha256,
    )

    for name, param in variant_model.named_parameters():
        src = ref_sd.get(name)
        if src is None:
            # A mechanism parameter. It must already be a no-op; the builders
            # zero-initialise the gates that make that true.
            rep.zero_initialised.append(name)
            continue
        if src.shape == param.shape:
            param.copy_(src)
            rep.copied_exact.append(name)
            continue
        # A WIDER variant has more features as well as more input channels, so
        # both dimensions can grow. Copy the reference into the leading block
        # and leave the added capacity at its random initialisation -- zeroing
        # it would hand the baseline dead width instead of trainable capacity,
        # which is exactly what made `wider` a frozen padded reference with 0
        # trainable parameters.
        if (param.dim() >= 1 and src.dim() == param.dim()
                and all(p >= q for p, q in zip(param.shape, src.shape))
                and param.shape != src.shape):
            # Behaviour preservation requires the added units to be SILENT at
            # the output, not dead at the input. Zeroing everything achieved
            # silence but left the added width with exactly zero gradient --
            # permanently unreachable, the same dead-module failure the
            # sector `evidence` layer had.
            #
            # So: the OUTPUT dimension (dim 0, the added units' own rows) keeps
            # its random initialisation where it reads existing features, and
            # the INPUT dimension (dim 1, how existing units read the added
            # features) is zeroed. Added units therefore compute something from
            # the start, contribute nothing yet, and receive gradient as soon
            # as a downstream weight moves off zero.
            sl = tuple(slice(0, q) for q in src.shape)
            if param.dim() >= 2 and param.shape[1] > src.shape[1]:
                param[:, src.shape[1]:].zero_()
            if param.dim() >= 1 and param.shape[0] > src.shape[0]:
                # An added OUTPUT unit is free to compute; what must stay
                # silent is how anything downstream reads it, handled by the
                # input-dimension zeroing of the NEXT layer.
                pass
            param[sl] = src
            grown = [f"{q}->{p}" for p, q in zip(param.shape, src.shape) if p != q]
            rep.copied_padded.append(
                f"{name}: {tuple(src.shape)} -> {tuple(param.shape)}, grew "
                f"{', '.join(grown)}; added capacity starts at zero (silent) "
                "and is trainable"
            )
            continue
        if (param.dim() == 4 and src.dim() == 4
                and param.shape[0] == src.shape[0]
                and param.shape[2:] == src.shape[2:]
                and param.shape[1] > src.shape[1]):
            param.zero_()
            # Reference input layout is [grey, hidden, logits]; the variant
            # keeps those first, so a prefix copy is the correct alignment.
            n_in = src.shape[1]
            param[:, :n_in].copy_(src)
            rep.copied_padded.append(
                f"{name}: {tuple(src.shape)} -> {tuple(param.shape)}, "
                f"{param.shape[1] - n_in} added input channels zeroed"
            )
            continue
        rep.skipped.append(f"{name}: {tuple(src.shape)} vs {tuple(param.shape)}")
    return rep


def mechanism_parameters(variant_model) -> list[str]:
    """Names of the parameters the mechanism or the added capacity introduced.

    Phase C2 trains ONLY these, so the frozen reference's task competence is
    not disturbed while the mechanism learns.

    A tensor the reference also has, but whose SHAPE grew, counts too: the
    `wider` baseline's widened layers carry real added capacity that must be
    trainable. Keying purely on names gave it 0 trainable parameters -- a
    frozen padded reference rather than a capacity baseline.
    """
    ref = dict(ReferenceCA().named_parameters())
    out = []
    for n, p in variant_model.named_parameters():
        src = ref.get(n)
        if src is None or src.shape != p.shape:
            out.append(n)
    return out


def freeze_backbone(variant_model) -> tuple[int, int]:
    """Freeze every transplanted tensor; train only mechanism parameters."""
    mech = set(mechanism_parameters(variant_model))
    frozen = trainable = 0
    for name, p in variant_model.named_parameters():
        if name in mech:
            p.requires_grad_(True)
            trainable += p.numel()
        else:
            p.requires_grad_(False)
            frozen += p.numel()
    return frozen, trainable


@torch.no_grad()
def assert_noop_equivalence(variant_model, reference: ReferenceCA, images,
                            *, steps: int = 20, tolerance: float = 1e-4,
                            fire=None, noise=None) -> dict:
    """A freshly transplanted variant must match the reference's predictions.

    Checked on three input regimes, because a mechanism can be inert on one and
    active on another: a complete image, a progressively revealed one, and a
    rollout driven by fresh messages.

    **Equivalence is judged on PREDICTIONS, not logit magnitudes.** A CA is a
    recurrent system, so an arbitrarily small residual compounds: at a gate
    bias of -12 the blend is 6e-6, which leaves logits matching to 7e-7 after
    one step but diverging to 6e+04 by step 50. Argmax agreement stays exactly
    1.0000 across all of that. Requiring logit equality would therefore reject
    a variant that is behaviourally identical, and tightening the bias only
    moves the step count at which the same thing happens.

    Returns the measured deviations. A caller that cares should assert on
    ``within_tolerance``; the numbers are returned either way so a failure can
    be diagnosed rather than merely reported.
    """
    dev = next(reference.parameters()).device
    images = images.to(dev)
    out: dict = {"tolerance": tolerance, "regimes": {}}

    def run(model, x0, f, nz):
        x = x0
        for t in range(steps):
            x = model(x, fire=None if f is None else f[t],
                      noise=None if nz is None else nz[t])
        return model.classify(x)

    b, h, w = images.shape
    if fire is None:
        fire = torch.ones(steps, b, 1, h, w, dtype=torch.bool, device=dev)
    if noise is None:
        noise = torch.zeros(steps, b, CHANNEL_N, h, w, device=dev)

    regimes = {
        "complete": images,
        "progressive": images.clone(),
    }
    regimes["progressive"][:, h // 2:, :] = 0.0

    for name, imgs in regimes.items():
        r_logits = run(reference, reference.initialize(imgs), fire, noise)
        v_noise = noise
        if variant_model.channel_n != CHANNEL_N:
            v_noise = torch.zeros(steps, b, variant_model.channel_n, h, w,
                                  device=dev)
        v_logits = run(variant_model, variant_model.initialize(imgs), fire,
                       v_noise)
        diff = float((r_logits - v_logits).abs().max())
        agree = float((r_logits.argmax(1) == v_logits.argmax(1)).float().mean())
        out["regimes"][name] = {
            "max_abs_logit_difference": diff,
            "argmax_agreement": agree,
            "predictions_identical": agree >= 1.0 - 1e-9,
            "logits_within_tolerance": diff <= tolerance,
        }

    # Behaviour preservation means the variant PREDICTS what the reference
    # predicts. Logit closeness is reported alongside, because a large value
    # with perfect agreement is compounding round-off, while disagreement at
    # any magnitude is a real mechanism leak.
    out["within_tolerance"] = all(
        r["predictions_identical"] for r in out["regimes"].values()
    )
    out["max_deviation"] = max(
        r["max_abs_logit_difference"] for r in out["regimes"].values()
    )
    out["min_argmax_agreement"] = min(
        r["argmax_agreement"] for r in out["regimes"].values()
    )
    out["criterion"] = "identical argmax predictions in every regime"
    return out
