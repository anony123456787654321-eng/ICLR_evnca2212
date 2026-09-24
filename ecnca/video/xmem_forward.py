"""The one place that speaks XMem's inference API.

Isolated so that every arm runs the IDENTICAL published forward and differs
only in the seam. If XMem's interface changes, exactly one file moves.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def label_at_original_resolution(prob, original_mask):
    """Official eval order: bilinear probability resize, then argmax."""
    target = tuple(original_mask.shape[-2:])
    p = prob.unsqueeze(0) if prob.dim() == 3 else prob
    if tuple(p.shape[-2:]) != target:
        p = F.interpolate(p, size=target, mode="bilinear",
                          align_corners=False)
    return p.argmax(1)[0]


def run_clip(xmem, seam, clip, device):
    """Run a clip through XMem with our controller on the memory readout.

    XMem's InferenceCore drives memory management; we intercept the readout
    feature grid. Raises rather than approximating: a hand-rolled substitute
    would not be the published baseline.
    """
    from inference.inference_core import InferenceCore   # from the XMem repo

    # XMem indexes objects POSITIONALLY and assumes labels 1..n; both DAVIS
    # and MOSE contain non-contiguous ids (DAVIS `drone` is [1,2,3,5]), which
    # indexes past the allocated channels. Give it positional labels and keep
    # the id mapping on our side.
    ids = list(clip.object_ids)
    if not ids:
        # No object is annotated in the prompt frame, so the semi-supervised
        # setting cannot be initialised for this clip. Raising a specific
        # error beats `stack expects a non-empty TensorList` from inside
        # XMem. ClipSampler must prevent this for calibration/training.
        raise ValueError(
            f"clip {clip.video!r} has no annotated object in its prompt "
            f"frame; nothing can be tracked")
    positional = list(range(1, len(ids) + 1))
    processor = InferenceCore(xmem.network, config=xmem.config)
    processor.set_all_labels(positional)

    # The seam point is named here and verified at run time. It is the ONE
    # assumption in this study about XMem's internals, and it is checked
    # against the pinned revision by the integration stage rather than
    # discovered mid-training. If the attribute is absent the study stops:
    # silently skipping the interception would train every arm as the
    # unmodified baseline and report the result as a comparison.
    if not hasattr(processor, "memory") or not hasattr(
            processor.memory, "match_memory"):
        raise RuntimeError(
            "XMem's InferenceCore does not expose memory.match_memory at the "
            "pinned revision. The integration point has moved; update "
            "ecnca/video/xmem_forward.py and re-run the integration gate "
            "rather than proceeding with an unhooked controller."
        )
    original = processor.memory.match_memory
    # A class-bound method must not be restored as an instance attribute:
    # memory -> bound method -> memory is a cycle retaining CUDA tensors.
    had_override = "match_memory" in processor.memory.__dict__
    prior_override = processor.memory.__dict__.get("match_memory")
    if seam.controller is not None:
        def patched(query_key, selection):
            # XMem returns (num_objects, CV, h, w) -- objects occupy the FIRST
            # dimension and the caller adds the batch dim afterwards
            # (inference_core.py:64). Our controller treats dimension 0 as
            # batch, so each object is processed independently, which is the
            # right semantics: a hypothesis about object A must not be mixed
            # with one about object B.
            readout = original(query_key, selection)
            out = seam(readout)
            if out.shape != readout.shape:
                raise RuntimeError(
                    f"the controller changed the readout shape "
                    f"{tuple(readout.shape)} -> {tuple(out.shape)}; XMem's "
                    f"decoder requires it unchanged")
            return out
        processor.memory.match_memory = patched

    out = []
    try:
        for i, frame in enumerate(clip.frames):
            img = frame[0] if frame.dim() == 4 else frame
            img = img.to(device, non_blocking=True)   # ONE frame resident
            if i == 0:
                # XMem's step() documents `mask: num_objects*H*W` -- a
                # per-object one-hot STACK, not an integer label map. Passing
                # the label map fails inside encode_value with a rank error,
                # so the conversion happens here, once, next to the contract.
                lab = clip.masks[0]
                while lab.dim() > 2:
                    lab = lab[0]
                mask = torch.stack(
                    [(lab == oid).float() for oid in ids], dim=0
                ).to(device, non_blocking=True)
                prob = processor.step(img, mask, positional)
            else:
                prob = processor.step(img)
            out.append(prob.unsqueeze(0) if prob.dim() == 3 else prob)
    finally:
        if seam.controller is not None:
            if had_override:
                processor.memory.match_memory = prior_override
            else:
                del processor.memory.match_memory
    return out
