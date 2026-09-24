"""Segmentation loss over a clip, with our controller in the readout path.

The loss is the same for every arm. The ONLY difference between arms is what
`seam` does to the memory readout, which is what makes the comparison a
comparison.
"""
from __future__ import annotations

# Bump when the probability/logit contract changes. v1 fed probabilities to
# cross_entropy (softmaxing them a second time) and to pred.softmax(1) (a
# third): gradients were 9,697x too small with cosine 0.19 to correct.
LOSS_CONTRACT_VERSION = 2

import torch
import torch.nn.functional as F


def clip_segmentation_loss(xmem, seam, clip, device) -> torch.Tensor:
    """Cross-entropy plus soft-IoU over the clip's annotated frames.

    The first frame provides the annotation (semi-supervised VOS); later
    ground-truth masks are used for the TRAINING loss only on training videos,
    and never on held-out ones.
    """
    from ecnca.video.xmem_forward import run_clip

    logits = run_clip(xmem, seam, clip, device)     # list of (1,K+1,H,W)
    if not logits:
        raise RuntimeError("the clip produced no predictions")

    # Predictions are on the device; ground truth stays on CPU so that whole
    # clips are never resident. The target therefore has to be moved here --
    # it previously was not, and cross_entropy raised
    # "found at least two devices, cuda:0 and cpu".
    #
    # The target also has to be REMAPPED. run_clip gives XMem positional
    # labels 1..n, so prediction channel k+1 corresponds to
    # clip.object_ids[k]. Using raw object ids as class indices was wrong for
    # any non-contiguous id set -- which MOSE has -- and clamping them merely
    # hid it by folding out-of-range ids onto the last channel.
    id_to_channel = {int(o): i + 1 for i, o in enumerate(clip.object_ids)}

    total = logits[0].new_zeros(())
    counted = 0
    for pred, gt in zip(logits[1:], clip.masks[1:]):   # frame 0 is the prompt
        target = gt if gt.dim() == 3 else gt.squeeze(1)
        target = target.to(pred.device, non_blocking=True)
        if pred.shape[-2:] != target.shape[-2:]:
            pred = F.interpolate(pred, size=target.shape[-2:],
                                 mode="bilinear", align_corners=False)
        k = pred.shape[1]
        # ids -> channels; anything unannotated stays background (0).
        remapped = torch.zeros_like(target)
        for oid, ch in id_to_channel.items():
            if ch < k:
                remapped = torch.where(target == oid,
                                       torch.full_like(target, ch), remapped)
        target = remapped

        # `InferenceCore.step()` returns pred_prob_with_bg -- PROBABILITIES,
        # already softmaxed in model/aggregate.py:12. Passing them to
        # cross_entropy softmaxed them a SECOND time and soft-IoU softmaxed a
        # third: measured gradients were 9,697x too small with cosine 0.19 to
        # the correct direction, so training moved almost not at all and what
        # movement there was pointed nearly orthogonally.
        #
        # Probabilities therefore take log + NLL, and soft-IoU uses them
        # directly. Interpolation can break normalisation slightly, so the
        # distribution is renormalised rather than re-softmaxed.
        prob = pred.clamp_min(0)
        prob = prob / prob.sum(dim=1, keepdim=True).clamp_min(1e-7)
        ce = F.nll_loss(torch.log(prob.clamp_min(1e-7)), target)
        oh = F.one_hot(target, k).permute(0, 3, 1, 2).to(prob.dtype)
        inter = (prob * oh).sum(dim=(2, 3))
        union = (prob + oh - prob * oh).sum(dim=(2, 3)).clamp_min(1e-6)
        soft_iou = 1.0 - (inter / union).mean()
        total = total + ce + soft_iou
        counted += 1
    return total / max(counted, 1)
