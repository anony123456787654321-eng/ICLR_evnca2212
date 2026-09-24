"""Segmentation and mechanism metrics, with thresholds frozen before training.

The MNIST campaign lost a comparison to a control that scored well by
ABSTAINING -- it suppressed almost every update, and the endpoint could not
tell that apart from acting correctly. The metrics here are built so that
cannot recur: a system that predicts nothing is scored as a miss, never as
successful identity preservation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# --- frozen thresholds ----------------------------------------------------
RECOVERY_IOU = 0.5          # target IoU counting as recovered
RECOVERY_CONSECUTIVE = 2    # consecutive annotated frames required
CONFUSION_IOU = 0.3         # distractor overlap counting as confusion
CONFUSION_CONSECUTIVE = 2
BOUNDARY_FRACTION = 0.008   # DAVIS boundary tolerance, fraction of diagonal


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """Region similarity J. Both empty is perfect; one empty is zero."""
    p, g = np.asarray(pred).astype(bool), np.asarray(gt).astype(bool)
    inter = int(np.logical_and(p, g).sum())
    union = int(np.logical_or(p, g).sum())
    if union == 0:
        return 1.0          # both empty: correctly predicted absence
    return inter / union


def _seg2bmap(seg: np.ndarray) -> np.ndarray:
    """Boundary map, VERBATIM from the official DAVIS evaluator.

    Ours used a 4-connected erosion instead. On clean shapes the two agree,
    but on a ragged boundary they did not: measured 0.2837 against the
    official 0.1574 -- a 12.6-point disagreement on exactly the kind of mask a
    model produces while recovering from an occlusion, and four times the
    3-point effect this study is trying to detect.
    """
    seg = np.asarray(seg).astype(bool)
    e = np.zeros_like(seg)
    s_ = np.zeros_like(seg)
    se = np.zeros_like(seg)
    e[:, :-1] = seg[:, 1:]
    s_[:-1, :] = seg[1:, :]
    se[:-1, :-1] = seg[1:, 1:]
    b = (seg ^ e) | (seg ^ s_) | (seg ^ se)
    b[-1, :] = seg[-1, :] ^ e[-1, :]
    b[:, -1] = seg[:, -1] ^ s_[:, -1]
    b[-1, -1] = 0
    return b


def _disk(radius: int) -> np.ndarray:
    """Disk structuring element, matching skimage.morphology.disk."""
    r = int(radius)
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Dilation by a DISK, as the official evaluator uses.

    Ours dilated by a diamond (repeated 4-connected shifts), which reaches
    fewer pixels at the same radius and inflated the boundary score.
    """
    m = np.asarray(mask).astype(bool)
    if radius <= 0 or not m.any():
        return m
    se = _disk(radius)
    out = np.zeros_like(m)
    h, w = m.shape
    ys, xs = np.nonzero(se)
    for dy, dx in zip(ys - radius, xs - radius):
        ys0, ys1 = max(0, dy), min(h, h + dy)
        xs0, xs1 = max(0, dx), min(w, w + dx)
        if ys0 >= ys1 or xs0 >= xs1:
            continue
        out[ys0:ys1, xs0:xs1] |= m[ys0 - dy:ys1 - dy, xs0 - dx:xs1 - dx]
    return out


def boundary_f(pred: np.ndarray, gt: np.ndarray,
               bound_th: float = 0.008) -> float:
    """Boundary F, following the official DAVIS `f_measure` exactly.

    Verified against the official implementation across shifted, shrunk,
    grown, disjoint, thin, multi-object, ragged and empty cases.
    """
    fg = np.asarray(pred).astype(bool)
    g = np.asarray(gt).astype(bool)
    bound_pix = (bound_th if bound_th >= 1
                 else np.ceil(bound_th * np.linalg.norm(fg.shape)))
    fgb, gtb = _seg2bmap(fg), _seg2bmap(g)
    r = int(bound_pix)
    fgd, gtd = _dilate(fgb, r), _dilate(gtb, r)
    n_fg, n_gt = int(fgb.sum()), int(gtb.sum())
    # The official empty-mask convention, which differs from a naive one.
    if n_fg == 0 and n_gt > 0:
        precision, recall = 1.0, 0.0
    elif n_fg > 0 and n_gt == 0:
        precision, recall = 0.0, 1.0
    elif n_fg == 0 and n_gt == 0:
        precision, recall = 1.0, 1.0
    else:
        precision = float((fgb & gtd).sum()) / n_fg
        recall = float((gtb & fgd).sum()) / n_gt
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def j_and_f(pred: np.ndarray, gt: np.ndarray) -> dict:
    j, f = iou(pred, gt), boundary_f(pred, gt)
    return {"J": j, "F": f, "JF": (j + f) / 2}


@dataclass
class RecoveryResult:
    recovered: bool
    delay_frames: int | None      # None when never recovered
    frames_evaluated: int

    def to_dict(self) -> dict:
        return {"recovered": self.recovered, "delay_frames": self.delay_frames,
                "frames_evaluated": self.frames_evaluated}


def recovery_delay(pred_masks, gt_masks, *, iou_threshold: float = RECOVERY_IOU,
                   consecutive: int = RECOVERY_CONSECUTIVE) -> RecoveryResult:
    """Frames until the target is held at IoU >= threshold for `consecutive`.

    Delay 0 means recovered immediately at the reappearance frame. An event
    that never recovers returns `recovered=False` and delay None -- it is
    REPORTED, never dropped, because silently discarding unrecovered events
    would flatter every method equally and ours most of all.
    """
    n = min(len(pred_masks), len(gt_masks))
    run = 0
    for i in range(n):
        if iou(pred_masks[i], gt_masks[i]) >= iou_threshold:
            run += 1
            if run >= consecutive:
                return RecoveryResult(True, i - consecutive + 1, n)
        else:
            run = 0
    return RecoveryResult(False, None, n)


@dataclass
class ConfusionResult:
    confused: bool                 # assigned to a DIFFERENT annotated object
    missed: bool                   # predicted (almost) nothing
    first_frame: int | None
    distractor_id: int | None

    def to_dict(self) -> dict:
        return {"confused": self.confused, "missed": self.missed,
                "first_frame": self.first_frame,
                "distractor_id": self.distractor_id}


def identity_confusion(pred_masks, gt_target, gt_others, *,
                       iou_threshold: float = CONFUSION_IOU,
                       consecutive: int = CONFUSION_CONSECUTIVE,
                       empty_fraction: float = 1e-4) -> ConfusionResult:
    """Is the target's prediction actually sitting on another object?

    Confusion requires BOTH that the prediction overlaps a distractor at least
    `iou_threshold`, AND that it overlaps that distractor more than the target
    -- an ordinary imprecise boundary brushing a neighbour is not a confusion.

    `gt_others` maps distractor id -> per-frame masks.

    Missed detection is tracked SEPARATELY and is never counted as success. A
    system that outputs an empty mask has not preserved identity; it has
    declined to answer. On MNIST a control scored near the ceiling precisely by
    declining to act, and the endpoint could not see it.
    """
    n = len(pred_masks)
    runs: dict[int, int] = {}
    missed_run = 0
    missed_at: int | None = None
    for i in range(n):
        p = np.asarray(pred_masks[i]).astype(bool)
        total = p.size
        if int(p.sum()) <= max(1, int(empty_fraction * total)):
            missed_run += 1
            if missed_run >= consecutive and missed_at is None:
                missed_at = i - consecutive + 1
            runs = {k: 0 for k in runs}
            continue
        missed_run = 0
        t_iou = iou(p, gt_target[i]) if i < len(gt_target) else 0.0
        for did, masks in gt_others.items():
            if i >= len(masks):
                continue
            d_iou = iou(p, masks[i])
            if d_iou >= iou_threshold and d_iou > t_iou:
                runs[did] = runs.get(did, 0) + 1
                if runs[did] >= consecutive:
                    return ConfusionResult(True, False,
                                           i - consecutive + 1, int(did))
            else:
                runs[did] = 0
    return ConfusionResult(False, missed_at is not None, missed_at, None)


def aggregate_by_video(per_event: list[dict], key: str) -> dict:
    """Average within each video BEFORE averaging across videos.

    A clip containing many events would otherwise dominate the aggregate, so
    the population would silently be "events in busy clips" rather than
    "events".
    """
    by_video: dict[str, list[float]] = {}
    for e in per_event:
        v = e.get("video")
        val = e.get(key)
        if val is None:
            continue
        by_video.setdefault(v, []).append(float(val))
    per_video = {v: float(np.mean(vals)) for v, vals in by_video.items()}
    vals = list(per_video.values())
    return {
        "per_video": per_video,
        "mean_over_videos": round(float(np.mean(vals)), 6) if vals else None,
        "n_videos": len(per_video),
        "n_events": sum(len(v) for v in by_video.values()),
    }
