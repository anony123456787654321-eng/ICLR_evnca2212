"""How densely is each split annotated, and is it scorable at all?

A split whose videos carry only a first-frame mask cannot produce a
disappearance/reappearance event: an absence needs at least
MIN_ABSENCE annotated frames showing the object gone, and one annotated frame
can never show that. The event extractor therefore returns zero for such a
split -- which is NOT the same fact as "these videos contain no reappearances".

Conflating the two is what produced the earlier claim that MOSE had
insufficient events. This module measures the difference and names it.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path

# A split needs at least this many annotated frames per video before an
# absence of MIN_ABSENCE frames plus a reappearance can even be represented.
from ecnca.video.events import MIN_ABSENCE, MIN_PRESENT_BEFORE, MIN_REAPPEAR

MIN_SCORABLE_FRAMES = MIN_PRESENT_BEFORE + MIN_ABSENCE + MIN_REAPPEAR


@dataclass
class SplitCoverage:
    """Mask-versus-image coverage for one official split."""

    split: str
    videos: int = 0
    videos_with_masks: int = 0
    image_frames: int = 0
    mask_frames: int = 0
    mask_frames_per_video_min: int = 0
    mask_frames_per_video_median: float = 0.0
    mask_frames_per_video_max: int = 0
    first_frame_only_videos: int = 0
    densely_annotated_videos: int = 0
    scorable: bool = False
    status: str = ""
    detail: str = ""
    examples: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def measure(images_dir, annotations_dir, split: str, *,
            sample: int = 0) -> SplitCoverage:
    """Count annotated frames against image frames, per video."""
    import numpy as np

    from ecnca.video.mose import _frames, _is_sidecar

    images_dir, annotations_dir = Path(images_dir), Path(annotations_dir)
    cov = SplitCoverage(split=split)
    vids = sorted(p for p in annotations_dir.iterdir()
                  if p.is_dir() and not _is_sidecar(p))
    if sample:
        vids = vids[:sample]
    counts = []
    for v in vids:
        cov.videos += 1
        masks = _frames(v)
        imgs = [f for f in (images_dir / v.name).glob("*.jpg")
                if not _is_sidecar(f)] if (images_dir / v.name).is_dir() else []
        cov.image_frames += len(imgs)
        cov.mask_frames += len(masks)
        if masks:
            cov.videos_with_masks += 1
            counts.append(len(masks))
            if len(masks) == 1:
                cov.first_frame_only_videos += 1
                if len(cov.examples) < 3:
                    cov.examples.append(
                        {"video": v.name, "mask_frames": 1,
                         "image_frames": len(imgs)})
            if len(masks) >= MIN_SCORABLE_FRAMES:
                cov.densely_annotated_videos += 1
    if counts:
        cov.mask_frames_per_video_min = int(min(counts))
        cov.mask_frames_per_video_median = float(np.median(counts))
        cov.mask_frames_per_video_max = int(max(counts))

    dense_frac = (cov.densely_annotated_videos / cov.videos
                  if cov.videos else 0.0)
    if cov.videos == 0:
        cov.status = "empty"
        cov.detail = "no annotated videos"
    elif cov.first_frame_only_videos >= 0.9 * cov.videos:
        cov.status = "first_frame_only"
        cov.scorable = False
        cov.detail = (
            f"{cov.first_frame_only_videos}/{cov.videos} videos carry a "
            f"single annotated frame. Local event scoring is UNAVAILABLE on "
            f"this split -- an absence needs at least {MIN_ABSENCE} annotated "
            f"frames showing the object gone. This is NOT evidence that the "
            f"videos contain no reappearances; the public masks are withheld "
            f"for benchmark submission."
        )
    elif dense_frac >= 0.5:
        cov.status = "densely_annotated"
        cov.scorable = True
        cov.detail = (
            f"{cov.densely_annotated_videos}/{cov.videos} videos carry at "
            f"least {MIN_SCORABLE_FRAMES} annotated frames; events can be "
            f"extracted and scored locally.")
    else:
        cov.status = "sparsely_annotated"
        cov.scorable = False
        cov.detail = (
            f"only {cov.densely_annotated_videos}/{cov.videos} videos carry "
            f"{MIN_SCORABLE_FRAMES}+ annotated frames; local scoring would "
            f"rest on a small and unrepresentative subset.")
    return cov


def report(layouts: dict, *, sample: int = 0) -> dict:
    """Coverage for every official split, reported SEPARATELY."""
    out = {"splits": {}, "min_scorable_frames": MIN_SCORABLE_FRAMES}
    for name, layout in layouts.items():
        if not layout.images or not layout.annotations:
            out["splits"][name] = SplitCoverage(
                split=name, status="missing",
                detail="the split has no image or annotation directory"
            ).to_dict()
            continue
        out["splits"][name] = measure(layout.images, layout.annotations,
                                      name, sample=sample).to_dict()
    scorable = [n for n, c in out["splits"].items() if c.get("scorable")]
    out["scorable_splits"] = scorable
    out["verdict"] = (
        f"local event scoring is available on {scorable}"
        if scorable else
        "NO official split carries dense enough masks for local event scoring"
    )
    return out
