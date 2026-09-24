"""Annotation-based disappearance/reappearance events.

The event population is defined from ANNOTATIONS ONLY, before any model is
run. Selecting clips because a baseline fails on them, or because our method
wins on them, would make the comparison meaningless -- so nothing here reads a
prediction.

Terminology is deliberate. An object with no annotated mask is *not annotated
as present*; that is not the same as being confirmed physically occluded. It
may be outside the frame, too small to annotate, or missed by the annotator.
Everything produced here is therefore an **annotated disappearance/reappearance
event**, and the report must say so rather than claiming occlusion.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Iterable, Sequence

import numpy as np

# Frozen protocol constants. Changing one after seeing results invalidates the
# comparison, so they live here and are hashed into the protocol file.
MIN_PRESENT_BEFORE = 1      # annotated frames with the object present, before
MIN_ABSENCE = 3             # consecutive annotated frames with no mask
MIN_REAPPEAR = 2            # consecutive annotated frames present again
RECOVERY_WINDOW = 10        # annotated frames after reappearance
MIN_PIXELS = 1              # a mask this small or smaller counts as absent


@dataclass
class Event:
    """One annotated disappearance/reappearance of one object."""

    video: str
    object_id: int
    last_seen_frame: int        # index into the annotated-frame sequence
    absence_start: int
    absence_end: int            # inclusive
    reappear_frame: int
    absence_length: int
    recovery_window: list[int] = field(default_factory=list)
    distractors_present: list[int] = field(default_factory=list)
    truncated_at_clip_end: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def presence_from_masks(masks: Sequence[np.ndarray], object_id: int,
                        *, min_pixels: int = MIN_PIXELS) -> np.ndarray:
    """Boolean presence per annotated frame for one object id.

    `masks[i]` is an integer label map for annotated frame i, where 0 is
    background. Presence means the object's label covers more than
    `min_pixels` pixels -- a single stray pixel is annotation noise, not a
    visible object.
    """
    out = np.zeros(len(masks), dtype=bool)
    for i, m in enumerate(masks):
        if m is None:
            continue
        out[i] = int((np.asarray(m) == object_id).sum()) > min_pixels
    return out


def object_ids(masks: Iterable[np.ndarray]) -> list[int]:
    """Every non-background label appearing anywhere in the clip."""
    ids: set[int] = set()
    for m in masks:
        if m is None:
            continue
        ids.update(int(v) for v in np.unique(np.asarray(m)) if int(v) != 0)
    return sorted(ids)


def _runs(flags: np.ndarray, value: bool):
    """Yield (start, end_inclusive) runs of `value` in a boolean array."""
    n = len(flags)
    i = 0
    while i < n:
        if bool(flags[i]) != value:
            i += 1
            continue
        j = i
        while j + 1 < n and bool(flags[j + 1]) == value:
            j += 1
        yield i, j
        i = j + 1


def extract_events(video: str, masks: Sequence[np.ndarray], *,
                   min_present_before: int = MIN_PRESENT_BEFORE,
                   min_absence: int = MIN_ABSENCE,
                   min_reappear: int = MIN_REAPPEAR,
                   recovery_window: int = RECOVERY_WINDOW,
                   min_pixels: int = MIN_PIXELS) -> list[Event]:
    """Every qualifying event in one clip.

    An object's identity is PERSISTENT across the absence by construction: the
    same annotation label before and after. That is what makes the event a test
    of memory rather than of redetection -- the question is whether the system
    still calls it the same object.
    """
    ids = object_ids(masks)
    events: list[Event] = []
    n = len(masks)
    for oid in ids:
        present = presence_from_masks(masks, oid, min_pixels=min_pixels)
        for start, end in _runs(present, False):
            length = end - start + 1
            if length < min_absence:
                continue
            # Must have been annotated present before the absence...
            before = present[:start]
            if int(before.sum()) < min_present_before:
                continue
            # ...and annotated present for long enough afterwards.
            after = present[end + 1:]
            if len(after) < min_reappear or not bool(after[:min_reappear].all()):
                continue

            reappear = end + 1
            window = list(range(reappear, min(reappear + recovery_window, n)))
            # Which OTHER objects are annotated present during the absence?
            # These are the distractors that could displace the memory.
            distractors = [
                other for other in ids
                if other != oid
                and bool(presence_from_masks(masks, other,
                                             min_pixels=min_pixels)[start:end + 1].any())
            ]
            last_seen = int(np.max(np.flatnonzero(before))) if before.any() else -1
            events.append(Event(
                video=video, object_id=oid, last_seen_frame=last_seen,
                absence_start=start, absence_end=end, reappear_frame=reappear,
                absence_length=length, recovery_window=window,
                distractors_present=distractors,
                truncated_at_clip_end=len(window) < recovery_window,
            ))
    return events


def audit(per_video_masks, *, progress_every: int = 0, **kw) -> dict:
    """Population statistics, reported BEFORE any training.

    Accepts either a mapping of video -> masks, or an ITERABLE of
    (video, masks) pairs. The iterable form matters at MOSE's scale: holding
    every mask of 1,818 videos in memory at once was killed by the OOM killer,
    and only the event records need to be retained.
    """
    items = (sorted(per_video_masks.items())
             if hasattr(per_video_masks, "items") else per_video_masks)

    all_events: list[Event] = []
    per_video: dict[str, int] = {}
    for i, (video, masks) in enumerate(items, 1):
        ev = extract_events(video, masks, **kw)
        per_video[video] = len(ev)
        all_events.extend(ev)
        del masks                    # release before the next video loads
        if progress_every and i % progress_every == 0:
            print(f"[audit]   {i} videos scanned, {len(all_events)} events "
                  f"so far", flush=True)

    lengths = [e.absence_length for e in all_events]
    with_distractor = sum(1 for e in all_events if e.distractors_present)
    objects = {(e.video, e.object_id) for e in all_events}
    return {
        "definition": {
            "min_present_before": kw.get("min_present_before", MIN_PRESENT_BEFORE),
            "min_absence": kw.get("min_absence", MIN_ABSENCE),
            "min_reappear": kw.get("min_reappear", MIN_REAPPEAR),
            "recovery_window": kw.get("recovery_window", RECOVERY_WINDOW),
            "min_pixels": kw.get("min_pixels", MIN_PIXELS),
            "note": (
                "Annotated disappearance/reappearance. Missing annotation is "
                "NOT confirmed physical occlusion and must not be reported as "
                "such."
            ),
        },
        # Counted from what was actually consumed: a stream has no len().
        "videos_total": len(per_video),
        "videos_with_events": sum(1 for v in per_video.values() if v),
        "objects_with_events": len(objects),
        "events_total": len(all_events),
        "events_per_video": per_video,
        "absence_length": {
            "min": int(min(lengths)) if lengths else None,
            "median": float(np.median(lengths)) if lengths else None,
            "max": int(max(lengths)) if lengths else None,
            "mean": round(float(np.mean(lengths)), 2) if lengths else None,
        },
        "events_with_a_visible_distractor": with_distractor,
        "events_truncated_at_clip_end": sum(1 for e in all_events
                                            if e.truncated_at_clip_end),
        "events": [e.to_dict() for e in all_events],
    }
