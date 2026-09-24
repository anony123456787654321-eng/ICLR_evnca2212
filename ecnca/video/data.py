"""Clip sampling. Splits are by VIDEO, never by frame.

Frames within a video are not independent evidence: two adjacent frames of the
same clip share almost everything, so splitting by frame would put near-copies
of the evaluation data into training. Every split here is a partition of
videos.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ecnca.video.mose import _frames, _is_sidecar

# Calibration/training clips require a foreground prompt and a scored frame.
# Included in controller fingerprints; EventClipLoader uses a separate policy.
CLIP_SAMPLING_VERSION = 2


@dataclass
class Clip:
    video: str
    frames: list           # list of HxWx3 uint8 arrays
    masks: list            # list of HxW integer label maps
    frame_indices: list
    object_ids: list
    # Set for event clips only.
    event: dict | None = None
    history_frames: int = 0
    window_positions: list = field(default_factory=list)
    # Ground truth at dataset resolution.  XMem consumes `masks` resized to
    # its official 480-short-side inference geometry; evaluation upsamples
    # probabilities and scores against these untouched masks.
    original_masks: list | None = None


def _dataset_roots(data_root: Path, dataset: str,
                   mose_split: str | None = None) -> tuple[Path, Path]:
    if dataset.startswith("DAVIS"):
        base = next(iter(sorted((data_root / "datasets").glob("DAVIS*"))), None)
        if base is None:
            raise SystemExit(f"no DAVIS extraction under {data_root}/datasets")
        for cand in (base / "DAVIS", base):
            if (cand / "Annotations" / "480p").exists():
                return cand / "JPEGImages" / "480p", cand / "Annotations" / "480p"
        raise SystemExit(f"DAVIS layout not recognised under {base}")
    # MOSE ships train AND valid. Returning the first layout that existed
    # meant that once train/ was present, valid/ was never looked at -- and
    # the evaluation split would have been drawn from training videos.
    # `mose_split` names which official split to read.
    from ecnca.video.mose import locate_splits

    base = data_root / "mosev1"
    for root in (base / "extracted", base):
        if not root.exists():
            continue
        layouts = locate_splits(root)
        wanted = mose_split or "train"
        layout = layouts.get(wanted)
        if layout is not None and not layout.problems:
            return Path(layout.images), Path(layout.annotations)
    raise SystemExit(
        f"no usable MOSE {mose_split or 'train'!r} split under {base}. "
        f"Run scripts/acquire_mose.sh, which validates the layout.")


class ClipSampler:
    """Samples contiguous clips from the videos of one split."""

    def __init__(self, data_root: Path, dataset: str, *, split: str,
                 clip_length: int = 8, seed: int = 0,
                 audit_path: Path | None = None):
        # Every split -- including eval -- is read from MOSE's official
        # TRAINING directory, because official validation ships first-frame
        # masks only and cannot be scored locally. Which videos are held out
        # is decided by the frozen manifest, never by the directory.
        #
        # This previously hardcoded eval -> official valid, which made the
        # held-out split unscorable and produced zero events.
        self.images, self.annotations = _dataset_roots(
            Path(data_root), dataset, mose_split="train")
        self.clip_length = clip_length
        if self.clip_length < 2:
            raise ValueError("clip_length must include a prompt and a following frame")
        self.split = split
        self._prompt_starts = {}
        videos = sorted(p.name for p in self.annotations.iterdir()
                        if p.is_dir() and not _is_sidecar(p))
        assignment = None
        if audit_path and Path(audit_path).exists():
            blob = json.loads(Path(audit_path).read_text())
            # The frozen manifest is authoritative; the audit embeds it.
            assignment = (blob.get("assignment")
                          or (blob.get("manifest") or {}).get("assignment")
                          or blob.get("splits"))
        # NO FALLBACK SPLIT. A sampler that invents its own assignment can
        # train on videos the evaluation holds out, and it does so silently:
        # the run completes, the numbers look ordinary, and the leak is
        # invisible in the output. Refusing is the only safe behaviour --
        # the frozen manifest is the single source of the split.
        if not assignment:
            raise SystemExit(
                f"no frozen video assignment was supplied (audit: "
                f"{audit_path!r}). Refusing to invent a {split!r} split: a "
                f"guessed assignment can train on held-out evaluation "
                f"videos. Pass the event audit whose manifest defines the "
                f"splits.")
        self.videos = [v for v in videos if assignment.get(v) == split]
        if not self.videos:
            raise SystemExit(f"no videos in split {split!r}")

    def sample(self, rng, *, device=None) -> Clip:
        import torch
        from PIL import Image

        # Draw from videos in this split, then uniformly from valid starts.
        # Empty masks are legitimate during occlusion, but cannot initialise
        # a semi-supervised tracker. Validate only the prompt; later frames
        # are retained even when the tracked object disappears completely.
        # Cached annotation scans never consume RNG, preserving resume pairing.
        remaining = list(self.videos)
        while remaining:
            video = remaining.pop(int(rng.integers(0, len(remaining))))
            frames = _frames(self.annotations / video)
            n = len(frames)
            take = min(self.clip_length, n)
            if video not in self._prompt_starts:
                starts = []
                if take >= 2:
                    for i in range(n - take + 1):
                        with Image.open(frames[i]) as image:
                            if np.any(np.asarray(image) != 0):
                                starts.append(i)
                self._prompt_starts[video] = starts
            starts = self._prompt_starts[video]
            if starts:
                start = starts[int(rng.integers(0, len(starts)))]
                break
        else:
            raise ValueError(
                f"no trackable clips in split {self.split!r}: "
                f"{len(self.videos)} videos checked; each clip needs an "
                "annotated object in its prompt frame and a following frame")
        idx = list(range(start, start + take))

        masks, imgs = [], []
        for i in idx:
            m = np.array(Image.open(frames[i]))
            masks.append(m)
            jpg = self.images / video / (frames[i].stem + ".jpg")
            imgs.append(np.array(Image.open(jpg).convert("RGB"))
                        if jpg.exists() else
                        np.zeros(m.shape + (3,), dtype=np.uint8))
        ids = sorted(int(v) for v in np.unique(masks[0]) if v != 0)
        clip = Clip(video=video, frames=imgs, masks=masks,
                    frame_indices=idx, object_ids=ids)
        # CPU tensors by default, matching EventClipLoader: callers stream
        # individual frames to the device. `device=` forces residency.
        clip = to_cpu_tensors(clip)
        if device is not None:
            clip = to_device(clip, device)
        return clip


def to_cpu_tensors(clip: Clip) -> Clip:
    """Frames and masks as CPU tensors. Nothing touches the accelerator.

    Images and ground truth stay on CPU for the whole run; only the single
    frame currently being segmented is streamed to the device. Materialising
    whole clips on CUDA exhausted the allocator at 93 events.
    """
    import torch

    # IDEMPOTENT. Both loaders convert before optionally calling to_device,
    # which converted again -- "expected np.ndarray (got Tensor)".
    if clip.frames and torch.is_tensor(clip.frames[0]):
        return clip
    # THE official transform: ToTensor + ImageNet normalisation. A bare
    # div(255) left the frozen encoder seeing inputs that differ from the
    # official pipeline by up to 2.1179, so every feature it produced was
    # wrong. See ecnca/video/preprocess.py.
    from ecnca.video.preprocess import frame_and_mask_to_tensors

    converted = [frame_and_mask_to_tensors(f, m)
                 for f, m in zip(clip.frames, clip.masks)]
    clip.frames = [x[0] for x in converted]
    clip.masks = [x[1] for x in converted]
    clip.original_masks = [x[2] for x in converted]
    return clip


def to_device(clip: Clip, device):
    """Kept for callers that genuinely want a resident clip.

    Prefer `to_cpu_tensors` plus per-frame streaming: a clip of 70 frames at
    480p is ~250 MiB resident, and holding many at once is what failed.
    """
    import torch

    clip = to_cpu_tensors(clip)
    clip.frames = [f.to(device) for f in clip.frames]
    clip.masks = [m.to(device) for m in clip.masks]
    if clip.original_masks is not None:
        clip.original_masks = [m.to(device) for m in clip.original_masks]
    return clip


def stream_frame(tensor, device):
    """One frame to the device, non-blocking where possible."""
    return tensor.to(device, non_blocking=True)


# --------------------------------------------------------------------------
# Event clips: deterministic, identical across configurations
# --------------------------------------------------------------------------
# The memory ablations must be evaluated on the PREDEFINED reappearance
# events, not on randomly sampled clips. Two requirements follow:
#
#   HISTORY   the clip must start early enough that the object was seen
#             BEFORE it disappeared. Without that there is nothing for memory
#             to remember, and a "long-term memory does not help" result would
#             be a statement about the clip, not about the task.
#
#   IDENTICAL every configuration must see the same frames in the same order.
#             Comparing configurations on different sequences measures the
#             sampler, not the ablation.
#
# So event clips are built deterministically from the event record.

# Enough preceding history that XMem's LONG-TERM memory actually engages.
#
# Measured on the pinned revision: long-term memory engages once working
# memory fills, i.e. after `max_mid_term_frames` (10) memory frames, and a
# frame enters memory every `mem_every` (5) frames -- so roughly 50 frames are
# needed. A 30-frame clip engaged it 0 times; a 52-frame clip engaged it 6
# times with 128 elements.
#
# With only 5 frames of history the long-term ablation changed nothing the
# model read, and the instrumentation correctly reported READS UNCHANGED. That
# is a property of the clip, not of the task, so the history is set from the
# measurement rather than from a guess.
MIN_HISTORY_FRAMES = 50     # annotated frames BEFORE the absence


class EventClipLoader:
    """Deterministic clips built around annotated reappearance events."""

    def __init__(self, data_root, dataset: str, *,
                 min_history: int = MIN_HISTORY_FRAMES,
                 mose_split: str = "train"):
        from pathlib import Path

        # Held-out event videos live in MOSE's official TRAINING directory:
        # official validation has first-frame masks only, so a recovery window
        # cannot be scored there. The manifest decides which are held out.
        self.images, self.annotations = _dataset_roots(
            Path(data_root), dataset, mose_split=mose_split)
        self.min_history = min_history

    def eligible(self, event: dict) -> tuple[bool, str]:
        """Enough preceding history for memory to have something to hold?"""
        history = int(event["absence_start"]) - int(event["last_seen_frame"]) + 1
        first_seen = int(event["last_seen_frame"]) - history + 1
        available = int(event["absence_start"])
        if available < self.min_history:
            return False, (
                f"only {available} annotated frames precede the absence; "
                f"{self.min_history} are required for the object to have been "
                f"established in memory")
        return True, "sufficient preceding history"

    def load(self, event: dict, *, device=None):
        """The clip for one event: history, absence, and the frozen window.

        Identical for every configuration -- nothing here is sampled.
        """
        import numpy as np
        from PIL import Image

        video = event["video"]
        ann_dir = self.annotations / video
        frames = _frames(ann_dir)

        start = max(0, int(event["absence_start"]) - self.min_history)
        window = [int(i) for i in event.get("recovery_window", [])]
        end = max(window) if window else int(event["reappear_frame"])
        end = min(end, len(frames) - 1)
        idx = list(range(start, end + 1))

        masks, imgs = [], []
        for i in idx:
            m = np.array(Image.open(frames[i]))
            masks.append(m)
            jpg = self.images / video / (frames[i].stem + ".jpg")
            imgs.append(np.array(Image.open(jpg).convert("RGB"))
                        if jpg.exists() else
                        np.zeros(m.shape + (3,), dtype=np.uint8))
        ids = sorted(int(v) for v in np.unique(masks[0]) if v != 0)
        if int(event["object_id"]) not in ids:
            # The target must be annotated in the prompt frame; otherwise the
            # semi-supervised setting cannot be initialised for it.
            ids = sorted(set(ids) | {int(event["object_id"])})
        clip = Clip(video=video, frames=imgs, masks=masks,
                    frame_indices=idx, object_ids=ids)
        clip.event = dict(event)
        clip.history_frames = int(event["absence_start"]) - start
        clip.window_positions = [idx.index(i) for i in window if i in idx]
        # CPU tensors by default: the caller streams individual frames to the
        # device. `device=` remains for callers that want residency.
        clip = to_cpu_tensors(clip)
        if device is not None:
            clip = to_device(clip, device)
        return clip
