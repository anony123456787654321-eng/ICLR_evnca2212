"""The frozen split manifest: which videos train, calibrate, and evaluate.

Built ONCE from densely annotated official training videos, written to disk,
and thereafter read rather than recomputed. Every consumer -- audit, training
sampler, calibration, memory-ablation gate, event loader, evaluator -- reads
this one file, so they cannot disagree about which videos are held out.

Why a custom split at all: MOSE's official validation ships first-frame masks
only, so local event scoring is impossible there. The official TRAINING split
is densely annotated, and the baseline checkpoint never trained on MOSE (see
ecnca/video/provenance.py), so holding out training videos is clean.

Results from it must be labelled CUSTOM HELD-OUT MOSE-v1 EVALUATION and never
reported as official validation performance.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

# Frozen. Changing any of these after seeing a method result invalidates the
# comparison, which is why they live here and are hashed into the manifest.
SPLIT_SEED = 20260917
SPLIT_FRACTIONS = {"train": 0.5, "calib": 0.2, "eval": 0.3}
MANIFEST_VERSION = 2          # v1 assumed official valid was scorable
LABEL = "custom held-out MOSE-v1 evaluation"


def source_group(video: str) -> str:
    """Videos from the same original source, kept together.

    Clips cut from one source share background, lighting and often the same
    objects, so splitting them across train and eval would leak. Where a
    common stem is identifiable (`name_01`, `name-clip2`, `name_part3`) it
    groups them; otherwise the video is its own group.
    """
    m = re.match(r"^(.*?)[._-](?:clip|part|seg|sub)?\d{1,3}$", video, re.I)
    return m.group(1) if m and len(m.group(1)) >= 3 else video


def build(eligible_videos, *, seed: int = SPLIT_SEED,
          fractions: dict | None = None) -> dict:
    """Assign GROUPS (not videos) to train / calib / eval, deterministically.

    Assigning groups rather than videos is what makes the split genuinely
    disjoint: two clips of one source can never land on opposite sides.
    """
    fractions = fractions or SPLIT_FRACTIONS
    groups: dict[str, list[str]] = {}
    for v in sorted(eligible_videos):
        groups.setdefault(source_group(v), []).append(v)

    keys = sorted(groups)
    rng = np.random.default_rng(seed)
    order = list(keys)
    rng.shuffle(order)

    n = len(order)
    n_tr = int(round(n * fractions["train"]))
    n_ca = int(round(n * fractions["calib"]))
    assign = {}
    for i, g in enumerate(order):
        split = ("train" if i < n_tr else
                 "calib" if i < n_tr + n_ca else "eval")
        for v in groups[g]:
            assign[v] = split

    by_split = {s: sorted(v for v, a in assign.items() if a == s)
                for s in ("train", "calib", "eval")}
    return {
        "version": MANIFEST_VERSION,
        "label": LABEL,
        "not_official_validation": (
            "These are MOSE-v1 official TRAINING videos held out by us. "
            "Results must never be reported as official validation "
            "performance."
        ),
        "seed": seed,
        "fractions": fractions,
        "unit": "source group (clips from one source stay together)",
        "groups_total": n,
        "videos_total": len(assign),
        "assignment": assign,
        "videos_by_split": by_split,
        "videos_per_split": {s: len(v) for s, v in by_split.items()},
        "groups_by_split": {
            s: sorted({source_group(v) for v in vs})
            for s, vs in by_split.items()},
    }


def check_disjoint(manifest: dict) -> dict:
    """No video, and no source group, may appear in two splits."""
    by = manifest["videos_by_split"]
    names = {s: set(v) for s, v in by.items()}
    overlaps = {}
    for a in ("train", "calib", "eval"):
        for b in ("train", "calib", "eval"):
            if a < b:
                shared = names[a] & names[b]
                if shared:
                    overlaps[f"{a}&{b}"] = sorted(shared)[:5]
    gs = {s: {source_group(v) for v in vs} for s, vs in by.items()}
    group_overlaps = {}
    for a in ("train", "calib", "eval"):
        for b in ("train", "calib", "eval"):
            if a < b:
                shared = gs[a] & gs[b]
                if shared:
                    group_overlaps[f"{a}&{b}"] = sorted(shared)[:5]
    return {
        "videos_disjoint": not overlaps,
        "groups_disjoint": not group_overlaps,
        "video_overlaps": overlaps,
        "group_overlaps": group_overlaps,
    }


def freeze(manifest: dict, path: Path) -> dict:
    """Write the manifest with its own hash. Refuses to overwrite silently."""
    path = Path(path)
    blob = json.dumps(manifest, indent=2, sort_keys=True, default=str)
    manifest = dict(manifest)
    manifest["self_sha256"] = hashlib.sha256(blob.encode()).hexdigest()
    if path.exists():
        existing = json.loads(path.read_text())
        if existing.get("self_sha256") != manifest["self_sha256"]:
            raise RuntimeError(
                f"{path} already holds a DIFFERENT frozen manifest "
                f"({str(existing.get('self_sha256'))[:16]} vs "
                f"{manifest['self_sha256'][:16]}). Refusing to overwrite: a "
                f"split changed after freezing invalidates every comparison "
                f"made against the old one. Delete it deliberately if you "
                f"intend to re-freeze."
            )
        return existing
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True,
                               default=str) + "\n")
    return manifest


def load(path: Path) -> dict:
    """Read the frozen manifest. The single source of split truth."""
    path = Path(path)
    if not path.exists():
        raise SystemExit(
            f"no frozen split manifest at {path}. Run the audit first; every "
            f"stage reads its splits from this file."
        )
    m = json.loads(path.read_text())
    if m.get("version") != MANIFEST_VERSION:
        raise SystemExit(
            f"{path} is manifest version {m.get('version')}, but this code "
            f"expects v{MANIFEST_VERSION}. The older split rests on a "
            f"different definition of which videos are scorable; re-freeze "
            f"deliberately rather than mixing them."
        )
    return m


def split_of(manifest: dict, video: str) -> str | None:
    return manifest.get("assignment", {}).get(video)
