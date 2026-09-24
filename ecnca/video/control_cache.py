"""Per-event result cache, versioned PER CONFIGURATION.

The first cache mixed one `ablation_v2` token into every configuration's key,
so correcting the ablations invalidated the 93 completed `normal` results too
-- an hour of GPU time recomputed for a configuration that had not changed.

Two rules follow:

  * a configuration's key contains only what affects THAT configuration, plus
    its own implementation version;
  * reuse is verified against the recorded provenance (checkpoint, event
    identity, preprocessing, scoring and configuration), never inferred from
    a filename.

Superseded entries are archived by their recorded `impl_version`, not by
control name: a name can be reused across incompatible implementations, and a
rename would silently orphan results rather than invalidate them.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, asdict, field
from pathlib import Path

# Implementation version PER CONFIGURATION. Bump only the one that changed.
#   normal            1 -- never modified
#   disrupted_content 1 -- never modified
#   no_long_term_read 2 -- replaced enable_long_term=False (unbounded memory)
#   recent_only       2 -- bound was previously never enforced
CONTROL_IMPL_VERSION = {
    "normal": 1,
    "disrupted_content": 1,
    "no_long_term_read": 2,
    "recent_only": 2,
    # Superseded names, recorded so their artifacts can be recognised.
    "no_long_term": 1,
}

# Version of the scoring path itself. A change here invalidates every
# configuration, because the numbers would no longer be comparable.
SCORING_VERSION = 1

# THE preprocessing identity, imported from the module that defines the
# transform rather than restated here. A second copy of this number is how
# the corrected preprocessing (version 2) failed to invalidate the cached
# per-event results: the transform changed, this constant did not, and every
# cached score computed from differently-scaled inputs stayed reusable.
from ecnca.video.preprocess import PREPROCESS_VERSION as PREPROCESSING_VERSION


@dataclass
class EventKey:
    """Everything that could change this event's number."""

    mode: str
    video: str
    object_id: int
    reappear_frame: int
    checkpoint_sha: str
    config_sha: str
    manifest_sha: str
    min_history: int
    impl_version: int
    scoring_version: int = SCORING_VERSION
    preprocessing_version: int = PREPROCESSING_VERSION

    def digest(self) -> str:
        return hashlib.sha256(
            json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()

    def filename(self) -> str:
        return self.digest()[:24] + ".json"

    def to_dict(self) -> dict:
        return asdict(self)


def config_sha(xmem_config: dict, mode: str) -> str:
    """Hash only the settings that reach the model for this control."""
    keys = ("enable_long_term", "enable_long_term_count_usage",
            "max_mid_term_frames", "min_mid_term_frames", "num_prototypes",
            "max_long_term_elements", "top_k", "mem_every",
            "deep_update_every", "key_dim", "value_dim", "hidden_dim")
    sub = {k: xmem_config.get(k) for k in keys}
    return hashlib.sha256(
        json.dumps(sub, sort_keys=True).encode()).hexdigest()[:16]


def make_key(*, mode, event, checkpoint_sha, xmem_config, manifest_sha,
             min_history) -> EventKey:
    return EventKey(
        mode=mode,
        video=str(event["video"]),
        object_id=int(event["object_id"]),
        reappear_frame=int(event["reappear_frame"]),
        checkpoint_sha=checkpoint_sha[:16],
        config_sha=config_sha(xmem_config, mode),
        manifest_sha=str(manifest_sha)[:16],
        min_history=int(min_history),
        impl_version=CONTROL_IMPL_VERSION.get(mode, 1),
    )


@dataclass
class CacheStats:
    hits: int = 0
    computed: int = 0
    remaining: int = 0
    migrated: int = 0
    rejected: int = 0
    reasons: dict = field(default_factory=dict)

    def note(self, reason: str) -> None:
        self.reasons[reason] = self.reasons.get(reason, 0) + 1

    def to_dict(self) -> dict:
        return asdict(self)


class ControlCache:
    """Reads and writes per-event results, verifying provenance on read."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.stats = CacheStats()

    def _path(self, key: EventKey) -> Path:
        return self.root / key.mode / key.filename()

    def load(self, key: EventKey):
        """Return a cached result, or None with the reason recorded.

        Compatibility is checked against the RECORDED key, so a matching
        filename with different provenance is rejected rather than trusted.
        """
        p = self._path(key)
        if not p.exists():
            return None
        try:
            blob = json.loads(p.read_text())
        except Exception:
            self.stats.rejected += 1
            self.stats.note("unreadable")
            return None
        recorded = blob.get("key")
        if not recorded:
            self.stats.rejected += 1
            self.stats.note("no recorded provenance")
            return None
        want = key.to_dict()
        mismatched = [k for k, v in want.items() if recorded.get(k) != v]
        if mismatched:
            self.stats.rejected += 1
            self.stats.note(f"provenance differs: {','.join(sorted(mismatched))}")
            return None
        self.stats.hits += 1
        return blob

    def save(self, key: EventKey, score, observation, memory) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps({
            "key": key.to_dict(),
            "score": score,
            "observation": observation,
            "memory": memory,
        }, indent=2, default=str) + "\n")
        tmp.replace(p)                      # atomic
        self.stats.computed += 1

    def migrate_from(self, legacy_dirs, *, keys_by_event) -> int:
        """Adopt compatible results from an older cache layout.

        A legacy entry is adopted only when every provenance field it records
        matches the key we would compute today. Filenames are never trusted.
        """
        adopted = 0
        for legacy in legacy_dirs:
            legacy = Path(legacy)
            if not legacy.is_dir():
                continue
            for f in sorted(legacy.glob("*.json")):
                try:
                    blob = json.loads(f.read_text())
                except Exception:
                    continue
                obs = blob.get("observation") or {}
                mode = obs.get("configuration")
                if mode is None:
                    continue
                key = keys_by_event.get((mode, blob.get("event_id")))
                if key is None:
                    continue
                if CONTROL_IMPL_VERSION.get(mode, 1) != key.impl_version:
                    continue
                dest = self._path(key)
                if dest.exists():
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                tmp = dest.with_suffix(".tmp")
                tmp.write_text(json.dumps({
                    "key": key.to_dict(),
                    "score": blob.get("score"),
                    "observation": obs,
                    "memory": blob.get("memory", {}),
                    "migrated_from": str(f),
                }, indent=2, default=str) + "\n")
                tmp.replace(dest)
                adopted += 1
        self.stats.migrated += adopted
        return adopted


def archive_superseded(root: Path, archive: Path, *, dry_run: bool = False):
    """Archive entries whose recorded impl_version is behind the current one.

    Keyed on the RECORDED version, not on a control name: a name can be reused
    across incompatible implementations, and renaming would otherwise orphan
    results rather than invalidate them.
    """
    root, archive = Path(root), Path(archive)
    moved, kept, entries = 0, 0, []
    for f in sorted(root.rglob("*.json")):
        try:
            blob = json.loads(f.read_text())
        except Exception:
            continue
        key = blob.get("key") or {}
        mode = key.get("mode") or (blob.get("observation") or {}).get(
            "configuration")
        recorded = key.get("impl_version")
        current = CONTROL_IMPL_VERSION.get(mode)
        if mode is None or current is None:
            kept += 1
            continue
        if recorded is None or recorded < current:
            entries.append({"file": str(f.relative_to(root)), "mode": mode,
                            "recorded_impl_version": recorded,
                            "current_impl_version": current})
            if not dry_run:
                dest = archive / f.relative_to(root)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(f), str(dest))
            moved += 1
        else:
            kept += 1
    return {"archived": moved, "kept": kept, "entries": entries}
