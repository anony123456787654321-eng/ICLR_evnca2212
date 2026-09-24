"""MOSE v1 acquisition: validate, extract safely, and check the dataset.

An arbitrary file in the drop directory must never make MOSE "present". The
previous check globbed for any file, so a stray .DS_Store would have reported
success and the pipeline would have failed much later with a confusing error.

What "present" requires here:
  1. The expected archives exist and are real archives, not HTML error pages.
  2. Extraction is SAFE -- no member escapes the destination directory.
  3. The extracted tree has the official layout, with both splits found.
  4. Images and masks align: same videos, same frame stems, same dimensions.
"""

from __future__ import annotations

import hashlib
import json
import tarfile
import zipfile
from dataclasses import dataclass, field, asdict
from pathlib import Path

# MOSE v1 ships a train and a valid split. The event experiment needs both:
# training videos to train on and held-out videos to evaluate on.
REQUIRED_SPLITS = ("train", "valid")
IMAGE_DIRNAMES = ("JPEGImages", "JPEGImages_full", "images")
MASK_DIRNAMES = ("Annotations", "annotations", "masks")
MIN_VIDEOS_PER_SPLIT = 1


def _is_sidecar(path) -> bool:
    """macOS AppleDouble / metadata companions, not dataset content.

    Archives written or repacked on macOS carry `._name` resource forks and
    `__MACOSX/` directories. They sit beside real frames and match a *.png
    glob, so handing them to an image loader raises UnidentifiedImageError on
    a dataset that is perfectly fine. They are skipped, and counted, rather
    than mistaken for corruption.
    """
    from pathlib import Path as _P

    p = _P(path)
    return p.name.startswith("._") or "__MACOSX" in p.parts


def _frames(directory) -> list:
    """Real annotation frames in a video directory, sidecars excluded."""
    from pathlib import Path as _P

    return sorted(f for f in _P(directory).glob("*.png") if not _is_sidecar(f))


class MoseInvalid(RuntimeError):
    """MOSE is present but not usable, with a specific reason."""


@dataclass
class SplitLayout:
    name: str
    images: str = ""
    annotations: str = ""
    videos: int = 0
    annotated_videos: int = 0
    problems: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _is_archive(path: Path) -> str | None:
    """The archive KIND, from content rather than from the file name."""
    try:
        if zipfile.is_zipfile(path):
            return "zip"
        if tarfile.is_tarfile(path):
            return "tar"
    except Exception:
        return None
    return None


def classify(drop_dir: Path) -> dict:
    """What is actually in the drop directory?

    Distinguishes archives from stray files, so 'a file exists' can never be
    mistaken for 'the dataset is here'.
    """
    drop_dir = Path(drop_dir)
    archives, extracted, stray = [], [], []
    if drop_dir.exists():
        for p in sorted(drop_dir.iterdir()):
            if p.is_dir():
                extracted.append(p.name)
            elif _is_archive(p):
                archives.append({"name": p.name, "kind": _is_archive(p),
                                 "bytes": p.stat().st_size})
            else:
                stray.append({"name": p.name, "bytes": p.stat().st_size,
                              "reason": "not an archive and not a directory"})
    return {"archives": archives, "extracted_dirs": extracted,
            "ignored_files": stray, "drop_dir": str(drop_dir)}


def safe_extract(archive: Path, dest: Path) -> dict:
    """Extract, refusing any member that would escape `dest`.

    A crafted archive can otherwise write outside the destination through
    absolute paths, `..` components, or links.
    """
    archive, dest = Path(archive), Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    root = dest.resolve()
    kind = _is_archive(archive)
    if kind is None:
        raise MoseInvalid(
            f"{archive.name} is not a zip or tar archive. A truncated "
            f"download or an HTML error page saved under an archive name "
            f"would look like this."
        )

    def unsafe(member_name: str) -> bool:
        target = (root / member_name).resolve()
        return not str(target).startswith(str(root))

    rejected, count = [], 0
    if kind == "zip":
        with zipfile.ZipFile(archive) as z:
            for info in z.infolist():
                if unsafe(info.filename):
                    rejected.append(info.filename)
                    continue
                z.extract(info, root)
                count += 1
    else:
        with tarfile.open(archive) as t:
            for member in t.getmembers():
                if unsafe(member.name) or member.issym() or member.islnk():
                    rejected.append(member.name)
                    continue
                # `filter="data"` is the hardened extraction policy; passing
                # it explicitly keeps behaviour identical across Python
                # versions rather than changing under us at 3.14.
                try:
                    t.extract(member, root, filter="data")
                except TypeError:          # Python < 3.11.4
                    t.extract(member, root)
                count += 1
    if rejected:
        raise MoseInvalid(
            f"{archive.name} contains {len(rejected)} member(s) that would "
            f"write outside {dest}: {rejected[:3]}. Refusing to extract it."
        )
    return {"archive": archive.name, "kind": kind, "members_extracted": count,
            "destination": str(dest)}


def _find(base: Path, names) -> Path | None:
    for n in names:
        p = base / n
        if p.is_dir():
            return p
    return None


def locate_splits(root: Path) -> dict[str, SplitLayout]:
    """Find each official split's image and annotation directories.

    Searches for BOTH splits rather than returning the first match. The loader
    previously returned on the first layout that existed, so once `train/` was
    present it never looked at `valid/` -- and the evaluation split would have
    been drawn from training videos.
    """
    root = Path(root)
    out: dict[str, SplitLayout] = {}
    for split in REQUIRED_SPLITS:
        layout = SplitLayout(name=split)
        # The split directory may sit at the top level or one level down,
        # depending on how the archives were packed.
        candidates = [root / split]
        candidates += [p / split for p in root.iterdir() if p.is_dir()] \
            if root.exists() else []
        base = next((c for c in candidates if c.is_dir()), None)
        if base is None:
            layout.problems.append(f"no '{split}' directory under {root}")
            out[split] = layout
            continue
        img = _find(base, IMAGE_DIRNAMES)
        ann = _find(base, MASK_DIRNAMES)
        if img is None:
            layout.problems.append(
                f"no image directory in {base} (looked for {IMAGE_DIRNAMES})")
        if ann is None:
            layout.problems.append(
                f"no annotation directory in {base} "
                f"(looked for {MASK_DIRNAMES})")
        layout.images = str(img or "")
        layout.annotations = str(ann or "")
        if img is not None:
            layout.videos = sum(1 for p in img.iterdir()
                                if p.is_dir() and not _is_sidecar(p))
        if ann is not None:
            layout.annotated_videos = sum(1 for p in ann.iterdir()
                                          if p.is_dir() and not _is_sidecar(p))
        out[split] = layout
    return out


def check_alignment(layout: SplitLayout, *, sample_videos: int = 5) -> dict:
    """Do images and masks describe the same frames at the same size?

    A split whose masks do not line up with its images produces silent
    garbage, so this is checked before the dataset is called usable.
    """
    if not layout.images or not layout.annotations:
        return {"checked": False,
                "reason": "the split has no image or annotation directory"}
    img_root, ann_root = Path(layout.images), Path(layout.annotations)
    img_videos = {p.name for p in img_root.iterdir()
                  if p.is_dir() and not _is_sidecar(p)}
    ann_videos = {p.name for p in ann_root.iterdir()
                  if p.is_dir() and not _is_sidecar(p)}

    problems = []
    missing_masks = sorted(img_videos - ann_videos)[:5]
    missing_images = sorted(ann_videos - img_videos)[:5]
    if missing_masks:
        problems.append(f"videos with images but no annotations: {missing_masks}")
    if missing_images:
        problems.append(f"videos with annotations but no images: {missing_images}")

    shared = sorted(img_videos & ann_videos)
    checked, mismatched = 0, []
    try:
        from PIL import Image
    except Exception:
        return {"checked": False, "reason": "PIL is unavailable"}

    for video in shared[:sample_videos]:
        frames = _frames(ann_root / video)
        if not frames:
            problems.append(f"{video}: no .png annotations")
            continue
        for f in frames[:3]:
            jpg = img_root / video / (f.stem + ".jpg")
            if not jpg.exists():
                mismatched.append(f"{video}/{f.stem}: no matching image")
                continue
            with Image.open(f) as m, Image.open(jpg) as i:
                if m.size != i.size:
                    mismatched.append(
                        f"{video}/{f.stem}: mask {m.size} vs image {i.size}")
            checked += 1
    if mismatched:
        problems.append(f"image/mask mismatches: {mismatched[:3]}")
    sidecars = sum(1 for v in shared[:sample_videos]
                   for f in (ann_root / v).glob("*.png") if _is_sidecar(f))
    return {
        "checked": True,
        "videos_with_both": len(shared),
        "frame_pairs_checked": checked,
        "sidecar_files_skipped": sidecars,
        "problems": problems,
        "aligned": not problems and checked > 0,
    }


def verify_publisher_checksums(drop_dir: Path) -> dict:
    """Check the archives against the maintainers' OWN published hashes.

    Stronger than a hash we record after the fact: a mismatch means the
    download is wrong or the artifact changed. The shipped SHA256SUMS has
    Windows line endings and names a test.tar.gz we do not use, so both are
    handled rather than left to fail confusingly.
    """
    from ecnca.video.acquisition import MOSE_V1_SHA256

    drop_dir = Path(drop_dir)
    out = {"source": "publisher", "results": {}}
    for name, want in MOSE_V1_SHA256.items():
        f = drop_dir / name
        if not f.exists():
            out["results"][name] = {"present": False}
            continue
        h = hashlib.sha256()
        with f.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 22), b""):
                h.update(chunk)
        got = h.hexdigest()
        out["results"][name] = {
            "present": True, "sha256": got, "expected": want,
            "verified": got == want, "bytes": f.stat().st_size,
        }
    checked = [r for r in out["results"].values() if r.get("present")]
    out["all_verified"] = bool(checked) and all(r["verified"] for r in checked)
    return out


def validate(drop_dir: Path, *, extract_to: Path | None = None) -> dict:
    """The full gate. Returns a report; raises MoseInvalid with a reason."""
    drop_dir = Path(drop_dir)
    extract_to = Path(extract_to or drop_dir / "extracted")
    report: dict = {"drop_dir": str(drop_dir), "root": str(extract_to)}

    found = classify(drop_dir)
    report["found"] = found
    checks = verify_publisher_checksums(drop_dir)
    report["publisher_checksums"] = checks
    bad = [n for n, r in checks["results"].items()
           if r.get("present") and not r["verified"]]
    if bad:
        raise MoseInvalid(
            f"publisher checksum MISMATCH for {bad}. The download is "
            f"corrupt or truncated; re-fetch before extracting."
        )
    if not found["archives"] and not found["extracted_dirs"]:
        raise MoseInvalid(
            f"nothing usable in {drop_dir}. "
            f"{len(found['ignored_files'])} file(s) present but none is an "
            f"archive or an extracted directory."
        )

    report["extractions"] = []
    for a in found["archives"]:
        marker = extract_to / f".extracted.{a['name']}"
        if marker.exists():
            report["extractions"].append({"archive": a["name"],
                                          "status": "already extracted"})
            continue
        rec = safe_extract(drop_dir / a["name"], extract_to)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps(rec) + "\n")
        report["extractions"].append(rec)

    # If the operator extracted by hand, the tree may be in the drop dir.
    search_roots = [extract_to, drop_dir]
    splits, chosen = None, None
    for r in search_roots:
        if not r.exists():
            continue
        cand = locate_splits(r)
        if all(not cand[s].problems for s in REQUIRED_SPLITS):
            splits, chosen = cand, r
            break
        if splits is None:
            splits, chosen = cand, r
    report["root"] = str(chosen)
    report["splits"] = {k: v.to_dict() for k, v in splits.items()}

    missing = [s for s in REQUIRED_SPLITS if splits[s].problems]
    if missing:
        detail = {s: splits[s].problems for s in missing}
        raise MoseInvalid(
            f"MOSE is incomplete: {detail}. Both the train and valid splits "
            f"are required -- without valid/, evaluation videos would be "
            f"drawn from the training split."
        )
    for s in REQUIRED_SPLITS:
        if splits[s].annotated_videos < MIN_VIDEOS_PER_SPLIT:
            raise MoseInvalid(
                f"split {s!r} has {splits[s].annotated_videos} annotated "
                f"videos; the archives may be truncated.")

    report["alignment"] = {s: check_alignment(splits[s])
                           for s in REQUIRED_SPLITS}
    bad = [s for s, a in report["alignment"].items() if not a.get("aligned")]
    if bad:
        raise MoseInvalid(
            f"image/mask alignment failed for {bad}: "
            f"{ {s: report['alignment'][s]['problems'] for s in bad} }")

    report["usable"] = True
    report["summary"] = {
        s: {"videos": splits[s].annotated_videos,
            "images": splits[s].images, "annotations": splits[s].annotations}
        for s in REQUIRED_SPLITS
    }
    return report
