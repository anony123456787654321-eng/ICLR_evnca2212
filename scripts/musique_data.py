#!/usr/bin/env python3
"""Restore and integrity-check the MuSiQue-Answerable v1.0 files.

data/raw/ is gitignored, so a machine that only pulls the repository has no
dataset and the sweep dies at its first load. MuSiQue is CC BY 4.0, which
permits redistribution with attribution, so the two audited files ship in this
repository xz-compressed under artifacts/musique/ (26.5 MB for 258 MB of
JSONL) and are decompressed on demand. See artifacts/musique/manifest.json for
the source and citation.

The expected digests are imported from analysis/musique_audit.py so there is a
single source of truth for what was audited.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.musique_audit import TRAIN_SHA256, DEV_SHA256

ROOT = Path("data/raw/musique/data")
ARCHIVE_DIR = Path("artifacts/musique")
MANIFEST = ARCHIVE_DIR / "manifest.json"
FILES = {"musique_ans_v1.0_train.jsonl": TRAIN_SHA256,
         "musique_ans_v1.0_dev.jsonl": DEV_SHA256}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def restore(root: Path, force: bool = False) -> int:
    """Decompress the shipped archives into root, verifying as we go."""
    if not MANIFEST.is_file():
        print(f"FATAL: missing {MANIFEST}", file=sys.stderr)
        return 2
    manifest = json.loads(MANIFEST.read_text())
    root.mkdir(parents=True, exist_ok=True)
    for name, expected in FILES.items():
        target = root / name
        if target.is_file() and not force and digest(target) == expected:
            print(f"present: {target}")
            continue
        archive = ARCHIVE_DIR / manifest["files"][name]["archive"]
        if not archive.is_file():
            print(f"FATAL: missing archive {archive}", file=sys.stderr)
            return 2
        found = digest(archive)
        if found != manifest["files"][name]["archive_sha256"]:
            print(f"FATAL: {archive} sha256 {found} != "
                  f"{manifest['files'][name]['archive_sha256']}", file=sys.stderr)
            return 2
        # Decompress via a temporary so an interrupted run cannot leave a
        # truncated file that later looks merely corrupt.
        tmp = target.with_suffix(target.suffix + ".partial")
        with lzma.open(archive, "rb") as src, open(tmp, "wb") as dst:
            shutil.copyfileobj(src, dst, 1 << 20)
        if (found := digest(tmp)) != expected:
            tmp.unlink(missing_ok=True)
            print(f"FATAL: {name} restored with sha256 {found} != {expected}",
                  file=sys.stderr)
            return 2
        tmp.replace(target)
        print(f"restored: {target} ({target.stat().st_size:,} bytes)")
    return 0


def verify(root: Path) -> int:
    problems = []
    for name, expected in FILES.items():
        path = root / name
        if not path.is_file():
            problems.append(f"MISSING {path}")
        elif (found := digest(path)) != expected:
            problems.append(f"CORRUPT {path}: sha256 {found} != {expected}")
    if problems:
        for line in problems:
            print(line, file=sys.stderr)
        print(f"\nRestore the audited MuSiQue-Answerable v1.0 files with:\n"
              f"  python scripts/musique_data.py restore --root {root}",
              file=sys.stderr)
        return 2
    print(f"ok: {len(FILES)} MuSiQue files present with audited sha256 under {root}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=("verify", "restore"))
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--force", action="store_true",
                    help="re-decompress even if the target already verifies")
    args = ap.parse_args()
    if args.command == "restore":
        return restore(Path(args.root), args.force)
    return verify(Path(args.root))


if __name__ == "__main__":
    raise SystemExit(main())
