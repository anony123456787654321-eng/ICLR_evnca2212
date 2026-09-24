"""Run manifests: commit, config, seeds, hardware, versions, output hashes."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from typing import Any, Dict


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unavailable"


def hardware() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
    }
    try:
        import torch  # noqa: F401
        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        info["cuda_device_count"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
        info["cuda_devices"] = [torch.cuda.get_device_name(i)
                                for i in range(info["cuda_device_count"])]
    except Exception:
        info["torch"] = None
    try:
        import numpy
        info["numpy"] = numpy.__version__
    except Exception:
        pass
    return info


def build_manifest(name: str, config: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "experiment": name,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": _git("status", "--porcelain") != "",
        "config": config,
        "hardware": hardware(),
    }


def write_manifest(path: str, manifest: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True, default=str)


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
