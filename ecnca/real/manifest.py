"""Dataset release manifest: exactly which files produced a result."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build(data_dir: str, repo: str = "chenxwh/AVeriTeC",
          repo_sha: Optional[str] = None, url: Optional[str] = None,
          knowledge_store_dir: Optional[str] = None,
          notes: str = "") -> Dict[str, Any]:
    files: Dict[str, Any] = {}
    for name in sorted(os.listdir(data_dir)):
        if not name.endswith(".json"):
            continue
        path = os.path.join(data_dir, name)
        with open(path) as fh:
            data = json.load(fh)
        files[name] = {"sha256": sha256(path), "bytes": os.path.getsize(path),
                       "n_records": len(data) if isinstance(data, list) else None}
    ks: Dict[str, Any] = {"present": False}
    if knowledge_store_dir and os.path.isdir(knowledge_store_dir):
        entries = sorted(os.listdir(knowledge_store_dir))
        ks = {"present": True, "n_files": len(entries),
              "bytes": sum(os.path.getsize(os.path.join(knowledge_store_dir, e))
                           for e in entries if os.path.isfile(
                               os.path.join(knowledge_store_dir, e)))}
    return {
        "repository": repo,
        "repository_commit": repo_sha,
        "download_url": url or f"https://huggingface.co/{repo}",
        "downloaded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "license": "CC BY-NC 4.0",
        "files": files,
        "split_counts": {k.replace(".json", ""): v["n_records"] for k, v in files.items()},
        "knowledge_store": ks,
        "evaluation_protocol": (
            "official AVeriTeC evaluator (src/prediction/evaluate_veracity.py); "
            "official AVeriTeC score plus verdict accuracy and evidence score"),
        "notes": notes,
    }


def write(path: str, manifest: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
