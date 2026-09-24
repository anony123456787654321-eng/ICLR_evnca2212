"""Shared experiment harness: args, resumable CSV, manifests, aggregation."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Dict, Iterable, List, Optional, Set

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ecnca.manifest import build_manifest, write_manifest  # noqa: E402


def base_parser(name: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(name)
    p.add_argument("--out", default=f"results/{name}")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--n-cells", type=int, default=16)
    p.add_argument("--n-unique", type=int, default=12)
    p.add_argument("--dim", type=int, default=4)
    p.add_argument("--capacity", type=int, default=32)
    p.add_argument("--dry-run", action="store_true", help="tiny sweep, two configs per axis")
    p.add_argument("--resume", choices=["auto", "off"], default="auto")
    return p


class ResumableCSV:
    """Append-only CSV that remembers which configurations already finished."""

    def __init__(self, path: str, key_fields: List[str], resume: bool = True):
        self.path, self.key_fields = path, key_fields
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.done: Set[tuple] = set()
        self.fieldnames: Optional[List[str]] = None
        if resume and os.path.exists(path):
            with open(path) as fh:
                for row in csv.DictReader(fh):
                    self.done.add(tuple(str(row.get(k, "")) for k in key_fields))
                    self.fieldnames = self.fieldnames or list(row)
        elif os.path.exists(path):
            os.remove(path)

    def key(self, row: Dict) -> tuple:
        return tuple(str(row.get(k, "")) for k in self.key_fields)

    def is_done(self, row: Dict) -> bool:
        return self.key(row) in self.done

    def write(self, row: Dict) -> None:
        new = not os.path.exists(self.path)
        if self.fieldnames is None:
            self.fieldnames = list(row)
        with open(self.path, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=self.fieldnames, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow(row)
        self.done.add(self.key(row))


def start_run(name: str, args) -> str:
    os.makedirs(args.out, exist_ok=True)
    write_manifest(os.path.join(args.out, "manifest.json"),
                   build_manifest(name, vars(args)))
    return args.out


def summarise(rows: List[Dict], group: Iterable[str], metrics: Iterable[str]) -> List[Dict]:
    """Mean / std / n over seeds for each group, plus a bootstrap 95% CI."""
    import collections
    buckets = collections.defaultdict(list)
    for r in rows:
        buckets[tuple(str(r[g]) for g in group)].append(r)
    out = []
    for key, rs in sorted(buckets.items()):
        rec = dict(zip(group, key))
        rec["n_seeds"] = len(rs)
        for m in metrics:
            vals = np.array([float(r[m]) for r in rs if r.get(m) not in (None, "")], dtype=float)
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                rec[f"{m}_mean"] = float("nan")
                continue
            rec[f"{m}_mean"] = float(vals.mean())
            rec[f"{m}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            if len(vals) > 1:
                rng = np.random.default_rng(0)
                boot = np.array([rng.choice(vals, len(vals), replace=True).mean()
                                 for _ in range(2000)])
                rec[f"{m}_lo"], rec[f"{m}_hi"] = map(float, np.percentile(boot, [2.5, 97.5]))
        out.append(rec)
    return out


def write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True, default=float)


def write_rows(path: str, rows: List[Dict]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fields: List[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
