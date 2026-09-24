"""Outcome-independent MuSiQue population and leakage audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ecnca.real.musique import load_musique, normalise_answer


TRAIN_SHA256 = "83a75b1e11e4e9bb8f8308e72ac40ca617ae4431b3a0d955b61cab259248490a"
DEV_SHA256 = "15fa63794d18a94ce12411aca6e2327e65b6e83b0b1490efab3f1962e48abf3b"


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def describe(records):
    linear = [r for r in records if r.is_linear]
    failures = Counter(reason for r in records if not r.is_linear
                       for reason in r.linear_failure.split(";") if reason)
    return {"records": len(records), "linear_records": len(linear),
            "linear_fraction": len(linear) / max(len(records), 1),
            "hops_all": dict(sorted(Counter(r.n_hops for r in records).items())),
            "hops_linear": dict(sorted(Counter(r.n_hops for r in linear).items())),
            "top_linear_failures": failures.most_common(20)}


def build_report(train_path, dev_path):
    train, dev = load_musique(train_path), load_musique(dev_path)
    train_ids, dev_ids = {r.record_id for r in train}, {r.record_id for r in dev}
    train_questions = {normalise_answer(r.question) for r in train}
    dev_questions = {normalise_answer(r.question) for r in dev}
    return {"dataset": "MuSiQue-Answerable v1.0", "license": "CC BY 4.0",
            "files": {"train": {"sha256": digest(train_path),
                                  "expected_sha256": TRAIN_SHA256},
                      "dev": {"sha256": digest(dev_path),
                              "expected_sha256": DEV_SHA256}},
            "train": describe(train), "dev": describe(dev),
            "leakage": {"shared_ids": len(train_ids & dev_ids),
                        "shared_normalised_questions": len(train_questions & dev_questions)},
            "ready": (digest(train_path) == TRAIN_SHA256 and
                      digest(dev_path) == DEV_SHA256 and not (train_ids & dev_ids))}


def main():
    ap = argparse.ArgumentParser()
    root = "data/raw/musique/data"
    ap.add_argument("--train", default=f"{root}/musique_ans_v1.0_train.jsonl")
    ap.add_argument("--dev", default=f"{root}/musique_ans_v1.0_dev.jsonl")
    ap.add_argument("--out", default="results/musique/audit.json")
    args = ap.parse_args()
    report = build_report(args.train, args.dev)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
