"""How the ledger's reader context compares with keeping one passage per source.

In the syndication experiment every arrival is version zero, so the ledger's
max-register join has no refinement to prefer and keeps, for each source
identifier, the copy with the largest payload hash. Grouping by source
identifier and keeping the first arrival is the obvious simpler baseline. This
script builds both contexts for every condition, with no reader loaded, and
reports the fraction of records whose contexts are byte-identical.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ecnca.real.musique import load_musique, split_linear  # noqa: E402
from ecnca.real.musique_duplication import resolve_stream  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "synd", ROOT / "experiments" / "musique_syndication_eval.py")
synd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(synd)


def first_per_source(events):
    kept, _, _ = resolve_stream(events, "first")
    held = {}
    for e in kept:
        held.setdefault(e.root_id, e)
    order = []
    for e in events:
        if e.root_id not in order:
            order.append(e.root_id)
    return [held[r].paragraph for r in order if r in held]


def main():
    os.chdir(ROOT)
    records = [r for r in split_linear(load_musique(
        "data/raw/musique/data/musique_ans_v1.0_dev.jsonl")) if r.n_hops == 2]
    items = synd.build_items(records, 300)
    out = {}
    for pos in ("last", "first"):
        for style in ("exact", "boilerplate"):
            for m in (1, 2, 4, 8, 16):
                same = 0
                for it in items:
                    ev = synd.syndication_stream(it, m, style, pos)
                    ledger, _, _ = synd.reader_context(ev, "full")
                    same += ledger == first_per_source(ev)
                out[f"{pos}/{style}/{m}"] = same / len(items)
    dest = ROOT / "results" / "musique_syndication" / "group_by_id_identity.json"
    dest.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
