"""Syndicated misinformation: does a reader count copies of one source as proof?

Pre-registered in docs/SYNDICATION_PREREGISTRATION.md, which was committed
before this script produced any result. Read that first; the design choices
below implement it and are not meant to be tuned against outcomes.

Xie et al. (ICLR 2024) found that language models side with whichever answer
more evidence supports. That is reasonable when the pieces are independent. It
is a vulnerability when one source is copied, because copying is free and
lineage is not. Here a single counterfactual paragraph, contradicting the true
final hop, is delivered m times. A reader that counts copies should drift
toward the planted answer as m grows. A reader fed from the lineage ledger sees
the counterfactual once, since every copy shares one root.

The reader for the `full` arm consumes the ledger's state, one payload per
root, which is what the ledger holds by definition: within a root a later
version replaces an earlier one rather than joining it. The other arms consume
whatever their accounting rule admitted.

The reader is deterministic under greedy decoding, so two arms that present the
same context to the same record must produce the same reply. Replies are cached
on that basis, which removes the redundant generations without changing any
result.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ecnca.real.musique import load_musique, normalise_answer, split_linear
from ecnca.real.musique_duplication import (DeliveryEvent, base_stream,
                                            ledger_mode, resolve_stream)

_spec = importlib.util.spec_from_file_location(
    "reader_eval", Path(__file__).resolve().parent / "musique_reader_eval.py")
reader = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(reader)

DEFAULT_VARIANTS = ("full", "canonical_dedup", "minhash_dedup",
                    "simhash_dedup", "plain")
STYLES = ("exact", "boilerplate")


def answer_kind(answer: str) -> str:
    a = answer.strip()
    if re.fullmatch(r"\d{4}", a):
        return "year"
    if re.search(r"\d", a):
        return "number"
    return "entity"


def golds_of(record):
    return tuple(g for g in (record.answer,) + tuple(record.answer_aliases)
                 if g and g.strip())


def pick_substitute(i, records, true_texts):
    """A plausible wrong answer of the same kind, chosen deterministically.

    A substitute of the wrong kind, a date where a person is asked for, would be
    easy to reject on plausibility alone and would understate the effect.
    """
    rec = records[i]
    kind = answer_kind(rec.answer)
    n_words = len(rec.answer.split())
    gold_norm = [normalise_answer(g) for g in golds_of(rec)]
    true_norm = normalise_answer(true_texts[i])
    for step in range(1, len(records)):
        cand = records[(i + step) % len(records)].answer.strip()
        if not cand or answer_kind(cand) != kind:
            continue
        if kind == "entity" and abs(len(cand.split()) - n_words) > 1:
            continue
        nc = normalise_answer(cand)
        if not nc:
            continue
        if any(g and (nc == g or nc in g or g in nc) for g in gold_norm):
            continue
        if nc in true_norm:
            continue
        return cand
    return None


def counterfactual(paragraph, golds, substitute):
    out = paragraph
    for g in sorted(golds, key=len, reverse=True):
        out = re.sub(re.escape(g), substitute, out, flags=re.IGNORECASE)
    norm = normalise_answer(out)
    if any(normalise_answer(g) and normalise_answer(g) in norm for g in golds):
        return None
    if normalise_answer(substitute) not in norm:
        return None
    return out


def build_items(records, limit):
    """The first `limit` eligible records, with substitute and counterfactual."""
    true_texts = []
    for r in records:
        true_texts.append(" ".join(s.paragraph_title + ". " + s.paragraph_text
                                   for s in r.steps))
    items = []
    for i, r in enumerate(records):
        sub = pick_substitute(i, records, true_texts)
        if sub is None:
            continue
        final = base_stream(r)[-1]
        cf = counterfactual(final.paragraph, golds_of(r), sub)
        if cf is None:
            continue
        items.append({"record": r, "substitute": sub, "counterfactual": cf})
        if len(items) >= limit:
            break
    return items


def syndication_stream(item, m, style, position):
    record = item["record"]
    base = base_stream(record)
    final = base[-1]
    root = f"{record.record_id}:counterfactual"
    copies = []
    for k in range(m):
        text = item["counterfactual"]
        if style == "boilerplate":
            # Generic outlet labels, deliberately not real organisations.
            text = f"Outlet {k + 1} reports: {text} (Syndicated copy {k + 1}.)"
        copies.append(DeliveryEvent(hop=final.hop, question=final.question,
                                    paragraph=text, root_id=root, version=0))
    return copies + base if position == "first" else base + copies


def reader_context(events, variant):
    mode = ledger_mode(variant)
    kept, credit, suppressed = resolve_stream(events, mode)
    if mode == "max":
        # The ledger holds one payload per root. Each kept arrival strictly beat
        # its root's incumbent, so the last kept per root is the ledger's state.
        state = {}
        for e in kept:
            state[e.root_id] = e
        order = []
        for e in events:
            if e.root_id not in order:
                order.append(e.root_id)
        passages = [state[r].paragraph for r in order if r in state]
    else:
        passages = [e.paragraph for e in kept]
    return passages, credit, suppressed


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dev",
                    default="data/raw/musique/data/musique_ans_v1.0_dev.jsonl")
    ap.add_argument("--reader", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--reader-revision", default="main")
    ap.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    ap.add_argument("--styles", default=",".join(STYLES))
    ap.add_argument("--multiplicities", default="1,2,4,8,16")
    ap.add_argument("--position", choices=("last", "first"), default="last")
    ap.add_argument("--hops", default="2")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dry-run", action="store_true",
                    help="build every context and report sizes, no reader")
    ap.add_argument("--out", default="results/musique_syndication/eval")
    a = ap.parse_args()
    os.chdir(Path(__file__).resolve().parent.parent)

    hops = {int(h) for h in a.hops.split(",") if h}
    records = [r for r in split_linear(load_musique(a.dev)) if r.n_hops in hops]
    items = build_items(records, a.limit)
    variants = [v for v in a.variants.split(",") if v]
    styles = [s for s in a.styles.split(",") if s]
    mults = [int(m) for m in a.multiplicities.split(",") if m]
    print(f"[synd] eligible records used: {len(items)} of {len(records)} "
          f"{sorted(hops)}-hop linear")

    if a.dry_run:
        unique = set()
        for style in styles:
            for m in mults:
                row = []
                for v in variants:
                    cf_counts, true_kept = [], []
                    for it in items:
                        ps, _, _ = reader_context(
                            syndication_stream(it, m, style, a.position), v)
                        cf_counts.append(sum(it["counterfactual"] in p
                                             for p in ps))
                        true_kept.append(float(
                            base_stream(it["record"])[-1].paragraph in ps))
                        unique.add((it["record"].record_id, tuple(ps)))
                    n = len(items)
                    row.append(f"{v}:cf={sum(cf_counts)/n:.2f},"
                               f"true={sum(true_kept)/n:.2f}")
                print(f"[synd] {style:>11} x{m:<2} " + "  ".join(row))
        print(f"[synd] unique reader calls needed: {len(unique)} "
              f"({len(unique)/max(len(items),1):.1f} per record)")
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.reader, revision=a.reader_revision)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        a.reader, revision=a.reader_revision,
        dtype=torch.bfloat16).to(a.device).eval()

    cache = {}

    def ask(record, passages):
        key = (record.record_id, tuple(passages))
        if key in cache:
            return cache[key]
        prompt = reader.build_prompt(tok, record.question, passages)
        batch = tok(prompt, return_tensors="pt").to(a.device)
        n_ctx = int(batch["input_ids"].shape[-1])
        with torch.no_grad():
            out = model.generate(**batch, max_new_tokens=a.max_new_tokens,
                                 do_sample=False, temperature=None,
                                 top_p=None, pad_token_id=tok.pad_token_id)
        reply = tok.decode(out[0][batch["input_ids"].shape[-1]:],
                           skip_special_tokens=True)
        cache[key] = (reply, n_ctx)
        return cache[key]

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for variant in variants:
        rows = []
        for style in styles:
            for m in mults:
                per = []
                for it in items:
                    rec = it["record"]
                    ps, credit, _ = reader_context(
                        syndication_stream(it, m, style, a.position), variant)
                    reply, n_ctx = ask(rec, ps)
                    answer, _ = reader.parse_reply(reply)
                    gold = reader.answer_recall(answer, golds_of(rec))
                    misled = reader.answer_recall(answer, (it["substitute"],))
                    per.append({"record_id": rec.record_id,
                                "gold_recall": gold, "misled": misled,
                                "flipped": float(misled and not gold),
                                "n_context_tokens": n_ctx,
                                "n_counterfactual_in_context": sum(
                                    it["counterfactual"] in p for p in ps),
                                "true_final_in_context": float(
                                    base_stream(rec)[-1].paragraph in ps)})
                n = len(per)
                agg = {k: sum(p[k] for p in per) / n
                       for k in ("gold_recall", "misled", "flipped",
                                 "n_context_tokens",
                                 "n_counterfactual_in_context",
                                 "true_final_in_context")}
                rows.append({"variant": variant, "seed": 0,
                             "intervention": style, "multiplicity": m,
                             "position": a.position, "n_records": n,
                             **agg, "per_record": per})
                print(f"[synd] {variant:>16} {style:>11} x{m:<2} "
                      f"gold={agg['gold_recall']:.4f} "
                      f"misled={agg['misled']:.4f} "
                      f"cf_in_ctx={agg['n_counterfactual_in_context']:.2f} "
                      f"ctx={agg['n_context_tokens']:.0f}  "
                      f"cache={len(cache)}")
        dest = out_dir / f"{variant}_s0.json"
        dest.write_text(json.dumps(
            {"variant": variant, "reader": a.reader,
             "reader_revision": a.reader_revision, "position": a.position,
             "n_records": len(items),
             "preregistration": "docs/SYNDICATION_PREREGISTRATION.md",
             "rows": rows}, indent=2) + "\n")
        print(f"[synd] wrote {dest}")


if __name__ == "__main__":
    main()
