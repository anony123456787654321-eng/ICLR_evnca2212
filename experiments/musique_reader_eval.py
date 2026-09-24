"""Does evidence conservation help a reader model, not just an accounting column?

Every result so far measures a learned readout of roughly half a million
trainable parameters. A reader may reasonably ask whether conservation matters
once a modern generative reader sits on top of the retrieved passages, which is
the configuration deployed retrieval systems actually use.

This experiment answers that question while changing as little as possible. The
retrieval, the duplication regimes and the ledger modes are the ones already
used elsewhere in this repository. The only new component is a frozen reader,
which never trains and never sees a gradient.

The design isolates the ledger. For each arm, a delivery stream is built under
a duplication regime, the arm's ledger decides which arrivals survive, and only
the surviving passages are placed in the reader's context. Every arm therefore
receives the same question, the same corpus and the same reader, and differs
only in which passages the accounting rule admitted. Measured are answer exact
match, token F1, citation precision against the gold supporting paragraphs, and
the context token count that each arm asks the reader to consume.

A conserving ledger should hold accuracy flat as duplication rises while
spending fewer context tokens, because suppressed arrivals never reach the
reader. A provenance-free arm should spend tokens in proportion to the delivery
count and crowd genuine evidence out of a bounded context.

No proprietary API is used. The reader is a local checkpoint at a pinned
revision, decoded greedily so a rerun reproduces the run exactly.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ecnca.real.musique import load_musique, normalise_answer, split_linear
from ecnca.real.musique_duplication import (INTERVENTIONS, MULTIPLICITIES,
                                            build_stream, ledger_mode,
                                            resolve_stream)

DEFAULT_VARIANTS = ("full", "filter_only", "plain", "canonical_dedup")
CITE = re.compile(r"\[(\d+)\]")


def exact_match(pred: str, golds) -> float:
    p = normalise_answer(pred)
    return float(any(p == normalise_answer(g) for g in golds))


def answer_recall(pred: str, golds) -> float:
    """Does a gold answer appear in the reply at all?

    A chat reader replies in a sentence, so exact match is near-vacuous even
    when the answer is present and correct. Containment is the metric that
    survives that, and it is reported alongside exact match rather than in
    place of it.
    """
    p = normalise_answer(pred)
    return float(any(normalise_answer(g) and normalise_answer(g) in p
                     for g in golds))


def token_f1(pred: str, golds) -> float:
    best = 0.0
    pt = normalise_answer(pred).split()
    for g in golds:
        gt = normalise_answer(g).split()
        common = Counter(pt) & Counter(gt)
        n = sum(common.values())
        if n == 0:
            continue
        precision, recall = n / len(pt), n / len(gt)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def build_prompt(tok, question: str, passages):
    listed = "\n".join(f"[{i + 1}] {t}" for i, t in enumerate(passages))
    ask = ("Answer the question using only the passages. Put the answer alone "
           "on the first line, as few words as possible, with no sentence and "
           "no explanation. On the second line list every passage number you "
           "used, each in square brackets.\n\n"
           f"Passages:\n{listed}\n\nQuestion: {question}")
    return tok.apply_chat_template([{"role": "user", "content": ask}],
                                   tokenize=False, add_generation_prompt=True)


def parse_reply(text: str):
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    answer = lines[0] if lines else ""
    cited = {int(m) for ln in lines[1:] for m in CITE.findall(ln)}
    if not cited and len(lines) > 1:
        cited = {int(m) for m in CITE.findall(" ".join(lines[1:]))}
    return answer, cited


def score_record(record, variant, intervention, multiplicity, scope, seed,
                 tok, model, device, max_new_tokens):
    import torch
    events = build_stream(record, intervention, multiplicity, scope, seed=seed)
    kept, credit, suppressed = resolve_stream(events, ledger_mode(variant))
    passages = [e.paragraph for e in kept]
    gold_titles = [s.paragraph_title for s in record.steps]
    # a kept arrival is supporting when its root is one of the chain's steps
    supporting = {i + 1 for i, e in enumerate(kept)
                  if any(e.paragraph.startswith(t[:24]) or t in e.paragraph
                         for t in gold_titles)}
    prompt = build_prompt(tok, record.question, passages)
    batch = tok(prompt, return_tensors="pt").to(device)
    n_context = int(batch["input_ids"].shape[-1])
    with torch.no_grad():
        out = model.generate(**batch, max_new_tokens=max_new_tokens,
                             do_sample=False, temperature=None, top_p=None,
                             pad_token_id=tok.pad_token_id)
    reply = tok.decode(out[0][batch["input_ids"].shape[-1]:],
                       skip_special_tokens=True)
    answer, cited = parse_reply(reply)
    golds = (record.answer,) + tuple(record.answer_aliases)
    cited_valid = {c for c in cited if 1 <= c <= len(passages)}
    # Scoring a reply that cites nothing as precision zero would conflate
    # declining to cite with citing wrongly. Precision is reported only over
    # replies that cited, and the rate of citing at all is reported beside it.
    cited_any = bool(cited_valid)
    cite_prec = (len(cited_valid & supporting) / len(cited_valid)
                 if cited_any else float("nan"))
    return {"record_id": record.record_id, "n_hops": record.n_hops,
            "n_delivered": len(events), "n_kept": len(kept),
            "n_suppressed": suppressed, "n_context_tokens": n_context,
            "credited_over_ceiling": sum(credit.values()) / max(len(gold_titles), 1),
            "exact_match": exact_match(answer, golds),
            "answer_recall": answer_recall(answer, golds),
            "token_f1": token_f1(answer, golds),
            "citation_precision": cite_prec,
            "cited_any": float(cited_any),
            "n_cited": len(cited_valid)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    root = "data/raw/musique/data"
    ap.add_argument("--dev", default=f"{root}/musique_ans_v1.0_dev.jsonl")
    ap.add_argument("--reader", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--reader-revision", default="main")
    # filter_only is retained as a control rather than as a contrast. The
    # refinement it removes lives in the learned transport, which a frozen
    # reader bypasses entirely, so this experiment measures the accounting
    # value of the ledger and not the value of refinement.
    ap.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    ap.add_argument("--interventions", default="exact,overlap")
    ap.add_argument("--multiplicities", default="1,4,16")
    ap.add_argument("--scope", default="all_hops")
    ap.add_argument("--hops", default="2")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--limit", type=int, default=120)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/musique_reader/eval")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--per-record", action="store_true",
                    help="store each record's score, so a bootstrap over "
                         "records can be computed after the run; costs "
                         "space, not time")
    a = ap.parse_args()
    os.chdir(Path(__file__).resolve().parent.parent)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hops = {int(h) for h in a.hops.split(",") if h}
    records = [r for r in split_linear(load_musique(a.dev))
               if r.n_hops in hops][:a.limit]
    if not records:
        raise SystemExit(f"no records with hops {sorted(hops)} in {a.dev}")

    tok = AutoTokenizer.from_pretrained(a.reader, revision=a.reader_revision)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        a.reader, revision=a.reader_revision,
        torch_dtype=torch.bfloat16).to(a.device).eval()

    out_dir = Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)
    variants = [v for v in a.variants.split(",") if v]
    interventions = [i for i in a.interventions.split(",") if i]
    mults = [int(m) for m in a.multiplicities.split(",") if m]
    seeds = [int(s) for s in a.seeds.split(",") if s]

    for variant in variants:
        for seed in seeds:
            dest = out_dir / f"{variant}_s{seed}.json"
            if dest.is_file() and not a.force:
                print(f"[reader] skip {dest.name}, already complete")
                continue
            rows = []
            for intervention in interventions:
                for m in mults:
                    if intervention not in INTERVENTIONS:
                        raise SystemExit(f"unknown intervention {intervention}")
                    if m not in MULTIPLICITIES:
                        raise SystemExit(f"unknown multiplicity {m}")
                    scored = [score_record(r, variant, intervention, m,
                                           a.scope, seed, tok, model,
                                           a.device, a.max_new_tokens)
                              for r in records]
                    agg = {k: sum(s[k] for s in scored) / len(scored)
                           for k in ("exact_match", "answer_recall",
                                     "token_f1", "cited_any",
                                     "n_context_tokens", "n_kept",
                                     "credited_over_ceiling")}
                    cited = [s["citation_precision"] for s in scored
                             if s["cited_any"]]
                    agg["citation_precision"] = (sum(cited) / len(cited)
                                                 if cited else float("nan"))
                    agg["n_cited_replies"] = len(cited)
                    row = {"variant": variant, "seed": seed,
                          "intervention": intervention,
                          "multiplicity": m, "scope": a.scope,
                          "n_records": len(scored), **agg}
                    if a.per_record:
                        # exact and overlap never seed randomness (see
                        # ecnca.real.musique_duplication.build_stream), so a
                        # sweep restricted to them is deterministic and three
                        # seeds of it are one replicate, not three. Per-record
                        # scores let a bootstrap over records, the axis that is
                        # actually random here, stand in for a seed interval.
                        row["per_record"] = [
                            {"record_id": s["record_id"],
                             "token_f1": s["token_f1"],
                             "answer_recall": s["answer_recall"],
                             "exact_match": s["exact_match"],
                             "n_context_tokens": s["n_context_tokens"]}
                            for s in scored]
                    rows.append(row)
                    print(f"[reader] {variant} s{seed} {intervention} x{m}  "
                          f"EM={agg['exact_match']:.4f} "
                          f"rec={agg['answer_recall']:.4f} "
                          f"F1={agg['token_f1']:.4f} "
                          f"cite={agg['citation_precision']:.4f}"
                          f"({agg['n_cited_replies']}/{len(scored)}) "
                          f"ctx={agg['n_context_tokens']:.0f}")
            dest.write_text(json.dumps(
                {"variant": variant, "seed": seed, "reader": a.reader,
                 "reader_revision": a.reader_revision, "scope": a.scope,
                 "hops": sorted(hops), "n_records": len(records),
                 "rows": rows}, indent=2) + "\n")
            print(f"[reader] wrote {dest}")


if __name__ == "__main__":
    main()
