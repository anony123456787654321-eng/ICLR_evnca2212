"""MuSiQue-Answerable transport records for the real multi-hop audit."""
from __future__ import annotations

import json
import re
import string
from dataclasses import dataclass
from pathlib import Path


_REF = re.compile(r"#(\d+)")


def normalise_answer(text: str) -> str:
    text = (text or "").lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    return " ".join(word for word in text.split() if word not in {"a", "an", "the"})


@dataclass(frozen=True)
class MuSiQueStep:
    question: str
    answer: str
    paragraph_title: str
    paragraph_text: str
    paragraph_idx: int
    dependencies: tuple[int, ...]


@dataclass(frozen=True)
class MuSiQueRecord:
    record_id: str
    question: str
    answer: str
    answer_aliases: tuple[str, ...]
    steps: tuple[MuSiQueStep, ...]
    is_linear: bool
    linear_failure: str

    @property
    def n_hops(self):
        return len(self.steps)


def parse_record(item: dict) -> MuSiQueRecord:
    paragraphs = {int(p["idx"]): p for p in item["paragraphs"]}
    steps = []
    failures = []
    for index, raw in enumerate(item["question_decomposition"]):
        paragraph_idx = int(raw["paragraph_support_idx"])
        paragraph = paragraphs.get(paragraph_idx)
        if paragraph is None:
            failures.append(f"missing_paragraph_{paragraph_idx}")
            paragraph = {"title": "", "paragraph_text": "", "is_supporting": False}
        if not paragraph.get("is_supporting", False):
            failures.append(f"paragraph_{paragraph_idx}_not_supporting")
        dependencies = tuple(sorted({int(x) for x in _REF.findall(raw["question"])}))
        expected = () if index == 0 else (index,)
        if dependencies != expected:
            failures.append(f"step_{index + 1}_dependencies_{dependencies}_expected_{expected}")
        steps.append(MuSiQueStep(
            question=raw["question"], answer=raw["answer"],
            paragraph_title=paragraph.get("title", ""),
            paragraph_text=paragraph.get("paragraph_text", ""),
            paragraph_idx=paragraph_idx, dependencies=dependencies))
    if not steps:
        failures.append("empty_decomposition")
    elif normalise_answer(steps[-1].answer) != normalise_answer(item["answer"]):
        failures.append("terminal_answer_mismatch")
    return MuSiQueRecord(
        record_id=str(item["id"]), question=item["question"], answer=item["answer"],
        answer_aliases=tuple(item.get("answer_aliases") or ()), steps=tuple(steps),
        is_linear=not failures, linear_failure=";".join(failures))


def load_musique(path: str | Path) -> list[MuSiQueRecord]:
    records = []
    with open(path) as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid MuSiQue JSONL at line {line_number}") from exc
            if not item.get("answerable", True):
                continue
            records.append(parse_record(item))
    return records


def split_linear(records):
    return [record for record in records if record.is_linear]
