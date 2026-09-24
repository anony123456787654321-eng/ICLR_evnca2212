"""AmbigNQ labels for variable-cardinality sector experiments.

The official data can contain multiple acceptable annotations for one question.
We preserve those annotations instead of inventing a consensus target: each
annotation is one admissible set of interpretations and can be scored as such.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Iterable, Iterator

import numpy as np


@dataclass(frozen=True)
class Interpretation:
    question: str
    answers: tuple[str, ...]


@dataclass(frozen=True)
class AmbigAnnotation:
    example_id: str
    question: str
    annotation_index: int
    interpretations: tuple[Interpretation, ...]

    @property
    def cardinality(self) -> int:
        return len(self.interpretations)


@dataclass(frozen=True)
class EvidenceChunk:
    root: str
    article_index: int
    chunk_index: int
    text: str
    score: float


def _nonempty_strings(values: Iterable[object], field: str) -> tuple[str, ...]:
    result = tuple(value.strip() for value in values if isinstance(value, str) and value.strip())
    if not result:
        raise ValueError(f"{field} must contain at least one non-empty string")
    return result


def parse_annotation(record: dict, annotation_index: int) -> AmbigAnnotation:
    try:
        example_id = str(record["id"])
        question = str(record["question"]).strip()
        annotation = record["annotations"][annotation_index]
        annotation_type = annotation["type"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("invalid AmbigNQ record") from exc
    if not question:
        raise ValueError("question must be non-empty")

    if annotation_type == "singleAnswer":
        interpretations = (
            Interpretation(question=question,
                           answers=_nonempty_strings(annotation.get("answer", ()), "answer")),
        )
    elif annotation_type == "multipleQAs":
        pairs = annotation.get("qaPairs")
        if not isinstance(pairs, list) or not pairs:
            raise ValueError("multipleQAs annotation must contain qaPairs")
        interpretations = tuple(
            Interpretation(
                question=str(pair.get("question", "")).strip(),
                answers=_nonempty_strings(pair.get("answer", ()), "qaPairs.answer"),
            )
            for pair in pairs
        )
        if any(not item.question for item in interpretations):
            raise ValueError("disambiguated question must be non-empty")
    else:
        raise ValueError(f"unknown AmbigNQ annotation type: {annotation_type!r}")

    return AmbigAnnotation(
        example_id=example_id,
        question=question,
        annotation_index=annotation_index,
        interpretations=interpretations,
    )


def iter_json_array(path: Path | str, chunk_size: int = 1 << 20) -> Iterator[dict]:
    """Stream a top-level JSON array, including the 1.25 GB evidence file."""
    decoder = json.JSONDecoder()
    with Path(path).open() as handle:
        buffer = ""
        position = 0
        started = False
        eof = False
        while True:
            if position > 0:
                buffer = buffer[position:]
                position = 0
            while not eof and len(buffer) < chunk_size:
                piece = handle.read(chunk_size)
                if piece:
                    buffer += piece
                else:
                    eof = True
            while position < len(buffer) and buffer[position].isspace():
                position += 1
            if not started:
                if position >= len(buffer) or buffer[position] != "[":
                    raise ValueError("AmbigNQ file must contain a JSON list")
                position += 1
                started = True
            while position < len(buffer) and (buffer[position].isspace()
                                               or buffer[position] == ","):
                position += 1
            if position < len(buffer) and buffer[position] == "]":
                return
            try:
                record, end = decoder.raw_decode(buffer, position)
            except json.JSONDecodeError:
                if eof:
                    raise ValueError("truncated or invalid AmbigNQ JSON")
                # Preserve the incomplete object and fetch another chunk.
                buffer = buffer[position:] + handle.read(chunk_size)
                position = 0
                continue
            if not isinstance(record, dict):
                raise ValueError("AmbigNQ array entries must be objects")
            yield record
            position = end


def article_title(article: str) -> str:
    """Stable provenance root from the first Markdown heading."""
    for line in article.splitlines():
        line = line.strip()
        if line.startswith("#"):
            title = line.lstrip("#").strip()
            if title:
                return " ".join(title.casefold().split())
        if line:
            break
    return ""


_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.casefold())


def select_root_balanced_chunks(
    question: str,
    articles: Iterable[str],
    *,
    per_article: int = 8,
    words_per_chunk: int = 100,
) -> tuple[EvidenceChunk, ...]:
    """Question-ranked chunks with an equal budget for every provenance root.

    Ranking is BM25 within each article.  It never consults annotations or
    answers, and a long article cannot suppress another root's entire message
    budget.  Chunk boundaries are fixed before encoding so all variants receive
    byte-identical text.
    """
    if per_article < 1 or words_per_chunk < 1:
        raise ValueError("chunk budgets must be positive")
    query = _tokens(question)
    selected: list[EvidenceChunk] = []
    for article_index, article in enumerate(articles):
        root = article_title(article)
        if not root:
            raise ValueError(f"article {article_index} has no Markdown title")
        words = article.split()
        chunks = [" ".join(words[start:start + words_per_chunk])
                  for start in range(0, len(words), words_per_chunk)]
        tokenized = [_tokens(chunk) for chunk in chunks]
        n_docs = max(len(chunks), 1)
        document_frequency = {
            token: sum(token in set(chunk) for chunk in tokenized)
            for token in set(query)
        }
        average_length = sum(map(len, tokenized)) / n_docs
        scored = []
        for chunk_index, (text, terms) in enumerate(zip(chunks, tokenized)):
            frequencies = {token: terms.count(token) for token in set(query)}
            score = 0.0
            for token in query:
                tf = frequencies.get(token, 0)
                if not tf:
                    continue
                df = document_frequency[token]
                inverse_frequency = np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
                normalizer = tf + 1.2 * (
                    0.25 + 0.75 * len(terms) / max(average_length, 1.0)
                )
                score += inverse_frequency * tf * 2.2 / normalizer
            scored.append(EvidenceChunk(root, article_index, chunk_index, text, score))
        selected.extend(sorted(scored, key=lambda item: (-item.score, item.chunk_index))[
            :per_article
        ])
    return tuple(selected)


def load_annotations(path: Path | str) -> Iterator[AmbigAnnotation]:
    for record in iter_json_array(path):
        annotations = record.get("annotations") if isinstance(record, dict) else None
        if not isinstance(annotations, list) or not annotations:
            raise ValueError("each AmbigNQ record must contain annotations")
        for annotation_index in range(len(annotations)):
            yield parse_annotation(record, annotation_index)


def cardinality_histogram(annotations: Iterable[AmbigAnnotation]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for annotation in annotations:
        counts[annotation.cardinality] = counts.get(annotation.cardinality, 0) + 1
    return dict(sorted(counts.items()))
