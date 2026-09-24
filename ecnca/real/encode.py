"""Frozen text encoding with an on-disk cache.

The encoder is frozen for the matched comparison and embeddings are computed
ONCE, so every variant sees byte-identical inputs.  Any difference between EC
and the no-provenance baseline is then attributable to the aggregation, not to
a different view of the text.

`HashEncoder` is a deterministic, dependency-free stand-in used by the tests: it
has no semantics, but it is stable and reproducible, which is all the structural
tests need.  Real runs use `SentenceTransformerEncoder`.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


class HashEncoder:
    """Deterministic pseudo-embedding. For tests only -- carries no meaning."""

    name = "hash-stub"

    def __init__(self, dim: int = 64):
        self.dim = dim

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            seed = int.from_bytes(hashlib.blake2b((t or "").encode("utf-8"),
                                                  digest_size=8).digest(), "little")
            v = np.random.default_rng(seed).normal(size=self.dim).astype(np.float32)
            out[i] = v / (np.linalg.norm(v) + 1e-9)
        return out


# BGE documents a retrieval instruction for the QUERY side only. Adding it to
# passages as well would destroy the asymmetry the model was trained with.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class SentenceTransformerEncoder:
    """Frozen sentence-transformers encoder, pinned to an exact revision."""

    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5",
                 revision: Optional[str] = None, device: str = "cpu",
                 batch_size: int = 64, max_seq_length: int = 512,
                 query_instruction: str = ""):
        from sentence_transformers import SentenceTransformer
        self.model_name, self.revision = model_name, revision
        self.batch_size = batch_size
        self.query_instruction = query_instruction
        self.model = SentenceTransformer(model_name, revision=revision, device=device)
        self.model.max_seq_length = max_seq_length
        self.model.eval()
        for prm in self.model.parameters():
            prm.requires_grad_(False)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.max_seq_length = max_seq_length
        # the fingerprint enters every cache key, so a revision or instruction
        # change can never silently reuse stale vectors
        self.name = (f"{model_name}@{revision or 'main'}"
                     f"|len{max_seq_length}|norm|instr={bool(query_instruction)}")

    @property
    def n_encoder_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def truncation_rate(self, texts: Sequence[str]) -> float:
        tok = self.model.tokenizer
        n = sum(1 for t in texts
                if len(tok(t, add_special_tokens=True)["input_ids"]) > self.max_seq_length)
        return n / max(len(texts), 1)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        prepped = [self.query_instruction + t for t in texts] \
            if self.query_instruction else list(texts)
        return np.asarray(self.model.encode(
            prepped, batch_size=self.batch_size, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=False), dtype=np.float32)


def bge_pair(revision: Optional[str] = None, device: str = "cpu",
             batch_size: int = 64, max_seq_length: int = 512):
    """(query encoder, passage encoder) -- instruction on the query side ONLY."""
    q = SentenceTransformerEncoder("BAAI/bge-base-en-v1.5", revision, device,
                                   batch_size, max_seq_length,
                                   query_instruction=BGE_QUERY_INSTRUCTION)
    p = SentenceTransformerEncoder("BAAI/bge-base-en-v1.5", revision, device,
                                   batch_size, max_seq_length, query_instruction="")
    p.model = q.model                      # one set of frozen weights, shared
    return q, p


class EmbeddingCache:
    """Content-addressed cache: key is (encoder name, normalised text).

    Keying on the TEXT rather than a passage id matters here: exact redelivery
    of a passage must reuse the identical vector, so an intervention cannot
    accidentally change the embedding it was supposed to hold fixed.
    """

    def __init__(self, path: str, encoder):
        self.path, self.encoder = path, encoder
        self.dim = getattr(encoder, "dim", None)
        self._mem: Dict[str, np.ndarray] = {}
        self._loaded = False

    def _key(self, text: str) -> str:
        h = hashlib.blake2b(f"{self.encoder.name}\x1f{text}".encode("utf-8"),
                            digest_size=16).hexdigest()
        return h

    def _load(self):
        if self._loaded:
            return
        if os.path.exists(self.path):
            with np.load(self.path, allow_pickle=False) as z:
                keys = json.loads(str(z["keys"]))
                mat = z["mat"]
                self._mem = {k: mat[i] for i, k in enumerate(keys)}
                self.dim = mat.shape[1] if mat.size else self.dim
        self._loaded = True

    def save(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        keys = sorted(self._mem)
        mat = np.stack([self._mem[k] for k in keys]) if keys else \
            np.zeros((0, self.dim or 1), dtype=np.float32)
        tmp = self.path + ".tmp.npz"
        np.savez_compressed(tmp, keys=json.dumps(keys), mat=mat)
        os.replace(tmp, self.path)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        self._load()
        keys = [self._key(t) for t in texts]
        missing = [t for t, k in zip(texts, keys) if k not in self._mem]
        if missing:
            uniq = sorted(set(missing))
            vecs = self.encoder.encode(uniq)
            self.dim = vecs.shape[1]
            for t, v in zip(uniq, vecs):
                self._mem[self._key(t)] = v
        return np.stack([self._mem[k] for k in keys])

    def __len__(self):
        self._load()
        return len(self._mem)
