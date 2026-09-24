"""MNIST acquisition, checksums, and the frozen split roles for the restart.

Split policy (`MNIST_RESTART_PLAN.md`, Phase 1):

* The original 60,000-image MNIST *training* population is partitioned into
  50,000 train / 5,000 development / 5,000 calibration, stratified by digit,
  with indices fixed by ``SPLIT_SEED``.
* The original 10,000-image MNIST *test* set is **evaluation-sealed**: its
  images and labels must not be used for training, evaluation, pair selection,
  threshold fitting, or architecture selection until Phase 3. ``load_split("test")`` raises
  unless the caller passes the explicit unseal token, so an accidental import
  cannot consume it.

Canonical checksums are the published Yann LeCun / mirror values for the four
IDX archives. They are recorded here so another worker can confirm byte
identity of the inputs rather than trusting a download.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import struct
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# --- provenance ----------------------------------------------------------

# Mirrors. The canonical http://yann.lecun.com/exdb/mnist/ host has been
# unreliable/403 for programmatic clients; these mirrors serve byte-identical
# archives, which is what the SHA-256 values below verify.
MIRRORS = (
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
)

ARCHIVES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

# SHA-256 of the gzip archives as published.
EXPECTED_SHA256 = {
    "train-images-idx3-ubyte.gz":
        "440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609",
    "train-labels-idx1-ubyte.gz":
        "3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c",
    "t10k-images-idx3-ubyte.gz":
        "8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6",
    "t10k-labels-idx1-ubyte.gz":
        "f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6",
}

# Frozen split definition. Changing either constant changes data roles and
# therefore invalidates every downstream comparison.
SPLIT_SEED = 20260911
SPLIT_SIZES = {"train": 50_000, "dev": 5_000, "calib": 5_000}

# Guard token for the sealed test set. Phase 3 passes this explicitly.
UNSEAL_TEST_TOKEN = "phase3-confirmatory-run"

DEFAULT_ROOT = Path("data/raw/mnist")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(root: Path = DEFAULT_ROOT, *, verify: bool = True) -> dict[str, str]:
    """Fetch the four IDX archives if absent; return {filename: sha256}."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    digests: dict[str, str] = {}
    for name in ARCHIVES.values():
        dest = root / name
        if not dest.exists():
            last: Exception | None = None
            for mirror in MIRRORS:
                try:
                    with urllib.request.urlopen(mirror + name, timeout=120) as r:
                        payload = r.read()
                    dest.write_bytes(payload)
                    last = None
                    break
                except Exception as exc:  # pragma: no cover - network path
                    last = exc
            if last is not None:
                raise RuntimeError(f"could not download {name}: {last}")
        digest = _sha256(dest)
        if verify and EXPECTED_SHA256[name] != digest:
            raise RuntimeError(
                f"checksum mismatch for {name}: expected "
                f"{EXPECTED_SHA256[name]}, got {digest}"
            )
        digests[name] = digest
    return digests


def _read_idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as fh:
        magic, count = struct.unpack(">II", fh.read(8))
        if magic == 2051:  # images
            rows, cols = struct.unpack(">II", fh.read(8))
            buf = fh.read(count * rows * cols)
            return np.frombuffer(buf, np.uint8).reshape(count, rows, cols)
        if magic == 2049:  # labels
            buf = fh.read(count)
            return np.frombuffer(buf, np.uint8)
        raise ValueError(f"unexpected IDX magic {magic} in {path}")


def load_raw(root: Path = DEFAULT_ROOT, *, verify: bool = True):
    """Return (train_images, train_labels, test_images, test_labels), uint8."""
    download(root, verify=verify)
    root = Path(root)
    return (
        _read_idx(root / ARCHIVES["train_images"]),
        _read_idx(root / ARCHIVES["train_labels"]),
        _read_idx(root / ARCHIVES["test_images"]),
        _read_idx(root / ARCHIVES["test_labels"]),
    )


def split_indices(labels: np.ndarray) -> dict[str, np.ndarray]:
    """Stratified, deterministic partition of the training population.

    Within each digit the shuffled order is fixed by ``SPLIT_SEED``; the digit's
    members are then dealt into train/dev/calib in proportion to SPLIT_SIZES so
    every split carries the population's class balance. Returns sorted indices.
    """
    total = int(sum(SPLIT_SIZES.values()))
    if labels.shape[0] != total:
        raise ValueError(
            f"expected {total} training labels, got {labels.shape[0]}"
        )
    rng = np.random.default_rng(SPLIT_SEED)
    order = ("train", "dev", "calib")
    members = {
        d: np.flatnonzero(labels == d)[
            rng.permutation(int((labels == d).sum()))
        ]
        for d in range(10)
    }

    # Allocation must satisfy two constraints at once: each split gets exactly
    # SPLIT_SIZES[k] items (column sums), and each digit contributes exactly as
    # many items as it has (row sums). Independent per-digit or per-split
    # largest-remainder rounding satisfies only one of the two, so do it in two
    # stages: round within each digit (row sums exact by construction), then
    # repair the split totals by transferring single items between splits of
    # the digit that is furthest from its proportional share. Every transfer
    # preserves row sums, so stratification degrades by at most one item per
    # repair while the totals become exact.
    counts = {d: dict.fromkeys(order, 0) for d in range(10)}
    for d in range(10):
        have = int(members[d].shape[0])
        exact = {k: have * SPLIT_SIZES[k] / total for k in order}
        base = {k: int(np.floor(exact[k])) for k in order}
        residual = have - sum(base.values())
        rank = sorted(order, key=lambda k: (-(exact[k] - base[k]), k))
        for k in rank[:residual]:
            base[k] += 1
        counts[d] = base

    def column(k: str) -> int:
        return sum(counts[d][k] for d in range(10))

    guard = 0
    while True:
        surplus = [k for k in order if column(k) > SPLIT_SIZES[k]]
        deficit = [k for k in order if column(k) < SPLIT_SIZES[k]]
        if not surplus:
            if deficit:
                raise RuntimeError("allocation deficit without surplus")
            break
        src, dst = surplus[0], deficit[0]
        # Move from the digit most over-represented in `src` relative to its
        # proportional share, so the repair costs the least stratification.
        def overshoot(d: int) -> float:
            have = int(members[d].shape[0])
            return counts[d][src] - have * SPLIT_SIZES[src] / total
        donor = max(
            (d for d in range(10) if counts[d][src] > 0),
            key=lambda d: (overshoot(d), -d),
        )
        counts[donor][src] -= 1
        counts[donor][dst] += 1
        guard += 1
        if guard > 1000:  # pragma: no cover - structural safety net
            raise RuntimeError("split repair failed to converge")

    buckets: dict[str, list[np.ndarray]] = {k: [] for k in SPLIT_SIZES}
    for d in range(10):
        if sum(counts[d].values()) != int(members[d].shape[0]):
            raise RuntimeError(f"digit {d}: row sum changed during repair")
        start = 0
        for k in order:
            buckets[k].append(members[d][start:start + counts[d][k]])
            start += counts[d][k]
    out = {k: np.sort(np.concatenate(v)) for k, v in buckets.items()}
    sizes = {k: int(v.shape[0]) for k, v in out.items()}
    if sizes != SPLIT_SIZES:
        raise RuntimeError(f"split sizes {sizes} != {SPLIT_SIZES}")
    everything = np.concatenate(list(out.values()))
    if np.unique(everything).shape[0] != total:
        raise RuntimeError("splits are not a partition of the training set")
    return out


@dataclass(frozen=True)
class Split:
    name: str
    images: np.ndarray  # float32 in [0, 1], (N, 28, 28)
    labels: np.ndarray  # int64, (N,)
    indices: np.ndarray  # int64 positions in the source population

    def __len__(self) -> int:
        return int(self.labels.shape[0])


def load_split(
    name: str,
    root: Path = DEFAULT_ROOT,
    *,
    unseal_token: str | None = None,
    verify: bool = True,
) -> Split:
    """Load one split. ``name="test"`` requires the explicit unseal token."""
    tr_x, tr_y, te_x, te_y = load_raw(root, verify=verify)
    if name == "test":
        if unseal_token != UNSEAL_TEST_TOKEN:
            raise PermissionError(
                "the MNIST test set is sealed until Phase 3; pass "
                "unseal_token=ecnca.vision.data.UNSEAL_TEST_TOKEN to open it"
            )
        return Split(
            "test",
            (te_x / 255.0).astype(np.float32),
            te_y.astype(np.int64),
            np.arange(te_y.shape[0], dtype=np.int64),
        )
    if name not in SPLIT_SIZES:
        raise KeyError(f"unknown split {name!r}")
    idx = split_indices(tr_y)[name]
    return Split(
        name,
        (tr_x[idx] / 255.0).astype(np.float32),
        tr_y[idx].astype(np.int64),
        idx,
    )


def split_manifest(root: Path = DEFAULT_ROOT, *, verify: bool = True) -> dict:
    """A reconstructible record of the data roles, with content hashes."""
    tr_x, tr_y, te_x, te_y = load_raw(root, verify=verify)
    idx = split_indices(tr_y)
    entries = {}
    for name, positions in idx.items():
        labels = tr_y[positions]
        entries[name] = {
            "size": int(positions.shape[0]),
            "index_sha256": hashlib.sha256(
                positions.astype("<i8").tobytes()
            ).hexdigest(),
            "label_histogram": {
                str(d): int((labels == d).sum()) for d in range(10)
            },
            "index_min": int(positions.min()),
            "index_max": int(positions.max()),
        }
    return {
        "source": {
            "mirrors": list(MIRRORS),
            "archive_sha256": {k: EXPECTED_SHA256[k] for k in sorted(EXPECTED_SHA256)},
            "train_population": int(tr_y.shape[0]),
            "test_population": int(te_y.shape[0]),
        },
        "split_seed": SPLIT_SEED,
        "split_sizes": dict(SPLIT_SIZES),
        "splits": entries,
        "test_set": {
            "status": "evaluation-sealed until Phase 3",
            "guard": "ecnca.vision.data.load_split('test') raises PermissionError "
                     "without UNSEAL_TEST_TOKEN",
        },
    }


def write_split_manifest(
    path: Path, root: Path = DEFAULT_ROOT, *, verify: bool = True
) -> dict:
    manifest = split_manifest(root, verify=verify)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest
