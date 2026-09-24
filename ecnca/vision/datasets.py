"""Datasets for the headroom gate: Fashion-MNIST, CIFAR-10, CIFAR-10-C.

Scope note. An earlier instruction excluded Fashion-MNIST from the campaign
unless a concrete blocker made the chosen domain unusable. The headroom gate
supersedes that: Fashion-MNIST enters as a **candidate only**, and is frozen as
the low-cost transfer experiment solely if it clears the gate's thresholds.

Split policy mirrors ``ecnca/vision/data.py``: the original *training*
population is partitioned into train/dev/calib with fixed stratified indices,
and the official test split is **evaluation-sealed** behind a token. Gate
pilots use development data only.

CIFAR-10-C ships as per-corruption .npy files of 50,000 images each: 10,000
official test images at each of five severities, stacked in order. Those are
the official held-out examples, so the gate reads them for **measurement only**
and never for tuning; that boundary is enforced by the caller, and recorded in
the manifest.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data import SPLIT_SEED, SPLIT_SIZES, UNSEAL_TEST_TOKEN, _read_idx, _sha256

# --- Fashion-MNIST -------------------------------------------------------
FASHION_MIRRORS = (
    "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/",
    "https://storage.googleapis.com/tensorflow/tf-keras-datasets/",
)
FASHION_ARCHIVES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}
FASHION_CLASSES = (
    "tshirt", "trouser", "pullover", "dress", "coat",
    "sandal", "shirt", "sneaker", "bag", "boot",
)

# CIFAR-10's training population is 50,000, not MNIST's 60,000, so it cannot
# reuse SPLIT_SIZES. Held-out fractions are kept proportional to the MNIST
# policy (1/12 dev, 1/12 calib) rather than the absolute counts, so the same
# share of data is reserved for selection and calibration.
CIFAR10_SPLIT_SIZES = {"train": 41_666, "dev": 4_167, "calib": 4_167}

# --- CIFAR-10 ------------------------------------------------------------
CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR10_SHA256 = "6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce"
CIFAR10_CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)

# --- CIFAR-10-C ----------------------------------------------------------
CIFAR10C_URL = "https://zenodo.org/records/2535967/files/CIFAR-10-C.tar"
CIFAR10C_CORRUPTIONS = (
    "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur",
    "glass_blur", "motion_blur", "zoom_blur", "snow", "frost", "fog",
    "brightness", "contrast", "elastic_transform", "pixelate",
    "jpeg_compression",
)
CIFAR10C_SEVERITIES = (1, 2, 3, 4, 5)

DEFAULT_ROOT = Path("data/raw")


@dataclass(frozen=True)
class VisionSplit:
    name: str
    dataset: str
    images: np.ndarray   # float32 in [0,1]; (N,28,28) grey or (N,32,32,3) RGB
    labels: np.ndarray   # int64
    indices: np.ndarray  # positions in the source population
    classes: tuple[str, ...]

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    @property
    def is_rgb(self) -> bool:
        return self.images.ndim == 4


def _download(url: str, dest: Path, timeout: int = 600) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=timeout) as r, tmp.open("wb") as fh:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
    tmp.replace(dest)


def _stratified_indices(labels: np.ndarray, sizes: dict[str, int],
                        seed: int) -> dict[str, np.ndarray]:
    """Same two-stage allocation as ``data.split_indices``.

    Per-digit and per-split largest-remainder rounding each satisfy only one of
    the two sum constraints, so round within classes then repair split totals by
    single transfers.
    """
    total = int(sum(sizes.values()))
    if labels.shape[0] != total:
        raise ValueError(f"expected {total} labels, got {labels.shape[0]}")
    n_cls = int(labels.max()) + 1
    rng = np.random.default_rng(seed)
    order = tuple(sizes.keys())
    members = {
        c: np.flatnonzero(labels == c)[rng.permutation(int((labels == c).sum()))]
        for c in range(n_cls)
    }
    counts = {c: dict.fromkeys(order, 0) for c in range(n_cls)}
    for c in range(n_cls):
        have = int(members[c].shape[0])
        exact = {k: have * sizes[k] / total for k in order}
        base = {k: int(np.floor(exact[k])) for k in order}
        for k in sorted(order, key=lambda k: (-(exact[k] - base[k]), k))[
                :have - sum(base.values())]:
            base[k] += 1
        counts[c] = base

    def column(k: str) -> int:
        return sum(counts[c][k] for c in range(n_cls))

    guard = 0
    while True:
        surplus = [k for k in order if column(k) > sizes[k]]
        deficit = [k for k in order if column(k) < sizes[k]]
        if not surplus:
            if deficit:
                raise RuntimeError("allocation deficit without surplus")
            break
        src, dst = surplus[0], deficit[0]
        donor = max(
            (c for c in range(n_cls) if counts[c][src] > 0),
            key=lambda c: (counts[c][src] - members[c].shape[0] * sizes[src] / total, -c),
        )
        counts[donor][src] -= 1
        counts[donor][dst] += 1
        guard += 1
        if guard > 5000:  # pragma: no cover
            raise RuntimeError("split repair failed to converge")

    buckets: dict[str, list[np.ndarray]] = {k: [] for k in order}
    for c in range(n_cls):
        start = 0
        for k in order:
            buckets[k].append(members[c][start:start + counts[c][k]])
            start += counts[c][k]
    out = {k: np.sort(np.concatenate(v)) for k, v in buckets.items()}
    got = {k: int(v.shape[0]) for k, v in out.items()}
    if got != sizes:
        raise RuntimeError(f"split sizes {got} != {sizes}")
    return out


# --- Fashion-MNIST loading ----------------------------------------------
def fashion_download(root: Path = DEFAULT_ROOT / "fashion_mnist") -> dict[str, str]:
    root = Path(root)
    digests = {}
    for name in FASHION_ARCHIVES.values():
        dest = root / name
        if not dest.exists():
            last = None
            for mirror in FASHION_MIRRORS:
                try:
                    _download(mirror + name, dest)
                    last = None
                    break
                except Exception as exc:  # pragma: no cover - network
                    last = exc
            if last is not None:
                raise RuntimeError(f"could not download {name}: {last}")
        digests[name] = _sha256(dest)
    return digests


def load_fashion(split: str, root: Path = DEFAULT_ROOT / "fashion_mnist", *,
                 unseal_token: str | None = None) -> VisionSplit:
    fashion_download(root)
    root = Path(root)
    if split == "test":
        if unseal_token != UNSEAL_TEST_TOKEN:
            raise PermissionError(
                "the Fashion-MNIST test split is evaluation-sealed; pass "
                "unseal_token=ecnca.vision.data.UNSEAL_TEST_TOKEN"
            )
        x = _read_idx(root / FASHION_ARCHIVES["test_images"])
        y = _read_idx(root / FASHION_ARCHIVES["test_labels"])
        return VisionSplit("test", "fashion_mnist",
                           (x / 255.0).astype(np.float32), y.astype(np.int64),
                           np.arange(y.shape[0], dtype=np.int64), FASHION_CLASSES)
    x = _read_idx(root / FASHION_ARCHIVES["train_images"])
    y = _read_idx(root / FASHION_ARCHIVES["train_labels"])
    idx = _stratified_indices(y.astype(np.int64), SPLIT_SIZES, SPLIT_SEED)[split]
    return VisionSplit(split, "fashion_mnist",
                       (x[idx] / 255.0).astype(np.float32), y[idx].astype(np.int64),
                       idx, FASHION_CLASSES)


# --- CIFAR-10 loading ----------------------------------------------------
def cifar10_download(root: Path = DEFAULT_ROOT / "cifar10", *,
                     verify: bool = True) -> str:
    root = Path(root)
    tar = root / "cifar-10-python.tar.gz"
    if not tar.exists():
        _download(CIFAR10_URL, tar)
    digest = _sha256(tar)
    if verify and digest != CIFAR10_SHA256:
        raise RuntimeError(
            f"CIFAR-10 checksum mismatch: expected {CIFAR10_SHA256}, got {digest}"
        )
    extracted = root / "cifar-10-batches-py"
    if not extracted.exists():
        with tarfile.open(tar) as t:
            t.extractall(root)
    return digest


def _cifar_batch(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as fh:
        d = pickle.load(fh, encoding="bytes")
    x = d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    return x, np.array(d[b"labels"], dtype=np.int64)


def load_cifar10(split: str, root: Path = DEFAULT_ROOT / "cifar10", *,
                 unseal_token: str | None = None,
                 verify: bool = True) -> VisionSplit:
    cifar10_download(root, verify=verify)
    base = Path(root) / "cifar-10-batches-py"
    if split == "test":
        if unseal_token != UNSEAL_TEST_TOKEN:
            raise PermissionError(
                "the CIFAR-10 test split is evaluation-sealed; pass "
                "unseal_token=ecnca.vision.data.UNSEAL_TEST_TOKEN"
            )
        x, y = _cifar_batch(base / "test_batch")
        return VisionSplit("test", "cifar10", (x / 255.0).astype(np.float32),
                           y, np.arange(y.shape[0], dtype=np.int64),
                           CIFAR10_CLASSES)
    xs, ys = [], []
    for i in range(1, 6):
        a, b = _cifar_batch(base / f"data_batch_{i}")
        xs.append(a)
        ys.append(b)
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    idx = _stratified_indices(y, CIFAR10_SPLIT_SIZES, SPLIT_SEED)[split]
    return VisionSplit(split, "cifar10", (x[idx] / 255.0).astype(np.float32),
                       y[idx], idx, CIFAR10_CLASSES)


# --- CIFAR-10-C loading --------------------------------------------------
def cifar10c_available(root: Path = DEFAULT_ROOT / "cifar10c") -> bool:
    root = Path(root)
    return (root / "CIFAR-10-C" / "labels.npy").exists() or (root / "labels.npy").exists()


def cifar10c_dir(root: Path = DEFAULT_ROOT / "cifar10c") -> Path:
    root = Path(root)
    inner = root / "CIFAR-10-C"
    return inner if (inner / "labels.npy").exists() else root


def load_cifar10c(
    corruption: str, severity: int, root: Path = DEFAULT_ROOT / "cifar10c",
    *, limit: int | None = None,
) -> VisionSplit:
    """One corruption at one severity.

    These are the official held-out test images corrupted. They are read for
    **measurement only**: no threshold, hyperparameter or architecture choice
    may be made from them. The gate uses them to size robustness headroom.
    """
    if corruption not in CIFAR10C_CORRUPTIONS:
        raise KeyError(f"unknown corruption {corruption!r}")
    if severity not in CIFAR10C_SEVERITIES:
        raise ValueError(f"severity must be 1-5, got {severity}")
    d = cifar10c_dir(root)
    xp, lp = d / f"{corruption}.npy", d / "labels.npy"
    if not xp.exists():
        raise FileNotFoundError(
            f"{xp} missing. Fetch CIFAR-10-C (2.9 GB) with "
            "ecnca.vision.datasets.cifar10c_download()"
        )
    x = np.load(xp, mmap_mode="r")
    y = np.load(lp)
    lo = (severity - 1) * 10_000
    hi = lo + (limit or 10_000)
    sl = slice(lo, min(hi, lo + 10_000))
    return VisionSplit(
        f"{corruption}_s{severity}", "cifar10c",
        (np.asarray(x[sl]) / 255.0).astype(np.float32),
        y[sl].astype(np.int64), np.arange(sl.start, sl.stop, dtype=np.int64),
        CIFAR10_CLASSES,
    )


def cifar10c_download(root: Path = DEFAULT_ROOT / "cifar10c") -> None:  # pragma: no cover
    """2.9 GB. Run on the compute node, not a laptop."""
    root = Path(root)
    tar = root / "CIFAR-10-C.tar"
    if not tar.exists():
        _download(CIFAR10C_URL, tar, timeout=3600)
    if not cifar10c_available(root):
        with tarfile.open(tar) as t:
            t.extractall(root)


# --- manifest ------------------------------------------------------------
def manifest(root: Path = DEFAULT_ROOT) -> dict:
    out: dict = {
        "split_seed": SPLIT_SEED,
        "split_sizes": {
            "mnist_and_fashion": dict(SPLIT_SIZES),
            "cifar10": dict(CIFAR10_SPLIT_SIZES),
        },
        "policy": (
            "Gate pilots use development data only. Official test splits are "
            "evaluation-sealed behind a token. CIFAR-10-C consists of the "
            "official held-out test images corrupted; it is read for "
            "measurement only and never for tuning."
        ),
        "fashion_mnist_scope_note": (
            "An earlier instruction excluded Fashion-MNIST unless a blocker "
            "made the chosen domain unusable. The headroom gate supersedes "
            "that: it enters as a candidate and is frozen only if it clears "
            "the gate thresholds."
        ),
    }
    try:
        out["fashion_mnist"] = {
            "archive_sha256": fashion_download(Path(root) / "fashion_mnist"),
            "classes": list(FASHION_CLASSES),
        }
        for s in ("train", "dev", "calib"):
            sp = load_fashion(s, Path(root) / "fashion_mnist")
            out["fashion_mnist"][s] = {
                "size": len(sp),
                "index_sha256": hashlib.sha256(
                    sp.indices.astype("<i8").tobytes()).hexdigest(),
            }
    except Exception as exc:
        out["fashion_mnist"] = {"status": f"unavailable: {exc}"}
    try:
        out["cifar10"] = {
            "archive_sha256": cifar10_download(Path(root) / "cifar10"),
            "classes": list(CIFAR10_CLASSES),
        }
        for s in ("train", "dev", "calib"):
            sp = load_cifar10(s, Path(root) / "cifar10")
            out["cifar10"][s] = {
                "size": len(sp),
                "index_sha256": hashlib.sha256(
                    sp.indices.astype("<i8").tobytes()).hexdigest(),
            }
    except Exception as exc:
        out["cifar10"] = {"status": f"unavailable: {exc}"}
    out["cifar10c"] = {
        "available": cifar10c_available(Path(root) / "cifar10c"),
        "corruptions": list(CIFAR10C_CORRUPTIONS),
        "severities": list(CIFAR10C_SEVERITIES),
        "note": "official held-out test images; measurement only",
    }
    return out
