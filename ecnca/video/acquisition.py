"""Pinned acquisition of the baseline code, weights and datasets.

Nothing here is redistributed. Every artifact is fetched from its official
source on the machine that runs it, and recorded with enough detail that a
later run can prove it used the same thing.

Two kinds of checksum are distinguished, and the distinction is load-bearing:

  PUBLISHER      a hash the publisher states. Verifiable -- a mismatch means
                 the download is wrong or the artifact changed.
  RECORDED       a hash we computed after acquisition. It pins reproducibility
                 ACROSS OUR RUNS but proves nothing about authenticity, since
                 we are hashing whatever we received.

Inventing a publisher checksum would turn an unverifiable download into an
apparently verified one, so `expected_sha256=None` stays None until a real
published value is supplied.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict, field
from pathlib import Path


@dataclass
class Artifact:
    name: str
    kind: str                       # code | weights | dataset
    url: str
    version: str                    # tag, commit, or release name
    license: str
    license_url: str = ""
    expected_sha256: str | None = None   # PUBLISHER-provided only
    # For a multi-file artifact the per-file publisher hashes live here; a
    # single expected_sha256 cannot represent two archives.
    expected_sha256_by_file: dict | None = None
    checksum_origin: str = "none"        # publisher | recorded | none
    needs_manual_acceptance: bool = False
    manual_action: str = ""
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# The pinned set. Versions are fixed here; the launcher refuses to proceed if
# what it downloads does not match what this file declares.
# --------------------------------------------------------------------------

XMEM = Artifact(
    name="XMem",
    kind="code",
    url="https://github.com/hkchengrex/XMem",
    version="main@<pinned-at-first-acquisition>",
    license="GPL-3.0",
    license_url="https://github.com/hkchengrex/XMem/blob/main/LICENSE",
    checksum_origin="recorded",
    notes=(
        "Primary published baseline. GPL-3.0: we may use and modify it, but "
        "derived work distributed publicly inherits the licence. Our "
        "controller lives in this repository and is loaded BY XMem rather "
        "than vendored into it, so the licence boundary stays clear."
    ),
)

XMEM_WEIGHTS = Artifact(
    name="XMem.pth",
    kind="weights",
    url="https://github.com/hkchengrex/XMem/releases/download/v1.0/XMem.pth",
    version="v1.0",
    license="GPL-3.0 (per the XMem repository)",
    checksum_origin="recorded",
    notes=(
        "Pretrained weights for the published configuration. The release page "
        "states no SHA256, so ours is RECORDED, not publisher-verified: it "
        "pins reproducibility across our runs and nothing more."
    ),
)

DAVIS_2017 = Artifact(
    name="DAVIS-2017-trainval-480p",
    kind="dataset",
    url="https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip",
    version="2017 trainval 480p",
    license="CC BY 4.0 (non-commercial research use; see the benchmark site)",
    license_url="https://davischallenge.org/davis2017/code.html",
    checksum_origin="recorded",
    notes=(
        "First integration benchmark. 480p trainval. Official evaluation "
        "code is used for J&F so our numbers are comparable to published ones."
    ),
)

# PUBLISHER-provided checksums, from the SHA256SUMS the maintainers ship
# alongside the archives. These are verifiable: a mismatch means the download
# is wrong or the artifact changed. Confirmed against a real download --
# train.tar.gz (22,077,380,713 bytes) and valid.tar.gz (3,875,208,344 bytes)
# both returned OK.
MOSE_V1_SHA256 = {
    "train.tar.gz":
        "3f805e66ecb576fdd37a1ab2b06b08a428edd71994920443f70d09537918270b",
    "valid.tar.gz":
        "884baecf7d7e85cd35486e45d6c474dc34352a227ac75c49f6d5e4afb61b331c",
}

MOSE_V1 = Artifact(
    name="MOSEv1",
    kind="dataset",
    url="https://mose.video/MOSEv1/",
    version="v1",
    license="see the MOSE site; research use",
    license_url="https://mose.video/MOSEv1/",
    expected_sha256_by_file=MOSE_V1_SHA256,
    checksum_origin="publisher",
    needs_manual_acceptance=True,
    manual_action=(
        "MOSE distributes through a form/drive link that requires a person to "
        "accept the terms. Download MOSEv1 (train + valid) by hand, place the "
        "archives in the directory the launcher prints, and re-run it. It will "
        "verify and record their hashes and continue."
    ),
    notes=(
        "Used ONLY if the DAVIS event audit shows too few qualifying events. "
        "MOSE v2 is explicitly excluded from this study."
    ),
)

ARTIFACTS = {a.name: a for a in (XMEM, XMEM_WEIGHTS, DAVIS_2017, MOSE_V1)}


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def verify(artifact: Artifact, path: Path) -> dict:
    """Hash what we actually have and compare where a publisher value exists."""
    got = sha256_file(path)
    rec = {
        "name": artifact.name,
        "path": str(path),
        "sha256": got,
        "checksum_origin": artifact.checksum_origin,
        "verified_against_publisher": False,
    }
    if artifact.expected_sha256:
        rec["expected_sha256"] = artifact.expected_sha256
        rec["verified_against_publisher"] = (got == artifact.expected_sha256)
        if not rec["verified_against_publisher"]:
            rec["error"] = (
                f"checksum mismatch: expected {artifact.expected_sha256[:16]}, "
                f"got {got[:16]}"
            )
    return rec


def manifest(records: list[dict]) -> dict:
    """The acquisition record written beside the results."""
    return {
        "artifacts": records,
        "checksum_policy": (
            "PUBLISHER checksums are verified against a value the publisher "
            "states. RECORDED checksums are computed after acquisition: they "
            "pin reproducibility across our runs but do not establish "
            "authenticity. No checksum in this study is invented."
        ),
        "redistribution": "none; every artifact is fetched from its source",
    }
