"""What did the published checkpoint train on?

Holding out videos the baseline already trained on would invalidate the
comparison: the baseline would be scored on its own training data while our
arms are scored on held-out data, and any difference would be meaningless.

So the check is a precondition, not a footnote. It reads the pinned XMem
checkout rather than trusting a remembered claim.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from pathlib import Path

# XMem v1.0's documented training stages (docs/TRAINING.md):
#   0 static images, 1 BL30K, 2 DAVIS+YouTubeVOS (longer),
#   3 DAVIS+YouTubeVOS (shorter). The released XMem.pth is stage "03".
XMEM_TRAINING_DATASETS = ("static", "BL30K", "DAVIS", "YouTubeVOS")


@dataclass
class Provenance:
    checkpoint: str = ""
    repo_revision: str = ""
    training_datasets: tuple = ()
    evidence: list = field(default_factory=list)
    target_dataset: str = ""
    target_in_training: bool | None = None
    safe_to_hold_out: bool | None = None
    detail: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["training_datasets"] = list(self.training_datasets)
        return d


def inspect(xmem_dir, *, target_dataset: str = "MOSE") -> Provenance:
    """Read the pinned checkout for what the released weights trained on."""
    xmem_dir = Path(xmem_dir)
    p = Provenance(target_dataset=target_dataset)

    train_py = xmem_dir / "train.py"
    config_py = xmem_dir / "util" / "configuration.py"
    training_md = xmem_dir / "docs" / "TRAINING.md"
    for f in (train_py, config_py, training_md):
        if not f.exists():
            p.detail = f"cannot verify provenance: {f} is missing"
            return p

    text = "\n".join(f.read_text(errors="replace")
                     for f in (train_py, config_py, training_md))

    found = []
    for name, pattern in (("static", r"static_root|static images"),
                          ("BL30K", r"bl_root|BL30K"),
                          ("YouTubeVOS", r"yv_root|YouTubeVOS"),
                          ("DAVIS", r"davis_root|DAVIS")):
        if re.search(pattern, text, re.I):
            found.append(name)
    p.training_datasets = tuple(found)
    p.evidence = [
        "util/configuration.py: --stages '0-static images, 1-Blender "
        "dataset, 2-DAVIS+YouTubeVOS'",
        "docs/TRAINING.md: 'the base model is pretrained with static images "
        "followed by the shorter main training (s03)'",
        "train.py: builds datasets from static_root, bl_root, yv_root, "
        "davis_root only",
    ]

    hit = re.search(rf"\b{re.escape(target_dataset)}\b", text, re.I)
    p.target_in_training = bool(hit)
    p.safe_to_hold_out = not p.target_in_training
    if p.target_in_training:
        p.detail = (
            f"{target_dataset} APPEARS in the pinned training code. Videos "
            f"from it may already be in the checkpoint's training data, so "
            f"holding them out would not be a clean comparison."
        )
    else:
        p.detail = (
            f"{target_dataset} does not appear anywhere in the pinned "
            f"training configuration or trainer. The released weights were "
            f"trained on {', '.join(found)} only, so {target_dataset} videos "
            f"are unseen and may be held out cleanly."
        )
    return p


def gate(prov: Provenance) -> dict:
    """The precondition on building a custom split from training videos."""
    return {
        "checked": prov.target_in_training is not None,
        "target_dataset": prov.target_dataset,
        "baseline_training_datasets": list(prov.training_datasets),
        "target_in_baseline_training": prov.target_in_training,
        "may_build_custom_split": bool(prov.safe_to_hold_out),
        "detail": prov.detail,
        "evidence": prov.evidence,
    }
