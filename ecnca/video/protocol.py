"""The frozen protocol for the video feasibility study.

Written before method training and hashed into the results. Changing any
threshold after seeing method results invalidates the comparison, which is
why they live in one file that is recorded rather than in scattered defaults.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ecnca.video import events as ev
from ecnca.video import metrics as mt

# -- advancement criteria, frozen (brief section 8) ------------------------
MIN_REAPPEAR_JF_GAIN = 3.0     # absolute J&F points over XMem at matched memory
MAX_OVERALL_JF_LOSS = 1.0      # absolute J&F points overall
REQUIRE_NCA_OVER_NONRECURRENT = True

# -- event sufficiency, frozen BEFORE any method result (decision 1) -------
# The DAVIS-vs-MOSE pivot is decided on annotation coverage and statistical
# adequacy alone. It must not depend on whether our method wins, so it is
# fixed here and evaluated by the audit stage before training.
MIN_EVENTS_TOTAL = 60
MIN_EVENT_VIDEOS = 20
MIN_EVENTS_EVAL_SPLIT = 25

# -- compute budget (brief section 9) --------------------------------------
TRAINING_GPU_HOURS_CAP = 4.0   # the WHOLE initial campaign, not per arm
                               # and not renewed by pivoting datasets

ARMS = ("xmem", "non_recurrent", "nca", "nca_no_hypotheses", "recent_only")


def event_sufficiency(audit: dict) -> dict:
    """Is this dataset adequate for the event experiment?

    Evaluated on the annotation audit only -- no model has run.
    """
    # The audit's own key. A mismatch here would silently read every dataset
    # as having zero events and pivot unconditionally, so both spellings are
    # accepted and the canonical one is first.
    n_events = int(audit.get("events_total", audit.get("total_events", 0)))
    n_videos = int(audit.get("videos_with_events", 0))
    n_eval = int(audit.get("events_by_split", {}).get("eval", 0))
    checks = {
        "events_total": (n_events, MIN_EVENTS_TOTAL, n_events >= MIN_EVENTS_TOTAL),
        "videos_with_events": (n_videos, MIN_EVENT_VIDEOS,
                               n_videos >= MIN_EVENT_VIDEOS),
        "events_in_eval_split": (n_eval, MIN_EVENTS_EVAL_SPLIT,
                                 n_eval >= MIN_EVENTS_EVAL_SPLIT),
    }
    passed = all(c[2] for c in checks.values())
    return {
        "sufficient": passed,
        "checks": {k: {"observed": v[0], "required": v[1], "passed": v[2]}
                   for k, v in checks.items()},
        "decision": (
            "dataset is adequate for the event experiment"
            if passed else
            "too few ELIGIBLE events after the real evaluation requirements "
            "(preceding history and a scorable recovery window); this "
            "dataset supports integration but not the event comparison"
        ),
        "basis": (
            "annotation coverage and statistical adequacy only. No method "
            "result is consulted."
        ),
    }


def freeze(out: Path, *, dataset: str, artifacts: list[dict],
           audit: dict, extra: dict | None = None) -> dict:
    """Write the protocol. Everything scored later is declared here."""
    proto = {
        "study": "NCA memory for video object segmentation (feasibility)",
        "objective": (
            "Whether a local recurrent update with competing hypotheses "
            "improves memory use during annotated disappearance and "
            "reappearance, against a strong published baseline."
        ),
        "propositions": {
            "memory_selection": "several retained observations beat recent-only",
            "competing_hypotheses": "alternatives help resolve ambiguity",
            "local_recurrence": "the NCA update beats a matched non-recurrent controller",
        },
        "dataset": dataset,
        "evaluation_population": (extra or {}).pop("evaluation_population", {}),
        "artifacts": artifacts,
        "arms": list(ARMS),
        "event_definition": {
            "min_present_before": ev.MIN_PRESENT_BEFORE,
            "min_absence": ev.MIN_ABSENCE,
            "min_reappear": ev.MIN_REAPPEAR,
            "recovery_window": ev.RECOVERY_WINDOW,
            "min_pixels": ev.MIN_PIXELS,
            "terminology": (
                "annotated disappearance/reappearance. Missing annotation is "
                "NOT confirmed physical occlusion."
            ),
        },
        "metrics": {
            "overall": "official per-video J&F",
            "primary_mechanism_endpoint": "reappearance-window J&F",
            "aggregation": (
                "events averaged WITHIN a video before averaging across "
                "videos, so a video with many events cannot dominate"
            ),
            "recovery_iou": mt.RECOVERY_IOU,
            "recovery_consecutive": mt.RECOVERY_CONSECUTIVE,
            "unrecovered_events": "reported explicitly, never dropped",
            "abstaining": "scored as a MISS, never as identity preservation",
        },
        "advancement_criteria": {
            "min_reappear_jf_gain": MIN_REAPPEAR_JF_GAIN,
            "max_overall_jf_loss": MAX_OVERALL_JF_LOSS,
            "require_nca_over_nonrecurrent": REQUIRE_NCA_OVER_NONRECURRENT,
            "note": "advancement thresholds, not promised outcomes",
        },
        "event_sufficiency": event_sufficiency(audit),
        "uncertainty": {
            "unit": "video",
            "intervals": "video-clustered paired",
            "seed_vs_sampling": "training-seed uncertainty reported separately",
            "status": "single-seed feasibility results remain EXPLORATORY",
        },
        "budget": {
            "training_gpu_hours_cap": TRAINING_GPU_HOURS_CAP,
            "scope": "the whole initial campaign; not per arm, not renewed by a pivot",
            "on_cap": "save and report an incomplete result; never silently shorten",
        },
        "splits": {"unit": "video", "held_out": "never used for training, "
                   "calibration or debugging"},
        "excluded": {"MOSE v2": "explicitly out of scope for this study"},
    }
    if extra:
        proto.update(extra)
    blob = json.dumps(proto, indent=2, sort_keys=True, default=str)
    proto["self_sha256"] = hashlib.sha256(blob.encode()).hexdigest()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(proto, indent=2, sort_keys=True, default=str) + "\n")
    return proto
