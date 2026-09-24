"""Deterministic execution, configured BEFORE CUDA initialises.

Bit-identical resume is only meaningful if two identical runs are themselves
bit-identical. On CUDA they are not by default: cuDNN benchmarking picks
algorithms by timing them, and several reductions accumulate in a
nondeterministic order. Those choices are made when CUDA initialises, so the
configuration has to happen before the first CUDA call -- importing this
module late, or calling it after a tensor has reached the device, silently
does nothing.

`CUBLAS_WORKSPACE_CONFIG` is read by cuBLAS at initialisation, so it must be
in the environment before then; setting it afterwards is ignored, which is why
this is an environment variable rather than a torch flag.

The same settings apply to uninterrupted runs, resumed runs and production
training. Using them only in the diagnostic would test a configuration that
the real training never runs under.
"""

from __future__ import annotations

import os

# Must be set before torch initialises cuBLAS. ":4096:8" is the setting that
# permits deterministic reductions while keeping a usable workspace; ":16:8"
# also works but is slower.
CUBLAS_WORKSPACE_CONFIG = ":4096:8"

_STATE: dict = {}


def configure(*, warn_only: bool = False) -> dict:
    """Configure deterministic execution. Call before any CUDA work.

    Returns the settings actually applied, so a run can RECORD what it ran
    under rather than assume.
    """
    applied: dict = {}

    prior = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if prior is None:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG
        applied["cublas_workspace_config"] = CUBLAS_WORKSPACE_CONFIG
        applied["cublas_workspace_preexisting"] = False
    else:
        # Respect an operator's explicit choice, but record that it differs.
        applied["cublas_workspace_config"] = prior
        applied["cublas_workspace_preexisting"] = True

    import torch

    if torch.cuda.is_available() and torch.cuda.is_initialized():
        # Not fatal, but the guarantee is weaker than the caller believes and
        # saying so is the whole point of this module.
        applied["warning"] = (
            "CUDA was already initialised when determinism was configured; "
            "cuBLAS may have read its workspace setting already"
        )

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    applied["cudnn_benchmark"] = False
    applied["cudnn_deterministic"] = True

    torch.use_deterministic_algorithms(True, warn_only=warn_only)
    applied["use_deterministic_algorithms"] = True
    applied["warn_only"] = bool(warn_only)

    applied["torch_version"] = torch.__version__
    applied["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        applied["cuda_version"] = torch.version.cuda
        applied["cudnn_version"] = torch.backends.cudnn.version()
        applied["device_count"] = torch.cuda.device_count()

    _STATE.clear()
    _STATE.update(applied)
    return dict(applied)


def settings() -> dict:
    """What was applied, for recording in a checkpoint or a result file."""
    return dict(_STATE)
