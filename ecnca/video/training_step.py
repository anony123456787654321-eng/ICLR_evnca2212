"""Shared optimizer step for training and workload measurement."""
import torch

from ecnca.video.losses import clip_segmentation_loss
from ecnca.video.integration import persistent_memory_bytes
from ecnca.video.profile import read_memory


def train_clip_step(xmem, seam, clip, device, optimizer, record=None):
    record = {} if record is None else record
    seam.reset()
    optimizer.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    record.update(video=clip.video, frames=list(clip.frame_indices),
                  objects=list(clip.object_ids),
                  frame_shape=list(clip.frames[0].shape),
                  before_forward=read_memory(device).to_dict())
    loss = None
    try:
        loss = clip_segmentation_loss(xmem, seam, clip, device)
        record["before_backward"] = read_memory(device).to_dict()
        loss.backward()
        parameters = [p for group in optimizer.param_groups for p in group["params"]]
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        return float(loss.detach())
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        # State and gradients must not overlap the next clip's forward pass.
        record["persistent_memory"] = persistent_memory_bytes(None, seam.state)
        loss = None
        seam.reset()
        optimizer.zero_grad(set_to_none=True)
        record["after_update"] = read_memory(device).to_dict()
        if device.type == "cuda":
            record["peak_allocated_mib"] = torch.cuda.max_memory_allocated(device) / 2**20
            record["peak_reserved_mib"] = torch.cuda.max_memory_reserved(device) / 2**20
