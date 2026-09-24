"""Input preprocessing, matched to the pinned official XMem pipeline.

Official inference applies `ToTensor()`, ImageNet normalization, and then
resizes the shorter side to 480 pixels (`eval.py --size`, default 480;
`inference/data/video_reader.py`). Ours originally omitted normalization and,
until version 3, also fed MOSE at native resolution.  The latter makes XMem's
object-group activations explode with both pixels and object count (a sampled
1080p/five-object clip exceeded a 40 GiB MIG slice).

    official  mean 0.6907  std 1.1350  range [-2.118, 2.640]
    ours      mean 0.6053  std 0.2646  range [ 0.000,  1.000]
    max |difference| 2.1179

Every feature, memory readout and score derived from those inputs. This module
is the single definition, used by integration, controls, training and
evaluation alike, and its version is recorded in cache and checkpoint
provenance so scores computed under different inputs are never mixed.
"""

from __future__ import annotations

import numpy as np
import torch

# Bump when the transform changes: recorded in provenance so a score computed
# under a different input pipeline can never be reused.
PREPROCESS_VERSION = 3

# Official XMem eval.py default.  -1 would mean native resolution; this study
# uses the published/default inference regime instead.
INFERENCE_SHORT_SIDE = 480

# Verbatim from dataset/range_transform.py at the pinned revision.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_MEAN = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
_STD = torch.tensor(IMAGENET_STD).view(3, 1, 1)


def image_to_tensor(array: np.ndarray) -> torch.Tensor:
    """HxWx3 uint8 -> normalized 1x3xHxW float, as official inference does.

    Equivalent to `transforms.Compose([ToTensor(), im_normalization])`:
    ToTensor scales uint8 to [0,1] and moves channels first; im_normalization
    subtracts the ImageNet mean and divides by its standard deviation.
    """
    t = torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1).float()
    t = t.div(255.0)
    t = (t - _MEAN.to(t.dtype)) / _STD.to(t.dtype)
    return t.unsqueeze(0)


def mask_to_tensor(array: np.ndarray) -> torch.Tensor:
    """HxW integer label map -> 1xHxW long. Masks are NOT normalized.

    The official reader passes palette masks through as raw uint8 arrays; they
    are object ids, not intensities.
    """
    return torch.from_numpy(np.ascontiguousarray(array).astype(np.int64)
                            ).unsqueeze(0)


def resized_shape(height: int, width: int,
                  short_side: int = INFERENCE_SHORT_SIDE) -> tuple[int, int]:
    """XMem's short-side resize geometry (aspect ratio preserved)."""
    if short_side < 0:
        return int(height), int(width)
    m = min(int(height), int(width))
    if m <= 0:
        raise ValueError(f"invalid image shape {(height, width)}")
    if height <= width:
        return int(short_side), int(width / height * short_side)
    return int(height / width * short_side), int(short_side)


def resize_image_tensor(tensor: torch.Tensor,
                        short_side: int = INFERENCE_SHORT_SIDE) -> torch.Tensor:
    """Resize normalized BCHW input exactly where official XMem does it."""
    import torch.nn.functional as F

    size = resized_shape(*tensor.shape[-2:], short_side)
    if tuple(tensor.shape[-2:]) == size:
        return tensor
    return F.interpolate(tensor, size=size, mode="bilinear",
                         align_corners=False, antialias=True)


def resize_mask_tensor(tensor: torch.Tensor,
                       short_side: int = INFERENCE_SHORT_SIDE) -> torch.Tensor:
    """Resize integer object ids with nearest-neighbour interpolation."""
    import torch.nn.functional as F

    size = resized_shape(*tensor.shape[-2:], short_side)
    if tuple(tensor.shape[-2:]) == size:
        return tensor
    dtype = tensor.dtype
    out = F.interpolate(tensor.unsqueeze(0).float(), size=size, mode="nearest")
    return out[0].to(dtype)


def frame_and_mask_to_tensors(image: np.ndarray, mask: np.ndarray,
                              short_side: int = INFERENCE_SHORT_SIDE
                              ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return official-resolution image/mask plus the untouched GT mask.

    XMem consumes the resized pair.  Evaluation restores probabilities to the
    untouched mask resolution before argmax, as official eval.py does.
    """
    if tuple(image.shape[:2]) != tuple(mask.shape[:2]):
        raise ValueError(
            f"image/mask geometry differs: {image.shape[:2]} vs {mask.shape[:2]}")
    original = mask_to_tensor(mask)
    image_t = resize_image_tensor(image_to_tensor(image), short_side)
    mask_t = resize_mask_tensor(original, short_side)
    return image_t, mask_t, original


def official_transform():
    """The official transform object, for parity testing.

    Imported lazily so the pipeline does not depend on the XMem checkout being
    importable at module load.
    """
    import torchvision.transforms as transforms
    from dataset.range_transform import im_normalization

    return transforms.Compose([transforms.ToTensor(), im_normalization])


def parity(array: np.ndarray) -> dict:
    """Numerical comparison against the official transform."""
    from PIL import Image

    ours = image_to_tensor(array)[0]
    official = official_transform()(Image.fromarray(array))
    diff = (ours - official).abs()
    return {
        "max_abs_difference": float(diff.max()),
        "mean_abs_difference": float(diff.mean()),
        "ours": {"mean": float(ours.mean()), "std": float(ours.std()),
                 "min": float(ours.min()), "max": float(ours.max())},
        "official": {"mean": float(official.mean()),
                     "std": float(official.std()),
                     "min": float(official.min()), "max": float(official.max())},
        "matches": bool(diff.max() < 1e-5),
        "preprocess_version": PREPROCESS_VERSION,
    }
