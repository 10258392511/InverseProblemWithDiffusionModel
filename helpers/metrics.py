import numpy as np

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
from collections import defaultdict
from typing import Dict, List


def add_first_channel(img: np.ndarray) -> np.ndarray:
    # (C, H, W) -> (1, C, H, W)
    if img.ndim == 3:
        img = img[None, ...]
    elif img.ndim == 4:
        pass
    else:
        raise ValueError("Input must have 3 or 4 dimensions.")
    
    return img


def compute_metrics(metric_names: List[str], img: np.ndarray, img_orig: np.ndarray) -> Dict[str, float]:
    out_dict = defaultdict(list)
    img = add_first_channel(img)
    img_orig = add_first_channel(img_orig)

    for metric_name in metric_names:
        assert metric_name in REGISTERED_METRICS
        metric_func = REGISTERED_METRICS[metric_name]
        for i in range(img.shape[0]):
            val = metric_func(img[i], img_orig[i])
            out_dict[metric_name].append(val)
    
    return out_dict


def MAE(img: np.ndarray, img_orig: np.ndarray) -> float:
    # img, img_orig: (C, H, W)
    error = np.abs(img - img_orig).mean()

    return error


def SSIM_wrapper(img: np.ndarray, img_orig: np.ndarray) -> float:
    # img, img_orig: (C, H, W)
    num_channels = img.shape[0]
    if_multi_channel = False
    if num_channels > 1:
        if_multi_channel = True

    if if_multi_channel:
        val = ssim(img, img_orig, channel_axis=0, multichannel=if_multi_channel)
    else:
        val = ssim(img[0], img_orig[0])

    return val


REGISTERED_METRICS = {
    "L2": mean_squared_error,
    "L1": MAE,
    "SSIM": SSIM_wrapper
}
