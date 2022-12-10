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


def compute_mean_and_std(imgs: np.ndarray):
    # img: (B, C, H, W)
    assert imgs.shape[0] > 1
    if imgs.dtype not in (np.complex, np.complex64):
        mean_img = np.mean(imgs, axis=0)
        std_img = np.std(np.abs(imgs), axis=0)
        # (C, H, W) each
        return mean_img, std_img
    else:
        imgs_mag = np.abs(imgs)
        imgs_phase = np.angle(imgs)
        mag_mean, mag_std = compute_mean_and_std(imgs_mag)
        phase_mean, phase_std = compute_mean_and_std(imgs_phase)

        # (C, H, W) each
        return mag_mean, phase_mean, mag_std, phase_std


REGISTERED_METRICS = {
    "L2": mean_squared_error,
    "L1": MAE,
    "SSIM": SSIM_wrapper
}
