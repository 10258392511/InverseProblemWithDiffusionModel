import numpy as np
import matplotlib.pyplot as plt
import torch

from torch.utils.data import Dataset
from InverseProblemWithDiffusionModel.helpers.utils import vis_signals


def select_at_idx(ds: Dataset, ds_name: str, idx: int):
    if "CINE" in ds_name:
        return ds[idx][0]

    elif ds_name == "SanityCheck1D":
        return ds[idx][0]
    
    raise NotImplementedError


def compute_max_dist(ds: Dataset, ds_name: str, max_num_pairs=1000, norm_order=2, verbose=False, num_prints=10):
    max_dist = -1
    print_interval = max_num_pairs // num_prints
    for i in range(max_num_pairs):
        if verbose:
            if i % print_interval == 0:
                print(f"current: {i + 1}/{max_num_pairs}")
        idx1, idx2 = np.random.choice(len(ds), (2,), replace=False)
        dist_cand = torch.norm(select_at_idx(ds, ds_name, idx1) - select_at_idx(ds, ds_name, idx2), p=norm_order).item()
        if dist_cand > max_dist:
            max_dist = dist_cand
    
    return max_dist


def compute_norm_hist(ds: Dataset, ds_name: str, tfm: str, norm_order=2, normalized=True, bins=50, if_plot=True, verbose=False, num_prints=10, figsize=(3.6, 3.6)):
    """
    normalized: divide norm by number of entries
    """
    tfm_func = REGISTERED_SIGNAL_TFM[tfm]
    norm_list = []
    print_interval = len(ds) // num_prints
    for i in range(len(ds)):
        if verbose:
            if i % print_interval == 0:
                print(f"current: {i + 1}/{len(ds)}")
        sample = select_at_idx(ds, ds_name, i)
        norm_iter = tfm_func(sample).item()

        if normalized:
            norm_iter /= sample.numel()
        
        # # ### debug only ###
        # th = 0.02
        # if norm_iter >= th:
        #     vis_signals(
        #         sample, 
        #         if_save=False,
        #         save_dir=f"/scratch/zhexwu/outputs/ds_stats/CINE64_1D/{tfm}/th_{str(th).replace('.', '_')}/",
        #         filename=f"sample_{i}.png")
        #     print("-" * 100)
        # ##################

        norm_list.append(norm_iter)
    
    norm_list = np.array(norm_list)
    hist, bin_edges = np.histogram(norm_list, bins=bins)
    if if_plot:
        fig, axis = plt.subplots(figsize=figsize)
        axis.hist(bin_edges[:-1], bin_edges, weights=hist, edgecolor="k")
        axis.set_title(f"{ds_name}, {len(ds)} samples")
        plt.show()

        return hist, bin_edges, fig
    
    return hist, bin_edges, None 


def signal_transform_norm(x: torch.Tensor, **kwargs):
    p = kwargs.get("p", 2)

    return torch.norm(x, p=p)


def signal_transform_TV(x: torch.Tensor, **kwargs):
    # x: (C, L)
    x_shift = torch.roll(x, -1, dims=1)
    x_diff = x - x_shift
    x_tv = torch.norm(x_diff, p=1)

    return x_tv

REGISTERED_SIGNAL_TFM = {
    "norm": signal_transform_norm,
    "TV": signal_transform_TV
}

def count_samples(counts: np.ndarray, bin_edges: np.ndarray, thresh: float):
    """
    Count number of samples on both sides of "thresh".
    """
    bin_edges = bin_edges[1:]
    mask_leq = bin_edges <= thresh
    num_leq = counts[mask_leq].sum()
    num_gt = counts[~mask_leq].sum()

    return num_leq, num_gt
