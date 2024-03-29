import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import os
import warnings
import pickle
import InverseProblemWithDiffusionModel.helpers.pytorch_utils as ptu

from scipy.stats import spearmanr
from collections import defaultdict
from torchvision.utils import make_grid
from einops import rearrange
from InverseProblemWithDiffusionModel.helpers.utils import load_pickle, compute_angle, save_vol_as_gif
from InverseProblemWithDiffusionModel.helpers.metrics import (
    REGISTERED_METRICS,
    REGISTERED_METRICS_3D,
    compute_snr,
    compute_metrics
)
from typing import Dict, Union, List


FIGSIZE_UNIT = 3.6


def add_text(axis: plt.Axes, text_dict: dict):
    df = pd.DataFrame(text_dict)
    axis.table(cellText=np.round(df.values.T, decimals=3), rowLabels=df.columns, loc="center", cellLoc="center")
    axis.set_axis_off()


def add_correlation_table(axis: plt.Axes, mag_std: np.ndarray, phase_std: np.ndarray, abs_error: np.ndarray, abs_mag_error: np.ndarray):
    #               mag std | phase std
    # abs error
    # abs mag error
    row_texts = ["abs error", "abs mag error"]
    col_texts = ["mag std", "phase std"]
    corr_vals = np.zeros((2, 2))
    for i, x_iter in enumerate([abs_error, abs_mag_error]):
        for j, y_iter in enumerate([mag_std, phase_std]):
            # # Pearson
            # data_iter = np.stack([x_iter.flatten(), y_iter.flatten()], axis=0)  # (2, N)
            # corr_val = np.corrcoef(data_iter)[0, 1]
            # # if not np.isnan(corr_val):
            # corr_vals[i, j] = corr_val

            # Spearman rank
            res = spearmanr(x_iter.flatten(), y_iter.flatten())
            # print(res)
            corr_vals[i, j] = res.correlation

    axis.table(cellText=corr_vals.round(decimals=4), rowLabels=row_texts, colLabels=col_texts,
               loc="center", cellLoc="center")
    axis.set_axis_off()


def create_sample_grid_plot(root_dir: str, orig_filename="original.pt", recons_filename="reconstructions.pt",
                            args_filename="args_dict.pkl", *args, **kwargs):
    """
    All: grayscale images.

    kwargs: if_save, save_dir, metrics
    """
    def dict2title(args_dict: dict):
        # customize with LATEX here
        title = r"$\lambda = $" + f"{args_dict['lr_scaled']: .2E}, " + \
                r"$\alpha$ = " + f"{args_dict['step_lr']: .2E}, "

        # title = r"$reg\_weight = $" + f"{args_dict['reg_weight']: .2E}"

        return title

    img_orig = torch.load(os.path.join(root_dir, orig_filename))  # (1, C, H, W)
    recons = torch.load(os.path.join(root_dir, recons_filename))  # (B, C, H, W)
    img_orig = ptu.to_numpy(img_orig)
    recons = ptu.to_numpy(recons)
    args_dict = load_pickle(os.path.join(root_dir, args_filename))
    col0_titles = ["mag gt", "phase gt"]
    mean_col_titles = ["mean mag", "mean phase"]
    std_col_titles = ["std mag", "std phase"]
    sample_col_titles = ["mag", "phase", "abs diff", "abs mag diff"]
    snr_array = compute_snr(recons)  # (B,)
    num_samples = recons.shape[0]
    num_cols = 3 + num_samples  # +3: orig, mean & std
    num_rows = 5

    # compute all metrics
    metrics = kwargs.get("metrics", ["NRMSE", "SSIM"])
    metric_vals = {}
    metric_vals["SNR"] = snr_array  # (B,)
    # on mag image:
    metric_vals.update(compute_metrics(metrics, np.abs(recons), np.abs(img_orig)))
    mag_std = np.abs(recons).std(axis=0)
    phase_std = compute_angle(recons).std(axis=0)


    fig, axes = plt.subplots(num_rows, num_cols, figsize=(FIGSIZE_UNIT * num_cols, FIGSIZE_UNIT * num_rows))
    for j in range(axes.shape[1]):
        if j == 0:
            # orig
            handle = axes[0, j].imshow(np.abs(img_orig[0, 0]), cmap="gray")
            plt.colorbar(handle, ax=axes[0, j])
            axes[0, j].set_title(col0_titles[0])

            handle = axes[1, j].imshow(compute_angle(img_orig[0, 0]), cmap="gray")
            plt.colorbar(handle, ax=axes[1, j])
            axes[1, j].set_title(col0_titles[1])

            for i in range(num_rows):
                if i not in [0, 1]:
                    axes[i, j].set_axis_off()
                    # continue

        elif j == num_cols - 2:
            # mean
            handle = axes[0, j].imshow(np.abs(recons[:, 0, ...]).mean(axis=0), cmap="gray")
            plt.colorbar(handle, ax=axes[0, j])
            axes[0, j].set_title(mean_col_titles[0])

            handle = axes[1, j].imshow(compute_angle(recons[:, 0, ...]).mean(axis=0), cmap="gray")
            plt.colorbar(handle, ax=axes[1, j])
            axes[1, j].set_title(mean_col_titles[1])

            text_dict = {metric_name: [metric_vals[metric_name].mean()] for metric_name in metric_vals}
            # add_text(axes[-2, j], text_dict)
            add_text(axes[2, j], text_dict)

            for i in range(num_rows):
                # if i not in [0, 1, num_rows - 2]:
                if i not in [0, 1, 2]:
                    axes[i, j].set_axis_off()

        elif j == num_cols - 1:
            # std
            handle = axes[0, j].imshow(np.abs(recons[:, 0, ...]).std(axis=0), cmap="gray")
            plt.colorbar(handle, ax=axes[0, j])
            axes[0, j].set_title(std_col_titles[0])

            handle = axes[1, j].imshow(compute_angle(recons[:, 0, ...]).std(axis=0), cmap="gray")
            plt.colorbar(handle, ax=axes[1, j])
            axes[1, j].set_title(std_col_titles[1])

            # text_dict = {metric_name: [metric_vals[metric_name].std()] for metric_name in metric_vals}
            # # add_text(axes[-2, j], text_dict)
            # add_text(axes[3, j], text_dict)

            for i in range(num_rows):
                # if i not in [0, 1, num_rows - 2]:
                if i not in [0, 1]:
                    axes[i, j].set_axis_off()

        else:
            # for each sample reconstruction
            idx = j - 1
            imgs_iter = [
                np.abs(recons[idx, 0]),
                compute_angle(recons[idx, 0]),
                np.abs(recons[idx, 0] - img_orig[0, 0]),
                np.abs(np.abs(recons[idx, 0]) - np.abs(img_orig[0, 0]))
            ]
            for i, img_iter in enumerate(imgs_iter):
                handle = axes[i, j].imshow(img_iter, cmap="gray")
                plt.colorbar(handle, ax=axes[i, j])
                title_iter = sample_col_titles[i]
                # no need to print SNR on mag title
                # if i == 0:
                #     title_iter = f"{title_iter}: SNR = {snr_array[idx]: .2f} dB"
                axes[i, j].set_title(title_iter)

            # # last but second row: metrics
            # text_dict = {metric_name: [metric_vals[metric_name][idx]] for metric_name in metric_vals}
            # add_text(axes[-2, j], text_dict)

            # last row: correlation subplot
            add_correlation_table(axes[-1, j], mag_std, phase_std, imgs_iter[-2], imgs_iter[-1])

    sup_title = dict2title(args_dict)
    fig.suptitle(sup_title)
    fig.tight_layout()

    save_filename = "samples"
    save_filename += ".png"
    if kwargs.get("if_save", False):
        save_dir = kwargs.get("save_dir", root_dir)
        save_path = os.path.join(save_dir, save_filename)
        fig.savefig(save_path)
    else:
        plt.show()
    plt.close()

    return fig


def metric_vs_hyperparam(root_dirs: list, metrics: list, params: list, defaults: dict, if_logscale_x: list = [],
                         orig_filename="original.pt", recons_filename="reconstructions.pt",
                         args_filename="args_dict.pkl", *args, **kwargs):
    """
    Assuming all "root_dirs" contain results for the same setting, e.g. ACDC without phase update.

    "params" contains hyper-parameters for plotting.

    "if_logscale_x" contains hyper-parameters for setting log-scale for x axis.

    "defaults" should contain all hyper-parameters with default values. Each plot will only plot for one metric vs
    one hyper-parameters.

    kwargs: if_save, save_dir
    """
    def param2str(param_name: str):
        # customize names of hyper-paramters with LATEX here
        param2str_dict = {
            "step_lr": r"$\alpha$",
            "lr_scaled": r"$\lambda$"
        }

        # param2str_dict = {
        #     "reg_weight": r"reg_weight",
        #     "num_sens": r"num_sens"
        # }

        assert param_name in param2str_dict

        return param2str_dict[param_name]

    def dict2title(args_dict: dict):
        # customize with LATEX here
        title = "defaults: " + \
                r"$\lambda = $" + f"{args_dict['lr_scaled']: .2E}, " + \
                r"$\alpha$ = " + f"{args_dict['step_lr']: .2E}"
        
        # title = "defaults: " + \
        #         r"reg_weight = " + f"{args_dict['reg_weight']}, " + \
        #         r"num_sens = " + f"{args_dict['num_sens']}"

        return title

    def check_default_except_one(vals: tuple, idx: int, param_names: list):
        for i, val in enumerate(vals):
            if i == idx:
                continue
            if val != defaults[param_names[i]]:
                return False

        return True

    for param_iter in params:
        assert param_iter in defaults, f"{param_iter} is not valid."
    for metric_iter in metrics:
        assert metric_iter in REGISTERED_METRICS, f"Metric {metric_iter} is not suppoerted."

    num_rows = len(params)
    num_cols = len(metrics)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(FIGSIZE_UNIT * num_cols, FIGSIZE_UNIT * num_rows))
    if num_rows == 1 and num_cols == 1:
        axes = np.array([[axes]])
    else:
        if num_rows == 1:
            axes = axes[None, :]
        if num_cols == 1:
            axes = axes[:, None]

    # compute all metrics: {(param (float)...): {metric: float...}...}
    metric_vals = {}
    for root_dir in root_dirs:
        img_orig = torch.load(os.path.join(root_dir, orig_filename))  # (1, C, H, W)
        recons = torch.load(os.path.join(root_dir, recons_filename))  # (B, C, H, W)
        img_orig = ptu.to_numpy(img_orig)
        recons = ptu.to_numpy(recons)
        args_dict = load_pickle(os.path.join(root_dir, args_filename))
        param_vals = tuple([args_dict[param_name_iter] for param_name_iter in params])
        metric_dict = compute_metrics(metrics, np.abs(recons), np.abs(img_orig))
        metric_vals[param_vals] = metric_dict

    # creating plots
    for i, param_name_iter in enumerate(params):
        # ordering params: in accordance to "params"
        # one row for one hyper-parameters
        for j, metric_name_iter in enumerate(metrics):
            # one col for one metric
            axis = axes[i, j]
            axis.set_xlabel(param2str(param_name_iter))
            axis.set_ylabel(metric_name_iter)

            xx, yy = [], []
            for key in metric_vals:
                # key: Tuple[float]
                if not check_default_except_one(key, i, params):
                    continue
                xx.append(key[i])
                yy.append(metric_vals[key][metric_name_iter][0])
            xx = np.array(xx)
            yy = np.array(yy)
            idx_arg_sort = np.argsort(xx)
            xx = xx[idx_arg_sort]
            yy = yy[idx_arg_sort]
            print(f"{(i, j)}:\n{xx}\n{yy}")
            axis.plot(xx, yy)
            if param_name_iter in if_logscale_x:
                axis.set_xscale("log")

    sup_title = dict2title(defaults)
    fig.suptitle(sup_title)
    fig.tight_layout()

    if kwargs.get("if_save", False):
        save_dir = kwargs.get("save_dir", None)
        assert save_dir is not None
        save_path = os.path.join(save_dir, "metrics.png")
        fig.savefig(save_path)
    else:
        plt.show()

    plt.close()

    return fig


def metric_vs_one_hyperparam(root_dirs: list, metrics: list, param_tune: str, param_defaults: dict,
                             orig_filename="original.pt", recons_filename="reconstructions.pt",
                             args_filename="args_dict.pkl", selection_func = None, *args, **kwargs):
    """
    kwargs: if_logscale_x: bool, save_dir: str, save_filename: str, if_compute_metrics_only: bool
    metric_vals: {param: {metric: val...}...}
    """
    def selection(args_dict: dict):
        """
        Customize criterion for selecting parameters.
        """
        return True

    def dict2str(args_dict: dict):
        out_str = ""
        for i, (key, val) in enumerate(args_dict.items()):
            prefix = ", " if i > 0 else ""
            out_str += prefix + f"{key}: {val: .2e}"

        return out_str

    if selection_func is None:
        selection_func = selection

    metric_vals = defaultdict(dict)
    for root_dir_iter in root_dirs:
        print(f"current: {root_dir_iter}")
        img = torch.load(os.path.join(root_dir_iter, orig_filename))
        recons = torch.load(os.path.join(root_dir_iter, recons_filename))
        img = ptu.to_numpy(img)  # (T, C, H, W)
        recons = ptu.to_numpy(recons)[0]  # (T, C, H, W)

        with open(os.path.join(root_dir_iter, args_filename), "rb") as rf:
            args_dict = pickle.load(rf)
        if not selection_func(args_dict):
            continue
        local_metric_dict = {}
        for metric_iter in metrics:
            assert metric_iter in REGISTERED_METRICS_3D
            metric_func = REGISTERED_METRICS_3D[metric_iter]
            metric_vals[args_dict[param_tune]][metric_iter] = metric_func(np.abs(recons), np.abs(img))
            local_metric_dict[metric_iter] = metric_vals[args_dict[param_tune]][metric_iter]
        with open(os.path.join(root_dir_iter, "metrics.txt"), "w") as wf:
            wf.write(f"{local_metric_dict}")

    if kwargs.get("if_compute_metrics_only", False):
        return

    params = sorted(list(metric_vals.keys()))
    print(f"params:\n{params}")
    metric_dict = defaultdict(list)  # {metric: list...}
    for param in params:
        for metric_iter in metrics:
            metric_dict[metric_iter].append(metric_vals[param][metric_iter])

    fig, axes = plt.subplots(1, len(metrics), figsize=(FIGSIZE_UNIT * len(metrics), FIGSIZE_UNIT))
    if len(metrics) == 1:
        axes = [axes]
    for axis, metric in zip(axes, metrics):
        metric_list = metric_dict[metric]
        print(f"{metric}:\n{metric_list}")
        axis.plot(params, metric_list)
        axis.set_xlabel(param_tune)
        axis.set_title(metric)
        axis.grid(axis="y")
        if kwargs.get("if_logscale_x", True):
            axis.set_xscale("log")

    fig.suptitle(dict2str(param_defaults))
    fig.tight_layout()

    save_dir = kwargs.get("save_dir", None)
    if save_dir is not None:
        save_filename = kwargs.get("save_filename", f"{param_tune}.png")
        assert ".png" in save_filename
        fig.savefig(os.path.join(save_dir, save_filename))

    return fig


def grid_of_temporal_samples(x: Union[torch.Tensor, List[str]], num_rows: int,
                             save_dir: Union[str, None] = None, **kwargs) -> torch.Tensor:
    """
    x: filenames or tensor: (B, C, T)
    kwargs: padding
    """
    padding = kwargs.get("padding", 2)
    if isinstance(x, list):
       x = list(map(lambda path: torch.load(path), x)) # list[(1, C, T)]
       x = torch.cat(x, dim=0)  # (B, C, T)

    assert x.shape[0] % num_rows == 0
    x = torch.clip(x, -1., 1.)
    # x = (x - torch.min(x, dim=-1, keepdim=True)[0]) / (torch.max(x, dim=-1, keepdim=True)[0] -
    #                                                 torch.min(x, dim=-1, keepdim=True)[0])
    x = (x + 1) / 2
    x = rearrange(x, "B (kx ky) T -> B T kx ky", ky=np.sqrt(x.shape[1]).astype(int))
    x_grid = make_grid(x, nrow=num_rows, padding=padding)  # (T, H', W')

    if save_dir is not None:
        save_vol_as_gif(x_grid.unsqueeze(1), save_dir=save_dir, filename="samples.gif", if_normalize=False)

    # (T, H', W')
    return x_grid
