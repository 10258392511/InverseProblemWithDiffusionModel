import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import warnings
import InverseProblemWithDiffusionModel.helpers.pytorch_utils as ptu

from InverseProblemWithDiffusionModel.helpers.utils import load_pickle, compute_angle
from InverseProblemWithDiffusionModel.helpers.metrics import REGISTERED_METRICS, compute_snr, compute_metrics


FIGSIZE_UNIT = 3.6


def create_sample_grid_plot(root_dir: str, orig_filename="original.pt", recons_filename="reconstructions.pt",
                            args_filename="args_dict.pkl", *args, **kwargs):
    """
    All: grayscale images.

    kwargs: if_save, save_dir
    """
    def dict2title(args_dict: dict):
        # customize with LATEX here
        title = r"$\lambda = $" + f"{args_dict['lr_scaled']}, " + \
                r"$\alpha$ = " + f"{args_dict['step_lr']}, " + \
                r"$\sigma_{data}^2 = $" + f"{args_dict['lamda']}"

        return title

    img_orig = torch.load(os.path.join(root_dir, orig_filename))  # (1, C, H, W)
    recons = torch.load(os.path.join(root_dir, recons_filename))  # (B, C, H, W)
    img_orig = ptu.to_numpy(img_orig)
    recons = ptu.to_numpy(recons)
    args_dict = load_pickle(os.path.join(root_dir, args_filename))
    col0_titles = ["mag gt", "phase gt"]
    last_col_titles = ["mean mag", "mean phase"]
    sample_col_titles = ["mag", "phase", "abs diff", "abs mag diff"]
    snr_array = compute_snr(recons)  # (B,)
    num_samples = recons.shape[0]
    num_cols = 2 + num_samples
    num_rows = 4

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(FIGSIZE_UNIT * num_cols, FIGSIZE_UNIT * num_rows))
    for j in range(axes.shape[1]):
        if j == 0:
            handle = axes[0, j].imshow(np.abs(img_orig[0, 0]), cmap="gray")
            plt.colorbar(handle, ax=axes[0, j])
            axes[0, j].set_title(col0_titles[0])

            handle = axes[1, j].imshow(compute_angle(img_orig[0, 0]), cmap="gray")
            plt.colorbar(handle, ax=axes[1, j])
            axes[1, j].set_title(col0_titles[1])

            for i in [2, 3]:
                axes[i, j].set_axis_off()
                continue


        elif j == num_cols - 1:
            handle = axes[0, j].imshow(np.abs(recons[:, 0, ...]).mean(axis=0), cmap="gray")
            plt.colorbar(handle, ax=axes[0, j])
            axes[0, j].set_title(last_col_titles[0])

            handle = axes[1, j].imshow(compute_angle(recons[:, 0, ...]).mean(axis=0), cmap="gray")
            plt.colorbar(handle, ax=axes[1, j])
            axes[1, j].set_title(last_col_titles[1])

            for i in [2, 3]:
                axes[i, j].set_axis_off()
                continue

        else:
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
                if i == 0:
                    title_iter = f"{title_iter}: SNR = {snr_array[idx]: .2f} dB"
                axes[i, j].set_title(title_iter)

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
    one hyper-parameter.

    kwargs: if_save, save_dir
    """
    def param2str(param_name: str):
        # customize names of hyper-paramters with LATEX here
        param2str_dict = {
            "step_lr": r"$\alpha$",
            "lr_scaled": r"$\lambda$"
        }
        assert param_name in param2str_dict

        return param2str_dict[param_name]

    def dict2title(args_dict: dict):
        # customize with LATEX here
        title = "defaults: " + \
                r"$\lambda = $" + f"{args_dict['lr_scaled']}, " + \
                r"$\alpha$ = " + f"{args_dict['step_lr']}"

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
        # one row for one hyper-parameter
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
            # print(f"{(i, j)}:\n{xx}\n{yy}")
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
