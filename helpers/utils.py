import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import SimpleITK as sitk
import InverseProblemWithDiffusionModel.helpers.pytorch_utils as ptu

from torchvision.utils import make_grid
from InverseProblemWithDiffusionModel.configs.general_configs import general_config


def expand_like(X_in, X_mimic):
    """
    e.g. X_in: (B,), X_mimic: (B, C, H, W); X_out: (B, 1, 1, 1)
    """
    X_expand_shape = [X_mimic.shape[0]]
    X_expand_shape += [1 for _ in range(len(X_mimic.shape) - 1)]

    return X_in.reshape(*X_expand_shape)


def get_data_inverse_scaler(is_centered):
  """Inverse data normalizer."""
  if is_centered:
    # Rescale [-1, 1] to [0, 1]
    return lambda x: (x + 1.) / 2.
  else:
    return lambda x: x


def vis_reverse_process(imgs: list, **kwargs):
    # imgs: list[(H, W, C)]
    figsize = kwargs.get("figsize", (18, 3))
    fig, axis = plt.subplots(figsize=figsize)
    imgs = torch.tensor(np.stack(imgs, axis=0))  # (B, H, W, C)
    imgs = imgs.permute(0, 3, 1, 2)  # (B, C, H, W)
    img_grid = make_grid(imgs, nrow=len(imgs)).permute(1, 2, 0)  # (C, H', W') -> (H', W', C)
    img_grid = ptu.to_numpy(img_grid)
    if img_grid.shape[-1] == 3:
        axis.imshow(img_grid)
    elif img_grid.shape[-1] == 1:
        axis.imshow[img_grid[..., 0]]
    else:
        raise Exception("Invalid number of channels")

    save_dir = kwargs.get("save_dir", None)
    if save_dir is not None:
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        save_filename = os.path.join(save_dir, "reverse_process.png")
        fig.savefig(save_filename)
    else:
        plt.show()
        plt.close()


def vis_volume(data):
    # data: (D, H, W) or (D, W, H)
    if isinstance(data, torch.Tensor):
        data = ptu.to_numpy(data)
    img_viewer = sitk.ImageViewer()
    img_sitk = sitk.GetImageFromArray(data)
    img_viewer.Execute(img_sitk)


def vis_images(*images, **kwargs):
    """
    kwargs: if_save, save_dir, filename, titles
    """
    num_imgs = len(images)
    fig, axes = plt.subplots(1, num_imgs, figsize=(general_config.figsize_unit * num_imgs, general_config.figsize_unit))
    if num_imgs == 1:
        axes = [axes]
    titles = kwargs.get("titles", None)
    if titles is not None:
        assert len(titles) == len(images)
    for i, (img_iter, axis) in enumerate(zip(images, axes)):
        channel = 0
        # channel = 0 if img_iter.shape[0] == 1 else 1
        if isinstance(img_iter, torch.Tensor):
            img_iter = ptu.to_numpy(img_iter)
        img_iter = img_iter[channel]
        handle = axis.imshow(img_iter, cmap="gray")
        plt.colorbar(handle)
        if titles is not None:
            axis.set_title(titles[i])

    fig.tight_layout()
    if_save = kwargs.get("if_save", False)
    if if_save:
        save_dir = kwargs.get("save_dir", "./outputs")
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        assert "filename" in kwargs
        filename = kwargs.get("filename")
        filename_full = os.path.join(save_dir, filename)
        fig.savefig(filename_full)
    else:
        plt.show()
    plt.close()


def collate_state_dict(state_dict: dict):
    out_dict = {}
    prefix = "model."
    for key, val in state_dict.items():
        idx = key.find(prefix)
        assert idx >= 0
        key_updated = key[len(prefix):]
        out_dict[key_updated] = state_dict[key]

    return out_dict
