import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import InverseProblemWithDiffusionModel.helpers.pytorch_utils as ptu

from torchvision.utils import make_grid


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
