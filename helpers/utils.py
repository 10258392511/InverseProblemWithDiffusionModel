import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import os
import SimpleITK as sitk
import yaml
import argparse
import pickle
import InverseProblemWithDiffusionModel.helpers.pytorch_utils as ptu

from torchvision.utils import make_grid
from InverseProblemWithDiffusionModel.configs.general_configs import general_config
from datetime import datetime
from typing import Union


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
        plt.colorbar(handle, ax=axis)
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


def load_yml_file(filename: str):
    assert ".yml" in filename
    with open(filename, "r") as rf:
        data = yaml.load(rf, yaml.Loader)

    data = dict2namespace(data)

    return data


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def get_optimizer(config, parameters):
    if config.optim.optimizer == 'Adam':
        return optim.Adam(parameters, lr=config.optim.lr, weight_decay=config.optim.weight_decay,
                          betas=(config.optim.beta1, 0.999), amsgrad=config.optim.amsgrad,
                          eps=config.optim.eps)
    elif config.optim.optimizer == 'RMSProp':
        return optim.RMSprop(parameters, lr=config.optim.lr, weight_decay=config.optim.weight_decay)
    elif config.optim.optimizer == 'SGD':
        return optim.SGD(parameters, lr=config.optim.lr, momentum=0.9)
    else:
        raise NotImplementedError('Optimizer {} not understood.'.format(config.optim.optimizer))


def logit_transform(image, lam=1e-6):
    image = lam + (1 - 2 * lam) * image
    return torch.log(image) - torch.log1p(-image)


def data_transform(config, X):
    if config.data.uniform_dequantization:
        X = X / 256. * 255. + torch.rand_like(X) / 256.
    if config.data.gaussian_dequantization:
        X = X + torch.randn_like(X) * 0.01

    if config.data.rescaled:
        X = 2 * X - 1.
    elif config.data.logit_transform:
        X = logit_transform(X)

    if hasattr(config, 'image_mean'):
        return X - config.image_mean.to(X.device)[None, ...]

    return X


def vis_tensor(X: torch.Tensor, **kwargs):
    """
    kwargs: figsize, if_colorbar
    """
    # X: (B, C, H, W)
    C = X.shape[1]
    assert C > 0
    if_colorbar = kwargs.get("if_colorbar", True)
    X = X.detach().cpu()
    img_grid = make_grid(X, nrow=X.shape[0])  # (C, H', W')
    figsize = kwargs.get("figsize", (general_config.figsize_unit * X.shape[0], general_config.figsize_unit))
    fig, axis = plt.subplots(figsize=figsize)
    if C == 3:
        axis.imshow(img_grid.permute(1, 2, 0).numpy())
    else:
        handle = axis.imshow(img_grid[0, ...].numpy(), cmap="gray")
        if if_colorbar:
            plt.colorbar(handle, ax=axis)


    # plt.show()
    # plt.close()

    return fig


def create_filename(args_dict: dict, suffix: str):
    str_out = ""
    for i, (key, val) in enumerate(args_dict.items()):
        if i == 0:
            str_out += f"{key}_{val}"
        else:
            str_out += f"_{key}_{val}"
    str_out = str_out.replace(".", "_")
    str_out += suffix

    return str_out


def load_pickle(filename: str):
    assert ".pkl" in filename
    with open(filename, "rb") as rf:
        data = pickle.load(rf)

    return data


def compute_angle(img: Union[torch.Tensor, np.ndarray], if_normalize=False):
    if isinstance(img, torch.Tensor):
        img = ptu.to_numpy(img)
    angle = np.angle(img)
    if if_normalize:
        angle -= angle.min()
        angle /= angle.max()

    return angle


def normalize(img: torch.Tensor, low_q: float = 0.02, high_q: float = 0.98, return_q=False) -> torch.Tensor:
    assert 0 <= low_q < high_q <= 1
    low_val = torch.quantile(img, low_q)
    high_val = torch.quantile(img, high_q)
    img_out = (img - low_val) / (high_val - low_val)
    img_out = torch.clip(img_out, 0., 1.)

    if return_q:
        return img_out, low_val, high_val

    return img_out


def denormalize(img: torch.Tensor, a_min: float, a_max: float) -> torch.Tensor:
    # assuming img is clean
    img_min, img_max = img.min(), img.max()
    img_out = (img - img_min) / (img_max - img_min) + a_min

    return img_out


def get_timestamp():
    time_stamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")

    return time_stamp


def undersample_seg_mask(label: torch.Tensor, fraction=1., seed=None):
    # label: (B, 1, H, W)
    assert 0. <= fraction <= 1.
    if seed is not None:
        torch.random.manual_seed(seed)
    non_zeros = torch.nonzero(label, as_tuple=True)
    num_samples = max(1, int(non_zeros[0].shape[0] * fraction))
    sample_indices = torch.randperm(non_zeros[0].shape[0])
    sample_indices = sample_indices[:num_samples]
    indices = [ind_iter[sample_indices] for ind_iter in non_zeros]
    label_out = torch.zeros_like(label)
    label_out[indices] = 1.

    return label_out


def reshape_temporal_dim(x: torch.Tensor, kx, ky, direction="forward", img_size=None):
    """
    "forward": (N, T, H, W) -> (N * H * W / (kx * ky), kx * ky, T)
    "backward": (N * H * W / (kx * ky), kx * ky, T) -> (N, T, H, W)
    """
    assert direction in ["forward", "backward"]
    if direction == "forward":
        N, T, H, W = x.shape
        assert H % kx == 0 and W % ky == 0
        x_out = x.permute(0, 2, 3, 1)  # (N, H, W, T)
        x_out = x_out.reshape(N, H // kx, kx, W // ky, ky, T)  # (N, H // kx, kx, W // ky, ky, T)
        x_out = x_out.permute(0, 1, 3, 2, 4, 5)  # (N, H // kx, W // ky, kx, ky, T)
        x_out = x_out.reshape(-1, kx * ky , T)  # (N', kx * ky, T)

        return x_out
    
    else:
        assert img_size is not None
        H, W = img_size
        assert H % kx == 0 and W % ky == 0
        N_out, C, T = x.shape
        assert C == kx * ky
        x_out = x.reshape(-1, H // kx, W // ky, kx, ky, T)  # (N, H // kx, W // ky, kx, ky, T)
        x_out = x_out.permute(0, 1, 3, 2, 4, 5)  # (N, H // kx, kx, W // ky, ky, T)
        x_out = x_out.reshape(-1, H, W, T)  # (N, H, W, T)
        x_out = x_out.permute(0, 3, 1, 2)  # (N, T, H, W)

        return x_out
