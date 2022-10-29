import torch
import scipy.io as sio
import os
import glob

from torch.utils.data import TensorDataset
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from monai.transforms import Resize as monai_Resize
from typing import Union


def load_mnist(root_dir, mode="train"):
    assert mode in ["train", "val", "test"]

    transforms = [
        ToTensor(),
        Resize(32),
        # Normalize(mean=[0.], std=[1.]),
    ]
    transforms = Compose(transforms)
    if_train = True if mode == "train" else False
    ds = MNIST(root_dir, train=if_train, transform=transforms, download=True)

    return ds


def load_cifar10(root_dir, mode="train"):
    assert mode in ["train", "val", "test"]
    if not os.path.isdir(root_dir):
        os.makedirs(root_dir)
    transforms = [
        ToTensor(),
        # Normalize(mean=[0.], std=[1.]),
    ]
    transforms = Compose(transforms)
    if_train = True if mode == "train" else False
    ds = CIFAR10(root_dir, train=if_train, transform=transforms, download=True)

    return ds


def load_cine(root_dir, mode="train", img_key="imgs", flatten=False,
              resize_shape: Union[int, None] = None):
    assert mode in ["train", "test"]
    filename = glob.glob(f"{root_dir}/*{mode}*.mat")[0]
    ds = sio.loadmat(filename)[img_key]  # (H, W, T, N)
    ds = ds.transpose(3, 2, 0, 1)  # (N, T, H, W)
    ds = (ds - ds.min()) / (ds.max() - ds.min())
    if flatten:
        N, T, H, W = ds.shape
        ds = ds.reshape(-1, H, W)  # (N', H, W)
        if resize_shape is not None and not (H == resize_shape and W == resize_shape):
            resizer = Compose([
                monai_Resize(spatial_size=(resize_shape, resize_shape)),
            ])
            ds = resizer(ds)  # (H, W, N') -> (N', H, W) -> (N', H0, W0)
        # (N', H0, W0) -> (N', 1, H0, W0)
        ds = ds[:, None, ...]
    ds = TensorDataset(torch.tensor(ds))

    return ds
