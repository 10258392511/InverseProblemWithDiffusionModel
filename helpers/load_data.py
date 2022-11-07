import numpy as np
import torch
import scipy.io as sio
import os
import glob
import random
import InverseProblemWithDiffusionModel.helpers.pytorch_utils as ptu

from monai.transforms import (
    Compose,
    Transform,
    MapTransform,
    ScaleIntensityd,
    CropForegroundd,
    Resized,
    RandRotated,
    RandAdjustContrastd
)
from monai.data import CacheDataset
from monai.data import Dataset as m_Dataset
from monai.utils import CommonKeys
from torch.utils.data import TensorDataset
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from monai.transforms import Resize as monai_Resize
from typing import Union
from InverseProblemWithDiffusionModel.helpers.utils import load_yml_file


parent_dir = os.path.dirname(os.path.dirname(__file__))
# change this
REGISTERED_DATA_ROOT_DIR = {
    "MNIST": os.path.join(parent_dir, "data"),
    "CINE64": os.path.join(parent_dir, "data/score_labs/data/cine_64"),
    "CINE127": os.path.join(parent_dir, "data/score_labs/data/cine_127"),
    "ACDC": "/scratch/zhexwu/data/ACDC_textures/data_slices",
    # "ACDC": "E:\Datasets\ACDC_textures\data_slices"
}

REGISTERED_DATA_CONFIG_FILENAME = {
    "MNIST": os.path.join(parent_dir, "ncsn/configs/mnist.yml"),
    "CINE64": os.path.join(parent_dir, "ncsn/configs/cine64.yml"),
    "CINE127": os.path.join(parent_dir, "ncsn/configs/cine127.yml"),
    "ACDC": os.path.join(parent_dir, "ncsn/configs/acdc.yml")
}


def load_data(ds_name, mode="train", **kwargs):
    assert ds_name in REGISTERED_DATA_ROOT_DIR.keys()
    assert mode in ["train", "val", "test"]

    ds_path = REGISTERED_DATA_ROOT_DIR[ds_name]
    ds_out = None

    if ds_name == "MNIST":
        ds_out = load_mnist(ds_path, mode, **kwargs)
    elif ds_name == "CINE64":
        ds_out = load_cine(ds_path, mode, **kwargs)
    elif ds_name == "CINE127":
        ds_out = load_cine(ds_path, mode, resize_shape=128, **kwargs)
    elif ds_name == "ACDC":
        ds_out = load_ACDC(ds_path, mode=mode, **kwargs)

    return ds_out


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


def load_cine(root_dir, mode="train", img_key="imgs", flatten=True,
              resize_shape: Union[int, None] = None):
    assert mode in ["train", "val", "test"]
    if mode == "val":
        mode = "test"
    filename = glob.glob(os.path.join(root_dir, f"*{mode}*.mat"))[0]
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
    if isinstance(ds, np.ndarray):
        ds = torch.tensor(ds)
    ds = TensorDataset(ds)

    return ds


def load_tissue_data(path_to_file):
    '''
    Read image intensity, multiclass-segmentations, and PD, T1, T2 (tissue properties)
    for multi-slice SAX images of the ACDC dataset
    dimensions are : [1,N_slices,Nx,Ny]

    all cases are have been previously resampled to 1x1 mm in-plane resolution and 10 mm slice thickness
    '''
    data = np.load(path_to_file)
    image = data['image']
    multi_class = data['multiClassMasks']
    PD = data['PD']
    T1 = data['T1']
    T2 = data['T2']

    return image, multi_class, PD, T1, T2


def vol2slice(root_dir, save_dir):
    """
    Save volume data as slices. All data: (1, H, W)
    """
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    all_filenames = glob.glob(os.path.join(root_dir, "*.npz"))
    for filename in all_filenames:
        image, multi_class, PD, T1, T2 = load_tissue_data(filename)
        for slice_idx in range(image.shape[1]):
            filename_base = os.path.basename(filename)
            suffix_idx = filename_base.find(".npz")
            save_filename = os.path.join(save_dir, f"{filename_base[:suffix_idx]}_{slice_idx}.npz")
            np.savez(save_filename,
                     image=image[:, slice_idx, ...],
                     multiClassMasks=multi_class[:, slice_idx, ...],
                     PD=PD[:, slice_idx, ...],
                     T1=T1[:, slice_idx, ...],
                     T2=T2[:, slice_idx, ...])


class LoadDataNumpyDict(Transform):
    def __init__(self, seg_labels: list):
        self.seg_labels = seg_labels

    def __call__(self, filename):
        seg_labels = self.seg_labels
        data_out = {
            CommonKeys.IMAGE: None,
            CommonKeys.LABEL: None
        }
        image, label, _, _, _ = load_tissue_data(filename)  # (1, H, W), (1, H, W)
        label_out = np.zeros_like(label, dtype=np.int64)
        for seg_label_iter in seg_labels:
            mask_iter = (label == seg_label_iter)
            label_out[mask_iter] = 1
        image = torch.tensor(image).float()
        label_out = torch.tensor(label_out).long()
        # data_out[CommonKeys.IMAGE] = image.permute(0, 2, 3, 1)  # (1, H, W, D)
        # data_out[CommonKeys.LABEL] = label_out.permute(0, 2, 3, 1)  # (1, H, W, D)
        data_out[CommonKeys.IMAGE] = image
        data_out[CommonKeys.LABEL] = label_out

        return data_out


class Flatten3D(MapTransform):
    def __call__(self, data):
        for key in self.keys:
            if key in data:
                # (C, H, W, D) -> [(C, H, W)]
                data[key] = [data[key][..., d] for d in range(data[key].shape[-1])]

        return data


def load_ACDC(root_dir, train_test_split=[0.8, 0.1], seg_labels=[3], mode="train", seed=0, num_workers=4, if_aug=True):
    # seg_label: 3: left MYO
    assert mode in ["train", "val", "test"]
    keys = [CommonKeys.IMAGE, CommonKeys.LABEL]

    all_filenames = glob.glob(os.path.join(root_dir, "*.npz"))
    random.seed(seed)
    random.shuffle(all_filenames)
    train_val_split_idx = int(len(all_filenames) * train_test_split[0])
    val_test_split_idx = int(len(all_filenames) * sum(train_test_split))
    if mode == "train":
        filenames = all_filenames[:train_val_split_idx]
    elif mode == "val":
        filenames = all_filenames[train_val_split_idx:val_test_split_idx]
    else:
        filenames = all_filenames[val_test_split_idx:]
    # print(filenames)

    transforms = [
        LoadDataNumpyDict(seg_labels=seg_labels),
        ScaleIntensityd(keys=[CommonKeys.IMAGE]),
        CropForegroundd(keys=keys, source_key=CommonKeys.IMAGE),
    ]

    if mode == "train" and if_aug:
        transforms += [
            RandRotated(keys=keys, range_x=np.deg2rad(15), mode=("bilinear", "nearest"), prob=.5),
            RandAdjustContrastd(keys=CommonKeys.IMAGE, prob=.5)
        ]

    transforms += [
        # Flatten3D(keys=keys),
        Resized(keys=keys, spatial_size=(256, 256), mode=("bilinear", "nearest"))
    ]

    transforms = Compose(transforms)
    if if_aug:
        ds_out = CacheDataset(filenames, transform=transforms, num_workers=num_workers)
    else:
        ds_out = m_Dataset(filenames, transform=transforms)

    return ds_out


def load_config(ds_name, mode="real-valued", device=None):
    assert mode in ["real-valued", "mag", "complex"]
    assert ds_name in REGISTERED_DATA_CONFIG_FILENAME.keys()
    if device is None:
        device = ptu.DEVICE
    config_path = REGISTERED_DATA_CONFIG_FILENAME[ds_name]
    config_namespace = load_yml_file(config_path)
    config_namespace.device = device
    if mode == "real-valued":
        pass

    elif mode == "mag":
        pass

    elif mode == "complex":
        config_namespace.data.channels = 2

    return config_namespace


def collate_batch(batch: torch.Tensor, mode="real-valued"):
    """
    To be used in LightningModule
    """
    assert mode in ["real-valued", "mag", "complex"]
    # batch: (B, 1, H, W)
    assert batch.shape[1] == 1
    if mode == "real-valued":
        pass

    elif mode == "mag":
        pass

    elif mode == "complex":
        batch = torch.cat([batch, torch.zeros_like(batch)], dim=1)  # (B, 2, H, W)

    return batch
