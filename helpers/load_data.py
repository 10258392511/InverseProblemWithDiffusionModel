import numpy as np
import torch
import scipy.io as sio
import einops
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
    Resize,
    Resized,
    RandRotated,
    RandAdjustContrastd,
    RandGaussianNoised
)
from monai.data import CacheDataset
from monai.data import Dataset as m_Dataset
from monai.utils import CommonKeys
from torch.utils.data import TensorDataset, Dataset
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from monai.transforms import Resize as monai_Resize
from typing import Union
from InverseProblemWithDiffusionModel.helpers.utils import load_yml_file, reshape_temporal_dim, vis_multi_channel_signal


parent_dir = os.path.dirname(os.path.dirname(__file__))
# change this
REGISTERED_DATA_ROOT_DIR = {
    "MNIST": os.path.join(parent_dir, "data"),
    "CINE64": os.path.join(parent_dir, "data/score_labs/data/cine_64"),
    "CINE127": os.path.join(parent_dir, "data/score_labs/data/cine_127"),
    "ACDC": "/scratch/zhexwu/data/ACDC_textures/data_slices",
    # "ACDC": "E:\Datasets\ACDC_textures\data_slices",
    "SanityCheck1D": None
}

REGISTERED_DATA_CONFIG_FILENAME = {
    "MNIST": os.path.join(parent_dir, "ncsn/configs/mnist.yml"),
    "CINE64": os.path.join(parent_dir, "ncsn/configs/cine64.yml"),
    "CINE64_1D": os.path.join(parent_dir, "ncsn/configs/cine64_1d.yml"),
    "CINE127": os.path.join(parent_dir, "ncsn/configs/cine127.yml"),
    "CINE127_1D": os.path.join(parent_dir, "ncsn/configs/cine127_1d.yml"),
    "ACDC": os.path.join(parent_dir, "ncsn/configs/acdc.yml"),
    "SanityCheck1D": os.path.join(parent_dir, "ncsn/configs/sanity_check_1D.yml")
}


def load_data(ds_name, mode="train", **kwargs):
    """
    kwargs: CINE64/127: flatten_type
    """
    assert ds_name in REGISTERED_DATA_ROOT_DIR.keys()
    assert mode in ["train", "val", "test"]

    ds_path = REGISTERED_DATA_ROOT_DIR[ds_name]
    ds_out = None

    if ds_name == "MNIST":
        ds_out = load_mnist(ds_path, mode, **kwargs)
    elif ds_name == "CINE64":
        flatten_type = kwargs.get("flatten_type", "spatial")
        if flatten_type == "spatial":
            ds_out = load_cine(ds_path, mode, **kwargs)
        else:
            ds_out = load_cine(ds_path, mode, resize_shape_T=32, **kwargs)
    elif ds_name == "CINE127":
        flatten_type = kwargs.get("flatten_type", "spatial")
        if flatten_type == "spatial":
            ds_out = load_cine(ds_path, mode, resize_shape=128, **kwargs)
        else:
            ds_out = load_cine(ds_path, mode, resize_shape=128, resize_shape_T=32, **kwargs)

    elif ds_name == "ACDC":
        ds_out = load_ACDC(ds_path, mode=mode, **kwargs)
    
    elif ds_name == "SanityCheck1D":
        if mode == "train":
            seed = 0
            num_samples = 1000
        else:
            seed = 10
            num_samples = 300
        ds_out = load_sanity_check_1D(seed=seed, num_samples=num_samples, **kwargs)

    return ds_out


def load_mnist(root_dir, mode="train", **kwargs):
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


def load_cine(root_dir, mode="train", img_key="imgs", flatten=True, flatten_type="spatial",
              resize_shape: Union[int, None] = None, resize_shape_T=None, win_size=2, **kwargs):
    assert mode in ["train", "val", "test"]
    assert flatten_type in ["spatial", "temporal"]
    if mode == "val":
        mode = "test"
    filename = glob.glob(os.path.join(root_dir, f"*{mode}*.mat"))[0]
    ds = sio.loadmat(filename)[img_key]  # (H, W, T, N)
    ds = ds.transpose(3, 2, 0, 1)  # (N, T, H, W)
    ds = (ds - ds.min()) / (ds.max() - ds.min())
    if flatten:
        if flatten_type == "spatial":
            N, T, H, W = ds.shape
            ds = ds.reshape(-1, H, W)  # (N', H, W)
            if resize_shape is not None and not (H == resize_shape and W == resize_shape):
                resizer = Compose([
                    monai_Resize(spatial_size=(resize_shape, resize_shape)),
                ])
                ds = resizer(ds)  # (H, W, N') -> (N', H, W) -> (N', H0, W0)
            # (N', H0, W0) -> (N', 1, H0, W0)
            ds = ds[:, None, ...]

        else:
            N, T, H, W = ds.shape
            resize_shape_T = T if resize_shape_T is None else resize_shape_T
            resize_shape_H = H if resize_shape is None else resize_shape
            resize_shape_W = W if resize_shape is None else resize_shape            
            resizer = Compose([
              monai_Resize(spatial_size=(resize_shape_T, resize_shape_H, resize_shape_W))  
            ])
            ds = resizer(ds)  # (N, T', H', W')
            if not isinstance(ds, torch.Tensor):
                ds = torch.tensor(ds)
            ds = reshape_temporal_dim(ds, win_size, win_size)  # (N', win_size^2, T')
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
            RandAdjustContrastd(keys=CommonKeys.IMAGE, prob=.5),
            RandGaussianNoised(keys=CommonKeys.IMAGE, prob=0.1, mean=0.0, std=0.5)
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


def load_sanity_check_1D(num_samples: int, num_channels: int, num_features: int, seed=0):
    # s(t) = a * t / T + b * sin(w * t) + eps(t) where a ~ Unif{-1, 1}, w = 1, eps(t) ~ GP(0, sigma * delta(t, t'))
    torch.manual_seed(seed)
    b, w = 0.2, 1.
    sigma = 0.01
    a = np.random.choice([-1., 1.], (num_samples, num_channels, 1))  # (N, C, 1)
    a = torch.tensor(a)
    t_grid = torch.arange(num_features, dtype=torch.float32)  # (T,)
    x = a * t_grid / num_features + b * torch.sin(w * t_grid)  # (N, C, T) + (T,) -> (N, C, T)
    x = x + torch.randn_like(x) * sigma
    ds = TensorDataset(x.float())

    return ds


def load_config(ds_name, mode="real-valued", device=None, **kwargs):
    """
    kwargs: flatten_type
    """
    assert mode in ["real-valued", "mag", "complex", "real-imag", "real-imag-random"]
    assert ds_name in REGISTERED_DATA_CONFIG_FILENAME.keys()
    if device is None:
        device = ptu.DEVICE

    # flatten_type = kwargs.get("flatten_type", "spatial")
    # if "CINE" in ds_name and flatten_type == "temporal":
    # # if "CINE" in ds_name:
    #     ds_name = f"{ds_name}_1D"

    config_path = REGISTERED_DATA_CONFIG_FILENAME[ds_name]
    config_namespace = load_yml_file(config_path)
    config_namespace.device = device
    if mode == "complex":
        config_namespace.data.channels = 2

    return config_namespace

# COUNTER = 0
def collate_batch(batch: torch.Tensor, mode="real-valued"):
    """
    To be used in LightningModule
    """
    # global COUNTER
    assert mode in ["real-valued", "mag", "complex", "real-imag", "real-imag-random"]
    # batch: (B, 1, H, W)
    batch_dim = batch.dim()
    if batch_dim == 3:
        # (B, C, T)
        batch = batch.unsqueeze(1)  # (B, 1, C, T)
    assert batch.shape[1] == 1
    if mode == "real-valued":
        pass

    elif mode == "mag":
        pass

    elif mode == "complex":
        assert batch_dim == 4
        batch = torch.cat([batch, torch.zeros_like(batch)], dim=1)  # (B, 2, H, W)
    
    elif mode == "real-imag":
        # u = x + 1j * y: u: [-1, 1] -> x, y
        phi = (torch.rand(batch.shape[0]) * 2 - 1) * torch.pi  # (B,)
        phi = phi.to(batch.device).reshape(batch.shape[0], 1, 1, 1)  # (B, 1, 1, 1)
        batch = batch * torch.exp(1j * phi)
        batch_real, batch_imag = torch.real(batch), torch.imag(batch)
        print(f"{batch_real.shape}, {batch_real.dtype}")
        # vis_multi_channel_signal(batch_real[-1, 0], if_save=True, save_dir="/scratch/zhexwu/outputs/debug_filter_batch/after_adding_phase/", filename=f"train_sample_{COUNTER}.png")
        # COUNTER += 1
        batch = [batch_real, batch_imag]
    
    elif mode == "real-imag-random":
        batch = add_phase(batch)
        batch = [torch.real(batch), torch.imag(batch)]
    
    if batch_dim == 3:
        # (B, 1, C, T) -> (B, C, T)
        if isinstance(batch, list):
            for i in range(len(batch)):
                batch[i] = batch[i].squeeze()
        else:
            batch = batch.squeeze()
            
    return batch


def add_phase(imgs: torch.Tensor, init_shape: Union[tuple, int] = (5, 5), seed=None, mode="spatial"):
    # imgs: (B, C, H, W) or (T, C, H, W)
    assert mode in ["spatial", "2D+time"]
    if seed is not None:
        torch.manual_seed(seed)
    imgs_out = imgs
    if mode == "spatial":
        # add smooth phase for each spatial slice
        B, C, H, W = imgs.shape
        imgs_out = torch.empty_like(imgs, dtype=torch.complex64)
        for i in range(B):
            img_iter = imgs[i, ...]  # (C, H, W)
            phase_init_patch = torch.randn(C, *init_shape, device=img_iter.device)
            resizer = monai_Resize((H, W), mode="bicubic", align_corners=True)
            phase = resizer(phase_init_patch)  # (C, H, W)
            imgs_out[i, ...] = img_iter * torch.exp(1j * phase)
    elif mode == "2D+time":
        # use 3D phase map for each channel for (T, C, H, W)
        assert len(init_shape) == 3
        T, C, H, W = imgs.shape
        phase = torch.randn(C, *init_shape, device=imgs.device)  # e.g. (init_x, init_y, init_z)
        resizer = monai_Resize((T, H, W), mode="trilinear", align_corners=True)
        phase = resizer(phase)  # (C, T, H, W)
        imgs_out = imgs * torch.exp(1j * einops.rearrange(phase, "C T H W -> T C H W"))

    return imgs_out


def compute_max_euclidean_dist(ds: Dataset, num_pairs=10 ** 3):
    max_dist = 0
    for _ in range(num_pairs):
        i, j = torch.randint(0, len(ds), (2,))
        if isinstance(ds[i], tuple):
            dist = torch.norm(ds[i][0] - ds[j][0])
        elif isinstance(ds[i], dict):
            dist = torch.norm(ds[i][CommonKeys.IMAGE] - ds[j][CommonKeys.IMAGE])
        else:
            dist = torch.norm(ds[i] - ds[j])
        
        if dist > max_dist:
            max_dist = dist
    
    return max_dist


def filter_batch(batch: torch.Tensor, config):
    # print("from filter_batch(.)")
    # print(f"before: {batch.shape}")
    # global COUNTER
    if batch.dim() == 3:  # 1D signal
        th = config.data.th
        prob = 1 / config.data.leq
        # batch: (B, C, L)
        B, C, L = batch.shape
        batch_shift = torch.roll(batch, -1, dims=-1)
        norm = torch.norm(batch_shift - batch, p=1, dim=(1, 2), keepdim=False) / (C * L)  # (B,)
        norm_mask = (norm > th)  # (B,)
        prob_mask = (torch.rand(B).to(batch.device) <= 0.)  # (B,)
        mask = torch.logical_or(norm_mask, prob_mask)
        mask[0:2] = True  # ensures at least 2 sample is kept
        batch = batch[mask]
        # vis_multi_channel_signal(batch[-1], if_save=True, save_dir="/scratch/zhexwu/outputs/debug_filter_batch/", filename=f"train_sample_{COUNTER}.png")
        # COUNTER += 1
        # print(f"norm_mask: {norm_mask}")
        # print(f"prob_mask: {prob_mask}")
        # print(f"mask: {mask}")
        print(f"after: {batch.shape}")

    return batch
