import sys
import os

path = "/scratch/zhexwu"
# path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if path not in sys.path:
    sys.path.append(path)

import argparse
import numpy as np
import torch
import InverseProblemWithDiffusionModel.helpers.pytorch_utils as ptu

from torch.utils.tensorboard import SummaryWriter
from InverseProblemWithDiffusionModel.helpers.load_data import load_data, load_config
from InverseProblemWithDiffusionModel.helpers.load_model import reload_model
from InverseProblemWithDiffusionModel.ncsn.models import get_sigmas
from InverseProblemWithDiffusionModel.ncsn.linear_transforms.masking import SkipLines
from InverseProblemWithDiffusionModel.ncsn.models.MAP_optimizers import Inpainting
from datetime import datetime


if __name__ == '__main__':
    """
    python scripts/mnist_MAP.py
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_skip_lines", type=int, default=2)
    parser.add_argument("--ds_idx", type=int, default=0)
    parser.add_argument("--save_dir", default="../outputs")
    parser.add_argument("--lamda", type=float, default=1e-2)
    args_dict = vars(parser.parse_args())

    ds_name = "MNIST"
    mode = "real-valued"
    device = ptu.DEVICE

    config = load_config(ds_name, mode, device)
    ds = load_data(ds_name, "val")
    data = ds[args_dict["ds_idx"]]
    # (1, H, W), (1, H, W)
    img, label = ds[args_dict["ds_idx"]]
    img = img.unsqueeze(0).to(device)  # (1, C, H, W)
    label = torch.tensor(label).view((1,)).to(device)  # (1,)

    scorenet = reload_model("Diffusion", ds_name, mode)

    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
    log_dir = os.path.join(args_dict["save_dir"], "MNIST_MAP", timestamp)
    logger = SummaryWriter(log_dir=log_dir)

    x_mod_shape = (
        1,
        config.data.channels,
        config.data.image_size,
        config.data.image_size
    )
    x_init = torch.rand(*x_mod_shape).to(device)
    linear_tfm = SkipLines(args_dict["num_skip_lines"], x_mod_shape[1:])
    measurement = linear_tfm(img)  # (1, 1, H, W)
    MAP_optimizer = Inpainting(
        x_init=x_init,
        measurement=measurement,
        scorenet=scorenet,
        linear_tfm=linear_tfm,
        lamda=args_dict["lamda"],
        config=config,
        logger=logger,
        device=device
    )
    MAP_optimizer()
