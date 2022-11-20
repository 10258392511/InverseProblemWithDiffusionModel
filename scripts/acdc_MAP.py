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
from InverseProblemWithDiffusionModel.ncsn.linear_transforms.undersampling_fourier import UndersamplingFourier
from InverseProblemWithDiffusionModel.ncsn.models.MAP_optimizers import UndersamplingFourier as UFMAP
from InverseProblemWithDiffusionModel.helpers.utils import vis_images, create_filename
from datetime import datetime
from monai.utils import CommonKeys


if __name__ == '__main__':
    """
    python scripts/acdc_MAP.py
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_skip_lines", type=int, default=2)
    parser.add_argument("--ds_idx", type=int, default=0)
    parser.add_argument("--save_dir", default="../outputs")
    parser.add_argument("--lamda", type=float, default=1e-2)
    args_dict = vars(parser.parse_args())

    ds_name = "ACDC"
    mode = "real-valued"
    device = ptu.DEVICE

    config = load_config(ds_name, mode, device)
    ds = load_data(ds_name, "val")
    data = ds[args_dict["ds_idx"]]
    # (1, H, W), (1, H, W)
    img, label = data[CommonKeys.IMAGE], data[CommonKeys.LABEL]
    img = img.unsqueeze(0).to(device)  # (1, 1, H, W)
    label = label.unsqueeze(0).to(device)

    scorenet = reload_model("Diffusion", ds_name, mode)

    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
    log_dir = os.path.join(args_dict["save_dir"], "ACDC_MAP", timestamp)
    logger = SummaryWriter(log_dir=log_dir)

    desc_dict = args_dict.copy()
    desc_dict.update(vars(config.MAP))
    with open(os.path.join(log_dir, "desc.txt"), "w") as wf:
        for key, val in desc_dict.items():
            wf.write(f"{key}: {val}\n")

    x_mod_shape = (
        1,
        config.data.channels,
        config.data.image_size,
        config.data.image_size
    )
    x_init = torch.rand(*x_mod_shape).to(device)
    linear_tfm = UndersamplingFourier(args_dict["num_skip_lines"], x_mod_shape[1:])
    measurement = linear_tfm(img.to(torch.complex64))  # (1, 1, H, W)

    # save image and measurement
    vis_images(img[0], if_save=True, save_dir=log_dir, filename="original.png")
    eps = 1e-6
    vis_images(torch.log(torch.abs(measurement[0]) + eps), if_save=True, save_dir=log_dir, filename="measurement.png")
    vis_images(torch.abs(linear_tfm.conj_op(measurement))[0], if_save=True, save_dir=log_dir,
               filename="zero_filled_recons.png")

    MAP_optimizer = UFMAP(
        x_init=x_init,
        measurement=measurement,
        scorenet=scorenet,
        linear_tfm=linear_tfm,
        lamda=args_dict["lamda"],
        config=config,
        logger=logger,
        device=device
    )
    img_out = MAP_optimizer()
    vis_images(img_out[0], if_save=True, save_dir=log_dir, filename="reconstruction.png")
