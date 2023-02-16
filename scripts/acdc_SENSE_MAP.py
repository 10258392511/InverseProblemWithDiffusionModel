import sys
import os

path = "/scratch/zhexwu"
# path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if path not in sys.path:
    sys.path.append(path)

import argparse
import numpy as np
import torch
import pickle
import InverseProblemWithDiffusionModel.helpers.pytorch_utils as ptu

from torch.utils.tensorboard import SummaryWriter
from InverseProblemWithDiffusionModel.helpers.load_data import load_data, load_config, add_phase
from InverseProblemWithDiffusionModel.helpers.load_model import reload_model
from InverseProblemWithDiffusionModel.ncsn.models import get_sigmas
from InverseProblemWithDiffusionModel.ncsn.linear_transforms.undersampling_fourier import SENSE
from InverseProblemWithDiffusionModel.ncsn.models.MAP_optimizers import SENSEMAP
from InverseProblemWithDiffusionModel.helpers.utils import vis_images, create_filename
from datetime import datetime
from monai.utils import CommonKeys

if __name__ == '__main__':
    """
    python scripts/acdc_SENSE_MAP.py
    """
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    parser = argparse.ArgumentParser()
    parser.add_argument("--R", type=int, default=6)
    parser.add_argument("--center_lines_frac", type=float, default=1 / 4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sens_type", default="exp")
    parser.add_argument("--num_sens", type=int, default=4)
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
    log_dir = os.path.join(args_dict["save_dir"], "logs")
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
    linear_tfm = SENSE(
        args_dict["sens_type"],
        args_dict["num_sens"],
        args_dict["R"], 
        args_dict["center_lines_frac"], 
        x_mod_shape[1:],
        args_dict["seed"]
    )
    img_complex = add_phase(img, init_shape=(5, 5), seed=args_dict["seed"])  # (1, 1, H, W)
    measurement = linear_tfm(img_complex.to(torch.complex64))  # (num_sens, 1, 1, H, W)
    direct_recons = linear_tfm.conj_op(measurement)  # (1, 1, H, W)
    x_init = direct_recons

    # save image and measurement
    vis_images(torch.abs(img_complex[0]), torch.angle(img_complex[0]), if_save=True, save_dir=args_dict["save_dir"], filename="original.png")
    eps = 1e-6
    vis_images(torch.log(torch.abs(measurement[0, 0]) + eps), torch.angle(measurement[0, 0]), if_save=True, save_dir=args_dict["save_dir"], filename="measurement.png")
    vis_images(torch.abs(direct_recons[0]), torch.angle(direct_recons[0]), if_save=True, save_dir=args_dict["save_dir"], filename="direct_recons.png")
    
    MAP_optimizer = SENSEMAP(
        x_init=x_init,
        measurement=measurement,
        scorenet=scorenet,
        linear_tfm=linear_tfm,
        lamda=args_dict["lamda"],
        config=config,
        logger=logger,
        device=device
    )

    filename_dict = {
        "ds_name": ds_name,
        "R": args_dict["R"],
        "center_lines_frac": args_dict["center_lines_frac"]
    }
    log_filename = create_filename(filename_dict, suffix=".txt")
    log_file = open(os.path.join(args_dict["save_dir"], log_filename), "w")
    sys.stdout = log_file
    sys.stderr = log_file

    img_out = MAP_optimizer()  # (1, 1, H, W)
    vis_images(torch.abs(img_out[0]), torch.angle(img_out[0]), if_save=True, save_dir=args_dict["save_dir"], filename="reconstruction.png")

    torch.save(img_out.detach().cpu(), os.path.join(args_dict["save_dir"], "reconstructions.pt"))
    print("-" * 100)
    print(args_dict)

    sys.stdout = original_stdout
    sys.stderr = original_stderr

    with open(os.path.join(args_dict["save_dir"], "args_dict.pkl"), "wb") as wf:
        pickle.dump(args_dict, wf)
