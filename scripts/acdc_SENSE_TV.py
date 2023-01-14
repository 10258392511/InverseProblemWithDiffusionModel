import sys
import os

path = "/scratch/zhexwu"
# path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if path not in sys.path:
    sys.path.append(path)

import argparse
import numpy as np
import pickle
import torch
import torch.nn as nn
import InverseProblemWithDiffusionModel.helpers.pytorch_utils as ptu

from kornia.losses import TotalVariation
from monai.data import DataLoader
from InverseProblemWithDiffusionModel.helpers.load_data import load_data, add_phase
from InverseProblemWithDiffusionModel.ncsn.linear_transforms.undersampling_fourier import SENSE
from InverseProblemWithDiffusionModel.helpers.utils import vis_images, get_timestamp, create_filename
from InverseProblemWithDiffusionModel.helpers.pl_helpers import TrainMAPModel
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from monai.utils import CommonKeys


if __name__ == '__main__':
    """
    python scripts/acdc_SENSE_TV.py
    """
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    parser = argparse.ArgumentParser()
    parser.add_argument("--R", type=int, default=5)
    parser.add_argument("--center_lines_frac", type=float, default=1 / 20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sens_type", default="exp")
    parser.add_argument("--num_sens", type=int, default=4)
    parser.add_argument("--ds_idx", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--reg_weight", type=float, default=0.01)
    parser.add_argument("--save_dir", default="../outputs")
    parser.add_argument("--log_dir", default="SENSE")
    args_dict = vars(parser.parse_args())

    ds_name = "ACDC"
    mode = "real-valued"
    device = ptu.DEVICE

    ds = load_data(ds_name, "val", if_aug=False)
    data = ds[args_dict["ds_idx"]]
    # (1, H, W), (1, H, W)
    img, label = data[CommonKeys.IMAGE], data[CommonKeys.LABEL]
    img = img.unsqueeze(0).to(device)  # (1, 1, H, W)
    label = label.unsqueeze(0).to(device)

    linear_tfm = SENSE(
        args_dict["sens_type"],
        args_dict["num_sens"],
        args_dict["R"], 
        args_dict["center_lines_frac"], 
        img.shape[1:],
        args_dict["seed"]
    )

    for i in range(linear_tfm.sens_maps.shape[0]):
        vis_images(linear_tfm.sens_maps[i:i + 1], if_save=True, save_dir=args_dict["save_dir"], 
                   filename=f"sens_map_{i}.png")

    img_complex = add_phase(img, init_shape=(5, 5), seed=args_dict["seed"])  # (1, 1, H, W)
    measurement = linear_tfm(img_complex.to(torch.complex64))  # (num_sens, 1, 1, H, W)
    
    reg = TotalVariation()
    lit_mode_params = {
        "num_workers": args_dict["num_workers"],
        "lr": args_dict["lr"]
    }
    lit_model = TrainMAPModel(measurement, linear_tfm, reg, args_dict["reg_weight"], lit_mode_params)

    eps = 1e-6
    vis_images(torch.abs(img_complex[0]), torch.angle(img_complex[0]), if_save=True, save_dir=args_dict["save_dir"], filename="original_acdc.png")
    vis_images(label[0], if_save=True, save_dir=args_dict["save_dir"], filename="original_seg.png")
    vis_images(torch.log(torch.abs(measurement[0, 0]) + eps), torch.angle(measurement[0, 0]), if_save=True, save_dir=args_dict["save_dir"],
               filename=f"acdc_measurement_R_{args_dict['R']}_frac_{args_dict['center_lines_frac']}.png")
    direct_recons = linear_tfm.conj_op(measurement)[0]
    vis_images(torch.abs(direct_recons), torch.angle(direct_recons), if_save=True, save_dir=args_dict["save_dir"],
               filename=f"acdc_zero_padded_recons_R_{args_dict['R']}_frac_{args_dict['center_lines_frac']}.png")

    filename_dict = {
        "ds_name": ds_name,
        "R": args_dict["R"],
        "center_lines_frac": args_dict["center_lines_frac"],
    }
    log_filename = create_filename(filename_dict, suffix=".txt")
    log_file = open(os.path.join(args_dict["save_dir"], log_filename), "w")
    sys.stdout = log_file
    sys.stderr = log_file
    original_error = torch.sum(torch.abs(linear_tfm(direct_recons.unsqueeze(0)).detach().cpu() - measurement.detach().cpu()) ** 2, dim=(1, 2, 3)).mean().item()
    print(f"original error: {original_error}")

    time_stamp = get_timestamp()
    logger = TensorBoardLogger(save_dir=args_dict["save_dir"], name=args_dict["log_dir"], version=time_stamp)
    callbacks = [ModelCheckpoint(os.path.join("./MAP_logs", args_dict["log_dir"], time_stamp, "checkpoints"))]
    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        callbacks=callbacks,
        logger=logger,
        max_epochs=args_dict["num_epochs"],
        log_every_n_steps=1
    )
    trainer.fit(lit_model)
    img_out = lit_model.model.get_reconstruction()

    filename = create_filename(filename_dict, suffix=".png")
    vis_images(torch.abs(img_out[0]), torch.angle(img_out[0]), if_save=True, save_dir=args_dict["save_dir"], filename=filename)

    # img_out, measurement: "(1, C, H, W)"
    l2_error = torch.sum(torch.abs(linear_tfm(img_out).detach().cpu() - measurement.detach().cpu()) ** 2,
                         dim=(1, 2, 3)).mean().item()
    print("-" * 100)
    print(args_dict)
    print(f"original error: {original_error}")
    print(f"reconstruction error: {l2_error}")

    save_dir = args_dict["save_dir"]
    torch.save(img_complex.detach().cpu(), os.path.join(save_dir, "original.pt"))
    torch.save(measurement.detach().cpu(), os.path.join(save_dir, "measurement.pt"))
    torch.save(img_out.detach().cpu(), os.path.join(save_dir, "reconstructions.pt"))

    log_file.close()

    sys.stdout = original_stdout
    sys.stderr = original_stderr

    with open(os.path.join(save_dir, "args_dict.pkl"), "wb") as wf:
        pickle.dump(args_dict, wf)
