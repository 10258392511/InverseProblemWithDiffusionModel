import sys
import os

path = "/scratch/zhexwu"
if path not in sys.path:
    sys.path.append(path)

import argparse
import numpy as np
import torch
import einops
import pickle
import time
import InverseProblemWithDiffusionModel.helpers.pytorch_utils as ptu

from monai.transforms import Resize
from InverseProblemWithDiffusionModel.helpers.load_data import load_data, load_config, add_phase
from InverseProblemWithDiffusionModel.helpers.load_model import reload_model
from InverseProblemWithDiffusionModel.ncsn.models import get_sigmas
from InverseProblemWithDiffusionModel.ncsn.linear_transforms.undersampling_fourier import SENSE
from InverseProblemWithDiffusionModel.ncsn.models.proximal_op import get_proximal
from InverseProblemWithDiffusionModel.ncsn.models.ALD_optimizers import ALD2DTime
from InverseProblemWithDiffusionModel.helpers.utils import (
    vis_images, 
    vis_signals, 
    create_filename,
    save_vol_as_gif, 
    normalize_phase
)


if __name__ == '__main__':
    """
    python scripts/cine_SENSE_real_img_2d_time.py
    """
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    parser = argparse.ArgumentParser()
    parser.add_argument("--ds_name", choices=["CINE64", "CINE127"])
    parser.add_argument("--R", type=int, default=6)
    parser.add_argument("--center_lines_frac", type=float, default=1 / 4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--step_lr", type=float, default=0.0001)  # overwriting config.sampling.step_lr
    parser.add_argument("--num_steps_each", type=int, default=3)
    parser.add_argument("--lr_scaled", type=float, default=1.)
    parser.add_argument("--proximal_type", default="L2Penalty")
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--sens_type", default="exp")
    parser.add_argument("--temporal_type", default="Diffusion3D")  # only for loading model
    parser.add_argument("--mode_T", choices=["tv", "diffusion1d", "none", "diffusion1d-only", "tv-only"], default="diffusion1d")
    parser.add_argument("--lamda_T", type=float, default=10.)
    parser.add_argument("--if_random_shift", action="store_true")
    parser.add_argument("--num_sens", type=int, default=4)
    parser.add_argument("--ds_idx", type=int, default=0)
    parser.add_argument("--save_dir", default="../outputs")
    args_dict = vars(parser.parse_args())

    ds_name = args_dict["ds_name"]
    mode = "real-valued"
    device = ptu.DEVICE

    config_spatial = load_config(ds_name, mode, device)
    print(f"config_spatial: {config_spatial.model.num_classes}")
    config_temporal = load_config(f"{ds_name}_1D", mode, device, flatten_type="temporal")
    print(f"config_temporal: {config_temporal.model.num_classes}")
    ds = load_data(ds_name, "val", if_aug=False, flatten=False)
    data = ds[args_dict["ds_idx"]]
    # ((T0, H0, W0),)
    img = data[0]
    img = img.unsqueeze(0)  # (1, T0, H0, W0)
    resizer = Resize(spatial_size=(config_temporal.data.image_size, config_spatial.data.image_size, config_spatial.data.image_size))
    img = resizer(img)  # (1, T, H, W)
    if not isinstance(img, torch.Tensor):
        img = torch.tensor(img)
    img = img.permute(1, 0, 2, 3).to(device)  # (T, 1, H, W)

    scorenet = reload_model("Diffusion", ds_name, mode)
    scorenet_T = reload_model(args_dict["temporal_type"], f"{ds_name}_1D", mode)
    ALD_sampler_params = {
        "n_steps_each": config_spatial.sampling.n_steps_each,
        "step_lr": config_spatial.sampling.step_lr
    }
    ALD_sampler_params["step_lr"] = args_dict["step_lr"]
    ALD_sampler_params["n_steps_each"] = args_dict["num_steps_each"]
    sigmas = get_sigmas(config_spatial, mode="recons")
    sigmas_T = get_sigmas(config_temporal, mode="recons")
    
    # (B, T, C, H, W)
    x_mod_shape = (
        args_dict["num_samples"],
        config_temporal.data.image_size,
        config_spatial.data.channels,
        config_spatial.data.image_size,
        config_spatial.data.image_size
    )
    linear_tfm = SENSE(
        args_dict["sens_type"],
        args_dict["num_sens"],
        args_dict["R"], 
        args_dict["center_lines_frac"], 
        x_mod_shape[2:],
        args_dict["seed"]
    )

    for i in range(linear_tfm.sens_maps.shape[0]):
        vis_images(linear_tfm.sens_maps[i:i + 1], if_save=True, save_dir=args_dict["save_dir"], 
                   filename=f"sens_map_{i}.png")

    proximal_constr = get_proximal(args_dict["proximal_type"])
    proximal = proximal_constr(linear_tfm)

    img_complex = add_phase(img, init_shape=(5, 5, 5), seed=args_dict["seed"], mode="2D+time")  # (T, 1, H, W)
    T, C, H, W = img_complex.shape
    measurement = linear_tfm(img_complex.to(torch.complex64))  # (num_sens, T, 1, H, W)
    measurement = measurement.unsqueeze(1)  # (num_sens, 1, T, 1, H, W)
    measurement = measurement.repeat(1, args_dict["num_samples"], 1, 1, 1, 1)  # (num_sens, B, T, 1, H, W)
    save_vol_as_gif(torch.abs(img_complex), save_dir=args_dict["save_dir"], filename=f"orig_mag.gif")
    save_vol_as_gif(normalize_phase(torch.angle(img_complex)), save_dir=args_dict["save_dir"], filename=f"orig_phase.gif")

    ALD_sampler = ALD2DTime(
        proximal,
        scorenet_T,
        sigmas_T,
        x_mod_shape,
        scorenet,
        sigmas,
        ALD_sampler_params,
        config_spatial,
        measurement,
        linear_tfm,
        device=device
    )

    eps = 1e-6
    # visualization: spatial: img_complex[0]; temporal: upper-left and center
    # img_complex: (T, 1, H, W), measurement: (num_sens, B, T, 1, H, W)
    vis_images(torch.log(torch.abs(measurement[0, 0, 0]) + eps), torch.angle(measurement[0, 0, 0]), if_save=True, save_dir=args_dict["save_dir"],
               filename=f"measurement.png")
    direct_recons = linear_tfm.conj_op(measurement.reshape(measurement.shape[0], -1, *measurement.shape[3:]))  # (num_sens, BT, 1, H, W) -> (BT, 1, H, W)
    direct_recons = einops.rearrange(direct_recons, "(B T) C H W -> B T C H W", T=measurement.shape[2])  # (BT, 1, H, W) -> (B, T, 1, H, W)
    # save the first batch
    x_init = direct_recons  # (B, T, 1, H, W)
    save_vol_as_gif(torch.abs(x_init[0]), save_dir=args_dict["save_dir"], filename=f"zf_mag.gif")
    save_vol_as_gif(normalize_phase(torch.angle(x_init[0])), save_dir=args_dict["save_dir"], filename=f"zf_phase.gif")
    ALD_sampler._screenshot(img_complex.unsqueeze(0), {"c": 0, "init_vis_dir": args_dict["save_dir"]})

    filename_dict = {
        "ds_name": ds_name,
        "R": args_dict["R"],
        "center_lines_frac": args_dict["center_lines_frac"],
    }
    log_filename = create_filename(filename_dict, suffix=".txt")
    log_file = open(os.path.join(args_dict["save_dir"], log_filename), "w")
    sys.stdout = log_file
    sys.stderr = log_file

    ALD_call_params = dict(save_dir=args_dict["save_dir"], lr_scaled=args_dict["lr_scaled"], mode_T=args_dict["mode_T"], lamda_T=args_dict["lamda_T"], if_random_shift=args_dict["if_random_shift"])
    time_start = time.time()
    img_out = ALD_sampler(**ALD_call_params)[0]  # (B, T, 1, H, W)
    time_end = time.time()

    # save the first batch
    save_vol_as_gif(torch.abs(img_out[0]), save_dir=args_dict["save_dir"], filename=f"recons_mag.gif")
    save_vol_as_gif(normalize_phase(torch.angle(img_out[0])), save_dir=args_dict["save_dir"], filename=f"recons_phase.gif")

    filename = create_filename(filename_dict, suffix=".png")
    vis_images(torch.abs(img_out[0, 0]), torch.angle(img_out[0, 0]), if_save=True, save_dir=args_dict["save_dir"], filename=filename)

    # img_out: (B, T, 1, H, W), measurement: (num_sens, B, T, 1, H, W)
    img_out_reshaped = img_out.reshape(-1, *img_out.shape[2:])  # (BT, 1, H, W)
    measurement_reshaped = measurement.reshape(measurement.shape[0], -1, *measurement.shape[3:])  # (num_sens, BT, 1, H, W)
    l2_error = torch.sum(torch.abs(linear_tfm(img_out_reshaped).detach().cpu() - measurement_reshaped.detach().cpu()) ** 2,
                         dim=(1, 2, 3)).mean().item()
    print("-" * 100)
    print(args_dict)
    print(f"reconstruction error: {l2_error}")
    print(f"reconstruction time: {time_end - time_start}")

    save_dir = args_dict["save_dir"]
    torch.save(img_complex.detach().cpu(), os.path.join(save_dir, "original.pt"))
    torch.save(measurement.detach().cpu(), os.path.join(save_dir, "measurement.pt"))
    torch.save(direct_recons.detach().cpu(), os.path.join(save_dir, "ZF.pt"))
    torch.save(img_out.detach().cpu(), os.path.join(save_dir, "reconstructions.pt"))

    log_file.close()

    sys.stdout = original_stdout
    sys.stderr = original_stderr

    with open(os.path.join(save_dir, "args_dict.pkl"), "wb") as wf:
        pickle.dump(args_dict, wf)
