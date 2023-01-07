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

from InverseProblemWithDiffusionModel.helpers.load_data import load_data, load_config, add_phase
from InverseProblemWithDiffusionModel.helpers.load_model import reload_model
from InverseProblemWithDiffusionModel.ncsn.models import get_sigmas
from InverseProblemWithDiffusionModel.ncsn.linear_transforms.undersampling_fourier import RandomUndersamplingFourier
from InverseProblemWithDiffusionModel.ncsn.models.proximal_op import get_proximal
from InverseProblemWithDiffusionModel.ncsn.models.ALD_optimizers import ALDInvSegProximalRealImag
from InverseProblemWithDiffusionModel.helpers.utils import vis_images, create_filename
from monai.utils import CommonKeys


if __name__ == '__main__':
    """
    python scripts/cine_inv_sampling_keep_center_prox_real_imag.py
    """
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    parser = argparse.ArgumentParser()
    parser.add_argument("--R", type=int, default=6)
    parser.add_argument("--center_lines_frac", type=float, default=1 / 4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seg_start_time", type=float, default=0.)
    parser.add_argument("--seg_step_type", default="linear")
    parser.add_argument("--lamda", type=float, default=0.1)
    parser.add_argument("--step_lr", type=float, default=0.0000009)  # overwriting config.sampling.step_lr
    parser.add_argument("--num_steps_each", type=int, default=3)
    parser.add_argument("--lr_scaled", type=float, default=1.)
    parser.add_argument("--proximal_type", default="L2Penalty")
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--ds_idx", type=int, default=0)
    parser.add_argument("--save_dir", default="../outputs")
    args_dict = vars(parser.parse_args())

    ds_name = "CINE64"
    mode = "real-valued"
    device = ptu.DEVICE

    config = load_config(ds_name, mode, device)
    ds = load_data(ds_name, "val")
    data = ds[args_dict["ds_idx"]]
    # (1, H, W), (1, H, W)
    img = data[0]
    img = img.unsqueeze(0).to(device)  # (1, 1, H, W)

    scorenet = reload_model("Diffusion", ds_name, mode)
    seg = reload_model("Seg", "ACDC", mode)
    ALD_sampler_params = {
        "n_steps_each": config.sampling.n_steps_each,
        "step_lr": config.sampling.step_lr,
        "final_only": config.sampling.final_only,
        "denoise": config.sampling.denoise
    }
    ALD_sampler_params["step_lr"] = args_dict["step_lr"]
    ALD_sampler_params["n_steps_each"] = args_dict["num_steps_each"]
    ALD_sampler_params["denoise"] = True
    # print(ALD_sampler_params)
    sigmas = get_sigmas(config, mode="recons")
    x_mod_shape = (
        args_dict["num_samples"],
        config.data.channels,
        config.data.image_size,
        config.data.image_size
    )
    linear_tfm = RandomUndersamplingFourier(args_dict["R"], args_dict["center_lines_frac"], x_mod_shape[1:],
                                            args_dict["seed"])
    proximal_constr = get_proximal(args_dict["proximal_type"])
    proximal = proximal_constr(linear_tfm)

    img_complex = add_phase(img, init_shape=(5, 5), seed=args_dict["seed"])  # (1, 1, H, W)
    measurement = linear_tfm(img_complex.to(torch.complex64))  # (1, 1, H, W)
    measurement = measurement.repeat(args_dict["num_samples"], 1, 1, 1)

    ALD_sampler = ALDInvSegProximalRealImag(
        proximal,
        args_dict["seg_start_time"],
        args_dict["seg_step_type"],
        x_mod_shape,
        scorenet,
        sigmas,
        ALD_sampler_params,
        config,
        measurement,
        linear_tfm,
        seg=seg,
        device=device
    )

    eps = 1e-6
    vis_images(torch.abs(img_complex[0]), torch.angle(img_complex[0]), if_save=True, save_dir=args_dict["save_dir"], filename="original_cine64.png")
    vis_images(torch.log(torch.abs(measurement[0]) + eps), torch.angle(measurement[0]), if_save=True, save_dir=args_dict["save_dir"],
               filename=f"cine64_measurement_R_{args_dict['R']}_frac_{args_dict['center_lines_frac']}.png")
    direct_recons = linear_tfm.conj_op(measurement)[0]
    vis_images(torch.abs(direct_recons), torch.angle(direct_recons), if_save=True, save_dir=args_dict["save_dir"],
               filename=f"cine64_zero_padded_recons_R_{args_dict['R']}_frac_{args_dict['center_lines_frac']}.png")

    filename_dict = {
        "ds_name": ds_name,
        "R": args_dict["R"],
        "center_lines_frac": args_dict["center_lines_frac"],
        "seg_step_type": args_dict["seg_step_type"],
        "seg_start_time": args_dict["seg_start_time"]
    }
    log_filename = create_filename(filename_dict, suffix=".txt")
    log_file = open(os.path.join(args_dict["save_dir"], log_filename), "w")
    sys.stdout = log_file
    sys.stderr = log_file
    original_error = torch.sum(torch.abs(linear_tfm(direct_recons.unsqueeze(0)).detach().cpu() - measurement.detach().cpu()) ** 2, dim=(1, 2, 3)).mean().item()
    print(f"original error: {original_error}")

    ALD_call_params = dict(label=None, lamda=args_dict["lamda"], save_dir=args_dict["save_dir"],
                           lr_scaled=args_dict["lr_scaled"])
    img_out = ALD_sampler(**ALD_call_params)[0]  # (B, C, H, W)

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
