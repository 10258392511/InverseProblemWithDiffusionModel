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

from InverseProblemWithDiffusionModel.helpers.load_data import load_data, load_config
from InverseProblemWithDiffusionModel.helpers.load_model import reload_model
from InverseProblemWithDiffusionModel.ncsn.models import get_sigmas
from InverseProblemWithDiffusionModel.ncsn.linear_transforms.undersampling_fourier import RandomUndersamplingFourier
from InverseProblemWithDiffusionModel.ncsn.models.ALD_optimizers import ALDInvSeg
from InverseProblemWithDiffusionModel.helpers.utils import vis_images, create_filename
from monai.utils import CommonKeys


if __name__ == '__main__':
    """
    python scripts/acdc_inv_seg_sampling_keep_center.py
    """
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    parser = argparse.ArgumentParser()
    parser.add_argument("--R", type=int, default=6)
    parser.add_argument("--center_lines_frac", type=float, default=1 / 4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seg_step_type", default="linear")
    parser.add_argument("--ds_idx", type=int, default=0)
    parser.add_argument("--save_dir", default="../outputs")
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
    seg = reload_model("Seg", ds_name, mode)
    ALD_sampler_params = {
        "n_steps_each": config.sampling.n_steps_each,
        "step_lr": config.sampling.step_lr,
        "final_only": config.sampling.final_only,
        "denoise": config.sampling.denoise
    }
    sigmas = get_sigmas(config)
    x_mod_shape = (
        1,
        config.data.channels,
        config.data.image_size,
        config.data.image_size
    )
    linear_tfm = RandomUndersamplingFourier(args_dict["R"], args_dict["center_lines_frac"], x_mod_shape[1:],
                                            args_dict["seed"])
    measurement = linear_tfm(img.to(torch.complex64))  # (1, 1, H, W)

    vis_images(img[0], if_save=True, save_dir=args_dict["save_dir"], filename="original_acdc.png")
    vis_images(label[0], if_save=True, save_dir=args_dict["save_dir"], filename="original_seg.png")
    vis_images(torch.log(torch.abs(measurement[0])), if_save=True, save_dir=args_dict["save_dir"],
               filename=f"acdc_measurement_skip_lines_{args_dict['num_skip_lines']}.png")
    vis_images(torch.abs(linear_tfm.conj_op(measurement)[0]), if_save=True, save_dir=args_dict["save_dir"],
               filename=f"acdc_zero_padded_recons_skip_lines_{args_dict['num_skip_lines']}.png")

    seg_start_time = np.linspace(0, 1, 2)
    # seg_start_time = [0.]
    for i, seg_start_time_iter in enumerate(seg_start_time):
        filename_dict = {
            "ds_name": ds_name,
            "num_skip_lines": args_dict["num_skip_lines"],
            "seg_step_type": args_dict["seg_step_type"],
            "seg_start_time": seg_start_time_iter
        }
        log_filename = create_filename(filename_dict, suffix=".txt")
        log_file = open(os.path.join(args_dict["save_dir"], log_filename), "w")
        sys.stdout = log_file
        sys.stderr = log_file

        print(f"current start time: {seg_start_time_iter} ({i + 1}/{len(seg_start_time)})")
        ALD_sampler = ALDInvSeg(
            seg_start_time_iter,
            seg_step_type=args_dict["seg_step_type"],
            x_mod_shape=x_mod_shape,
            scorenet=scorenet,
            sigmas=sigmas,
            params=ALD_sampler_params,
            config=config,
            measurement=measurement,
            linear_tfm=linear_tfm,
            seg=seg,
            device=device
        )
        img_out = ALD_sampler(label=label, save_dir=os.path.join(args_dict["save_dir"], "temp/"))[0]

        filename = create_filename(filename_dict, suffix=".png")
        vis_images(torch.abs(img_out[0]), if_save=True, save_dir=args_dict["save_dir"], filename=filename)

        del img_out
        log_file.close()

    sys.stdout = original_stdout
    sys.stderr = original_stderr
