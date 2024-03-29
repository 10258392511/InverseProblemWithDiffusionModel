import sys
import os

path = "/scratch/zhexwu"
# path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if path not in sys.path:
    sys.path.append(path)

import argparse
import numpy as np
import torch
import einops
import InverseProblemWithDiffusionModel.helpers.pytorch_utils as ptu

from InverseProblemWithDiffusionModel.helpers.load_data import load_config, REGISTERED_DATA_CONFIG_FILENAME
from InverseProblemWithDiffusionModel.helpers.load_model import reload_model, TASK_NAME_TO_MODEL_CTOR
from InverseProblemWithDiffusionModel.ncsn.models.ALD_optimizers import ALDUnconditionalSampler
from InverseProblemWithDiffusionModel.ncsn.models import get_sigmas
from InverseProblemWithDiffusionModel.helpers.utils import (
    vis_tensor, 
    create_filename, 
    vis_multi_channel_signal,
    save_vol_as_gif
)


if __name__ == '__main__':
    """
    python scripts/unconditioned_sampling.py --ds_name CINE127 --mode complex
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds_name", required=True, choices=list(REGISTERED_DATA_CONFIG_FILENAME.keys()))
    parser.add_argument("--task_name", required=True, choices=["Diffusion", "Diffusion1D", "Diffusion3D"], default="Diffusion")
    parser.add_argument("--mode", required=True, choices=["real-valued", "mag", "complex"])
    # parser.add_argument("--if_conditioned", action="store_true")
    parser.add_argument("--num_steps_each", type=int, default=3)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--save_dir", default="../outputs")
    parser.add_argument("--step_lr", type=float, default=0.0000009)
    parser.add_argument("--if_save_fig", action="store_true")  # for 1D only; 2D images are always saved
    parser.add_argument("--if_save_as_gif", action="store_true")  # for 1D only
    args_dict = vars(parser.parse_args())
    args_dict["if_conditioned"] = False
    device = ptu.DEVICE

    # if "1D" in args_dict["task_name"]:
    #     config_ds_name = f"{args_dict['ds_name']}_1D"
    # else:
    #     config_ds_name = args_dict["ds_name"]
    # print(config_ds_name)
    config = load_config(args_dict["ds_name"], args_dict["mode"], device, )
    print(config.model.num_classes)
    scorenet = reload_model(args_dict["task_name"], args_dict["ds_name"], args_dict["mode"])
    ALD_sampler_params = {
        "n_steps_each": args_dict["num_steps_each"],
        # "step_lr": config.sampling.step_lr,
        "step_lr": args_dict["step_lr"],
        "final_only": config.sampling.final_only,
        "denoise": config.sampling.denoise
    }
    sigmas = get_sigmas(config)
    if args_dict["task_name"] in ["Diffusion1D", "Diffusion3D"]:
        x_mod_shape = (
            args_dict["num_samples"],
            config.data.channels,
            config.data.image_size
        )
    else:
        x_mod_shape = (
            args_dict["num_samples"],
            config.data.channels,
            config.data.image_size,
            config.data.image_size
        )
    ALD_sampler = ALDUnconditionalSampler(
        x_mod_shape,
        scorenet,
        sigmas,
        ALD_sampler_params,
        config,
        device=device
    )
    images = ALD_sampler()[0]  # (B, C, H, W) or (B, C, T)

    if not os.path.isdir(args_dict["save_dir"]):
        os.makedirs(args_dict["save_dir"])
    
    if args_dict["task_name"] == "Diffusion":
        fig = vis_tensor(images[:, 0:1, ...])  # save only the 1st channel
        filename = create_filename(
            {
                "ds_name": args_dict["ds_name"],
                "mode": args_dict["mode"],
                "if_conditioned": args_dict["if_conditioned"]
            },
            suffix=".png"
        )
        save_path = os.path.join(args_dict["save_dir"], filename)
        fig.savefig(save_path)

    elif args_dict["task_name"] in ["Diffusion1D", "Diffusion3D"]:
        # images: (B, C, T)
        if args_dict["if_save_fig"]:
            # all channels
            ylim = (-2, 2)
            vis_multi_channel_signal(images[0][:4], if_save=True, save_dir=args_dict["save_dir"], filename="unconditioned_sample.png", ylim=ylim)
        if args_dict["if_save_as_gif"]:
            # save the first batch only
            B, C, T  = images.shape
            images_as_vol = einops.rearrange(images[0], "(C1 kx ky) T -> T C1 kx ky", kx=int(np.sqrt(C)), C1=1)
            save_vol_as_gif(images_as_vol, args_dict["save_dir"], "samples.gif")
    
    torch.save(images, os.path.join(args_dict["save_dir"], "samples.pt"))
