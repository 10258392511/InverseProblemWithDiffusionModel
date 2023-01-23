import sys
import os

path = "/scratch/zhexwu"
# path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if path not in sys.path:
    sys.path.append(path)

import argparse
import InverseProblemWithDiffusionModel.helpers.pytorch_utils as ptu

from InverseProblemWithDiffusionModel.helpers.load_data import load_config, REGISTERED_DATA_CONFIG_FILENAME
from InverseProblemWithDiffusionModel.helpers.load_model import reload_model, TASK_NAME_TO_MODEL_CTOR
from InverseProblemWithDiffusionModel.ncsn.models.ALD_optimizers import ALDUnconditionalSampler
from InverseProblemWithDiffusionModel.ncsn.models import get_sigmas
from InverseProblemWithDiffusionModel.helpers.utils import vis_tensor, create_filename, vis_multi_channel_signal


if __name__ == '__main__':
    """
    python scripts/unconditioned_sampling.py --ds_name CINE127 --mode complex
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds_name", required=True, choices=list(REGISTERED_DATA_CONFIG_FILENAME.keys()))
    parser.add_argument("--task_name", required=True, choices=["Diffusion", "Diffusion1D"], default="Diffusion")
    parser.add_argument("--mode", required=True, choices=["real-valued", "mag", "complex"])
    # parser.add_argument("--if_conditioned", action="store_true")
    parser.add_argument("--num_steps_each", type=int, default=3)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--save_dir", default="../outputs")
    args_dict = vars(parser.parse_args())
    args_dict["if_conditioned"] = False
    device = ptu.DEVICE

    config = load_config(args_dict["ds_name"], args_dict["mode"], device)
    print(config.model.num_classes)
    scorenet = reload_model(args_dict["task_name"], args_dict["ds_name"], args_dict["mode"])
    ALD_sampler_params = {
        "n_steps_each": args_dict["num_steps_each"],
        "step_lr": config.sampling.step_lr,
        "final_only": config.sampling.final_only,
        "denoise": config.sampling.denoise
    }
    sigmas = get_sigmas(config)
    if args_dict["task_name"] == "Diffusion1D":
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

    elif args_dict["task_name"] == "Diffusion1D":
        # images: (B, C, T)
        
        vis_multi_channel_signal(images[0], if_save=True, save_dir=args_dict["save_dir"], filename="unconditioned_sample.png")
