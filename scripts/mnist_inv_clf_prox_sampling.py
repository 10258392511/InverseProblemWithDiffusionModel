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
from InverseProblemWithDiffusionModel.ncsn.linear_transforms.masking import SkipLines
from InverseProblemWithDiffusionModel.ncsn.models.proximal_op import get_proximal
from InverseProblemWithDiffusionModel.ncsn.models.ALD_optimizers import ALDInvClfProximal
from InverseProblemWithDiffusionModel.helpers.utils import vis_images, create_filename


if __name__ == '__main__':
    """
    python scripts/mnist_inv_clf_prox_sampling.py
    """
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_skip_lines", type=int, default=2)
    parser.add_argument("--clf_start_time", type=float, default=0.)
    parser.add_argument("--clf_step_type", default="linear")
    parser.add_argument("--lamda", type=float, default=1.)
    parser.add_argument("--proximal_type", default="L2Penalty")
    parser.add_argument("--ds_idx", type=int, default=0)
    parser.add_argument("--save_dir", default="../outputs")
    args_dict = vars(parser.parse_args())

    if not os.path.isdir(args_dict["save_dir"]):
        os.makedirs(args_dict["save_dir"])

    ds_name = "MNIST"
    mode = "real-valued"
    device = ptu.DEVICE

    config = load_config(ds_name, mode, device)
    ds = load_data(ds_name, "val")
    img, label = ds[args_dict["ds_idx"]]
    img = img.unsqueeze(0).to(device)  # (1, C, H, W)
    label = torch.tensor(label).view((1,)).to(device)  # (1,)

    scorenet = reload_model("Diffusion", ds_name, mode)
    clf = reload_model("Clf", ds_name, mode)
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
    linear_tfm = SkipLines(args_dict["num_skip_lines"], x_mod_shape[1:])
    proximal_constr = get_proximal(args_dict["proximal_type"])
    proximal = proximal_constr(linear_tfm)
    measurement = linear_tfm(img)
    ALD_sampler = ALDInvClfProximal(
        proximal,
        args_dict["clf_start_time"],
        args_dict["clf_step_type"],
        x_mod_shape,
        scorenet,
        sigmas,
        ALD_sampler_params,
        config,
        measurement,
        linear_tfm,
        clf,
        device=device
    )

    vis_images(img[0], if_save=True, save_dir=args_dict["save_dir"], filename="original_mnist.png",
               titles=[f"cls: {label[0].item()}"])
    vis_images(linear_tfm.conj_op(measurement)[0], if_save=True, save_dir=args_dict["save_dir"],
               filename=f"downsampled_mnist_R_{args_dict['num_skip_lines']}.png", titles=[f"cls: {label[0].item()}"])

    # for i, lamda_iter in enumerate(lamda_grid):
    filename_dict = {
        "ds_name": ds_name,
        "num_skip_lines": args_dict["num_skip_lines"],
        "lamda": args_dict["lamda"],
        "clf_start": args_dict["clf_start_time"],
        "clf_type": args_dict["clf_step_type"]
    }
    log_filename = create_filename(filename_dict, suffix=".txt")
    log_file = open(os.path.join(args_dict["save_dir"], log_filename), "w")
    sys.stdout = log_file
    sys.stderr = log_file

    ALD_call_params = dict(cls=label, lamda=args_dict["lamda"])
    img_out = ALD_sampler(**ALD_call_params)[0]

    filename = create_filename(filename_dict, suffix=".png")
    vis_images(img_out[0], if_save=True, save_dir=args_dict["save_dir"], filename=filename,
               titles=[f"cls: {label[0].item()}"])

    del img_out
    log_file.close()

    sys.stdout = original_stdout
    sys.stderr = original_stderr
