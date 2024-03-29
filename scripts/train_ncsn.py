import sys
import os

path = "/scratch/zhexwu"
# path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if path not in sys.path:
    sys.path.append(path)

import argparse
import numpy as np

from InverseProblemWithDiffusionModel.helpers.load_data import load_data, load_config
from InverseProblemWithDiffusionModel.helpers.pl_helpers import TrainScoreModelDiscrete, get_score_model_trainer
from InverseProblemWithDiffusionModel.helpers.load_model import load_model
from InverseProblemWithDiffusionModel.helpers.pl_callbacks import EMA, ValVisualizationDiscrete


if __name__ == '__main__':
    """
    python scripts/train_ncsn.py --ds_name ACDC --task_name Diffusion --mode complex
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds_name", required=True)  # CINE64 or CINE 127, "1D" will be appended later
    parser.add_argument("--task_name", required=True)  # Diffusion, Diffusion1D or Diffusion3D
    parser.add_argument("--mode", required=True)  # real-imag &etc
    parser.add_argument("--flatten_type", default="spatial")
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--if_centering", action="store_true")
    parser.add_argument("--notes", default="")
    args = vars(parser.parse_args())
    ds_name = args["ds_name"]
    task_name = args["task_name"]
    mode = args["mode"]

    log_dir = "./"
    log_name = "ncsn_logs"

    if "CINE" in args["ds_name"] and args["flatten_type"] == "temporal":
        config = load_config(f"{ds_name}_1D", mode=mode, flatten_type=args["flatten_type"])
    else:
        config = load_config(ds_name, mode=mode)
        
    if "CINE" in args["ds_name"] and args["flatten_type"] == "temporal":
        win_size = int(np.sqrt(config.data.channels))
        train_ds = load_data(ds_name, "train", if_aug=False, flatten_type=args["flatten_type"], win_size=win_size, resize_shape_T=config.data.image_size)
        val_ds = load_data(ds_name, "val", if_aug=False, flatten_type=args["flatten_type"], win_size=win_size, resize_shape_T=config.data.image_size)
        ds_name = f"{ds_name}_1D"
    elif ds_name == "SanityCheck1D":
        num_channels = config.data.channels
        num_features = config.data.image_size
        train_ds = load_data(ds_name, "train", num_channels=num_channels, num_features=num_features)
        val_ds = load_data(ds_name, "val", num_channels=num_channels, num_features=num_features)
    else:
        train_ds = load_data(ds_name, "train", if_aug=False)
        val_ds = load_data(ds_name, "val", if_aug=False)
    ds_dict = {
        "train": train_ds,
        "val": val_ds
    }
    print(f"batch_size: {config.training.batch_size}")
    print("-" * 100)
    model = load_model(config, task_name)
    print(f"model: {type(model)}")
    params = {
        "batch_size": config.training.batch_size,
        "lr": config.optim.lr,
        "data_mode": mode,
        "num_workers": args["num_workers"],
        "if_centering": True
    }
    lit_model = TrainScoreModelDiscrete(model, ds_dict, params)
    callbacks = [EMA(config.model.ema_rate)]
    trainer, log_dir_full = get_score_model_trainer(
        callbacks=callbacks,
        mode="train",
        log_dir=".",
        log_name="ncsn_logs",
        num_epochs=config.training.n_epochs,
        return_log_dir=True,
        if_monitor=False
    )

    if not os.path.isdir(log_dir_full):
        os.makedirs(log_dir_full)

    args.update({
        "num_channels": config.data.channels,
        "sigma_begin": config.model.sigma_begin,
        "num_classes": config.model.num_classes
    })
    with open(f"{log_dir_full}/desc.txt", "w") as wf:
        for key, val in args.items():
            wf.write(f"{key}: {val}\n")

    trainer.fit(lit_model)
