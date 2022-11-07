import sys
import os

path = "/scratch/zhexwu"
# path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if path not in sys.path:
    sys.path.append(path)

import argparse

from InverseProblemWithDiffusionModel.helpers.load_data import load_data, load_config
from InverseProblemWithDiffusionModel.helpers.pl_helpers import TrainClf, get_score_model_trainer
from InverseProblemWithDiffusionModel.helpers.load_model import load_model


if __name__ == '__main__':
    """
    python scripts/train_clf.py --ds_name MNIST --task_name Clf --mode complex
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds_name", required=True)
    parser.add_argument("--task_name", required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--num_workers", type=int, default=0)
    args = vars(parser.parse_args())
    ds_name = args["ds_name"]
    task_name = args["task_name"]
    mode = args["mode"]

    log_dir = "./"
    log_name = "clf_logs"

    train_ds = load_data(ds_name, "train")
    val_ds = load_data(ds_name, "val")
    ds_dict = {
        "train": train_ds,
        "val": val_ds
    }
    config = load_config(ds_name, mode=mode)
    model = load_model(config, task_name, use_net_params=True)
    params = {
        "batch_size": config.training.clf_batch_size,
        "lr": config.optim.lr,
        "data_mode": mode,
        "num_workers": args["num_workers"]
    }
    lit_model = TrainClf(model, ds_dict, params, config)
    callbacks = []
    trainer, log_dir_full = get_score_model_trainer(
        callbacks=callbacks,
        mode="train",
        log_dir=log_dir,
        log_name=log_name,
        num_epochs=100,
        return_log_dir=True
    )

    if not os.path.isdir(log_dir_full):
        os.makedirs(log_dir_full)

    with open(f"{log_dir_full}/desc.txt", "w") as wf:
        for key, val in args.items():
            wf.write(f"{key}: {val}\n")

    trainer.fit(lit_model)
