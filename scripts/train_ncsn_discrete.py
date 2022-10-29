import sys

path = "D:\\testings\\Python\\TestingPython\\"
if path not in sys.path:
    sys.path.append(path)


import numpy as np
import torch
import yaml
import InverseProblemWithDiffusionModel.helpers.pytorch_utils as ptu

from InverseProblemWithDiffusionModel.ncsn.models import get_sigmas
from InverseProblemWithDiffusionModel.ncsn.models.ncsnv2 import NCSNv2
from InverseProblemWithDiffusionModel.helpers.utils import load_yml_file
from InverseProblemWithDiffusionModel.helpers.load_data import load_mnist
from InverseProblemWithDiffusionModel.helpers.pl_helpers import TrainScoreModelDiscrete, get_score_model_trainer
from InverseProblemWithDiffusionModel.helpers.pl_callbacks import EMA, ValVisualizationDiscrete
from torch.utils.data import DataLoader


if __name__ == '__main__':
    train_ds = load_mnist("../data/", mode="train")
    test_ds = load_mnist("../data/", mode="val")

    config_path = "../ncsn/configs/mnist.yml"
    config_namespace = load_yml_file(config_path)
    config_namespace.device = ptu.DEVICE
    model = NCSNv2(config_namespace)

    ds_dict = {
        "train": train_ds,
        "val": test_ds
    }
    params = {
        "batch_size": config_namespace.training.batch_size,
        "lr": config_namespace.optim.lr
    }
    lit_model = TrainScoreModelDiscrete(model, ds_dict, params)

    callbacks = [EMA(config_namespace.model.ema_rate),
                 ValVisualizationDiscrete(config_namespace.sampling.batch_size)]

    trainer = get_score_model_trainer(
        callbacks=callbacks,
        mode="train",
        log_dir="../logs/",
        log_name="ncsn_logs"
    )

    trainer.fit(lit_model)
