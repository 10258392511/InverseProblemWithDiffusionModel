import torch
import pytorch_lightning as pl
import os

from InverseProblemWithDiffusionModel.helpers.losses import get_loss_fn
from InverseProblemWithDiffusionModel.configs.general_configs import general_config
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from datetime import datetime
from typing import List, Any


class TrainScoreModel(pl.LightningModule):
    def __init__(self, score_model, ds_dict, params):
        """
        ds_dict: keys: train, val
        params: batch_size, lr, sde
        """
        super(TrainScoreModel, self).__init__()
        self.params = params
        self.model = score_model
        self.ds_dict = ds_dict
        self.batch_size = self.params["batch_size"]
        self.lr = self.params["lr"]
        self.loss_fn = get_loss_fn(params["sde"])

    def training_step(self, batch, batch_idx):
        if isinstance(batch, list):
            X = batch[0]  # (B, C, H, W)
        else:
            X = batch
        loss = self.loss_fn(self.model, X)

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        out_dict = {"loss": loss}

        return out_dict

    def validation_step(self, batch, batch_idx):
        if isinstance(batch, list):
            X = batch[0]  # (B, C, H, W)
        else:
            X = batch
        loss = self.loss_fn(self.model, X)

        self.log("val_loss", loss, prog_bar=True, on_epoch=True)

    def train_dataloader(self):
        ds_key = "train"
        ds = self.ds_dict[ds_key]
        dataloader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        return dataloader

    def val_dataloader(self):
        ds_key = "val"
        ds = self.ds_dict[ds_key]
        dataloader = DataLoader(ds, batch_size=self.batch_size)

        return dataloader

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        opt_config = {
            "optimizer": opt
        }

        return opt_config


def get_score_model_trainer(callbacks: List[pl.Callback], mode="train"):
    assert mode in ["train", "debug"]
    time_stamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
    logger = TensorBoardLogger(".", version=time_stamp)
    callbacks = [ModelCheckpoint(os.path.join("lightning_logs", time_stamp, "checkpoints"), monitor="val_loss")] \
                + callbacks

    trainer = None
    train_params = dict(accelerator="gpu",
                        devices=1,
                        logger=logger,
                        callbacks=callbacks)

    if mode == "debug":
        train_params_cp = train_params.copy()
        train_params_cp.update({
            "fast_dev_run": 2
        })
        trainer = pl.Trainer(**train_params_cp)

    else:
        train_params_cp = train_params.copy()
        train_params_cp.update({
            "precision": 16,
            "num_sanity_val_steps": -1,
            "max_epochs": general_config.max_epochs
        })
        trainer = pl.Trainer(**train_params_cp)

    return trainer

