import torch
import pytorch_lightning as pl
import os

from InverseProblemWithDiffusionModel.helpers.losses import get_loss_fn
from InverseProblemWithDiffusionModel.ncsn.losses.dsm import anneal_dsm_score_estimation
from InverseProblemWithDiffusionModel.ncsn.losses.clf_loss import clf_loss_with_perturbation
from InverseProblemWithDiffusionModel.ncsn.losses.seg_loss import seg_loss_with_perturbation
from InverseProblemWithDiffusionModel.configs.general_configs import general_config
from InverseProblemWithDiffusionModel.ncsn.models import get_sigmas
from InverseProblemWithDiffusionModel.helpers.utils import get_optimizer
from torch.utils.data import DataLoader
from monai.data import DataLoader as m_DataLoader
from torchmetrics import MetricCollection, Accuracy
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from monai.metrics import DiceMetric
from monai.data import decollate_batch
from InverseProblemWithDiffusionModel.helpers.load_data import collate_batch
from monai.optimizers import Novograd
from monai.transforms import Compose, AsDiscrete
from monai.utils import CommonKeys
from datetime import datetime
from typing import List, Any


class TrainScoreModel(pl.LightningModule):
    def __init__(self, score_model, ds_dict, params):
        """
        ds_dict: keys: train, val
        params: batch_size, lr, sde, if_centering
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
        if self.params["if_centering"]:
            X = 2 * X - 1
        loss = self.loss_fn(self.model, X)

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        out_dict = {"loss": loss}

        return out_dict

    def validation_step(self, batch, batch_idx):
        if isinstance(batch, list):
            X = batch[0]  # (B, C, H, W)
        else:
            X = batch
        if self.params["if_centering"]:
            X = 2 * X - 1

        loss = self.loss_fn(self.model, X)

        self.log("val_loss", loss, prog_bar=True, on_epoch=True)

    def forward(self, X, t):
        X = X.float()
        t = t.float()
        return self.model(X, t)

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


class TrainScoreModelDiscrete(pl.LightningModule):
    def __init__(self, score_model, ds_dict, params):
        """
        ds_dict: keys: train, val
        params: batch_size, lr, num_workers, data_mode

        Data normalization goes into "transforms" when loading the data
        """
        super(TrainScoreModelDiscrete, self).__init__()
        self.params = params
        self.model = score_model
        self.config = score_model.config
        self.sigmas = get_sigmas(self.config)
        self.ds_dict = ds_dict
        self.batch_size = self.params["batch_size"]
        self.lr = self.params["lr"]
        self.num_workers = params.get("num_workers", 0)
        self.loss_fn = anneal_dsm_score_estimation

    def training_step(self, batch, batch_idx):
        if isinstance(batch, list):
            X = batch[0]
        elif isinstance(batch, dict):
            X = batch[CommonKeys.IMAGE]
        else:
            X = batch
        # X: (B, C, H, W)
        X = collate_batch(X, self.params["data_mode"])
        loss = self.loss_fn(self.model, X, self.sigmas)

        out_dict = {"loss": loss}
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        return out_dict

    def validation_step(self, batch, batch_idx):
        if isinstance(batch, list):
            X = batch[0]
        elif isinstance(batch, dict):
            X = batch[CommonKeys.IMAGE]
        else:
            X = batch
        # X: (B, C, H, W)
        X = collate_batch(X, self.params["data_mode"])
        loss = self.loss_fn(self.model, X, self.sigmas)

        self.log("val_loss", loss, prog_bar=True, on_epoch=True)

    def train_dataloader(self):
        ds_key = "train"
        ds = self.ds_dict[ds_key]
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

        return loader

    def val_dataloader(self):
        ds_key = "val"
        ds = self.ds_dict[ds_key]
        loader = DataLoader(ds, batch_size=self.batch_size, num_workers=self.num_workers)

        return loader

    def configure_optimizers(self):
        opt = get_optimizer(self.config, self.model.parameters())
        opt_config = {
            "optimizer": opt
        }

        return opt_config


class TrainClf(pl.LightningModule):
    def __init__(self, model, ds_dict, params, config):
        """
        ds_dict: keys: train, val
        params: batch_size, lr, num_workers, data_mode
        """
        super(TrainClf, self).__init__()
        self.params = params
        self.ds_dict = ds_dict
        self.config = config
        self.model = model
        self.sigmas = get_sigmas(self.config)
        self.batch_size = self.params["batch_size"]
        self.lr = self.params["lr"]
        self.num_workers = params.get("num_workers", 0)
        self.loss_fn = clf_loss_with_perturbation
        metrics = MetricCollection([Accuracy()])
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")

    def training_step(self, batch, batch_idx):
        X, y = batch
        X = collate_batch(X, self.params["data_mode"])
        loss, y_pred = clf_loss_with_perturbation(self.model, X, y, self.sigmas)
        self.train_metrics(y_pred, y)
        out_dict = {"loss": loss}
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        return out_dict

    def training_epoch_end(self, outputs):
        metrics = self.train_metrics.compute()
        self.log_dict(metrics, prog_bar=True, on_epoch=True)
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        X, y = batch
        X = collate_batch(X, self.params["data_mode"])
        loss, y_pred = clf_loss_with_perturbation(self.model, X, y, self.sigmas)
        self.val_metrics(y_pred, y)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)

    def validation_epoch_end(self, outputs):
        metrics = self.val_metrics.compute()
        self.log_dict(metrics, prog_bar=True, on_epoch=True)
        self.val_metrics.reset()

    def train_dataloader(self):
        ds_key = "train"
        ds = self.ds_dict[ds_key]
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

        return loader

    def val_dataloader(self):
        ds_key = "val"
        ds = self.ds_dict[ds_key]
        loader = DataLoader(ds, batch_size=self.batch_size, num_workers=self.num_workers)

        return loader

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.model.parameters(), self.lr)
        opt_config = {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, min_lr=1e-6),
                "monitor": "val_loss"
            }
        }

        return opt_config


class TrainSeg(pl.LightningModule):
    def __init__(self, model, ds_dict, params, config):
        """
       ds_dict: keys: train, val
       params: batch_size, lr, num_workers, num_cls, data_mode
       """
        super(TrainSeg, self).__init__()
        self.params = params
        self.ds_dict = ds_dict
        self.config = config
        self.model = model
        self.sigmas = get_sigmas(self.config)
        self.batch_size = self.params["batch_size"]
        self.lr = self.params["lr"]
        self.num_workers = params.get("num_workers", 0)
        self.loss_fn = seg_loss_with_perturbation
        self.val_metric = DiceMetric(include_background=False, reduction="mean")
        self.post_processing_pred = Compose([AsDiscrete(argmax=True, to_onehot=self.params["num_cls"])])
        self.post_processing_label = Compose([AsDiscrete(to_onehot=self.params["num_cls"])])

    def training_step(self, batch, batch_idx):
        # (B, C, H, W)
        img, label = batch[CommonKeys.IMAGE], batch[CommonKeys.LABEL]
        img = collate_batch(img, self.params["data_mode"])
        loss, pred = self.loss_fn(self.model, img, label, self.sigmas)
        out_dict = {"loss": loss}
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        return out_dict

    def validation_step(self, batch, batch_idx):
        img, label = batch[CommonKeys.IMAGE], batch[CommonKeys.LABEL]
        img = collate_batch(img, self.params["data_mode"])
        loss, pred = self.loss_fn(self.model, img, label, self.sigmas)
        self.log("val_loss", loss, on_epoch=True)
        # label: (B, 1, H, W); pred: (B, C, H, W)
        label = [self.post_processing_label(item) for item in decollate_batch(label)]
        pred = [self.post_processing_pred(item) for item in decollate_batch(pred)]
        self.val_metric(y_pred=pred, y=label)

    def validation_epoch_end(self, outputs):
        metric = self.val_metric.aggregate().item()
        self.log("val_dsc", metric, prog_bar=True, on_epoch=True)
        self.val_metric.reset()

    def train_dataloader(self):
        ds_key = "train"
        ds = self.ds_dict[ds_key]
        loader = m_DataLoader(ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

        return loader

    def val_dataloader(self):
        ds_key = "val"
        ds = self.ds_dict[ds_key]
        loader = m_DataLoader(ds, batch_size=self.batch_size, num_workers=self.num_workers)

        return loader

    def configure_optimizers(self):
        opt = Novograd(self.model.parameters(), lr=self.lr)
        opt_config = {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, min_lr=1e-6),
                "monitor": "val_dsc"
            }
        }

        return opt_config


def get_score_model_trainer(callbacks: List[pl.Callback], mode="train", log_dir=".", log_name="lightning_logs",
                            if_monitor=True):
    assert mode in ["train", "debug"]
    time_stamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
    logger = TensorBoardLogger(log_dir, name=log_name, version=time_stamp)
    ckpt_path = os.path.join(log_dir, log_name, time_stamp, "checkpoints")

    if if_monitor:
        callbacks = [ModelCheckpoint(ckpt_path, monitor="val_loss")] \
                    + callbacks

    trainer = None
    train_params = dict(
        # default_root_dir=os.path.dirname(ckpt_path),
        accelerator="cpu",
        devices=1,
        logger=logger,
        callbacks=callbacks,
        limit_train_batches=2,
        limit_val_batches=2
    )

    if mode == "debug":
        train_params_cp = train_params.copy()
        train_params_cp.update({
            "fast_dev_run": 2
        })
        trainer = pl.Trainer(**train_params_cp)

    else:
        train_params_cp = train_params.copy()
        train_params_cp.update({
            # "precision": 16,
            "num_sanity_val_steps": -1,
            "max_epochs": general_config.max_epochs,
            "check_val_every_n_epoch": 1
        })
        trainer = pl.Trainer(**train_params_cp)

    return trainer
