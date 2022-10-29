from copy import deepcopy
from typing import Optional, Union, Dict, Any

import pytorch_lightning as pl
import torch
import InverseProblemWithDiffusionModel.helpers.pytorch_utils as ptu

from overrides import overrides
from pytorch_lightning.utilities import rank_zero_only
from InverseProblemWithDiffusionModel.helpers.utils import get_data_inverse_scaler, data_transform
from InverseProblemWithDiffusionModel.sde.sampling import get_sampling_fn
from InverseProblemWithDiffusionModel.ncsn.models import anneal_Langevin_dynamics
from torchvision.utils import make_grid


# Implementation source: https://github.com/Lightning-AI/lightning/issues/10914
# "EMA weights are saved as "ema_state_dict" in the callback state. So, you can retrieve them manually
# outside lightning."
class EMA(pl.Callback):
    """Implements EMA (exponential moving average) to any kind of model.
    EMA weights will be used during validation and stored separately from original model weights.

    How to use EMA:
        - Sometimes, last EMA checkpoint isn't the best as EMA weights metrics can show long oscillations in time. See
          https://github.com/rwightman/pytorch-image-models/issues/102
        - Batch Norm layers and likely any other type of norm layers doesn't need to be updated at the end. See
          discussions in: https://github.com/rwightman/pytorch-image-models/issues/106#issuecomment-609461088 and
          https://github.com/rwightman/pytorch-image-models/issues/224
        - For object detection, SWA usually works better. See   https://github.com/timgaripov/swa/issues/16

    Implementation detail:
        - See EMA in Pytorch Lightning: https://github.com/PyTorchLightning/pytorch-lightning/issues/10914
        - When multi gpu, we broadcast ema weights and the original weights in order to only hold 1 copy in memory.
          This is specially relevant when storing EMA weights on CPU + pinned memory as pinned memory is a limited
          resource. In addition, we want to avoid duplicated operations in ranks != 0 to reduce jitter and improve
          performance.
    """

    def __init__(self, decay: float = 0.9999, ema_device: Optional[Union[torch.device, str]] = None, pin_memory=True):
        super().__init__()
        self.decay = decay
        self.ema_device: str = f"{ema_device}" if ema_device else None  # perform ema on different device from the model
        self.ema_pin_memory = pin_memory if torch.cuda.is_available() else False  # Only works if CUDA is available
        self.ema_state_dict: Dict[str, torch.Tensor] = {}
        self.original_state_dict = {}
        self._ema_state_dict_ready = False

    @staticmethod
    def get_state_dict(pl_module: pl.LightningModule):
        """Returns state dictionary from pl_module. Override if you want filter some parameters and/or buffers out.
        For example, in pl_module has metrics, you don't want to return their parameters.

        code:
            # Only consider modules that can be seen by optimizers. Lightning modules can have others nn.Module attached
            # like losses, metrics, etc.
            patterns_to_ignore = ("metrics1", "metrics2")
            return dict(filter(lambda i: i[0].startswith(patterns), pl_module.state_dict().items()))
        """
        return pl_module.state_dict()

    @overrides
    def on_train_start(self, trainer: "pl.Trainer", pl_module: pl.LightningModule) -> None:
        # Only keep track of EMA weights in rank zero.
        if not self._ema_state_dict_ready and pl_module.global_rank == 0:
            self.ema_state_dict = deepcopy(self.get_state_dict(pl_module))
            if self.ema_device:
                self.ema_state_dict = {k: tensor.to(device=self.ema_device) for k, tensor in
                                       self.ema_state_dict.items()}

            if self.ema_device == "cpu" and self.ema_pin_memory:
                self.ema_state_dict = {k: tensor.pin_memory() for k, tensor in self.ema_state_dict.items()}

        self._ema_state_dict_ready = True

    @rank_zero_only
    def on_train_batch_end(self, trainer: "pl.Trainer", pl_module: pl.LightningModule, *args, **kwargs) -> None:
        # Update EMA weights
        with torch.no_grad():
            for key, value in self.get_state_dict(pl_module).items():
                ema_value = self.ema_state_dict[key]
                ema_value.copy_(self.decay * ema_value + (1. - self.decay) * value, non_blocking=True)

    @overrides
    def on_validation_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not self._ema_state_dict_ready:
            return  # Skip Lightning sanity validation check if no ema weights has been loaded from a checkpoint.

        self.original_state_dict = deepcopy(self.get_state_dict(pl_module))
        pl_module.trainer.training_type_plugin.broadcast(self.ema_state_dict, 0)
        assert self.ema_state_dict.keys() == self.original_state_dict.keys(), \
            f"There are some keys missing in the ema static dictionary broadcasted. " \
            f"They are: {self.original_state_dict.keys() - self.ema_state_dict.keys()}"
        pl_module.load_state_dict(self.ema_state_dict, strict=False)

        if pl_module.global_rank > 0:
            # Remove ema state dict from the memory. In rank 0, it could be in ram pinned memory.
            self.ema_state_dict = {}

    @overrides
    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not self._ema_state_dict_ready:
            return  # Skip Lightning sanity validation check if no ema weights has been loaded from a checkpoint.

        # Replace EMA weights with training weights
        pl_module.load_state_dict(self.original_state_dict, strict=False)

    @overrides
    def on_save_checkpoint(
            self, trainer: pl.Trainer, pl_module: pl.LightningModule, checkpoint: Dict[str, Any]
    ) -> dict:
        return {"ema_state_dict": self.ema_state_dict, "_ema_state_dict_ready": self._ema_state_dict_ready}

    @overrides
    def on_load_checkpoint(
            self, trainer: pl.Trainer, pl_module: pl.LightningModule, callback_state: Dict[str, Any]
    ) -> None:
        self._ema_state_dict_ready = callback_state["_ema_state_dict_ready"]
        self.ema_state_dict = callback_state["ema_state_dict"]


class ValVisualization(pl.Callback):
    def __init__(self, params):
        """
        params: config, sde, shape (B, C, H, W), eps (in consistency with get_sampling_fn)
        """
        super(ValVisualization, self).__init__()
        self.params = params
        self.params["shape"] = self._collate_shape(self.params["shape"])
        self.epoch_cnt = 0

    @torch.no_grad()
    def on_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        inverse_scaler = get_data_inverse_scaler(pl_module.params["if_centering"])
        sampler_fn = get_sampling_fn(**self.params, inverse_scaler=inverse_scaler)
        X_sample, n = sampler_fn(pl_module)  # X_sample: (1, C, H, W)
        pl_module.logger.experiment.add_image("val_sample", ptu.to_numpy(X_sample), self.epoch_cnt)
        self.epoch_cnt += 1

    def _collate_shape(self, shape):
        # shape: (B, C, H, W): take the first sample
        if len(shape) == 4:
            shape = (1, *shape[1:])

        # TODO: collate shape of higher-dim tensors: all to (1, C, H, W)

        # (1, C, H, W)
        return shape


class ValVisualizationDiscrete(pl.Callback):
    def __init__(self, num_samples=1):
        self.num_samples = num_samples
        self.num_epochs = 0

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        B = self.num_samples  # sample only one image
        x_mod = torch.rand(B, pl_module.config.data.channels, pl_module.config.data.image_size,
                           pl_module.config.data.image_size, device=pl_module.config.device)
        x_mod = data_transform(pl_module.config, x_mod)
        images = anneal_Langevin_dynamics(x_mod,
                                          pl_module.model,
                                          pl_module.sigmas,
                                          n_steps_each=pl_module.config.sampling.n_steps_each,
                                          step_lr=pl_module.config.sampling.step_lr,
                                          final_only=pl_module.config.sampling.final_only,
                                          denoise=pl_module.config.sampling.denoise)[0]  # (B, C, H, W)
        # images = x_mod[0]
        images = images.detach().cpu()
        image_grid = make_grid(images, nrow=B)
        trainer.logger.experiment.add_image("val_sample", image_grid, self.num_epochs)
        self.num_epochs += 1
