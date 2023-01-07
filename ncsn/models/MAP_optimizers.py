import torch
import torch.nn as nn
import abc
import InverseProblemWithDiffusionModel.helpers.pytorch_utils as ptu

from torch.utils.tensorboard import SummaryWriter
# from monai.data import MetaTensor
from InverseProblemWithDiffusionModel.ncsn.linear_transforms import LinearTransform
from InverseProblemWithDiffusionModel.ncsn.regularizers import AbstractRegularizer
from InverseProblemWithDiffusionModel.ncsn.models import get_sigmas
from tqdm import trange
from typing import Any, Union


class MAPModel(nn.Module):
    def __init__(self, S: Union[torch.Tensor, Any], lin_tfm: LinearTransform, reg: Union[Any, AbstractRegularizer],
                 reg_weight: float):
        """
        S: measurement, (B, C', H', W')
        """
        super(MAPModel, self).__init__()
        self.S = S.to(ptu.DEVICE)
        self.lin_tfm = lin_tfm
        X_conj = self.lin_tfm.conj_op(self.S)
        # X_conj = torch.zeros_like(X_conj)
        self.X = nn.Parameter(X_conj, requires_grad=True)
        self.reg = reg
        self.reg_weight = reg_weight

    def forward(self, *args, **kwargs):
        # X: img; S: measurement
        AX = self.lin_tfm(self.X)  # (B, C', H', W')
        data_loss = (torch.abs(AX - self.S) ** 2).sum() / 2
        reg_loss = self.reg(self.X, *args, **kwargs).squeeze()
        loss = data_loss + self.reg_weight * reg_loss

        return data_loss, reg_loss, loss

    def get_reconstruction(self):

        return self.X.detach().cpu()
    

class MAPOptimizer(object):
    def __init__(self, x_init: torch.Tensor, measurement: torch.Tensor, scorenet, linear_tfm: LinearTransform, lamda,
                 config, logger: SummaryWriter, device=None):
        """
        x_init: (1, C, H, W)
        logger: initialized outside the scope
                tags: data_error, grad_data, grad_prior, grad_norm, recons_img
        lamda: regularization strength
        """
        self.x_init = x_init
        self.measurement = measurement
        self.scorenet = scorenet
        self.linear_tfm = linear_tfm
        self.lamda = lamda
        self.config = config
        self.device = ptu.DEVICE if device is None else device
        self.logger = logger
        self.plot_interval = self.config.MAP.n_iters // 50
        self.sigma_val = self.scorenet.sigmas[-1]
        self.sigma = -1
        self.lr = self.config.MAP.lr
    
    @torch.no_grad()
    def __call__(self):
        n_iters = self.config.MAP.n_iters
        pbar = trange(n_iters, desc="optimizing")
        x = self.x_init
        for iter in pbar:
            x = self._step(x, iter)
            data_error = 0.5 * (torch.norm(self.linear_tfm(x) - self.measurement) ** 2)
            self.logger.add_scalar("data_error", data_error.item(), global_step=iter)
            pbar.set_description(desc=f"data error: {data_error.item()}")

            if iter % self.plot_interval == 0 or iter == n_iters - 1:
                # x: (1, 1, H, W)
                self.logger.add_image("recons_img", x.detach().cpu()[0], global_step=iter, dataformats="CHW")

        return x

    def _step(self, x, iter):
        grad_data = self.linear_tfm.log_lh_grad(x, self.measurement, 1.)  # (1, C, H, W)
        grad_prior = self.scorenet(x, self.sigma) * self.sigma_val  # (1, C, H, W)
        # grad_prior = self.scorenet(x, self.sigma)  # (1, C, H, W)
        grad = grad_data + self.lamda * grad_prior
        x += grad * self.lr  # maximizing log posterior

        # logging
        self.logger.add_scalar("grad_data", torch.norm(grad_data).item(), global_step=iter)
        self.logger.add_scalar("grad_prior", torch.norm(grad_prior).item(), global_step=iter)
        self.logger.add_scalar("grad", torch.norm(grad).item(), global_step=iter)

        return x


class Inpainting(MAPOptimizer):
    pass


class UndersamplingFourier(MAPOptimizer):
    """
    x: real image
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x_complex = self.linear_tfm.conj_op(self.measurement)
        self.x_complex = torch.abs(self.x_init) * torch.sgn(self.x_complex)
        # self.x_init = torch.abs(self.x_complex)
        # self.x_complex = self.x_init * torch.exp(1j * (torch.rand_like(self.x_init).to(self.x_init.device) * 2 - 1) * torch.pi)

    def _step(self, x, iter):
        grad_prior = self.scorenet(x, self.sigma) * self.sigma_val
        x += self.lamda * grad_prior * self.lr
        self.x_complex = torch.maximum(x, torch.tensor(0).to(x.device)) * torch.sgn(self.x_complex)  # (1, C, H, W)
        for _ in range(self.config.MAP.complex_inner_n_steps):
            grad_data = self.linear_tfm.log_lh_grad(self.x_complex, self.measurement, 1.)
            self.x_complex += grad_data * self.lr
        x = torch.abs(self.x_complex)

        # logging
        self.logger.add_scalar("grad_data", torch.norm(grad_data).item(), global_step=iter)
        self.logger.add_scalar("grad_prior", torch.norm(grad_prior).item(), global_step=iter)

        return x
