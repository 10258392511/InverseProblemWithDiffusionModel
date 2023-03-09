import numpy as np
import torch
import torch.nn as nn
import einops
import abc
import os
import InverseProblemWithDiffusionModel.helpers.pytorch_utils as ptu

from torch.utils.tensorboard import SummaryWriter
# from monai.data import MetaTensor
from InverseProblemWithDiffusionModel.ncsn.linear_transforms import LinearTransform
from InverseProblemWithDiffusionModel.ncsn.regularizers import AbstractRegularizer
from InverseProblemWithDiffusionModel.ncsn.models import get_sigmas
from InverseProblemWithDiffusionModel.helpers.utils import (
    reshape_temporal_dim, 
    save_vol_as_gif, 
    normalize_phase,
    vis_images,
    vis_multi_channel_signal
)
from InverseProblemWithDiffusionModel.ncsn.linear_transforms.finite_diff import FiniteDiff
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
                 config, logger: SummaryWriter, device=None, opt_class=None, opt_params=None):
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
        # self.sigma_val = self.scorenet.sigmas[-1]
        # self.sigma = -1
        self.lr = self.config.MAP.lr
        if opt_class is None:
            opt_class = torch.optim.Adam
            opt_params = {"betas": (0.5, 0.5)}
        self.opt = opt_class([self.x_init], lr=self.lr, **opt_params)  # LBFGS is not supported
    
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
        # grad_prior = self.scorenet(x, self.sigma) * self.sigma_val  # (1, C, H, W)
        labels = torch.ones(x.shape[0], device=self.device).long()
        grad_prior_real = self.scorenet(torch.real(x), labels)
        grad_prior_imag = self.scorenet(torch.imag(x), labels)
        grad_prior = grad_prior_real + 1j * grad_prior_imag
        grad = grad_data + self.lamda * grad_prior
        # x += grad * self.lr  # maximizing log posterior
        self.opt.zero_grad()
        self.x_init.grad = -grad
        self.opt.step()

        # logging
        self.logger.add_scalar("grad_data", torch.norm(grad_data).item(), global_step=iter)
        self.logger.add_scalar("grad_prior", torch.norm(grad_prior).item(), global_step=iter)
        self.logger.add_scalar("grad", torch.norm(grad).item(), global_step=iter)

        return x


class Inpainting(MAPOptimizer):
    pass


class SENSEMAP(MAPOptimizer):
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


class MAPOptimizer2DTime(object):
    def __init__(self, x_init: torch.Tensor, measurement: torch.Tensor, scorenet_S: nn.Module, scorenet_T: nn.Module, linear_tfm: LinearTransform, logger: SummaryWriter, params: dict):
        """
        x_init: (B, T, C, H, W), complex, already sent to "device"
        measurement: (num_sens, B, T, C, H, W)

        params: lr, opt_class, (device), num_iters, num_plot_times, win_size, prior_weight, spatial_step_weight, temporal_step_weight, save_dir, opt_params: dict, mode_T: [diffusion1d, tv], if_random_shift
        """
        self.params = params
        # self.x = torch.zeros_like(x_init)
        self.x = x_init
        self.x_real = torch.real(self.x)
        self.x_imag = torch.imag(self.x)
        self.measurement = measurement
        self.scorenet_S = scorenet_S
        self.scorenet_T = scorenet_T
        self.win_size = np.sqrt(self.scorenet_T.config.data.channels).astype(int)
        self.linear_tfm = linear_tfm
        device = params.get("device", None)
        self.device = ptu.DEVICE if device is None else device
        self.logger = logger
        self.opt_real = self.params["opt_class"]([self.x_real], lr=self.params["lr"], **self.params.get("opt_params", {}))
        self.opt_imag = self.params["opt_class"]([self.x_imag], lr=self.params["lr"], **self.params.get("opt_params", {}))
        self.plot_interval = self.params["num_iters"] // self.params["num_plot_times"]
        self.grad_log = {"grad_data": None, "grad_S": None, "grad_T": None, "grad": None, "data_error": None}
        self.finite_diff = None
    
    @torch.no_grad()
    def __call__(self):
        pbar = trange(self.params["num_iters"], desc="optimizing")
        save_dir = os.path.join(self.params["save_dir"], "screenshots/")
        save_dir_2d = os.path.join(self.params["save_dir"], "screenshots_2d/")

        for iter in pbar:
            # all grad_*: (B, T, C, H, W) 
            @torch.no_grad()
            def closure_real():
                self.opt_real.zero_grad()
                grad_data, data_error = self.data_step()
                grad_S = self.spatial_step()
                grad_T = self.temporal_step(mode_T=self.params["mode_T"], if_random_shift=self.params["if_random_shift"])
                
                grad = grad_data + self.params["prior_weight"] * (self.params["spatial_step_weight"] * grad_S + self.params["temporal_step_weight"] * grad_T)
                self.x_real.grad = -torch.real(grad)

                self.grad_log["grad_data"] = grad_data
                self.grad_log["grad_S"] = grad_S
                self.grad_log["grad_T"] = grad_T
                self.grad_log["grad"] = grad
                self.grad_log["data_error"] = data_error

                return 0
            
            @torch.no_grad()
            def closure_imag():
                self.opt_imag.zero_grad()
                grad_data, data_error = self.data_step()
                grad_S = self.spatial_step()
                grad_T = self.temporal_step(mode_T=self.params["mode_T"], if_random_shift=self.params["if_random_shift"])
                
                grad = grad_data + self.params["prior_weight"] * (self.params["spatial_step_weight"] * grad_S + self.params["temporal_step_weight"] * grad_T)
                self.x_imag.grad = -torch.imag(grad)

                self.grad_log["grad_data"] = grad_data
                self.grad_log["grad_S"] = grad_S
                self.grad_log["grad_T"] = grad_T
                self.grad_log["grad"] = grad
                self.grad_log["data_error"] = data_error

                return 0
            
            self.opt_real.step(closure_real)
            self.opt_imag.step(closure_imag)
            self.x = self.x_real + 1j * self.x_imag

            # logging
            grad_data = self.grad_log["grad_data"]
            grad_S = self.grad_log["grad_S"]
            grad_T = self.grad_log["grad_T"]
            grad = self.grad_log["grad"]
            data_error = self.grad_log["data_error"]

            self.logger.add_scalar("data_error", data_error.item(), global_step=iter)
            self.logger.add_scalar("grad_data", torch.norm(grad_data).item(), global_step=iter)
            self.logger.add_scalar("grad_S", torch.norm(grad_S).item(), global_step=iter)
            self.logger.add_scalar("grad_T", torch.norm(grad_T).item(), global_step=iter)
            self.logger.add_scalar("grad", torch.norm(grad).item(), global_step=iter)
            pbar.set_description(f"data_error: {data_error.item()}")

            if iter % self.plot_interval == 0 or iter == self.params["num_iters"] - 1:
                # only save the first batch
                save_vol_as_gif(torch.abs(self.x[0]), save_dir=save_dir, filename=f"mag_{iter + 1}.gif")
                save_vol_as_gif(normalize_phase(torch.angle(self.x[0])), save_dir=save_dir, filename=f"phase_{iter + 1}.gif")
                self._screenshot(self.x, {"c": iter, "save_dir": save_dir_2d})

        return self.get_reconstruction()

    def data_step(self):
        B, T, C, H, W = self.x.shape
        x = einops.rearrange(self.x, "B T C H W -> (B T) C H W")  # (BT, C, H, W)
        measurement = einops.rearrange(self.measurement, "num_sens B T C H W -> num_sens (B T) C H W")  # (num_sens, BT, C, H, W)
        grad = self.linear_tfm.log_lh_grad(x, measurement)  # (BT, C, H, W)
        grad = einops.rearrange(grad, "(B T) C H W -> B T C H W", T=T)
        data_error = 0.5 * ((torch.abs(self.linear_tfm(x) - measurement) ** 2).sum(dim=(1, 2, 3)).mean())

        return grad, data_error

    def spatial_step(self):
        B, T, C, H, W = self.x.shape
        x = einops.rearrange(self.x, "B T C H W -> (B T) C H W")  # (BT, C, H, W)
        labels = torch.ones(x.shape[0], device=self.device).long()
        x_real, x_imag = torch.real(x), torch.imag(x)
        grad_real = self.scorenet_S(x_real, labels)
        grad_imag = self.scorenet_S(x_imag, labels)
        grad = grad_real + 1j * grad_imag
        grad = einops.rearrange(grad, "(B T) C H W -> B T C H W", T=T)

        return grad

    # for NCSN1D
    def temporal_step(self, mode_T="diffusion1d", if_random_shift=False):
        # self.x: (B, T, C, H, W)
        if mode_T == "tv":
            if self.finite_diff is None:
                self.finite_diff = FiniteDiff(dims=1)
            x_real, x_imag = torch.real(self.x), torch.imag(self.x)
            grad_real = self.finite_diff.log_lh_grad(x_real)
            grad_imag = self.finite_diff.log_lh_grad(x_imag)
            grad = grad_real + 1j * grad_imag

        elif mode_T == "diffusion1d":
            B, T, C, H, W = self.x.shape
            x = einops.rearrange(self.x, "B T C H W -> (B C) T H W")  # (BC, T, H, W)
            if if_random_shift:
                shifts_np = np.random.randint(0, self.win_size, (2,))
                shifts = tuple(shifts_np.tolist())
                # print(f"shifts: {shifts}")
                x = torch.roll(x, shifts=shifts, dims=(-2, -1))
            x = reshape_temporal_dim(x, self.params["win_size"], self.params["win_size"], "forward")  # (B', kx * ky, T)
            labels = torch.ones(x.shape[0], device=self.device).long()
            x_real, x_imag = torch.real(x), torch.imag(x)
            grad_real = self.scorenet_T(x_real, labels)
            grad_imag = self.scorenet_T(x_imag, labels)
            grad = grad_real + 1j * grad_imag
            grad = reshape_temporal_dim(grad, self.params["win_size"], self.params["win_size"], "backward", img_size=(H, W))  # (BC, T, H, W)
            # print(f"grad_T shape: {grad.shape}")
            # print("-" * 100)
            if if_random_shift:
                shifts = -shifts_np
                shifts = tuple(shifts.tolist())
                # print(f"shifts back: {shifts}")
                grad = torch.roll(grad, shifts=shifts, dims=(-2, -1))
            grad = einops.rearrange(grad, "(B C) T H W -> B T C H W", C=C)

        return grad

    # # for NCSN3D (conv)
    # def temporal_step(self, mode_T="diffusion1d", if_random_shift=False):
    #     # self.x: (B, T, C, H, W)
    #     if mode_T == "tv":
    #         if self.finite_diff is None:
    #             self.finite_diff = FiniteDiff(dims=1)
    #         x_real, x_imag = torch.real(self.x), torch.imag(self.x)
    #         grad_real = self.finite_diff.log_lh_grad(x_real)
    #         grad_imag = self.finite_diff.log_lh_grad(x_imag)
    #         grad = grad_real + 1j * grad_imag

    #     elif mode_T == "diffusion1d":
    #         B, T, C, H, W = self.x.shape
    #         x = einops.rearrange(self.x, "B T C H W -> B C H W T")
    #         labels = torch.ones(x.shape[0], device=self.device).long()
    #         x_real, x_imag = torch.real(x), torch.imag(x)
    #         grad_real = self.scorenet_T(x_real, labels)
    #         grad_imag = self.scorenet_T(x_imag, labels)
    #         grad = grad_real + 1j * grad_imag
    #         # print(f"grad_T shape: {grad.shape}")
    #         # print("-" * 100)
    #         grad = einops.rearrange(grad, "B C H W T -> B T C H W")

    #     return grad

    def get_reconstruction(self):

        return self.x.detach().cpu()
    
    def _screenshot(self, x_mod, print_args: dict):
        """
        x_mod: (B, T, C, H, W)
        print_args: keys: c, save_dir

        Screenshots: first batch: first image; upper-left corner and center temporal slices (first image channel);
            all in magnitude-phase format
        """
        B, T, C, H, W = x_mod.shape
        save_dir = print_args.get("init_vis_dir", None)
        if save_dir is None:
            save_dir = os.path.join(print_args["save_dir"], "screenshots/")
        vis_images(torch.abs(x_mod[0, 0]), torch.angle(x_mod[0, 0]), if_save=True, save_dir=save_dir, 
                   filename=f"step_{print_args['c']}_spatial.png")
        h_center, w_center = H // 2, W // 2
        x_mod_temporal_slice = x_mod[0, :, 0, ...].unsqueeze(0)  # (T, H, W) -> (1, T, H, W)
        upper_left_corner = x_mod_temporal_slice[:, :, 0:self.win_size, 0:self.win_size]  # (1, T, kx, ky)
        center = x_mod_temporal_slice[:, :, h_center:h_center + self.win_size, w_center:w_center + self.win_size]  # (1, T, kx, ky)
        upper_left_corner = reshape_temporal_dim(upper_left_corner, self.win_size, self.win_size)  # (1, kx * ky, T)
        center = reshape_temporal_dim(center, self.win_size, self.win_size)  # (1, kx * ky, T)
        
        num_first_channels = 4
        vis_multi_channel_signal(torch.real(upper_left_corner[0]), num_channels=num_first_channels, if_save=True, save_dir=save_dir, filename=f"step_{print_args['c']}_T_upper_left_corner_real.png")
        vis_multi_channel_signal(torch.imag(upper_left_corner[0]), num_channels=num_first_channels, if_save=True, save_dir=save_dir, filename=f"step_{print_args['c']}_T_upper_left_corner_imag.png")
        vis_multi_channel_signal(torch.real(center[0]), num_channels=num_first_channels, if_save=True, save_dir=save_dir, filename=f"step_{print_args['c']}_T_center_real.png")
        vis_multi_channel_signal(torch.imag(center[0]), num_channels=num_first_channels, if_save=True, save_dir=save_dir, filename=f"step_{print_args['c']}_T_center_imag.png")
        