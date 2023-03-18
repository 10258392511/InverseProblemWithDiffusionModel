import abc
import numpy as np
import torch
import torch.nn.functional as F
import os
import InverseProblemWithDiffusionModel.helpers.pytorch_utils as ptu

from . import freeze_model, compute_clf_grad, compute_seg_grad
from .proximal_op import Proximal, L2Penalty, Constrained, SingleCoil
from InverseProblemWithDiffusionModel.helpers.utils import (
    data_transform, 
    vis_images, 
    normalize, 
    denormalize,
    reshape_temporal_dim,
    vis_multi_channel_signal
)
from InverseProblemWithDiffusionModel.ncsn.linear_transforms import i2k_complex, k2i_complex
from InverseProblemWithDiffusionModel.ncsn.linear_transforms.finite_diff import FiniteDiff
from einops import rearrange


def get_lh_weights(sigmas, start_time, curve_type="linear"):
    assert 0 <= start_time <= 1
    lh_weights = torch.zeros_like(sigmas)

    if start_time == 1:
        return lh_weights

    start_idx = int(len(sigmas) * start_time)

    if curve_type == "linear":
        lh_weights[start_idx:] = torch.linspace(0, 1, len(sigmas) - start_idx, device=sigmas.device)

        return lh_weights

    else:
        raise NotImplementedError


def round_sign(X: torch.Tensor):
    # angle(X): [-pi, pi]
    X_angle = torch.angle(X)
    sign_out = (torch.abs(X_angle) >= torch.pi / 2).float() * 2 - 1

    return sign_out


class ALDOptimizer(abc.ABC):
    def __init__(self, x_mod_shape, scorenet, sigmas, params, config,
                 measurement=None, linear_tfm=None, clf=None, seg=None, device=None):
        """
        params: n_steps_each, step_lr, denoise, final_only
        """
        self.x_mod_shape = x_mod_shape  # (B, C, H, W), C = 1 or 2 which is set outside the scope
        self.scorenet = scorenet
        self.sigmas = sigmas
        self.params = params
        self.config = config
        self.measurement = measurement
        self.linear_tfm = linear_tfm
        self.clf = clf
        self.seg = seg
        self.device = device if device is not None else ptu.DEVICE

    def __call__(self, **kwargs):
        """
        kwargs:
            ALDInvClf: lamda
            ALDInvSeg: lamda
        """
        torch.set_grad_enabled(False)
        scorenet = self.scorenet
        sigmas = self.sigmas
        n_steps_each = self.params["n_steps_each"]
        step_lr = self.params["step_lr"]
        denoise = self.params["denoise"]
        final_only = self.params["final_only"]

        print("sampling...")
        # x_mod = torch.rand(*self.x_mod_shape).to(self.device)
        ### inserting pt ###
        x_mod = self.init_x_mod()
        ####################

        x_mod = data_transform(self.config, x_mod)

        images = []
        print_interval = len(sigmas) // 10

        ### inserting pt ###
        self.preprocessing_steps(**kwargs)
        ####################

        for c, sigma in enumerate(sigmas):
            if c % print_interval == 0:
                print(f"{c + 1}/{len(sigmas)}")

            labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c  # (B,)
            labels = labels.long()
            step_size = step_lr * (sigma / sigmas[-1]) ** 2  # (L_noise,)

            ### inserting pt ###
            x_mod = self.init_estimation(x_mod, alpha=step_size, **kwargs)
            ####################

            for s in range(n_steps_each):
                grad = scorenet(x_mod, labels)  # (B, C, H, W)

                ### inserting pt ###
                grad = self.adjust_grad(grad, x_mod, sigma=sigma, **kwargs)
                ####################

                noise = torch.randn_like(x_mod)
                grad_norm = torch.norm(grad.view(grad.shape[0], -1), dim=-1).mean()
                noise_norm = torch.norm(noise.view(noise.shape[0], -1), dim=-1).mean()
                x_mod = x_mod + step_size * grad + noise * torch.sqrt(step_size * 2)

                print(f"x_mod: {(x_mod.max(), x_mod.min())}")

                image_norm = torch.norm(x_mod.view(x_mod.shape[0], -1), dim=-1).mean()
                snr = torch.sqrt(step_size / 2.) * grad_norm / noise_norm
                grad_mean_norm = torch.norm(grad.mean(dim=0).view(-1)) ** 2 * sigma ** 2

                if not final_only:
                    images.append(x_mod.to('cpu'))

        if denoise:
            last_noise = (len(sigmas) - 1) * torch.ones(x_mod.shape[0], device=x_mod.device)
            last_noise = last_noise.long()
            x_mod = x_mod + sigmas[-1] ** 2 * scorenet(x_mod, last_noise)
            images.append(x_mod.to('cpu'))

        if final_only:
            return [x_mod.to('cpu')]
        else:
            return images

    def preprocessing_steps(self, **kwargs):
        # e.g. freeze_model(clf)
        pass

    def init_x_mod(self):
        # x_mod = 2 * torch.rand(*self.x_mod_shape).to(self.device) - 1
        x_mod = torch.rand(*self.x_mod_shape).to(self.device)

        return x_mod

    def init_estimation(self, x_mod, **kwargs):
        # e.g. projection
        return x_mod

    def adjust_grad(self, grad, x_mod, **kwargs):
        # add log-lh
        return grad


class ALDUnconditionalSampler(ALDOptimizer):
    pass

class ALDInvSegProximalRealImag(ALDOptimizer):
    def __init__(self, proximal: Proximal, seg_start_time, seg_step_type, *args, **kwargs):
        super(ALDInvSegProximalRealImag, self).__init__(*args, **kwargs)
        self.proximal = proximal
        self.seg_start_time = seg_start_time
        self.seg_step_type = seg_step_type
        # print(f"seg_start_time: {self.seg_start_time}")
        self.lh_weights = get_lh_weights(self.sigmas, self.seg_start_time, self.seg_step_type)  # (L,)
        self.if_print = False
        self.print_args = {}

    def __call__(self, **kwargs):
        """
        kwargs: label, lamda, save_dir, lr_scaled
        """
        torch.set_grad_enabled(False)
        scorenet = self.scorenet
        sigmas = self.sigmas
        n_steps_each = self.params["n_steps_each"]
        step_lr = self.params["step_lr"]
        denoise = self.params["denoise"]
        final_only = self.params["final_only"]

        print("sampling...")
        ### inserting pt ###
        # m_mod = self.init_x_mod()
        ####################
        # m_mod = data_transform(self.config, m_mod)

        x_mod = self.linear_tfm.conj_op(self.measurement)
        x_mod_real, x_mod_imag = torch.real(x_mod), torch.imag(x_mod)

        # images = []
        print_interval = len(sigmas) // 10

        ### inserting pt ###
        self.preprocessing_steps(**kwargs)
        ####################

        for c, sigma in enumerate(sigmas):
            self.if_print = False

            if c % print_interval == 0:
                self.if_print = True
                self.print_args = {
                    "c": c,
                    "save_dir": os.path.join(kwargs.get("save_dir"), "phase_images/")
                }

                print(f"{c + 1}/{len(sigmas)}")
                vis_images(x_mod_real[0], x_mod_imag[0], if_save=True,
                           save_dir=os.path.join(kwargs.get("save_dir"), "sampling_snapshots/"),
                           filename=f"step_{c}_start_time_{self.seg_start_time}_acdc.png")

            labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c  # (B,)
            labels = labels.long()
            step_size = step_lr * (sigma / sigmas[-1]) ** 2  # (L_noise,)

            # m_mod = self.init_estimation(x_mod, **kwargs)
            lh_seg_weight = self.lh_weights[c]

            ### inserting pt ###
            # x_mod = self.init_estimation(x_mod)
            ####################

            for s in range(n_steps_each):
                grad_prior_real = scorenet(x_mod_real, labels)  # (B, C, H, W)
                grad_prior_imag = scorenet(x_mod_imag, labels)

                ### inserting pt ###
                # TODO: consider whether using seg-net on real & imag; or mag only
                grad_real = self.adjust_grad(grad_prior_real, x_mod_real, sigma=sigma, seg_lamda=lh_seg_weight, **kwargs)
                grad_imag = self.adjust_grad(grad_prior_imag, x_mod_imag, sigma=sigma, seg_lamda=lh_seg_weight, **kwargs)
                # grad_real = grad_prior_real
                # grad_imag = grad_prior_imag
                ####################

                noise_real = torch.randn_like(x_mod_real)
                x_mod_real = x_mod_real + step_size * grad_real + noise_real * torch.sqrt(step_size * 2)
                noise_imag = torch.randn_like(x_mod_imag)
                x_mod_imag = x_mod_imag + step_size * grad_imag + noise_imag * torch.sqrt(step_size * 2)

                print(f"x_mod_real, {s + 1}/{n_steps_each}: {(x_mod_real.max(), x_mod_real.min())}")  ###
                print(f"x_mod_imag, {s + 1}/{n_steps_each}: {(x_mod_imag.max(), x_mod_imag.min())}")

                ### inserting pt ###
                x_mod_real, x_mod_imag = self.post_processing(x_mod_real, x_mod_imag, alpha=step_lr, sigma=sigma, **kwargs)
                print("after prox:")
                print(f"x_mod_real, {s + 1}/{n_steps_each}: {(x_mod_real.max(), x_mod_real.min())}")  ###
                print(f"x_mod_imag, {s + 1}/{n_steps_each}: {(x_mod_imag.max(), x_mod_imag.min())}")
                ####################

        if denoise:
            last_noise = (len(sigmas) - 1) * torch.ones(x_mod.shape[0], device=x_mod.device)
            last_noise = last_noise.long()
            x_mod_real = x_mod_real + sigmas[-1] ** 2 * scorenet(x_mod_real, last_noise)
            x_mod_imag = x_mod_imag + sigmas[-1] ** 2 * scorenet(x_mod_imag, last_noise)

        ### inserting pt ###
        # x_mod_real = denormalize(x_mod_real, low_q_real, high_q_real)
        # x_mod_imag = denormalize(x_mod_imag, low_q_imag, high_q_imag)
        x_mod = x_mod_real + 1j * x_mod_imag
        # torch.save(x_mod.detach().cpu(), os.path.join(kwargs.get("save_dir"), "before_last_prox.pt"))
        ####################

        if final_only:
            return [x_mod.to('cpu')]
        else:
            # return images
            return [x_mod.to('cpu')]
    
    def adjust_grad(self, grad, m_mod, **kwargs):
        """
        kwargs: label, seg_lamda, sigma, seg_mode
        """
        # m_mod: (B, C, H, W)
        label = kwargs["label"]
        lamda = kwargs["seg_lamda"]
        sigma = kwargs["sigma"]
        seg_mode = kwargs["seg_mode"]

        grad_log_lh_seg = compute_seg_grad(self.seg, m_mod, label, seg_mode)  # (B, C, H, W)
        grad = grad + grad_log_lh_seg / sigma * lamda
        # grad = grad + grad_log_lh_seg  * lamda

        return grad

    def post_processing(self, x_mod_real, x_mod_imag, **kwargs):
        """
        kwargs:
            sigma: noise std
            L2Penalty, SingleCoil:
                alpha: step-size for the ALD step, unscaled (i.e lr)
                lamda: hyper-param, for ||Ax - y||^2 / (lamda^2 + sigma^2) * lr_scaled
                       i.e lamda = sigma_data^2
                lr_scaled: empirical scaling
        """
        sigma = kwargs["sigma"]
        measurement = self.measurement
        print(f"sigma: {sigma}")
        
        x_mod = x_mod_real + 1j * x_mod_imag

        alpha = kwargs["alpha"]
        # lamda = kwargs["lamda"]  # not used (sigma_{data} ^ 2)
        lr_scaled = kwargs["lr_scaled"]
        if self.if_print:
            vis_images(torch.abs(x_mod[0]), torch.angle(x_mod[0]), if_save=True,
                       save_dir=self.print_args["save_dir"], filename=f"step_{self.print_args['c']}_before.png")
            mag_before = torch.abs(x_mod[0])
            phase_before = torch.angle(x_mod[0])

        coeff = alpha * lr_scaled
        print(f"coeff: {coeff}")
        x_mod = self.proximal(x_mod, measurement, coeff, 1.)

        if self.if_print:
            vis_images(torch.abs(x_mod[0]), torch.angle(x_mod[0]), if_save=True,
                       save_dir=self.print_args["save_dir"], filename=f"step_{self.print_args['c']}_after.png")
            mag_diff = torch.abs(x_mod[0]) - mag_before
            phase_diff = torch.angle(x_mod[0]) - phase_before
            vis_images(mag_diff, phase_diff, if_save=True,
                       save_dir=self.print_args["save_dir"], filename=f"step_{self.print_args['c']}_diff.png")

        x_mod_real, x_mod_imag = torch.real(x_mod), torch.imag(x_mod)

        return x_mod_real, x_mod_imag


class ALD2DTime(ALDOptimizer):
    def __init__(self, proximal: Proximal, scorenet_T, sigmas_T, *args, **kwargs):
        """
        params: n_steps_each, step_lr

        x_mod_shape: (B, T, C, H, W)
        measurement: (num_sens, B, T, C, H, W)
        """
        super(ALD2DTime, self).__init__(*args, **kwargs)
        self.proximal = proximal
        self.scorenet_T = scorenet_T
        self.sigmas_T_orig = sigmas_T
        sigmas_T_interp_len = (self.sigmas <= sigmas_T[0]).sum()
        self.sigmas_T = torch.ones_like(self.sigmas) * (-1)
        self.sigmas_T[-sigmas_T_interp_len:] = F.interpolate(sigmas_T.view(1, 1, -1), sigmas_T_interp_len, mode="nearest").squeeze()  # (T', 1, 1) -> (T',)
        self.scorenet_T.sigmas = self.sigmas_T
        self.win_size = np.sqrt(self.scorenet_T.config.data.channels).astype(int)
        self.finite_diff = None
        self.if_print = False
        self.print_args = {}

    def __call__(self, **kwargs):
        """
        kwargs: save_dir, lr_scaled, mode_T: ["tv", "diffusion1d", "none", "diffusion1d-only", "tv-only"], lamda_T: weighting for temporal step, if_random_shift
        """
        mode_T = kwargs.get("mode_T")
        if_skip_spatial = False
        if mode_T in ["diffusion1d-only", "tv-only"]:
            self.sigmas_T = self.sigmas_T_orig
            self.scorenet_T.sigmas = self.sigmas_T_orig
            self.sigmas = self.sigmas_T_orig
            if_skip_spatial = True
        # print(f"scorenet_T: {self.scorenet_T.sigmas}, self.sigmas: {self.sigmas}")
        torch.set_grad_enabled(False)
        ### inserting pt ###
        self.preprocessing_steps(**kwargs)
        ####################

        ### inserting pt ###
        x_mod = self.init_x_mod()  # (B, T, C, H, W)
        ####################

        print_interval = len(self.sigmas) // 10

        for c in range(self.sigmas.shape[0]):
            print(f"current: {c + 1}/{self.sigmas.shape[0]}")
            print(f"sigma: {self.sigmas[c]}, sigma_T: {self.sigmas_T[c]}")
            self.if_print = False

            if c % print_interval == 0:
                self.if_print = True
                self.print_args = {
                    "c": c,
                    "save_dir": kwargs.get("save_dir")
                }
           
            for s in range(self.params["n_steps_each"]):
                # spatial_step
                x_mod = self.spatial_step(x_mod, c, if_skip_spatial)
                ###########################################
                x_mod_real, x_mod_imag = torch.real(x_mod), torch.imag(x_mod)
                print("after spatial: ")
                print(f"x_mod_real, {s + 1}/{self.params['n_steps_each']}: {(x_mod_real.max(), x_mod_real.min())}")  ###
                print(f"x_mod_imag, {s + 1}/{self.params['n_steps_each']}: {(x_mod_imag.max(), x_mod_imag.min())}")
                ###########################################
                
                # temporal_step
                mode_T = kwargs.get("mode_T", "diffusion1d")
                lamda_T = kwargs.get("lamda_T", 1.)
                if_random_shift = kwargs.get("if_random_shift", False)
                x_mod = self.temporal_step(x_mod, c, mode_T, lamda_T, if_random_shift)
                ###########################################
                x_mod_real, x_mod_imag = torch.real(x_mod), torch.imag(x_mod)
                print("after temporal: ")
                print(f"x_mod_real, {s + 1}/{self.params['n_steps_each']}: {(x_mod_real.max(), x_mod_real.min())}")  ###
                print(f"x_mod_imag, {s + 1}/{self.params['n_steps_each']}: {(x_mod_imag.max(), x_mod_imag.min())}")
                ###########################################

                # proximal
                lr_scaled = kwargs["lr_scaled"]
                x_mod = self.proximal_step(x_mod, self.params["step_lr"], lr_scaled)

            if self.if_print:
                self._screenshot(x_mod, self.print_args)

        # no need to do the last denoising step
        self._screenshot(x_mod, {"c": c, "save_dir": kwargs.get("save_dir")})

        return [x_mod.to("cpu")]

    def init_x_mod(self):
        num_sens, B, T, C, H, W = self.measurement.shape
        measurement = self.measurement.reshape(num_sens, -1, C, H, W)  # (nums_sens, BT, C, H, W)
        x_mod = self.linear_tfm.conj_op(measurement)  # (BT, C, H, W)
        x_mod = x_mod.reshape(B, T, C, H, W)

        return x_mod
    
    def spatial_step(self, x_mod, c, if_skip_spatial):
        # x_mod: (B, T, C, H, W), labels: (B,)
        if if_skip_spatial:
            return x_mod

        B, T, C, H, W = x_mod.shape
        x_mod = x_mod.reshape(-1, C, H, W)  # (BT, C, H, W)
        labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
        labels = labels.long()
        x_mod_real, x_mod_imag = torch.real(x_mod), torch.imag(x_mod)
        step_size = self.params["step_lr"] * (self.sigmas[c] / self.sigmas[-1]) ** 2

        grad_real = self.scorenet(x_mod_real, labels)  # (BT, C, H, W)
        grad_imag = self.scorenet(x_mod_imag, labels)
        noise_real = torch.randn_like(x_mod_real)
        noise_imag = torch.randn_like(x_mod_imag)
        x_mod_real = x_mod_real + step_size * grad_real + noise_real * torch.sqrt(step_size * 2)
        x_mod_imag = x_mod_imag + step_size * grad_imag + noise_imag * torch.sqrt(step_size * 2)

        x_mod = (x_mod_real + 1j * x_mod_imag).reshape(B, T, C, H, W)
        
        return x_mod

    # # for NCSN1D    
    # def temporal_step(self, x_mod, c, mode_T, lamda_T, if_random_shift):
    #     # x_mod: (B, T, C, H, W), labels: (B,)
    #     # TV_t
    #     if "tv" in mode_T:
    #         if self.finite_diff is None:
    #             self.finite_diff = FiniteDiff(dims=1)
    #         x_mod_real, x_mod_imag = torch.real(x_mod), torch.imag(x_mod)
    #         x_mod_real = x_mod_real + self.finite_diff.log_lh_grad(x_mod_real, lamda=lamda_T)
    #         x_mod_imag = x_mod_imag + self.finite_diff.log_lh_grad(x_mod_imag, lamda=lamda_T)

    #         x_mod = x_mod_real + 1j * x_mod_imag

    #     # scorenet_T
    #     elif "diffusion1d" in mode_T:
    #         if self.sigmas_T[c] == -1:
    #             return x_mod
    #         B, T, C, H, W = x_mod.shape
    #         x_mod = x_mod.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
    #         x_mod = x_mod.reshape(-1, T, H, W)  # (BC, T, H, W)
    #         if if_random_shift:
    #             shifts_np = np.random.randint(0, self.win_size, (2,))
    #             shifts = tuple(shifts_np.tolist())
    #             print(f"shifts: {shifts}")
    #             x_mod = torch.roll(x_mod, shifts=shifts, dims=(-2, -1))
    #         x_mod = reshape_temporal_dim(x_mod, self.win_size, self.win_size, "forward")  # (B', kx * ky, T)
    #         labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
    #         labels = labels.long()
    #         x_mod_real, x_mod_imag = torch.real(x_mod), torch.imag(x_mod)
    #         step_size = self.params["step_lr"] * (self.sigmas_T[c] / self.sigmas_T[-1]) ** 2

    #         grad_real = self.scorenet_T(x_mod_real, labels)  # (B', kx * ky, T)
    #         grad_imag = self.scorenet_T(x_mod_imag, labels)
            
    #         step_size = step_size * lamda_T
    #         noise_real = torch.randn_like(x_mod_real)
    #         noise_imag = torch.randn_like(x_mod_imag)

    #         x_mod_real = x_mod_real + step_size * grad_real + noise_real * torch.sqrt(step_size * 2)
    #         x_mod_imag = x_mod_imag + step_size * grad_imag + noise_imag * torch.sqrt(step_size * 2)

            
    #         x_mod = reshape_temporal_dim(x_mod_real + 1j * x_mod_imag, self.win_size, self.win_size, "backward", img_size=(H, W))  # (BC, T, H, W)
    #         if if_random_shift:
    #             shifts = -shifts_np
    #             shifts = tuple(shifts.tolist())
    #             print(f"shifts back: {shifts}")
    #             x_mod = torch.roll(x_mod, shifts=shifts, dims=(-2, -1))

    #         x_mod = x_mod.reshape(B, C, T, H, W).permute(0, 2, 1, 3, 4)  # (B, C, T, H, W) -> (B, T, C, H, W)
    
    #     return x_mod
    
    # for NCSN3D (conv)
    def temporal_step(self, x_mod, c, mode_T, lamda_T, if_random_shift):
        # x_mod: (B, T, C, H, W), labels: (B,)
        # TV_t
        if "tv" in mode_T:
            if self.finite_diff is None:
                self.finite_diff = FiniteDiff(dims=1)
            x_mod_real, x_mod_imag = torch.real(x_mod), torch.imag(x_mod)
            x_mod_real = x_mod_real + self.finite_diff.log_lh_grad(x_mod_real, lamda=lamda_T)
            x_mod_imag = x_mod_imag + self.finite_diff.log_lh_grad(x_mod_imag, lamda=lamda_T)

            x_mod = x_mod_real + 1j * x_mod_imag

        # scorenet_T
        elif "diffusion1d" in mode_T:
            if self.sigmas_T[c] == -1:
                return x_mod
            B, T, C, H, W = x_mod.shape
            x_mod = rearrange(x_mod, "B T C H W -> B C H W T")
            labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
            labels = labels.long()
            x_mod_real, x_mod_imag = torch.real(x_mod), torch.imag(x_mod)
            step_size = self.params["step_lr"] * (self.sigmas_T[c] / self.sigmas_T[-1]) ** 2

            grad_real = self.scorenet_T(x_mod_real, labels)  # (B', kx * ky, T)
            grad_imag = self.scorenet_T(x_mod_imag, labels)
            
            step_size = step_size * lamda_T
            noise_real = torch.randn_like(x_mod_real)
            noise_imag = torch.randn_like(x_mod_imag)

            x_mod_real = x_mod_real + step_size * grad_real + noise_real * torch.sqrt(step_size * 2)
            x_mod_imag = x_mod_imag + step_size * grad_imag + noise_imag * torch.sqrt(step_size * 2)
            x_mod = x_mod_real + 1j * x_mod_imag
            
            x_mod = rearrange(x_mod, "B C H W T -> B T C H W")
    
        return x_mod
    
    def proximal_step(self, x_mod, alpha, lr_scaled):
        # x_mod: (B, T, C, H, W)
        B, T, C, H, W = x_mod.shape
        x_mod = x_mod.reshape(-1, C, H, W)  # (BT, C, H, W)
        num_sens = self.measurement.shape[0]
        measurement = self.measurement.reshape(num_sens, -1, *self.measurement.shape[3:])  # (num_sens, BT, C, H, W)
        coeff = alpha * lr_scaled
        print(f"coeff: {coeff}")
        x_mod = self.proximal(x_mod, measurement, coeff, 1.)  # (BT, C, H, W)
        x_mod = x_mod.reshape(B, T, C, H, W)

        return x_mod

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
        