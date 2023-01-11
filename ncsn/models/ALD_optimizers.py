import abc
import torch
import os
import InverseProblemWithDiffusionModel.helpers.pytorch_utils as ptu

from . import freeze_model, compute_clf_grad, compute_seg_grad
from .proximal_op import Proximal, L2Penalty, Constrained, SingleCoil
from InverseProblemWithDiffusionModel.helpers.utils import data_transform, vis_images, normalize, denormalize
from InverseProblemWithDiffusionModel.ncsn.linear_transforms import i2k_complex, k2i_complex


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
                # grad_real = self.adjust_grad(grad_prior_real, x_mod_real, sigma=sigma, seg_lamda=lh_seg_weight, **kwargs)
                # grad_imag = self.adjust_grad(grad_prior_imag, x_mod_imag, sigma=sigma, seg_lamda=lh_seg_weight, **kwargs)
                grad_real = grad_prior_real
                grad_imag = grad_prior_imag
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
        torch.save(x_mod.detach().cpu(), os.path.join(kwargs.get("save_dir"), "before_last_prox.pt"))
        ####################

        if final_only:
            return [x_mod.to('cpu')]
        else:
            # return images
            return [x_mod.to('cpu')]
    
    def adjust_grad(self, grad, m_mod, **kwargs):
        """
        kwargs: label, seg_lamda, sigma
        """
        # m_mod: (B, C, H, W)
        label = kwargs["label"]
        lamda = kwargs["seg_lamda"]
        sigma = kwargs["sigma"]

        grad_log_lh_seg = compute_seg_grad(self.seg, m_mod, label)  # (B, C, H, W)
        # grad = grad + grad_log_lh_seg / sigma * lamda
        grad = grad + grad_log_lh_seg  * lamda

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
