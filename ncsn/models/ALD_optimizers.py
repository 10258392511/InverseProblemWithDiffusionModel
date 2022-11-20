import abc
import torch
import InverseProblemWithDiffusionModel.helpers.pytorch_utils as ptu

from . import freeze_model, compute_clf_grad, compute_seg_grad
from InverseProblemWithDiffusionModel.helpers.utils import data_transform, vis_images


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
        x_mod = self.init_x_mod()
        x_mod = data_transform(self.config, x_mod)

        images = []
        print_interval = len(sigmas) // 10

        self.preprocessing_steps(**kwargs)

        for c, sigma in enumerate(sigmas):
            if c % print_interval == 0:
                print(f"{c + 1}/{len(sigmas)}")

            labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c  # (B,)
            labels = labels.long()
            step_size = step_lr * (sigma / sigmas[-1]) ** 2  # (L_noise,)

            x_mod = self.init_estimation(x_mod, **kwargs)

            for s in range(n_steps_each):
                grad = scorenet(x_mod, labels)  # (B, C, H, W)

                grad = self.adjust_grad(grad, x_mod, sigma=sigma, **kwargs)

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


class ALDInvClf(ALDOptimizer):
    def preprocessing_steps(self, **kwargs):
        freeze_model(self.clf)

    def adjust_grad(self, grad, x_mod, **kwargs):
        """
        kwargs: cls, lamda (on clf)
        """
        cls = kwargs["cls"]
        lamda = kwargs["lamda"]
        sigma = kwargs["sigma"]
        grad_norm = torch.sqrt((grad ** 2).sum(dim=(1, 2, 3), keepdim=True))  # (B, 1, 1, 1)
        # grad_norm = torch.norm(grad)
        grad_log_lh_clf = compute_clf_grad(self.clf, x_mod, cls=kwargs["cls"])
        # (B, C, H, W)
        grad_log_lh_measurement = self.linear_tfm.log_lh_grad(x_mod, self.measurement, 1.)
        # grad_log_lh_measurement_norm = torch.sqrt((grad_log_lh_measurement ** 2).sum(dim=(1, 2, 3), keepdim=True))  # (B, 1, 1, 1)
        # grad_log_lh_measurement_norm = torch.norm(grad_log_lh_measurement)
        # grad_log_lh_measurement = grad_log_lh_measurement / (grad_log_lh_measurement_norm) * grad_norm
        grad += (grad_log_lh_clf * lamda + grad_log_lh_measurement * (1 - lamda)) / sigma

        return grad


class ALDInvSeg(ALDOptimizer):
    def __init__(self, seg_start_time, seg_step_type="linear", **kwargs):
        """
        seg_start_time: in [0, 1]
        """
        super(ALDInvSeg, self).__init__(**kwargs)
        self.seg_start_time = seg_start_time
        self.seg_step_type = seg_step_type
        self.lh_weights = get_lh_weights(self.sigmas, self.seg_start_time, self.seg_step_type)
        self.last_m_mod_sign = None

    def __call__(self, **kwargs):
        """
        kwargs: label, save_dir, (lamda (lh_weight)),
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
        m_mod = self.init_x_mod()
        m_mod = data_transform(self.config, m_mod)
        x_mod = m_mod * torch.exp(1j * (torch.rand(m_mod.shape, device=m_mod.device) * 2 - 1) * torch.pi)

        images = []
        print_interval = len(sigmas) // 10

        self.preprocessing_steps(**kwargs)

        for c, sigma in enumerate(sigmas):
            if c % print_interval == 0:
                print(f"{c + 1}/{len(sigmas)}")

                # vis_images(torch.abs(x_mod[0]), if_save=True, save_dir=kwargs.get("save_dir"),
                #            filename=f"step_{c}_start_time_{self.seg_start_time}_acdc.png")
                vis_images(m_mod[0], if_save=True, save_dir=kwargs.get("save_dir"),
                           filename=f"step_{c}_start_time_{self.seg_start_time}_acdc.png")

            labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c  # (B,)
            labels = labels.long()
            step_size = step_lr * (sigma / sigmas[-1]) ** 2  # (L_noise,)

            # m_mod = self.init_estimation(x_mod, **kwargs)
            lh_seg_weight = self.lh_weights[c]

            for s in range(n_steps_each):
                grad_prior = scorenet(m_mod, labels)  # (B, C, H, W)

                grad = self.adjust_grad(grad_prior, m_mod, sigma=sigma, lamda=lh_seg_weight, **kwargs)

                noise = torch.randn_like(m_mod)
                grad_norm = torch.norm(grad.view(grad.shape[0], -1), dim=-1).mean()
                noise_norm = torch.norm(noise.view(noise.shape[0], -1), dim=-1).mean()

                m_mod = m_mod + step_size * grad + noise * torch.sqrt(step_size * 2)

                image_norm = torch.norm(x_mod.view(x_mod.shape[0], -1), dim=-1).mean()
                snr = torch.sqrt(step_size / 2.) * grad_norm / noise_norm
                grad_mean_norm = torch.norm(grad.mean(dim=0).view(-1)) ** 2 * sigma ** 2

                x_mod, m_mod = self.mag2complex(x_mod, m_mod, grad_prior, sigma, step_size)
                print(f"grad_prior: {torch.norm(grad_prior)}")  ###
                print(f"m_mod: {(m_mod.max(), m_mod.min())}")  ###

                if not final_only:
                    images.append(m_mod.to('cpu'))

        if denoise:
            last_noise = (len(sigmas) - 1) * torch.ones(x_mod.shape[0], device=x_mod.device)
            last_noise = last_noise.long()
            m_mod = m_mod + sigmas[-1] ** 2 * scorenet(m_mod, last_noise)
            # x_mod = m_mod * torch.exp(1j * torch.angle(x_mod))
            images.append(m_mod.to('cpu'))

        if final_only:
            return [m_mod.to('cpu')]
        else:
            return images

    def preprocessing_steps(self, **kwargs):
        freeze_model(self.seg)

    def init_x_mod(self):
        x_mod = torch.rand(*self.x_mod_shape).to(self.device)
        return x_mod

    def init_estimation(self, x_mod, **kwargs):
        m_mod = torch.abs(x_mod)

        return m_mod

    def adjust_grad(self, grad, m_mod, **kwargs):
        """
        kwargs: label, lamda
        """
        label = kwargs["label"]
        lh_weight = kwargs["lamda"]
        sigma = kwargs["sigma"]
        grad_log_lh_seg = compute_seg_grad(self.seg, m_mod, label=label)
        # grad_norm = torch.sqrt((torch.abs(grad) ** 2).sum(dim=(1, 2, 3), keepdim=True))  # (B, 1, 1, 1)
        # grad_log_lh_seg_norm = torch.sqrt((torch.abs(grad_log_lh_seg) ** 2).sum(dim=(1, 2, 3), keepdim=True))
        # grad_log_lh_seg = grad_log_lh_seg / grad_log_lh_seg_norm * grad_norm
        print(f"grad_log_lh_seg: {torch.norm(grad_log_lh_seg)}")  ###
        grad += grad_log_lh_seg * lh_weight / sigma

        return grad

    def mag2complex(self, x_mod, m_mod, grad, sigma, step_size):
        """
        (1). x_mod <- m_mod * angle(x_mod)
        (2). Inv log-lh step: update x_mod
        (3). Update m_mod
        """
        # # x_mod = m_mod * torch.exp(1j * torch.angle(x_mod))
        # x_mod_angle_in = round_sign(x_mod)
        # if self.last_m_mod_sign is None:
        #     # m_sign_flip_mask = torch.ones(m_mod.shape).to(m_mod.device)
        #     self.last_m_mod_sign = torch.ones(m_mod.shape).to(m_mod.device)
        # # else:
        # #     m_mod_sign = torch.sign(m_mod)
        # #     # m_sign_flip_mask = m_mod_sign * self.last_m_mod_sign
        # #     self.last_m_mod_sign = m_mod_sign
        #
        # # pass sign flip from m_mod to x_mod
        # # x_mod = m_mod * self.last_m_mod_sign * torch.sgn(x_mod)
        # self.last_m_mod_sign = torch.sign(m_mod)
        #
        # x_mod = torch.abs(m_mod) * torch.sgn(x_mod)
        #
        # grad_norm = torch.sqrt((torch.abs(grad) ** 2).sum(dim=(1, 2, 3), keepdim=True))  # (B, 1, 1, 1)
        # grad_log_lh_inv = self.linear_tfm.log_lh_grad(x_mod, self.measurement, 1.)
        # grad_log_lh_inv_norm = torch.sqrt((torch.abs(grad_log_lh_inv) ** 2).sum(dim=(1, 2, 3), keepdim=True))
        # grad_log_lh_inv = grad_log_lh_inv / grad_log_lh_inv_norm * grad_norm
        # print(f"grad_log_lh_inv: {torch.norm(grad_log_lh_inv)}")  ###
        # x_mod += grad_log_lh_inv * step_size
        #
        # x_mod_angle_out = round_sign(x_mod)
        # sign_flip_mask = x_mod_angle_in * x_mod_angle_out
        #
        # # pass sign flip from x_mod to m_mod
        # # m_mod = torch.abs(x_mod) * torch.sign(m_mod) * sign_flip_mask
        # m_mod = torch.abs(x_mod) * torch.sign(m_mod)
        #
        # return x_mod, m_mod

        m_mod = torch.maximum(m_mod, torch.tensor(0).to(m_mod.device))
        x_mod = m_mod * torch.sgn(x_mod)

        grad_log_lh_inv = self.linear_tfm.log_lh_grad(x_mod, self.measurement, 1.) / sigma
        print(f"grad_log_lh_inv: {torch.norm(grad_log_lh_inv)}")  ###

        for _ in range(self.config.sampling.complex_inner_n_steps):
            noise = torch.randn_like(x_mod)
            x_mod = x_mod + step_size * grad_log_lh_inv + noise * torch.sqrt(step_size * 2)

        m_mod = torch.abs(x_mod)

        return x_mod, m_mod
