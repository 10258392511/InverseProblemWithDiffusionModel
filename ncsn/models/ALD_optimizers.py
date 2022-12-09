import abc
import torch
import os
import InverseProblemWithDiffusionModel.helpers.pytorch_utils as ptu

from . import freeze_model, compute_clf_grad, compute_seg_grad
from .proximal_op import Proximal, L2Penalty, Constrained, SingleCoil
from InverseProblemWithDiffusionModel.helpers.utils import data_transform, vis_images
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
        grad_log_lh_measurement_norm = torch.sqrt((grad_log_lh_measurement ** 2).sum(dim=(1, 2, 3), keepdim=True))  # (B, 1, 1, 1)
        grad_log_lh_measurement_norm = torch.norm(grad_log_lh_measurement)
        grad_log_lh_measurement = grad_log_lh_measurement / (grad_log_lh_measurement_norm) * grad_norm
        print(f"grad: {torch.norm(grad_norm)}, grad_log_lh_clf: {torch.norm(grad_log_lh_clf)}, grad_log_lh_measurement: {torch.norm(grad_log_lh_measurement)}l")
        # grad += (grad_log_lh_clf / sigma * lamda + grad_log_lh_measurement * (1 - lamda) / sigma)   
        grad += (grad_log_lh_clf / sigma * lamda + grad_log_lh_measurement * (1 - lamda))
        # grad += (grad_log_lh_clf * lamda + grad_log_lh_measurement * (1 - lamda) / sigma)   

        return grad


class ALDInvSeg(ALDOptimizer):
    def __init__(self, seg_start_time, seg_step_type, *args, **kwargs):
        """
        seg_start_time: in [0, 1]
        """
        super(ALDInvSeg, self).__init__(*args, **kwargs)
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
        x_mod = self.linear_tfm.conj_op(self.measurement)  # (B, C, H, W)
        # x_mod = m_mod * torch.exp(1j * (torch.rand(m_mod.shape, device=m_mod.device) * 2 - 1) * torch.pi)
        x_mod = m_mod * torch.sgn(x_mod)

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
        
        grad_norm = torch.sqrt((torch.abs(grad) ** 2).sum(dim=(1, 2, 3), keepdim=True))  # (B, 1, 1, 1)
        # # grad_log_lh_inv = self.linear_tfm.log_lh_grad(x_mod, self.measurement, 1.) / sigma
        # grad_log_lh_inv = self.linear_tfm.log_lh_grad(x_mod, self.measurement, 1.)
        # grad_log_lh_inv_norm = torch.sqrt((torch.abs(grad_log_lh_inv) ** 2).sum(dim=(1, 2, 3), keepdim=True))
        # grad_log_lh_inv = grad_log_lh_inv / grad_log_lh_inv_norm * grad_norm

        for _ in range(self.config.sampling.complex_inner_n_steps):
            # grad_log_lh_inv = self.linear_tfm.log_lh_grad(x_mod, self.measurement, 1.) / sigma
            grad_log_lh_inv = self.linear_tfm.log_lh_grad(x_mod, self.measurement, 1.)
            grad_log_lh_inv_norm = torch.sqrt((torch.abs(grad_log_lh_inv) ** 2).sum(dim=(1, 2, 3), keepdim=True))
            grad_log_lh_inv = grad_log_lh_inv / grad_log_lh_inv_norm * grad_norm
     
            print(f"grad_log_lh_inv: {torch.norm(grad_log_lh_inv)}")  ###
            
            noise = torch.randn_like(x_mod)
            x_mod = x_mod + step_size * grad_log_lh_inv + noise * torch.sqrt(step_size * 2)
            # x_mod = x_mod + step_size * grad_log_lh_inv

        # m_mod = torch.abs(x_mod) * torch.sign(m_mod) 
        m_mod = torch.abs(x_mod)

        return x_mod, m_mod


class ALDInvClfProximal(ALDInvClf):
    def __init__(self, proximal: Proximal, clf_start_time, clf_step_type="linear", *args, **kwargs):
        super(ALDInvClfProximal, self).__init__(*args, **kwargs)
        self.proximal = proximal
        self.clf_start_time = clf_start_time
        self.clf_step_type = clf_step_type
        self.lh_weights = get_lh_weights(self.sigmas, self.clf_start_time, self.clf_step_type)  # (L,)

    def __call__(self, **kwargs):
        """
        kwargs: cls, lamda, save_dir, lr_scaled
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
                vis_images(x_mod[0], if_save=True, save_dir=os.path.join(kwargs.get("save_dir"), "sampling_snapshots/"),
                           filename=f"step_{c}_start_time_{self.clf_start_time}.png")

            labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c  # (B,)
            labels = labels.long()
            step_size = step_lr * (sigma / sigmas[-1]) ** 2  # (L_noise,)
            clf_lamda = self.lh_weights[c]

            ### inserting pt ###
            # x_mod = self.init_estimation(x_mod, alpha=step_size, sigma=sigma, **kwargs)
            # x_mod = self.init_estimation(x_mod)
            ####################

            for s in range(n_steps_each):
                grad = scorenet(x_mod, labels)  # (B, C, H, W)

                ### inserting pt ###
                grad = self.adjust_grad(grad, x_mod, sigma=sigma, clf_lamda=clf_lamda, **kwargs)
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

            ### inserting pt ###
            x_mod = self.post_processing(x_mod, alpha=step_lr, sigma=sigma, **kwargs)
            ####################
    
        if denoise:
            last_noise = (len(sigmas) - 1) * torch.ones(x_mod.shape[0], device=x_mod.device)
            last_noise = last_noise.long()
            x_mod = x_mod + sigmas[-1] ** 2 * scorenet(x_mod, last_noise)
            images.append(x_mod.to('cpu'))

        if final_only:
            return [x_mod.to('cpu')]
        else:
            return images

    def adjust_grad(self, grad, x_mod, **kwargs):
        """
        kwargs: cls, clf_lamda, sigma
        """
        # x_mod: (B, C, H, W)
        cls = kwargs["cls"]
        lamda = kwargs["clf_lamda"]
        sigma = kwargs["sigma"]

        grad_log_lh_clf = compute_clf_grad(self.clf, x_mod, cls)  # (B, C, H, W)
        grad = grad + grad_log_lh_clf / sigma * lamda

        return grad

    def post_processing(self, x_mod, **kwargs):
        """
        kwargs:
            sigma: noise std
            L2Penalty:
                alpha: step-size for the ALD step, unscaled (i.e lr)
                lamda: hyper-param, for ||Ax - y||^2 / (lamda^2 + sigma^2)
                       i.e lamda = sigma_data
                lr_scaled: empirical step-length
            Constrained:
                lamda: hyper-param, for balancing info retained and not retained
        """
        sigma = kwargs["sigma"]
        measurement = self.measurement + sigma * self.linear_tfm(torch.rand_like(x_mod))

        if isinstance(self.proximal, L2Penalty):
            alpha = kwargs["alpha"]
            lamda = kwargs["lamda"]
            lr_scaled = kwargs["lr_scaled"]
            # x_mod = self.proximal(x_mod, self.measurement, alpha, lamda + sigma ** 2)
            x_mod = self.proximal(x_mod, self.measurement, lr_scaled, lamda + sigma ** 2)

            return x_mod

        elif isinstance(self.proximal, Constrained):
            lamda = kwargs["lamda"]
            x_mod = self.proximal(x_mod, self.measurement, lamda)

            return x_mod

        else:
            return x_mod


class ALDInvSegProximal(ALDInvSeg):
    def __init__(self, proximal: Proximal, seg_start_time, seg_step_type, *args, **kwargs):
        super(ALDInvSegProximal, self).__init__(seg_start_time, seg_step_type, *args, **kwargs)
        self.proximal = proximal
        # self.seg_start_time = seg_start_time
        # self.seg_step_type = seg_step_type
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
        # x_mod = torch.rand(*self.x_mod_shape).to(self.device)
        ### inserting pt ###
        m_mod = self.init_x_mod()
        ####################
        m_mod = data_transform(self.config, m_mod)
        # x_mod = self.linear_tfm.conj_op(self.measurement)  # (B, C, H, W)
        # x_mod = m_mod * torch.sgn(x_mod)
        # x_mod = m_mod * torch.exp(1j * (torch.rand(m_mod.shape, device=m_mod.device) * 2 - 1) * torch.pi)
        x_mod = m_mod.to(torch.complex64)

        images = []
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
                # vis_images(torch.abs(x_mod[0]), if_save=True, save_dir=kwargs.get("save_dir"),
                #            filename=f"step_{c}_start_time_{self.seg_start_time}_acdc.png")
                vis_images(m_mod[0], if_save=True, save_dir=os.path.join(kwargs.get("save_dir"), "sampling_snapshots/"),
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
                grad_prior = scorenet(m_mod, labels)  # (B, C, H, W)

                ### inserting pt ###
                grad = self.adjust_grad(grad_prior, m_mod, sigma=sigma, seg_lamda=lh_seg_weight, **kwargs)
                ####################

                noise = torch.randn_like(m_mod)
                grad_norm = torch.norm(grad.view(grad.shape[0], -1), dim=-1).mean()
                noise_norm = torch.norm(noise.view(noise.shape[0], -1), dim=-1).mean()

                m_mod = m_mod + step_size * grad + noise * torch.sqrt(step_size * 2)

                print(f"m_mod, {s + 1}/{n_steps_each}: {(m_mod.max(), m_mod.min())}")  ###

                image_norm = torch.norm(x_mod.view(x_mod.shape[0], -1), dim=-1).mean()
                snr = torch.sqrt(step_size / 2.) * grad_norm / noise_norm
                grad_mean_norm = torch.norm(grad.mean(dim=0).view(-1)) ** 2 * sigma ** 2


                if not final_only:
                    images.append(m_mod.to('cpu'))

            ### inserting pt ###
            # m_mod, x_mod = self.post_processing(m_mod, x_mod, alpha=step_lr, sigma=sigma, **kwargs)
            m_mod, _ = self.post_processing(m_mod, x_mod, alpha=step_lr, sigma=sigma, **kwargs)
            ####################

        if denoise:
            last_noise = (len(sigmas) - 1) * torch.ones(x_mod.shape[0], device=x_mod.device)
            last_noise = last_noise.long()
            m_mod = m_mod + sigmas[-1] ** 2 * scorenet(m_mod, last_noise)
        
        ### inserting pt ###
        m_mod = self.last_prox_step(m_mod)
        ####################
        images.append(m_mod.to('cpu'))

        if final_only:
            return [m_mod.to('cpu')]
        else:
            return images

    def adjust_grad(self, grad, m_mod, **kwargs):
        """
        kwargs: label, seg_lamda, sigma
        """
        # m_mod: (B, C, H, W)
        label = kwargs["label"]
        lamda = kwargs["seg_lamda"]
        sigma = kwargs["sigma"]

        grad_log_lh_seg = compute_seg_grad(self.seg, m_mod, label)  # (B, C, H, W)
        grad = grad + grad_log_lh_seg / sigma * lamda

        return grad

    def post_processing(self, m_mod, x_mod, **kwargs):
        """
        kwargs:
            sigma: noise std
            L2Penalty:
                alpha: step-size for the ALD step, unscaled (i.e lr)
                lamda: hyper-param, for ||Ax - y||^2 / (lamda^2 + sigma^2)
                       i.e lamda = sigma_data
                lr_scaled: empirical step-length
            Constrained:
                lamda: hyper-param, for balancing info retained and not retained
        """
        sigma = kwargs["sigma"]
        ###
        # x_mod = m_mod * torch.sgn(x_mod)
        x_mod = torch.abs(m_mod) * torch.sgn(x_mod)
        ###
        if isinstance(self.proximal, L2Penalty) or isinstance(self.proximal, SingleCoil):
            alpha = kwargs["alpha"]
            lamda = kwargs["lamda"]
            lr_scaled = kwargs["lr_scaled"]

            if self.if_print:
                vis_images(torch.abs(x_mod[0]), torch.angle(x_mod[0]), if_save=True,
                           save_dir=self.print_args["save_dir"], filename=f"step_{self.print_args['c']}_before.png")
                mag_before = torch.abs(x_mod[0])
                phase_before = torch.angle(x_mod[0])

            # x_mod = self.proximal(x_mod, self.measurement, lr_scaled, lamda + sigma ** 2)
            x_mod = self.proximal(x_mod, self.measurement, 2 * alpha * (sigma / self.sigmas[-1]) ** 2,
                                  lamda + sigma ** 2)

            if self.if_print:
                vis_images(torch.abs(x_mod[0]), torch.angle(x_mod[0]), if_save=True,
                           save_dir=self.print_args["save_dir"], filename=f"step_{self.print_args['c']}_after.png")
                mag_diff = torch.abs(x_mod[0]) - mag_before
                phase_diff = torch.angle(x_mod[0]) - phase_before
                vis_images(mag_diff, phase_diff, if_save=True, 
                           save_dir=self.print_args["save_dir"], filename=f"step_{self.print_args['c']}_diff.png")

            ###
            # m_mod = torch.abs(x_mod) * torch.sign(m_mod)
            m_mod = torch.abs(x_mod)
            ###

            return m_mod, x_mod

        elif isinstance(self.proximal, Constrained):
            lamda = kwargs["lamda"]
            x_mod = self.proximal(x_mod, self.measurement, lamda)
            ###
            # m_mod = torch.abs(x_mod) * torch.sign(m_mod)
            m_mod = torch.abs(x_mod)
            ###

            return m_mod, x_mod

        else:
            return m_mod, x_mod

    def last_prox_step(self, m_mod, **kwargs):
        """
        kwargs: num_steps, lr

        x0 <- z0.to(complex64)
        x0 <- argmin_x ||Ax - y0||_2^2
        z0 <- |x0|
        """
        # torch.set_grad_enabled(True)
        # num_steps = kwargs.get("num_steps", 10)
        # lr = kwargs.get("lr", 1e-2)

        # x_mod = m_mod.to(torch.complex64)  # (B, C, H, W)
        # x_mod.requires_grad = True
        # opt = torch.optim.Adam([x_mod], lr=lr)
        # for i in range(num_steps):
        #     s_pred = self.linear_tfm(x_mod)
        #     loss = (torch.abs(s_pred - self.measurement) ** 2).sum(dim=(1, 2, 3)).mean()
        #     opt.zero_grad()
        #     loss.backward()
        #     opt.step()
        #     print(f"last prox step {i + 1}/{num_steps}: loss {loss.item()}")
        
        # torch.set_grad_enabled(False)
        # m_mod = torch.abs(x_mod.detach())
        
        # return m_mod
        assert self.linear_tfm.mask is not None
        mask = self.linear_tfm.mask.to(m_mod.device)
        m_mod = k2i_complex(self.measurement + (1 - mask) * i2k_complex(m_mod))

        return m_mod
