import numpy as np
import torch
import torch.nn as nn


from InverseProblemWithDiffusionModel.ncsn.linear_transforms import LinearTransform
from InverseProblemWithDiffusionModel.helpers.utils import vis_tensor


def get_sigmas(config, mode="unconditioned"):
    assert mode in ("unconditioned", "recons")
    if mode == "recons":
        if config.recons.sigma_dist == 'geometric':
            sigmas = torch.tensor(
                np.exp(np.linspace(np.log(config.recons.sigma_begin), np.log(config.recons.sigma_end),
                                config.recons.num_classes))).float().to(config.device)
        elif config.recons.sigma_dist == 'uniform':
            sigmas = torch.tensor(
                np.linspace(config.recons.sigma_begin, config.recons.sigma_end, config.recons.num_classes)
            ).float().to(config.device)

        else:
            raise NotImplementedError('sigma distribution not supported')
        
    else:
        if config.model.sigma_dist == 'geometric':
            sigmas = torch.tensor(
                np.exp(np.linspace(np.log(config.model.sigma_begin), np.log(config.model.sigma_end),
                                config.model.num_classes))).float().to(config.device)
        elif config.model.sigma_dist == 'uniform':
            sigmas = torch.tensor(
                np.linspace(config.model.sigma_begin, config.model.sigma_end, config.model.num_classes)
            ).float().to(config.device)

        else:
            raise NotImplementedError('sigma distribution not supported')

    return sigmas

@torch.no_grad()
def anneal_Langevin_dynamics(x_mod, scorenet, sigmas, n_steps_each=200, step_lr=0.000008,
                             final_only=False, verbose=False, denoise=True):
    print("sampling...")
    images = []
    print_interval = len(sigmas) // 10

    with torch.no_grad():
        for c, sigma in enumerate(sigmas):
            if c % print_interval == 0:
                print(f"{c + 1}/{len(sigmas)}")
            labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
            labels = labels.long()
            step_size = step_lr * (sigma / sigmas[-1]) ** 2

            for s in range(n_steps_each):
                grad = scorenet(x_mod, labels)

                noise = torch.randn_like(x_mod)
                grad_norm = torch.norm(grad.view(grad.shape[0], -1), dim=-1).mean()
                noise_norm = torch.norm(noise.view(noise.shape[0], -1), dim=-1).mean()
                x_mod = x_mod + step_size * grad + noise * torch.sqrt(step_size * 2)

                image_norm = torch.norm(x_mod.view(x_mod.shape[0], -1), dim=-1).mean()
                snr = torch.sqrt(step_size / 2.) * grad_norm / noise_norm
                grad_mean_norm = torch.norm(grad.mean(dim=0).view(-1)) ** 2 * sigma ** 2

                if not final_only:
                    images.append(x_mod.to('cpu'))
                if verbose:
                    print("level: {}, step_size: {}, grad_norm: {}, image_norm: {}, snr: {}, grad_mean_norm: {}".format(
                        c, step_size, grad_norm.item(), image_norm.item(), snr.item(), grad_mean_norm.item()))

        if denoise:
            last_noise = (len(sigmas) - 1) * torch.ones(x_mod.shape[0], device=x_mod.device)
            last_noise = last_noise.long()
            x_mod = x_mod + sigmas[-1] ** 2 * scorenet(x_mod, last_noise)
            images.append(x_mod.to('cpu'))

        if final_only:
            return [x_mod.to('cpu')]
        else:
            return images


@torch.no_grad()
def anneal_Langevin_dynamics_inpainting(x_mod, refer_image, scorenet, sigmas, image_size,
                                        n_steps_each=100, step_lr=0.000008):
    """
    Currently only good for 32x32 images. Assuming the right half is missing.
    """

    images = []

    refer_image = refer_image.unsqueeze(1).expand(-1, x_mod.shape[1], -1, -1, -1)
    refer_image = refer_image.contiguous().view(-1, 3, image_size, image_size)
    x_mod = x_mod.view(-1, 3, image_size, image_size)
    cols = image_size // 2
    half_refer_image = refer_image[..., :cols]
    with torch.no_grad():
        for c, sigma in enumerate(sigmas):
            labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
            labels = labels.long()
            step_size = step_lr * (sigma / sigmas[-1]) ** 2

            for s in range(n_steps_each):
                images.append(x_mod.to('cpu'))
                corrupted_half_image = half_refer_image + torch.randn_like(half_refer_image) * sigma
                x_mod[:, :, :, :cols] = corrupted_half_image
                noise = torch.randn_like(x_mod) * torch.sqrt(step_size * 2)
                grad = scorenet(x_mod, labels)
                x_mod = x_mod + step_size * grad + noise
                print("class: {}, step_size: {}, mean {}, max {}".format(c, step_size, grad.abs().mean(),
                                                                         grad.abs().max()))

        return images

@torch.no_grad()
def anneal_Langevin_dynamics_interpolation(x_mod, scorenet, sigmas, n_interpolations, n_steps_each=200, step_lr=0.000008,
                             final_only=False, verbose=False):
    images = []

    n_rows = x_mod.shape[0]

    x_mod = x_mod[:, None, ...].repeat(1, n_interpolations, 1, 1, 1)
    x_mod = x_mod.reshape(-1, *x_mod.shape[2:])

    for c, sigma in enumerate(sigmas):
        labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
        labels = labels.long()
        step_size = step_lr * (sigma / sigmas[-1]) ** 2
        for s in range(n_steps_each):
            grad = scorenet(x_mod, labels)

            noise_p = torch.randn(n_rows, x_mod.shape[1], x_mod.shape[2], x_mod.shape[3],
                                  device=x_mod.device)
            noise_q = torch.randn(n_rows, x_mod.shape[1], x_mod.shape[2], x_mod.shape[3],
                                  device=x_mod.device)
            angles = torch.linspace(0, np.pi / 2., n_interpolations, device=x_mod.device)

            noise = noise_p[:, None, ...] * torch.cos(angles)[None, :, None, None, None] + \
                        noise_q[:, None, ...] * torch.sin(angles)[None, :, None, None, None]

            noise = noise.reshape(-1, *noise.shape[2:])
            grad_norm = torch.norm(grad.view(grad.shape[0], -1), dim=-1).mean()
            noise_norm = torch.norm(noise.view(noise.shape[0], -1), dim=-1).mean()
            image_norm = torch.norm(x_mod.view(x_mod.shape[0], -1), dim=-1).mean()

            x_mod = x_mod + step_size * grad + noise * torch.sqrt(step_size * 2)

            snr = torch.sqrt(step_size / 2.) * grad_norm / noise_norm

            if not final_only:
                images.append(x_mod.to('cpu'))
            if verbose:
                print(
                    "level: {}, step_size: {}, image_norm: {}, grad_norm: {}, snr: {}".format(
                        c, step_size, image_norm.item(), grad_norm.item(), snr.item()))


    if final_only:
        return [x_mod.to('cpu')]
    else:
        return images


def freeze_model(model: nn.Module):
    model.eval()
    for param in model.parameters():
        param.requires_grad = False


def compute_clf_grad(clf, X, cls):
    # X: (B, C, H, W); cls: (B,)
    torch.set_grad_enabled(True)
    # grad_out = torch.empty_like(X)
    # for idx in range(cls.shape[0]):
    #     X_iter = X[idx : idx + 1, ...]  # (1, C, H, W)
    #     X_iter.requires_grad = True
    #     y_pred = clf(X_iter)  # (1, num_cls)
    #     y_pred = torch.softmax(y_pred, dim=-1)  # (1, num_cls)
    #     y_sel = torch.log(y_pred[0, cls[idx]])  # float
    #     y_sel.backward()
    #     grad_out[idx : idx + 1] = X_iter.grad
    X.requires_grad = True
    y_pred = clf(X)  # (B,, num_cls)
    y_pred = torch.softmax(y_pred, dim=1)  # (B, num_cls)
    y_pred_sel = torch.gather(y_pred, dim=1, index=cls.unsqueeze(1)).squeeze()  # (B, 1) -> (B,)
    loss = torch.log(y_pred_sel).sum()
    loss.backward()
    grad_out = X.grad

    torch.set_grad_enabled(False)

    return grad_out


def compute_seg_grad(seg, X, label, mode="full"):
    # X: (B, C, H, W); label: (B, 1, H, W)
    assert mode in ["full", "FG"]
    torch.set_grad_enabled(True)
    X.requires_grad = True
    y_pred = seg(X)  # (B, num_cls, H, W)
    y_pred = torch.softmax(y_pred, dim=1)  # (B, num_cls, H, W)
    y_pred_sel = torch.gather(y_pred, dim=1, index=label)  # (B, 1, H, W)
    # (B, 1, H, W) -> (B,) -> float
    # requiring re-scale the grad later, so for each image "mean" is used to avoid numerical instability
    loss = torch.log(y_pred_sel).sum(dim=(1, 2, 3)).sum()
    loss.backward()
    grad_out = X.grad

    torch.set_grad_enabled(False)
    if mode == "FG":
        grad_out = grad_out * label

    return grad_out


@torch.no_grad()
def anneal_Langevin_dynamics_cls_conditioned(x_mod, cls, scorenet, clf, sigmas, n_steps_each=200, step_lr=0.000008,
                             final_only=False, verbose=False, denoise=True):
    # x_mod: (B, C, H, W), cls: (B,)
    print("sampling...")
    images = []
    print_interval = len(sigmas) // 10
    freeze_model(clf)

    for c, sigma in enumerate(sigmas):
        if c % print_interval == 0:
            print(f"{c + 1}/{len(sigmas)}")

        labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c  # (B,)
        labels = labels.long()
        step_size = step_lr * (sigma / sigmas[-1]) ** 2  # (L_noise,)

        for s in range(n_steps_each):
            grad = scorenet(x_mod, labels)  # (B, C, H, W)
            grad_log_lh = compute_clf_grad(clf, x_mod, cls)  # (B, C, H, W)
            # grad_log_lh = compute_clf_grad(clf, x_mod.clone(), cls)  # (B, C, H, W)
            grad += grad_log_lh

            noise = torch.randn_like(x_mod)
            grad_norm = torch.norm(grad.view(grad.shape[0], -1), dim=-1).mean()
            noise_norm = torch.norm(noise.view(noise.shape[0], -1), dim=-1).mean()
            x_mod = x_mod + step_size * grad + noise * torch.sqrt(step_size * 2)

            image_norm = torch.norm(x_mod.view(x_mod.shape[0], -1), dim=-1).mean()
            snr = torch.sqrt(step_size / 2.) * grad_norm / noise_norm
            grad_mean_norm = torch.norm(grad.mean(dim=0).view(-1)) ** 2 * sigma ** 2

            if not final_only:
                images.append(x_mod.to('cpu'))
            if verbose:
                print("level: {}, step_size: {}, grad_norm: {}, image_norm: {}, snr: {}, grad_mean_norm: {}".format(
                    c, step_size, grad_norm.item(), image_norm.item(), snr.item(), grad_mean_norm.item()))

    if denoise:
        last_noise = (len(sigmas) - 1) * torch.ones(x_mod.shape[0], device=x_mod.device)
        last_noise = last_noise.long()
        x_mod = x_mod + sigmas[-1] ** 2 * scorenet(x_mod, last_noise)
        images.append(x_mod.to('cpu'))

    if final_only:
        return [x_mod.to('cpu')]
    else:
        return images


@torch.no_grad()
def anneal_Langevin_dynamics_inverse_problem(x_mod, measurement, scorenet, linear_tfm: LinearTransform, sigmas, lamdas,
                                             n_steps_each=100, step_lr=0.000008, denoise=True, final_only=False,
                                             perturb_measurement=False):
    """
    x_mod: (B, C, H, W)
    measurement: (B, C, H_s, W_s)
    sigmas: (L_noise,)
    lamdas: float or (L_noise,)
    """
    print("sampling...")
    if isinstance(lamdas, float):
        lamdas = torch.ones_like(sigmas) * lamdas  # (L_noise)

    images = []
    print_interval = len(sigmas) // 10

    for c, sigma in enumerate(sigmas):
        if c % print_interval == 0:
            print(f"{c + 1}/{len(sigmas)}")
            # vis_tensor(x_mod)

        labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c  # (B,)
        labels = labels.long()
        step_size = step_lr * (sigma / sigmas[-1]) ** 2  # (L_noise,)
        lamda_iter = lamdas[c]
        measurement_iter = measurement
        if perturb_measurement:
            measurement_iter += sigma * linear_tfm(torch.randn_like(x_mod))

        for s in range(n_steps_each):
            grad = scorenet(x_mod, labels)  # (B, C, H, W)
            grad_log_lh = linear_tfm.log_lh_grad(x_mod, measurement_iter, lamda_iter)  # (B, C, H, W)
            grad_norm = torch.norm(grad)
            grad_log_lh_norm = torch.norm(grad_log_lh)
            # print(f"log_lh_norm: {grad_log_lh_norm}, grad_norm: {grad_norm}")
            grad += grad_log_lh / grad_log_lh_norm * grad_norm

            noise = torch.randn_like(x_mod)
            grad_norm = torch.norm(grad.view(grad.shape[0], -1), dim=-1).mean()
            noise_norm = torch.norm(noise.view(noise.shape[0], -1), dim=-1).mean()
            x_mod = x_mod + step_size * grad + noise * torch.sqrt(step_size * 2)

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


@torch.no_grad()
def anneal_Langevin_dynamics_inverse_problem_proj(x_mod, measurement, scorenet, linear_tfm: LinearTransform,
                                                  sigmas, lamdas, n_steps_each=100, step_lr=0.000008,
                                                  denoise=True, final_only=False, perturb_measurement=False):
    """
    x_mod: (B, C, H, W)
    measurement: (B, C, H_s, W_s)
    sigmas: (L_noise,)
    lamdas: float or (L_noise,)
    """
    if isinstance(lamdas, float):
        lamdas = torch.ones_like(sigmas) * lamdas  # (L_noise)
    """
    x_mod: (B, C, H, W)
    measurement: (B, C, H_s, W_s)
    sigmas: (L_noise,)
    lamdas: float or (L_noise,)
    """
    print("sampling...")
    if isinstance(lamdas, float):
        lamdas = torch.ones_like(sigmas) * lamdas  # (L_noise)

    images = []
    print_interval = len(sigmas) // 10

    for c, sigma in enumerate(sigmas):
        if c % print_interval == 0:
            print(f"{c + 1}/{len(sigmas)}")
            # vis_tensor(x_mod)

        labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c  # (B,)
        labels = labels.long()
        step_size = step_lr * (sigma / sigmas[-1]) ** 2  # (L_noise,)
        lamda_iter = lamdas[c]
        measurement_iter = measurement
        if perturb_measurement:
            measurement_iter += sigma * linear_tfm(torch.randn_like(x_mod))

        x_mod = linear_tfm.projection(x_mod, measurement_iter, lamda_iter)
        for s in range(n_steps_each):
            grad = scorenet(x_mod, labels)  # (B, C, H, W)

            noise = torch.randn_like(x_mod)
            grad_norm = torch.norm(grad.view(grad.shape[0], -1), dim=-1).mean()
            noise_norm = torch.norm(noise.view(noise.shape[0], -1), dim=-1).mean()
            x_mod = x_mod + step_size * grad + noise * torch.sqrt(step_size * 2)

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
