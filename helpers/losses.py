import torch

from InverseProblemWithDiffusionModel.sde.sde_lib import SDE
from InverseProblemWithDiffusionModel.helpers.utils import expand_like


def get_loss_fn(sde: SDE, eps=1e-5):
    def loss_fn(model, X):
        # X: e.g: (B, C, H, W)
        # (B,)
        T = torch.rand(X.shape[0], device=X.device, dtype=X.dtype) * (sde.T - eps) + eps  # sde.T: end-time
        Z = torch.randn_like(X)
        # (B, C, H, W), (B,)
        mean, std = sde.marginal_prob(X, T)  # diagonal Cov with same std
        std = expand_like(std, Z)  # (B, 1, 1, 1)
        X_perturbed = mean + Z * std  # (B, C, H, W)
        score = model(X_perturbed, T)  # (B, C, H, W)
        # (B, C, H, W) -> (B,) -> (1,)
        # loss = ((score * std + Z) ** 2).mean(dim=[1, 2, 3]).mean()
        loss = ((score + Z / std) ** 2).mean(dim=[1, 2, 3]).mean()

        return loss

    return loss_fn
