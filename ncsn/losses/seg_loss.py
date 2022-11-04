import torch
import torch.nn as nn

from monai.losses import DiceCELoss


def seg_loss_with_perturbation(model: nn.Module, X: torch.Tensor, y: torch.Tensor, sigmas, labels=None):
    # X: (B, C, H, W); y: (B,)
    sigmas = sigmas.to(X.device)  # (L_noise,)
    B = X.shape[0]
    if labels is None:
        labels = torch.randint(0, len(sigmas), (B,), device=X.device)
    # (L_noise,) -> (B,) -> (B, 1, 1, 1)
    used_sigmas = sigmas[labels].view(B, *([1] * len(X.shape[1:])))
    noise = torch.randn_like(X) * used_sigmas  # (B, C, H, W)
    X_perturbed = X + noise
    y_pred = model(X_perturbed)  # (B, 2, H, W)

    loss_fn = DiceCELoss(
        include_background=False,
        to_onehot_y=True,
        softmax=True,
        squared_pred=True,
        lambda_ce=0.5,
        lambda_dice=0.5
    )
    loss = loss_fn(y_pred, y)

    return loss, y_pred