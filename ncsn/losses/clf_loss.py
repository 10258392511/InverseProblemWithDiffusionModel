import torch
import torch.nn as nn


def clf_loss_with_perturbation(model: nn.Module, X: torch.Tensor, y: torch.Tensor, sigmas, labels=None):
    # X: (B, C, H, W); y: (B,)
    sigmas = sigmas.to(X.device)  # (L_noise,)
    B = X.shape[0]
    if labels is None:
        labels = torch.randint(0, len(sigmas), (B,), device=X.device)
    # (L_noise,) -> (B,) -> (B, 1, 1, 1)
    used_sigmas = sigmas[labels].view(B, *([1] * len(X.shape[1:])))
    noise = torch.randn_like(X) * used_sigmas  # (B, C, H, W)
    X_perturbed = X + noise
    y_pred = model(X_perturbed)  # (B,)

    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(y_pred, y)

    return loss, y_pred
