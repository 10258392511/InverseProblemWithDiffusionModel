import torch

from . import LinearTransform


class SkipLines(LinearTransform):
    """
    A = P * M where P is extraction
    """
    def __init__(self, num_skip_lines, in_shape):
        super(SkipLines, self).__init__()
        self.num_skip_lines = num_skip_lines
        self.in_shape = in_shape

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        # X: (B, C, H, W)
        X_out = X[:, :, 0::self.num_skip_lines, :]

        return X_out

    def conj_op(self, S: torch.Tensor) -> torch.Tensor:
        """
        Zero-padded
        """
        S_out = torch.zeros((S.shape[0], *self.in_shape), dtype=S.dtype, device=S.device)
        S_out[:, :, 0::self.num_skip_lines] = S

        return S_out

    def projection(self, X: torch.Tensor, S: torch.Tensor, lamda: float) -> torch.Tensor:
        """
        x <- lamda * s + (1 - lamda) * M * x + (I - M) * x
        X: (B, C, H, W), perturbed part (masked out part); S: (B, C, H_s, W_s)
        """
        B, C, H, W = X.shape
        X_out = torch.empty_like(X)
        line_mask_retained = torch.zeros(H, device=X.device, dtype=bool)
        line_mask_retained[::self.num_skip_lines] = True
        masked = X[:, :, torch.logical_not(line_mask_retained), :]  # masked out
        unmasked = lamda * S + (1 - lamda) * self(X)  # (B, C, H', W)
        X_out[:, :, line_mask_retained, :] = unmasked
        X_out[:, :, torch.logical_not(line_mask_retained), :] = masked

        return X_out
