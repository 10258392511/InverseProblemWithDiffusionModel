import torch
import warnings

from . import i2k_complex, k2i_complex, LinearTransform
from .masking import SkipLines


class UndersamplingFourier(LinearTransform):
    def __init__(self, num_skip_lines, in_shape):
        self.skip_lines = SkipLines(num_skip_lines, in_shape)

    def __call__(self, X: torch.Tensor):
        # X: (B, C, H, W)
        # assert X.dtype == torch.complex64
        X = X.to(torch.complex64)
        S = i2k_complex(X)
        S = self.skip_lines(S)

        # (B, C, H', W)
        return S

    def conj_op(self, S: torch.Tensor) -> torch.Tensor:
        # S: (B, C, H, W)
        S = S.to(torch.complex64)
        X = self.skip_lines.conj_op(S)
        X = k2i_complex(X)

        # (B, C, H, W)
        return X

    def projection(self, X: torch.Tensor, S: torch.Tensor, lamda: float) -> torch.Tensor:
        warnings.warn("Not used!")

        return X


class RandomUndersamplingFourier(LinearTransform):
    def __init__(self, R, center_lines_frac, in_shape, seed=None):
        """
        in_shape: (C, H, W)
        """
        self.R = R
        self.center_lines_frac = center_lines_frac
        self.in_shape = in_shape
        self.seed = seed
        self.mask = self._generate_mask()

    def _generate_mask(self):
        # mask: (C, H, W)
        torch.random.manual_seed(self.seed)
        C, H, W = self.in_shape
        mask = (torch.rand(self.in_shape) <= 1 / self.R).float()
        win_size = int(H * self.center_lines_frac)
        half_win_size = H // 2
        start_idx = half_win_size - win_size // 2
        end_idx = start_idx + win_size
        mask[:, start_idx:end_idx, :] = 1.

        return mask

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        # X: (B, C, H, W)
        mask = self.mask.to(X.device)
        S = mask * i2k_complex(X)  # (B, C, H, W)

        return S

    def conj_op(self, S: torch.Tensor) -> torch.Tensor:
        X = k2i_complex(S)

        return X

    def projection(self, X: torch.Tensor, S: torch.Tensor, lamda: float) -> torch.Tensor:
        warnings.warn("Not used!")

        return X

