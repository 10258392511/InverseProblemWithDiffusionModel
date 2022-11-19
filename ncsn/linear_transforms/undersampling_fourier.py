import torch
import warnings

from . import i2k_complex, k2i_complex, LinearTransform
from .masking import SkipLines


# TODO: implement this
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
