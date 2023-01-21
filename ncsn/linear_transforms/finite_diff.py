import torch

from . import LinearTransform
from typing import Union, Tuple


class FiniteDiff(LinearTransform):
    def __init__(self, dims: Union[int, Tuple[int]]):
        self.dims = dims

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        # X: (B, C, T, H, W)
        X_forward = torch.roll(X, -1, self.dims)
        X_out = X_forward - X

        return X_out
    
    def conj_op(self, S: torch.Tensor) -> torch.Tensor:
        # S: (B, C, T, H, W)
        S_backward = torch.roll(S, 1, self.dims)
        S_out = S_backward - S

        return S_out
    
    def projection(self, X: torch.Tensor, S: torch.Tensor, lamda: float) -> torch.Tensor:
        
        return super().projection(X, S, lamda)

    def log_lh_grad(self, X: torch.Tensor, S: torch.Tensor = None, lamda: float = 1) -> torch.Tensor:
        """
        grad = -lamda * nabla' @ (sign(nabla @ X)) 
        """
        grad = -lamda * self.conj_op(torch.sign(self(X)))

        return grad
