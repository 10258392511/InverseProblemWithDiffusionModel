import abc
import torch


class LinearTransform(abc.ABC):
    """
    All inputs: (B, C, H, W)
    """
    @abc.abstractmethod
    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        return X

    @abc.abstractmethod
    def conj_op(self, S: torch.Tensor) -> torch.Tensor:
        return S

    @abc.abstractmethod
    def projection(self, X: torch.Tensor, S: torch.Tensor, lamda: float) -> torch.Tensor:
        """
        A = M * T
        x <- T^{-1} * (lamda * M * P^{-1}(M) * s + (1 - lamda) * M * T * x + (I - M) * T * x)
        """
        return X

    def log_lh_grad(self, X: torch.Tensor, S: torch.Tensor, lamda: float) -> torch.Tensor:
        """
        grad = -lamda * A'(Ax - s)
        """
        diff = self(X) - S  # (B, C, H_s, W_s)
        grad = -self.conj_op(diff) * lamda

        return grad
