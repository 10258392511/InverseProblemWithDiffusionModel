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

    def log_lh_grad(self, X: torch.Tensor, S: torch.Tensor, lamda: float = 1.) -> torch.Tensor:
        """
        grad = -lamda * A'(Ax - s)
        """
        diff = self(X) - S  # (B, C, H_s, W_s)
        grad = -self.conj_op(diff) * lamda

        return grad


def i2k_complex(X):
    """
    X: (B, C, D, H, W) or (B, C, H, W)
    """
    X = X.to(torch.complex64)
    X = torch.fft.ifftshift(X, dim=[-1, -2])
    X_k_space = torch.fft.fftn(X, dim=[-1, -2], norm="ortho")
    X_k_space_shifted = torch.fft.fftshift(X_k_space, dim=[-1, -2])

    return X_k_space_shifted


def k2i_complex(X):
    """
    X: (B, C, D, H, W) or (B, C, H, W)
    """
    X = X.to(torch.complex64)
    X_i_shifted = torch.fft.ifftshift(X, dim=[-1, -2])
    X_img = torch.fft.ifftn(X_i_shifted, dim=[-1, -2], norm="ortho")
    X_img = torch.fft.fftshift(X_img, dim=[-1, -2])

    return X_img
