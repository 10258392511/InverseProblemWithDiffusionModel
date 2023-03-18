import numpy as np
import torch
import warnings

from scipy.spatial import distance_matrix
from . import i2k_complex, k2i_complex, LinearTransform, generate_mask
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

    # def _generate_mask(self):
    #     # mask: (1, 1, W)
    #     torch.random.manual_seed(self.seed)
    #     C, H, W = self.in_shape
    #     mask = (torch.rand(1, 1, W) <= 1 / self.R).float()
    #     win_size = int(W * self.center_lines_frac)
    #     half_win_size = W // 2
    #     start_idx = half_win_size - win_size // 2
    #     end_idx = start_idx + win_size
    #     mask[..., start_idx:end_idx] = 1.

    #     return mask

    def _generate_mask(self):
        # for (T, 1, H, W) only
        torch.random.manual_seed(self.seed)
        C, H, W = self.in_shape
        # T = 24
        # mask = generate_mask(T, W, sw=0.07, sm=0.3, sa=0.01782)  # (T, 1, W), R = 20
        # mask = mask.unsqueeze(1)  # (T, 1, 1, W)
        T = 1
        mask = generate_mask(T, W, sw=0.07, sm=0.3, sa=0.01782, seed=self.seed)  # (1, W), R = 20

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
        # X, S: (B, C, H, W)
        mask = self.mask.to(X.device)  # (1, 1, W)
        S_from_X = i2k_complex(X)
        S_retrained_mixture = lamda * S + (1 - lamda) * mask * S_from_X
        S_unretrained = (1 - mask) * S_from_X
        X_out = k2i_complex(S_retrained_mixture + S_unretrained)

        return X_out


class SENSE(LinearTransform):
    def __init__(self, sens_type, num_sens, R, center_lines_frac, in_shape, seed):
        assert sens_type in ["exp"]
        self.random_under_fourier = RandomUndersamplingFourier(R, center_lines_frac, in_shape, seed)
        sens_maps = []
        for i in range(num_sens):
            seed = self.random_under_fourier.seed
            if seed is not None:
                seed += i
            sens_maps.append(self._generate_sens_map(sens_type, seed))

        sens_maps = torch.stack(sens_maps, dim=0)  # [(H, W)...] -> (num_sens, H, W)
        normalize_fractor = (torch.abs(sens_maps) ** 2).sum(dim=0)  # (num_sens, H, W) -> (H, W)
        normalize_fractor = torch.sqrt(normalize_fractor)  # (H, W)
        self.sens_maps = sens_maps / normalize_fractor  # (num_sens, H, W)

        energy = (torch.abs(self.sens_maps) ** 2).sum(dim=0)
        assert torch.allclose(energy,  torch.ones_like(energy))

    def _generate_sens_map(self, sens_type, seed=0, **kwargs):
        sens_map = torch.ones(self.random_under_fourier.in_shape)
        if sens_type == "exp":
            # kwargs: l, anchor: np.ndarray
            # exp(- ||x - x0||^2 / (2 * l))
            anchor = kwargs.get("anchor", None)

            if anchor is None:
                H, W = self.random_under_fourier.in_shape[-2:]
                np.random.seed(seed)
                anchor_h, anchor_w = np.random.choice(H), np.random.choice(W)
                anchor = np.array([anchor_h, anchor_w])[None, :]  # (1, 2)
                ww, hh = np.mgrid[0:W, 0:H]  # (H, W) each
                coords = np.stack([ww.flatten(), hh.flatten()], axis=1)  # (HW, 2)
                dist_mat = distance_matrix(coords, anchor, p=2)  # (HW, 1), not squared
                dist_mat_tensor = torch.tensor(dist_mat.reshape((H, W)))  # (H, W)
                l = kwargs.get("l", dist_mat.max() / 2)
                sens_map = torch.exp(- dist_mat_tensor / (2 * l))

        return sens_map

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        # X: (B, C, H, W)
        S = []
        sens_maps = self.sens_maps.to(X.device)
        for i in range(sens_maps.shape[0]):
            sens_map_iter = sens_maps[i]  # (H, W)
            S.append(self.random_under_fourier(sens_map_iter * X))

        S = torch.stack(S, dim=0)  # [(B, C, H, W)...] -> (num_sens, B, C, H, W)

        return S

    def conj_op(self, S: torch.Tensor) -> torch.Tensor:
        # S: (num_sens, B, C, H, W)
        sens_maps = self.sens_maps.to(S.device)
        X_out = torch.zeros(S.shape[1:], dtype=S.dtype).to(S.device)  # (B, C, H, W)
        for i in range(S.shape[0]):
            X_out += sens_maps[i].conj() * self.random_under_fourier.conj_op(S[i])

        # (B, C, H, W)
        return X_out

    def SSOS(self, S: torch.Tensor) -> torch.Tensor:
        # S: (num_sens, B, C, H, W)
        X_out = torch.zeros(S.shape[1:], dtype=torch.float32).to(S.device)  # (B, C, H, W)
        for i in range(S.shape[0]):
            X_out += torch.abs(self.random_under_fourier.conj_op(S[i])) ** 2

        X_out = torch.sqrt(X_out)

        return X_out

    def projection(self, X: torch.Tensor, S: torch.Tensor, lamda: float) -> torch.Tensor:
        warnings.warn("Not implemented!")
        X_proj = super(SENSE, self).projection(X, S)

        return X_proj
