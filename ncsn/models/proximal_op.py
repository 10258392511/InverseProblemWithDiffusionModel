import numpy as np
import warnings
import torch
import InverseProblemWithDiffusionModel.helpers.pytorch_utils as ptu

from scipy.sparse.linalg import cg
from InverseProblemWithDiffusionModel.ncsn.linear_transforms import LinearTransform, i2k_complex, k2i_complex
from InverseProblemWithDiffusionModel.ncsn.linear_transforms.undersampling_fourier import RandomUndersamplingFourier


class Proximal(object):
    def __init__(self, lin_tfm: LinearTransform):
        self.lin_tfm = lin_tfm

    def __call__(self, *args, **kwargs):
        pass


class L2Penalty(Proximal):
    def __call__(self, z, y, alpha, lamda, num_steps=10):
        """
        x <- argmin_x 1 / 2 * norm(x - z)^2 + 1 / 2 * alpha / lamda * norm(Ax - y)

        y: y_0 + sigma_k * A * eps_k
        """
        def loss(x):
            # x: (B, C, H, W)
            data_error = 0.5 * (torch.abs(x - z) ** 2).sum(dim=(1, 2, 3)).mean()
            reg = 0.5 * alpha / lamda * (torch.abs(self.lin_tfm(x) - y) ** 2).sum(dim=(1, 2, 3)).mean()
            loss_val = data_error + reg

            return loss_val

        x_sol = z.clone()
        # x_sol = torch.randn_like(z)
        x_sol.requires_grad = True
        # opt = torch.optim.LBFGS([x_sol])
        opt = torch.optim.Adam([x_sol], lr=5e-1)

        for _ in range(num_steps):
            def closure():
                opt.zero_grad()
                loss_val = loss(x_sol)
                loss_val.backward()
                # print(loss_val.item())

                return loss_val
            opt.step(closure)

        return x_sol.detach()

    @torch.no_grad()
    def check_solution(self, x_sol, z, y, alpha, lamda):
        warnings.warn("For testing only, don't use this in iterations.")
        b = z + alpha / lamda * self.lin_tfm.conj_op(y)
        lhs = x_sol + alpha / lamda * self.lin_tfm.conj_op(self.lin_tfm(x_sol))

        return (torch.abs(lhs - b) ** 2).sum(dim=(1, 2, 3)).mean()


class Constrained(Proximal):
    """
    Proximal operator from Yang et al (MRI).
    """
    def __call__(self, X: torch.Tensor, S: torch.Tensor, lamda: float):
        X_out = self.lin_tfm.projection(X, S, lamda)

        return X_out


class SingleCoil(Proximal):
    def __init__(self, lin_tfm: RandomUndersamplingFourier):
        super(SingleCoil, self).__init__(lin_tfm)
        assert isinstance(self.lin_tfm, RandomUndersamplingFourier), "only supporting RandomUnversamplingFourier"

    def __call__(self, z, y, alpha, lamda):
        """
        Closed-form solution of
        x <- argmin_x 1 / 2 * norm(x - z)^2 + 1 / 2 * alpha / lamda * norm(Ax - y)
        x = F' diag(1 / (1 + alpha * M_{ii})) F (z + alpha F'y)
        """
        alpha = alpha / lamda
        print(f"alpha: {alpha}")
        mask = self.lin_tfm.mask.to(z.device)
        x_out = z + alpha * k2i_complex(y)
        x_out = i2k_complex(x_out)
        mask_inv = 1 / (1 + mask * alpha)
        x_out = mask_inv * x_out
        x_out = k2i_complex(x_out)

        # print(f"diff: {torch.norm(x_out - z)}")

        return x_out

    @torch.no_grad()
    def check_solution(self, x_out, z, y, alpha, lamda):
        warnings.warn("For testing only, don't use this in iterations.")
        alpha = alpha / lamda
        lhs = x_out + alpha * self.lin_tfm.conj_op(self.lin_tfm(x_out))
        rhs = alpha * self.lin_tfm.conj_op(y) + z

        # x_out: (B, C, H, W)
        return (torch.abs(lhs - rhs) ** 2).sum(dim=(1, 2, 3)).mean()


def get_proximal(proximal_name: str):
    assert proximal_name in ["L2Penalty", "Constrained", "SingleCoil"]
    if proximal_name  == "L2Penalty":
        return L2Penalty

    elif proximal_name == "Constrained":
        return Constrained

    elif proximal_name == "SingleCoil":
        return SingleCoil

    else:
        raise NotImplementedError
