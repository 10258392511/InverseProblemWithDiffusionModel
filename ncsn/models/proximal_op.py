import numpy as np
import warnings
import torch
import InverseProblemWithDiffusionModel.helpers.pytorch_utils as ptu

from scipy.sparse.linalg import cg
from InverseProblemWithDiffusionModel.ncsn.linear_transforms import LinearTransform


class Proximal(object):
    def __init__(self, lin_tfm: LinearTransform):
        self.lin_tfm = lin_tfm

    def __call__(self, *args, **kwargs):
        pass


class L2Penalty(Proximal):
    def __call__(self, z, y, alpha, lamda, num_steps=200):
        """
        x <- argmin_x 1 / 2 * norm(x - z)^2 + 1 / 2 * alpha / lamda * norm(Ax - y)

        y: y_0 + sigma_k * A * eps_k
        """
        # def A_op(x):
        #     out = x + alpha / lamda * self.lin_tfm.conj_op(self.lin_tfm(x))
        #
        #     return out
        #
        # b = z + alpha / lamda * self.lin_tfm.conj_op(y)
        # x, exit_code = cg(A_op, ptu.to_numpy(b))
        # x = torch.tensor(x).to(z.device)
        #
        # if exit_code != 0:
        #     warnings.warn(f"CG not successful: with exit code {exit_code}")
        #
        # return x

        # keep z as real image for now
        # z = z.to(torch.complex64): accurate solution but still needs complex2mag

        # alpha = 1 / alpha  # empirical obs: reg strength should increase as number of iterations increases
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
        # alpha = 1 / alpha
        b = z + alpha / lamda * self.lin_tfm.conj_op(y)
        lhs = x_sol + alpha / lamda * self.lin_tfm.conj_op(self.lin_tfm(x_sol))
        # b = self.lin_tfm.conj_op(y)
        # lhs = self.lin_tfm.conj_op(self.lin_tfm(x_sol))

        return (torch.abs(lhs - b) ** 2).sum(dim=(1, 2, 3)).mean()


class Constrained(Proximal):
    """
    Proximal operator from Yang et al (MRI).
    """
    def __call__(self, X: torch.Tensor, S: torch.Tensor, lamda: float):
        X_out = self.lin_tfm.projection(X, S, lamda)

        return X_out


def get_proximal(proximal_name: str):
    assert proximal_name in ["L2Penalty", "Constrained"]
    if proximal_name  == "L2Penalty":
        return L2Penalty

    elif proximal_name == "Constrained":
        return Constrained

    else:
        raise NotImplementedError
