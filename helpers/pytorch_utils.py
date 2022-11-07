import torch

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# DEVICE = torch.device("cpu")

def to_numpy(X: torch.Tensor):

    return X.detach().cpu().numpy()
