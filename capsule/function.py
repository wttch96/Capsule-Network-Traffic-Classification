import torch


def squash(x: torch.Tensor, dim: int = -1):
    # p=2: L2 范式
    norm = torch.norm(x, p=2, dim=dim, keepdim=True)
    norm_p = norm ** 2
    scale = norm_p / (1 + norm_p) / (norm + 1e-8)
    return x * scale
