import torch
import numpy as np

Tensor = torch.Tensor


def tensor(data, requires_grad=False, dtype=None):
    if isinstance(data, torch.Tensor):
        return data.detach().requires_grad_(requires_grad)
    if dtype is None:
        dtype = torch.float32 if requires_grad else None
    return torch.tensor(np.asarray(data), requires_grad=requires_grad, dtype=dtype)
