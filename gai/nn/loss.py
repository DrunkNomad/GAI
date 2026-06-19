import torch
import torch.nn.functional as F
from .module import Module


class MSELoss(Module):
    def forward(self, pred, target):
        return F.mse_loss(pred, target)


class CrossEntropyLoss(Module):
    def forward(self, pred, target):
        if target.dtype not in (torch.long, torch.int):
            target = target.long()
        return F.cross_entropy(pred, target)
