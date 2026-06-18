import numpy as np
from ..tensor import Tensor
from .module import Module


class MSELoss(Module):
    def forward(self, pred, target):
        diff = pred - target
        return (diff ** 2).mean()


class CrossEntropyLoss(Module):
    def forward(self, pred, target):
        if isinstance(target, Tensor):
            target = target.data
        target = np.asarray(target, dtype=np.int64)
        batch_size = pred.shape[0]
        pred_exp = np.exp(pred.data - pred.data.max(axis=1, keepdims=True))
        softmax = pred_exp / pred_exp.sum(axis=1, keepdims=True)
        log_softmax = np.log(softmax + 1e-10)
        loss_val = -log_softmax[np.arange(batch_size), target].mean()

        loss = Tensor(loss_val, requires_grad=pred.requires_grad)
        if loss.requires_grad:
            class CrossEntropyFn:
                @staticmethod
                def backward(depends_on, grad):
                    p, _ = depends_on[0]
                    s = softmax.copy()
                    s[np.arange(batch_size), target] -= 1.0
                    return [s / batch_size * grad]
            loss._depends_on = [(pred, {})]
            loss._creation_op = CrossEntropyFn
        return loss
