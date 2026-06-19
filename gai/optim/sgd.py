import torch


class SGD:
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        self.params = list(params)
        self.optim = torch.optim.SGD(self.params, lr=lr, momentum=momentum, weight_decay=weight_decay)

    def step(self):
        self.optim.step()

    def zero_grad(self):
        self.optim.zero_grad()
