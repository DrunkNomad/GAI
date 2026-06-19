import torch


class Adam:
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        self.params = list(params)
        self.optim = torch.optim.Adam(self.params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

    def step(self):
        self.optim.step()

    def zero_grad(self):
        self.optim.zero_grad()
