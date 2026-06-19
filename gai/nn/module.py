import torch


class Module(torch.nn.Module):
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def parameters(self, recurse=True):
        return list(super().parameters(recurse))

    def zero_grad(self):
        for p in self.parameters():
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

    def to(self, device):
        return super().to(device)


class Sequential(Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def __getitem__(self, idx):
        return self.layers[idx]

    def __len__(self):
        return len(self.layers)
