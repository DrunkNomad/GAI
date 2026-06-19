from .module import Module


class LayerNorm(Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()
        import torch
        shape = normalized_shape if isinstance(normalized_shape, tuple) else (normalized_shape,)
        self.ln = torch.nn.LayerNorm(shape, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, x):
        return self.ln(x)
