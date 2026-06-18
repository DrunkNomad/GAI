from ..tensor import Tensor


class Module:
    def __init__(self):
        self._parameters = {}
        self._modules = {}
        self._buffers = {}

    def __setattr__(self, name, value):
        if isinstance(value, Module):
            self._modules[name] = value
        elif isinstance(value, Tensor) and value.requires_grad:
            self._parameters[name] = value
        super().__setattr__(name, value)

    def parameters(self):
        params = []
        for p in self._parameters.values():
            params.append(p)
        for m in self._modules.values():
            params.extend(m.parameters())
        return params

    def named_parameters(self, prefix=""):
        for name, p in self._parameters.items():
            yield f"{prefix}.{name}" if prefix else name, p
        for name, m in self._modules.items():
            yield from m.named_parameters(prefix=f"{prefix}.{name}" if prefix else name)

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def train(self):
        self.training = True
        for m in self._modules.values():
            m.train()
        return self

    def eval(self):
        self.training = False
        for m in self._modules.values():
            m.eval()
        return self

    def to(self, device):
        return self


class Sequential(Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = layers
        for i, layer in enumerate(layers):
            setattr(self, f"_{i}", layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def __getitem__(self, idx):
        return self.layers[idx]

    def __len__(self):
        return len(self.layers)
