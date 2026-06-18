class SGD:
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocities = [None for _ in self.params]

    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            grad = p.grad + self.weight_decay * p.data
            if self.momentum > 0:
                if self.velocities[i] is None:
                    self.velocities[i] = grad
                else:
                    self.velocities[i] = self.momentum * self.velocities[i] + grad
                update = self.velocities[i]
            else:
                update = grad
            p.data -= self.lr * update

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()
