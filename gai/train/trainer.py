import time
import numpy as np
from ..tensor import Tensor


class Trainer:
    def __init__(self, model, optimizer, dataset, tokenizer):
        self.model = model
        self.optimizer = optimizer
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.history = {"loss": []}

    def train_step(self, batch_size):
        xs, ys = self.dataset.get_batch(batch_size)
        x = Tensor(xs, requires_grad=False)
        y = Tensor(ys, requires_grad=False)
        _, loss = self.model(x, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train(self, steps=1000, batch_size=16, log_every=100, eval_every=None):
        start = time.time()
        for step in range(steps):
            loss_val = self.train_step(batch_size)
            self.history["loss"].append(loss_val)

            if step % log_every == 0:
                elapsed = time.time() - start
                print(f"Step {step}/{steps} | loss: {loss_val:.4f} | time: {elapsed:.1f}s")

            if eval_every and step % eval_every == 0 and step > 0:
                self.generate_sample()

        print(f"Training done in {time.time() - start:.1f}s. Final loss: {loss_val:.4f}")

    def generate_sample(self, prompt="", max_tokens=50, temperature=1.0):
        if prompt:
            tokens = self.tokenizer.encode(prompt)
            x = Tensor(np.array([tokens], dtype=np.int64), requires_grad=False)
        else:
            x = Tensor(np.zeros((1, 1), dtype=np.int64), requires_grad=False)
        out = self.model.generate(x, max_tokens, temperature)
        text = self.tokenizer.decode(out[0, x.shape[1]:].tolist())
        print(f"Sample: {prompt}{text}")
        return text
