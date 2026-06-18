import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gai import Tensor
from gai.nn import Linear, CrossEntropyLoss, Sequential, ReLU
from gai.optim import SGD
import numpy as np


def test_autograd():
    print("=== Test 1: Basic autograd ===")
    a = Tensor([2.0, 3.0], requires_grad=True)
    b = Tensor([1.0, 4.0], requires_grad=True)
    c = a * b + a
    c.backward()
    assert np.allclose(a.grad, [b.data[0] + 1, b.data[1] + 1]), f"a.grad = {a.grad}"
    assert np.allclose(b.grad, a.data), f"b.grad = {b.grad}"
    print("  Basic ops: OK")


def test_linear():
    print("=== Test 2: Linear layer ===")
    layer = Linear(4, 3)
    x = Tensor(np.random.randn(2, 4), requires_grad=False)
    out = layer(x)
    out.mean().backward()
    assert out.shape == (2, 3)
    assert layer.weight.grad is not None
    print("  Linear: OK")


def test_mse():
    print("=== Test 3: MSELoss ===")
    from gai.nn import MSELoss
    pred = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    target = Tensor([1.5, 2.5, 3.5])
    loss = MSELoss()
    l = loss(pred, target)
    l.backward()
    assert l.item() > 0
    assert pred.grad is not None
    print("  MSE Loss: OK")


def test_sequential():
    print("=== Test 4: Sequential model ===")
    model = Sequential(
        Linear(10, 20),
        ReLU(),
        Linear(20, 5),
    )
    x = Tensor(np.random.randn(3, 10), requires_grad=False)
    out = model(x)
    out.mean().backward()
    assert out.shape == (3, 5)
    print("  Sequential: OK")


def test_optimizer():
    print("=== Test 5: SGD optimizer ===")
    layer = Linear(2, 1)
    x = Tensor([[1.0, 2.0]], requires_grad=False)
    y = Tensor([[5.0]], requires_grad=False)
    optim = SGD(layer.parameters(), lr=0.01)
    pred = layer(x)
    loss = ((pred - y) ** 2).mean()
    optim.zero_grad()
    loss.backward()
    optim.step()
    print("  SGD step: OK")


def test_gpt_init():
    print("=== Test 6: GPT initialization ===")
    from gai.model import GPT
    model = GPT(vocab_size=100, embed_dim=32, num_heads=2, num_layers=2, ff_dim=64, max_seq_len=16)
    x = np.array([[1, 2, 3, 4, 5]], dtype=np.int64)
    from gai import Tensor
    x_t = Tensor(x, requires_grad=False)
    logits, loss = model(x_t, x_t)
    assert logits.shape == (1, 5, 100)
    print(f"  GPT: logits shape {logits.shape}, loss={loss.item():.4f}")


def test_generate():
    print("=== Test 7: GPT generation ===")
    from gai.model import GPT
    model = GPT(vocab_size=100, embed_dim=32, num_heads=2, num_layers=2, ff_dim=64, max_seq_len=16)
    x = np.array([[1, 2, 3]], dtype=np.int64)
    from gai import Tensor
    x_t = Tensor(x, requires_grad=False)
    out = model.generate(x_t, max_new_tokens=10, temperature=1.0)
    assert out.shape == (1, 13)
    print(f"  Generation: shape {out.shape}")


if __name__ == "__main__":
    test_autograd()
    test_linear()
    test_mse()
    test_sequential()
    test_optimizer()
    test_gpt_init()
    test_generate()
    print("\nAll tests passed!")
