import numpy as np
from ..tensor import Tensor
from .module import Module


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = Tensor(
            np.random.randn(num_embeddings, embedding_dim) * 0.01,
            requires_grad=True,
        )

    def forward(self, x):
        idx = x.data.astype(np.int64) if isinstance(x, Tensor) else np.asarray(x, dtype=np.int64)
        out = Tensor(self.weight.data[idx], requires_grad=self.weight.requires_grad)
        if out.requires_grad:
            class EmbeddingFn:
                @staticmethod
                def backward(depends_on, grad):
                    w, _ = depends_on[0]
                    w_grad = np.zeros_like(w.data)
                    np.add.at(w_grad, idx, grad)
                    return [w_grad]
            out._depends_on = [(self.weight, {})]
            out._creation_op = EmbeddingFn
        return out

    def __repr__(self):
        return f"Embedding({self.num_embeddings}, {self.embedding_dim})"
