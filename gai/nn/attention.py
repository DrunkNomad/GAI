import numpy as np
from ..tensor import Tensor
from .module import Module
from .linear import Linear
from .dropout import Dropout


class MultiHeadAttention(Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, causal=True):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.causal = causal

        self.q_proj = Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = Linear(embed_dim, embed_dim, bias=False)
        self.dropout = Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, C = x.shape
        H = self.num_heads
        D = self.head_dim

        q = self.q_proj(x).reshape(B, T, H, D).transpose((0, 2, 1, 3))
        k = self.k_proj(x).reshape(B, T, H, D).transpose((0, 2, 1, 3))
        v = self.v_proj(x).reshape(B, T, H, D).transpose((0, 2, 1, 3))

        score = q @ k.transpose((0, 1, 3, 2)) / np.sqrt(D)

        if self.causal:
            causal_mask = np.triu(np.ones((T, T), dtype=np.float32) * -1e9, k=1)
            score = score + Tensor(causal_mask, requires_grad=False)

        if mask is not None:
            score = score + mask

        attn = Tensor(
            np.exp(score.data - score.data.max(axis=-1, keepdims=True)) / (
                np.exp(score.data - score.data.max(axis=-1, keepdims=True)).sum(axis=-1, keepdims=True) + 1e-10
            ),
            requires_grad=score.requires_grad,
        )
        if attn.requires_grad:
            class AttnFn:
                @staticmethod
                def backward(depends_on, grad):
                    s, _ = depends_on[0]
                    a = attn.data
                    return [a * (grad - (a * grad).sum(axis=-1, keepdims=True))]
            attn._depends_on = [(score, {})]
            attn._creation_op = AttnFn

        attn = self.dropout(attn)
        out = (attn @ v).transpose((0, 2, 1, 3)).reshape(B, T, C)
        out = self.out_proj(out)
        return out
