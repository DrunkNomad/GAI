import torch
import torch.nn.functional as F
from .module import Module
from .linear import Linear
from .dropout import Dropout


class MultiHeadAttention(Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, causal=True):
        super().__init__()
        assert embed_dim % num_heads == 0
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

        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)
        k = self.k_proj(x).view(B, T, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, D).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * (D ** -0.5)

        if self.causal:
            causal_mask = torch.triu(
                torch.full((T, T), float("-inf"), device=x.device, dtype=x.dtype), diagonal=1
            )
            attn = attn + causal_mask

        if mask is not None:
            attn = attn + mask

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(out)
        return out
