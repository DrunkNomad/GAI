from .module import Module
from .linear import Linear
from .normalization import LayerNorm
from .dropout import Dropout
from .activation import GELU
from .attention import MultiHeadAttention


class FeedForward(Module):
    def __init__(self, embed_dim, ff_dim, dropout=0.0):
        super().__init__()
        self.fc1 = Linear(embed_dim, ff_dim)
        self.act = GELU()
        self.fc2 = Linear(ff_dim, embed_dim)
        self.dropout = Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(self.act(self.fc1(x))))


class TransformerBlock(Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.0, causal=True):
        super().__init__()
        self.ln1 = LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout, causal)
        self.ln2 = LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim, ff_dim, dropout)
        self.dropout = Dropout(dropout)

    def forward(self, x, mask=None):
        x = x + self.dropout(self.attn(self.ln1(x), mask))
        x = x + self.dropout(self.ff(self.ln2(x)))
        return x
