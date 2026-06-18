import numpy as np
from ..tensor import Tensor
from ..nn.module import Module
from ..nn.embedding import Embedding
from ..nn.linear import Linear
from ..nn.normalization import LayerNorm
from ..nn.dropout import Dropout
from ..nn.transformer import TransformerBlock


class GPT(Module):
    def __init__(self, vocab_size, embed_dim=256, num_heads=4, num_layers=4, ff_dim=512, max_seq_len=512, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        self.token_embedding = Embedding(vocab_size, embed_dim)
        self.pos_embedding = Embedding(max_seq_len, embed_dim)
        self.dropout = Dropout(dropout)

        self.blocks = [TransformerBlock(embed_dim, num_heads, ff_dim, dropout, causal=True) for _ in range(num_layers)]
        for i, block in enumerate(self.blocks):
            setattr(self, f"block_{i}", block)

        self.ln_f = LayerNorm(embed_dim)
        self.lm_head = Linear(embed_dim, vocab_size, bias=False)

    def forward(self, x, targets=None):
        B, T = x.shape
        assert T <= self.max_seq_len, f"Sequence length {T} exceeds max_seq_len {self.max_seq_len}"

        pos = Tensor(np.arange(T, dtype=np.int64), requires_grad=False)
        tok_emb = self.token_embedding(x)
        pos_emb = self.pos_embedding(pos)
        x = tok_emb + pos_emb
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            from ..nn.loss import CrossEntropyLoss
            loss_fn = CrossEntropyLoss()
            loss = loss_fn(logits.reshape(-1, self.vocab_size), targets.reshape(-1))

        return logits, loss

    def generate(self, idx, max_new_tokens=100, temperature=1.0, top_k=None):
        self.eval()
        if isinstance(idx, Tensor):
            idx_np = idx.data.astype(np.int64)
        else:
            idx_np = np.asarray(idx, dtype=np.int64)
        for _ in range(max_new_tokens):
            idx_cond = idx_np if idx_np.shape[1] <= self.max_seq_len else idx_np[:, -self.max_seq_len:]
            logits, _ = self.forward(Tensor(idx_cond, requires_grad=False))
            logits_np = logits.data[:, -1, :] / temperature

            if top_k is not None:
                top_k_vals = np.sort(logits_np)[:, -top_k]
                logits_np = np.where(logits_np < top_k_vals[:, None], -float("inf"), logits_np)

            probs = np.exp(logits_np - logits_np.max(axis=-1, keepdims=True))
            probs = probs / probs.sum(axis=-1, keepdims=True)
            next_token = np.array([[np.random.choice(probs.shape[1], p=probs[0])]], dtype=np.int64)
            idx_np = np.concatenate([idx_np, next_token], axis=1)
        return idx_np

    def __repr__(self):
        total = sum(p.data.size for p in self.parameters())
        return f"GPT(vocab_size={self.vocab_size}, embed_dim={self.embed_dim}, params={total:,})"
