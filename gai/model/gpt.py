import torch
import torch.nn.functional as F
from ..nn.module import Module
from ..nn.embedding import Embedding
from ..nn.linear import Linear
from ..nn.normalization import LayerNorm
from ..nn.dropout import Dropout
from ..nn.transformer import TransformerBlock
from ..tensor import tensor


class GPT(Module):
    def __init__(self, vocab_size, embed_dim=256, num_heads=4, num_layers=4, ff_dim=512, max_seq_len=512, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        self.token_embedding = Embedding(vocab_size, embed_dim)
        self.pos_embedding = Embedding(max_seq_len, embed_dim)
        self.dropout = Dropout(dropout)

        self.blocks = torch.nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout, causal=True)
            for _ in range(num_layers)
        ])

        self.ln_f = LayerNorm(embed_dim)
        self.lm_head = Linear(embed_dim, vocab_size, bias=False)

    def forward(self, x, targets=None):
        B, T = x.shape
        assert T <= self.max_seq_len

        pos = torch.arange(T, device=x.device, dtype=torch.long)
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
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))

        return logits, loss

    def generate(self, idx, max_new_tokens=100, temperature=1.0, top_k=None):
        was_training = self.training
        self.eval()
        with torch.no_grad():
            if isinstance(idx, torch.Tensor):
                idx_np = idx.clone()
            else:
                idx_np = torch.tensor(idx, dtype=torch.long)
            for _ in range(max_new_tokens):
                idx_cond = idx_np if idx_np.shape[1] <= self.max_seq_len else idx_np[:, -self.max_seq_len:]
                logits, _ = self.forward(idx_cond)
                logits_np = logits[:, -1, :] / temperature
                if top_k is not None:
                    top_k_vals = torch.topk(logits_np, top_k, dim=-1).values[:, -1:]
                    logits_np = torch.where(logits_np < top_k_vals, float("-inf"), logits_np)
                probs = F.softmax(logits_np, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                idx_np = torch.cat([idx_np, next_token], dim=1)
        if was_training:
            self.train()
        return idx_np

    def __repr__(self):
        total = sum(p.numel() for p in self.parameters())
        return f"GPT(vocab_size={self.vocab_size}, embed_dim={self.embed_dim}, params={total:,})"

    def save_pickle(self, path, tokenizer):
        import pickle
        state = {k: v.cpu().numpy() for k, v in self.state_dict().items()}
        data = {
            "model_state": state,
            "tokenizer": tokenizer,
            "config": {
                "vocab_size": self.vocab_size,
                "embed_dim": self.embed_dim,
                "num_heads": self.blocks[0].attn.num_heads,
                "num_layers": len(self.blocks),
                "ff_dim": self.embed_dim * 4,
                "max_seq_len": self.max_seq_len,
            }
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load_pickle(cls, path):
        import pickle
        import numpy as np
        with open(path, "rb") as f:
            data = pickle.load(f)
        config = data["config"]
        model = cls(
            vocab_size=config["vocab_size"],
            embed_dim=config["embed_dim"],
            num_heads=config.get("num_heads", 4),
            num_layers=config.get("num_layers", 4),
            ff_dim=config.get("ff_dim", config["embed_dim"] * 4),
            max_seq_len=config["max_seq_len"],
            dropout=0.0,
        )
        state = {k: torch.from_numpy(np.asarray(v)) for k, v in data["model_state"].items()}
        model.load_state_dict(state)
        model.eval()
        return model, data["tokenizer"], config
