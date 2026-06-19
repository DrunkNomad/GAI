import torch
from .module import Module


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(num_embeddings, embedding_dim)
        torch.nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)

    def forward(self, x):
        if x.dtype not in (torch.long, torch.int):
            x = x.long()
        return self.embedding(x)

    def __repr__(self):
        return f"Embedding({self.embedding.num_embeddings}, {self.embedding.embedding_dim})"
