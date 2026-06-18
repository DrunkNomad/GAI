import pickle
from pathlib import Path

import numpy as np

from gai.model.gpt import GPT
from gai.tensor import Tensor


class CustomGPTGenerator:
    def __init__(self, model_path: Path, temperature: float = 0.6, top_k: int = 30) -> None:
        self.temperature = temperature
        self.top_k = top_k
        self.model, self.tokenizer, self.config = self._load(model_path)

    def _load(self, path: Path) -> tuple[GPT, object, dict]:
        with open(path, "rb") as f:
            data = pickle.load(f)

        config = data["config"]
        model = GPT(
            vocab_size=config["vocab_size"],
            embed_dim=config["embed_dim"],
            num_heads=config["num_heads"],
            num_layers=config["num_layers"],
            ff_dim=config["ff_dim"],
            max_seq_len=config["max_seq_len"],
            dropout=0.0,
        )
        model.eval()

        loaded_params = dict(data["model_state"])
        for param in model.parameters():
            if param.label in loaded_params:
                param.data = loaded_params[param.label]

        tokenizer = data["tokenizer"]
        return model, tokenizer, config

    def generate(self, prompt: str, max_new_tokens: int = 128) -> str:
        tokens = self.tokenizer.encode(prompt)
        seq_len = self.config["max_seq_len"]
        x = np.array([tokens[-seq_len:]], dtype=np.int64)
        out = self.model.generate(
            Tensor(x, requires_grad=False),
            max_new_tokens=max_new_tokens,
            temperature=self.temperature,
            top_k=self.top_k,
        )
        generated_ids = out[0, x.shape[1]:].tolist()
        return self.tokenizer.decode(generated_ids)

    @property
    def model_name(self) -> str:
        cfg = self.config
        return f"custom-gpt-{cfg['embed_dim']}dim-{cfg['num_layers']}l"
