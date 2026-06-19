import pickle
from pathlib import Path

import torch
import numpy as np

from gai.model.gpt import GPT


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
            num_heads=config.get("num_heads", 4),
            num_layers=config.get("num_layers", 4),
            ff_dim=config.get("ff_dim", config["embed_dim"] * 4),
            max_seq_len=config["max_seq_len"],
            dropout=0.0,
        )
        model.eval()

        state = data["model_state"]
        if isinstance(state, list):
            state_dict = {}
            for i, (label, weight) in enumerate(state):
                state_dict[f"_list_param_{i}"] = torch.from_numpy(np.asarray(weight))
            model.load_state_dict(state_dict, strict=False)
        else:
            torch_state = {k: torch.from_numpy(np.asarray(v)) for k, v in state.items()}
            model.load_state_dict(torch_state)

        tokenizer = data["tokenizer"]
        return model, tokenizer, config

    def generate(self, prompt: str, max_new_tokens: int = 128) -> str:
        tokens = self.tokenizer.encode(prompt)
        seq_len = self.config["max_seq_len"]
        x = torch.tensor([tokens[-seq_len:]], dtype=torch.long)
        out = self.model.generate(
            x,
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
