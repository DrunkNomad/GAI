import pickle
import os
import sys
import numpy as np
from ..tensor import Tensor
from ..model import GPT


LOCAL_SYSTEM_PROMPT = (
    "Ты — GAI, интеллектуальный ИИ-агент. "
    "Отвечай по-русски. Будь дружелюбным и полезным."
)


class LocalLLM:
    def __init__(self, model_path=None):
        if model_path is None:
            base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            model_path = os.path.join(base, "gai_model.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Модель не найдена: {model_path}\n"
                f"Обучи модель: python scripts/train_model_final.py"
            )

        with open(model_path, "rb") as f:
            data = pickle.load(f)

        config = data["config"]
        self.model = GPT(
            vocab_size=config["vocab_size"],
            embed_dim=config["embed_dim"],
            num_heads=config["num_heads"],
            num_layers=config["num_layers"],
            ff_dim=config["ff_dim"],
            max_seq_len=config["max_seq_len"],
        )
        for p, (label, weight) in zip(self.model.parameters(), data["model_state"]):
            p.data = weight

        self.tokenizer = data["tokenizer"]
        self.model.tokenizer = self.tokenizer
        self.max_seq_len = config["max_seq_len"]
        self.messages = []

    def chat(self, user_message, tools_enabled=False):
        self.messages.append({"role": "user", "content": user_message})

        prompt = self._build_prompt()
        tokens = self.tokenizer.encode(prompt)

        if len(tokens) > self.max_seq_len - 80:
            tokens = tokens[-(self.max_seq_len - 80):]

        x = np.array([tokens], dtype=np.int64)
        out = self.model.generate(
            Tensor(x, requires_grad=False),
            max_new_tokens=150,
            temperature=0.6,
            top_k=30,
        )
        generated = out[0, len(tokens):].tolist()
        response = self.tokenizer.decode(generated)

        if "\n\n" in response:
            response = response[:response.index("\n\n")].strip()
        if "User:" in response:
            response = response.split("User:")[0].strip()
        if response.startswith("Assistant:"):
            response = response[len("Assistant:"):].strip()

        self.messages.append({"role": "assistant", "content": response})
        return response

    def _build_prompt(self):
        parts = []
        for m in self.messages[-4:]:
            if m["role"] == "user":
                parts.append(f"User: {m['content']}\nAssistant: ")
            else:
                parts.append(f"Assistant: {m['content']}\n")
        return "".join(parts)
