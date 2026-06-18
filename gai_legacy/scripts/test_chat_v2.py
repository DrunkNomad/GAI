import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import numpy as np
from gai.tensor import Tensor
from gai.model import GPT


class CharTokenizer:
    def __init__(self):
        self.char_to_id = {}
        self.id_to_char = {}

    def train(self, texts):
        chars = set()
        for text in texts:
            chars.update(text)
        chars = sorted(chars)
        self.char_to_id = {c: i for i, c in enumerate(chars)}
        self.id_to_char = {i: c for i, c in enumerate(chars)}

    def encode(self, text):
        return [self.char_to_id.get(c, 0) for c in text]

    def decode(self, ids):
        return "".join(self.id_to_char.get(i, "") for i in ids)

    def __len__(self):
        return len(self.char_to_id)


with open("gai_model_v2.pkl", "rb") as f:
    data = pickle.load(f)

config = data["config"]
model = GPT(
    vocab_size=config["vocab_size"],
    embed_dim=config["embed_dim"],
    num_heads=config["num_heads"],
    num_layers=config["num_layers"],
    ff_dim=config["ff_dim"],
    max_seq_len=config["max_seq_len"],
)
for p, (label, weight) in zip(model.parameters(), data["model_state"]):
    p.data = weight

tokenizer = data["tokenizer"]

print(f"Модель загружена: {config}")
print(f"Словарь: {len(tokenizer)} символов")
print()

while True:
    try:
        user = input("Вы: ")
    except (EOFError, KeyboardInterrupt):
        break
    if user.lower() in ("/exit", "/quit"):
        break
    if not user.strip():
        continue

    prompt = f"User: {user}\nAssistant: "
    tokens = tokenizer.encode(prompt)
    x = np.array([tokens[:config["max_seq_len"]]], dtype=np.int64)
    out = model.generate(
        Tensor(x, requires_grad=False),
        max_new_tokens=80,
        temperature=0.7,
        top_k=30,
    )
    generated = out[0, len(tokens):].tolist()
    response = tokenizer.decode(generated)
    if "\n\n" in response:
        response = response[:response.index("\n\n")]
    if "User:" in response:
        response = response.split("User:")[0].strip()

    print(f"GAI: {response}")
