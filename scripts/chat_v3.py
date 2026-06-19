import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from gai.model import GPT

model, tokenizer, config = GPT.load_pickle("gai_model_v3_max.pkl")
print("GAI готов. Напиши exit для выхода.")

while True:
    prompt = input("You: ")
    if prompt.lower() in ("exit", "quit"):
        break
    tokens = tokenizer.encode(f"User: {prompt}\nAssistant: ")
    x = torch.tensor([tokens[-config["max_seq_len"]:]], dtype=torch.long)
    out = model.generate(x, max_new_tokens=128, temperature=0.6, top_k=30)
    result = tokenizer.decode(out[0, len(tokens) :].tolist())
    if "\n\n" in result:
        result = result[: result.index("\n\n")]
    print(f"GAI: {result}")
