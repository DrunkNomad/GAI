"""Test v5 - save to UTF-8 file."""
import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gai.model import GPT

model, tokenizer, config = GPT.load_pickle("gai_model_v5_max.pkl")
print(f"Model: {model}, Vocab: {len(tokenizer)}")

prompts = [
    "Привет",
    "Что такое нейросеть?",
    "Искусственный интеллект",
    "Москва — столица",
    "Вторая мировая война",
    "Россия — это",
    "Кошка",
]

results = []
with torch.no_grad():
    for prompt in prompts:
        tokens = tokenizer.encode(prompt)
        x = torch.tensor([tokens[:200]], dtype=torch.long)
        out = model.generate(x, max_new_tokens=150, temperature=0.6, top_k=30)
        gen = out[0, len(tokens):].tolist()
        result = tokenizer.decode(gen)
        if "\n\n" in result:
            result = result[:result.index("\n\n")]
        results.append((prompt, result))

with open("test_v5_output.txt", "w", encoding="utf-8") as f:
    for p, r in results:
        f.write(f"Q: {p}\nA: {r}\n\n---\n\n")

print(f"Saved to test_v5_output.txt ({os.path.getsize('test_v5_output.txt')} bytes)")
