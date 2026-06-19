"""Test fine-tuned chat model."""
import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gai.model import GPT

m, tok, cfg = GPT.load_pickle("gai_model_v5_chat_max.pkl")
m.eval()
print(f"Model: {sum(p.numel() for p in m.parameters()):,} params, vocab={len(tok)}")

test_qs = [
    "Привет", "Кто ты?", "Что такое нейросеть?",
    "Столица России?", "Расскажи шутку", "Спасибо",
    "Пока", "Что такое Python?", "Как дела?",
    "Что такое трансформер?", "Что ты умеешь?",
    "В чём смысл жизни?", "Напиши функцию на Python",
]

results = []
with torch.no_grad():
    for q in test_qs:
        ids = tok.encode(f"{q} ")
        x = torch.tensor([ids[:200]], dtype=torch.long)
        out = m.generate(x, max_new_tokens=150, temperature=0.6, top_k=30)
        gen = out[0, len(ids):].tolist()
        result = tok.decode(gen)
        if "\n\n" in result:
            result = result[:result.index("\n\n")]
        results.append((q, result))

with open("test_chat_output.txt", "w", encoding="utf-8") as f:
    for q, r in results:
        f.write(f"Q: {q}\nA: {r}\n\n---\n\n")

print(f"Saved to test_chat_output.txt ({os.path.getsize('test_chat_output.txt')} bytes)")
