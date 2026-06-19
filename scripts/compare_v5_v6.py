"""Compare v5 (char-level) vs v6 (BPE) model outputs."""
import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gai.model import GPT

prompts = [
    "Привет",
    "Что такое нейросеть?",
    "Искусственный интеллект",
    "Москва",
    "Россия",
    "Вторая мировая война",
    "Кошка",
    "Python",
]

models = {
    "v5 (char-level, loss 1.25)": "gai_model_v5_max.pkl",
    "v6 (BPE, loss 2.57)": "gai_model_v6_max.pkl",
}

results = {}
for label, path in models.items():
    print(f"Loading {label}...")
    m, tok, cfg = GPT.load_pickle(path)
    m.eval()
    out = []
    with torch.no_grad():
        for p in prompts:
            ids = tok.encode(p)
            x = torch.tensor([ids[:200]], dtype=torch.long)
            gen = m.generate(x, max_new_tokens=120, temperature=0.6, top_k=30)
            text = tok.decode(gen[0, len(ids):].tolist())
            if "\n\n" in text:
                text = text[:text.index("\n\n")]
            out.append(text)
    results[label] = out

with open("model_comparison.txt", "w", encoding="utf-8") as f:
    for i, p in enumerate(prompts):
        f.write(f"Prompt: {p}\n")
        f.write("=" * 60 + "\n")
        for label, out in results.items():
            f.write(f"\n{label}:\n  {out[i][:200]}\n")
        f.write("\n" + "-" * 60 + "\n\n")

print(f"Saved to model_comparison.txt")
