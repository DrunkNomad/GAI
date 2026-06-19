"""
Test v4 model generation - save to file for proper UTF-8 viewing.
"""
import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gai.model import GPT

model, tokenizer, config = GPT.load_pickle("gai_model_v4_max.pkl")
print(f"Model: {model}")
print(f"Vocab: {len(tokenizer)}")

test_prompts = [
    "Привет",
    "Что такое нейросеть?",
    "Искусственный интеллект",
    "История России",
    "Python — это",
    "Вторая мировая война",
    "Москва — столица",
]

results = []
with torch.no_grad():
    for prompt in test_prompts:
        tokens = tokenizer.encode(prompt)
        x = torch.tensor([tokens[:min(len(tokens), 200)]], dtype=torch.long)
        out = model.generate(x, max_new_tokens=150, temperature=0.6, top_k=30)
        generated = out[0, len(tokens):].tolist()
        result = tokenizer.decode(generated)
        if "\n\n" in result:
            result = result[:result.index("\n\n")]
        results.append((prompt, result))

with open("test_v4_output.txt", "w", encoding="utf-8") as f:
    for prompt, result in results:
        f.write(f"Q: {prompt}\n")
        f.write(f"A: {result}\n")
        f.write("\n" + "=" * 60 + "\n\n")

print(f"Results saved to test_v4_output.txt ({os.path.getsize('test_v4_output.txt')} bytes)")
