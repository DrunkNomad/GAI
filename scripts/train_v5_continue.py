"""
Continue v5 training — best char-level model, more steps.
"""
import sys, os, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gai.model import GPT
from gai.optim import Adam
from gai.train import Trainer


class CharDataset:
    def __init__(self, text_path, tokenizer, seq_len=256, chunk_chars=10_000_000):
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        print("  Tokenizing chars in chunks...")
        chunks = []
        with open(text_path, "r", encoding="utf-8") as f:
            while True:
                raw = f.read(chunk_chars)
                if not raw:
                    break
                ids = tokenizer.encode(raw)
                chunks.append(np.array(ids, dtype=np.uint16))
        self.tokens = np.concatenate(chunks)
        del chunks
        print(f"  {len(self.tokens):,} tokens ({self.tokens.nbytes/1024/1024:.1f} MB)")

    def __len__(self):
        return max(0, len(self.tokens) - self.seq_len)

    def get_batch(self, batch_size):
        indices = np.random.randint(0, len(self), batch_size)
        xs = np.stack([self.tokens[i:i + self.seq_len].astype(np.int64) for i in indices])
        ys = np.stack([self.tokens[i + 1:i + self.seq_len + 1].astype(np.int64) for i in indices])
        return xs, ys


DATA_FILE = "training_data_full.txt"
CHECKPOINT = "gai_model_v5_max.pkl"
MODEL_NAME = "gai_model_v5"
SEQ_LEN = 256
BATCH_SIZE = 8
LR = 0.0005
TOTAL_STEPS = 70000
LOG_EVERY = 500
EVAL_EVERY = 5000

print("=" * 60)
print("  GAI v5 — continue training (char-level)")
print("=" * 60)

model, tokenizer, config = GPT.load_pickle(CHECKPOINT)
model.train()
print(f"Model: {sum(p.numel() for p in model.parameters()):,} params, vocab={len(tokenizer)}")

# Rebuild dataset with the loaded tokenizer
print("Rebuilding dataset...")
t = time.time()
dataset = CharDataset(DATA_FILE, tokenizer, seq_len=SEQ_LEN)
print(f"  {time.time()-t:.1f}s, {len(dataset)//BATCH_SIZE:,} steps/epoch")

# Lower LR for fine-tuning
optimizer = Adam(model.parameters(), lr=LR)
trainer = Trainer(model, optimizer, dataset, tokenizer)

# Safe generate
def safe_gen(self, prompt="", max_tokens=50, temperature=1.0):
    try:
        tokens = self.tokenizer.encode(prompt) if prompt else []
        x = torch.tensor([tokens[:100] or [0]], dtype=torch.long) if tokens else torch.zeros((1,1), dtype=torch.long)
        out = self.model.generate(x, max_tokens, temperature)
        text = self.tokenizer.decode(out[0, x.shape[1]:].tolist())
        safe = text.encode('cp1251', errors='replace').decode('cp1251')
        print(f"Sample: {safe[:100]}")
        return text
    except Exception as e:
        print(f"Sample: <{e}>")
trainer.generate_sample = safe_gen.__get__(trainer, type(trainer))

steps_remaining = TOTAL_STEPS - 40000
print(f"\nTraining: {steps_remaining} more steps ({40000} -> {TOTAL_STEPS})")
print(f"Estimated: ~3 hours")

checkpoint_steps = set(range(45000, TOTAL_STEPS+1, 5000))
t_start = time.time()
for step in range(40000, TOTAL_STEPS):
    loss_val = trainer.train_step(BATCH_SIZE)
    trainer.history["loss"].append(loss_val)
    if step % LOG_EVERY == 0:
        elapsed = time.time() - t_start
        print(f"Step {step}/{TOTAL_STEPS} | loss: {loss_val:.4f} | time: {elapsed:.0f}s")
    if EVAL_EVERY and step > 0 and step % EVAL_EVERY == 0:
        trainer.generate_sample()
    if step in checkpoint_steps or step == TOTAL_STEPS - 1:
        p = f"{MODEL_NAME}_step{step}.pkl"
        model.save_pickle(p, tokenizer)
        print(f"  Saved: {p}")

print(f"\nDone in {time.time()-t_start:.1f}s. Final loss: {loss_val:.4f}")

final = f"{MODEL_NAME}_finetuned.pkl"
model.save_pickle(final, tokenizer)
print(f"Saved: {final}")

print("\nTesting...")
results = []
with torch.no_grad():
    for prompt in ["Привет", "Что такое нейросеть?", "Искусственный интеллект", "Москва"]:
        ids = tokenizer.encode(prompt)
        x = torch.tensor([ids[:SEQ_LEN]], dtype=torch.long)
        out = model.generate(x, max_new_tokens=150, temperature=0.6, top_k=30)
        gen = out[0, len(ids):].tolist()
        result = tokenizer.decode(gen)
        if "\n\n" in result:
            result = result[:result.index("\n\n")]
        results.append((prompt, result))

with open("test_v5_finetune.txt", "w", encoding="utf-8") as f:
    for p, r in results:
        f.write(f"Q: {p}\nA: {r}\n\n---\n\n")
print(f"Tests saved to test_v5_finetune.txt")
