"""
Fine-tune v5 on dialog data.
Repeats dialog data for effective fine-tuning with Wikipedia mix.
"""
import sys, os, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gai.model import GPT
from gai.optim import Adam
from gai.train import Trainer
from gai.tokenizer.char import CharTokenizer

MODEL_CHECKPOINT = "gai_model_v5_max.pkl"
DIALOG_FILE = "training_data_dialog.txt"
WIKI_FILE = "training_data_full.txt"
OUTPUT_NAME = "gai_model_v5_chat"
SEQ_LEN = 256
BATCH_SIZE = 8
LR = 0.0002
STEPS = 10000
LOG_EVERY = 200
EVAL_EVERY = 1000

print("=" * 60)
print("  GAI v5 — dialog fine-tune")
print("=" * 60)

model, tokenizer, config = GPT.load_pickle(MODEL_CHECKPOINT)
model.train()
print(f"Model: {sum(p.numel() for p in model.parameters()):,} params, vocab={len(tokenizer)}")

def load_tokens(path, tokenizer, label="", repeat=1):
    print(f"Loading {label}...")
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        while True:
            raw = f.read(10_000_000)
            if not raw:
                break
            ids = tokenizer.encode(raw)
            chunks.append(np.array(ids, dtype=np.uint16))
    arr = np.concatenate(chunks)
    if repeat > 1:
        arr = np.tile(arr, repeat)
    print(f"  {len(arr):,} tokens ({arr.nbytes/1024/1024:.1f} MB) x{repeat}")
    return arr

# Repeat dialog data 30x to balance with a slice of Wikipedia
dialog_tokens = load_tokens(DIALOG_FILE, tokenizer, "dialog", repeat=30)
# Use first 100 MB of Wikipedia to avoid catastrophic forgetting
wiki_tokens = load_tokens(WIKI_FILE, tokenizer, "wikipedia")[:50_000_000]

class MixedDataset:
    def __init__(self, arr_a, arr_b, ratio_a=0.7, seq_len=256):
        self.a = arr_a; self.b = arr_b; self.ratio_a = ratio_a; self.seq_len = seq_len
    def __len__(self):
        return max(1, len(self.a) + len(self.b) - self.seq_len)
    def get_batch(self, batch_size):
        n_a = max(1, int(batch_size * self.ratio_a))
        n_b = batch_size - n_a
        xs, ys = [], []
        for n, arr in [(n_a, self.a), (n_b, self.b)]:
            if n <= 0: continue
            indices = np.random.randint(0, len(arr) - self.seq_len, n)
            for i in indices:
                xs.append(arr[i:i+self.seq_len].astype(np.int64))
                ys.append(arr[i+1:i+self.seq_len+1].astype(np.int64))
        return np.stack(xs), np.stack(ys)

dataset = MixedDataset(dialog_tokens, wiki_tokens, ratio_a=0.7, seq_len=SEQ_LEN)

optimizer = Adam(model.parameters(), lr=LR)
trainer = Trainer(model, optimizer, dataset, tokenizer)

def safe_gen(self, prompt="", max_tokens=50, temperature=1.0):
    try:
        tokens = self.tokenizer.encode(prompt) if prompt else []
        x = (torch.tensor([tokens[:100]], dtype=torch.long) if tokens else torch.zeros((1,1), dtype=torch.long))
        out = self.model.generate(x, max_tokens, temperature)
        text = self.tokenizer.decode(out[0, x.shape[1]:].tolist())
        safe = text.encode('cp1251', errors='replace').decode('cp1251')[:120]
        print(f"Sample: {safe}")
        return text
    except Exception as e:
        print(f"Sample: <{e}>")
trainer.generate_sample = safe_gen.__get__(trainer, type(trainer))

print(f"\nFine-tuning: {STEPS} steps (70% dialog, 30% wiki)")
t_start = time.time()
for step in range(STEPS):
    loss_val = trainer.train_step(BATCH_SIZE)
    if step % LOG_EVERY == 0:
        print(f"Step {step}/{STEPS} | loss: {loss_val:.4f} | time: {time.time()-t_start:.0f}s")
    if EVAL_EVERY and step > 0 and step % EVAL_EVERY == 0:
        trainer.generate_sample()
    if step > 0 and (step == STEPS-1 or step % 2000 == 0):
        model.save_pickle(f"{OUTPUT_NAME}_step{step}.pkl", tokenizer)

print(f"\nDone in {time.time()-t_start:.1f}s")

final = f"{OUTPUT_NAME}_max.pkl"
model.save_pickle(final, tokenizer)
print(f"Saved: {final}")

# Test
print("\n=== DIALOG TEST ===")
test_qs = ["Привет", "Кто ты?", "Что такое нейросеть?", "Столица России?", "Расскажи шутку", "Спасибо", "Пока", "Что такое Python?"]
with torch.no_grad():
    for q in test_qs:
        ids = tokenizer.encode(f"{q} ")
        x = torch.tensor([ids[:SEQ_LEN]], dtype=torch.long)
        out = model.generate(x, max_new_tokens=150, temperature=0.6, top_k=30)
        gen = out[0, len(ids):].tolist()
        result = tokenizer.decode(gen)
        if "\n\n" in result:
            result = result[:result.index("\n\n")]
        print(f"\nQ: {q}\nA: {result[:200]}")
