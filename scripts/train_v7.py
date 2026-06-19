"""
v7: train on larger corpus (1 GB) + dialog data.
Continues from v5_chat fine-tuned model.
"""
import sys, os, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gai.model import GPT
from gai.optim import Adam
from gai.train import Trainer
from gai.tokenizer.char import CharTokenizer

MODEL_CHECKPOINT = "gai_model_v5_chat_max.pkl"
WIKI_FILE = "training_data_full.txt"
DIALOG_FILE = "training_data_dialog.txt"
OUTPUT_NAME = "gai_model_v7"
SEQ_LEN = 256
BATCH_SIZE = 8
LR = 0.0003
STEPS = 30000
LOG_EVERY = 500
EVAL_EVERY = 3000
SUBSET_BYTES = 1_000_000_000  # First 1 GB of wiki

print("=" * 60)
print("  GAI v7 — more data + dialog")
print("=" * 60)

model, tokenizer, config = GPT.load_pickle(MODEL_CHECKPOINT)
model.train()
print(f"Model: {sum(p.numel() for p in model.parameters()):,} params, vocab={len(tokenizer)}")

def load_tokens_chunked(path, tokenizer, label="", max_bytes=None, repeat=1):
    print(f"Loading {label}...")
    chunks = []
    total = 0
    with open(path, "r", encoding="utf-8") as f:
        while True:
            if max_bytes and total >= max_bytes:
                break
            chunk_size = min(10_000_000, max_bytes - total) if max_bytes else 10_000_000
            raw = f.read(chunk_size)
            if not raw:
                break
            total += len(raw.encode("utf-8"))
            ids = tokenizer.encode(raw)
            chunks.append(np.array(ids, dtype=np.uint16))
    arr = np.concatenate(chunks)
    if repeat > 1:
        arr = np.tile(arr, repeat)
    print(f"  {len(arr):,} tokens ({arr.nbytes/1024/1024:.1f} MB)")
    return arr

# Load Wikipedia (first 1 GB) + dialog (repeated)
wiki_tokens = load_tokens_chunked(WIKI_FILE, tokenizer, "wikipedia (1 GB)", max_bytes=SUBSET_BYTES)
dialog_tokens = load_tokens_chunked(DIALOG_FILE, tokenizer, "dialog", repeat=50)

# Mixed dataset
class MixedDataset:
    def __init__(self, arr_a, arr_b, ratio_a=0.6, seq_len=256):
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

dataset = MixedDataset(wiki_tokens, dialog_tokens, ratio_a=0.6, seq_len=SEQ_LEN)

optimizer = Adam(model.parameters(), lr=LR)
trainer = Trainer(model, optimizer, dataset, tokenizer)

def safe_gen(self, prompt="", max_tokens=50, temperature=1.0):
    try:
        tokens = self.tokenizer.encode(prompt) if prompt else []
        x = torch.tensor([tokens[:100] or [0]], dtype=torch.long) if tokens else torch.zeros((1,1), dtype=torch.long)
        out = self.model.generate(x, max_tokens, temperature)
        text = self.tokenizer.decode(out[0, x.shape[1]:].tolist())
        safe = text.encode('cp1251', errors='replace').decode('cp1251')[:120]
        print(f"Sample: {safe}")
    except Exception as e:
        print(f"Sample: <{e}>")
trainer.generate_sample = safe_gen.__get__(trainer, type(trainer))

print(f"\nTraining: {STEPS} steps, LR={LR}")
print(f"  Wiki: {len(wiki_tokens):,} tokens | Dialog: {len(dialog_tokens):,} tokens")
t_start = time.time()
for step in range(STEPS):
    loss_val = trainer.train_step(BATCH_SIZE)
    if step % LOG_EVERY == 0:
        print(f"Step {step}/{STEPS} | loss: {loss_val:.4f} | time: {time.time()-t_start:.0f}s")
    if EVAL_EVERY and step > 0 and step % EVAL_EVERY == 0:
        trainer.generate_sample()
    if step > 0 and (step == STEPS-1 or step % 5000 == 0):
        model.save_pickle(f"{OUTPUT_NAME}_step{step}.pkl", tokenizer)

print(f"\nDone in {time.time()-t_start:.1f}s")
model.save_pickle(f"{OUTPUT_NAME}_max.pkl", tokenizer)
print(f"Saved: {OUTPUT_NAME}_max.pkl")

# Test to file
print("\n=== TEST ===")
test_qs = ["Привет", "Кто ты?", "Что такое нейросеть?", "Столица России?", "Расскажи шутку"]
results = []
with torch.no_grad():
    for q in test_qs:
        ids = tokenizer.encode(f"{q} ")
        x = torch.tensor([ids[:SEQ_LEN]], dtype=torch.long)
        out = model.generate(x, max_new_tokens=150, temperature=0.6, top_k=30)
        gen = out[0, len(ids):].tolist()
        r = tokenizer.decode(gen)
        if "\n\n" in r: r = r[:r.index("\n\n")]
        results.append((q, r))

with open("test_v7_output.txt", "w", encoding="utf-8") as f:
    for q, r in results:
        f.write(f"Q: {q}\nA: {r}\n\n---\n\n")
print(f"Tests saved")
