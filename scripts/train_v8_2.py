"""
Train v8_2: Phase 1 = pure dialog (memorize reactions), Phase 2 = balanced.
"""
import sys, os, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gai.model import GPT
from gai.optim import Adam
from gai.train import Trainer
from gai.tokenizer.char import CharTokenizer

MODEL_CHECKPOINT = "gai_model_v5_chat_max.pkl"
DIALOG_FILE = "training_data_dialog_v2.txt"
WIKI_FILE = "training_data_clean.txt"
OUTPUT_NAME = "gai_model_v8_2"
SEQ_LEN = 256
BATCH_SIZE = 8
LR = 0.0002
LOG_EVERY = 200
SAVE_EVERY = 5000

print("=" * 60)
print("  GAI v8.2 — pure dialog then balanced")
print("=" * 60)

model, tokenizer, config = GPT.load_pickle(MODEL_CHECKPOINT)
model.train()
print(f"Model: {sum(p.numel() for p in model.parameters()):,} params, vocab={len(tokenizer)}")

def load_tokens(path, tokenizer, label="", repeat=1, max_tokens=None):
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
    if max_tokens and len(arr) > max_tokens:
        arr = arr[:max_tokens]
    if repeat > 1:
        arr = np.tile(arr, repeat)
    print(f"  {len(arr):,} tokens ({arr.nbytes/1024/1024:.1f} MB) x{repeat}")
    return arr

# Dialog: repeat 500x for pure dialog phase
dialog_tokens = load_tokens(DIALOG_FILE, tokenizer, "dialog v2", repeat=500)
wiki_tokens = load_tokens(WIKI_FILE, tokenizer, "clean wiki", max_tokens=200_000_000)

class MixedDataset:
    def __init__(self, arr_a, arr_b, ratio_a=1.0, seq_len=256):
        self.a = arr_a; self.b = arr_b; self.ratio_a = ratio_a; self.seq_len = seq_len
    def set_ratio(self, ratio):
        self.ratio_a = ratio
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

dataset = MixedDataset(dialog_tokens, wiki_tokens, ratio_a=1.0, seq_len=SEQ_LEN)

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

# ============================================================
# Phase 1: Pure dialog (15000 steps)
# ============================================================
PHASE1_STEPS = 15000
print(f"\n=== Phase 1: {PHASE1_STEPS} steps (100% dialog) ===")
t_start = time.time()
for step in range(PHASE1_STEPS):
    loss_val = trainer.train_step(BATCH_SIZE)
    if step % LOG_EVERY == 0:
        elapsed = time.time() - t_start
        remaining = (elapsed / (step+1)) * (PHASE1_STEPS - step - 1)
        print(f"Phase1 Step {step}/{PHASE1_STEPS} | loss: {loss_val:.4f} | {elapsed:.0f}s | ETA: {remaining:.0f}s")
    if step > 0 and step % 1000 == 0:
        trainer.generate_sample()

print(f"Phase 1 done in {time.time()-t_start:.1f}s")

# Save phase1 checkpoint
model.save_pickle(f"{OUTPUT_NAME}_phase1.pkl", tokenizer)
print(f"Saved: {OUTPUT_NAME}_phase1.pkl")

# ============================================================
# Phase 2: Balanced (10000 steps, 50% dialog, 50% wiki)
# ============================================================
PHASE2_STEPS = 10000
dataset.set_ratio(0.5)
optimizer = Adam(model.parameters(), lr=LR * 0.5)
trainer = Trainer(model, optimizer, dataset, tokenizer)
trainer.generate_sample = safe_gen.__get__(trainer, type(trainer))

print(f"\n=== Phase 2: {PHASE2_STEPS} steps (50% dialog, 50% wiki) ===")
t_start = time.time()
for step in range(PHASE2_STEPS):
    loss_val = trainer.train_step(BATCH_SIZE)
    if step % LOG_EVERY == 0:
        elapsed = time.time() - t_start
        remaining = (elapsed / (step+1)) * (PHASE2_STEPS - step - 1)
        print(f"Phase2 Step {step}/{PHASE2_STEPS} | loss: {loss_val:.4f} | {elapsed:.0f}s | ETA: {remaining:.0f}s")
    if step > 0 and step % 1000 == 0:
        trainer.generate_sample()

print(f"Phase 2 done in {time.time()-t_start:.1f}s")

# Save final
final = f"{OUTPUT_NAME}_max.pkl"
model.save_pickle(final, tokenizer)
print(f"\nSaved: {final}")

# ============================================================
# Test
# ============================================================
print("\n" + "=" * 60)
print("  EVALUATION")
print("=" * 60)

test_cases = [
    "Привет",
    "Здравствуйте",
    "Кто ты?",
    "Что ты умеешь?",
    "Как дела?",
    "Что такое нейросеть?",
    "Столица России?",
    "Что такое Python?",
    "Сколько будет 7*8?",
    # Reactions
    "круто!",
    "класс",
    "ок",
    "понятно",
    "здорово",
    "интересно",
    # Follow-ups
    "расскажи подробнее",
    "а ещё?",
    "почему?",
    # I don't know
    "не знаю что спросить",
    # Off-topic
    "как настроение?",
    "что делаешь?",
    "мне скучно",
    # Complex
    "расскажи про нейросети",
    "это сложно",
    "молодец",
    "зачем ты нужен",
    "придумай историю",
    "Расскажи шутку",
    "проверка связи",
]

with torch.no_grad():
    for i, q in enumerate(test_cases):
        ids = tokenizer.encode(f"{q} ")
        x = torch.tensor([ids[:SEQ_LEN]], dtype=torch.long)
        out = model.generate(x, max_new_tokens=150, temperature=0.6, top_k=30)
        gen = out[0, len(ids):].tolist()
        result = tokenizer.decode(gen)
        if "\n\n" in result:
            result = result[:result.index("\n\n")]
        clean = result[:150].encode('cp1251', errors='replace').decode('cp1251')
        print(f"\n{i+1:2d}. Q: {q}\n    A: {clean}")
