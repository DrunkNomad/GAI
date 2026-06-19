"""
Train v8_3: 100% dialog, no wiki. Start from v5_chat (already wiki-trained).
Focus exclusively on memorizing all reaction patterns perfectly.
"""
import sys, os, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gai.model import GPT
from gai.optim import Adam
from gai.train import Trainer
from gai.tokenizer.char import CharTokenizer

MODEL_CHECKPOINT = "gai_model_v5_chat_max.pkl"
DIALOG_FILE = "training_data_dialog_v2.txt"
OUTPUT_NAME = "gai_model_v8_3"
SEQ_LEN = 256
BATCH_SIZE = 8
LR = 0.0002
STEPS = 25000
LOG_EVERY = 200

print("=" * 60)
print("  GAI v8.3 — 100% dialog (no wiki)")
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

# Dialog: repeat 1000x for maximum exposure
dialog_tokens = load_tokens(DIALOG_FILE, tokenizer, "dialog v2", repeat=1000)

class DialogDataset:
    def __init__(self, arr, seq_len=256):
        self.arr = arr
        self.seq_len = seq_len
    def __len__(self):
        return max(1, len(self.arr) - self.seq_len)
    def get_batch(self, batch_size):
        xs, ys = [], []
        indices = np.random.randint(0, len(self.arr) - self.seq_len, batch_size)
        for i in indices:
            xs.append(self.arr[i:i+self.seq_len].astype(np.int64))
            ys.append(self.arr[i+1:i+self.seq_len+1].astype(np.int64))
        return np.stack(xs), np.stack(ys)

dataset = DialogDataset(dialog_tokens, seq_len=SEQ_LEN)

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

print(f"\nTraining: {STEPS} steps (100% dialog)")
t_start = time.time()
for step in range(STEPS):
    loss_val = trainer.train_step(BATCH_SIZE)
    if step % LOG_EVERY == 0:
        elapsed = time.time() - t_start
        remaining = (elapsed / (step+1)) * (STEPS - step - 1)
        print(f"Step {step}/{STEPS} | loss: {loss_val:.4f} | {elapsed:.0f}s | ETA: {remaining:.0f}s")
    if step > 0 and step % 1000 == 0:
        trainer.generate_sample()

print(f"\nDone in {time.time()-t_start:.1f}s")

final = f"{OUTPUT_NAME}_max.pkl"
model.save_pickle(final, tokenizer)
print(f"Saved: {final}")

# ============================================================
# Test: comprehensive dialog evaluation
# ============================================================
print("\n" + "=" * 60)
print("  COMPREHENSIVE EVALUATION")
print("=" * 60)

test_cases = [
    # Greetings
    ("Привет",),
    ("Здравствуйте",),
    # Core QA
    ("Кто ты?",),
    ("Что ты умеешь?",),
    ("Как дела?",),
    # Knowledge
    ("Что такое нейросеть?",),
    ("Столица России?",),
    ("Что такое Python?",),
    ("Сколько будет 7*8?",),
    # Reactions (CRITICAL - these were failing)
    ("круто!",),
    ("класс",),
    ("ок",),
    ("понятно",),
    ("здорово",),
    ("интересно",),
    # Follow-ups
    ("расскажи подробнее",),
    ("а ещё?",),
    ("почему?",),
    # I don't know
    ("не знаю что спросить",),
    # Off-topic
    ("как настроение?",),
    ("что делаешь?",),
    ("мне скучно",),
    (("это сложно"),),
    (("молодец"),),
    (("зачем ты нужен"),),
    (("придумай историю"),),
    (("Расскажи шутку"),),
    (("проверка связи"),),
]

with torch.no_grad():
    for i, case in enumerate(test_cases):
        q = case[0]
        ids = tokenizer.encode(f"{q} ")
        x = torch.tensor([ids[:SEQ_LEN]], dtype=torch.long)
        out = model.generate(x, max_new_tokens=150, temperature=0.6, top_k=30)
        gen = out[0, len(ids):].tolist()
        result = tokenizer.decode(gen)
        if "\n\n" in result:
            result = result[:result.index("\n\n")]
        clean = result[:200].encode('cp1251', errors='replace').decode('cp1251')
        print(f"\n{i+1:2d}. Q: {q}\n    A: {clean}")
