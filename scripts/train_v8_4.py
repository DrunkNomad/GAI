"""
Train v8_4: prefixed dialog format (>question answer).
Start from v5_chat, 100% dialog.
The > prefix cleanly separates dialog from wiki: model learns ">" means "respond as chatbot".
"""
import sys, os, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gai.model import GPT
from gai.optim import Adam
from gai.train import Trainer
from gai.tokenizer.char import CharTokenizer

MODEL_CHECKPOINT = "gai_model_v5_chat_max.pkl"
DIALOG_FILE = "training_data_dialog_v4.txt"
OUTPUT_NAME = "gai_model_v8_4"
SEQ_LEN = 256
BATCH_SIZE = 8
LR = 0.0002
STEPS = 15000
LOG_EVERY = 200

print("=" * 60)
print("  GAI v8.4 — prefixed dialog (>question)")
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

dialog_tokens = load_tokens(DIALOG_FILE, tokenizer, "dialog v4 (prefixed)", repeat=2000)

class DialogDataset:
    def __init__(self, arr, seq_len=256):
        self.arr = arr; self.seq_len = seq_len
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

print(f"\nTraining: {STEPS} steps")
t_start = time.time()
for step in range(STEPS):
    loss_val = trainer.train_step(BATCH_SIZE)
    if step % LOG_EVERY == 0:
        elapsed = time.time() - t_start
        remaining = (elapsed / (step+1)) * (STEPS - step - 1)
        print(f"Step {step}/{STEPS} | loss: {loss_val:.4f} | {elapsed:.0f}s | ETA: {remaining:.0f}s")
    if step > 0 and step % 2000 == 0:
        trainer.generate_sample()

print(f"\nDone in {time.time()-t_start:.1f}s")

final = f"{OUTPUT_NAME}_max.pkl"
model.save_pickle(final, tokenizer)
print(f"Saved: {final}")

# Test with > prefix
print("\n" + "=" * 60)
print("  EVALUATION (with > prefix)")
print("=" * 60)

tests = [
    "привет", "здравствуйте", "кто ты?", "что ты умеешь?", "как дела?",
    "что такое нейросеть?", "столица россии?", "что такое python?", "сколько будет 7*8?",
    "круто!", "класс", "ок", "понятно", "здорово", "интересно",
    "расскажи подробнее", "а ещё?", "почему?", "не знаю что спросить",
    "как настроение?", "что делаешь?", "мне скучно", "это сложно",
    "молодец", "зачем ты нужен", "придумай историю", "расскажи шутку", "проверка связи",
]

with torch.no_grad():
    for i, q in enumerate(tests, 1):
        # Use > prefix for dialog mode
        ids = tokenizer.encode(f">{q} ")
        x = torch.tensor([ids[:SEQ_LEN]], dtype=torch.long)
        out = model.generate(x, max_new_tokens=80, temperature=0.6, top_k=30)
        gen = out[0, len(ids):].tolist()
        result = tokenizer.decode(gen)
        if "\n\n" in result:
            result = result[:result.index("\n\n")]
        safe = result[:200]
        print(f"{i:2d}. >{q}\n    -> {safe}\n")
