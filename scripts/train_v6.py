"""
v6: BPE on full Wikipedia corpus.
Safe console output + checkpoints.
"""
import sys, os, time, numpy as np, torch, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gai.model import GPT
from gai.optim import Adam
from gai.train import Trainer
from gai.tokenizer.bpe_fast import BPETokenizer


class BPEDataset:
    def __init__(self, text_path, tokenizer, seq_len=256, chunk_chars=5_000_000):
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        print("  Encoding BPE in chunks...")
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
MODEL_NAME = "gai_model_v6"
SEQ_LEN = 256
EMBED_DIM = 256
NUM_HEADS = 8
NUM_LAYERS = 6
DROPOUT = 0.1
BATCH_SIZE = 8
LR = 0.001
STEPS = 30000
LOG_EVERY = 500
EVAL_EVERY = 5000
BPE_VOCAB = 2048

print(f"GAI v6 — BPE={BPE_VOCAB}, embed={EMBED_DIM}, layers={NUM_LAYERS}")

if not os.path.exists(DATA_FILE):
    sys.exit(f"File not found: {DATA_FILE}")

print("Training BPE...")
with open(DATA_FILE, "r", encoding="utf-8") as f:
    sample = f.read(5_000_000)
t = time.time()
tokenizer = BPETokenizer(vocab_size=BPE_VOCAB)
tokenizer.train([sample])
del sample
print(f"  Vocab: {len(tokenizer)} ({time.time()-t:.1f}s)")

print("Building dataset...")
t = time.time()
dataset = BPEDataset(DATA_FILE, tokenizer, seq_len=SEQ_LEN)
print(f"  {time.time()-t:.1f}s, {len(dataset)//BATCH_SIZE:,} steps/epoch")

model = GPT(vocab_size=len(tokenizer), embed_dim=EMBED_DIM,
            num_heads=NUM_HEADS, num_layers=NUM_LAYERS,
            ff_dim=EMBED_DIM*4, max_seq_len=SEQ_LEN, dropout=DROPOUT)
print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

trainer = Trainer(model, Adam(model.parameters(), lr=LR), dataset, tokenizer)

# Safe generate_sample for console
def safe_generate(self, prompt="", max_tokens=50, temperature=1.0):
    try:
        import torch
        tokens = self.tokenizer.encode(prompt) if prompt else []
        x = torch.tensor([tokens or [0]], dtype=torch.long) if tokens else torch.zeros((1,1), dtype=torch.long)
        out = self.model.generate(x, max_tokens, temperature)
        text = self.tokenizer.decode(out[0, x.shape[1]:].tolist())
        print(f"Sample: {text.encode('cp1251', errors='replace').decode('cp1251')[:100]}")
        return text
    except Exception as e:
        print(f"Sample: <{e}>")
trainer.generate_sample = safe_generate.__get__(trainer, type(trainer))

print(f"Training {STEPS} steps...")
checkpoint_steps = set(range(5000, STEPS+1, 5000))
t_start = time.time()
for step in range(STEPS):
    loss_val = trainer.train_step(BATCH_SIZE)
    trainer.history["loss"].append(loss_val)
    if step % LOG_EVERY == 0:
        print(f"Step {step}/{STEPS} | loss: {loss_val:.4f} | time: {time.time()-t_start:.0f}s")
    if EVAL_EVERY and step > 0 and step % EVAL_EVERY == 0:
        trainer.generate_sample()
    if step in checkpoint_steps or step == STEPS - 1:
        p = f"{MODEL_NAME}_step{step}.pkl"
        model.save_pickle(p, tokenizer)
        print(f"  Saved: {p}")

print(f"Done in {time.time()-t_start:.1f}s. Final loss: {loss_val:.4f}")

# Final save
final = f"{MODEL_NAME}_max.pkl"
model.save_pickle(final, tokenizer)
print(f"Saved: {final}")

# Test to file
print("\nTesting...")
results = []
with torch.no_grad():
    for prompt in ["Привет", "Что такое нейросеть?", "Искусственный интеллект", "Москва"]:
        ids = tokenizer.encode(prompt)
        x = torch.tensor([ids[:SEQ_LEN]], dtype=torch.long)
        out = model.generate(x, max_new_tokens=100, temperature=0.6, top_k=30)
        gen = out[0, len(ids):].tolist()
        result = tokenizer.decode(gen)
        if "\n\n" in result:
            result = result[:result.index("\n\n")]
        results.append((prompt, result))

with open("test_v6_output.txt", "w", encoding="utf-8") as f:
    for p, r in results:
        f.write(f"Q: {p}\nA: {r}\n\n---\n\n")

print(f"Tests saved to test_v6_output.txt")
