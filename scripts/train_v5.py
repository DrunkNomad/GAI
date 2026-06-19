"""
Training v5: bigger model on full Wikipedia corpus.
Memory-efficient dataset with numpy arrays,
chunked tokenization to avoid giant Python lists.
"""
import sys, os, pickle, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gai.model import GPT
from gai.optim import Adam
from gai.train import Trainer
from gai.tokenizer.char import CharTokenizer


class MemEfficientDataset:
    """Memory-efficient dataset using numpy uint16."""
    def __init__(self, text_path, tokenizer, seq_len=64, chunk_size=10_000_000):
        self.seq_len = seq_len
        self.tokenizer = tokenizer

        # Tokenize in chunks to avoid giant Python lists
        print("  Tokenizing in chunks...")
        chunks = []
        with open(text_path, "r", encoding="utf-8") as f:
            while True:
                chunk_text = f.read(chunk_size)
                if not chunk_text:
                    break
                tokens_list = tokenizer.encode(chunk_text)
                chunks.append(np.array(tokens_list, dtype=np.uint16))
                del tokens_list, chunk_text

        self.tokens = np.concatenate(chunks)
        del chunks
        print(f"  Tokens: {len(self.tokens):,} @ {self.tokens.nbytes/1024/1024:.1f} MB")

    def __len__(self):
        return max(0, len(self.tokens) - self.seq_len)

    def get_batch(self, batch_size):
        indices = np.random.randint(0, len(self), batch_size)
        xs = np.stack([self.tokens[idx:idx + self.seq_len].astype(np.int64) for idx in indices])
        ys = np.stack([self.tokens[idx + 1:idx + self.seq_len + 1].astype(np.int64) for idx in indices])
        return xs, ys


DATA_FILE = "training_data_full.txt"
MODEL_NAME = "gai_model_v5"

SEQ_LEN = 256
EMBED_DIM = 256
NUM_HEADS = 8
NUM_LAYERS = 6
FF_MULT = 4
DROPOUT = 0.1
BATCH_SIZE = 8
LR = 0.001
STEPS = 40000
LOG_EVERY = 500
EVAL_EVERY = 5000

print("=" * 60)
print("  GAI v5")
print(f"  embed={EMBED_DIM}, heads={NUM_HEADS}, layers={NUM_LAYERS}")
print(f"  data: {DATA_FILE}")
print("=" * 60)

if not os.path.exists(DATA_FILE):
    print(f"File not found: {DATA_FILE}")
    sys.exit(1)

# Train tokenizer on a sample (don't load full file)
print("\nTraining tokenizer on sample...")
with open(DATA_FILE, "r", encoding="utf-8") as f:
    sample_for_vocab = f.read(5_000_000)

tokenizer = CharTokenizer()
tokenizer.train([sample_for_vocab])
vocab_size = len(tokenizer)
del sample_for_vocab
print(f"Vocab: {vocab_size} unique chars")

print(f"\nBuilding dataset...")
start = time.time()
dataset = MemEfficientDataset(DATA_FILE, tokenizer, seq_len=SEQ_LEN)
print(f"Dataset: {time.time()-start:.1f}s, {len(dataset)//BATCH_SIZE:,} steps/epoch")

model = GPT(
    vocab_size=vocab_size,
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    ff_dim=EMBED_DIM * FF_MULT,
    max_seq_len=SEQ_LEN,
    dropout=DROPOUT,
)

total_params = sum(p.numel() for p in model.parameters())
print(f"Params: {total_params:,}")

optimizer = Adam(model.parameters(), lr=LR)
trainer = Trainer(model, optimizer, dataset, tokenizer)

print(f"\nTraining: {STEPS} steps, batch={BATCH_SIZE}")
print(f"Estimated: ~4-5 hours")
trainer.train(steps=STEPS, batch_size=BATCH_SIZE, log_every=LOG_EVERY, eval_every=EVAL_EVERY)

output_path = f"{MODEL_NAME}_max.pkl"
model.save_pickle(output_path, tokenizer)
print(f"\nSaved: {output_path}")

print("\n=== TEST ===")
test_prompts = ["Привет", "Что такое нейросеть?", "Искусственный интеллект"]
with torch.no_grad():
    for prompt in test_prompts:
        tokens = tokenizer.encode(prompt)
        x = torch.tensor([tokens[:SEQ_LEN]], dtype=torch.long)
        out = model.generate(x, max_new_tokens=100, temperature=0.6, top_k=30)
        gen = out[0, len(tokens):].tolist()
        result = tokenizer.decode(gen)
        if "\n\n" in result:
            result = result[:result.index("\n\n")]
        print(f"\nQ: {prompt}\nA: {result}")

print(f"\nDone. Model: {output_path}")
