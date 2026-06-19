"""
Training v4: GPT model on real Russian Wikipedia data.
"""
import sys, os, pickle, random, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gai.model import GPT
from gai.optim import Adam
from gai.train import TextDataset, Trainer
from gai.tokenizer.char import CharTokenizer

DATA_FILE = "training_data_raw.txt"
MODEL_NAME = "gai_model_v4"

# Hyperparams
SEQ_LEN = 256
EMBED_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 6
FF_MULT = 4
DROPOUT = 0.1
BATCH_SIZE = 8
LR = 0.001
STEPS = 15000
LOG_EVERY = 200
EVAL_EVERY = 2000
TEST_PROMPTS = [
    "Привет",
    "Что такое нейросеть?",
    "Искусственный интеллект",
    "История России",
    "Python — это",
]

print("=" * 60)
print("  GAI v4 — обучение на корпусе русской Википедии")
print("=" * 60)

if not os.path.exists(DATA_FILE):
    print(f"Data file not found: {DATA_FILE}")
    sys.exit(1)

with open(DATA_FILE, "r", encoding="utf-8") as f:
    data = f.read()

print(f"\nData loaded: {len(data):,} chars ({len(data)/1024/1024:.1f} MB)")

tokenizer = CharTokenizer()
tokenizer.train([data])
vocab_size = len(tokenizer)
print(f"Vocabulary size: {vocab_size}")

dataset = TextDataset([data], tokenizer, seq_len=SEQ_LEN)
print(f"Tokens: {len(dataset.tokens):,}, Batches: {len(dataset)}")

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
print(f"Model params: {total_params:,}")

optimizer = Adam(model.parameters(), lr=LR)
trainer = Trainer(model, optimizer, dataset, tokenizer)

print(f"\nTraining: {STEPS} steps, batch={BATCH_SIZE}")
trainer.train(steps=STEPS, batch_size=BATCH_SIZE, log_every=LOG_EVERY, eval_every=EVAL_EVERY)

output_path = f"{MODEL_NAME}_max.pkl"
model.save_pickle(output_path, tokenizer)
print(f"\nModel saved: {output_path}")

print("\n=== TEST GENERATION ===")
with torch.no_grad():
    for prompt in TEST_PROMPTS:
        tokens = tokenizer.encode(prompt)
        x = torch.tensor([tokens[:SEQ_LEN]], dtype=torch.long)
        out = model.generate(x, max_new_tokens=100, temperature=0.6, top_k=30)
        generated = out[0, len(tokens):].tolist()
        result = tokenizer.decode(generated)
        if "\n\n" in result:
            result = result[:result.index("\n\n")]
        print(f"\nQ: {prompt}")
        print(f"A: {result}")

print(f"\nDone. Saved as {output_path}")
