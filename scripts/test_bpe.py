"""Test BPE tokenizer on Russian Wikipedia sample."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gai.tokenizer.bpe_fast import BPETokenizer

# Load a small sample to test
with open("training_data_full.txt", "r", encoding="utf-8") as f:
    sample = f.read(2_000_000)  # 2 MB sample

print(f"Sample: {len(sample):,} chars")

print("Training BPE (vocab=2048)...")
start = time.time()
tokenizer = BPETokenizer(vocab_size=2048)
tokenizer.train([sample])
print(f"Done: {time.time()-start:.1f}s, vocab={len(tokenizer)}")

# Test encode/decode
test_texts = [
    "Привет",
    "Искусственный интеллект",
    "Нейросеть — это математическая модель",
    "Москва — столица Российской Федерации",
]

for text in test_texts:
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)
    ratio = len(text) / len(ids) if ids else 1
    print(f"\nOriginal: {text}")
    print(f"Tokens:   {len(ids)} (compression ratio: {ratio:.1f}x)")
    print(f"Decoded:  {decoded}")
    print(f"Match:    {text == decoded}")

# Test on a larger sample to measure encoding speed
print(f"\n\nEncoding larger sample...")
sample2 = sample[:500_000]
start = time.time()
ids = tokenizer.encode(sample2)
elapsed = time.time() - start
print(f"500K chars -> {len(ids):,} tokens in {elapsed:.2f}s ({len(ids)/elapsed:.0f} tokens/s)")

# Test pickling
import pickle
print(f"\nTesting pickle...")
state = pickle.dumps(tokenizer)
t2 = pickle.loads(state)
test_ids = t2.encode("Привет мир!")
print(f"Pickle OK: {test_ids[:10]}")

print(f"\nAll BPE tests passed!")
