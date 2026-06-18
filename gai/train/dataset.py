import numpy as np


class TextDataset:
    def __init__(self, texts, tokenizer, seq_len=64):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.tokens = []
        for text in texts:
            tokens = tokenizer.encode(text)
            self.tokens.extend(tokens)

    def __len__(self):
        return max(0, len(self.tokens) - self.seq_len)

    def __getitem__(self, idx):
        x = np.array(self.tokens[idx:idx + self.seq_len], dtype=np.int64)
        y = np.array(self.tokens[idx + 1:idx + self.seq_len + 1], dtype=np.int64)
        return x, y

    def get_batch(self, batch_size):
        indices = np.random.randint(0, len(self), batch_size)
        xs = []
        ys = []
        for idx in indices:
            x, y = self[idx]
            xs.append(x)
            ys.append(y)
        return np.stack(xs), np.stack(ys)
