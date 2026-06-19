class CharTokenizer:
    def __init__(self, vocab_size=256):
        self.vocab_size = vocab_size

    def train(self, texts):
        chars = set()
        for text in texts:
            chars.update(text)
        self.vocab = {ch: i for i, ch in enumerate(sorted(chars))}
        self.vocab["<PAD>"] = len(self.vocab)
        self.vocab["<UNK>"] = len(self.vocab)
        self.vocab["<BOS>"] = len(self.vocab)
        self.vocab["<EOS>"] = len(self.vocab)
        self.id_to_token = {v: k for k, v in self.vocab.items()}

    def encode(self, text):
        unk = self.vocab.get("<UNK>", 0)
        return [self.vocab.get(ch, unk) for ch in text]

    def decode(self, ids):
        return "".join(self.id_to_token.get(i, "") for i in ids)

    def __len__(self):
        return len(self.vocab)
