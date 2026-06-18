import re
import json
from collections import Counter


class BPETokenizer:
    def __init__(self, vocab_size=512):
        self.vocab_size = vocab_size
        self.merges = {}
        self.vocab = {}
        self.pattern = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\w+| ?\d+| ?[^\w\s]+|\s+(?!\S)|\s+""")

    def _get_stats(self, words):
        pairs = Counter()
        for word, freq in words.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs

    def _merge(self, words, pair, new_token):
        new_words = {}
        bigram = " ".join(pair)
        for word, freq in words.items():
            new_word = word.replace(bigram, new_token)
            new_words[new_word] = freq
        return new_words

    def train(self, texts):
        words = Counter()
        for text in texts:
            tokens = [" ".join(list(word)) + " </w>" for word in text.split()]
            for t in tokens:
                words[t] += 1

        self.vocab = {}
        all_chars = set()
        for word in words:
            for ch in word.split():
                all_chars.add(ch)
        for i, ch in enumerate(sorted(all_chars)):
            self.vocab[ch] = i

        next_id = len(self.vocab)
        while len(self.vocab) < self.vocab_size:
            stats = self._get_stats(words)
            if not stats:
                break
            best_pair = max(stats, key=stats.get)
            new_token = "".join(best_pair)
            self.merges[best_pair] = new_token
            self.vocab[new_token] = next_id
            next_id += 1
            words = self._merge(words, best_pair, new_token)

        self.vocab["<PAD>"] = next_id + 1
        self.vocab["<UNK>"] = next_id + 2
        self.vocab["<BOS>"] = next_id + 3
        self.vocab["<EOS>"] = next_id + 4

    def encode(self, text):
        words = self.pattern.findall(text)
        tokens = []
        for word in words:
            symbols = list(word) + ["</w>"]
            while len(symbols) > 1:
                stats = Counter()
                for i in range(len(symbols) - 1):
                    stats[(symbols[i], symbols[i + 1])] = 1
                pair = None
                for p in stats:
                    if p in self.merges:
                        pair = p
                        break
                if pair is None:
                    break
                new_token = self.merges[pair]
                new_symbols = []
                i = 0
                while i < len(symbols):
                    if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
                        new_symbols.append(new_token)
                        i += 2
                    else:
                        new_symbols.append(symbols[i])
                        i += 1
                symbols = new_symbols
            for s in symbols:
                tokens.append(self.vocab.get(s, self.vocab.get("<UNK>", 0)))
        return tokens

    def decode(self, ids):
        id_to_token = {v: k for k, v in self.vocab.items()}
        tokens = []
        for i in ids:
            token = id_to_token.get(i, "<UNK>")
            if token == "</w>":
                tokens.append(" ")
            elif token in ("<PAD>", "<UNK>", "<BOS>", "<EOS>"):
                tokens.append("")
            else:
                tokens.append(token.replace("</w>", ""))
        return "".join(tokens).strip()

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"merges": [list(k) for k in self.merges], "vocab": self.vocab}, f)

    def load(self, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.merges = {tuple(k): "".join(k) for k in data["merges"]}
        self.vocab = data["vocab"]

    @property
    def vocab_size(self):
        return self._vocab_size

    @vocab_size.setter
    def vocab_size(self, value):
        self._vocab_size = value

    def __len__(self):
        return len(self.vocab)
