"""
BPE tokenizer using HuggingFace tokenizers library (Rust-backed, fast).
"""
import os, json, tempfile
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors


class BPETokenizer:
    def __init__(self, vocab_size=2048, special_tokens=None):
        self.vocab_size = vocab_size
        self._special_tokens = special_tokens or ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
        self._tokenizer = None

    def train(self, texts):
        """Train BPE on a list of text strings."""
        tokenizer = Tokenizer(models.BPE(unk_token="<UNK>"))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=self._special_tokens,
            min_frequency=2,
            show_progress=True,
        )

        if isinstance(texts, str):
            texts = [texts]

        tokenizer.train_from_iterator(texts, trainer)
        self._tokenizer = tokenizer
        self._tokenizer_id = id(tokenizer)

    def encode(self, text):
        """Encode text to token IDs."""
        if self._tokenizer is None:
            raise ValueError("Tokenizer not trained")
        return self._tokenizer.encode(text).ids

    def decode(self, ids):
        """Decode token IDs to text."""
        if self._tokenizer is None:
            raise ValueError("Tokenizer not trained")
        return self._tokenizer.decode(ids)

    def __len__(self):
        if self._tokenizer is None:
            return len(self._special_tokens)
        return self._tokenizer.get_vocab_size()

    def __getstate__(self):
        """Support pickling by serializing the tokenizer to JSON."""
        state = self.__dict__.copy()
        if self._tokenizer is not None:
            state["_tokenizer_json"] = self._tokenizer.to_str()
            state["_tokenizer"] = None
        return state

    def __setstate__(self, state):
        """Support unpickling by restoring from JSON."""
        json_str = state.pop("_tokenizer_json", None)
        self.__dict__.update(state)
        if json_str is not None:
            import tempfile, os
            self._tokenizer = Tokenizer.from_str(json_str)

    def __repr__(self):
        if self._tokenizer is None:
            return f"BPETokenizer(vocab_size={self.vocab_size}, not trained)"
        return f"BPETokenizer(vocab_size={len(self)})"
