import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gai.model import GPT
from gai.optim import Adam
from gai.tokenizer import BPETokenizer
from gai.train import TextDataset, Trainer


def main():
    print("=" * 60)
    print("  GAI — тренировка своей нейросети")
    print("=" * 60)

    text = input("Путь к файлу с текстом для обучения (enter = tiny demo): ").strip()
    if text and os.path.isfile(text):
        with open(text, "r", encoding="utf-8") as f:
            data = f.read()
    else:
        data = """
        The quick brown fox jumps over the lazy dog.
        Neural networks are computing systems inspired by biological neural networks.
        They consist of connected units called neurons organized in layers.
        Deep learning uses multiple layers to progressively extract higher-level features.
        Transformers are a type of neural network architecture particularly suited for sequence data.
        The attention mechanism allows the model to focus on relevant parts of the input.
        Self-attention computes weighted representations of input sequences.
        """
        data = data * 50
        print("  Использую демо-текст (", len(data), "символов)")

    print("\nПараметры модели (enter = стандартные):")
    vocab_size = int(input("  vocab_size (512): ") or "512")
    embed_dim = int(input("  embed_dim (64): ") or "64")
    num_heads = int(input("  num_heads (4): ") or "4")
    num_layers = int(input("  num_layers (2): ") or "2")
    seq_len = int(input("  seq_len (32): ") or "32")

    print("\nТренируем токенизатор...")
    tokenizer = BPETokenizer(vocab_size=vocab_size)
    tokenizer.train([data])
    print(f"  Слов в словаре: {len(tokenizer)}")

    print("Создаём датасет...")
    dataset = TextDataset([data], tokenizer, seq_len=seq_len)
    print(f"  Всего токенов: {len(dataset.tokens)}, батчей: {len(dataset)}")

    print("Создаём модель...")
    model = GPT(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        ff_dim=embed_dim * 4,
        max_seq_len=seq_len,
    )
    print(f"  {model}")
    total_params = sum(p.data.size for p in model.parameters())
    print(f"  Всего параметров: {total_params:,}")

    lr = float(input(f"\nLearning rate (0.001): ") or "0.001")
    steps = int(input(f"Шагов обучения (500): ") or "500")
    batch_size = int(input(f"Batch size (8): ") or "8")

    optimizer = Adam(model.parameters(), lr=lr)
    trainer = Trainer(model, optimizer, dataset, tokenizer)

    print(f"\nНачинаем обучение на {steps} шагах...")
    trainer.train(steps=steps, batch_size=batch_size, log_every=50, eval_every=200)

    save = input("\nСохранить модель? (y/N): ").strip().lower()
    if save == "y":
        import pickle
        path = "gai_model.pkl"
        with open(path, "wb") as f:
            pickle.dump({
                "model_state": [(p.label, p.data) for p in model.parameters()],
                "tokenizer": tokenizer,
                "config": {
                    "vocab_size": vocab_size,
                    "embed_dim": embed_dim,
                    "num_heads": num_heads,
                    "num_layers": num_layers,
                    "ff_dim": embed_dim * 4,
                    "max_seq_len": seq_len,
                }
            }, f)
        print(f"Модель сохранена в {path}")

    print("\nГенерация текста:")
    prompt = input("Промпт (enter = ничего): ").strip()
    trainer.generate_sample(prompt, max_tokens=100, temperature=0.8)


if __name__ == "__main__":
    main()
