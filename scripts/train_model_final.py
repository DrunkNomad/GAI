"""
Универсальный скрипт обучения GPT (PyTorch).
Поддержка загрузки текстов из файла и автоматической генерации данных.

Запуск: python scripts/train_model_final.py
  или:  python scripts/train_model_final.py --file my_text.txt --seq 256
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import pickle
import argparse
import numpy as np
import torch

from gai.model import GPT
from gai.optim import Adam
from gai.train import TextDataset, Trainer


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


def generate_default_data():
    pairs = [
        ("Привет", "Привет! Я GAI, твой ИИ-помощник. Чем могу помочь?"),
        ("Привет!", "Привет! Я GAI, твой ИИ-помощник. Чем могу помочь?"),
        ("Здравствуйте", "Здравствуйте! Я GAI, рад помочь. Спрашивайте что угодно."),
        ("Здравствуй", "Здравствуй! Чем я могу тебе помочь сегодня?"),
        ("Как дела?", "У меня всё отлично! Спасибо, что спросил. Чем могу помочь?"),
        ("Кто ты?", "Я GAI — General Artificial Intelligence. Моя нейросеть написана с нуля на Python."),
        ("Что ты умеешь?", "Я умею отвечать на вопросы, вести диалог, решать задачи. Могу помочь с программированием, математикой и другими темами."),
        ("Какая погода?", "У меня нет доступа к интернету, поэтому я не могу узнать текущую погоду."),
        ("Спасибо", "Пожалуйста! Обращайся ещё."),
        ("Пока", "До свидания! Хорошего дня!"),
        ("Что такое нейросеть?", "Нейросеть — это математическая модель, вдохновлённая работой мозга."),
        ("Что такое ИИ?", "Искусственный интеллект — область информатики, создающая интеллектуальные системы."),
        ("Как работает трансформер?", "Трансформер использует механизм внимания для нахождения связей между словами."),
        ("Что такое машинное обучение?", "ML — метод, при котором модель учится на данных без явного программирования."),
        ("Напиши код", "print('Hello, World!')\nЭто простой код на Python."),
        ("Напиши функцию", "def hello(name):\n    print(f'Привет, {name}!')"),
        ("Расскажи про Python", "Python — популярный язык программирования, простой и понятный."),
        ("Что такое алгоритм?", "Алгоритм — последовательность шагов для решения задачи."),
        ("2 + 2", "2 + 2 = 4"),
        ("Столица России?", "Столица России — Москва."),
        ("Столица Франции?", "Столица Франции — Париж."),
        ("Ты умный?", "Я учусь каждый день! Моя нейросеть постоянно совершенствуется."),
        ("Помоги мне", "Конечно! Расскажи, какая у тебя задача, и я постараюсь помочь."),
        ("Расскажи шутку", "Почему программисты любят тёмную тему? Потому что свет привлекает баги!"),
        ("Ты человек?", "Нет, я искусственный интеллект. Я программа, которая учится и помогает людям."),
        ("В чём смысл жизни?", "42! Это отсылка к книге Дугласа Адамса 'Автостопом по галактике'."),
    ]
    conversations = []
    for q, a in pairs:
        conversations.append(f"User: {q}\nAssistant: {a}")
    random.shuffle(conversations)
    return "\n\n".join(conversations)


def main():
    parser = argparse.ArgumentParser(description="Обучение GPT модели (PyTorch)")
    parser.add_argument("--file", type=str, default=None, help="Путь к текстовому файлу для обучения")
    parser.add_argument("--vocab", type=int, default=256, help="Размер словаря (char-level)")
    parser.add_argument("--seq", type=int, default=256, help="Длина последовательности (seq_len)")
    parser.add_argument("--embed", type=int, default=128, help="Размер эмбеддингов")
    parser.add_argument("--heads", type=int, default=4, help="Количество голов в attention")
    parser.add_argument("--layers", type=int, default=4, help="Количество слоёв трансформера")
    parser.add_argument("--steps", type=int, default=5000, help="Количество шагов обучения")
    parser.add_argument("--batch", type=int, default=8, help="Размер батча")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--output", type=str, default="gai_model.pkl", help="Путь для сохранения модели")
    args = parser.parse_args()

    print("=" * 60)
    print("  GAI — обучение GPT модели (PyTorch)")
    print("=" * 60)

    if args.file:
        print(f"Загружаем данные из файла: {args.file}")
        if not os.path.exists(args.file):
            print(f"Файл не найден: {args.file}")
            sys.exit(1)
        with open(args.file, "r", encoding="utf-8") as f:
            data = f.read()
        print(f"  Загружено: {len(data)} символов")
    else:
        print("Данные не указаны, генерирую встроенные диалоги...")
        data = generate_default_data()
        multiplier = 50
        data = data * multiplier
        print(f"  Сгенерировано: {len(data)} символов")

    print(f"\nТренируем токенизатор (char-level, vocab<={args.vocab})...")
    tokenizer = CharTokenizer(vocab_size=args.vocab)
    tokenizer.train([data])
    actual_vocab_size = len(tokenizer)
    print(f"  Фактический размер словаря: {actual_vocab_size}")

    print(f"\nПараметры модели:")
    print(f"  vocab_size={actual_vocab_size}, embed_dim={args.embed}")
    print(f"  num_heads={args.heads}, num_layers={args.layers}, seq_len={args.seq}")

    dataset = TextDataset([data], tokenizer, seq_len=args.seq)
    print(f"  Токенов: {len(dataset.tokens)}, батчей: {len(dataset)}")

    model = GPT(
        vocab_size=actual_vocab_size,
        embed_dim=args.embed,
        num_heads=args.heads,
        num_layers=args.layers,
        ff_dim=args.embed * 4,
        max_seq_len=args.seq,
        dropout=0.1,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Всего параметров: {total_params:,}")

    optimizer = Adam(model.parameters(), lr=args.lr)
    trainer = Trainer(model, optimizer, dataset, tokenizer)

    est_per_step = 2.0 * (args.seq / 128) * (args.layers / 4) * (args.embed / 128) ** 2
    est_s = est_per_step * args.steps / 60
    print(f"\nОбучаем {args.steps} шагов (batch_size={args.batch})...")
    print(f"  Примерное время: ~{est_s:.0f} минут")
    trainer.train(steps=args.steps, batch_size=args.batch, log_every=100, eval_every=2000)

    model.save_pickle(args.output, tokenizer)
    print(f"\nМодель сохранена в {args.output}")

    print("\n\n=== ТЕСТ ГЕНЕРАЦИИ ===")
    test_prompts = ["Привет!", "Как дела?", "Кто ты?", "Что ты умеешь?", "Напиши код", "Спасибо", "2 + 2"]
    with torch.no_grad():
        for prompt in test_prompts:
            tokens = tokenizer.encode(f"User: {prompt}\nAssistant: ")
            x = torch.tensor([tokens[:args.seq]], dtype=torch.long)
            out = model.generate(
                x,
                max_new_tokens=80,
                temperature=0.6,
                top_k=30,
            )
            generated = out[0, len(tokens):].tolist()
            result = tokenizer.decode(generated)
            if "\n\n" in result:
                result = result[:result.index("\n\n")]
            print(f"\n  Q: {prompt}")
            print(f"  A: {result}")


if __name__ == "__main__":
    main()
