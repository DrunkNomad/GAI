"""
Улучшенное обучение GPT с char-level токенизацией.
Генерирует много разнообразных русских текстов.
Запуск: python scripts/train_model_v2.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import pickle
import numpy as np

from gai.model import GPT
from gai.optim import Adam
from gai.tensor import Tensor
from gai.train import TextDataset, Trainer


class CharTokenizer:
    def __init__(self):
        self.char_to_id = {}
        self.id_to_char = {}
        self.vocab = {}

    def train(self, texts):
        chars = set()
        for text in texts:
            chars.update(text)
        chars = sorted(chars)
        self.char_to_id = {c: i for i, c in enumerate(chars)}
        self.id_to_char = {i: c for i, c in enumerate(chars)}
        self.vocab = self.char_to_id.copy()

    def encode(self, text):
        return [self.char_to_id.get(c, 0) for c in text]

    def decode(self, ids):
        return "".join(self.id_to_char.get(i, "") for i in ids)

    def __len__(self):
        return len(self.char_to_id)


def generate_diverse_russian_text():
    topics = [
        "нейросеть", "искусственный интеллект", "машинное обучение",
        "программирование", "компьютеры", "наука", "технологии",
        "математика", "физика", "биология", "космос", "роботы",
    ]

    subjects = [
        "нейросеть", "модель", "алгоритм", "программа", "система",
        "компьютер", "робот", "процессор", "память", "данные",
        "трансформер", "токен", "вектор", "градиент", "функция",
    ]

    predicates = [
        "учится на данных", "обрабатывает информацию", "решает задачи",
        "генерирует текст", "понимает язык", "анализирует данные",
        "находит закономерности", "делает предсказания", "оптимизирует процесс",
        "преобразует входные данные", "использует алгоритмы", "работает быстро",
    ]

    questions = [
        "Что такое", "Как работает", "Зачем нужен", "Где применяется",
        "Кто создал", "Когда появился", "Какие бывают", "Чем отличается",
    ]

    answers = [
        "Это метод машинного обучения.", "Она состоит из множества нейронов.",
        "Он преобразует входной сигнал в выходной.", "Данные поступают на вход модели.",
        "Модель обучается на размеченных данных.", "Градиент показывает направление улучшения.",
        "Функция потерь измеряет ошибку предсказания.", "Оптимизатор обновляет веса модели.",
        "Эпоха это полный проход по данным.", "Батч это группа примеров для одного шага.",
        "Скорость обучения определяет размер шага.", "Переобучение это запоминание данных.",
    ]

    greetings = [
        "Привет! Я нейросеть GAI.", "Здравствуйте! Чем могу помочь?",
        "Приветствую! Я искусственный интеллект.", "Добрый день! Я готова отвечать на вопросы.",
    ]

    sentences = []

    for _ in range(10000):
        subj = random.choice(subjects)
        pred = random.choice(predicates)
        sentences.append(f"{subj.capitalize()} {pred}.")

    for _ in range(5000):
        q = random.choice(questions)
        subj = random.choice(subjects + topics)
        sentences.append(f"{q} {subj}?")

    for _ in range(5000):
        sentences.append(random.choice(answers))

    for _ in range(2000):
        sentences.append(random.choice(greetings))

    for _ in range(3000):
        a = random.choice(subjects)
        b = random.choice(subjects)
        v = random.choice(predicates)
        sentences.append(f"{a.capitalize()} и {b} {v}.")

    for _ in range(2000):
        sentences.append(f"Я умею {random.choice(['отвечать на вопросы', 'генерировать текст', 'анализировать данные', 'решать задачи', 'помогать людям'])}.")

    for _ in range(2000):
        n1 = random.randint(1, 100)
        n2 = random.randint(1, 100)
        op = random.choice(["+", "-", "*"])
        if op == "+":
            r = n1 + n2
        elif op == "-":
            r = n1 - n2
        else:
            r = n1 * n2
        sentences.append(f"{n1} {op} {n2} = {r}.")

    for _ in range(2000):
        sentences.append(f"Слово '{random.choice(subjects)}' состоит из {random.randint(3, 10)} букв.")

    for _ in range(2000):
        sentences.append(f"Это {random.choice(['просто', 'сложно', 'интересно', 'полезно', 'важно', 'нужно', 'хорошо', 'плохо'])}.")

    random.shuffle(sentences)
    return " ".join(sentences)


def main():
    print("=" * 60)
    print("  GAI — улучшенное обучение (char-level)")
    print("=" * 60)

    print("Генерируем данные...")
    raw_text = generate_diverse_russian_text()

    multiplier = 10
    data = raw_text * multiplier
    print(f"  Сгенерировано: {len(raw_text)} уникальных символов")
    print(f"  Всего данных (x{multiplier}): {len(data)} символов")

    print("\nТренируем char-level токенизатор...")
    tokenizer = CharTokenizer()
    tokenizer.train([data])
    actual_vocab_size = len(tokenizer)
    print(f"  Уникальных символов: {actual_vocab_size}")
    print(f"  Символы: {''.join(list(tokenizer.char_to_id.keys())[:20])}...")

    embed_dim = 128
    num_heads = 4
    num_layers = 4
    seq_len = 128

    print(f"\nПараметры модели:")
    print(f"  vocab_size={actual_vocab_size}, embed_dim={embed_dim}")
    print(f"  num_heads={num_heads}, num_layers={num_layers}, seq_len={seq_len}")

    print("Создаём датасет...")
    dataset = TextDataset([data], tokenizer, seq_len=seq_len)
    print(f"  Всего токенов: {len(dataset.tokens)}, батчей: {len(dataset)}")

    print("Создаём модель...")
    model = GPT(
        vocab_size=actual_vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        ff_dim=embed_dim * 4,
        max_seq_len=seq_len,
        dropout=0.1,
    )
    total_params = sum(p.data.size for p in model.parameters())
    print(f"  Всего параметров: {total_params:,}")

    steps = 5000
    batch_size = 16
    lr = 0.002

    optimizer = Adam(model.parameters(), lr=lr)
    trainer = Trainer(model, optimizer, dataset, tokenizer)

    print(f"\nНачинаем обучение на {steps} шагах (lr={lr})...")
    print(f"  Ожидаемое время: ~{steps * 3 / 60:.0f} минут")
    trainer.train(steps=steps, batch_size=batch_size, log_every=200, eval_every=1000)

    path = "gai_model_v2.pkl"
    with open(path, "wb") as f:
        pickle.dump({
            "model_state": [(p.label, p.data) for p in model.parameters()],
            "tokenizer": tokenizer,
            "config": {
                "vocab_size": actual_vocab_size,
                "embed_dim": embed_dim,
                "num_heads": num_heads,
                "num_layers": num_layers,
                "ff_dim": embed_dim * 4,
                "max_seq_len": seq_len,
            }
        }, f)
    print(f"\nМодель сохранена в {path}")

    print("\n\n=== ТЕСТ ГЕНЕРАЦИИ ===")
    test_prompts = [
        "Привет!",
        "Что такое нейросеть?",
        "Как работает трансформер?",
        "Какая сегодня погода?",
        "Напиши код.",
    ]

    for prompt in test_prompts:
        tokens = tokenizer.encode(prompt)
        x = np.array([tokens[:seq_len]], dtype=np.int64)
        out = model.generate(
            Tensor(x, requires_grad=False),
            max_new_tokens=80,
            temperature=0.6,
            top_k=30,
        )
        generated = out[0, len(tokens):].tolist()
        result = tokenizer.decode(generated)
        print(f"\n  Q: {prompt}")
        print(f"  A: {result}")


if __name__ == "__main__":
    main()
