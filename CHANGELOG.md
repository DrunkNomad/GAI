# Changelog

## v3 — PyTorch порт + расширенный датасет (18.06.2026)

### Изменения

#### Полный порт на PyTorch

- `gai/tensor.py` — `Tensor = torch.Tensor` + фабрика `tensor()`
- `gai/nn/*` — все слои переписаны на `torch.nn` (Linear, Embedding, LayerNorm,
  GELU, Dropout, CrossEntropyLoss, MultiHeadAttention, TransformerBlock)
- `gai/nn/module.py` — наследует `torch.nn.Module`
- `gai/model/gpt.py` — forward через PyTorch ops, `torch.nn.ModuleList`,
  `F.cross_entropy`, `torch.multinomial` для генерации
- `gai/optim/adam.py`, `sgd.py` — обёртки над `torch.optim.Adam/SGD`
- `gai/train/trainer.py` — `torch.tensor()` вместо самодельного `Tensor()`
- `gai/train/dataset.py` — без изменений (numpy для данных)
- `scripts/train_model_final.py` — обновлён под PyTorch API
- `rag_service/src/rag/generator.py` — обновлён под новый pickle формат
  (поддерживает и старый numpy формат совместимости)

#### Производительность

| Метрика | NumPy | PyTorch | Ускорение |
|---------|-------|---------|-----------|
| Шаг обучения | 2.0 с | 0.1 с | **20x** |
| Память (3MB датасет) | 8+ GB | ~630 MB | **12x** |
| 10000 шагов | 5.5 часов | 17 минут | **20x** |

#### Датасет

- `generate_data_v2.py` — первый шаг к разнообразию (875 уникальных диалогов)
- `generate_data_v3.py` — расширенный генератор (1029 уникальных диалогов,
  шаблоны, код, математика, диалоги)
- `training_data_big.txt` — объединённый датасет 4.76MB из всех источников

### Гиперпараметры v3
| Параметр | Значение |
|----------|----------|
| vocab    | ~130 (char-level) |
| embed    | 128 |
| heads    | 4 |
| layers   | 6 |
| seq_len  | 256 |
| batch    | 8 |
| lr       | 0.001 |
| steps    | 10000 |
| params   | 1,260,032 |

---

## v2 — NumPy оптимизация (18.06.2026)

### Изменения

#### gai/tensor.py
- Добавлен параметр `retain_graph=False` в `backward()` — граф вычислений
  автоматически очищается после обратного прохода (`_depends_on = []`,
  `_creation_op = None`), предотвращая утечку памяти на длинных тренировках
- Добавлен метод `detach()` для отрыва тензора от графа вычислений

#### scripts/train_model_final.py
- Добавлен класс `CharTokenizer` — лёгкий char-level токенизатор
  (encode/decode через ord/chr), обходит BPE-оверхед на больших данных
- BPE токенизатор заменён на char-level по умолчанию
- `log_every` изменён с 500 → 100 для более частого логирования
- `multiplier` для `--file` режима: 3 → 1 (данные не дублируются)

#### Новые файлы
- `gai_model_v1_max.pkl` — бэкап первой обученной модели (1,286,144 params)
- `training_data.txt` — сгенерированный датасет, 11 стилей, 21M символов
- `training_data_small.txt` — срез 3MB для обучения (помещается в память)

### Гиперпараметры v2
| Параметр | Значение |
|----------|----------|
| vocab    | 130 (char-level) |
| embed    | 128 |
| heads    | 4 |
| layers   | 6 |
| seq_len  | 256 |
| batch    | 8 |
| lr       | 0.002 |
| steps    | 10000 |
| params   | 1,251,840 |

---

## v1 — Первое обучение (17.06.2026)

### Запуск
- Первая тренировка модели на 25 QA-парах × 50 повторов
- BPE токенизатор, vocab=256 (фактически char-level, 260 токенов)
- Гиперпараметры: embed=128, heads=4, layers=6, seq=256, steps=5000, batch=8
- Параметров: 1,286,144
- Время: 9036 сек (~2.5 часа)
- Финальный loss: 3.9082
- Сохранена в `gai_model_max.pkl`
