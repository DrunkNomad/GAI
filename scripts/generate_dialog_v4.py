"""Generate dialog_v4 with > prefix to distinguish from wiki text.
Format: ">user_input model_response" """
import random
random.seed(42)

# Read dialog_v2
with open("training_data_dialog_v2.txt", "r", encoding="utf-8") as f:
    text = f.read()

pairs = []
for chunk in text.strip().split("\n\n"):
    chunk = chunk.strip()
    if not chunk:
        continue
    # Find first space after first word
    idx = chunk.index(" ") if " " in chunk else -1
    if idx > 0:
        q, a = chunk[:idx].strip(), chunk[idx:].strip()
        pairs.append((q, a))

# Add some variations with > prefix already in the answer (multi-turn feel)
more_pairs = [
    (">привет", "Привет! Я GAI, искусственный интеллект."),
    (">кто ты", "Я GAI — искусственный интеллект."),
    (">расскажи шутку", "Почему программисты любят тёмную тему? Потому что свет привлекает баги!"),
    (">класс", "Рад стараться! Спрашивай ещё."),
    (">ок", "Окей! Жду следующих вопросов."),
    (">круто", "Спасибо! Могу ещё рассказать."),
    (">что ты умеешь", "Я умею отвечать на вопросы, объяснять концепции, помогать с кодом."),
]

all_pairs = pairs + more_pairs
random.shuffle(all_pairs)

# Two variants:
# 1. No prefix (original format) - for backward compat
# 2. With prefix - for the new chat interface

# Write prefixed version
prefixed_lines = [f">{q} {a}" for q, a in all_pairs]
data = "\n\n".join(prefixed_lines)

with open("training_data_dialog_v4.txt", "w", encoding="utf-8") as f:
    f.write(data)

print(f"Dialog v4 (prefixed): {len(all_pairs):,} pairs, {len(data):,} chars ({len(data)/1024:.1f} KB)")
print("First 5:")
for i in range(min(5, len(prefixed_lines))):
    print(f"  {prefixed_lines[i][:80]}")
