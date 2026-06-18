import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sys
sys.stdout.reconfigure(encoding="utf-8")

from gai.llm.local_client import LocalLLM

llm = LocalLLM()  # loads gai_model.pkl from legacy root
print("=== Старая модель (BPE, 1024 vocab) ===")
print()

tests = [
    "Привет!",
    "Что такое нейросеть?",
    "Расскажи про Python",
    "Напиши код",
    "Как работает трансформер?",
]

for q in tests:
    res = llm.chat(q)
    print(f"Q: {q}")
    print(f"A: {res}")
    print()
