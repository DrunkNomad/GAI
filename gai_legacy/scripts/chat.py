import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from gai.tensor import Tensor
from gai.model import GPT
import pickle


def load_model(model_path=None):
    if model_path is None:
        model_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "gai_model.pkl",
        )

    with open(model_path, "rb") as f:
        data = pickle.load(f)

    config = data["config"]
    model = GPT(
        vocab_size=config["vocab_size"],
        embed_dim=config["embed_dim"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        ff_dim=config["ff_dim"],
        max_seq_len=config["max_seq_len"],
    )
    model.eval()

    loaded_params = dict(data["model_state"])
    for param in model.parameters():
        if param.label in loaded_params:
            param.data = loaded_params[param.label]

    tokenizer = data["tokenizer"]

    return model, tokenizer, config


def generate(model, tokenizer, config, prompt, max_new_tokens=100, temperature=0.7, top_k=30):
    tokens = tokenizer.encode(prompt)
    if len(tokens) > config["max_seq_len"]:
        tokens = tokens[-config["max_seq_len"] :]

    x = np.array([tokens], dtype=np.int64)
    out = model.generate(
        Tensor(x, requires_grad=False),
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )
    generated = out[0, len(tokens) :].tolist()
    response = tokenizer.decode(generated)

    if "\n\n" in response:
        response = response[: response.index("\n\n")]
    if "User:" in response:
        response = response.split("User:")[0].strip()
    if response.startswith("Assistant:"):
        response = response[len("Assistant:") :].strip()

    return response


def main():
    model_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "gai_model.pkl",
    )

    print("=" * 60)
    print("  GAI Legacy — интерактивный чат")
    print("=" * 60)

    if not os.path.exists(model_path):
        print(f"Модель не найдена: {model_path}")
        print("Запусти обучение: python scripts/train_model.py")
        return

    print("Загрузка модели...")
    model, tokenizer, config = load_model(model_path)
    print(f"  embed_dim={config['embed_dim']}, num_layers={config['num_layers']}")
    print(f"  vocab={config['vocab_size']}, seq_len={config['max_seq_len']}")
    print(f"  Команды: /exit, /temp N (сменить temperature)")
    print()

    temperature = 0.7
    history = []

    while True:
        try:
            user_input = input("Вы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nДо свидания!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("/exit", "/quit"):
            print("До свидания!")
            break
        if user_input.startswith("/temp"):
            try:
                temperature = float(user_input.split()[1])
                print(f"Temperature = {temperature}")
            except (ValueError, IndexError):
                print("Использование: /temp 0.8")
            continue

        history.append(f"User: {user_input}")
        context = "\n".join(history[-4:])
        prompt = f"{context}\nAssistant: "

        response = generate(
            model, tokenizer, config, prompt,
            temperature=temperature, top_k=30,
        )

        history.append(f"Assistant: {response}")
        if len(history) > 10:
            history = history[-10:]

        print(f"GAI: {response}")
        print()


if __name__ == "__main__":
    main()
