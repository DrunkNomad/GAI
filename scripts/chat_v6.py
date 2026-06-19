"""
Interactive chat with GAI v6 (BPE model).
Просто вводи текст — модель продолжает как языковая модель.
Команды: /reset, /temp 0.7, /topk 40, /exit
"""
import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gai.model import GPT

MODEL_PATH = "gai_model_v6_max.pkl"

SYS_PROMPT = ""
TEMPERATURE = 0.7
TOP_K = 40
MAX_NEW = 150
SEQ_LEN = 256
HISTORY = []

def safe_print(text):
    try:
        print(text)
    except:
        print(text.encode('cp1251', errors='replace').decode('cp1251'))

print("Loading GAI v6...")
model, tokenizer, config = GPT.load_pickle(MODEL_PATH)
model.eval()
safe_print(f"Model: {sum(p.numel() for p in model.parameters()):,} params")
safe_print(f"Vocab: {len(tokenizer)}")
safe_print("")

safe_print("=== GAI v6 (BPE) — языковая модель ===")
safe_print("Просто вводи текст. Модель продолжит его.")
safe_print("Команды: /temp N, /topk N, /len N, /reset, /exit")
safe_print("")

with torch.no_grad():
    while True:
        try:
            user = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if user == "/exit":
            break
        elif user == "/reset":
            HISTORY.clear()
            safe_print("Context cleared.")
            continue
        elif user.startswith("/temp"):
            try:
                TEMPERATURE = float(user.split()[1])
                safe_print(f"Temperature = {TEMPERATURE}")
            except:
                safe_print("Usage: /temp 0.7")
            continue
        elif user.startswith("/topk"):
            try:
                TOP_K = int(user.split()[1])
                safe_print(f"Top-K = {TOP_K}")
            except:
                safe_print("Usage: /topk 40")
            continue
        elif user.startswith("/len"):
            try:
                MAX_NEW = int(user.split()[1])
                safe_print(f"Max new tokens = {MAX_NEW}")
            except:
                safe_print("Usage: /len 150")
            continue
        elif user.startswith("/"):
            safe_print("Commands: /temp N, /topk N, /len N, /reset, /exit")
            continue
        elif not user:
            continue

        # Build prompt: user text as continuation prefix
        prompt = SYS_PROMPT + user + " "
        ids = tokenizer.encode(prompt)
        if len(ids) > SEQ_LEN:
            ids = ids[-SEQ_LEN:]

        x = torch.tensor([ids], dtype=torch.long)
        out = model.generate(x, max_new_tokens=MAX_NEW, temperature=TEMPERATURE, top_k=TOP_K)
        gen_ids = out[0, len(ids):].tolist()
        result = tokenizer.decode(gen_ids)

        safe_print(f"\n{user} »{result}\n")