"""
Interactive chat with GAI v5 (char-level model, лучшая).
Просто вводи текст — модель продолжает.
Команды: /temp N, /topk N, /len N, /reset, /exit
"""
import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gai.model import GPT

MODEL_PATH = "gai_model_v5_max.pkl"
TEMPERATURE = 0.6
TOP_K = 30
MAX_NEW = 200
SEQ_LEN = 256

def safe_print(text):
    try:
        print(text)
    except:
        print(text.encode('cp1251', errors='replace').decode('cp1251'))

print("Loading GAI v5...")
model, tokenizer, config = GPT.load_pickle(MODEL_PATH)
model.eval()
safe_print(f"Model: {sum(p.numel() for p in model.parameters()):,} params")
safe_print(f"Vocab: {len(tokenizer)}")
safe_print("\n=== GAI v5 (char-level) ===")
safe_print("Вводи текст — модель продолжит как языковая модель.")
safe_print("Commands: /temp N, /topk N, /len N, /reset, /exit\n")

with torch.no_grad():
    while True:
        try:
            user = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!"); break

        if user == "/exit": break
        elif user == "/reset": safe_print("Cleared."); continue
        elif user.startswith("/temp"):
            try: TEMPERATURE = float(user.split()[1]); safe_print(f"temp={TEMPERATURE}")
            except: safe_print("/temp 0.7")
            continue
        elif user.startswith("/topk"):
            try: TOP_K = int(user.split()[1]); safe_print(f"topk={TOP_K}")
            except: safe_print("/topk 30")
            continue
        elif user.startswith("/len"):
            try: MAX_NEW = int(user.split()[1]); safe_print(f"len={MAX_NEW}")
            except: safe_print("/len 200")
            continue
        elif user.startswith("/"): safe_print("Commands: /temp N, /topk N, /len N, /reset, /exit"); continue
        elif not user: continue

        prompt = user + " "
        ids = tokenizer.encode(prompt)
        if len(ids) > SEQ_LEN: ids = ids[-SEQ_LEN:]

        x = torch.tensor([ids], dtype=torch.long)
        out = model.generate(x, max_new_tokens=MAX_NEW, temperature=TEMPERATURE, top_k=TOP_K)
        gen = out[0, len(ids):].tolist()
        result = tokenizer.decode(gen)

        safe_print(f"\n{user} »{result}\n")
