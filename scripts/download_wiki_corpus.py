"""
Download Russian Wikipedia corpus via API.
Single-query approach: generator=random + prop=extracts in one call.
"""

import requests, time, os, re

API_URL = "https://ru.wikipedia.org/w/api.php"
TARGET_SIZE = 20 * 1024 * 1024
OUTPUT_FILE = "training_data_raw.txt"
SESSION = requests.Session()
USER_AGENT = "GAI-Training/1.0"

def api_call(params, retries=5):
    params["format"] = "json"
    params["maxlag"] = 5
    headers = {"User-Agent": USER_AGENT}
    for attempt in range(retries):
        try:
            resp = SESSION.get(API_URL, params=params, headers=headers, timeout=60)
            if resp.status_code == 429:
                wait = 10 * (attempt + 1)
                print(f"  429, waiting {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            if attempt == retries - 1:
                raise
            wait = 5 * (attempt + 1)
            print(f"  Error: {e}, retry in {wait}s...")
            time.sleep(wait)

def fetch_batch():
    """Get 50 random pages with extracts in ONE API call."""
    params = {
        "action": "query",
        "generator": "random",
        "grnnamespace": 0,
        "grnlimit": 50,
        "prop": "extracts",
        "exlimit": 50,
        "explaintext": 1,
        "exintro": 0,
    }
    data = api_call(params)
    texts = []
    for pid, page in data.get("query", {}).get("pages", {}).items():
        if "extract" in page and page["extract"]:
            texts.append(page["extract"].strip())
    return texts

def clean_text(text):
    text = re.sub(r'\[\d+(?:,\s*\d+)*\]', '', text)
    text = re.sub(r'\[(?:citation\s+needed|who\?|when\?|where\?|dubious|disputed|verify)\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[edit\]', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()

def main():
    total_size = 0
    total_articles = 0
    batch = 1

    if os.path.exists(OUTPUT_FILE):
        total_size = os.path.getsize(OUTPUT_FILE)
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            total_articles = f.read().count('\n\n')
        print(f"Resuming: {total_size/1024/1024:.1f} MB, ~{total_articles} articles")

    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        while total_size < TARGET_SIZE:
            texts = fetch_batch()
            written = 0
            for text in texts:
                if len(text) < 100:
                    continue
                text = clean_text(text)
                if len(text) < 100:
                    continue
                f.write(text + "\n\n")
                f.flush()
                total_size += len(text.encode("utf-8"))
                total_articles += 1
                written += 1

            pct = total_size / TARGET_SIZE * 100
            print(f"Batch {batch}: +{written} art, total {total_articles} art, {total_size/1024/1024:.1f} MB ({pct:.0f}%)")

            batch += 1
            time.sleep(1.5)

    print(f"\nDone! {total_articles} articles, {total_size/1024/1024:.2f} MB -> {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
