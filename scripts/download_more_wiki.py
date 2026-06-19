"""
Download remaining Wikipedia chunks (articles5-6) and append to corpus.
"""
import requests, bz2, xml.parsers.expat, re, os, time

BASE_URL = "https://dumps.wikimedia.org/ruwiki/latest/"
OUTPUT_FILE = "training_data_full.txt"
CHUNK_SIZE = 65536
headers = {"User-Agent": "GAI-Training/1.0"}
MIN_ARTICLE_LENGTH = 100

CHUNKS = [
    "ruwiki-latest-pages-articles5.xml-p3835773p5335772.bz2",
    "ruwiki-latest-pages-articles5.xml-p5335773p6585765.bz2",
    "ruwiki-latest-pages-articles6.xml-p6585766p8085765.bz2",
    "ruwiki-latest-pages-articles6.xml-p8085766p9585765.bz2",
    "ruwiki-latest-pages-articles6.xml-p9585766p11085765.bz2",
    "ruwiki-latest-pages-articles6.xml-p11085766p11486309.bz2",
    "ruwiki-latest-pages-articles6.xml-p11085766p11513555.bz2",
]

def strip_wiki(text):
    if not text: return ""
    text = re.sub(r'<ref[^>]*>[^<]*(?:</ref>)?', '', text, flags=re.DOTALL)
    text = re.sub(r'<ref[^/][^>]*/>', '', text)
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    for _ in range(5):
        new = re.sub(r'\{\{[^{}]*\}\}', '', text)
        if new == text: break
        text = new
    text = re.sub(r'\[\[(?:Category|Категория|File|Файл|Image|Изображение):[^\]]*\]\]', '', text)
    text = re.sub(r'\[\[([^\]]*\|)?([^\]]+)\]\]', r'\2', text)
    text = re.sub(r"'''?", '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    lines = [l for l in text.split('\n') if l.strip()]
    text = '\n'.join(lines)
    return text.strip()

def main():
    existing = os.path.getsize(OUTPUT_FILE) if os.path.exists(OUTPUT_FILE) else 0
    print(f"Existing corpus: {existing/1024/1024:.1f} MB")

    total_articles = 0
    total_added = 0
    temp_file = "wiki_chunk.bz2"

    for i, name in enumerate(CHUNKS):
        url = BASE_URL + name
        print(f"\n[{i+1}/{len(CHUNKS)}] {name}")

        # Download
        resp = requests.get(url, headers=headers, stream=True, timeout=1200)
        resp.raise_for_status()
        total = int(resp.headers.get('content-length', 0))
        downloaded = 0
        start = time.time()
        with open(temp_file, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    speed = downloaded / (time.time()-start) / 1024
                    print(f"\r  {downloaded/1024/1024:.0f}/{total/1024/1024:.0f} MB @ {speed:.0f} KB/s", end='')
        print(f"\n  Download: {time.time()-start:.0f}s")

        # Extract
        class Handler:
            def __init__(self):
                self.in_text = False; self.chars = []; self.n = 0; self.added = 0
            def start(self, name, attrs):
                if name == 'text': self.in_text = True; self.chars = []
            def end(self, name):
                if name == 'text' and self.in_text:
                    self.in_text = False
                    raw = ''.join(self.chars); self.chars = []
                    if len(raw) < MIN_ARTICLE_LENGTH: return
                    cleaned = strip_wiki(raw)
                    if len(cleaned) < 100: return
                    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                        f.write(cleaned + "\n\n")
                    self.added += len(cleaned.encode("utf-8"))
                    self.n += 1
            def data(self, d):
                if self.in_text: self.chars.append(d)

        h = Handler()
        parser = xml.parsers.expat.ParserCreate()
        parser.StartElementHandler = h.start
        parser.EndElementHandler = h.end
        parser.CharacterDataHandler = h.data
        parser.buffer_text = True

        with bz2.open(temp_file, "rb") as f:
            while True:
                chunk = f.read(CHUNK_SIZE)
                if not chunk: break
                parser.Parse(chunk)

        total_articles += h.n
        total_added += h.added
        print(f"  +{h.n} articles, +{h.added/1024/1024:.1f} MB (total: {total_articles} art, {existing+total_added:.0f}/{existing+total_added/1024/1024:.1f} MB)")

        os.remove(temp_file)

    print(f"\n{'='*60}")
    print(f"Done! Total added: {total_articles} articles, {total_added/1024/1024:.1f} MB")
    print(f"Final corpus: {existing+total_added/1024/1024:.1f} MB")

if __name__ == "__main__":
    main()
