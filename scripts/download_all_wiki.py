"""
Download ALL Russian Wikipedia dump chunks and extract text.
Appends all text to a single output file.
"""

import requests, bz2, xml.parsers.expat, re, os, time, sys

BASE_URL = "https://dumps.wikimedia.org/ruwiki/latest/"
OUTPUT_FILE = "training_data_full.txt"
TARGET_SIZE = 400 * 1024 * 1024  # 400 MB target
CHUNK_SIZE = 65536
headers = {"User-Agent": "GAI-Training/1.0"}
MIN_ARTICLE_LENGTH = 100

CHUNKS = [
    "ruwiki-latest-pages-articles1.xml-p1p224167.bz2",
    "ruwiki-latest-pages-articles2.xml-p224168p1042043.bz2",
    "ruwiki-latest-pages-articles3.xml-p1042044p2198269.bz2",
    "ruwiki-latest-pages-articles4.xml-p2198270p3698269.bz2",
    "ruwiki-latest-pages-articles4.xml-p3698270p3835772.bz2",
    "ruwiki-latest-pages-articles5.xml-p3835773p5335772.bz2",
    "ruwiki-latest-pages-articles5.xml-p5335773p6585765.bz2",
    "ruwiki-latest-pages-articles6.xml-p6585766p8085765.bz2",
    "ruwiki-latest-pages-articles6.xml-p8085766p9585765.bz2",
    "ruwiki-latest-pages-articles6.xml-p9585766p11085765.bz2",
    "ruwiki-latest-pages-articles6.xml-p11085766p11486309.bz2",
    "ruwiki-latest-pages-articles6.xml-p11085766p11513555.bz2",
]


def strip_wiki(text):
    if not text:
        return ""
    text = re.sub(r'<ref[^>]*>[^<]*(?:</ref>)?', '', text, flags=re.DOTALL)
    text = re.sub(r'<ref[^/][^>]*/>', '', text)
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    for _ in range(5):
        new_text = re.sub(r'\{\{[^{}]*\}\}', '', text)
        if new_text == text:
            break
        text = new_text
    text = re.sub(r'\[\[(?:Category|Категория|File|Файл|Image|Изображение):[^\]]*\]\]', '', text)
    text = re.sub(r'\[\[([^\]]*\|)?([^\]]+)\]\]', r'\2', text)
    text = re.sub(r'^={1,6}\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s*={1,6}\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r"'''?", '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\{\{DEFAULTSORT:[^}]*\}\}', '', text)
    text = re.sub(r'__(?:NOTOC|FORCETOC|TOC|NOEDITSECTION|NEWSECTIONLINK|NONEWSECTIONLINK)__', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    lines = [l for l in text.split('\n') if l.strip() and not re.match(r'^[-=*#|!]+$', l.strip())]
    text = '\n'.join(lines)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


class WikiExtractor:
    def __init__(self, output_file, target_size, existing_size):
        self.in_text = False
        self.text_chars = []
        self.total_size = existing_size
        self.article_count = 0
        self.chunk_articles = 0
        self.output_file = output_file
        self.target_size = target_size
        self.done = False

    def start_element(self, name, attrs):
        if name == 'text':
            self.in_text = True
            self.text_chars = []

    def end_element(self, name):
        if name == 'text' and self.in_text:
            self.in_text = False
            raw = ''.join(self.text_chars)
            self.text_chars = []
            if not raw or len(raw) < MIN_ARTICLE_LENGTH:
                return
            cleaned = strip_wiki(raw)
            if len(cleaned) < MIN_ARTICLE_LENGTH:
                return
            with open(self.output_file, "a", encoding="utf-8") as out:
                out.write(cleaned + "\n\n")
            self.total_size += len(cleaned.encode("utf-8"))
            self.article_count += 1
            self.chunk_articles += 1
            if self.total_size >= self.target_size:
                self.done = True

    def char_data(self, data):
        if self.in_text:
            self.text_chars.append(data)


def download_chunk(url, dest):
    print(f"  Downloading...")
    resp = requests.get(url, headers=headers, stream=True, timeout=1200)
    resp.raise_for_status()
    total = int(resp.headers.get('content-length', 0))
    downloaded = 0
    start = time.time()
    with open(dest, 'wb') as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    speed = downloaded / (time.time() - start) / 1024
                    print(f"\r    {downloaded/1024/1024:.0f}/{total/1024/1024:.0f} MB ({pct:.0f}%) @ {speed:.0f} KB/s", end='')
    print()
    return dest


def extract_from_bz2(bz2_path, extractor):
    parser = xml.parsers.expat.ParserCreate()
    parser.StartElementHandler = extractor.start_element
    parser.EndElementHandler = extractor.end_element
    parser.CharacterDataHandler = extractor.char_data
    parser.buffer_text = True

    with bz2.open(bz2_path, "rb") as f:
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break
            parser.Parse(chunk)
            if extractor.done:
                return True
    return False


def main():
    if os.path.exists(OUTPUT_FILE):
        existing_size = os.path.getsize(OUTPUT_FILE)
        print(f"Existing corpus: {existing_size/1024/1024:.1f} MB")
    else:
        existing_size = 0
        print("Starting fresh corpus")

    extractor = WikiExtractor(OUTPUT_FILE, TARGET_SIZE, existing_size)
    temp_bz2 = "wiki_temp_chunk.bz2"

    total_start = time.time()
    for i, chunk_name in enumerate(CHUNKS):
        if extractor.done:
            print(f"\nTarget {TARGET_SIZE/1024/1024:.0f} MB reached, stopping.")
            break

        print(f"\n[{i+1}/{len(CHUNKS)}] {chunk_name}")

        if not os.path.exists(temp_bz2):
            url = BASE_URL + chunk_name
            download_chunk(url, temp_bz2)
        else:
            print(f"  Using existing temp file ({os.path.getsize(temp_bz2)/1024/1024:.1f} MB)")

        print(f"  Extracting...")
        extractor.chunk_articles = 0
        start = time.time()
        done = extract_from_bz2(temp_bz2, extractor)
        elapsed = time.time() - start
        print(f"  +{extractor.chunk_articles} articles, total: {extractor.article_count} art, {extractor.total_size/1024/1024:.1f} MB ({elapsed:.1f}s)")

        if os.path.exists(temp_bz2):
            os.remove(temp_bz2)

    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"Total: {extractor.article_count} articles, {extractor.total_size/1024/1024:.1f} MB")
    print(f"Time: {total_elapsed/60:.1f} min")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
