"""
Download Russian Wikipedia dump chunk, extract clean text.
Uses xml.parsers.expat for streaming XML parsing (no tree in memory).
"""

import requests, bz2, xml.parsers.expat, re, os, time, sys

DUMP_URL = "https://dumps.wikimedia.org/ruwiki/latest/ruwiki-latest-pages-articles1.xml-p1p224167.bz2"
OUTPUT_FILE = "training_data_raw.txt"
TARGET_SIZE = 30 * 1024 * 1024
CHUNK_SIZE = 65536
headers = {"User-Agent": "GAI-Training/1.0"}

MIN_ARTICLE_LENGTH = 100  # minimum chars for an article to keep


def strip_wiki(text):
    """Strip basic wiki markup from article text."""
    if not text:
        return ""
    text = re.sub(r'<ref[^>]*>[^<]*(?:</ref>)?', '', text, flags=re.DOTALL)
    text = re.sub(r'<ref[^/][^>]*/>', '', text)
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    # templates (multiple passes for nested)
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


def download_file(url, dest):
    """Download a file with progress."""
    print(f"Downloading {url.split('/')[-1]}...")
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
                    print(f"\r  {downloaded/1024/1024:.1f}/{total/1024/1024:.1f} MB ({pct:.0f}%) @ {speed:.0f} KB/s", end='')
    print()
    return dest


def extract_text_from_dump(bz2_path):
    """Extract article text from Wikipedia XML dump using streaming expat parser."""
    total_size = 0
    article_count = 0

    # Reset output file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        pass

    class WikiHandler:
        def __init__(self):
            self.in_text = False
            self.text_chars = []
            self.text_len = 0
            self.done = False

        def start_element(self, name, attrs):
            if name == 'text':
                self.in_text = True
                self.text_chars = []
                self.text_len = 0

        def end_element(self, name):
            nonlocal total_size, article_count
            if name == 'text' and self.in_text:
                self.in_text = False
                raw = ''.join(self.text_chars)
                self.text_chars = []
                if len(raw) < MIN_ARTICLE_LENGTH:
                    return
                cleaned = strip_wiki(raw)
                if len(cleaned) < MIN_ARTICLE_LENGTH:
                    return
                with open(OUTPUT_FILE, "a", encoding="utf-8") as out:
                    out.write(cleaned + "\n\n")
                total_size += len(cleaned.encode("utf-8"))
                article_count += 1
                if article_count % 500 == 0:
                    print(f"  {article_count} articles, {total_size/1024/1024:.1f} MB")
                if total_size >= TARGET_SIZE:
                    self.done = True

        def char_data(self, data):
            if self.in_text:
                self.text_chars.append(data)

    handler = WikiHandler()
    parser = xml.parsers.expat.ParserCreate()
    parser.StartElementHandler = handler.start_element
    parser.EndElementHandler = handler.end_element
    parser.CharacterDataHandler = handler.char_data
    parser.buffer_text = True

    with bz2.open(bz2_path, "rb") as f:
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break
            parser.Parse(chunk)
            if handler.done:
                print("  Target reached, stopping parse early")
                # Feed remaining data to avoid warning
                while True:
                    remaining = f.read(CHUNK_SIZE)
                    if not remaining:
                        break
                break

    return article_count, total_size


def main():
    temp_bz2 = "wiki_temp_dump.bz2"

    if not os.path.exists(temp_bz2):
        start = time.time()
        download_file(DUMP_URL, temp_bz2)
        print(f"Downloaded {os.path.getsize(temp_bz2)/1024/1024:.1f} MB in {(time.time()-start)/60:.1f} min")
    else:
        print(f"Using existing file {temp_bz2} ({os.path.getsize(temp_bz2)/1024/1024:.1f} MB)")

    print("\nExtracting text from XML (streaming)...")
    start = time.time()
    articles, text_size = extract_text_from_dump(temp_bz2)
    elapsed = time.time() - start
    print(f"\nExtracted: {articles} articles, {text_size/1024/1024:.2f} MB clean text")
    print(f"Extraction time: {elapsed/60:.1f} min")
    print(f"Output: {OUTPUT_FILE} ({os.path.getsize(OUTPUT_FILE)/1024/1024:.1f} MB)")

    # Cleanup
    if os.path.exists(temp_bz2):
        os.remove(temp_bz2)
        print("Temp file cleaned up")


if __name__ == "__main__":
    main()
