"""
Clean up remaining wiki artifacts from training corpus.
Reads existing corpus, applies additional cleanup, writes clean version.
"""
import re, os

INPUT = "training_data_full.txt"
OUTPUT = "training_data_clean.txt"
MAX_BYTES = 1_200_000_000  # Process first 1.2 GB (what we use for training)
CHUNK_CHARS = 20_000_000

def clean_text(text):
    """Second-pass cleanup: remove artifacts missed by strip_wiki."""
    # Remove wiki signatures ~~~~ and UTC timestamps
    text = re.sub(r'--~~~~', '', text)
    text = re.sub(r'~~~~', '', text)
    text = re.sub(r'\d{1,2}:\d{2}, \d{1,2} \w+ \d{4} \(UTC\)', '', text)
    text = re.sub(r'\d{1,2}:\d{2}, \d{1,2} \w+ \d{4}', '', text)

    # Remove wiki table markup lines
    text = re.sub(r'^\{|\|[\|\-\+\!\}]', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\|-\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\|\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\!\s*$', '', text, flags=re.MULTILINE)

    # Remove style/align/class attributes leftover from tables
    text = re.sub(r'\|\s*(?:style|align|valign|width|bgcolor|colspan|rowspan)="[^"]*"', '', text)
    text = re.sub(r'\|\s*(?:style|align|valign|width|bgcolor|colspan|rowspan)=\'[^\']*\'', '', text)
    # standalone style lines
    text = re.sub(r'^\s*(?:style|align|valign|width|bgcolor|colspan|rowspan)\s*=.*$', '', text, flags=re.MULTILINE)

    # Remove remaining HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)

    # Remove single pipes and double pipes at line start
    text = re.sub(r'^\s*\|\|?\s*', '', text, flags=re.MULTILINE)

    # Remove lines that are just whitespace/punctuation
    lines = []
    for line in text.split('\n'):
        stripped = line.strip()
        # Skip lines that are just separators or wiki artifacts
        if stripped and not re.match(r'^[-=*#|!:.\[\]{}<>]+$', stripped):
            lines.append(line)
    text = '\n'.join(lines)

    # Collapse multiple newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)

    return text.strip()

def main():
    in_size = os.path.getsize(INPUT)
    print(f"Input: {INPUT} ({in_size/1024/1024/1024:.2f} GB)")
    print(f"Processing first {MAX_BYTES/1024/1024:.0f} MB...")

    with open(OUTPUT, "w", encoding="utf-8") as out:
        pass  # truncate

    total_read = 0
    total_written = 0
    char_count = 0

    with open(INPUT, "r", encoding="utf-8") as f:
        while total_read < MAX_BYTES:
            raw = f.read(CHUNK_CHARS)
            if not raw:
                break
            total_read += len(raw.encode("utf-8"))

            cleaned = clean_text(raw)
            with open(OUTPUT, "a", encoding="utf-8") as out:
                out.write(cleaned + "\n\n")

            char_count += len(cleaned)
            total_written += len(cleaned.encode("utf-8"))

            if char_count % 5_000_000 < CHUNK_CHARS:
                print(f"  Read: {total_read/1024/1024:.0f} MB | Written: {total_written/1024/1024:.0f} MB | Chars: {char_count:,}")

    # Final
    final_size = os.path.getsize(OUTPUT)
    compression = 1 - final_size / total_read if total_read else 0
    print(f"\nDone! Cleaned {total_read/1024/1024:.0f} MB -> {final_size/1024/1024:.0f} MB ({compression*100:.0f}% reduction)")
    print(f"Output: {OUTPUT}")

    # Show sample comparison
    print("\nSample (first 300 chars):")
    with open(OUTPUT, "r", encoding="utf-8") as f:
        sample = f.read(300)
    print(sample)

if __name__ == "__main__":
    main()
