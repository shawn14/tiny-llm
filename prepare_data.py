"""
Download a 10M-token sample of FineWeb-Edu — the modern open-source
educational web-text dataset used to train small LMs.

Saves to: corpus.txt  (~50 MB raw text)

Why FineWeb-Edu over tinyshakespeare:
  - 50× more data → less overfitting at modern model size
  - Modern English instead of one author's archaic style
  - Mirrors what real small-LLM training uses

Run:
    pip install datasets
    python prepare_data.py
"""
import os
import re
import sys

try:
    from datasets import load_dataset
except ImportError:
    sys.exit("Need `datasets`: run  pip install datasets")

OUT_FILE = "corpus.txt"
TARGET_BYTES = 50_000_000   # ~10M tokens at ~5 bytes/token

PRINTABLE_RE = re.compile(r'[^\x20-\x7E\n]')      # keep only printable ASCII + newline
WS_RE        = re.compile(r'\n{3,}')               # collapse triple+ newlines


def normalize(text: str) -> str:
    text = PRINTABLE_RE.sub('', text)
    text = WS_RE.sub('\n\n', text)
    return text.strip()


def main():
    if os.path.exists(OUT_FILE):
        size_mb = os.path.getsize(OUT_FILE) / 1e6
        print(f"{OUT_FILE} already exists ({size_mb:.1f} MB). Delete it to re-download.")
        return

    print("Streaming FineWeb-Edu (sample-10BT)...")
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        "sample-10BT",
        split="train",
        streaming=True,
    )

    written = 0
    docs = 0
    with open(OUT_FILE, "w") as f:
        for ex in ds:
            text = normalize(ex["text"])
            if len(text) < 200:
                continue
            f.write(text + "\n\n")
            written += len(text) + 2
            docs += 1
            if docs % 200 == 0:
                print(f"  ...{docs} docs, {written/1e6:.1f} MB")
            if written >= TARGET_BYTES:
                break

    final_mb = os.path.getsize(OUT_FILE) / 1e6
    print(f"\nWrote {docs} documents to {OUT_FILE} ({final_mb:.1f} MB)")
    print(f"Approx tokens (chars/5): {int(written/5):,}")
    print(f"\nVocab preview:")
    chars = sorted(set(open(OUT_FILE).read()))
    print(f"  {len(chars)} unique characters")
    print(f"  {''.join(chars[:80])}...")


if __name__ == "__main__":
    main()
