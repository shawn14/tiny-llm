"""
Build a richer training corpus for v4.

Combines:
  - Tiny Shakespeare (input.txt — what train.py was using)
  - Complete Works of Shakespeare (Project Gutenberg #100)
  - Plays of Christopher Marlowe (Project Gutenberg #779)

Strips Gutenberg headers/footers and filters to the EXACT 65-character
vocab from input.txt — preserves the site's narrative + lets v4 use the
same tokenizer/decoder code as v1/v2/v3.

Output: input_v4.txt + diagnostic stats.
"""
import os
import urllib.request

# 1) Anchor vocab on the original tiny shakespeare file
if not os.path.exists("input.txt"):
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        "input.txt",
    )
ORIGINAL = open("input.txt").read()
VOCAB = set(ORIGINAL)
print(f"Anchor vocab: {len(VOCAB)} chars from input.txt ({len(ORIGINAL)/1e6:.2f}M chars)")
print(f"Vocab: {''.join(sorted(VOCAB))!r}")
print()

# 2) Source list — Gutenberg cache URLs are stable
SOURCES = [
    ("complete_shakespeare.txt",
     "https://www.gutenberg.org/cache/epub/100/pg100.txt"),
    ("marlowe_plays.txt",
     "https://www.gutenberg.org/cache/epub/779/pg779.txt"),
]

START_MARKERS = (
    "*** START OF THE PROJECT GUTENBERG",
    "*** START OF THIS PROJECT GUTENBERG",
    "***START OF THE PROJECT GUTENBERG",
)
END_MARKERS = (
    "*** END OF THE PROJECT GUTENBERG",
    "*** END OF THIS PROJECT GUTENBERG",
    "***END OF THE PROJECT GUTENBERG",
)

def strip_gutenberg(text):
    """Cut anything before the START marker and after the END marker."""
    for marker in START_MARKERS:
        i = text.find(marker)
        if i >= 0:
            nl = text.find("\n", i)
            text = text[nl + 1:] if nl >= 0 else text[i + len(marker):]
            break
    for marker in END_MARKERS:
        i = text.find(marker)
        if i >= 0:
            text = text[:i]
            break
    return text

def fetch(url, cache_path):
    if not os.path.exists(cache_path):
        print(f"  downloading {url}")
        urllib.request.urlretrieve(url, cache_path)
    return open(cache_path, encoding="utf-8", errors="replace").read()

# 3) Build corpus
parts = [ORIGINAL]  # keep tinyshakespeare in the mix
print("Fetching extra sources:")
for cache, url in SOURCES:
    raw = fetch(url, cache)
    body = strip_gutenberg(raw)
    kept = "".join(c for c in body if c in VOCAB)
    print(f"  {cache:30s} raw {len(raw)/1e6:5.2f}M → body {len(body)/1e6:5.2f}M → "
          f"in-vocab {len(kept)/1e6:5.2f}M  ({(1 - len(kept)/max(len(body),1))*100:4.1f}% chars dropped)")
    parts.append(kept)

combined = "\n\n".join(parts)
with open("input_v4.txt", "w") as f:
    f.write(combined)

# 4) Final stats
final_vocab = set(combined)
print()
print(f"Wrote input_v4.txt: {len(combined)/1e6:.2f}M chars "
      f"({len(combined) / len(ORIGINAL):.1f}× tiny shakespeare)")
print(f"Vocab preserved: {final_vocab == VOCAB}  "
      f"(set len {len(final_vocab)} vs anchor {len(VOCAB)})")
if final_vocab != VOCAB:
    print(f"  drift: anchor−combined = {VOCAB - final_vocab}")
    print(f"         combined−anchor = {final_vocab - VOCAB}")
