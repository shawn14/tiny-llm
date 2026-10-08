"""
Build a finance corpus from S&P 500 earnings call transcripts (FMP API).

Output:
  finance_corpus.txt        — all transcripts concatenated, doc-separated
  finance_corpus_meta.json  — manifest of what was fetched (resume-safe)

Usage:
  export FMP_API_KEY=...
  python prepare_finance_data.py --years 2024-2024 --limit-tickers 5    # smoke test
  python prepare_finance_data.py --years 2015-2025                      # full run
"""
import argparse
import json
import os
import re
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

API_KEY = os.environ.get("FMP_API_KEY")
BASE = "https://financialmodelingprep.com/api/v3"
OUT_FILE = Path("finance_corpus.txt")
META_FILE = Path("finance_corpus_meta.json")
DOC_SEP = "\n\n=== END OF TRANSCRIPT ===\n\n"


def get_sp500_tickers():
    r = requests.get(f"{BASE}/sp500_constituent", params={"apikey": API_KEY}, timeout=30)
    r.raise_for_status()
    return [c["symbol"] for c in r.json()]


def fetch_transcript(ticker, year, quarter):
    r = requests.get(
        f"{BASE}/earning_call_transcript/{ticker}",
        params={"year": year, "quarter": quarter, "apikey": API_KEY},
        timeout=30,
    )
    if r.status_code != 200:
        return None
    data = r.json()
    if not data:
        return None
    return data[0]


def clean_transcript(raw_text: str) -> str:
    """
    TODO (you write this): turn FMP's raw transcript text into clean training text.

    Input shape: a single string. FMP transcripts look roughly like:

        Operator: Good afternoon, and welcome to the Apple Q4 2024 earnings call...
        Tim Cook: Thank you, operator. Today we're reporting...
        [some transcripts have timestamps like "10:32 AM ET"]
        [unicode quotes and dashes mixed with ASCII]
        [occasional HTML entities like &amp;, &nbsp;]
        Analyst (Goldman Sachs): My question is about...

    You decide what stays and what goes. Each choice shapes what the model learns:

      • Speaker tags ("Tim Cook:", "Operator:") — keep them and the model learns
        dialogue structure (who speaks, turn-taking). Strip them and you get pure
        prose, which trains a better "writing finance" model but loses the call
        format. RECOMMENDED: keep.

      • Section markers ("Prepared Remarks", "Question-and-Answer Session") —
        keep them and the model learns that calls have structure. RECOMMENDED: keep.

      • Timestamps ("10:32 AM ET", "[2:15]") — almost always noise. RECOMMENDED: strip.

      • Smart quotes / em-dashes / non-ASCII — for char-level training (95-char
        vocab), MUST normalize to ASCII or it'll explode the vocab. For BPE
        (Llama), keep them. RECOMMENDED for now: NFKC normalize, then map common
        unicode (smart quotes → straight, em-dash → "--").

      • Whitespace — collapse runs of spaces, cap blank lines at 2.

      • HTML entities — decode them (&amp; → &, &nbsp; → space).

    Return the cleaned string. Aim for ~5-15 lines.
    """
    raise NotImplementedError("clean_transcript: see docstring above")


def format_transcript(t) -> str:
    header = f"# {t['symbol']} Q{t['quarter']} {t['year']} Earnings Call ({t['date']})"
    body = clean_transcript(t["content"])
    return f"{header}\n\n{body}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default="2015-2025", help="e.g. 2015-2025 inclusive")
    ap.add_argument("--quarters", default="1,2,3,4")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit-tickers", type=int, default=None,
                    help="for smoke testing — only fetch first N tickers")
    args = ap.parse_args()

    if not API_KEY:
        raise SystemExit("Set FMP_API_KEY environment variable.")

    y0, y1 = (int(x) for x in args.years.split("-"))
    quarters = [int(q) for q in args.quarters.split(",")]

    print(f"[1/3] Fetching S&P 500 constituents...")
    tickers = get_sp500_tickers()
    if args.limit_tickers:
        tickers = tickers[: args.limit_tickers]
    print(f"      {len(tickers)} tickers")

    jobs = [(t, y, q) for t in tickers for y in range(y0, y1 + 1) for q in quarters]
    print(f"[2/3] {len(jobs)} (ticker, year, quarter) combinations")

    fetched = []
    docs_written = 0
    chars_written = 0
    if META_FILE.exists():
        meta = json.loads(META_FILE.read_text())
        fetched = meta.get("fetched", [])
        docs_written = meta.get("docs_written", 0)
        chars_written = meta.get("chars_written", 0)
        already = set(fetched)
        before = len(jobs)
        jobs = [j for j in jobs if f"{j[0]}_{j[1]}_{j[2]}" not in already]
        print(f"      resuming — {before - len(jobs)} already done, {len(jobs)} remaining")

    print(f"[3/3] Fetching transcripts ({args.workers} workers)...")
    out = OUT_FILE.open("a", encoding="utf-8")
    t_start = time.time()
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(fetch_transcript, t, y, q): (t, y, q) for (t, y, q) in jobs}
            for i, fut in enumerate(as_completed(futures)):
                t, y, q = futures[fut]
                key = f"{t}_{y}_{q}"
                try:
                    transcript = fut.result()
                except Exception as e:
                    print(f"      ✗ {key}: {e}")
                    continue
                fetched.append(key)
                if transcript and transcript.get("content"):
                    text = format_transcript(transcript)
                    out.write(text + DOC_SEP)
                    docs_written += 1
                    chars_written += len(text)
                if (i + 1) % 50 == 0:
                    rate = (i + 1) / (time.time() - t_start)
                    eta = (len(jobs) - (i + 1)) / max(rate, 0.01) / 60
                    print(f"      [{i+1}/{len(jobs)}] {docs_written} docs, "
                          f"{chars_written/1e6:.1f}M chars, "
                          f"{rate:.1f} req/s, ETA {eta:.1f} min")
                if (i + 1) % 200 == 0:
                    META_FILE.write_text(json.dumps({
                        "fetched": fetched,
                        "docs_written": docs_written,
                        "chars_written": chars_written,
                    }))
                    out.flush()
    finally:
        out.close()
        META_FILE.write_text(json.dumps({
            "fetched": fetched,
            "docs_written": docs_written,
            "chars_written": chars_written,
        }))

    print()
    print(f"Done. {docs_written} transcripts → {OUT_FILE}")
    print(f"  {chars_written/1e6:.1f}M chars")
    print(f"  ~{chars_written/4/1e6:.0f}M tokens (BPE estimate, 4 chars/token)")
    print(f"  ~{chars_written/1e6:.0f}M chars for char-level model")


if __name__ == "__main__":
    main()
