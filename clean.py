"""
Input:   data/filings.json
         data/raw/*.html

Output:  data/chunks.jsonl
         (one chunk per line, ready for RAG + fine-tuning)

Pipeline:
  HTML  → strip_html() → raw text
        → clean_text() → noise removed
        → extract_sections() → structured SEC sections
        → split_sections() → sub-section refinement (MD&A)
        → chunk_section() → semantic + token-aware chunks
"""

import json
import re
from pathlib import Path
from bs4 import BeautifulSoup


# ─────────────────────────────────────────────────────────────────────────────
# SECTION MAPPING (SEC 10-K structure)
# ─────────────────────────────────────────────────────────────────────────────

SECTION_MAP = {
    "item 1": "business",
    "item 1a": "risk_factors",
    "item 7": "mda",
    "item 7a": "market_risk",
    "item 8": "financial_statements",
}

ITEM_RE = re.compile(r"^\s*(?:ITEM|Item)\s+(\d+[A-Za-z]?)[.\s]", re.MULTILINE)


# ─────────────────────────────────────────────────────────────────────────────
# HTML CLEANING
# ─────────────────────────────────────────────────────────────────────────────

def strip_html(html: str) -> str:
    if len(html) > 20 * 1024 * 1024:
        html = html[:20 * 1024 * 1024]

    html = re.sub(r'\s+ix:[a-z]+=["\'][^"\']*["\']', "", html, flags=re.I)

    soup = BeautifulSoup(html, "lxml")

    for tag in soup(["script", "style", "head", "nav", "footer", "form"]):
        tag.decompose()

    return soup.get_text(separator="\n")


# ─────────────────────────────────────────────────────────────────────────────
# TEXT CLEANING
# ─────────────────────────────────────────────────────────────────────────────

NOISE_PATTERNS = [
    re.compile(r"^[_\-=]{4,}$"),
    re.compile(r"^\s*Exhibit\s+\d+", re.I),
    re.compile(r"^\s*(Table of Contents|PART [IVX]+)\s*$", re.I),
]

def clean_text(text: str) -> str:
    lines = []
    for line in text.split("\n"):
        s = line.strip()

        if len(s) < 4:
            continue
        if any(p.match(s) for p in NOISE_PATTERNS):
            continue

        lines.append(line)

    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_sections(text: str) -> dict[str, str]:
    sections = {}
    current_key = None
    buffer = []

    for line in text.split("\n"):
        m = ITEM_RE.match(line)

        if m:
            if current_key and buffer:
                sections[current_key] = "\n".join(buffer).strip()

            item_label = f"item {m.group(1).lower()}"
            current_key = SECTION_MAP.get(item_label)
            buffer = []

        elif current_key:
            buffer.append(line)

    if current_key and buffer:
        sections[current_key] = "\n".join(buffer).strip()

    return sections


# ─────────────────────────────────────────────────────────────────────────────
# MD&A SUB-STRUCTURING (IMPORTANT IMPROVEMENT)
# ─────────────────────────────────────────────────────────────────────────────

MDA_SUBHEADERS = [
    "results of operations",
    "liquidity and capital resources",
    "critical accounting estimates",
]

def split_mda(text: str) -> list[str]:
    """
    Splits MD&A into sub-blocks if possible.
    Falls back to full text if no headers found.
    """
    pattern = re.compile(r"(" + "|".join(MDA_SUBHEADERS) + r")", re.I)

    splits = pattern.split(text)
    if len(splits) <= 1:
        return [text]

    blocks = []
    current = []

    for part in splits:
        if pattern.match(part):
            if current:
                blocks.append("\n".join(current))
            current = [part]
        else:
            current.append(part)

    if current:
        blocks.append("\n".join(current))

    return [b.strip() for b in blocks if len(b.strip()) > 0]


# ─────────────────────────────────────────────────────────────────────────────
# SENTENCE SPLITTING
# ─────────────────────────────────────────────────────────────────────────────

ABBREVS = [
    "Inc", "Corp", "Ltd", "Co", "vs", "U.S", "p.m", "a.m",
    "Dr", "Mr", "Ms", "No", "approx", "est"
]

def split_sentences(text: str) -> list[str]:
    protected = text

    for ab in ABBREVS:
        protected = protected.replace(f"{ab}.", f"{ab}DOT")

    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z"\'])', protected)

    return [s.replace("DOT", ".").strip() for s in sentences if s.strip()]


# ─────────────────────────────────────────────────────────────────────────────
# CHUNKING (IMPROVED CORE LOGIC)
# ─────────────────────────────────────────────────────────────────────────────

TOPIC_KEYWORDS = [
    re.compile(r"\b(revenue|income|earnings|net sales)\b", re.I),
    re.compile(r"\b(segment|geographic|product)\b", re.I),
    re.compile(r"\b(risk|uncertainty|forward-looking)\b", re.I),
]

def is_topic_shift(prev: str, curr: str) -> bool:
    if not prev:
        return False

    for p in TOPIC_KEYWORDS:
        if bool(p.search(prev)) != bool(p.search(curr)):
            return True

    return False


def chunk_text(text: str, max_tokens: int = 400, overlap_tokens: int = 60) -> list[str]:
    sentences = split_sentences(text)

    chunks = []
    current = []
    current_len = 0

    i = 0
    while i < len(sentences):
        sent = sentences[i]
        sent_len = len(sent.split())

        # ── FORCE CHUNK BREAK ON TOPIC SHIFT ──
        if (
            current
            and current_len > 150
            and is_topic_shift(current[-1], sent)
        ):
            chunk = " ".join(current).strip()
            if len(chunk) > 80:
                chunks.append(chunk)

            current, current_len = [], 0

        if current_len + sent_len > max_tokens and current:
            chunk = " ".join(current).strip()
            if len(chunk) > 80:
                chunks.append(chunk)

            # overlap via token tail
            overlap = []
            total = 0

            for s in reversed(current):
                wc = len(s.split())
                if total + wc > overlap_tokens:
                    break
                overlap.insert(0, s)
                total += wc

            current = overlap
            current_len = total

        current.append(sent)
        current_len += sent_len
        i += 1

    if current:
        chunk = " ".join(current).strip()
        if len(chunk) > 80:
            chunks.append(chunk)

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def main():
    filings_path = Path("data/filings.json")
    if not filings_path.exists():
        print("Run fetch step first.")
        return

    filings = json.loads(filings_path.read_text())
    out_path = Path("data/chunks.jsonl")
    out_path.parent.mkdir(exist_ok=True)

    total_chunks = 0
    section_counts = {}

    with open(out_path, "w", encoding="utf-8") as f:

        for filing in filings:
            html_path = Path(filing["html_path"])

            if not html_path.exists():
                continue

            print(f"Processing {filing['company']} {filing['period']}...")

            html = html_path.read_text(encoding="utf-8", errors="ignore")

            raw = strip_html(html)
            clean = clean_text(raw)
            sections = extract_sections(clean)

            for section_name, text in sections.items():

                if len(text.split()) < 100:
                    continue

                # ── MD&A SUB-SPLITTING (KEY IMPROVEMENT) ──
                if section_name == "mda":
                    sub_blocks = split_mda(text)
                else:
                    sub_blocks = [text]

                for block in sub_blocks:
                    chunks = chunk_text(block)

                    for idx, chunk in enumerate(chunks):
                        record = {
                            "text": chunk,

                            "messages": [
                                {
                                    "role": "system",
                                    "content": (
                                        f"You are a financial analyst. "
                                        f"This is from the {section_name.upper()} "
                                        f"section of {filing['company']} "
                                        f"for fiscal year ending {filing['period']}."
                                    ),
                                },
                                {
                                    "role": "user",
                                    "content": "What are the key points from this section?",
                                },
                                {
                                    "role": "assistant",
                                    "content": chunk,
                                },
                            ],

                            "metadata": {
                                "company": filing["company"],
                                "cik": filing["cik"],
                                "accession": filing["accession"],
                                "period": filing["period"],
                                "section": section_name,
                                "chunk_idx": idx,
                            },
                        }

                        f.write(json.dumps(record) + "\n")
                        total_chunks += 1
                        section_counts[section_name] = section_counts.get(section_name, 0) + 1

    print(f"\n✓ {total_chunks} chunks written to {out_path}")

    print("\nSection breakdown:")
    for k, v in sorted(section_counts.items(), key=lambda x: -x[1]):
        print(f"{k:<20} {v}")


if __name__ == "__main__":
    main()