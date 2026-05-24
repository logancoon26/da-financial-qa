"""
Input:   data/chunks.jsonl      (from clean.py)
Output:  data/faiss.index       (the vector index)
         data/chunk_store.json  (the text + metadata for each indexed chunk)
"""

import json
import re
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# ── Config ────────────────────────────────────────────────────────────────────

CHUNKS_PATH  = "data/chunks.jsonl"
INDEX_PATH   = "data/faiss.index"
STORE_PATH   = "data/chunk_store.json"

EMBED_MODEL  = "BAAI/bge-m3"
BATCH_SIZE   = 2


# ── Build index ───────────────────────────────────────────────────────────────

def load_chunks(path: str) -> tuple[list[str], list[dict]]:
    """Load chunk texts and metadata from JSONL."""
    texts, metas = [], []

    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            texts.append(rec["text"])
            metas.append(rec["metadata"])

    print(f"Loaded {len(texts):,} chunks")
    return texts, metas


def embed_chunks(texts: list[str], model: SentenceTransformer) -> np.ndarray:
    """
    Embed all chunks into dense vectors.

    normalize_embeddings=True makes cosine similarity = dot product.
    """
    print(f"Embedding {len(texts):,} chunks with {EMBED_MODEL}…")

    all_embeddings = []

    for i in tqdm(range(0, len(texts), BATCH_SIZE)):
        batch = texts[i:i + BATCH_SIZE]

        embeddings = model.encode(
            batch,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        all_embeddings.append(embeddings)

    return np.vstack(all_embeddings).astype("float32")


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build exact cosine similarity FAISS index.
    """
    dim = embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    print(f"FAISS index built: {index.ntotal:,} vectors, dim={dim}")

    return index

def normalize_company_name(name: str) -> str:
    """
    Normalize company names for fuzzy matching + entity detection.
    """

    name = name.lower()

    # 1. remove possessive "'s"
    name = re.sub(r"'s\b", "", name)

    # 2. remove remaining apostrophes
    name = name.replace("'", "")

    # 3. remove punctuation (but keep spaces/numbers)
    name = re.sub(r"[^\w\s]", " ", name)

    # 4. normalize whitespace
    name = re.sub(r"\s+", " ", name).strip()

    # 5. suffix + stopword removal (important upgrade)
    stopwords = {
        "of", "the", "and", "amp"
    }

    suffixes = {
        "inc", "incorporated",
        "corp", "corporation",
        "co", "company",
        "ltd", "limited",
        "llc", "plc",
        "holdings", "group"
    }

    words = [
        w for w in name.split()
        if w not in suffixes and w not in stopwords
    ]

    return " ".join(words).strip()

def company_match(a: str, b: str) -> bool:
    return normalize_company_name(a) == normalize_company_name(b)


# ── Retrieval ─────────────────────────────────────────────────────────────────

class Retriever:
    """
    Wrapper around FAISS index + chunk store.

    Added metadata-aware retrieval:
    If a company is detected in the query, retrieval is filtered
    to chunks from that company only.
    """

    def __init__(
        self,
        index_path: str = INDEX_PATH,
        store_path: str = STORE_PATH,
        embed_model: str = EMBED_MODEL,
    ):
        print("Loading retriever…")

        self.index = faiss.read_index(index_path)

        with open(store_path, encoding="utf-8") as f:
            store = json.load(f)

        self.texts = store["texts"]
        self.metas = store["metadata"]

        self.model = SentenceTransformer(embed_model)

        print(f"Retriever ready: {self.index.ntotal:,} chunks indexed")

    # ── Company detection ─────────────────────────────────────────────────────

    def detect_company(self, query: str):
        """
        Detect company mentioned in query using normalized matching.
        """

        query_norm = normalize_company_name(query)

        best_match = None
        best_score = 0

        for meta in self.metas:
            company = meta["company"]
            company_norm = normalize_company_name(company)

            # 1. exact match (strongest signal)
            if company_norm in query_norm:
                return company

        # 2. weak match (score-based, not first-hit)
        query_tokens = set(query_norm.split())

        for meta in self.metas:
            company = meta["company"]
            company_norm = normalize_company_name(company)

            company_tokens = set(company_norm.split())
            overlap = company_tokens & query_tokens

            score = len(overlap)

            if score > best_score and score >= 2:   # important threshold
                best_score = score
                best_match = company

        return best_match

    # ── Retrieval ────────────────────────────────────────────────────────────

    def retrieve(self, query: str, k: int = 5) -> list[dict]:
        """
        Retrieve top-k relevant chunks.

        If company detected:
            search larger candidate pool
            then filter to matching company
        """

        query_vec = self.model.encode(
            [query],
            normalize_embeddings=True,
        ).astype("float32")

        detected_company = self.detect_company(query)

        # Search more candidates if filtering afterward
        search_k = 50 if detected_company else k

        scores, indices = self.index.search(query_vec, search_k)

        results = []

        for score, idx in zip(scores[0], indices[0]):

            if idx == -1:
                continue

            meta = self.metas[idx]

            # Metadata-aware filtering
            if detected_company:
                if not company_match(meta["company"], detected_company):
                    continue

            results.append({
                "text": self.texts[idx],
                "score": float(score),
                "metadata": meta,
            })

            # Stop once enough valid chunks collected
            if len(results) >= k:
                break

        return results


# ── Main ─────────────────────────────────────────────────────────────────────

def main():

    # Load chunks
    texts, metas = load_chunks(CHUNKS_PATH)

    # Load embedding model
    print(f"\nLoading embedding model: {EMBED_MODEL}")

    model = SentenceTransformer(
        EMBED_MODEL,
        device="cuda",
    )

    # Embed chunks
    embeddings = embed_chunks(texts, model)

    # Build FAISS index
    print()

    index = build_faiss_index(embeddings)

    # Save outputs
    Path(INDEX_PATH).parent.mkdir(exist_ok=True)

    faiss.write_index(index, INDEX_PATH)
    print(f"✓ FAISS index saved → {INDEX_PATH}")

    with open(STORE_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "texts": texts,
            "metadata": metas,
        }, f)

    print(f"✓ Chunk store saved → {STORE_PATH}")

    # ── Sanity check ────────────────────────────────────────────────────────

    print("\n── Sanity check ─────────────────────────────────────────────")

    retriever = Retriever()

    query = "What were Apple's main revenue drivers?"

    results = retriever.retrieve(query, k=3)

    print(f"\nQuery: {query}")

    detected = retriever.detect_company(query)

    print(f"Detected company: {detected}")

    for i, r in enumerate(results):

        print(
            f"\n[{i+1}] "
            f"Score={r['score']:.3f} | "
            f"{r['metadata']['company']} "
            f"{r['metadata']['period']} | "
            f"{r['metadata']['section']}"
        )

        print(f"    {r['text'][:200]}…")


if __name__ == "__main__":
    main()