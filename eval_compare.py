"""
Configurations compared:
  1. Fine-tuned model + RAG
  2. Base model + RAG
  3. Fine-tuned model, no RAG
  4. Base model, no RAG

Metrics:
  - Answer Similarity : cosine similarity(answer, reference)
  - Answer Relevance  : cosine similarity(question, answer)
  - Faithfulness      : ROUGE-L precision(answer vs retrieved context)
  - Context Relevance : cosine similarity(question, retrieved chunks)

Output:
  outputs/eval_results.json
"""

import json
from pathlib import Path

import numpy as np
from rouge_score import rouge_scorer as rouge_lib
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

from rag import Retriever
from rag_core import load_generator, generate, build_no_rag_prompt, build_rag_prompt


# ── Config ────────────────────────────────────────────────────────────────────

BASE_MODEL      = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MERGED_DIR      = "models/merged"

TOP_K           = 3
MAX_NEW_TOKENS  = 256


SYSTEM_PROMPT = """\
You are a financial analyst assistant with expertise in SEC filings.
Answer the user's question using ONLY the provided context excerpts.
If the context doesn't contain enough information, say so clearly.\
"""


SYSTEM_PROMPT_NO_RAG = """\
You are a financial analyst assistant with expertise in SEC filings.
Answer the question as accurately as possible based on your knowledge.\
"""


# ── Golden QA set ─────────────────────────────────────────────────────────────

GOLDEN_QA = [
    {
        "question": "What were Apple's main product revenue categories?",
        "reference": (
            "Apple's revenue categories include iPhone, Mac, iPad, "
            "Wearables and Services. iPhone is the largest segment."
        ),
    },
    {
        "question": "How did NVIDIA's data center revenue change with AI demand?",
        "reference": (
            "NVIDIA's data center revenue grew significantly driven by "
            "demand for AI and machine learning workloads."
        ),
    },
    {
        "question": "What risks does Microsoft identify around cloud computing competition?",
        "reference": (
            "Microsoft identifies risks including intense competition "
            "from AWS and Google Cloud and the need for continued "
            "infrastructure investment."
        ),
    },
    {
        "question": "What is Berkshire Hathaway's approach to capital allocation?",
        "reference": (
            "Berkshire focuses on acquiring businesses with durable "
            "competitive advantages and maintaining large cash reserves."
        ),
    },
    {
        "question": "How does Amazon describe its fulfillment network strategy?",
        "reference": (
            "Amazon describes continued investment in fulfillment "
            "center expansion and last-mile delivery to reduce "
            "delivery times and costs."
        ),
    },
    {
        "question": "What were JPMorgan's main sources of revenue?",
        "reference": (
            "JPMorgan's main revenue sources include net interest income, "
            "investment banking fees, and asset management."
        ),
    },
    {
        "question": "How did Alphabet describe its advertising business?",
        "reference": (
            "Alphabet's advertising revenue is driven primarily by "
            "Google Search and YouTube ads."
        ),
    },
    {
        "question": "What did Meta say about its Reality Labs segment?",
        "reference": (
            "Meta's Reality Labs segment reported significant operating "
            "losses while investing in virtual and augmented reality products."
        ),
    },
]

# ── Metrics ───────────────────────────────────────────────────────────────────

def answer_similarity(answer: str, reference: str, embed_model) -> float:
    """
    Cosine similarity between generated answer and reference answer.
    """

    vecs = embed_model.encode(
        [answer, reference],
        normalize_embeddings=True,
    )

    return float(np.dot(vecs[0], vecs[1]))


def answer_relevance(question: str, answer: str, embed_model) -> float:
    """
    Cosine similarity between question and generated answer.
    """

    vecs = embed_model.encode(
        [question, answer],
        normalize_embeddings=True,
    )

    return float(np.dot(vecs[0], vecs[1]))


def faithfulness(answer: str, chunks: list[dict]) -> float:
    """
    ROUGE-L precision against retrieved chunks.
    """

    scorer = rouge_lib.RougeScorer(
        ["rougeL"],
        use_stemmer=True,
    )

    scores = [
        scorer.score(chunk["text"], answer)["rougeL"].precision
        for chunk in chunks
    ]

    return max(scores) if scores else 0.0


def context_relevance(
    question: str,
    chunks: list[dict],
    embed_model,
) -> float:
    """
    Max cosine similarity between question and retrieved chunks.
    """

    if not chunks:
        return 0.0

    q_vec = embed_model.encode(
        [question],
        normalize_embeddings=True,
    )

    c_vecs = embed_model.encode(
        [c["text"] for c in chunks],
        normalize_embeddings=True,
    )

    sims = q_vec @ c_vecs.T

    return float(np.max(sims))


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_config(
    name: str,
    generator,
    use_rag: bool,
    qa_pairs: list[dict],
    retriever: Retriever,
    embed_model,
) -> dict:

    print(f"\n  Evaluating: {name}")

    results = []

    for qa in qa_pairs:

        question = qa["question"]
        reference = qa["reference"]

        # ── Retrieve ────────────────────────────────────────────────────────

        chunks = (
            retriever.retrieve(question, k=TOP_K)
            if use_rag else []
        )

        # ── Generate ────────────────────────────────────────────────────────

        prompt = (
            build_rag_prompt(question, chunks)
            if use_rag
            else build_no_rag_prompt(question)
        )

        answer = generate(generator, prompt)

        # ── Metrics ─────────────────────────────────────────────────────────

        result = {
            "question": question,
            "reference": reference,
            "answer": answer,

            "answer_similarity": round(
                answer_similarity(answer, reference, embed_model),
                3,
            ),

            "answer_relevance": round(
                answer_relevance(question, answer, embed_model),
                3,
            ),

            "faithfulness": round(
                faithfulness(answer, chunks),
                3,
            ),

            "context_relevance": round(
                context_relevance(question, chunks, embed_model),
                3,
            ),
        }

        results.append(result)

        print(
            f"    Q: {question[:50]}…  "
            f"sim={result['answer_similarity']:.2f}"
        )

    # ── Aggregate metrics ───────────────────────────────────────────────────

    agg = {
        k: round(float(np.mean([r[k] for r in results])), 3)
        for k in [
            "answer_similarity",
            "answer_relevance",
            "faithfulness",
            "context_relevance",
        ]
    }

    return {
        "name": name,
        "use_rag": use_rag,
        "per_question": results,
        "aggregate": agg,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():

    Path("outputs").mkdir(exist_ok=True)

    # ── Load retriever ──────────────────────────────────────────────────────

    print("Loading retriever…")

    retriever = Retriever()

    embed_model = retriever.model

    # ── Load models ─────────────────────────────────────────────────────────

    print("\nLoading models…")

    finetuned_path = (
        MERGED_DIR
        if Path(MERGED_DIR).exists()
        else None
    )

    base_gen = load_generator(BASE_MODEL)

    finetuned_gen = (
        load_generator(finetuned_path)
        if finetuned_path
        else None
    )

    if not finetuned_gen:
        print(
            "  WARNING: No fine-tuned model found "
            "at models/merged"
        )

    # ── Run evaluations ────────────────────────────────────────────────────

    configs = []

    if finetuned_gen:

        configs.append(
            evaluate_config(
                "Fine-tuned + RAG",
                finetuned_gen,
                use_rag=True,
                qa_pairs=GOLDEN_QA,
                retriever=retriever,
                embed_model=embed_model,
            )
        )

    configs.append(
        evaluate_config(
            "Base + RAG",
            base_gen,
            use_rag=True,
            qa_pairs=GOLDEN_QA,
            retriever=retriever,
            embed_model=embed_model,
        )
    )

    if finetuned_gen:

        configs.append(
            evaluate_config(
                "Fine-tuned, no RAG",
                finetuned_gen,
                use_rag=False,
                qa_pairs=GOLDEN_QA,
                retriever=retriever,
                embed_model=embed_model,
            )
        )

    configs.append(
        evaluate_config(
            "Base, no RAG",
            base_gen,
            use_rag=False,
            qa_pairs=GOLDEN_QA,
            retriever=retriever,
            embed_model=embed_model,
        )
    )

    # ── Save results ───────────────────────────────────────────────────────

    out_path = Path("outputs/eval_results.json")

    out_path.write_text(
        json.dumps(configs, indent=2)
    )

    # ── Print summary ──────────────────────────────────────────────────────

    print(f"\n{'=' * 75}")
    print("EVALUATION SUMMARY")
    print(f"{'=' * 75}")

    print(
        f"{'Configuration':<25} "
        f"{'Ans Sim':>8} "
        f"{'Ans Rel':>8} "
        f"{'Faithful':>9} "
        f"{'Ctx Rel':>8}"
    )

    print(
        f"{'─' * 25} "
        f"{'─' * 8} "
        f"{'─' * 8} "
        f"{'─' * 9} "
        f"{'─' * 8}"
    )

    for c in configs:

        a = c["aggregate"]

        print(
            f"{c['name']:<25} "
            f"{a['answer_similarity']:>8.3f} "
            f"{a['answer_relevance']:>8.3f} "
            f"{a['faithfulness']:>9.3f} "
            f"{a['context_relevance']:>8.3f}"
        )

    print(f"{'=' * 75}")

    print("\nMetric guide:")

    print(
        "  Answer Similarity  : "
        "how close answer is to reference"
    )

    print(
        "  Answer Relevance   : "
        "how on-topic the answer is"
    )

    print(
        "  Faithfulness       : "
        "how grounded in retrieved context"
    )

    print(
        "  Context Relevance  : "
        "how relevant retrieved chunks are"
    )

    print(f"\n✓ Full results saved → {out_path}")


if __name__ == "__main__":
    main()