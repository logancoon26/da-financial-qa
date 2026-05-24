import streamlit as st
import time

from rag import Retriever
from rag_core import load_generator, build_rag_prompt, build_no_rag_prompt, generate

# import metrics from eval_compare.py
from eval_compare import (
    answer_similarity,
    answer_relevance,
    faithfulness,
    context_relevance,
)

from sentence_transformers import SentenceTransformer


st.set_page_config(page_title="RAG Dashboard", layout="wide")


# ── Cached resources ─────────────────────────────────────────────────────────

@st.cache_resource
def load_resources():
    retriever = Retriever()

    generator = load_generator("models/merged")  # or BASE_MODEL if needed

    embed_model = SentenceTransformer("BAAI/bge-m3")

    return {
        "retriever": retriever,
        "generator": generator,
        "embed_model": embed_model,
    }


resources = load_resources()
retriever = resources["retriever"]
generator = resources["generator"]
embed_model = resources["embed_model"]


# ── UI ───────────────────────────────────────────────────────────────────────

st.title("Finetuned LLM RAG Dashboard")

user_query = st.text_input(
    "Enter your financial question:",
    placeholder="Ask something..."
)

st.caption("💡 Example: What is the main driver of revenue for Apple?")


# ── Run inference ────────────────────────────────────────────────────────────

if st.button("Submit"):

    if not user_query.strip():
        st.warning("Please enter a question.")
        st.stop()

    # ── Retrieval timing ─────────────────────────────
    t0 = time.time()
    chunks = retriever.retrieve(user_query, k=2)
    retrieval_latency = (time.time() - t0) * 1000

    # ── Prompt + generation timing ───────────────────
    prompt = build_rag_prompt(user_query, chunks)

    t1 = time.time()
    output = generator(prompt)[0]["generated_text"]
    answer = output[len(prompt):].strip()
    generation_latency = (time.time() - t1) * 1000

    end_to_end = retrieval_latency + generation_latency

    # ── Metrics ───────────────────────────────────────
    ans_sim = answer_similarity(answer, "N/A", embed_model)  # no reference at runtime
    ans_rel = answer_relevance(user_query, answer, embed_model)
    faith = faithfulness(answer, chunks)
    ctx_rel = context_relevance(user_query, chunks, embed_model)

    # token throughput (approx)
    token_throughput = len(answer.split()) / max(generation_latency / 1000, 1e-6)

    # ── Output ────────────────────────────────────────

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Metrics")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Retrieval Latency", f"{retrieval_latency:.2f} ms")
        st.metric("Generation Latency", f"{generation_latency:.2f} ms")
        st.metric("End-to-End Latency", f"{end_to_end:.2f} ms")

    with col2:
        st.metric("Token Throughput", f"{token_throughput:.2f} tok/sec")
        st.metric("Faithfulness", f"{faith:.2f}")
        st.metric("Relevance", f"{ans_rel:.2f}")

    st.subheader("Retrieved Sources")

    for i, source in enumerate(chunks):
        with st.expander(
            f"{source['metadata']['company']} | "
            f"{source['metadata']['period']} | "
            f"{source['metadata']['section']} (score={source['score']:.3f})"
        ):
            st.write(source["text"])