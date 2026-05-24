import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


MAX_NEW_TOKENS = 256


def load_generator(model_path: str):
    print(f"Loading model: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map={"": 0} if torch.cuda.is_available() else None,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )

    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=0.1,
        do_sample=False,
        repetition_penalty=1.1,
    )

def build_rag_prompt(question: str, chunks: list[dict]) -> str:
    context = "\n\n".join(
        f"[{i+1}] {c['metadata']['company']} | {c['metadata']['period']} | "
        f"{c['metadata']['section'].upper()}\n{c['text'][:500]}"
        for i, c in enumerate(chunks)
    )

    return (
        "You are a financial analyst assistant with expertise in SEC filings.\n"
        "Answer ONLY using the context.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {question}\n\nANSWER:"
    )


def build_no_rag_prompt(question: str) -> str:
    return (
        "You are a financial analyst assistant.\n\n"
        f"QUESTION: {question}\n\nANSWER:"
    )


def generate(generator, prompt: str) -> str:
    output = generator(prompt)[0]["generated_text"]
    return output[len(prompt):].strip()