"""The RAG pipeline: retrieve chunks, then answer only from them.

Command line:  python -m src.pipeline "your question" [--semantic]
"""
import argparse

from .generate import build_messages, chat
from .retry import with_retry
from .search import retrieve
from .stats import stats


def rag_answer(question, use_semantic=False):
    # Both external calls are wrapped in the retry logic.
    chunks = with_retry(retrieve, question, use_semantic)
    messages = build_messages(question, chunks)
    response = with_retry(chat, messages)

    usage = response.usage
    stats["calls"] += 1
    stats["tokens"] += usage.total_tokens

    return {
        "answer": response.choices[0].message.content,
        "retrieved_ids": [c["id"] for c in chunks],
        "retrieved_texts": [c["content"] for c in chunks],
        "finish_reason": response.choices[0].finish_reason,
        "usage": {
            "prompt": usage.prompt_tokens,
            "completion": usage.completion_tokens,
            "total": usage.total_tokens,
        },
    }


if __name__ == "__main__":
    # Runs only when started as a program (python -m src.pipeline), not when imported.
    parser = argparse.ArgumentParser(description="Ask the RAG pipeline one question.")
    parser.add_argument("question")
    parser.add_argument("--semantic", action="store_true", help="turn the semantic ranker on")
    args = parser.parse_args()

    out = rag_answer(args.question, use_semantic=args.semantic)
    print("retrieved ids:", out["retrieved_ids"])
    print("answer:", out["answer"])
    print("tokens:", out["usage"]["total"])
