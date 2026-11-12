"""Smoke test: send ONE real question through the pipeline (1 search query + 1 chat call,
about 500 tokens) and compare it with the answer saved in results/run1_baseline_inputs.json.

    python -m src.smoke_test          # preview only, no calls
    python -m src.smoke_test --yes    # make the single real call
"""
import argparse
import json
from pathlib import Path

from .pipeline import rag_answer

QUESTION_ID = "q06"  # a two-part question whose answer cites two chunks

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="One real pipeline call vs the saved run1 answer.")
    parser.add_argument("--yes", action="store_true", help="really make the call")
    args = parser.parse_args()

    rows = json.loads(Path("results/run1_baseline_inputs.json").read_text(encoding="utf-8"))
    saved = next(r for r in rows if r["id"] == QUESTION_ID)
    print(f"Question {QUESTION_ID}: {saved['question']}")

    if not args.yes:
        print("Preview only: 1 search query + 1 chat call, about 500 tokens. Add --yes to run it.")
    else:
        out = rag_answer(saved["question"], use_semantic=False)  # run1 used the ranker OFF
        print("retrieved ids match run1:", out["retrieved_ids"] == saved["retrieved_ids"],
              "|", out["retrieved_ids"], "vs", saved["retrieved_ids"])
        print("answer identical to run1:", out["answer"] == saved["answer"])
        print("tokens:", out["usage"]["total"], "(run1 used", saved["usage"]["total"], ")")
        print("--- new answer ---")
        print(out["answer"])
