"""Run the whole frozen test set through the pipeline and save the rows the other tools read
(results/<name>_inputs.json: question, retrieved ids and texts, answer, usage).

    python -m src.run_testset run3_example                   # preview only, no calls
    python -m src.run_testset run3_example --yes             # real run: one search query + one chat call per question
    python -m src.run_testset run3_example --yes --semantic  # same, with the semantic ranker ON
"""
import argparse
import hashlib
import json
from pathlib import Path

from .pipeline import rag_answer

TESTSET_PATH = Path("data/testsets/testset_v1.json")
TESTSET_SHA256_PREFIX = "5182ab0763243b46"  # fingerprint of the frozen test set v1


def check_testset_is_frozen(path=TESTSET_PATH):
    """Refuse to run if the test set's bytes changed: results would no longer be comparable."""
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if not digest.startswith(TESTSET_SHA256_PREFIX):
        raise RuntimeError("testset_v1.json does not match its frozen SHA-256; refusing to run.")


def run_testset(testset, use_semantic, answer_fn=rag_answer):
    """Answer every test question. `answer_fn` can be replaced by a fake in offline tests."""
    rows = []
    for sample in testset:
        out = answer_fn(sample["user_input"], use_semantic=use_semantic)
        rows.append(
            {
                "id": sample["id"],
                "type": sample["type"],
                "question": sample["user_input"],
                "retrieved_ids": out["retrieved_ids"],
                "retrieved_texts": out["retrieved_texts"],
                "answer": out["answer"],
                "finish_reason": out["finish_reason"],
                "usage": out["usage"],
            }
        )
        print(sample["id"], out["retrieved_ids"], out["usage"]["total"], "tokens")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the frozen test set through the pipeline.")
    parser.add_argument("name", help="run name, e.g. run3_prompt_fix (file: results/<name>_inputs.json)")
    parser.add_argument("--semantic", action="store_true", help="turn the semantic ranker on")
    parser.add_argument("--yes", action="store_true", help="really run (spends search queries and tokens)")
    args = parser.parse_args()

    check_testset_is_frozen()
    testset = json.loads(TESTSET_PATH.read_text(encoding="utf-8"))
    out_path = Path("results") / f"{args.name}_inputs.json"

    if not args.yes:
        n = len(testset)
        print(f"Preview: {n} questions = {n} search queries + {n} chat calls, about {n * 450:,} tokens.")
        print(f"Semantic ranker: {'ON' if args.semantic else 'off'}. Would write {out_path}. Add --yes to run.")
    else:
        rows = run_testset(testset, use_semantic=args.semantic)
        with open(out_path, "x", encoding="utf-8") as f:  # "x": never overwrite an existing run
            json.dump(rows, f, ensure_ascii=False, indent=2)
        print("saved", out_path)
