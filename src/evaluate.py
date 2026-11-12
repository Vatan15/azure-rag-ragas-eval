"""RAGAS evaluation: score a run's saved inputs with an Azure OpenAI judge.

Command line (the real run makes ~7 judge calls per question, ~70K tokens for 10):
    python -m src.evaluate results/run2_semantic_inputs.json          # preview only
    python -m src.evaluate results/run2_semantic_inputs.json --yes    # actually run

Note: ragas 0.4.3's "classic" metrics and wrappers used here are deprecated and will be
removed in ragas 1.0 (see README, limitations).
"""
import argparse
import json
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import AzureChatOpenAI
from ragas import EvaluationDataset, RunConfig, evaluate
from ragas.cost import get_token_usage_for_openai
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)

from .config import API_VERSION, EMBEDDING_MODEL_NAME, require_env

# Held identical across runs so score differences come from the pipeline, not the judge.
# Low concurrency keeps requests under the deployment's requests-per-minute limit.
RUN_CONFIG = {"max_workers": 2, "timeout": 120, "max_retries": 3, "max_wait": 30}

SCORE_COLUMNS = [
    "faithfulness",
    "answer_relevancy",
    "llm_context_precision_with_reference",
    "context_recall",
]

# Counts judge calls that completed (filled in by counting_parser below).
judge_stats = {"calls": 0}


def counting_parser(llm_result):
    """RAGAS calls this after every judge call to read its token usage; we also count it."""
    judge_stats["calls"] += 1
    return get_token_usage_for_openai(llm_result)


def make_judge():
    """The judge model (Azure OpenAI, temperature 0) and local embeddings for Response Relevancy."""
    judge_llm = LangchainLLMWrapper(
        AzureChatOpenAI(
            azure_deployment=require_env("AZURE_OPENAI_DEPLOYMENT"),
            azure_endpoint=require_env("AZURE_OPENAI_ENDPOINT"),
            api_key=require_env("AZURE_OPENAI_API_KEY"),
            api_version=API_VERSION,
            temperature=0,
            max_retries=0,  # retries are handled by RAGAS's RunConfig, not stacked here
        )
    )
    embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=f"sentence-transformers/{EMBEDDING_MODEL_NAME}")
    )
    return judge_llm, embeddings


def build_dataset(rows, testset):
    """RAGAS needs four fields per sample. The pipeline supplies retrieved_contexts and
    response; the frozen test set supplies user_input's reference answer."""
    ref_by_id = {s["id"]: s["reference"] for s in testset}
    return EvaluationDataset.from_list(
        [
            {
                "user_input": r["question"],
                "retrieved_contexts": r["retrieved_texts"],
                "response": r["answer"],
                "reference": ref_by_id[r["id"]],
            }
            for r in rows
        ]
    )


def run_ragas(rows, testset):
    judge_llm, embeddings = make_judge()
    judge_stats["calls"] = 0
    metrics = [
        Faithfulness(),
        ResponseRelevancy(),
        LLMContextPrecisionWithReference(),
        LLMContextRecall(),
    ]
    return evaluate(
        dataset=build_dataset(rows, testset),
        metrics=metrics,
        llm=judge_llm,
        embeddings=embeddings,
        run_config=RunConfig(**RUN_CONFIG),
        token_usage_parser=counting_parser,
        raise_exceptions=False,  # a failed judge call becomes NaN, not a crash
        show_progress=True,
    )


def save_results(result, rows, name, out_dir=Path("results")):
    """Write <name>_scores.csv and <name>_judge_usage.json ("x" mode: never overwrite)."""
    df = result.to_pandas()
    df.insert(0, "id", [r["id"] for r in rows])
    df.insert(1, "type", [r["type"] for r in rows])
    out = df[["id", "type"] + SCORE_COLUMNS].copy()
    out["retrieved_ids"] = ["|".join(r["retrieved_ids"]) for r in rows]
    with open(out_dir / f"{name}_scores.csv", "x", encoding="utf-8", newline="") as f:
        out.to_csv(f, index=False)

    usage = result.total_tokens()
    with open(out_dir / f"{name}_judge_usage.json", "x", encoding="utf-8") as f:
        json.dump(
            {
                "judge_calls": judge_stats["calls"],
                "judge_input_tokens": usage.input_tokens,
                "judge_output_tokens": usage.output_tokens,
                "judge_model_reported": usage.model,
            },
            f,
            indent=2,
        )
    return out, int(out[SCORE_COLUMNS].isna().sum().sum())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score a run's saved inputs with RAGAS.")
    parser.add_argument("inputs", help="e.g. results/run2_semantic_inputs.json")
    parser.add_argument("--yes", action="store_true", help="really run (spends judge tokens)")
    args = parser.parse_args()

    inputs_path = Path(args.inputs)
    rows = json.loads(inputs_path.read_text(encoding="utf-8"))
    testset = json.loads(Path("data/testsets/testset_v1.json").read_text(encoding="utf-8"))
    name = inputs_path.stem.replace("_inputs", "")

    if not args.yes:
        n = len(rows)
        print(f"Preview: {n} samples x 4 metrics = about {n * 7} judge calls, about {n * 7000:,} tokens.")
        print("Nothing was run. Add --yes to run it.")
    else:
        result = run_ragas(rows, testset)
        _, nan_cells = save_results(result, rows, name)
        print(result)
        print("judge calls:", judge_stats["calls"], "| NaN cells:", nan_cells, f"of {len(rows) * 4}")
