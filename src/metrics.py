"""Exact retrieval metrics computed from reference_chunk_ids: no LLM, no randomness.

    hit rate@k    1 if any correct chunk is in the top k, else 0
    MRR           1 / rank of the FIRST correct chunk (0 if none), averaged over questions
    full coverage 1 if ALL the reference chunks are in the top k, else 0

Only in-scope questions are scored: an out-of-scope question has no correct chunk, and
"all of nothing was found" would count as full coverage (vacuously true).

Command line (recomputes the published table from the saved runs, no Azure needed):
    python -m src.metrics
"""
import json
from pathlib import Path


def retrieval_metrics(retrieved_ids, needed_ids):
    """Return (hit, reciprocal_rank, full_coverage) for ONE question."""
    ranks = [retrieved_ids.index(c) + 1 for c in needed_ids if c in retrieved_ids]
    hit = 1.0 if ranks else 0.0
    rr = 1.0 / min(ranks) if ranks else 0.0
    full = 1.0 if len(ranks) == len(needed_ids) else 0.0
    return hit, rr, full


def summarize(rows, testset):
    """Score a run's saved rows against the test set, in-scope questions only.

    Returns (per_question_table, hit_rate, mrr, full_coverage).
    """
    needed = {s["id"]: s["reference_chunk_ids"] for s in testset}
    in_scope_ids = [s["id"] for s in testset if s["type"] != "out_of_scope"]

    table = {}
    for r in rows:
        if r["id"] in in_scope_ids:
            table[r["id"]] = retrieval_metrics(r["retrieved_ids"], needed[r["id"]])
    n = len(table)
    hit = sum(v[0] for v in table.values()) / n
    mrr = sum(v[1] for v in table.values()) / n
    full = sum(v[2] for v in table.values()) / n
    return table, hit, mrr, full


if __name__ == "__main__":
    testset = json.loads(Path("data/testsets/testset_v1.json").read_text(encoding="utf-8"))
    for path in sorted(Path("results").glob("run*_inputs.json")):
        rows = json.loads(path.read_text(encoding="utf-8"))
        _, hit, mrr, full = summarize(rows, testset)
        name = path.stem.replace("_inputs", "")
        print(f"{name:<16} hit@3={hit:.3f}  MRR={mrr:.3f}  full={full:.3f}")
