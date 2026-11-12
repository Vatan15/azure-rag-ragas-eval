"""Tests for the exact retrieval metrics (src/metrics.py). Offline: no Azure, no LLM."""
import json
from pathlib import Path

import pytest

from src.metrics import retrieval_metrics, summarize

ROOT = Path(__file__).resolve().parents[1]


def test_correct_chunk_ranked_third():
    hit, rr, full = retrieval_metrics(["3", "5", "1"], ["1"])
    assert hit == 1.0
    assert rr == pytest.approx(1 / 3)
    assert full == 1.0


def test_missing_second_chunk_breaks_full_coverage():
    hit, rr, full = retrieval_metrics(["2", "6", "1"], ["6", "7"])
    assert (hit, rr, full) == (1.0, 0.5, 0.0)


def test_mrr_uses_only_the_first_correct_chunk():
    # Needed chunks 8 and 2 are at ranks 2 and 3: MRR is 1/2, not 1/2 + 1/3.
    hit, rr, full = retrieval_metrics(["1", "8", "2"], ["8", "2"])
    assert (hit, rr, full) == (1.0, 0.5, 1.0)


def test_no_correct_chunk_retrieved():
    assert retrieval_metrics(["0", "3", "9"], ["7"]) == (0.0, 0.0, 0.0)


def test_empty_reference_list_is_vacuously_full():
    # Why out-of-scope questions must be excluded from the averages.
    assert retrieval_metrics(["0", "3", "9"], []) == (0.0, 0.0, 1.0)


def test_summarize_skips_out_of_scope_questions():
    testset = [
        {"id": "a", "type": "meaning", "reference_chunk_ids": ["1"]},
        {"id": "b", "type": "out_of_scope", "reference_chunk_ids": []},
    ]
    rows = [
        {"id": "a", "retrieved_ids": ["1", "2", "3"]},
        {"id": "b", "retrieved_ids": ["4", "5", "6"]},
    ]
    table, hit, mrr, full = summarize(rows, testset)
    assert list(table) == ["a"]
    assert (hit, mrr, full) == (1.0, 1.0, 1.0)


def test_published_numbers_are_reproduced_from_saved_runs():
    testset = json.loads((ROOT / "data/testsets/testset_v1.json").read_text(encoding="utf-8"))
    run1 = json.loads((ROOT / "results/run1_baseline_inputs.json").read_text(encoding="utf-8"))
    run2 = json.loads((ROOT / "results/run2_semantic_inputs.json").read_text(encoding="utf-8"))

    _, hit1, mrr1, full1 = summarize(run1, testset)
    _, hit2, mrr2, full2 = summarize(run2, testset)

    assert (round(hit1, 3), round(mrr1, 3), round(full1, 3)) == (1.0, 0.792, 0.875)
    assert (round(hit2, 3), round(mrr2, 3), round(full2, 3)) == (1.0, 1.0, 0.875)
