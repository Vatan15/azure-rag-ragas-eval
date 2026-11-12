"""Offline test of src/run_testset.py using a FAKE pipeline function (no Azure calls)."""
import json
from pathlib import Path

from src.run_testset import check_testset_is_frozen, run_testset

ROOT = Path(__file__).resolve().parents[1]


def fake_answer(question, use_semantic=False):
    return {
        "answer": f"fake answer to: {question}",
        "retrieved_ids": ["1", "2", "3"] if use_semantic else ["3", "2", "1"],
        "retrieved_texts": ["a", "b", "c"],
        "finish_reason": "stop",
        "usage": {"prompt": 10, "completion": 5, "total": 15},
    }


def load_testset():
    return json.loads((ROOT / "data/testsets/testset_v1.json").read_text(encoding="utf-8"))


def test_rows_have_exactly_the_structure_of_the_saved_runs():
    saved = json.loads((ROOT / "results/run1_baseline_inputs.json").read_text(encoding="utf-8"))
    rows = run_testset(load_testset(), use_semantic=False, answer_fn=fake_answer)
    assert len(rows) == 10
    assert set(rows[0]) == set(saved[0])
    assert [r["id"] for r in rows] == [r["id"] for r in saved]
    assert [r["type"] for r in rows] == [r["type"] for r in saved]


def test_semantic_flag_is_forwarded_to_the_pipeline():
    off = run_testset(load_testset(), use_semantic=False, answer_fn=fake_answer)
    on = run_testset(load_testset(), use_semantic=True, answer_fn=fake_answer)
    assert off[0]["retrieved_ids"] == ["3", "2", "1"]
    assert on[0]["retrieved_ids"] == ["1", "2", "3"]


def test_frozen_testset_check_passes_for_the_real_file_and_fails_for_a_changed_one(tmp_path):
    check_testset_is_frozen(ROOT / "data/testsets/testset_v1.json")  # must not raise
    changed = tmp_path / "changed.json"
    changed.write_text("[]", encoding="utf-8")
    try:
        check_testset_is_frozen(changed)
    except RuntimeError:
        return
    raise AssertionError("a modified test set should have been refused")
