"""Guards for the things the published results depend on: the frozen test set, the chunking
and the prompt. Offline: no Azure, and the embedding model is never loaded."""
import hashlib
import json
from pathlib import Path

from src.generate import SYSTEM_PROMPT
from src.ingest import chunk_text, load_document

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_testset_is_unchanged():
    data = (ROOT / "data/testsets/testset_v1.json").read_bytes()
    assert hashlib.sha256(data).hexdigest().startswith("5182ab0763243b46")


def test_testset_has_ten_questions_with_two_out_of_scope():
    testset = json.loads((ROOT / "data/testsets/testset_v1.json").read_text(encoding="utf-8"))
    assert len(testset) == 10
    assert sum(1 for s in testset if s["type"] == "out_of_scope") == 2


def test_chunking_reproduces_the_saved_chunks():
    # reference_chunk_ids in the test set are positions in this exact list.
    doc = load_document(ROOT / "data/raw_documents/drug_info_00.json")
    saved = json.loads((ROOT / "data/processed/drug_info_00_chunks.json").read_text(encoding="utf-8"))
    assert chunk_text(doc["markdown"]) == saved
    assert len(saved) == 10


def test_system_prompt_contains_the_refusal_sentence():
    assert "I don't know based on the provided documents." in SYSTEM_PROMPT
    assert len(SYSTEM_PROMPT) == 340  # the length used for runs 1 and 2
