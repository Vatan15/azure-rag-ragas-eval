# Stage A #3 — run2: semantic ranker ON vs run1 baseline

## 1. Held constant (verified, not assumed)
Only `use_semantic` changed (False → True). A programmatic diff of `run1_baseline_config.json` vs `run2_semantic_config.json` lists only: `retrieval.semantic_ranker`, `retrieval.mode`, `run`, `notes`, and token usage (a consequence of different chunks). Identical: test set `testset_v1.json` (SHA-256 `5182ab0763243b46…`, re-verified), index `rag-chunks`, top_k 3, hybrid search, generator `chat-mini` (gpt-4.1-mini, temperature 0, max_tokens 300, same system prompt in the unchanged `rag_pipeline.py`), judge `chat-mini` temperature 0 (reported model gpt-4.1-mini-2025-04-14), the same 4 RAGAS metrics, `RunConfig` (2 workers, 120 s, 3 retries, 30 s max wait), local MiniLM embeddings for Response Relevancy, and all package versions (ragas 0.4.3, openai 3.3.0, etc.).
One infrastructure change, approved by the mentor (option A): the chat-mini deployment quota was raised from 13K TPM / 13 RPM to 53K TPM / 53 RPM before the successful judge run (see section 8). It does not affect scores.

## 2. Exact retrieval metrics (8 in-scope questions, no LLM, deterministic)
| metric | run1 | run2 | diff |
|---|---|---|---|
| hit rate@3 | 1.000 | 1.000 | +0.000 |
| MRR | 0.792 | 1.000 | **+0.208** |
| full coverage | 0.875 (7/8) | 0.875 (7/8) | +0.000 |

Run1 matches the mentor's expected values. Out-of-scope questions are excluded (empty reference list makes "full coverage" vacuously 1.0).

## 3. Retrieval per question (retrieved chunk ids, rank order)
| id | needed | run1 | run2 |
|---|---|---|---|
| q01 | 0 | 0,4,9 | 0,4,2 |
| q02 | 1 | 3,5,1 | **1**,3,5 |
| q03 | 3 | 3,5,4 | 3,0,4 |
| q04 | 0 | 0,4,1 | 0,4,9 |
| q05 | 9 | 9,7,0 | 9,6,4 |
| q06 | 4,5 | 4,5,3 | 4,5,3 |
| q07 | 8,2 | 1,8,2 | **8**,1,2 |
| q10 | 6,7 | 2,6,1 | **6**,9,8 |
| q08 / q09 (out of scope) | – | 4,3,0 / 0,3,9 | 3,1,4 / 3,8,9 |

## 4. RAGAS comparison per question (run1 → run2)
| id | type | faithfulness | relevancy | precision | recall |
|---|---|---|---|---|---|
| q01 | meaning | 1.00 → 1.00 | 0.96 → 0.94 | 1.00 → 1.00 | 1.00 → 1.00 |
| q02 | meaning | 1.00 → 1.00 | 0.00 → 0.00 | 0.33 → **1.00** | 1.00 → 1.00 |
| q03 | meaning | 1.00 → 1.00 | 0.96 → 0.96 | 1.00 → 1.00 | 1.00 → 1.00 |
| q04 | exact_term | 0.00 → 0.00 | 0.00 → 0.00 | 1.00 → 1.00 | 1.00 → 1.00 |
| q05 | exact_term | 1.00 → 1.00 | 0.73 → 0.68 | 1.00 → 1.00 | 1.00 → 1.00 |
| q06 | multi_part | 1.00 → 1.00 | 0.93 → 0.93 | 0.00 → 0.00 | 1.00 → 1.00 |
| q07 | multi_part | 1.00 → 1.00 | 0.74 → 0.70 | 0.33 → 0.33 | 1.00 → 1.00 |
| q10 | multi_part | 1.00 → 1.00 | 0.95 → 0.99 | 0.50 → **1.00** | 0.50 → 0.50 |
| q08 | out_of_scope | 1.00 → 0.00 | 0.00 → 0.00 | 0.00 → 0.00 | 1.00 → 1.00 |
| q09 | out_of_scope | 1.00 → 1.00 | 0.00 → 0.00 | 0.00 → 0.00 | 1.00 → 1.00 |

## 5. In-scope averages (8 questions) with noise flags (rule: |diff| < 0.05 = within noise)
| metric | run1 | run2 | diff | flag |
|---|---|---|---|---|
| hit@3 | 1.000 | 1.000 | +0.000 | unchanged |
| MRR | 0.792 | 1.000 | +0.208 | better |
| full coverage | 0.875 | 0.875 | +0.000 | unchanged |
| faithfulness | 0.875 | 0.875 | +0.000 | within noise |
| response relevancy | 0.658 | 0.652 | -0.006 | within noise |
| context precision | 0.646 | 0.792 | +0.146 | better |
| context recall | 0.938 | 0.938 | +0.000 | within noise |

Caveats: the 0.05 rule is a convention, not a statistical test; with 8 questions one question moving by 0.4 shifts an average by 0.05. The three exact metrics are deterministic, so "unchanged" means exactly unchanged.

## 6. Step 6 checks
- **q02:** needed chunk 1 moved from rank 3 to rank 1 (precision 0.33 → 1.00).
- **q07:** chunk 8 moved from rank 2 to rank 1; chunk 2 stayed at rank 3. RAGAS precision stayed 0.33 in both runs although chunk 8 plainly contains the fracture-risk facts from the reference, so the judge's verdict disagrees with the text (a fair score would be about 0.83).
- **q10:** chunk 6 moved from rank 2 to rank 1 (precision 0.50 → 1.00). **Chunk 7 was not retrieved** (run2 top 3 = 6, 9, 8); recall stays 0.50 and full coverage stays 7/8.
- **q04:** still answers "I don't know based on the provided documents." (prompt unchanged; chunk 0 is rank 1 in both runs). This is a generator-side strictness issue, not a retrieval issue, so the ranker cannot fix it.
- Out-of-scope q08 and q09 still refuse with the exact sentence.

## 7. Judge reliability observations
Context Precision verdicts looked wrong for q06 (0.00 in both runs although chunks 4 and 5 hold exactly the reference facts and retrieval is identical) and q07 (0.33 in both runs). Identical inputs gave identical wrong verdicts at temperature 0, so this error looks systematic, not random: repeating the run would not average it out. Response Relevancy is 0 for refusals by design and for q02 (a correct, cited answer, likely flagged noncommittal; hypothesis). q08 faithfulness flipped 1.0 → 0.0 with an identical refusal answer, so faithfulness on refusals is unstable (out-of-scope, excluded from averages). Response Relevancy also logged "LLM returned 1 generations instead of requested 3" (once in run1, twice in run2).

## 8. Cost, tokens, errors
| | calls | in | out | total | est. cost |
|---|---|---|---|---|---|
| Pipeline (10 semantic queries + 10 answers) | 10 | 3,772 | 675 | 4,447 | $0.0026 |
| Judge (successful run) | 70 | 62,098 | 9,559 | 71,657 | $0.0401 |
| Judge (first attempt, scores lost) | 49 | 41,797 | 6,038 | 47,835 | $0.0264 |
| **Total** | 129 | | | **123,939** (41% of the 300K cap) | **$0.0691** |

Assumed list price $0.40 / $1.60 per 1M input/output tokens (check the Azure pricing page); Azure AI Search fixed fee not included. Semantic queries used: 10 (Free-tier allowance), no billing or quota errors on search.

Errors and fixes:
1. First judge attempt: HTTP 429 `rate_limit_exceeded` on 15 of 40 jobs (retries exhausted, NaN-bound). The portal showed the deployment at 13,000 TPM / 13 RPM, not the 50K assumed (Modified-on equalled Created-on). Run1 had used about 13.5 calls/min, i.e. it sat at the limit and was probably carried by silent retries. I stopped and reported per the rules; the mentor approved raising the quota (option A), done to 53K TPM / 53 RPM.
2. A kernel restart lost the in-memory scores of that first attempt (47,835 tokens wasted). Fix: reload everything from saved files; the re-run saves the CSV and usage file immediately after `evaluate()`.
3. Setup-cell mix-ups after the restart (an old cell needing `rp`, an evaluation guard firing) caused no spend and were corrected.
4. A typo in the saved config note ("13 TPM-K") was corrected to "13K TPM / 13 RPM".

## 9. Recommendation: keep the semantic ranker ON
- Evidence: MRR +0.208 and context precision +0.146 (above the 0.05 noise rule); the first correct chunk is at rank 1 for all 8 in-scope questions; no metric got worse beyond noise; refusals and answer quality are intact.
- It fixes the ranking problems (q02, q07, q10) but not the two remaining failures, which need different single-variable changes: q10's missing chunk 7 (recall 0.5, full coverage 7/8) points to a larger top_k (e.g. 5) or chunking change; q04's refusal points to the system prompt (re-check q08/q09 refusals after any prompt change).
- Cautions: only 10 questions, one judge model with some systematic errors; added latency and the Free tier's monthly semantic-query allowance were not measured here, so check both before relying on it in a real deployment.
