# Stage A #3 — RAGAS baseline (run1)

## 1. Versions
Python 3.11.9 (venv `Rag_Project\.venv`) · ragas 0.4.3 · openai 3.3.0 · langchain-openai 1.7.3 · langchain-core 1.7.4 · langchain-community 0.3.31 · langchain-huggingface 1.2.3 · instructor 1.17.0 · sentence-transformers 6.1.0 · azure-search-documents 12.0.0 · pandas 3.1.0.
Full freeze: `results/run1_requirements.txt`.

## 2. Run config
- Test set: `testset_v1.json` (frozen, read-only), SHA-256 `5182ab0763243b46b5b5541574854831a591ce8064d6b6fd2798b53cebdcfe41`; 10 questions (3 meaning, 2 exact_term, 3 multi_part, 2 out_of_scope).
- Retrieval: Azure AI Search index `rag-chunks`, hybrid (BM25 + vector), top-k 3, semantic ranker OFF; local all-MiniLM-L6-v2 query embeddings.
- Generator: deployment `chat-mini` (gpt-4.1-mini, version pinned), api_version 2024-12-01-preview, temperature 0, max_tokens 300, answers only from chunks with `[chunk N]` citations, else the exact refusal sentence. Retry with backoff (3 retries); 0 retries needed in this run.
- Judge: same `chat-mini` deployment (responses report gpt-4.1-mini-2025-04-14), temperature 0. Kept identical across runs so score changes come from the pipeline, not the judge.
- Response Relevancy embeddings: local all-MiniLM-L6-v2 (0 Azure embedding calls).
- RAGAS RunConfig: max_workers 2, timeout 120 s, max_retries 3, max_wait 30 s. Classic route (`ragas.metrics` classic metrics + `evaluate()`), which is deprecated and must be migrated before ragas 1.0.
- Files: `results/run1_baseline_inputs.json`, `run1_baseline_scores.csv`, `run1_baseline_config.json`, `rag_pipeline.py`.

## 3. Per-question scores
| id | type | faithfulness | response relevancy | context precision | context recall |
|---|---|---|---|---|---|
| q01 | meaning | 1.00 | 0.96 | 1.00 | 1.00 |
| q02 | meaning | 1.00 | 0.00 | 0.33 | 1.00 |
| q03 | meaning | 1.00 | 0.96 | 1.00 | 1.00 |
| q04 | exact_term | 0.00 | 0.00 | 1.00 | 1.00 |
| q05 | exact_term | 1.00 | 0.73 | 1.00 | 1.00 |
| q06 | multi_part | 1.00 | 0.93 | 0.00 | 1.00 |
| q07 | multi_part | 1.00 | 0.74 | 0.33 | 1.00 |
| q10 | multi_part | 1.00 | 0.95 | 0.50 | 0.50 |
| q08 | out_of_scope | 1.00 | 0.00 | 0.00 | 1.00 |
| q09 | out_of_scope | 1.00 | 0.00 | 0.00 | 1.00 |

## 4. In-scope averages (8 questions: q01–q07, q10)
Faithfulness 0.875 · Response Relevancy 0.658 · Context Precision 0.646 · Context Recall 0.938.

## 5. Out-of-scope behaviour (q08, q09)
Both answered with the exact refusal sentence ("I don't know based on the provided documents."), so the guardrail worked, including q09, which is near-topic (medicine for piles). Their metric values are not meaningful: Response Relevancy is 0 because RAGAS scores noncommittal answers as 0; precision is 0 because no chunk is useful; recall is 1.0 and faithfulness is 1.0 because a refusal has nothing to contradict. They are excluded from the averages.

## 6. NaNs
None in any metric (0 of 40 cells). One warning appeared: "LLM returned 1 generations instead of requested 3" (Response Relevancy asks the judge for 3 candidate questions per call). It did not fail any cell, but the affected sample's relevancy is based on fewer samples and is noisier. Not traced to a specific question.

## 7. Two lowest in-scope questions (mean of the 4 metrics)
- **q04 "what is GERD?" (mean 0.50): generation problem.** Retrieval was perfect (chunk 0 at rank 1; precision 1.0, recall 1.0), but the model answered with the refusal sentence. Chunk 0 only names "gastroesophageal reflux disease (GERD)" and does not define it, so the strict "answer only from chunks" rule made the model refuse. Faithfulness 0.0 is a plausible judge reading that the refusal claim ("documents don't contain this") is false for this question (hypothesis, judge statements not inspected). Relevancy 0 is the usual refusal artefact.
- **q02 "how to decide whether to take this medicine?" (mean 0.58): ranking problem plus a suspicious score.** The answer is correct and cited, but the needed chunk 1 was retrieved at rank 3 behind two irrelevant chunks (3, 5), so precision is 0.33. Relevancy 0.00 looks wrong for a good answer; likely the noncommittal check treated "decide together with your doctor" as vague (hypothesis).
- **Judge-noise finding (q06):** chunks 4 and 5 contain exactly the reference facts and the answer is correct and cited, yet Context Precision is 0.00. This is a judge error, so precision (and relevancy) from this judge on 10 samples is noisy. q10's recall of 0.5 is genuine: chunk 7 was not retrieved.

## 8. Calls, tokens, cost
| | calls | input tokens | output tokens | total | est. cost |
|---|---|---|---|---|---|
| Pipeline | 10 | 3,879 | 590 | 4,469 | $0.0025 |
| Judge | 70 | 62,074 | 8,513 | 70,587 | $0.0385 |
| **Total** | **80** | 65,953 | 9,103 | **75,056** | **$0.0409** |

25% of the 300,000-token cap. Cost assumes list prices of $0.40 per 1M input and $1.60 per 1M output tokens (to be checked against the Azure pricing page for the region); Azure AI Search's flat service fee is not included. 0 Azure embedding calls. Judge calls counted are completed calls only.

## 9. Errors and fixes
1. `pip install ragas` downgraded `openai` 3.41.0 → 3.3.0 (ragas 0.4.3 depends on instructor 1.17.0, which needs an older openai). Verified `AzureOpenAI`, `max_retries`, `timeout` and the error classes still exist; recorded.
2. `import ragas` failed with `ModuleNotFoundError: langchain_community.chat_models.vertexai`. ragas 0.4.3 imports it unconditionally and langchain-community 0.4.x removed it; ragas does not pin it. Fix: `pip install "langchain-community<0.4"` (0.3.31).
3. Harmless pip warning about a leftover `~iter` temp folder (file in use during upgrade); folder removed.
4. DeprecationWarnings: `ragas.metrics` classic imports, `LangchainLLMWrapper`, `LangchainEmbeddingsWrapper` (removal in v1.0). Accepted for the baseline; migration to `ragas.metrics.collections` and `llm_factory` needed later.
5. Earlier phase: transient `ServiceResponseError: Connection aborted` on Azure Search led to the retry wrapper; no retries were needed in this run.
