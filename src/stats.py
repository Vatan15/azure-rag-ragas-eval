"""Running totals for the pipeline's own model calls.

It is a dictionary (not separate variables) on purpose: `from src.stats import stats`
gives every module the SAME object, so a change made in one module is visible in all.
The RAGAS judge's calls are counted separately in src/evaluate.py.
"""

stats = {"calls": 0, "tokens": 0, "retries": 0}
