# TODO

## Next Session Starting Point

1. Review current retrieval baseline metrics.
- Run `python scripts/eval_retrieval.py --top-k 5 --output-file evaluation/latest_report.json` in the app container.
- Confirm mean precision and mean traceability are stable.

2. Expand benchmark coverage.
- Add 20 to 50 clinically diverse queries to `evaluation/benchmark_queries.jsonl`.
- Include expected PMIDs and, where possible, expected parent_ids.

## Cleanup

1. Reduce runtime pip installs inside Airflow DAG tasks.
- Build a custom Airflow image with project dependencies preinstalled.
- Remove pip install commands from DAG bash operators.

2. Consolidate duplicated configuration.
- Keep env defaults aligned across bootstrap, ingest, and retrieval scripts.
- Add a shared config module if drift starts appearing.

3. Improve logging consistency.
- Add consistent run identifiers to bootstrap, ingest, retrieval, and eval outputs.

## Tests

1. Unit tests.
- XML parsing and sentence splitting behavior.
- Deterministic parent_id and child_id generation.
- Retrieval reconstruction logic for parent mapping.

2. Integration tests.
- Bootstrap schema + Qdrant collection creation.
- Sample ingest writes to Postgres and Qdrant.
- Retrieval returns traceable parent contexts.

3. Regression guardrails.
- Assert parent/child row and point counts after sample ingest.

## CI Checks

1. Add lint and format checks.
- Ruff check and Ruff format verification.

2. Add test workflow.
- Run unit tests on pull requests.
- Run selected integration smoke tests.

3. Add evaluation smoke gate.
- Run `eval_retrieval.py` on a small benchmark subset and publish metrics artifact.

## Generation Layer

1. Add grounded answer assembly.
- Create a generation script that accepts parent_contexts output from retrieval.
- Produce answers with explicit PMID and parent_id citations.

2. Add safety behavior.
- Return insufficient-evidence when retrieval coverage is weak.
- Prevent uncited claims in final responses.

3. Add generation quality checks.
- Validate citation correctness against retrieved contexts.
- Track answerability vs hallucination rate on benchmark prompts.
