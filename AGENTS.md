# AGENTS

Guidance for AI coding agents working in this repository.

## Purpose
- Build and maintain a local PubMed RAG pipeline using PostgreSQL (parent docs), Qdrant (child vectors), Ollama embeddings, and Airflow orchestration.

## Start Here
- Architecture: [docs/01_pipeline_architecture.md](docs/01_pipeline_architecture.md)
- Project concepts and rationale: [docs/02_project_guide.md](docs/02_project_guide.md)
- Repo quickstart and service commands: [README.md](README.md)
- Current priorities: [TODO.md](TODO.md)

## Primary Workflow Commands
- Start services:
  - docker compose -f .devcontainer/docker-compose.yml up -d
- Install dependencies in app container:
  - docker compose -f .devcontainer/docker-compose.yml exec -T app python -m pip install -r requirements.txt
- Bootstrap schema and vector collection:
  - docker compose -f .devcontainer/docker-compose.yml exec -T app python scripts/bootstrap_parent_child_chunks.py
- Ingest sample XML:
  - docker compose -f .devcontainer/docker-compose.yml exec -T app python scripts/ingest_pubmed_xml.py --use-sample
- Retrieve context:
  - docker compose -f .devcontainer/docker-compose.yml exec -T app python scripts/retrieve_parent_context.py --query "early kidney injury biomarkers" --top-k 3
- Evaluate retrieval:
  - docker compose -f .devcontainer/docker-compose.yml exec -T app python scripts/eval_retrieval.py --top-k 5 --output-file evaluation/latest_report.json
- Run unit tests:
  - docker compose -f .devcontainer/docker-compose.yml exec -T app python -m unittest discover -s tests -v

## Important Conventions
- Deterministic IDs are required for idempotent ingestion.
  - parent_id format: PMID:p{paragraph_index}
  - child_id format: {parent_id}:c{child_index}
  - qdrant_point_id uses UUID5 over parent_id:child_index
- Keep config defaults centralized in [scripts/shared_config.py](scripts/shared_config.py). Avoid reintroducing duplicated env defaults.
- Use run-level traceability in scripts via [scripts/shared_run.py](scripts/shared_run.py).
  - Prefer passing --run-id for reproducible logs in multi-step runs.
- Retrieval/evaluation logic should preserve parent-child traceability.

## Airflow Notes
- DAGs should not install dependencies at runtime.
- Airflow image bakes required runtime libs via [.devcontainer/docker-compose.yml](.devcontainer/docker-compose.yml).
- Relevant DAG files:
  - [dags/pubmed_parent_child_bootstrap_dag.py](dags/pubmed_parent_child_bootstrap_dag.py)
  - [dags/pubmed_ingest_xml_dag.py](dags/pubmed_ingest_xml_dag.py)

## Known Pitfalls
- SQLAlchemy must stay on 1.4.x with current Airflow dependency pinning.
- Ollama model must exist before non-dry-run ingestion; default is nomic-embed-text.
- Embedding dimension must match Qdrant collection size (default 768).
- Host port 11434 may conflict if another Ollama instance is running.

## Validation After Changes
- For script changes:
  - Run unit tests.
  - Run retrieval evaluation and confirm summary metrics and traceability output.
- For schema/ingest changes:
  - Verify parent and child counts after sample ingest.
  - Confirm retrieval still reconstructs parent contexts correctly.
