# PubMed RAG (Local Dev Stack)

Local Python-first development environment for a PubMed Retrieval-Augmented Generation pipeline focused on high-precision clinical decision support.

This project uses hierarchical parent-child chunking:

- Parent chunks (full paragraphs): PostgreSQL
- Child chunks (sentence vectors): Qdrant
- Embeddings: Ollama
- Orchestration: Airflow standalone

## Included Services

- app: Python workspace container
- postgres: parent and mapping metadata store
- qdrant: vector index for child chunks
- ollama: local model and embedding runtime
- airflow-standalone: pipeline orchestration

## Quick Start

1. Start services.

```bash
docker compose -f .devcontainer/docker-compose.yml up -d
```

2. Install Python dependencies in app container.

```bash
docker compose -f .devcontainer/docker-compose.yml exec -T app python -m pip install -r requirements.txt
```

3. Bootstrap storage schema and vector collection.

```bash
docker compose -f .devcontainer/docker-compose.yml exec -T app python scripts/bootstrap_parent_child_chunks.py
```

4. Ingest sample PubMed XML.

```bash
docker compose -f .devcontainer/docker-compose.yml exec -T app python scripts/ingest_pubmed_xml.py --use-sample
```

5. Run retrieval.

```bash
docker compose -f .devcontainer/docker-compose.yml exec -T app python scripts/retrieve_parent_context.py --query "early kidney injury biomarkers" --top-k 3
```

6. Run evaluation.

```bash
docker compose -f .devcontainer/docker-compose.yml exec -T app python scripts/eval_retrieval.py --top-k 3 --output-file evaluation/latest_report.json
```

## Key Paths

- scripts/: bootstrap, ingestion, retrieval, and evaluation scripts
- dags/: Airflow DAGs for bootstrap and ingest workflows
- evaluation/: benchmark inputs and generated reports
- docs/: architecture and project guides
- TODO.md: prioritized next-session work

## Notes

- Default embedding model is nomic-embed-text.
- If missing in Ollama, pull it first:

```bash
docker compose -f .devcontainer/docker-compose.yml exec -T ollama ollama pull nomic-embed-text
```
