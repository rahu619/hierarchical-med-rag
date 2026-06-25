# PubMed RAG Pipeline Architecture

## 1. Purpose
This document describes the end-to-end local architecture for a high-precision Clinical Decision Support (CDS) PubMed Retrieval-Augmented Generation (RAG) environment built with hierarchical parent-child chunking.

- Parent chunks: full clinical paragraphs (stored in PostgreSQL)
- Child chunks: sentence-level units (embedded and indexed in Qdrant)

## 2. System Components

- App container: Python development/runtime environment for scripts and services
- PostgreSQL: source-of-truth store for parent documents and parent-child mappings
- Qdrant: vector index for child chunks
- Ollama: local embedding/LLM inference endpoint
- Airflow standalone: orchestration for bootstrap and ingestion tasks

## 3. Architecture Diagram

```mermaid
flowchart LR
    A[PubMed XML Input\nSample | File | NCBI Fetch] --> B[Ingestion Script\nscripts/ingest_pubmed_xml.py]

    B --> C[Parent Chunk Extractor\nParagraph-level]
    B --> D[Child Chunk Extractor\nSentence-level]

    C --> E[(PostgreSQL\nparent_documents)]
    D --> F[Embedding Call\nOllama /api/embeddings]
    F --> G[(Qdrant\npubmed_child_chunks)]

    D --> H[(PostgreSQL\nchild_chunks mapping)]
    H --> E

    I[Bootstrap Script\nscripts/bootstrap_parent_child_chunks.py] --> E
    I --> G

    J[Airflow DAG\npubmed_parent_child_bootstrap] --> I
    K[Airflow DAG\npubmed_ingest_xml] --> B
    K --> I
```

## 4. Data Model

### PostgreSQL Tables

1. `schema_migrations`
- version
- applied_at

2. `parent_documents`
- parent_id (PK)
- pmid
- source
- title
- section_title
- paragraph_text
- created_at

3. `child_chunks`
- child_id (PK)
- parent_id (FK -> parent_documents.parent_id)
- child_index
- qdrant_point_id (unique)
- child_text
- created_at

### Qdrant Collection

- Collection name: `pubmed_child_chunks` (configurable)
- Vector size: `EMBEDDING_DIM` (default 768)
- Distance metric: cosine
- Payload fields (example): pmid, parent_id, child_index, section_title, source, child_text

## 5. Pipeline Stages

1. Bootstrap stage
- Create relational schema and indexes
- Ensure migration marker exists
- Ensure Qdrant collection exists with expected embedding dimension

2. Ingestion stage
- Read XML input (sample/file/NCBI)
- Parse article abstracts into parent paragraphs
- Split each paragraph into sentence-level child chunks
- Insert/update parent and child metadata in PostgreSQL
- Generate embeddings for child chunks through Ollama
- Upsert vectors and metadata payloads into Qdrant

3. Retrieval stage (conceptual)
- Embed user query
- Search Qdrant for nearest child chunks
- Join child results to parent_documents in PostgreSQL
- Build context windows from parent paragraph text
- Send grounded context to generation layer

## 6. Runtime Execution Paths

### Direct CLI path
- Bootstrap:
  - `python scripts/bootstrap_parent_child_chunks.py`
- Ingest sample:
  - `python scripts/ingest_pubmed_xml.py --use-sample`

### Airflow path
- DAG `pubmed_parent_child_bootstrap`
  - validates environment
  - runs bootstrap script
- DAG `pubmed_ingest_xml`
  - ensures storage bootstrap
  - runs ingest script

## 7. Precision-Oriented Design Notes

- Parent-child separation improves traceability and recall balance:
  - child vectors improve semantic match granularity
  - parent text preserves full clinical context for answer grounding
- Deterministic IDs (`parent_id`, `child_id`, `qdrant_point_id`) support idempotent upserts and repeatable ingestion.
- Explicit schema versioning supports controlled evolution.

## 8. Operational Checks

- Service health:
  - PostgreSQL on 5432
  - Qdrant on 6333/6334
  - Ollama on 11434
  - Airflow UI on 8080
- Data checks:
  - row counts in `parent_documents` and `child_chunks`
  - point count in `pubmed_child_chunks`
- Orchestration checks:
  - Airflow DAG listing and task test execution

## 9. Known Constraints

- Airflow and SQLAlchemy compatibility is pinned for stability.
- Airflow container may require runtime package installation for custom scripts unless baked into a custom image.
- Ollama model must be available locally (for example `nomic-embed-text`) before non-dry-run ingestion.
