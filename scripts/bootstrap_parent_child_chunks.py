"""Bootstrap scaffold for hierarchical parent-child chunking.

This script is intentionally minimal and production-oriented:
- Parent chunks are expected to be full PubMed paragraphs stored in Postgres.
- Child chunks are sentence-level segments embedded and indexed in Qdrant.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import requests
from qdrant_client import QdrantClient, models
from shared_config import (
    get_embedding_dim,
    get_postgres_dsn,
    get_qdrant_collection,
    get_qdrant_url,
)
from shared_run import log_event, resolve_run_id
from sqlalchemy import create_engine, text

SCHEMA_VERSION = "2026_06_25_001"


@dataclass
class Settings:
    postgres_dsn: str
    qdrant_url: str
    qdrant_collection: str
    embedding_dim: int


def load_settings() -> Settings:
    return Settings(
        postgres_dsn=get_postgres_dsn(),
        qdrant_url=get_qdrant_url(),
        qdrant_collection=get_qdrant_collection(),
        embedding_dim=get_embedding_dim(),
    )


def init_postgres(settings: Settings, dry_run: bool, run_id: str) -> None:
    log_event(run_id, "info", f"postgres_dsn={settings.postgres_dsn}")

    if dry_run:
        log_event(run_id, "dry-run", "Skipping Postgres writes. Connectivity checks only.")
    else:
        log_event(
            run_id,
            "run",
            "Initializing migration metadata and parent-child schema in Postgres.",
        )

    engine = create_engine(settings.postgres_dsn, future=True)
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
        if not dry_run:
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS schema_migrations (
                        version TEXT PRIMARY KEY,
                        applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
            )
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS parent_documents (
                        parent_id TEXT PRIMARY KEY,
                        pmid TEXT NOT NULL,
                        source TEXT NOT NULL,
                        title TEXT,
                        section_title TEXT,
                        paragraph_text TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
            )
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS child_chunks (
                        child_id TEXT PRIMARY KEY,
                        parent_id TEXT NOT NULL REFERENCES parent_documents(parent_id) ON DELETE CASCADE,
                        child_index INTEGER NOT NULL,
                        qdrant_point_id TEXT NOT NULL UNIQUE,
                        child_text TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(parent_id, child_index)
                    )
                    """
                )
            )
            conn.execute(
                text(
                    """
                    CREATE INDEX IF NOT EXISTS idx_parent_documents_pmid
                    ON parent_documents (pmid)
                    """
                )
            )
            conn.execute(
                text(
                    """
                    CREATE INDEX IF NOT EXISTS idx_child_chunks_parent_id
                    ON child_chunks (parent_id)
                    """
                )
            )
            conn.execute(
                text(
                    """
                    INSERT INTO schema_migrations (version)
                    VALUES (:version)
                    ON CONFLICT (version) DO NOTHING
                    """
                ),
                {"version": SCHEMA_VERSION},
            )
            conn.commit()

    log_event(run_id, "ok", "Postgres connectivity check succeeded.")


def init_qdrant(settings: Settings, dry_run: bool, run_id: str) -> None:
    log_event(run_id, "info", f"qdrant_url={settings.qdrant_url}")
    log_event(run_id, "info", f"qdrant_collection={settings.qdrant_collection}")
    log_event(run_id, "info", f"embedding_dim={settings.embedding_dim}")

    resp = requests.get(f"{settings.qdrant_url}/healthz", timeout=10)
    resp.raise_for_status()
    client = QdrantClient(url=settings.qdrant_url, timeout=30)

    if dry_run:
        collections = client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        log_event(
            run_id,
            "ok",
            f"Qdrant health check succeeded. collections={collection_names}",
        )
        return

    if not client.collection_exists(collection_name=settings.qdrant_collection):
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=models.VectorParams(
                size=settings.embedding_dim,
                distance=models.Distance.COSINE,
            ),
        )
        log_event(run_id, "ok", "Qdrant collection created.")
        return

    collection = client.get_collection(collection_name=settings.qdrant_collection)
    existing_dim = collection.config.params.vectors.size
    if existing_dim != settings.embedding_dim:
        raise ValueError(
            "Existing Qdrant collection dimension mismatch: "
            f"found {existing_dim}, expected {settings.embedding_dim}."
        )
    log_event(run_id, "ok", "Qdrant collection already exists with matching embedding dimension.")


def bootstrap_schema(settings: Settings, dry_run: bool, run_id: str) -> None:
    init_postgres(settings, dry_run=dry_run, run_id=run_id)
    init_qdrant(settings, dry_run=dry_run, run_id=run_id)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bootstrap parent-child storage scaffolding for PubMed RAG."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run connectivity checks without creating tables.",
    )
    parser.add_argument(
        "--run-id",
        default="",
        help="Optional run identifier to include in logs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_settings()
    run_id = resolve_run_id(args.run_id)
    log_event(run_id, "run", "Starting bootstrap workflow.")
    bootstrap_schema(settings, dry_run=args.dry_run, run_id=run_id)
    log_event(run_id, "ok", "Bootstrap workflow completed.")


if __name__ == "__main__":
    main()
