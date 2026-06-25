"""Bootstrap scaffold for hierarchical parent-child chunking.

This script is intentionally minimal and production-oriented:
- Parent chunks are expected to be full PubMed paragraphs stored in Postgres.
- Child chunks are sentence-level segments embedded and indexed in Qdrant.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import requests
from qdrant_client import QdrantClient, models
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
        postgres_dsn=os.getenv(
            "POSTGRES_DSN",
            "postgresql+psycopg2://admin:password@postgres:5432/pubmed_rag",
        ),
        qdrant_url=os.getenv("QDRANT_URL", "http://qdrant:6333"),
        qdrant_collection=os.getenv("QDRANT_COLLECTION", "pubmed_child_chunks"),
        embedding_dim=int(os.getenv("EMBEDDING_DIM", "768")),
    )


def init_postgres(settings: Settings, dry_run: bool) -> None:
    print(f"[info] postgres_dsn={settings.postgres_dsn}")

    if dry_run:
        print("[dry-run] Skipping Postgres writes. Connectivity checks only.")
    else:
        print("[run] Initializing migration metadata and parent-child schema in Postgres.")

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

    print("[ok] Postgres connectivity check succeeded.")


def init_qdrant(settings: Settings, dry_run: bool) -> None:
    print(f"[info] qdrant_url={settings.qdrant_url}")
    print(f"[info] qdrant_collection={settings.qdrant_collection}")
    print(f"[info] embedding_dim={settings.embedding_dim}")

    resp = requests.get(f"{settings.qdrant_url}/healthz", timeout=10)
    resp.raise_for_status()
    client = QdrantClient(url=settings.qdrant_url, timeout=30)

    if dry_run:
        collections = client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        print(f"[ok] Qdrant health check succeeded. collections={collection_names}")
        return

    if not client.collection_exists(collection_name=settings.qdrant_collection):
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=models.VectorParams(
                size=settings.embedding_dim,
                distance=models.Distance.COSINE,
            ),
        )
        print("[ok] Qdrant collection created.")
        return

    collection = client.get_collection(collection_name=settings.qdrant_collection)
    existing_dim = collection.config.params.vectors.size
    if existing_dim != settings.embedding_dim:
        raise ValueError(
            "Existing Qdrant collection dimension mismatch: "
            f"found {existing_dim}, expected {settings.embedding_dim}."
        )
    print("[ok] Qdrant collection already exists with matching embedding dimension.")


def bootstrap_schema(settings: Settings, dry_run: bool) -> None:
    init_postgres(settings, dry_run=dry_run)
    init_qdrant(settings, dry_run=dry_run)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bootstrap parent-child storage scaffolding for PubMed RAG."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run connectivity checks without creating tables.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_settings()
    bootstrap_schema(settings, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
