"""Retrieve grounded parent context for a user query.

Pipeline:
1. Embed the query with Ollama.
2. Search child chunk vectors in Qdrant.
3. Reconstruct full parent paragraph context from Postgres.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any

import requests
from qdrant_client import QdrantClient
from sqlalchemy import create_engine, text


@dataclass
class Settings:
    postgres_dsn: str
    qdrant_url: str
    qdrant_collection: str
    ollama_url: str
    embedding_model: str
    embedding_dim: int


def load_settings() -> Settings:
    return Settings(
        postgres_dsn=os.getenv(
            "POSTGRES_DSN",
            "postgresql+psycopg2://admin:password@postgres:5432/pubmed_rag",
        ),
        qdrant_url=os.getenv("QDRANT_URL", "http://qdrant:6333"),
        qdrant_collection=os.getenv("QDRANT_COLLECTION", "pubmed_child_chunks"),
        ollama_url=os.getenv("OLLAMA_URL", "http://ollama:11434"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "nomic-embed-text"),
        embedding_dim=int(os.getenv("EMBEDDING_DIM", "768")),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrieve top-k child hits and reconstructed parent context."
    )
    parser.add_argument(
        "--query",
        required=True,
        help="User query to retrieve relevant PubMed context.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top child vector hits to retrieve.",
    )
    return parser.parse_args()


def embed_query(settings: Settings, query_text: str) -> list[float]:
    resp = requests.post(
        f"{settings.ollama_url}/api/embeddings",
        json={"model": settings.embedding_model, "prompt": query_text},
        timeout=60,
    )
    if resp.status_code >= 400:
        raise RuntimeError(
            "Query embedding failed. Ensure Ollama is running and model "
            f"'{settings.embedding_model}' is pulled. "
            f"Status={resp.status_code} Body={resp.text[:300]}"
        )
    payload = resp.json()
    vector = payload.get("embedding")
    if not isinstance(vector, list) or not vector:
        raise ValueError("Embedding response did not include a valid query vector.")
    if len(vector) != settings.embedding_dim:
        raise ValueError(
            "Embedding dimension mismatch: "
            f"got {len(vector)} expected {settings.embedding_dim}."
        )
    return vector


def fetch_parent_rows(settings: Settings, parent_ids: list[str]) -> dict[str, dict[str, Any]]:
    if not parent_ids:
        return {}

    engine = create_engine(settings.postgres_dsn, future=True)
    parent_map: dict[str, dict[str, Any]] = {}

    sql = text(
        """
        SELECT
            parent_id,
            pmid,
            source,
            title,
            section_title,
            paragraph_text,
            created_at
        FROM parent_documents
        WHERE parent_id = :parent_id
        """
    )
    with engine.connect() as conn:
        for parent_id in parent_ids:
            row = conn.execute(sql, {"parent_id": parent_id}).mappings().first()
            if row is None:
                continue
            parent_map[parent_id] = dict(row)

    return parent_map


def retrieve(settings: Settings, query_text: str, top_k: int) -> dict[str, Any]:
    query_vector = embed_query(settings, query_text)
    qdrant = QdrantClient(url=settings.qdrant_url, timeout=60)

    hits = qdrant.search(
        collection_name=settings.qdrant_collection,
        query_vector=query_vector,
        limit=top_k,
        with_payload=True,
        with_vectors=False,
    )

    child_hits: list[dict[str, Any]] = []
    ordered_parent_ids: list[str] = []
    seen_parent_ids: set[str] = set()

    for rank, hit in enumerate(hits, start=1):
        payload = hit.payload or {}
        parent_id = str(payload.get("parent_id", ""))
        if parent_id and parent_id not in seen_parent_ids:
            seen_parent_ids.add(parent_id)
            ordered_parent_ids.append(parent_id)

        child_hits.append(
            {
                "rank": rank,
                "score": hit.score,
                "point_id": str(hit.id),
                "pmid": payload.get("pmid"),
                "parent_id": payload.get("parent_id"),
                "child_index": payload.get("child_index"),
                "section_title": payload.get("section_title"),
                "child_text": payload.get("child_text"),
            }
        )

    parent_map = fetch_parent_rows(settings, ordered_parent_ids)

    parent_contexts: list[dict[str, Any]] = []
    for parent_rank, parent_id in enumerate(ordered_parent_ids, start=1):
        parent_row = parent_map.get(parent_id)
        if parent_row is None:
            continue

        supporting_children = [
            child for child in child_hits if child.get("parent_id") == parent_id
        ]
        parent_contexts.append(
            {
                "parent_rank": parent_rank,
                "parent_id": parent_row["parent_id"],
                "pmid": parent_row["pmid"],
                "source": parent_row["source"],
                "title": parent_row["title"],
                "section_title": parent_row["section_title"],
                "paragraph_text": parent_row["paragraph_text"],
                "supporting_children": supporting_children,
            }
        )

    return {
        "query": query_text,
        "top_k": top_k,
        "embedding_model": settings.embedding_model,
        "child_hits": child_hits,
        "parent_contexts": parent_contexts,
    }


def main() -> None:
    args = parse_args()
    settings = load_settings()
    result = retrieve(settings, query_text=args.query, top_k=args.top_k)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
