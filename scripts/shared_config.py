"""Shared environment-backed configuration defaults for PubMed RAG scripts."""

from __future__ import annotations

import os

DEFAULT_POSTGRES_DSN = "postgresql+psycopg2://admin:password@postgres:5432/pubmed_rag"
DEFAULT_QDRANT_URL = "http://qdrant:6333"
DEFAULT_QDRANT_COLLECTION = "pubmed_child_chunks"
DEFAULT_OLLAMA_URL = "http://ollama:11434"
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
DEFAULT_EMBEDDING_DIM = 768


def get_postgres_dsn() -> str:
    return os.getenv("POSTGRES_DSN", DEFAULT_POSTGRES_DSN)


def get_qdrant_url() -> str:
    return os.getenv("QDRANT_URL", DEFAULT_QDRANT_URL)


def get_qdrant_collection() -> str:
    return os.getenv("QDRANT_COLLECTION", DEFAULT_QDRANT_COLLECTION)


def get_ollama_url() -> str:
    return os.getenv("OLLAMA_URL", DEFAULT_OLLAMA_URL)


def get_embedding_model() -> str:
    return os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)


def get_embedding_dim() -> int:
    return int(os.getenv("EMBEDDING_DIM", str(DEFAULT_EMBEDDING_DIM)))
