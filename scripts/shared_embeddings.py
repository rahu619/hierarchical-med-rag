"""Shared embedding request helper for Ollama-backed scripts."""

from __future__ import annotations

import requests


def request_embedding(
    *,
    ollama_url: str,
    model: str,
    input_text: str,
    expected_dim: int,
    error_prefix: str,
) -> list[float]:
    resp = requests.post(
        f"{ollama_url}/api/embeddings",
        json={"model": model, "prompt": input_text},
        timeout=60,
    )
    if resp.status_code >= 400:
        raise RuntimeError(
            f"{error_prefix}. Ensure model '{model}' is available in Ollama. "
            "Use 'ollama pull <model>' in the Ollama environment and retry. "
            f"Status={resp.status_code} Body={resp.text[:300]}"
        )

    payload = resp.json()
    vector = payload.get("embedding")
    if not isinstance(vector, list) or not vector:
        raise ValueError("Ollama embedding response did not include a valid vector.")
    if len(vector) != expected_dim:
        raise ValueError(
            f"Embedding dimension mismatch: got {len(vector)}, expected {expected_dim}."
        )
    return vector