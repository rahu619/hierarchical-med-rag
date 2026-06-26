"""Ingest PubMed XML into parent-child storage for RAG retrieval.

Flow:
1. Parse PubMed XML into parent paragraphs (parent chunks).
2. Split parent paragraphs into sentence-level child chunks.
3. Store parents and child mapping metadata in Postgres.
4. Embed child chunks via Ollama and upsert vectors to Qdrant.
"""

from __future__ import annotations

import argparse
import re
import uuid
from dataclasses import dataclass
from typing import Iterable
from xml.etree import ElementTree as ET

import requests
from qdrant_client import QdrantClient, models
from shared_config import (
    get_embedding_dim,
    get_embedding_model,
    get_ollama_url,
    get_postgres_dsn,
    get_qdrant_collection,
    get_qdrant_url,
)
from shared_run import log_event, resolve_run_id
from sqlalchemy import create_engine, text

SAMPLE_XML = """
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>00000001</PMID>
      <Article>
        <ArticleTitle>Sample clinical abstract for local ingestion testing</ArticleTitle>
        <Abstract>
          <AbstractText Label="BACKGROUND">Acute kidney injury is common in critical care and can increase mortality risk.</AbstractText>
          <AbstractText Label="METHODS">We evaluated early serum biomarkers in a small observational cohort.</AbstractText>
          <AbstractText Label="RESULTS">Biomarker-guided assessment improved early risk stratification and intervention timing.</AbstractText>
        </Abstract>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>
""".strip()

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Settings:
    postgres_dsn: str
    qdrant_url: str
    qdrant_collection: str
    ollama_url: str
    embedding_model: str
    embedding_dim: int


@dataclass
class ParentParagraph:
    parent_id: str
    pmid: str
    source: str
    title: str
    section_title: str
    paragraph_text: str


def load_settings() -> Settings:
    return Settings(
        postgres_dsn=get_postgres_dsn(),
        qdrant_url=get_qdrant_url(),
        qdrant_collection=get_qdrant_collection(),
        ollama_url=get_ollama_url(),
        embedding_model=get_embedding_model(),
        embedding_dim=get_embedding_dim(),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest PubMed XML into Postgres parent docs and Qdrant child vectors."
    )
    parser.add_argument(
        "--pmids",
        default="31452104",
        help="Comma-separated PubMed IDs used when fetching XML from NCBI.",
    )
    parser.add_argument(
        "--xml-file",
        default="",
        help="Local XML file path. If set, this input is used instead of remote fetch.",
    )
    parser.add_argument(
        "--use-sample",
        action="store_true",
        help="Use built-in sample XML to avoid external network calls.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and validate only. Do not persist to Postgres or Qdrant.",
    )
    parser.add_argument(
        "--run-id",
        default="",
        help="Optional run identifier to include in logs.",
    )
    return parser.parse_args()


def make_parent_id(pmid: str, paragraph_index: int) -> str:
    return f"{pmid}:p{paragraph_index}"


def make_child_id(parent_id: str, child_index: int) -> str:
    return f"{parent_id}:c{child_index}"


def make_qdrant_point_id(parent_id: str, child_index: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{parent_id}:{child_index}"))


def read_xml(args: argparse.Namespace) -> str:
    if args.use_sample:
        return SAMPLE_XML

    if args.xml_file:
        with open(args.xml_file, "r", encoding="utf-8") as handle:
            return handle.read()

    resp = requests.get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
        params={
            "db": "pubmed",
            "id": args.pmids,
            "retmode": "xml",
            "rettype": "abstract",
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.text


def parse_parent_paragraphs(xml_payload: str) -> list[ParentParagraph]:
    root = ET.fromstring(xml_payload)
    rows: list[ParentParagraph] = []

    for article in root.findall(".//PubmedArticle"):
        pmid_text = article.findtext(".//PMID", default="unknown")
        pmid = pmid_text.strip()
        title_text = article.findtext(".//ArticleTitle", default="")
        title = " ".join(title_text.split())

        abstract_nodes = article.findall(".//AbstractText")
        for idx, abstract_node in enumerate(abstract_nodes):
            paragraph_text = " ".join("".join(abstract_node.itertext()).split())
            if not paragraph_text:
                continue

            section_title = abstract_node.attrib.get("Label", "")
            parent_id = make_parent_id(pmid, idx)
            rows.append(
                ParentParagraph(
                    parent_id=parent_id,
                    pmid=pmid,
                    source="pubmed",
                    title=title,
                    section_title=section_title,
                    paragraph_text=paragraph_text,
                )
            )

    return rows


def split_sentences(paragraph: str) -> list[str]:
    parts = [piece.strip() for piece in SENTENCE_SPLIT.split(paragraph)]
    return [piece for piece in parts if len(piece) >= 20]


def embed_text(settings: Settings, input_text: str) -> list[float]:
    resp = requests.post(
        f"{settings.ollama_url}/api/embeddings",
        json={"model": settings.embedding_model, "prompt": input_text},
        timeout=60,
    )
    if resp.status_code >= 400:
        raise RuntimeError(
            "Ollama embedding request failed. Ensure model "
            f"'{settings.embedding_model}' is available in Ollama. "
            "Use 'ollama pull <model>' in the Ollama environment and retry. "
            f"Status={resp.status_code} Body={resp.text[:300]}"
        )
    payload = resp.json()
    vector = payload.get("embedding")
    if not isinstance(vector, list) or not vector:
        raise ValueError("Ollama embedding response did not include a valid vector.")
    if len(vector) != settings.embedding_dim:
        raise ValueError(
            f"Embedding dimension mismatch: got {len(vector)}, expected {settings.embedding_dim}."
        )
    return vector


def persist_records(
    settings: Settings,
    parents: Iterable[ParentParagraph],
    dry_run: bool,
    run_id: str,
) -> None:
    parent_list = list(parents)
    child_count = sum(len(split_sentences(parent.paragraph_text)) for parent in parent_list)

    log_event(run_id, "info", f"parents={len(parent_list)}")
    log_event(run_id, "info", f"children={child_count}")

    if dry_run:
        log_event(run_id, "dry-run", "Skipping writes to Postgres, Ollama, and Qdrant.")
        return

    engine = create_engine(settings.postgres_dsn, future=True)
    qdrant = QdrantClient(url=settings.qdrant_url, timeout=60)

    with engine.connect() as conn:
        for parent in parent_list:
            conn.execute(
                text(
                    """
                    INSERT INTO parent_documents (
                        parent_id,
                        pmid,
                        source,
                        title,
                        section_title,
                        paragraph_text
                    )
                    VALUES (
                        :parent_id,
                        :pmid,
                        :source,
                        :title,
                        :section_title,
                        :paragraph_text
                    )
                    ON CONFLICT (parent_id)
                    DO UPDATE SET
                        title = EXCLUDED.title,
                        section_title = EXCLUDED.section_title,
                        paragraph_text = EXCLUDED.paragraph_text
                    """
                ),
                {
                    "parent_id": parent.parent_id,
                    "pmid": parent.pmid,
                    "source": parent.source,
                    "title": parent.title,
                    "section_title": parent.section_title,
                    "paragraph_text": parent.paragraph_text,
                },
            )

            points: list[models.PointStruct] = []
            for child_index, child_text in enumerate(split_sentences(parent.paragraph_text)):
                point_id = make_qdrant_point_id(parent.parent_id, child_index)
                vector = embed_text(settings, child_text)

                conn.execute(
                    text(
                        """
                        INSERT INTO child_chunks (
                            child_id,
                            parent_id,
                            child_index,
                            qdrant_point_id,
                            child_text
                        )
                        VALUES (
                            :child_id,
                            :parent_id,
                            :child_index,
                            :qdrant_point_id,
                            :child_text
                        )
                        ON CONFLICT (child_id)
                        DO UPDATE SET
                            qdrant_point_id = EXCLUDED.qdrant_point_id,
                            child_text = EXCLUDED.child_text
                        """
                    ),
                    {
                        "child_id": make_child_id(parent.parent_id, child_index),
                        "parent_id": parent.parent_id,
                        "child_index": child_index,
                        "qdrant_point_id": point_id,
                        "child_text": child_text,
                    },
                )

                points.append(
                    models.PointStruct(
                        id=point_id,
                        vector=vector,
                        payload={
                            "pmid": parent.pmid,
                            "parent_id": parent.parent_id,
                            "child_index": child_index,
                            "section_title": parent.section_title,
                            "source": parent.source,
                            "child_text": child_text,
                        },
                    )
                )

            if points:
                qdrant.upsert(
                    collection_name=settings.qdrant_collection,
                    points=points,
                    wait=True,
                )

        conn.commit()

    log_event(run_id, "ok", "Ingestion completed successfully.")


def main() -> None:
    args = parse_args()
    settings = load_settings()
    run_id = resolve_run_id(args.run_id)
    log_event(run_id, "run", "Starting PubMed XML ingestion workflow.")
    xml_payload = read_xml(args)
    parents = parse_parent_paragraphs(xml_payload)

    if not parents:
        raise ValueError("No PubMed abstracts found in input XML payload.")

    persist_records(settings, parents, dry_run=args.dry_run, run_id=run_id)
    log_event(run_id, "ok", "Ingestion workflow completed.")


if __name__ == "__main__":
    main()
