"""DAG to ingest PubMed XML into parent-child storage for local development."""

from __future__ import annotations

from datetime import datetime

from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator

with DAG(
    dag_id="pubmed_ingest_xml",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    tags=["pubmed", "rag", "ingest", "parent-child"],
) as dag:
    ensure_bootstrap = BashOperator(
        task_id="ensure_storage_bootstrap",
        bash_command=(
            "python /opt/airflow/scripts/bootstrap_parent_child_chunks.py"
        ),
    )

    ingest_sample = BashOperator(
        task_id="ingest_sample_pubmed_xml",
        bash_command=(
            "python /opt/airflow/scripts/ingest_pubmed_xml.py "
            "--use-sample"
        ),
    )

    ensure_bootstrap >> ingest_sample
