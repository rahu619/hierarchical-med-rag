"""Starter DAG for validating parent-child PubMed chunking bootstrap flows."""

from __future__ import annotations

from datetime import datetime

from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator

with DAG(
    dag_id="pubmed_parent_child_bootstrap",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    tags=["pubmed", "rag", "parent-child", "bootstrap"],
) as dag:
    check_env = BashOperator(
        task_id="check_python_environment",
        bash_command=(
            "python --version && "
            "echo 'QDRANT_URL='${QDRANT_URL:-http://qdrant:6333} && "
            "echo 'POSTGRES_HOST='${POSTGRES_HOST:-postgres}"
        ),
    )

    run_bootstrap = BashOperator(
        task_id="run_parent_child_scaffold",
        bash_command=(
            "python /opt/airflow/scripts/bootstrap_parent_child_chunks.py "
            "--dry-run"
        ),
    )

    check_env >> run_bootstrap
