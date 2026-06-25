"""Evaluate retrieval precision and traceability.

Precision is measured from benchmark relevance labels.
Traceability is measured by whether child hits can be reconstructed to parent rows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from retrieve_parent_context import load_settings, retrieve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval precision@k and traceability metrics."
    )
    parser.add_argument(
        "--benchmark-file",
        default="evaluation/benchmark_queries.jsonl",
        help="Path to JSONL benchmark file.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-k child hits to evaluate.",
    )
    parser.add_argument(
        "--output-file",
        default="",
        help="Optional path to save JSON evaluation report.",
    )
    return parser.parse_args()


def load_benchmark_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in benchmark file at line {line_number}: {exc}"
                ) from exc

            if "query" not in row:
                raise ValueError(
                    f"Benchmark line {line_number} is missing required 'query' field."
                )
            rows.append(row)
    return rows


def compute_query_metrics(
    result: dict[str, Any],
    expected_pmids: set[str],
    expected_parent_ids: set[str],
) -> dict[str, Any]:
    child_hits = result.get("child_hits", [])
    parent_contexts = result.get("parent_contexts", [])

    traceable_parent_ids = {
        str(parent_context.get("parent_id")) for parent_context in parent_contexts
    }

    relevant_hits = 0
    traceable_hits = 0
    missing_parent_ids: list[str] = []

    for hit in child_hits:
        hit_pmid = str(hit.get("pmid", ""))
        hit_parent_id = str(hit.get("parent_id", ""))

        pmid_match = bool(expected_pmids) and hit_pmid in expected_pmids
        parent_match = bool(expected_parent_ids) and hit_parent_id in expected_parent_ids
        if pmid_match or parent_match:
            relevant_hits += 1

        if hit_parent_id in traceable_parent_ids:
            traceable_hits += 1
        elif hit_parent_id:
            missing_parent_ids.append(hit_parent_id)

    denominator = max(len(child_hits), 1)
    precision_at_k = relevant_hits / denominator
    traceability_at_k = traceable_hits / denominator

    return {
        "query": result.get("query"),
        "top_k_returned": len(child_hits),
        "relevant_hits": relevant_hits,
        "precision_at_k": precision_at_k,
        "hit_at_k": relevant_hits > 0,
        "traceable_hits": traceable_hits,
        "traceability_at_k": traceability_at_k,
        "missing_parent_ids": sorted(set(missing_parent_ids)),
    }


def evaluate(benchmark_rows: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    settings = load_settings()
    query_reports: list[dict[str, Any]] = []

    for row in benchmark_rows:
        query = str(row["query"])
        expected_pmids = {str(item) for item in row.get("expected_pmids", [])}
        expected_parent_ids = {str(item) for item in row.get("expected_parent_ids", [])}

        retrieval_result = retrieve(settings, query_text=query, top_k=top_k)
        query_report = compute_query_metrics(
            retrieval_result,
            expected_pmids=expected_pmids,
            expected_parent_ids=expected_parent_ids,
        )
        query_report["id"] = row.get("id", query)
        query_report["expected_pmids"] = sorted(expected_pmids)
        query_report["expected_parent_ids"] = sorted(expected_parent_ids)
        query_reports.append(query_report)

    count = len(query_reports)
    mean_precision = (
        sum(report["precision_at_k"] for report in query_reports) / max(count, 1)
    )
    hit_rate = sum(1 for report in query_reports if report["hit_at_k"]) / max(count, 1)
    mean_traceability = (
        sum(report["traceability_at_k"] for report in query_reports) / max(count, 1)
    )

    return {
        "summary": {
            "query_count": count,
            "top_k": top_k,
            "mean_precision_at_k": mean_precision,
            "hit_rate_at_k": hit_rate,
            "mean_traceability_at_k": mean_traceability,
        },
        "queries": query_reports,
    }


def main() -> None:
    args = parse_args()
    benchmark_path = Path(args.benchmark_file)
    if not benchmark_path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {benchmark_path}")

    rows = load_benchmark_rows(benchmark_path)
    report = evaluate(rows, top_k=args.top_k)

    report_text = json.dumps(report, indent=2)
    print(report_text)

    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report_text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
