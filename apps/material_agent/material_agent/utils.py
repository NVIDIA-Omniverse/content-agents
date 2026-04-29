# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared utilities for material agent workflows."""

import statistics
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from rich.console import Console
from rich.table import Table

console = Console()


# Version info
def get_version() -> str:
    try:
        return version("material-agent")
    except PackageNotFoundError:
        return "0.0.1-dev"


def calculate_metrics(
    scores: list[int], evaluations: list[dict], success_threshold: float = 4.0
) -> dict[str, Any]:
    """Calculate evaluation metrics from scores and evaluations.

    Args:
        scores: List of judge scores (1-5)
        evaluations: List of evaluation results
        success_threshold: Score threshold for success (default: 4.0)

    Returns:
        Dictionary with calculated metrics
    """
    if not scores:
        return {
            "functional_correctness_score": 0,
            "success_rate": 0,
            "exact_match_rate": 0,
            "total_cases": 0,
            "successful_cases": 0,
            "exact_matches": 0,
            "score_distribution": {},
            "failure_count": 0,
        }

    # Filter out zero scores (errors)
    valid_scores = [s for s in scores if s > 0]

    # Functional Correctness Score (average)
    fcs = statistics.mean(valid_scores) if valid_scores else 0

    # Success Rate (percentage scoring >= threshold)
    success_count = sum(1 for s in valid_scores if s >= success_threshold)
    success_rate = (success_count / len(valid_scores)) * 100 if valid_scores else 0

    # Exact Match Rate (absolute matching without reasoning)
    exact_matches = sum(1 for e in evaluations if e.get("exact_match", False))
    exact_match_rate = (exact_matches / len(evaluations)) * 100 if evaluations else 0

    # Score distribution
    score_dist = {i: scores.count(i) for i in range(1, 6)}

    # Failure analysis
    failures = [e for e in evaluations if e.get("score", 0) < success_threshold]

    return {
        "functional_correctness_score": round(fcs, 2),
        "success_rate": round(success_rate, 1),
        "exact_match_rate": round(exact_match_rate, 1),
        "total_cases": len(scores),
        "valid_cases": len(valid_scores),
        "successful_cases": success_count,
        "exact_matches": exact_matches,
        "score_distribution": score_dist,
        "failure_count": len(failures),
    }


def display_results(metrics: dict[str, Any], title: str = "Benchmark Results") -> None:
    """Display evaluation results in a formatted table.

    Args:
        metrics: Calculated metrics to display
        title: Title to display (default: "Benchmark Results")
    """
    console.print(f"\n[bold cyan]{title}[/bold cyan]")
    console.print("=" * 50)

    table = Table(title="Performance Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row(
        "Functional Correctness Score (FCS)",
        f"{metrics['functional_correctness_score']}/5.0",
    )
    table.add_row("Success Rate (Judge)", f"{metrics['success_rate']}%")
    table.add_row("Exact Match Rate", f"{metrics.get('exact_match_rate', 0)}%")
    table.add_row("Total Cases", str(metrics["total_cases"]))
    table.add_row(
        "Valid Cases", str(metrics.get("valid_cases", metrics["total_cases"]))
    )
    table.add_row("Successful Cases (Judge)", str(metrics["successful_cases"]))
    table.add_row("Exact Matches", str(metrics.get("exact_matches", 0)))
    table.add_row("Failed Cases", str(metrics["failure_count"]))

    console.print(table)

    # Score distribution
    if metrics.get("score_distribution"):
        console.print("\n[bold]Score Distribution:[/bold]")
        for score, count in sorted(metrics["score_distribution"].items()):
            if count > 0:
                bar = "█" * count
                console.print(f"  Score {score}: {bar} ({count})")


def format_prediction_output(prediction: dict, include_confidence: bool = True) -> dict:
    """Format a prediction for output.

    Args:
        prediction: Raw prediction from VLM
        include_confidence: Whether to include confidence scores

    Returns:
        Formatted prediction dictionary
    """
    output = {
        "id": prediction["id"],
        "image_path": prediction.get("image_path", ""),
        "materials": prediction.get("vlm_response", ""),
    }

    if include_confidence and "confidence" in prediction:
        output["confidence"] = prediction["confidence"]

    return output
