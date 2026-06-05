from __future__ import annotations

from pathlib import Path

from .artifacts import write_json, write_text
from .config import ReadinessConfig, Job


def _prompt_for(config: ReadinessConfig, jobs: list[Job]) -> str:
    lines = [
        f"You are evaluating agent readiness for {config.name}.",
        f"Repository path: {config.repo_path}",
        "",
        "Attempt each job through shipped docs and product-supported surfaces.",
        "Do not count evaluator-written glue as a full pass.",
        "",
    ]
    for job in jobs:
        lines += [
            f"==={job.id}===",
            f"Priority: {job.priority}",
            f"Title: {job.title}",
            "Success criteria:",
        ]
        lines += [f"- {criterion}" for criterion in job.success_criteria]
        lines.append("")
    lines.append("After the final job, print ===END RUN=== and summarize pass/fail evidence.")
    return "\n".join(lines)


def run_dry(config: ReadinessConfig, run_path: Path, agent: str) -> dict:
    run_root = run_path / "live" / "runs" / agent
    run_root.mkdir(parents=True, exist_ok=True)
    prompt = _prompt_for(config, config.jobs)
    write_text(run_root / "prompt.txt", prompt + "\n")
    write_json(
        run_root / "result.json",
        {
            "agent": agent,
            "mode": "dry-run",
            "status": "skipped",
            "jobs_in_scope": [job.id for job in config.jobs],
            "reason": "dry-run generated the prompt but did not invoke a coding agent",
        },
    )
    rows = [
        {
            "id": job.id,
            "priority": job.priority,
            "status": "Skipped",
            "method": "Skipped",
            "evidence": f"Dry run only; prompt generated at live/runs/{agent}/prompt.txt.",
        }
        for job in config.jobs
    ]
    payload = {
        "product_id": config.product_id,
        "name": config.name,
        "lane": "headless",
        "agent": agent,
        "rows": rows,
    }
    write_scorecard(run_path, payload)
    return payload


def write_scorecard(run_path: Path, payload: dict) -> None:
    write_json(run_path / "scorecard.json", payload)
    lines = [
        f"# Scorecard - {payload['name']} x {payload.get('agent', 'agent')}",
        "",
        "| ID | Priority | Status | Method | Evidence |",
        "|---|---|---|---|---|",
    ]
    for row in payload["rows"]:
        lines.append(f"| {row['id']} | {row['priority']} | {row['status']} | {row['method']} | {row['evidence']} |")
    lines.append("")
    write_text(run_path / "scorecard.md", "\n".join(lines))
    write_text(run_path / "journey-report.md", "# Journey Report\n\nDry-run mode generated prompts only.\n")
    write_text(run_path / "issues.md", "# Issues\n\nNo issues generated in dry-run mode.\n")
    write_text(run_path / "executive-summary.md", "# Executive Summary\n\nDry-run mode does not produce a release verdict.\n")
