from __future__ import annotations

from pathlib import Path

from .artifacts import write_json, write_text
from .config import ReadinessConfig


def _exists_any(repo: Path, names: list[str]) -> bool:
    return any((repo / name).exists() for name in names)


def _contains_any(repo: Path, names: list[str], needles: list[str]) -> bool:
    for name in names:
        path = repo / name
        if not path.exists() or not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        if any(needle.lower() in text for needle in needles):
            return True
    return False


def _has_code_block(path: Path) -> bool:
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8", errors="ignore")
    return "```" in text or "$ " in text or "python " in text or "npm " in text or "pip " in text


def _check(status: str, category: str, summary: str, evidence: list[str], recommendation: str) -> dict:
    severity = "none" if status == "Pass" else ("major" if status == "Fail" else "minor")
    return {
        "category": category,
        "status": status,
        "method": "Static",
        "severity": severity,
        "summary": summary,
        "evidence": evidence,
        "recommendation": recommendation,
    }


def scan(config: ReadinessConfig) -> dict:
    repo = config.repo_path
    checks: list[dict] = []

    readme = repo / "README.md"
    has_readme = readme.exists()
    checks.append(
        _check(
            "Pass" if has_readme else "Fail",
            "readme_first_screen",
            "README.md is present." if has_readme else "README.md is missing.",
            ["README.md"] if has_readme else [],
            "Add a README with product purpose, install, quickstart, and validation commands.",
        )
    )

    has_entrypoint = _exists_any(repo, ["AGENTS.md", "CLAUDE.md", "skills.md", ".agents"])
    checks.append(
        _check(
            "Pass" if has_entrypoint else "Fail",
            "agent_entrypoints",
            "Agent-facing instructions are present." if has_entrypoint else "No agent-facing entrypoint found.",
            [name for name in ["AGENTS.md", "CLAUDE.md", "skills.md", ".agents"] if (repo / name).exists()],
            "Add AGENTS.md or equivalent guidance covering setup, commands, validation, and safety boundaries.",
        )
    )

    has_env_docs = _exists_any(repo, [".env.example", "env.example", "requirements.txt", "pyproject.toml", "Dockerfile"])
    checks.append(
        _check(
            "Pass" if has_env_docs else "Partial",
            "environment_reproducibility",
            "Environment hints are present." if has_env_docs else "Environment requirements are not explicit.",
            [name for name in [".env.example", "env.example", "requirements.txt", "pyproject.toml", "Dockerfile"] if (repo / name).exists()],
            "Declare required env vars, package manager, OS/hardware assumptions, and a clean setup command.",
        )
    )

    has_quickstart = has_readme and _has_code_block(readme)
    has_examples = _exists_any(repo, ["examples", "example", "samples", "sample"])
    checks.append(
        _check(
            "Pass" if has_quickstart and has_examples else "Partial",
            "examples_quickstarts",
            "Quickstart commands and examples are present." if has_quickstart and has_examples else "Quickstart or examples are incomplete.",
            [item for item in ["README.md", "examples/"] if (repo / item).exists() or item == "README.md" and has_quickstart],
            "Provide a copy-paste hello world, sample input, expected output, and at least one negative/error case.",
        )
    )

    has_api_surface = _exists_any(repo, ["openapi.yaml", "openapi.json", "api", "src", "python"]) or _contains_any(
        repo, ["README.md", "pyproject.toml", "package.json"], ["cli", "api", "[project.scripts]", "entry_points"]
    )
    checks.append(
        _check(
            "Pass" if has_api_surface else "Partial",
            "api_cli_surface",
            "An automatable API/CLI surface is discoverable." if has_api_surface else "No clear automatable API or CLI surface found.",
            [name for name in ["openapi.yaml", "openapi.json", "api", "src", "python", "pyproject.toml", "package.json"] if (repo / name).exists()],
            "Document the supported CLI/API/SDK entrypoints with examples and authentication behavior.",
        )
    )

    has_troubleshooting = _contains_any(repo, ["README.md", "AGENTS.md", "docs/troubleshooting.md", "TROUBLESHOOTING.md"], ["troubleshoot", "error", "fails", "auth"])
    checks.append(
        _check(
            "Pass" if has_troubleshooting else "Partial",
            "error_recovery",
            "Troubleshooting or recovery guidance is present." if has_troubleshooting else "Error recovery guidance is thin.",
            [name for name in ["docs/troubleshooting.md", "TROUBLESHOOTING.md", "AGENTS.md", "README.md"] if (repo / name).exists()],
            "Add common failure modes with exact symptoms, likely causes, and next commands.",
        )
    )

    return {
        "product_id": config.product_id,
        "name": config.name,
        "lane": "static",
        "checks": checks,
        "totals": {
            "pass": sum(1 for item in checks if item["status"] == "Pass"),
            "partial": sum(1 for item in checks if item["status"] == "Partial"),
            "fail": sum(1 for item in checks if item["status"] == "Fail"),
        },
    }


def write_static_artifacts(run_path: Path, payload: dict) -> None:
    write_json(run_path / "static-readiness.json", payload)
    lines = [
        f"# Static Readiness - {payload['name']}",
        "",
        "| Category | Status | Evidence | Recommendation |",
        "|---|---|---|---|",
    ]
    for item in payload["checks"]:
        evidence = ", ".join(item["evidence"]) if item["evidence"] else "none"
        lines.append(f"| {item['category']} | {item['status']} | {evidence} | {item['recommendation']} |")
    lines.append("")
    write_text(run_path / "static-readiness.md", "\n".join(lines))
