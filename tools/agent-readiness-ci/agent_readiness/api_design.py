from __future__ import annotations

from pathlib import Path

from .artifacts import write_json, write_text
from .config import ReadinessConfig


MAX_FILE_BYTES = 512_000
MAX_CANDIDATE_FILES = 500


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")[:MAX_FILE_BYTES]


def _rel(repo: Path, path: Path) -> str:
    return path.relative_to(repo).as_posix()


def _candidate_files(repo: Path) -> list[Path]:
    exact_names = {
        "README.md",
        "AGENTS.md",
        "CLAUDE.md",
        "llms.txt",
        "llms-full.txt",
        "openapi.yaml",
        "openapi.yml",
        "openapi.json",
        "swagger.yaml",
        "swagger.yml",
        "swagger.json",
        "schema.graphql",
        "package.json",
        "pyproject.toml",
    }
    suffixes = {".md", ".yaml", ".yml", ".json", ".graphql", ".proto"}
    files: set[Path] = set()
    for name in exact_names:
        path = repo / name
        if path.is_file():
            files.add(path)
    for root_name in ("docs", "api", "apis", "openapi", "schema", "schemas"):
        root = repo / root_name
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in suffixes:
                files.add(path)
    for pattern in (
        "**/openapi*.yaml",
        "**/openapi*.yml",
        "**/openapi*.json",
        "**/swagger*.yaml",
        "**/swagger*.yml",
        "**/swagger*.json",
        "**/*api*.md",
        "**/*.graphql",
        "**/*.proto",
        "**/mcp*.json",
    ):
        for path in repo.glob(pattern):
            if _is_reasonable_candidate(repo, path):
                files.add(path)
    return sorted(files, key=lambda item: item.as_posix())[:MAX_CANDIDATE_FILES]


def _is_reasonable_candidate(repo: Path, path: Path) -> bool:
    if not path.is_file():
        return False
    rel = _rel(repo, path)
    blocked_parts = {
        ".git",
        ".venv",
        "node_modules",
        "__pycache__",
        ".pytest_cache",
        "dist",
        "build",
    }
    if set(Path(rel).parts) & blocked_parts:
        return False
    return path.suffix.lower() in {".md", ".yaml", ".yml", ".json", ".graphql", ".proto"}


def _build_corpus(repo: Path) -> list[tuple[Path, str, str]]:
    corpus: list[tuple[Path, str, str]] = []
    for path in _candidate_files(repo):
        text = _read_text(path)
        corpus.append((path, _rel(repo, path), text))
    return corpus


def _matching_paths(corpus: list[tuple[Path, str, str]], terms: list[str]) -> list[str]:
    matches: list[str] = []
    for _, rel, text in corpus:
        lower = text.lower()
        if any(term.lower() in lower for term in terms):
            matches.append(rel)
    return sorted(set(matches))


def _count_terms(corpus: list[tuple[Path, str, str]], terms: list[str]) -> int:
    combined = "\n".join(text.lower() for _, _, text in corpus)
    return sum(1 for term in terms if term.lower() in combined)


def _exists_path(repo: Path, names: list[str]) -> list[str]:
    return [name for name in names if (repo / name).exists()]


def _criterion(
    category: str,
    status: str,
    summary: str,
    evidence: list[str],
    recommendation: str,
    blocker: bool = False,
) -> dict:
    if status == "Pass":
        severity = "none"
        points = 10
    elif status == "Partial":
        severity = "minor"
        points = 5
    else:
        severity = "major" if blocker else "minor"
        points = 0
    return {
        "category": category,
        "status": status,
        "method": "Static API Design",
        "severity": severity,
        "blocker": blocker and status == "Fail",
        "points": points,
        "max_points": 10,
        "summary": summary,
        "evidence": evidence,
        "recommendation": recommendation,
    }


def scan_api_design(config: ReadinessConfig) -> dict:
    repo = config.repo_path
    corpus = _build_corpus(repo)
    checks: list[dict] = []

    contract_files = _matching_paths(
        corpus,
        ["openapi:", '"openapi"', "swagger", "schema.graphql", "type Query", "mcpServers", "tools/list"],
    )
    contract_fallback = _exists_path(repo, ["api", "apis", "docs/api.md", "pyproject.toml", "package.json"])
    if contract_files:
        contract_status = "Pass"
        contract_summary = "Machine-readable API/tool contracts are discoverable."
        contract_evidence = contract_files
    elif contract_fallback:
        contract_status = "Partial"
        contract_summary = "API surface exists, but a machine-readable contract was not found."
        contract_evidence = contract_fallback
    else:
        contract_status = "Fail"
        contract_summary = "No machine-readable API, GraphQL, MCP, SDK, or CLI contract was found."
        contract_evidence = []
    checks.append(
        _criterion(
            "machine_readable_contract",
            contract_status,
            contract_summary,
            contract_evidence,
            "Publish OpenAPI/GraphQL/MCP schemas, typed SDK contracts, or CLI command schemas in a stable path.",
            blocker=True,
        )
    )

    task_terms = [
        "operationid",
        "create",
        "start",
        "run",
        "submit",
        "validate",
        "deploy",
        "provision",
        "migrate",
        "publish",
        "session",
        "job",
        "workflow",
    ]
    task_count = _count_terms(corpus, task_terms)
    checks.append(
        _criterion(
            "task_shaped_operations",
            "Pass" if task_count >= 5 else ("Partial" if task_count >= 2 else "Fail"),
            "API operations appear task-shaped." if task_count >= 5 else "API operations may be too low-level or under-documented.",
            _matching_paths(corpus, task_terms),
            "Expose task-level operations such as validate, deploy, run, submit, create job, cancel, and resume.",
        )
    )

    safety_terms = ["dry_run", "dry run", "preview", "sandbox", "test mode", "idempotency", "idempotent", "rollback", "cancel"]
    safety_count = _count_terms(corpus, safety_terms)
    checks.append(
        _criterion(
            "safe_execution_modes",
            "Pass" if safety_count >= 3 else ("Partial" if safety_count >= 1 else "Fail"),
            "Safe execution controls are documented." if safety_count >= 3 else "Safe execution controls are incomplete.",
            _matching_paths(corpus, safety_terms),
            "Add dry-run/preview, sandbox or test mode, idempotency keys, cancel, rollback, and production-safety guidance.",
            blocker=True,
        )
    )

    error_terms = ["error_code", "error code", "status code", "retryable", "remediation", "troubleshoot", "docs link", "400", "401", "403", "429"]
    error_count = _count_terms(corpus, error_terms)
    checks.append(
        _criterion(
            "structured_error_recovery",
            "Pass" if error_count >= 3 else ("Partial" if error_count >= 1 else "Fail"),
            "Structured error and remediation guidance is present." if error_count >= 3 else "Error recovery is not structured enough for agents.",
            _matching_paths(corpus, error_terms),
            "Return stable error codes, retryability, field-level validation, missing permission/env names, and remediation links.",
            blocker=True,
        )
    )

    auth_terms = ["oauth", "scope", "scopes", "token", "api key", "permission", "permissions", "least privilege", "actor", "service account"]
    auth_count = _count_terms(corpus, auth_terms)
    checks.append(
        _criterion(
            "scoped_auth_and_permissions",
            "Pass" if auth_count >= 3 else ("Partial" if auth_count >= 1 else "Fail"),
            "Authentication and permission boundaries are documented." if auth_count >= 3 else "Auth scopes and permission behavior are underspecified.",
            _matching_paths(corpus, auth_terms),
            "Document scoped credentials, app/user identity, least privilege scopes, and explicit permission failure behavior.",
            blocker=True,
        )
    )

    async_terms = ["job", "session", "status", "progress", "logs", "artifacts", "cancel", "resume", "webhook", "event", "events"]
    async_count = _count_terms(corpus, async_terms)
    checks.append(
        _criterion(
            "long_running_operation_model",
            "Pass" if async_count >= 4 else ("Partial" if async_count >= 2 else "Fail"),
            "Long-running operation state is documented." if async_count >= 4 else "Long-running operation semantics are incomplete.",
            _matching_paths(corpus, async_terms),
            "Represent async work as jobs/sessions with status, progress, logs, artifacts, cancel/resume, and events/webhooks.",
        )
    )

    example_paths = _matching_paths(corpus, ["```", "request:", "response:", "curl ", "example", "examples:"])
    request_response_count = _count_terms(corpus, ["request:", "response:", "curl ", "examples:"])
    has_code_block = bool(_matching_paths(corpus, ["```"]))
    checks.append(
        _criterion(
            "deterministic_examples",
            "Pass" if has_code_block and request_response_count >= 2 else ("Partial" if has_code_block or request_response_count else "Fail"),
            "Request/response examples are available." if has_code_block and request_response_count >= 2 else "API examples are not deterministic enough.",
            example_paths,
            "Provide copy-paste request and response examples that run in sandbox/test mode with expected outputs.",
        )
    )

    observability_terms = ["request_id", "request id", "trace_id", "trace id", "audit", "logs", "artifact", "artifacts", "telemetry", "event"]
    observability_count = _count_terms(corpus, observability_terms)
    checks.append(
        _criterion(
            "observable_outputs",
            "Pass" if observability_count >= 3 else ("Partial" if observability_count >= 1 else "Fail"),
            "API outputs expose useful operational evidence." if observability_count >= 3 else "API outputs lack observable evidence for agents.",
            _matching_paths(corpus, observability_terms),
            "Return request IDs, trace IDs, audit records, logs, events, and artifact URLs so CI can prove what happened.",
        )
    )

    version_terms = ["api version", "version:", '"version"', "changelog", "deprecat", "/v1", "/v2"]
    version_count = _count_terms(corpus, version_terms)
    checks.append(
        _criterion(
            "versioning_and_compatibility",
            "Pass" if version_count >= 1 else "Partial",
            "Versioning or compatibility signals are present." if version_count >= 1 else "Versioning behavior is not explicit.",
            _matching_paths(corpus, version_terms),
            "Document API versions, deprecation policy, changelog behavior, and backwards-compatibility expectations.",
        )
    )

    tool_terms = ["mcp", "cli", "sdk", "client", "openapi", "graphql", "llms.txt", "agent"]
    tool_count = _count_terms(corpus, tool_terms)
    checks.append(
        _criterion(
            "agent_tool_surface_parity",
            "Pass" if tool_count >= 4 else ("Partial" if tool_count >= 2 else "Fail"),
            "Agent-facing tool surfaces are visible." if tool_count >= 4 else "Tool-surface parity across API/CLI/SDK/MCP is unclear.",
            _matching_paths(corpus, tool_terms),
            "Keep API, CLI, SDK, MCP, and docs concepts aligned with matching operation names and examples.",
        )
    )

    points = sum(item["points"] for item in checks)
    max_points = sum(item["max_points"] for item in checks)
    blockers = [item["category"] for item in checks if item["blocker"]]
    return {
        "product_id": config.product_id,
        "name": config.name,
        "lane": "api-design",
        "checks": checks,
        "score": {
            "points": points,
            "max_points": max_points,
            "ratio": round(points / max_points, 4) if max_points else 0.0,
        },
        "totals": {
            "pass": sum(1 for item in checks if item["status"] == "Pass"),
            "partial": sum(1 for item in checks if item["status"] == "Partial"),
            "fail": sum(1 for item in checks if item["status"] == "Fail"),
        },
        "blockers": blockers,
        "candidate_files": [rel for _, rel, _ in corpus],
    }


def write_api_design_artifacts(run_path: Path, payload: dict) -> None:
    write_json(run_path / "api-design-readiness.json", payload)
    lines = [
        f"# API Design Readiness - {payload['name']}",
        "",
        f"Score: {payload['score']['points']}/{payload['score']['max_points']} ({payload['score']['ratio']:.2f})",
        "",
        "| Category | Status | Evidence | Recommendation |",
        "|---|---|---|---|",
    ]
    for item in payload["checks"]:
        evidence = ", ".join(item["evidence"]) if item["evidence"] else "none"
        lines.append(f"| {item['category']} | {item['status']} | {evidence} | {item['recommendation']} |")
    if payload["blockers"]:
        lines.extend(["", "## Blocking Gaps", ""])
        for blocker in payload["blockers"]:
            lines.append(f"- {blocker}")
    lines.append("")
    write_text(run_path / "api-design-readiness.md", "\n".join(lines))
