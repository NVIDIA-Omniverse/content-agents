from __future__ import annotations

from pathlib import Path

from .artifacts import read_json, write_json
from .config import ReadinessConfig


LIVE_METHODS = {"Live", "Remote Live", "Hosted"}


def summarize(config: ReadinessConfig, run_path: Path) -> dict:
    scorecard_path = run_path / "scorecard.json"
    rows = read_json(scorecard_path).get("rows", []) if scorecard_path.exists() else []
    p0_rows = [row for row in rows if row.get("priority") == "P0"]
    p0_live_passes = [
        row
        for row in p0_rows
        if row.get("status") == "Pass" and row.get("method") in LIVE_METHODS
    ]
    p0_fails = [row for row in p0_rows if row.get("status") == "Fail"]
    p0_live_pass_rate = (len(p0_live_passes) / len(p0_rows)) if p0_rows else 0.0

    failures: list[str] = []
    if p0_live_pass_rate < config.thresholds.p0_live_pass_rate_min:
        failures.append(
            f"P0 live pass rate {p0_live_pass_rate:.2f} is below threshold {config.thresholds.p0_live_pass_rate_min:.2f}"
        )
    if config.thresholds.fail_on_p0_fail and p0_fails:
        failures.append(f"{len(p0_fails)} P0 job(s) failed")

    static_path = run_path / "static-readiness.json"
    static = read_json(static_path) if static_path.exists() else None
    static_failures = []
    if static:
        static_failures = [item for item in static.get("checks", []) if item.get("status") == "Fail"]
        if static_failures:
            failures.append(f"{len(static_failures)} static readiness check(s) failed")

    api_design_path = run_path / "api-design-readiness.json"
    api_design = read_json(api_design_path) if api_design_path.exists() else None
    api_design_score = None
    api_design_failures: list[str] = []
    api_design_blockers: list[str] = []
    if api_design:
        api_design_score = float(api_design.get("score", {}).get("ratio", 0.0))
        api_design_failures = [
            item["category"]
            for item in api_design.get("checks", [])
            if item.get("status") == "Fail"
        ]
        api_design_blockers = list(api_design.get("blockers", []))
        if api_design_score < config.thresholds.api_design_score_min:
            failures.append(
                f"API design score {api_design_score:.2f} is below threshold {config.thresholds.api_design_score_min:.2f}"
            )
        if config.thresholds.fail_on_unverified_blockers and api_design_blockers:
            failures.append(f"{len(api_design_blockers)} blocking API design check(s) failed")

    payload = {
        "product_id": config.product_id,
        "name": config.name,
        "status": "passed" if not failures else "failed",
        "p0_jobs": len(p0_rows),
        "p0_live_passes": len(p0_live_passes),
        "p0_live_pass_rate": round(p0_live_pass_rate, 4),
        "thresholds": {
            "p0_live_pass_rate_min": config.thresholds.p0_live_pass_rate_min,
            "api_design_score_min": config.thresholds.api_design_score_min,
            "fail_on_p0_fail": config.thresholds.fail_on_p0_fail,
            "fail_on_unverified_blockers": config.thresholds.fail_on_unverified_blockers,
        },
        "failures": failures,
        "static_failures": [item["category"] for item in static_failures],
        "api_design_score": api_design_score,
        "api_design_failures": api_design_failures,
        "api_design_blockers": api_design_blockers,
    }
    write_json(run_path / "ci-summary.json", payload)
    write_junit(run_path, payload)
    return payload


def write_junit(run_path: Path, payload: dict) -> None:
    failures = payload.get("failures", [])
    if failures:
        failure_xml = "\n".join(
            f'    <failure message="{_xml_escape(item)}">{_xml_escape(item)}</failure>'
            for item in failures
        )
    else:
        failure_xml = ""
    text = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<testsuite name="agent-readiness" tests="1" failures="{1 if failures else 0}">\n'
        f'  <testcase classname="agent-readiness" name="{_xml_escape(payload["product_id"])}">\n'
        f"{failure_xml}\n"
        "  </testcase>\n"
        "</testsuite>\n"
    )
    (run_path / "junit.xml").write_text(text, encoding="utf-8")


def _xml_escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
