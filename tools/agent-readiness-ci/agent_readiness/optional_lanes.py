from __future__ import annotations

from pathlib import Path

from .artifacts import write_json, write_text
from .config import ReadinessConfig


def _mode(config: ReadinessConfig, key: str) -> dict:
    raw = ((config.raw.get("modes") or {}).get(key) or {})
    return raw if isinstance(raw, dict) else {}


def run_remote_placeholder(config: ReadinessConfig, run_path: Path) -> dict:
    remote_cfg = _mode(config, "remote_deployment")
    enabled = bool(remote_cfg.get("enabled", False))
    payload = {
        "product_id": config.product_id,
        "lane": "remote",
        "status": "Skipped" if not enabled else "Blocked",
        "method": "Remote Live",
        "reason": "remote_deployment.enabled is false"
        if not enabled
        else "remote deployment is enabled but no generic runner adapter has been configured",
    }
    write_json(run_path / "remote" / "remote-result.json", payload)
    write_text(run_path / "remote" / "README.md", payload["reason"] + "\n")
    return payload


def run_hosted_litmus_placeholder(config: ReadinessConfig, run_path: Path) -> dict:
    litmus_cfg = _mode(config, "litmus_vdr")
    enabled = bool(litmus_cfg.get("enabled", False))
    payload = {
        "product_id": config.product_id,
        "lane": "hosted-litmus",
        "status": "Skipped" if not enabled else "Blocked",
        "method": "Hosted",
        "reason": "litmus_vdr.enabled is false"
        if not enabled
        else "Litmus VDR is enabled but no service URL or MCP adapter has been configured",
    }
    write_json(run_path / "hosted" / "litmus-run.json", payload)
    write_text(run_path / "hosted" / "litmus-report.md", f"# Litmus VDR\n\n{payload['reason']}\n")
    return payload


def run_fvr_placeholder(config: ReadinessConfig, run_path: Path) -> dict:
    fvr_cfg = _mode(config, "fvr_rc")
    enabled = bool(fvr_cfg.get("enabled", False))
    payload = {
        "product_id": config.product_id,
        "lane": "fvr-rc",
        "status": "Skipped" if not enabled else "Blocked",
        "method": "FVR RC",
        "reason": "fvr_rc.enabled is false"
        if not enabled
        else "FVR RC validation is enabled but no execution adapter has been configured",
        "native_artifacts": [
            "00-test-plan.yaml",
            "01-doc-review.md",
            "01d-agentic-readiness.md",
            "02-commands.jsonl",
            "03-test-results.md",
            "04-perf-results.md",
            "05-quality-review.md",
            "07-grounded-report.md",
        ],
    }
    write_json(run_path / "fvr" / "fvr-summary.json", payload)
    write_text(run_path / "fvr" / "report.md", f"# FVR RC\n\n{payload['reason']}\n")
    return payload
