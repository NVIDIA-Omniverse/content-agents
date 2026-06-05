from __future__ import annotations

import html
import shutil
from pathlib import Path

from .artifacts import read_json
from .config import ReadinessConfig


def publish(config: ReadinessConfig, runs_root: Path, public: Path) -> None:
    run_path = runs_root / config.product_id / "latest"
    public.mkdir(parents=True, exist_ok=True)
    summary = read_json(run_path / "ci-summary.json") if (run_path / "ci-summary.json").exists() else {}
    static_md = _read_optional(run_path / "static-readiness.md")
    api_design_md = _read_optional(run_path / "api-design-readiness.md")
    scorecard_md = _read_optional(run_path / "scorecard.md")
    status = summary.get("status", "unknown")
    rate = summary.get("p0_live_pass_rate", "n/a")
    api_score = summary.get("api_design_score", "n/a")
    body = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Agent Readiness - {html.escape(config.name)}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; line-height: 1.45; }}
    code, pre {{ background: #f5f5f5; padding: 2px 4px; }}
    pre {{ padding: 16px; overflow: auto; }}
    .status {{ display: inline-block; padding: 4px 8px; border-radius: 4px; background: #eee; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: left; }}
  </style>
</head>
<body>
  <h1>Agent Readiness - {html.escape(config.name)}</h1>
  <p>Status: <strong class="status">{html.escape(str(status))}</strong></p>
  <p>P0 live pass rate: <strong>{html.escape(str(rate))}</strong></p>
  <p>API design score: <strong>{html.escape(str(api_score))}</strong></p>
  <h2>Static Readiness</h2>
  <pre>{html.escape(static_md)}</pre>
  <h2>API Design Readiness</h2>
  <pre>{html.escape(api_design_md)}</pre>
  <h2>Scorecard</h2>
  <pre>{html.escape(scorecard_md)}</pre>
</body>
</html>
"""
    (public / "index.html").write_text(body, encoding="utf-8")
    if run_path.exists():
        artifact_dir = public / "artifacts"
        if artifact_dir.exists():
            shutil.rmtree(artifact_dir)
        shutil.copytree(run_path, artifact_dir)


def _read_optional(path: Path) -> str:
    if not path.exists():
        return "Not generated."
    return path.read_text(encoding="utf-8")
