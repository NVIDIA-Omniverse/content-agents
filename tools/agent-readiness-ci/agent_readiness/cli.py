from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .api_design import scan_api_design, write_api_design_artifacts
from .artifacts import run_dir
from .config import ConfigError, load_config
from .headless import run_dry
from .optional_lanes import run_fvr_placeholder, run_hosted_litmus_placeholder, run_remote_placeholder
from .publish import publish as publish_site
from .static_scan import scan, write_static_artifacts
from .summary import summarize as summarize_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agent-readiness")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate-config", help="Validate agent-readiness.yaml")
    validate.add_argument("config", type=Path)

    run = sub.add_parser("run", help="Run one readiness lane")
    run.add_argument("--lane", required=True, choices=["static", "api-design", "headless", "remote", "hosted-litmus", "fvr-rc"])
    run.add_argument("--config", type=Path, default=Path("agent-readiness.yaml"))
    run.add_argument("--out", type=Path, default=Path("agent-readiness-runs"))
    run.add_argument("--agent", default="dry-run")

    summarize = sub.add_parser("summarize", help="Create ci-summary.json")
    summarize.add_argument("--config", type=Path, default=Path("agent-readiness.yaml"))
    summarize.add_argument("--out", type=Path, default=Path("agent-readiness-runs"))
    summarize.add_argument("--fail-on-threshold", action="store_true")

    publish = sub.add_parser("publish", help="Publish a minimal static dashboard")
    publish.add_argument("--config", type=Path, default=Path("agent-readiness.yaml"))
    publish.add_argument("--runs", type=Path, default=Path("agent-readiness-runs"))
    publish.add_argument("--out", type=Path, default=Path("public"))

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "validate-config":
            config = load_config(args.config)
            print(f"OK: {config.product_id} ({len(config.jobs)} job(s))")
            return 0

        if args.command == "run":
            config = load_config(args.config)
            path = run_dir(args.out, config)
            if args.lane == "static":
                payload = scan(config)
                write_static_artifacts(path, payload)
                print(f"wrote {path / 'static-readiness.json'}")
                return 0
            if args.lane == "api-design":
                payload = scan_api_design(config)
                write_api_design_artifacts(path, payload)
                print(f"wrote {path / 'api-design-readiness.json'}")
                return 0
            if args.lane == "headless":
                if args.agent != "dry-run":
                    raise ConfigError("only --agent dry-run is implemented in this local toolkit version")
                run_dry(config, path, args.agent)
                print(f"wrote {path / 'scorecard.json'}")
                return 0
            if args.lane == "remote":
                run_remote_placeholder(config, path)
                print(f"wrote {path / 'remote' / 'remote-result.json'}")
                return 0
            if args.lane == "hosted-litmus":
                run_hosted_litmus_placeholder(config, path)
                print(f"wrote {path / 'hosted' / 'litmus-run.json'}")
                return 0
            if args.lane == "fvr-rc":
                run_fvr_placeholder(config, path)
                print(f"wrote {path / 'fvr' / 'fvr-summary.json'}")
                return 0

        if args.command == "summarize":
            config = load_config(args.config)
            payload = summarize_run(config, run_dir(args.out, config))
            print(f"{payload['status']}: P0 live pass rate {payload['p0_live_pass_rate']:.2f}")
            if args.fail_on_threshold and payload["status"] != "passed":
                return 1
            return 0

        if args.command == "publish":
            config = load_config(args.config)
            publish_site(config, args.runs, args.out)
            print(f"wrote {args.out / 'index.html'}")
            return 0
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
