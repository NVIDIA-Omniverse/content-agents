# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Minimal client for the OVRTX Rendering API — used by CI smoke tests.

Encodes a local USD file as a ``data:`` URI and POSTs it to ``/render``,
then verifies a non-empty ``images`` map comes back. Works identically
against a local docker container and an NVCF function URL.

Usage:
    python apps/ovrtx_rendering_api/client/client.py \\
        --base-url http://localhost:8000 \\
        --usd apps/ovrtx_rendering_api/tests/renders/smoke_cube.usda
"""

from __future__ import annotations

import argparse
import base64
import json
import pathlib
import sys
from typing import Any

import requests


def _encode_usd_as_data_uri(path: pathlib.Path) -> str:
    """Base64-encode a USD file into a data: URI the renderer understands."""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:application/octet-stream;base64,{encoded}"


def _build_request(usd_data_uri: str) -> dict[str, Any]:
    return {
        "url": usd_data_uri,
        "force_render": True,
        "render_settings": {
            "camera_paths": ["/World/Camera"],
            "frame_range": {"start": 0, "end": 0},
            "camera_parameters": {"width": 256, "height": 256},
            "sensors": None,
            "apply_background_mask": False,
        },
    }


def health_check(base_url: str, token: str | None, timeout: float) -> None:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    r = requests.get(f"{base_url.rstrip('/')}/health", headers=headers, timeout=timeout)
    r.raise_for_status()
    body = r.json()
    print(f"health: {body}")
    if not body.get("gpu_initialized", False):
        raise SystemExit(
            "✗ /health returned but gpu_initialized=false — renderer did not start"
        )
    print("✓ /health passed, gpu_initialized=true")


def render_smoke(
    base_url: str, usd_path: pathlib.Path, token: str | None, timeout: float
) -> None:
    data_uri = _encode_usd_as_data_uri(usd_path)
    payload = _build_request(data_uri)
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    headers["Content-Type"] = "application/json"

    print(f"POST /render with {usd_path.name} ({usd_path.stat().st_size} bytes)")
    r = requests.post(
        f"{base_url.rstrip('/')}/render",
        data=json.dumps(payload),
        headers=headers,
        timeout=timeout,
    )
    r.raise_for_status()
    body = r.json()
    status = body.get("status", "unknown")
    error = body.get("error")
    images = body.get("images", {})

    if status != "success":
        raise SystemExit(f"✗ /render status={status} error={error}")
    if not images:
        raise SystemExit("✗ /render returned empty images map")

    # Structure: images[frame][camera][sensor] = base64
    total_images = sum(
        1 for frame in images.values() for cam in frame.values() for _ in cam.values()
    )
    print(f"✓ /render produced {total_images} image(s)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", required=True, help="Service base URL")
    p.add_argument(
        "--usd",
        type=pathlib.Path,
        default=pathlib.Path(__file__).parent.parent
        / "tests"
        / "renders"
        / "smoke_cube.usda",
        help="Path to a small USD file to render",
    )
    p.add_argument("--token", default=None, help="Bearer token (for NVCF)")
    p.add_argument("--skip-health", action="store_true", help="Skip /health probe")
    p.add_argument(
        "--timeout", type=float, default=300.0, help="Per-request HTTP timeout"
    )
    args = p.parse_args()

    if not args.usd.exists():
        print(f"✗ USD fixture not found: {args.usd}", file=sys.stderr)
        sys.exit(1)

    if not args.skip_health:
        health_check(args.base_url, args.token, args.timeout)

    render_smoke(args.base_url, args.usd, args.token, args.timeout)
    print("✓ all checks passed")


if __name__ == "__main__":
    main()
