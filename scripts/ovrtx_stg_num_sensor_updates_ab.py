# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A/B test the num_sensor_updates field against the deployed OVRtx STG NVCF service.

Sends the same scene at multiple sample counts and records wall-clock latency
plus the first rendered PNG so quality can be compared by eye.

Usage:
    NVCF_API_KEY=... python scripts/ovrtx_stg_num_sensor_updates_ab.py \
        --usd apps/ovrtx_rendering_api/tests/renders/smoke_cube.usda \
        --samples 1,4,16,25,64 \
        --out-dir /tmp/ovrtx_ab

The default --base-url points at the ovrtx-stg NVCF function; override with
--base-url to hit prd or a local docker container.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import pathlib
import time
from typing import Any

import requests

# ovrtx-stg NVCF function ID (see .gitlab-ci.yml OVRTX_RENDERING_API_STG_NVCF_ID).
STG_BASE_URL = (
    "https://babc4c06-dad7-460b-acba-2076d1b52c81.invocation.api.nvcf.nvidia.com"
)


def _encode_usd_as_data_uri(path: pathlib.Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:application/octet-stream;base64,{encoded}"


def _request_body(
    usd_data_uri: str,
    num_sensor_updates: int,
    width: int,
    height: int,
    camera: str,
) -> dict[str, Any]:
    return {
        "url": usd_data_uri,
        "force_render": True,
        "render_settings": {
            "camera_paths": [camera],
            "frame_range": {"start": 0, "end": 0},
            "camera_parameters": {"width": width, "height": height},
            "sensors": None,
            "apply_background_mask": False,
            "num_sensor_updates": num_sensor_updates,
        },
    }


def _render(
    base_url: str,
    token: str,
    body: dict[str, Any],
    timeout: float,
) -> tuple[float, dict[str, Any]]:
    url = f"{base_url.rstrip('/')}/render"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    t0 = time.perf_counter()
    r = requests.post(url, data=json.dumps(body), headers=headers, timeout=timeout)
    elapsed = time.perf_counter() - t0
    r.raise_for_status()
    return elapsed, r.json()


def _decode_first_png(resp: dict[str, Any]) -> bytes | None:
    """Pull the first base64 PNG out of the nested images dict, if any."""
    images = resp.get("images") or {}
    for frame in images.values():
        for cam in frame.values():
            for payload in cam.values():
                # Service returns either "data:image/png;base64,..." or bare base64.
                if isinstance(payload, str):
                    if payload.startswith("data:"):
                        payload = payload.split(",", 1)[1]
                    return base64.b64decode(payload)
    return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", default=STG_BASE_URL)
    p.add_argument(
        "--usd",
        type=pathlib.Path,
        default=pathlib.Path(__file__).parent.parent
        / "apps/ovrtx_rendering_api/tests/renders/smoke_cube.usda",
    )
    p.add_argument(
        "--samples",
        default="1,4,16,25,64",
        help="Comma-separated num_sensor_updates values to sweep",
    )
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--camera", default="/World/Camera")
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=pathlib.Path("/tmp/ovrtx_ab"),
        help="Directory for per-sample PNG + a summary.json",
    )
    args = p.parse_args()

    token = os.environ.get("NVCF_API_KEY") or os.environ.get("NVIDIA_API_KEY")
    if not token:
        raise SystemExit("Set NVCF_API_KEY (or NVIDIA_API_KEY) in the environment")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    usd_uri = _encode_usd_as_data_uri(args.usd)
    sample_counts = [int(s) for s in args.samples.split(",") if s.strip()]

    summary: list[dict[str, Any]] = []
    for n in sample_counts:
        print(f"-> num_sensor_updates={n}")
        body = _request_body(usd_uri, n, args.width, args.height, args.camera)
        elapsed, resp = _render(args.base_url, token, body, args.timeout)
        png = _decode_first_png(resp)
        out_png = args.out_dir / f"num_sensor_updates_{n:03d}.png"
        if png is not None:
            out_png.write_bytes(png)
        status = resp.get("status", "unknown")
        error = resp.get("error")
        print(
            f"   status={status} elapsed={elapsed:.2f}s png={out_png if png else '-'}"
        )
        summary.append(
            {
                "num_sensor_updates": n,
                "status": status,
                "error": error,
                "elapsed_seconds": round(elapsed, 3),
                "png_bytes": len(png) if png else 0,
                "png_path": str(out_png) if png else None,
            }
        )

    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {args.out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
