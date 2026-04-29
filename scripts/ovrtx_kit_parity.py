# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render the same scene on OVRTX + Kit NVCF and compare quality.

Long-term goal: replace the Kit-based rendering-api with OVRTX. This script
produces the evidence for that claim — render identical input on both, sweep
OVRTX ``num_sensor_updates``, and report per-sample quality vs. the Kit reference.

Kit and OVRTX return different wire formats:

* Kit  : ``{total_cameras, total_frames, rendered_data:
         {<cam_basename>: {<frame_str>: {<sensor>: {type: "array",
         data: "<base64 raw RGBA bytes>"}}}}}``
* OVRTX: ``{status, error, images: {<frame_str>: {<full_cam_path>:
         {"images": "<base64 PNG>"}}}}`` (V1 format invented for the OVRTX
         service; see ``apps/ovrtx_rendering_api/service/renderer.py:
         _to_v1_response``).

Both collapse to an HxWx3 uint8 array for comparison. Metrics are the
straight-forward ones that do not need extra deps beyond Pillow + numpy:

* ``l1_rgb``  : mean |Δ|, 0-255 scale. Lower is closer to Kit.
* ``psnr_db`` : 10 log10(255² / MSE), dB. Higher is closer to Kit.
* ``bg_std``  : std-dev across the top-left 32x32 patch (a denoising proxy).
                Drops as sample count rises.

Typical use:

    NVCF_API_KEY=... python scripts/ovrtx_kit_parity.py \\
        --usd /home/horde/dev/kit-gen-ai-service/kit/source/extensions/omni.gen_ai.core/data/scene.usd \\
        --camera /Cameras/CornerViewCamera_posx_posy_posz \\
        --samples 1,4,16,25,64,128
    # (omit --out-dir to write to a fresh tmp dir printed at the end)

Outputs per sample count + a ``summary.json`` + a short Markdown table.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import math
import os
import pathlib
import tempfile
import time
import zipfile
from typing import Any

import numpy as np
import requests
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

KIT_PROD_FUNCTION_ID = "a9f09b24-59d1-4600-b1e1-cc0f21526dbc"
OVRTX_STG_FUNCTION_ID = "babc4c06-dad7-460b-acba-2076d1b52c81"
OVRTX_DEFAULT_LOCAL_URL = "http://localhost:8001"

# Files the USD stage may reference at runtime. Anything sitting next to
# the root USD layer with one of these suffixes is bundled into the ZIP
# we upload so both renderers can resolve relative paths.
_ASSET_SUFFIXES = {
    ".usd",
    ".usda",
    ".usdc",
    ".exr",
    ".hdr",
    ".png",
    ".jpg",
    ".jpeg",
    ".mdl",
    ".tga",
    ".tif",
    ".tiff",
}


def _bundle_or_raw(usd_path: pathlib.Path) -> tuple[bytes, str, str]:
    """Return (payload_bytes, data_uri_mime, nice_label).

    If the USD lives alone in its directory, send the raw bytes. Otherwise
    ZIP the directory (recognized asset suffixes only) so relative refs
    resolve on both renderers.
    """
    parent = usd_path.parent
    siblings = [
        p
        for p in parent.iterdir()
        if p.is_file() and p != usd_path and p.suffix.lower() in _ASSET_SUFFIXES
    ]
    if not siblings:
        return usd_path.read_bytes(), "model/vnd.usda", f"raw {usd_path.name}"

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # Name the root layer identically in the archive so
        # `main.*/scene.*/stage.*` priority picks the one we want. We use
        # the file's own basename — both renderers will find any USD layer.
        zf.write(usd_path, usd_path.name)
        for sib in siblings:
            zf.write(sib, sib.name)
    return (
        buf.getvalue(),
        "application/zip",
        f"ZIP({usd_path.name} + {len(siblings)} sibling asset(s))",
    )


def _data_uri(payload: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(payload).decode('ascii')}"


def _post(
    url: str, token: str, body: dict[str, Any], timeout: float = 300
) -> tuple[float, dict[str, Any]]:
    t0 = time.perf_counter()
    r = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        json=body,
        timeout=timeout,
    )
    elapsed = time.perf_counter() - t0
    r.raise_for_status()
    return elapsed, r.json()


def _kit_decode(resp: dict[str, Any], width: int, height: int) -> np.ndarray:
    """Extract the first RGB frame from a Kit /render response as HxWx3 uint8."""
    rd = resp.get("rendered_data") or {}
    if not rd:
        raise RuntimeError(f"Kit response missing rendered_data: {resp!r}")
    cam_name = next(iter(rd))
    frame = next(iter(rd[cam_name]))
    sensor_obj = rd[cam_name][frame].get("rgb") or next(
        iter(rd[cam_name][frame].values())
    )
    raw = base64.b64decode(sensor_obj["data"])
    # Kit returns raw RGBA8: H*W*4 bytes. Some builds emit RGB24 — detect.
    if len(raw) == width * height * 4:
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 4)[:, :, :3]
    elif len(raw) == width * height * 3:
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
    else:
        raise RuntimeError(
            f"Unexpected Kit raw payload size: {len(raw)} bytes for {width}x{height}"
        )
    return arr.copy()  # copy so we can free `raw`


def _ovrtx_decode(resp: dict[str, Any]) -> np.ndarray:
    """Extract the first RGB frame from an OVRTX /render response as HxWx3 uint8."""
    images = resp.get("images") or {}
    if not images:
        raise RuntimeError(
            f"OVRTX response missing images: status={resp.get('status')!r} error={resp.get('error')!r}"
        )
    frame = next(iter(images.values()))
    cam_data = next(iter(frame.values()))
    # V1 format keys the PNG under "images"; tolerate alternate keys.
    png_b64 = (
        cam_data.get("images") or cam_data.get("rgb") or next(iter(cam_data.values()))
    )
    if png_b64.startswith("data:"):
        png_b64 = png_b64.split(",", 1)[1]
    img = Image.open(io.BytesIO(base64.b64decode(png_b64))).convert("RGB")
    return np.asarray(img)


def _metrics(ref: np.ndarray, test: np.ndarray) -> dict[str, float]:
    """Return {l1_rgb, psnr_db, bg_std} comparing test vs reference."""
    if ref.shape != test.shape:
        raise RuntimeError(f"shape mismatch: ref={ref.shape} test={test.shape}")
    diff = ref.astype(np.int32) - test.astype(np.int32)
    l1 = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff.astype(np.float64) ** 2))
    psnr = float("inf") if mse == 0 else 10 * math.log10(255.0 * 255.0 / mse)
    # Top-left 32x32 patch as a background denoising proxy.
    patch = test[:32, :32, :].astype(np.float64)
    bg_std = float(np.std(patch))
    return {
        "l1_rgb": round(l1, 3),
        "psnr_db": round(psnr, 3),
        "bg_std": round(bg_std, 3),
    }


def _render_settings(
    camera: str, width: int, height: int, num_sensor_updates: int | None
) -> dict[str, Any]:
    rs: dict[str, Any] = {
        "camera_paths": [camera],
        "frame_range": {"start": 0, "end": 0},
        "camera_parameters": {"width": width, "height": height},
        "sensors": None,
        "apply_background_mask": False,
    }
    if num_sensor_updates is not None:
        rs["num_sensor_updates"] = num_sensor_updates
    return rs


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--usd", type=pathlib.Path, required=True, help="Path to the root USD layer"
    )
    p.add_argument(
        "--camera", required=True, help="Full camera prim path, e.g. /Cameras/Foo"
    )
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=512)
    p.add_argument(
        "--samples",
        default="1,4,16,25,64,128",
        help="Comma-separated OVRTX num_sensor_updates values to sweep",
    )
    p.add_argument(
        "--ovrtx-base-url",
        default=OVRTX_DEFAULT_LOCAL_URL,
        help=(
            f"OVRTX service base URL "
            f"(default: {OVRTX_DEFAULT_LOCAL_URL} — a local docker container)"
        ),
    )
    p.add_argument(
        "--kit-function-id",
        default=KIT_PROD_FUNCTION_ID,
        help="NVCF function ID for the Kit rendering-api reference (default: prod)",
    )
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=None,
        help=(
            "Output directory for sweep PNGs + summary (default: a fresh "
            "tempfile.mkdtemp(prefix='ovrtx_kit_parity_') created per run)"
        ),
    )
    args = p.parse_args()
    if args.out_dir is None:
        args.out_dir = pathlib.Path(tempfile.mkdtemp(prefix="ovrtx_kit_parity_"))

    token = os.environ.get("NVCF_API_KEY") or os.environ.get("NVIDIA_API_KEY")
    if not token:
        raise SystemExit("Set NVCF_API_KEY (or NVIDIA_API_KEY) for the Kit NVCF call")

    if not args.usd.exists():
        raise SystemExit(f"USD not found: {args.usd}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sample_counts = [int(s) for s in args.samples.split(",") if s.strip()]

    # Bundle once; reuse for both renderers.
    payload, mime, label = _bundle_or_raw(args.usd)
    uri = _data_uri(payload, mime)
    logger.info("Input: %s  -> %s  (%s bytes)", args.usd, label, f"{len(payload):,}")
    logger.info("Camera: %s  @ %sx%s", args.camera, args.width, args.height)

    # ---- Kit reference ----
    kit_url = f"https://{args.kit_function_id}.invocation.api.nvcf.nvidia.com/render"
    logger.info("\n[Kit] POST %s", kit_url)
    # Kit ignores num_sensor_updates (operator setting baked into its container), so
    # we send it without the field.
    kit_body = {
        "url": uri,
        "force_render": True,
        "render_settings": _render_settings(
            args.camera, args.width, args.height, num_sensor_updates=None
        ),
    }
    kit_elapsed, kit_resp = _post(kit_url, token, kit_body, args.timeout)
    kit_rgb = _kit_decode(kit_resp, args.width, args.height)
    kit_png = args.out_dir / "kit_reference.png"
    Image.fromarray(kit_rgb, "RGB").save(kit_png)
    logger.info(
        "  elapsed=%.2fs  saved=%s  shape=%s", kit_elapsed, kit_png, kit_rgb.shape
    )

    # ---- OVRTX sweep ----
    ovrtx_url = f"{args.ovrtx_base_url.rstrip('/')}/render"
    logger.info("\n[OVRTX] POST %s", ovrtx_url)
    summary: list[dict[str, Any]] = []
    for n in sample_counts:
        body = {
            "url": uri,
            "force_render": True,
            "render_settings": _render_settings(
                args.camera, args.width, args.height, num_sensor_updates=n
            ),
        }
        try:
            elapsed, resp = _post(ovrtx_url, token, body, args.timeout)
        except requests.RequestException as exc:
            # Don't abort the whole sweep on a transient HTTP / timeout /
            # 5xx — record the failure and keep going so surrounding
            # data points still land in summary.json.
            logger.error(
                "  num_sensor_updates=%s: HTTP_ERROR %s: %s",
                n,
                type(exc).__name__,
                exc,
            )
            summary.append(
                {
                    "num_sensor_updates": n,
                    "status": "http_error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "elapsed_seconds": None,
                }
            )
            continue
        if resp.get("status") != "success":
            logger.error(
                "  num_sensor_updates=%s: FAILED status=%s err=%s",
                n,
                resp.get("status"),
                resp.get("error"),
            )
            summary.append(
                {
                    "num_sensor_updates": n,
                    "status": resp.get("status"),
                    "error": resp.get("error"),
                    "elapsed_seconds": round(elapsed, 3),
                }
            )
            continue
        rgb = _ovrtx_decode(resp)
        out_png = args.out_dir / f"ovrtx_n{n:03d}.png"
        Image.fromarray(rgb, "RGB").save(out_png)
        m = _metrics(kit_rgb, rgb)
        row = {
            "num_sensor_updates": n,
            "status": "success",
            "elapsed_seconds": round(elapsed, 3),
            "png": str(out_png),
            **m,
        }
        logger.info(
            "  num_sensor_updates=%4d: elapsed=%6.2fs  "
            "l1_rgb=%6.2f  psnr=%6.2fdB  bg_std=%6.2f",
            n,
            elapsed,
            m["l1_rgb"],
            m["psnr_db"],
            m["bg_std"],
        )
        summary.append(row)

    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Markdown table for the MR description.
    md_lines = [
        "# OVRTX ↔ Kit parity sweep",
        "",
        f"- Scene: `{args.usd}`",
        f"- Camera: `{args.camera}` @ {args.width}x{args.height}",
        f"- Kit function: `{args.kit_function_id}`",
        f"- OVRTX endpoint: `{args.ovrtx_base_url}`",
        "",
        "| num_sensor_updates | elapsed (s) | L1 RGB | PSNR (dB) | bg std |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        if row.get("status") == "success":
            md_lines.append(
                f"| {row['num_sensor_updates']} | {row['elapsed_seconds']} "
                f"| {row['l1_rgb']} | {row['psnr_db']} | {row['bg_std']} |"
            )
        else:
            md_lines.append(
                f"| {row['num_sensor_updates']} | {row['elapsed_seconds']} | FAIL | FAIL | FAIL |"
            )
    (args.out_dir / "summary.md").write_text("\n".join(md_lines) + "\n")

    logger.info("\nWrote %s", args.out_dir / "summary.json")
    logger.info("Wrote %s", args.out_dir / "summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
