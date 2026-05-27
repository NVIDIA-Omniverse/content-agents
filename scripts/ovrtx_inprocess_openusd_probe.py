# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Probe whether OVRTX and OpenUSD can coexist in one Python process.

This is an opt-in maintenance probe for issue #283. It intentionally runs each
case in a child process so a native crash or renderer hang does not take down
the caller.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

IMPORT_ORDERS = ("pxr_then_ovrtx", "ovrtx_then_pxr")


CONSTRUCTOR_SNIPPET = r"""
import os
import sys
import traceback
from importlib import metadata

order = os.environ["WU_OVRTX_INPROCESS_ORDER"]
run = os.environ["WU_OVRTX_INPROCESS_RUN"]
print(f"run={run} order={order}", flush=True)
print(f"python={sys.executable}", flush=True)

try:
    print(f"usd_core_dist={metadata.version('usd-core')}", flush=True)
    print(f"ovrtx_dist={metadata.version('ovrtx')}", flush=True)

    if order == "pxr_then_ovrtx":
        from pxr import Usd

        print(f"pxr_import=ok module={getattr(Usd, '__name__', '<module>')}", flush=True)
        import ovrtx

        print(f"ovrtx_import=ok version={getattr(ovrtx, '__version__', '<none>')}", flush=True)
    elif order == "ovrtx_then_pxr":
        import ovrtx

        print(f"ovrtx_import=ok version={getattr(ovrtx, '__version__', '<none>')}", flush=True)
        from pxr import Usd

        print(f"pxr_import=ok module={getattr(Usd, '__name__', '<module>')}", flush=True)
    else:
        raise RuntimeError(f"unknown import order: {order}")

    renderer = ovrtx.Renderer()
    print(f"renderer_constructed={type(renderer).__name__}", flush=True)
    del renderer
except BaseException as exc:
    print(f"probe_failed={type(exc).__name__}: {exc}", flush=True)
    traceback.print_exc()
    raise SystemExit(2)
"""


MINIMAL_RENDER_SNIPPET = r"""
import os
import json
import sys
import tempfile
import traceback
from importlib import metadata
from pathlib import Path

try:
    order = os.environ["WU_OVRTX_INPROCESS_ORDER"]
    if order == "pxr_then_ovrtx":
        from pxr import Usd
        import ovrtx
    elif order == "ovrtx_then_pxr":
        import ovrtx
        from pxr import Usd
    else:
        raise RuntimeError(f"unknown import order: {order}")

    import numpy as np
    from PIL import Image

    smoke_usd = Path(os.environ["WU_OVRTX_SMOKE_USD"])
    tmp_root = Path(os.environ.get("WU_OVRTX_PROBE_TMP", tempfile.gettempdir()))
    tmp_dir = Path(tempfile.mkdtemp(prefix="wu_ovrtx_inprocess_", dir=tmp_root))

    render_products = tmp_dir / "render_products.usda"
    render_products.write_text(
        '''#usda 1.0
(
)

def Scope "Render"
{
    def RenderProduct "Product_World_Camera" (
        prepend apiSchemas = ["OmniRtxSettingsCommonAdvancedAPI_1", "OmniRtxSettingsRtAdvancedAPI_1", "OmniRtxSettingsPtAdvancedAPI_1"]
    )
    {
        rel camera = </World/Camera>
        rel orderedVars = [</Render/Vars/LdrColor>]
        uniform int2 resolution = (64, 64)
        token omni:rtx:rendermode = "PathTracing"
    }

    def Scope "Vars"
    {
        def RenderVar "LdrColor"
        {
            uniform string sourceName = "LdrColor"
        }
    }
}
''',
        encoding="utf-8",
    )

    def asset_path(path):
        return "@" + str(path).replace("\\", "/").replace("@", "%40") + "@"

    combined = tmp_dir / "combined.usda"
    combined.write_text(
        "#usda 1.0\n"
        "(\n"
        "    subLayers = [\n"
        f"        {asset_path(smoke_usd)},\n"
        f"        {asset_path(render_products)}\n"
        "    ]\n"
        ")\n",
        encoding="utf-8",
    )

    print(f"usd_core_dist={metadata.version('usd-core')}", flush=True)
    print(f"ovrtx_dist={metadata.version('ovrtx')}", flush=True)
    print(f"ovrtx_version={getattr(ovrtx, '__version__', '<none>')}", flush=True)
    stage = Usd.Stage.Open(str(smoke_usd))
    stage_open = stage is not None
    print(f"stage_open={stage_open}", flush=True)
    if not stage_open:
        raise RuntimeError(f"smoke USD did not open: {smoke_usd}")
    print(f"combined_usd={combined}", flush=True)

    renderer = ovrtx.Renderer()
    renderer.open_usd(str(combined))
    renderer.update_from_usd_time(0.0)
    renderer.reset()

    products = None
    product_paths = {"/Render/Product_World_Camera"}
    for step_idx in range(4):
        products = renderer.step(render_products=product_paths, delta_time=0.0)
        print(f"step={step_idx + 1} products={bool(products)}", flush=True)

    if not products:
        raise RuntimeError("renderer returned no products")

    image_stats = None
    for _name, product in products.items():
        for product_frame in product.frames:
            if "LdrColor" not in product_frame.render_vars:
                continue
            with product_frame.render_vars["LdrColor"].map(device=ovrtx.Device.CPU) as var:
                pixels = np.from_dlpack(var).copy()
            image_path = tmp_dir / "minimal.png"
            Image.fromarray(pixels).save(image_path)
            rgb = pixels[..., :3] if pixels.ndim == 3 and pixels.shape[-1] >= 3 else pixels
            image_stats = {
                "path": str(image_path),
                "shape": list(pixels.shape),
                "max": int(pixels.max()),
                "nonzero": int((pixels > 0).sum()),
                "max_rgb": int(rgb.max()),
                "nonzero_rgb": int((rgb > 0).sum()),
            }
            break
        if image_stats is not None:
            break

    del products
    del renderer
    if image_stats is None:
        raise RuntimeError("renderer returned no LdrColor")
    if image_stats["max_rgb"] <= 0 or image_stats["nonzero_rgb"] <= 0:
        raise RuntimeError("rendered buffer has no visible/nonzero RGB content")
    print("minimal_render=" + json.dumps(image_stats, sort_keys=True), flush=True)
except Exception as exc:
    print(f"render_probe_failed={type(exc).__name__}: {exc}", flush=True)
    traceback.print_exc()
    raise SystemExit(2)
"""


def _run_snippet(
    python: Path,
    snippet: str,
    *,
    env: dict[str, str],
    timeout_s: float,
) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            [str(python), "-u", "-c", snippet],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        return {
            "returncode": None,
            "timed_out": True,
            "launch_failed": False,
            "timeout_s": timeout_s,
            "stdout": stdout,
            "stderr_tail": stderr[-4000:],
        }
    except OSError as exc:
        return {
            "returncode": None,
            "timed_out": False,
            "launch_failed": True,
            "timeout_s": timeout_s,
            "stdout": "",
            "stderr_tail": str(exc)[-4000:],
        }

    return {
        "returncode": proc.returncode,
        "timed_out": False,
        "launch_failed": False,
        "timeout_s": timeout_s,
        "stdout": proc.stdout,
        "stderr_tail": proc.stderr[-4000:],
    }


def _gpu_summary() -> str:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"nvidia-smi unavailable: {exc}"
    if proc.returncode != 0:
        return f"nvidia-smi failed: {proc.stderr.strip()[-300:]}"
    return proc.stdout.strip()


def _git_rev(repo_root: Path, rev: str) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", rev],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def _ok(result: dict[str, Any]) -> bool:
    return (
        not result["timed_out"]
        and not result.get("launch_failed", False)
        and result["returncode"] == 0
        and "validation_error" not in result
    )


def _minimal_render_validation_error(result: dict[str, Any]) -> str | None:
    """Return a validation error for false-positive minimal render outputs."""
    if result["returncode"] != 0 or result["timed_out"] or result.get("launch_failed"):
        return None

    stdout = str(result.get("stdout", ""))
    if "stage_open=False" in stdout:
        return "smoke USD did not open"

    for line in reversed(stdout.splitlines()):
        if not line.startswith("minimal_render="):
            continue
        try:
            stats = json.loads(line.removeprefix("minimal_render="))
        except json.JSONDecodeError as exc:
            return f"minimal render stats are not valid JSON: {exc}"
        if int(stats.get("max_rgb", 0)) <= 0 or int(stats.get("nonzero_rgb", 0)) <= 0:
            return "minimal render RGB output is blank"
        return None

    return "minimal render stats are missing"


def run_probe(
    *,
    python: Path,
    repo_root: Path,
    runs: int,
    constructor_timeout_s: float,
    minimal_render: bool,
    render_timeout_s: float,
    smoke_usd: Path | None = None,
    tmp_dir: Path | None = None,
) -> tuple[dict[str, Any], bool]:
    if runs < 1:
        raise ValueError("runs must be at least 1")

    env_base = os.environ.copy()
    if tmp_dir is not None:
        env_base["WU_OVRTX_PROBE_TMP"] = str(tmp_dir)

    results = []
    constructor_success = True
    for run_idx in range(1, runs + 1):
        for order in IMPORT_ORDERS:
            env = env_base.copy()
            env["WU_OVRTX_INPROCESS_RUN"] = str(run_idx)
            env["WU_OVRTX_INPROCESS_ORDER"] = order
            result = _run_snippet(
                python,
                CONSTRUCTOR_SNIPPET,
                env=env,
                timeout_s=constructor_timeout_s,
            )
            result.update({"run": run_idx, "order": order, "case": "constructor"})
            constructor_success = constructor_success and _ok(result)
            results.append(result)

    render_results: list[dict[str, Any]] = []
    minimal_render_success: bool | None = None
    if minimal_render:
        smoke = smoke_usd or (
            repo_root
            / "apps"
            / "ovrtx_rendering_api"
            / "tests"
            / "renders"
            / "smoke_cube.usda"
        )
        minimal_render_success = True
        for order in IMPORT_ORDERS:
            env = env_base.copy()
            env["WU_OVRTX_INPROCESS_ORDER"] = order
            env["WU_OVRTX_SMOKE_USD"] = str(smoke)
            render_result = _run_snippet(
                python,
                MINIMAL_RENDER_SNIPPET,
                env=env,
                timeout_s=render_timeout_s,
            )
            render_result.update(
                {"case": "minimal_render", "order": order, "smoke_usd": str(smoke)}
            )
            validation_error = _minimal_render_validation_error(render_result)
            if validation_error is not None:
                render_result["validation_error"] = validation_error
            minimal_render_success = minimal_render_success and _ok(render_result)
            render_results.append(render_result)

    success = constructor_success and (
        True if minimal_render_success is None else minimal_render_success
    )

    report = {
        "probe": "ovrtx_inprocess_openusd",
        "python": str(python),
        "platform": platform.platform(),
        "gpu": _gpu_summary(),
        "repo_head": _git_rev(repo_root, "HEAD"),
        "runs": runs,
        "import_orders": list(IMPORT_ORDERS),
        "constructor_timeout_s": constructor_timeout_s,
        "render_timeout_s": render_timeout_s,
        "constructor_results": results,
        "minimal_render_results": render_results,
        "phase_success": {
            "constructor": constructor_success,
            "minimal_render": minimal_render_success,
        },
        "decision_hint": (
            "Do not remove the isolated OVRTX venv/subprocess path unless "
            "this succeeds on a supported Linux GPU/service host and "
            "regression tests cover service startup and cached runtime "
            "provisioning."
        ),
    }
    return report, success


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(sys.executable),
        help="Python executable containing both ovrtx and a pxr/OpenUSD provider",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=repo_root,
        help="world-understanding repository root",
    )
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--constructor-timeout", type=float, default=120.0)
    parser.add_argument("--minimal-render", action="store_true")
    parser.add_argument("--render-timeout", type=float, default=240.0)
    parser.add_argument("--smoke-usd", type=Path, default=None)
    parser.add_argument("--tmp-dir", type=Path, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report, success = run_probe(
        python=args.python,
        repo_root=args.repo_root,
        runs=args.runs,
        constructor_timeout_s=args.constructor_timeout,
        minimal_render=args.minimal_render,
        render_timeout_s=args.render_timeout,
        smoke_usd=args.smoke_usd,
        tmp_dir=args.tmp_dir,
    )
    output = json.dumps(report, indent=2)
    if args.json_out is not None:
        args.json_out.write_text(output + "\n", encoding="utf-8")
    print(output)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
