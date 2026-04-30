# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""USD rendering functions using OvRTX local RTX renderer.

This module provides rendering functions that use the ovrtx library for
local, in-process RTX rendering. It combines the quality of RTX rendering
with the low latency of local execution (no cloud overhead).

Because ovrtx bundles its own USD C libraries which conflict with the pxr
(usd-core) package at the shared-library level, all ovrtx work runs in an
isolated subprocess using a separate virtual environment that has ovrtx
installed *without* usd-core. The main process exports the stage to a temp
file and the subprocess does the actual rendering.

Requires: ovrtx >= 0.1.0
"""

import atexit
import json
import logging
import os
import selectors
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from pxr import Usd

logger = logging.getLogger(__name__)

# Mapping from WU sensor names to ovrtx render variable names
_SENSOR_TO_RENDER_VAR: dict[str, str] = {
    "depth": "Depth",
    "normal": "Normal",
    "albedo": "Albedo",
}

# Default location for the auto-provisioned ovrtx venv.
# Honour WU_OVRTX_VENV_DIR env var so Docker images can ship a pre-built venv.
_OVRTX_VENV_DIR = Path(
    os.environ.get(
        "WU_OVRTX_VENV_DIR", str(Path.home() / ".cache" / "wu" / "ovrtx_venv")
    )
)

# Cached path to the ovrtx venv Python executable
_ovrtx_python: str | None = None

# Number of ``renderer.step(delta_time=0)`` iterations per frame. OVRtx's
# path tracer accumulates samples across successive step() calls when
# ``delta_time`` is zero — this is the only quality knob that actually
# has effect in ovrtx 0.2.0. Empirically (see the ovrtx_kit_parity.py
# convergence-cap sweep on the kit-gen-ai-service golden scene), PT mode
# plateaus at ~500 steps / ~39.7 dB PSNR vs the Kit reference. Anything
# past 500 is wasted work; anything below ~100 is visibly noisy.
#
# The field is named ``num_sensor_updates`` on the wire for historical reasons
# (and Kit's rendering-api uses ``num_sensor_updates`` for the same
# concept — outer update loop) but is semantically *iteration count*,
# not samples-per-pixel. The schema-level ``omni:rtx:pt:samplesPerPixel``
# attribute is silently ignored by ovrtx 0.2.0.
DEFAULT_NUM_SENSOR_UPDATES = 500

# Default RTX render mode. ``pt`` maps to ``PathTracing`` — Kit's
# ground-truth mode. rt2 is available as an override for callers that
# want real-time-path-tracing speed, but pt is the quality-parity target.
DEFAULT_RENDER_MODE = "pt"

# Map the short mode tokens accepted on the wire (kit-gen-ai-service's
# ``RenderMode`` enum values) to the ``omni:rtx:rendermode`` token the
# RTX engine expects.
_RENDER_MODE_TOKENS: dict[str, str] = {
    "rt1": "RaytracedLighting",
    "rt2": "RealTimePathTracing",
    "pt": "PathTracing",
}

# RTX USD api schemas. Applying these on the RenderProduct is required
# so OVRtx's ``omni:rtx:rendermode`` attribute is honored — without the
# schema prepend, the RenderProduct is treated as a bare RenderProduct
# and the engine falls back to its internal default (RaytracedLighting,
# which isn't even a supported token at runtime, and gets remapped to
# RealTimePathTracing).
#
# The schema names come from ovrtx's bundled rtx_settings plugin —
# ``.ovrtx_venv/.../ovrtx/bin/usd_plugins/rtx_settings/generatedSchema.usda``.
_RTX_RENDER_PRODUCT_API_SCHEMAS = (
    "OmniRtxSettingsCommonAdvancedAPI_1",
    "OmniRtxSettingsRtAdvancedAPI_1",
    "OmniRtxSettingsPtAdvancedAPI_1",
)


def _parse_frames(frames: str) -> list[int]:
    """Parse a frames string into a list of integer frame numbers.

    Supports three formats:
    - Single frame: "0", "42"
    - Frame range: "0:10" (inclusive, produces [0, 1, ..., 10])
    - Comma-separated: "0,5,10"

    Args:
        frames: Frame specification string.

    Returns:
        Sorted list of integer frame numbers.

    Raises:
        ValueError: If the frames string cannot be parsed.

    Examples:
        >>> _parse_frames("0")
        [0]
        >>> _parse_frames("0:3")
        [0, 1, 2, 3]
        >>> _parse_frames("0,5,10")
        [0, 5, 10]
    """
    frames = frames.strip()

    if ":" in frames:
        parts = frames.split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid frame range: '{frames}'. Expected 'start:end'.")
        start = int(parts[0])
        end = int(parts[1])
        return list(range(start, end + 1))
    elif "," in frames:
        return sorted(int(f.strip()) for f in frames.split(",") if f.strip())
    else:
        return [int(frames)]


def _build_visibility_frame_updates(
    visibility_schedule: dict[str, dict[str, str]],
    frames: list[int],
) -> dict[str, dict[str, str]]:
    """Collapse full-frame visibility samples to per-frame deltas.

    ``render_all_cameras`` strips time-sampled visibility from the exported USD
    because OVRTX 0.2.0 can crash on those authored samples. The stripped
    export leaves affected prims visible by default, so the daemon only needs
    to write values that differ from the currently applied visibility state.
    This avoids hammering the native renderer with thousands of redundant
    ``write_attribute`` calls for every rendered frame.
    """
    if not visibility_schedule:
        return {}

    current_visibility: dict[str, str] = {}
    frame_updates: dict[str, dict[str, str]] = {}

    for frame_num in frames:
        frame_key = str(float(frame_num))
        vis_map = visibility_schedule.get(frame_key)
        if not vis_map:
            continue

        changes: dict[str, str] = {}
        for prim_path, vis_value in vis_map.items():
            token = "inherited" if vis_value == "inherited" else "invisible"
            if current_visibility.get(prim_path, "inherited") == token:
                continue
            current_visibility[prim_path] = token
            changes[prim_path] = token

        if changes:
            frame_updates[frame_key] = changes

    return frame_updates


def _build_render_products_usda(
    cameras: list[str],
    image_width: int,
    image_height: int,
    sensors: list[str] | None = None,
    render_mode: str = DEFAULT_RENDER_MODE,
) -> tuple[str, list[str]]:
    """Generate USDA content defining RenderProduct prims for each camera.

    Each RenderProduct references a camera, specifies the render resolution,
    and carries the ``omni:rtx:rendermode`` USD attribute so OVRtx's
    ``step()`` picks the right RTX mode. The api schema prepend is the
    tripwire that unlocks the attribute on the RenderProduct.

    Known ovrtx 0.2.0 limitation: the schema-level attributes
    ``omni:rtx:pt:samplesPerPixel`` and ``omni:rtx:rt:accumulationLimit``
    are defined in the bundled rtx_settings plugin but are silently
    ignored by the render path (verified by the step-cost + noise-proxy
    timing test in /tmp/ovrtx_verify.py — extreme values produce
    bitwise-identical timings and noise). The only quality knob that
    actually has effect is the number of ``renderer.step(delta_time=0)``
    iterations in the daemon. We therefore do not emit those attributes
    here — keeping the USDA to the two things that do matter: api
    schemas + rendermode.

    Args:
        cameras: List of camera prim paths (e.g., ["/Cameras/Camera1"]).
        image_width: Render width in pixels.
        image_height: Render height in pixels.
        sensors: Optional list of sensor names to include (e.g., ["depth"]).
        render_mode: ``rt1``/``rt2``/``pt`` short token — translated to
            ``omni:rtx:rendermode`` (RaytracedLighting / RealTimePathTracing /
            PathTracing) on the RenderProduct.

    Returns:
        Tuple of (usda_string, product_paths):
            - usda_string: USDA layer content to sublayer into the stage
            - product_paths: List of RenderProduct prim paths for step()

    Raises:
        ValueError: If ``render_mode`` is not one of ``rt1``/``rt2``/``pt``.
    """
    if render_mode not in _RENDER_MODE_TOKENS:
        raise ValueError(
            f"Unknown render_mode: {render_mode!r}. "
            f"Expected one of {sorted(_RENDER_MODE_TOKENS)}."
        )
    rendermode_token = _RENDER_MODE_TOKENS[render_mode]
    api_schema_list = ", ".join(f'"{s}"' for s in _RTX_RENDER_PRODUCT_API_SCHEMAS)
    product_paths = []
    render_var_defs = []
    render_var_refs = []

    # Always include LdrColor (matches ovrtx reference format)
    render_var_defs.append(
        '        def RenderVar "LdrColor"\n'
        "        {\n"
        '            uniform string sourceName = "LdrColor"\n'
        "        }\n"
    )
    render_var_refs.append("</Render/Vars/LdrColor>")

    # Add sensor render vars
    for sensor_name in sensors or []:
        render_var_name = _map_sensor_to_render_var(sensor_name)
        if render_var_name is None:
            logger.warning(
                "Unknown sensor '%s', skipping render var generation", sensor_name
            )
            continue

        render_var_defs.append(
            f'        def RenderVar "{render_var_name}"\n'
            f"        {{\n"
            f'            uniform string sourceName = "{render_var_name}"\n'
            f"        }}\n"
        )
        render_var_refs.append(f"</Render/Vars/{render_var_name}>")

    # Build render var references string
    ordered_vars = ", ".join(render_var_refs)

    # Build product definitions. Only the api schema prepend and
    # ``omni:rtx:rendermode`` token actually influence rendering in
    # ovrtx 0.2.0 — the sample/accumulation-limit attributes defined in
    # the bundled schema are silently ignored (see the docstring above).
    product_defs = []
    for camera_path in cameras:
        # Sanitize camera path for use as prim name
        safe_name = camera_path.strip("/").replace("/", "_")
        product_prim_name = f"Product_{safe_name}"
        product_path = f"/Render/{product_prim_name}"
        product_paths.append(product_path)

        product_defs.append(
            f'    def RenderProduct "{product_prim_name}" (\n'
            f"        prepend apiSchemas = [{api_schema_list}]\n"
            f"    )\n"
            f"    {{\n"
            f"        rel camera = <{camera_path}>\n"
            f"        rel orderedVars = [{ordered_vars}]\n"
            f"        uniform int2 resolution = ({image_width}, {image_height})\n"
            f'        token omni:rtx:rendermode = "{rendermode_token}"\n'
            f"    }}\n"
        )

    # Assemble full USDA (matching ovrtx reference structure)
    products_block = "\n".join(product_defs)
    vars_block = "\n".join(render_var_defs)

    usda = (
        "#usda 1.0\n"
        "(\n"
        ")\n"
        "\n"
        'def Scope "Render"\n'
        "{\n"
        f"{products_block}\n"
        "\n"
        '    def Scope "Vars"\n'
        "    {\n"
        f"{vars_block}"
        "    }\n"
        "}\n"
    )

    return usda, product_paths


def _map_sensor_to_render_var(sensor_name: str) -> str | None:
    """Map a WU sensor name to an ovrtx render variable name.

    Args:
        sensor_name: WU sensor name (e.g., "depth", "normal").

    Returns:
        OvRTX render variable name, or None if no mapping exists.

    Examples:
        >>> _map_sensor_to_render_var("depth")
        'Depth'
        >>> _map_sensor_to_render_var("unknown")
    """
    return _SENSOR_TO_RENDER_VAR.get(sensor_name)


# ---------------------------------------------------------------------------
# Isolated ovrtx venv management
# ---------------------------------------------------------------------------

# Pre-built ovrtx from PyPI (includes native libovrtx-dynamic.so).
# 0.2.0.280040 is the build with the native library bundled.
_OVRTX_PACKAGE = "ovrtx==0.2.0.280040"


def _get_ovrtx_python(venv_dir: Path | None = None) -> str:
    """Return the path to the Python executable in the ovrtx venv.

    If the venv does not exist, it is created and ovrtx + dependencies
    are installed into it. The venv intentionally does NOT have usd-core
    to avoid native library conflicts.

    Args:
        venv_dir: Override directory for the venv. Defaults to
            ``~/.cache/wu/ovrtx_venv``.

    Returns:
        Absolute path to the venv's python executable.

    Raises:
        RuntimeError: If venv creation or package installation fails.
    """
    global _ovrtx_python
    if _ovrtx_python is not None and os.path.exists(_ovrtx_python):
        return _ovrtx_python

    venv_dir = venv_dir or _OVRTX_VENV_DIR
    python_path = venv_dir / "bin" / "python"

    if python_path.exists():
        # Quick smoke-test: can ovrtx be imported?
        try:
            probe = subprocess.run(
                [str(python_path), "-c", "import ovrtx"],
                capture_output=True,
                timeout=30,
                check=False,
            )
            if probe.returncode == 0:
                _ovrtx_python = str(python_path)
                return _ovrtx_python
        except subprocess.TimeoutExpired:
            logger.warning("ovrtx import probe timed out, recreating venv")

        # Venv exists but ovrtx is broken — recreate
        logger.warning("Existing ovrtx venv broken, recreating: %s", venv_dir)
        shutil.rmtree(venv_dir, ignore_errors=True)

    # Create the venv
    logger.info("Creating isolated ovrtx venv at %s", venv_dir)
    venv_dir.mkdir(parents=True, exist_ok=True)

    # Try uv first, fall back to stdlib venv.
    # shutil.which may miss uv inside venvs, so also check next to sys.executable.
    uv_bin = shutil.which("uv")
    if uv_bin is None:
        _candidate = Path(sys.executable).parent / "uv"
        if _candidate.exists():
            uv_bin = str(_candidate)
    if uv_bin:
        _run_checked(
            [
                uv_bin,
                "venv",
                str(venv_dir),
                "--python",
                f"{sys.version_info.major}.{sys.version_info.minor}",
            ],
            "uv venv creation",
        )
        _run_checked(
            [
                uv_bin,
                "pip",
                "install",
                "--python",
                str(python_path),
                _OVRTX_PACKAGE,
                "numpy",
                "pillow",
            ],
            "uv pip install ovrtx",
        )
    else:
        # Fallback: install packages into a target directory using the
        # current Python's pip.  This avoids venv creation entirely which
        # is fragile on minimal cloud images (missing ensurepip, mismatched
        # libpython, etc.).  We create a thin wrapper script so the rest of
        # the code can still invoke ``venv_dir / "bin" / "python"``.
        logger.info("uv not found, using pip --target fallback")
        site_dir = venv_dir / "lib" / "python" / "site-packages"
        site_dir.mkdir(parents=True, exist_ok=True)

        _run_checked(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--target",
                str(site_dir),
                _OVRTX_PACKAGE,
                "numpy",
                "pillow",
            ],
            "pip install --target ovrtx",
        )

        # Create a wrapper so venv_dir/bin/python launches the system
        # Python with PYTHONPATH pointing at the --target directory.
        bin_dir = venv_dir / "bin"
        bin_dir.mkdir(parents=True, exist_ok=True)
        wrapper = bin_dir / "python"
        real_python = shutil.which("python3") or shutil.which("python") or "python3"
        wrapper.write_text(f'#!/bin/sh\nexec {real_python} "$@"\n')
        wrapper.chmod(0o755)

        # Record site_dir so the subprocess scripts can prepend it to
        # sys.path.  The daemon/worker scripts already run inside the
        # ovrtx venv python, so this env var is only needed for the
        # --target fallback.
        os.environ["_WU_OVRTX_SITE_DIR"] = str(site_dir)

    if not python_path.exists():
        raise RuntimeError(f"Failed to create ovrtx venv at {venv_dir}")

    # Symlink MaterialX standard data libraries so MaterialX shaders
    # (ND_tiledimage, OpenPBR, etc.) resolve correctly.  ovrtx ships
    # them under ovrtx/bin/library/ but looks for them at library/.
    _site_pkgs = venv_dir / "lib"
    for d in _site_pkgs.rglob("ovrtx/bin/library"):
        expected = d.parent.parent.parent / "library"
        if not expected.exists():
            expected.symlink_to(d)
            logger.info("Created MaterialX library symlink: %s -> %s", expected, d)
        break

    _ovrtx_python = str(python_path)
    logger.info("OvRTX venv ready: %s", _ovrtx_python)
    return _ovrtx_python


def _run_checked(cmd: list[str], label: str) -> None:
    """Run a command and raise RuntimeError on failure."""
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"{label} failed (exit {result.returncode}): {result.stderr[-500:]}"
        )


def _copy_exported_relative_assets(stage: "Usd.Stage", export_dir: Path) -> int:
    """Copy local relative texture assets next to an exported render stage.

    ``render_all_cameras`` exports the caller's stage into an OVRTX IPC temp
    directory before the isolated daemon opens it. Relative texture paths in
    that exported layer are interpreted relative to the new temp directory, not
    the original stage directory, so extracted ZIP/USDZ bundles lose their
    textures unless we mirror those files here.
    """
    from world_understanding.utils.usd.material import get_local_texture_file_assets

    root_layer = stage.GetRootLayer()
    if root_layer.realPath:
        base_dir = Path(root_layer.realPath).parent
    else:
        base_dir = Path.cwd()

    copied = 0
    for asset in get_local_texture_file_assets(stage, base_dir=base_dir):
        if not asset.get("is_local") or not asset.get("resolved_path"):
            continue

        asset_path = str(asset.get("file_path", ""))
        if not asset_path or asset_path.startswith(("http://", "https://")):
            continue

        source = Path(str(asset["resolved_path"]))
        if not source.is_file():
            continue

        path = Path(asset_path)
        if path.is_absolute():
            # Absolute texture paths remain valid after export.
            continue
        if any(part in ("", ".", "..") for part in path.parts):
            logger.warning("Skipping unsafe relative texture path: %s", asset_path)
            continue

        destination = export_dir / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.resolve() == source.resolve():
            continue

        shutil.copy2(source, destination)
        copied += 1

    return copied


# ---------------------------------------------------------------------------
# Subprocess worker script (executed in the ovrtx venv without pxr)
#
# This follows the same pattern as the ovrtx reference example
# (render_to_disk.py):
#   1. ovrtx.Renderer() — plain constructor, no config
#   2. renderer.add_usd(path) — load scene with render products sublayered
#   3. Multiple step() calls for path-tracer convergence
#   4. var.tensor.numpy() to extract pixels
#   5. Image.fromarray(pixels) to save
# ---------------------------------------------------------------------------

_WORKER_SCRIPT = r'''
"""OvRTX subprocess worker — runs in the isolated ovrtx venv."""
import json, os, sys

# pip --target fallback: prepend site dir so ovrtx can be imported.
_site = os.environ.get("_WU_OVRTX_SITE_DIR")
if _site:
    sys.path.insert(0, _site)

# Compat: ovrtx >=110.1.0.273788 changed map(device=) from str to enum.
def _cpu_device():
    import ovrtx
    return getattr(ovrtx, "Device", None) and ovrtx.Device.CPU or "cpu"


def main():
    params = json.loads(sys.argv[1])
    usd_path = params["usd_path"]
    fps = params.get("fps", 24.0)
    cameras = params["cameras"]
    frames = params["frames"]
    sensors = params["sensors"]
    output_dir = params["output_dir"]
    product_paths = params["product_paths"]
    # ``num_sensor_updates`` drives the progressive
    # ``renderer.step(delta_time=0)`` accumulation loop below — the only
    # quality knob OVRtx 0.2.0 honours. OVRtx silently ignores the
    # ``omni:rtx:pt:samplesPerPixel`` / ``omni:rtx:rt:accumulationLimit``
    # schema attributes, so we control convergence here, not in USD.
    num_sensor_updates = params.get("num_sensor_updates", 1)
    visibility_schedule = params.get("visibility_schedule", {})
    visibility_updates = params.get("visibility_updates")
    log_level = params.get("log_level", "warn")

    import logging
    _LOG_MAP = {"error": logging.ERROR, "warn": logging.WARNING,
                "info": logging.INFO, "debug": logging.DEBUG}
    logging.basicConfig(level=_LOG_MAP.get(log_level, logging.WARNING))

    import ovrtx
    import numpy as np
    from PIL import Image

    SENSOR_MAP = {"depth": "Depth", "normal": "Normal", "albedo": "Albedo"}

    all_product_paths = set(product_paths)

    cam_data = [
        {"images": [], "sensor_files": {s: {} for s in sensors}}
        for _ in cameras
    ]

    renderer = ovrtx.Renderer()
    renderer.add_usd(usd_path)

    for frame_num in frames:
        renderer.update_from_usd_time(float(frame_num) / fps)

        # Apply visibility via write_attribute API (workaround for
        # ovrtx 0.2.0 segfault on time-sampled visibility in USD).
        # Prefer pre-computed per-frame deltas; fall back to the legacy
        # full-frame schedule for direct callers using older params.
        frame_key = str(float(frame_num))
        vis_map = None
        if visibility_updates is not None:
            vis_map = visibility_updates.get(frame_key)
        elif frame_key in visibility_schedule:
            vis_map = visibility_schedule[frame_key]
        if vis_map:
            for prim_path, vis_value in vis_map.items():
                token = "inherited" if vis_value == "inherited" else "invisible"
                renderer.write_attribute([prim_path], "visibility", [token])

        renderer.reset()

        # Progressive path-tracer accumulation: step() with delta_time=0
        # keeps simulation time fixed so OVRtx's accumulator layers more
        # samples onto the same frame. Convergence plateaus near
        # ~500 iterations on the kit golden scene (PSNR climbs ~12 dB
        # over 1→100, another ~1.3 dB over 100→500, flat past there).
        # See the convergence sweep in /tmp/ovrtx_cap.py.
        all_products = None
        for _ in range(num_sensor_updates):
            all_products = renderer.step(
                render_products=all_product_paths,
                delta_time=0.0,
            )

        if all_products:
            for cam_idx, product_path in enumerate(product_paths):
                if product_path not in all_products:
                    continue
                product = all_products[product_path]
                for frame in product.frames:
                    if "LdrColor" in frame.render_vars:
                        with frame.render_vars["LdrColor"].map(device=_cpu_device()) as var:
                            pixels = var.tensor.numpy().copy()
                        fname = f"cam{cam_idx}_f{frame_num}.png"
                        fpath = os.path.join(output_dir, fname)
                        Image.fromarray(pixels).save(fpath)
                        cam_data[cam_idx]["images"].append(fname)

                    for sname in sensors:
                        rv = SENSOR_MAP.get(sname)
                        if rv and rv in frame.render_vars:
                            with frame.render_vars[rv].map(device=_cpu_device()) as var:
                                sarr = var.tensor.numpy().copy()
                            sfname = f"cam{cam_idx}_f{frame_num}_{sname}.npy"
                            np.save(os.path.join(output_dir, sfname), sarr)
                            cam_data[cam_idx]["sensor_files"][sname][frame_num] = sfname

    del renderer

    results = []
    for cam_idx, camera in enumerate(cameras):
        results.append({
            "camera": camera,
            "image_files": cam_data[cam_idx]["images"],
            "sensor_files": cam_data[cam_idx]["sensor_files"],
            "frame_count": len(cam_data[cam_idx]["images"]),
        })

    with open(os.path.join(output_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(results, f)

if __name__ == "__main__":
    main()
'''

# ---------------------------------------------------------------------------
# Persistent daemon script (executed in the ovrtx venv without pxr)
#
# Unlike _WORKER_SCRIPT which processes a single batch and exits, this
# daemon creates ovrtx.Renderer() once and loops on stdin reading JSON
# commands.  This amortises the ~5.5s GPU init across all render calls.
# ---------------------------------------------------------------------------

_DAEMON_SCRIPT = r'''
"""OvRTX persistent daemon — runs in the isolated ovrtx venv.

Creates Renderer() once, then loops on stdin for JSON-line commands.
Protocol:
  startup  → stdout: {"status": "ready"}
  request  → stdin:  {"command": "render", ...}
           ← stdout: {"status": "ok", "manifest": [...]}
  shutdown → stdin:  {"command": "shutdown"}
  error    ← stdout: {"status": "error", "error": "msg"}
"""
import json
import os
import sys
import traceback

# pip --target fallback: prepend site dir so ovrtx can be imported.
_site = os.environ.get("_WU_OVRTX_SITE_DIR")
if _site:
    sys.path.insert(0, _site)

# Compat: ovrtx >=110.1.0.273788 changed map(device=) from str to enum.
def _cpu_device():
    import ovrtx
    return getattr(ovrtx, "Device", None) and ovrtx.Device.CPU or "cpu"


def _redirect_native_stdout():
    """Redirect C-level fd 1 (stdout) to fd 2 (stderr).

    ovrtx / Vulkan may print init messages via the C runtime which bypass
    Python's sys.stdout and would corrupt our JSON protocol on the pipe.
    Returns the saved fd so it can be restored later.
    """
    saved_fd = os.dup(1)
    os.dup2(2, 1)  # fd 1 now points to stderr
    return saved_fd


def _restore_native_stdout(saved_fd):
    """Restore the original C-level stdout from a previously saved fd."""
    sys.stdout.flush()
    os.dup2(saved_fd, 1)
    os.close(saved_fd)


def main():
    # Configure Python logging from env var set by parent process
    import logging
    _LOG_MAP = {"error": logging.ERROR, "warn": logging.WARNING,
                "info": logging.INFO, "debug": logging.DEBUG}
    _lvl = os.environ.get("OVRTX_LOG_LEVEL", "warn")
    logging.basicConfig(level=_LOG_MAP.get(_lvl, logging.WARNING))

    # Redirect native stdout → stderr while importing ovrtx and creating
    # the Renderer, so Vulkan/driver messages don't corrupt our JSON pipe.
    saved_fd = _redirect_native_stdout()
    import ovrtx
    import numpy as np
    from PIL import Image

    SENSOR_MAP = {"depth": "Depth", "normal": "Normal", "albedo": "Albedo"}

    renderer = ovrtx.Renderer()
    _restore_native_stdout(saved_fd)

    # Signal readiness to parent
    sys.stdout.write(json.dumps({"status": "ready"}) + "\n")
    sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            sys.stdout.write(
                json.dumps({"status": "error", "error": f"bad JSON: {exc}"}) + "\n"
            )
            sys.stdout.flush()
            continue

        command = request.get("command")

        if command == "shutdown":
            break

        if command != "render":
            sys.stdout.write(
                json.dumps({"status": "error", "error": f"unknown command: {command}"}) + "\n"
            )
            sys.stdout.flush()
            continue

        try:
            usd_path = request["usd_path"]
            fps = request.get("fps", 24.0)
            cameras = request["cameras"]
            frames = request["frames"]
            sensors = request["sensors"]
            output_dir = request["output_dir"]
            product_paths = request["product_paths"]
            # num_sensor_updates drives the progressive
            # renderer.step(delta_time=0) accumulation loop below.
            # OVRtx 0.2.0 ignores the RenderProduct SPP/accum USD
            # attributes, so this is the only quality knob that fires.
            num_sensor_updates = request.get("num_sensor_updates", 1)
            visibility_schedule = request.get("visibility_schedule", {})
            visibility_updates = request.get("visibility_updates")

            all_product_paths = set(product_paths)

            cam_data = [
                {"images": [], "sensor_files": {s: {} for s in sensors}}
                for _ in cameras
            ]

            # Redirect native stdout during ovrtx calls so Vulkan log
            # messages don't corrupt the JSON protocol on the pipe.
            saved_fd = _redirect_native_stdout()
            try:
                renderer.reset_stage()
                renderer.add_usd(usd_path)

                for frame_num in frames:
                    renderer.update_from_usd_time(float(frame_num) / fps)

                    # Apply visibility via write_attribute API. Prefer
                    # pre-computed per-frame deltas; fall back to the legacy
                    # full-frame schedule for direct callers using older params.
                    frame_key = str(float(frame_num))
                    vis_map = None
                    if visibility_updates is not None:
                        vis_map = visibility_updates.get(frame_key)
                    elif frame_key in visibility_schedule:
                        vis_map = visibility_schedule[frame_key]
                    if vis_map:
                        for prim_path, vis_value in vis_map.items():
                            token = "inherited" if vis_value == "inherited" else "invisible"
                            renderer.write_attribute([prim_path], "visibility", [token])

                    renderer.reset()

                    # Progressive accumulation via dt=0 loop (see the
                    # one-shot worker block above for rationale).
                    all_products = None
                    for _ in range(num_sensor_updates):
                        all_products = renderer.step(
                            render_products=all_product_paths,
                            delta_time=0.0,
                        )

                    if all_products:
                        for cam_idx, product_path in enumerate(product_paths):
                            if product_path not in all_products:
                                continue
                            product = all_products[product_path]
                            for frame in product.frames:
                                if "LdrColor" in frame.render_vars:
                                    with frame.render_vars["LdrColor"].map(device=_cpu_device()) as var:
                                        pixels = var.tensor.numpy().copy()
                                    fname = f"cam{cam_idx}_f{frame_num}.png"
                                    fpath = os.path.join(output_dir, fname)
                                    Image.fromarray(pixels).save(fpath)
                                    cam_data[cam_idx]["images"].append(fname)

                                for sname in sensors:
                                    rv = SENSOR_MAP.get(sname)
                                    if rv and rv in frame.render_vars:
                                        with frame.render_vars[rv].map(device=_cpu_device()) as var:
                                            sarr = var.tensor.numpy().copy()
                                        sfname = f"cam{cam_idx}_f{frame_num}_{sname}.npy"
                                        np.save(os.path.join(output_dir, sfname), sarr)
                                        cam_data[cam_idx]["sensor_files"][sname][frame_num] = sfname
            finally:
                _restore_native_stdout(saved_fd)

            manifest = []
            for cam_idx, camera in enumerate(cameras):
                manifest.append({
                    "camera": camera,
                    "image_files": cam_data[cam_idx]["images"],
                    "sensor_files": cam_data[cam_idx]["sensor_files"],
                    "frame_count": len(cam_data[cam_idx]["images"]),
                })

            sys.stdout.write(
                json.dumps({"status": "ok", "manifest": manifest}) + "\n"
            )
            sys.stdout.flush()

        except Exception:
            sys.stdout.write(
                json.dumps({"status": "error", "error": traceback.format_exc()[-2000:]}) + "\n"
            )
            sys.stdout.flush()

    # Clean shutdown
    del renderer


if __name__ == "__main__":
    main()
'''


class _OvRTXDaemon:
    """Manages a persistent OvRTX renderer subprocess.

    The daemon creates ``ovrtx.Renderer()`` once at startup and loops on
    stdin for JSON-line render commands, avoiding the ~5.5 s GPU init cost
    on every call.  If the daemon crashes, the next ``render()`` call
    restarts it transparently.
    """

    def __init__(
        self,
        ovrtx_python: str,
        daemon_script_path: str,
        log_level: str = "warn",
    ) -> None:
        self._ovrtx_python = ovrtx_python
        self._daemon_script_path = daemon_script_path
        self._log_level = log_level
        self._process: subprocess.Popen[str] | None = None
        self._stderr_thread: threading.Thread | None = None
        self._stdout_buffer = b""
        self._lock = threading.Lock()
        self._start_timeout_s = float(
            os.environ.get("OVRTX_DAEMON_START_TIMEOUT", "600")
        )
        self._render_timeout_s = float(
            os.environ.get("OVRTX_DAEMON_RENDER_TIMEOUT", "1800")
        )
        atexit.register(self.shutdown)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def ensure_running(self) -> None:
        """Start the daemon if it is not already running."""
        with self._lock:
            if self._is_running():
                return
            self._start()

    def _start(self) -> None:
        """Launch the daemon subprocess and wait for its *ready* signal."""
        env = os.environ.copy()
        # Remove PYTHONPATH so the isolated ovrtx venv doesn't pick up
        # the app's usd-core (which conflicts with ovrtx's bundled USD).
        env.pop("PYTHONPATH", None)
        if not env.get("DISPLAY"):
            env["DISPLAY"] = ":0"
        env["OVRTX_LOG_LEVEL"] = self._log_level

        logger.info("Starting OvRTX daemon subprocess …")
        self._stdout_buffer = b""
        self._process = subprocess.Popen(
            [self._ovrtx_python, self._daemon_script_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )

        # Background thread to drain stderr so the pipe never fills up
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()

        # Wait for the "ready" JSON line. Do not let a wedged daemon pin
        # service startup forever; kill it so the next request can retry cleanly.
        ready_line = self._read_stdout_line(self._start_timeout_s, "startup")
        if not ready_line:
            rc = self._process.wait(timeout=10)
            raise RuntimeError(f"OvRTX daemon exited during init (exit code {rc})")
        msg = json.loads(ready_line)
        if msg.get("status") != "ready":
            raise RuntimeError(f"OvRTX daemon unexpected init msg: {msg}")
        logger.info("OvRTX daemon ready (pid %d)", self._process.pid)

    def _drain_stderr(self) -> None:
        """Read stderr lines and send them to the logger."""
        proc = self._process
        assert proc is not None and proc.stderr is not None
        for line in proc.stderr:
            logger.debug("[ovrtx-daemon] %s", line.rstrip())

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def render(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        """Send a render request and return the manifest.

        If the daemon is not running (or has crashed), it is (re)started
        automatically.
        """
        with self._lock:
            if not self._is_running():
                logger.warning("OvRTX daemon not running — restarting")
                self._start()

            assert self._process is not None
            assert self._process.stdin is not None
            assert self._process.stdout is not None

            request = {"command": "render", **params}
            try:
                self._process.stdin.write(json.dumps(request) + "\n")
                self._process.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                rc = self._process.poll()
                self._process = None
                raise RuntimeError(
                    f"OvRTX daemon pipe failed before render response (exit code {rc})"
                ) from exc

            response_line = self._read_stdout_line(self._render_timeout_s, "render")
            if not response_line:
                rc = self._process.poll() if self._process is not None else None
                self._process = None
                raise RuntimeError(f"OvRTX daemon died during render (exit code {rc})")
            response = json.loads(response_line)

        if response.get("status") == "error":
            raise RuntimeError(f"OvRTX daemon render error: {response.get('error')}")
        if response.get("status") != "ok":
            raise RuntimeError(f"OvRTX daemon unexpected response: {response}")
        manifest: list[dict[str, Any]] = response["manifest"]
        return manifest

    def _read_stdout_line(self, timeout_s: float, phase: str) -> str:
        """Read one daemon stdout line with a timeout.

        ``readline()`` on a subprocess pipe blocks indefinitely if the daemon
        stays alive but stops writing. A single ``select()`` before
        ``readline()`` is not enough because a partial line makes the fd
        readable and ``readline()`` can then block waiting for ``\n``. Read the
        pipe bytes directly until newline, EOF, or the real deadline.
        """
        assert self._process is not None
        assert self._process.stdout is not None

        buffered_line = self._pop_stdout_line()
        if buffered_line is not None:
            return buffered_line

        if timeout_s <= 0:
            if self._stdout_buffer:
                prefix = self._stdout_buffer.decode(errors="replace")
                self._stdout_buffer = b""
                return prefix + self._process.stdout.readline()
            return self._process.stdout.readline()

        fd = self._process.stdout.fileno()
        deadline = time.monotonic() + timeout_s
        selector = selectors.DefaultSelector()
        try:
            selector.register(fd, selectors.EVENT_READ)
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break

                events = selector.select(remaining)
                if not events:
                    break

                chunk = os.read(fd, 4096)
                if not chunk:
                    line = self._stdout_buffer.decode(errors="replace")
                    self._stdout_buffer = b""
                    return line

                self._stdout_buffer += chunk
                buffered_line = self._pop_stdout_line()
                if buffered_line is not None:
                    return buffered_line
        finally:
            selector.close()

        logger.error(
            "OvRTX daemon %s timed out after %.1fs; killing subprocess",
            phase,
            timeout_s,
        )
        self._kill_process()
        raise TimeoutError(f"OvRTX daemon {phase} timed out after {timeout_s:.1f}s")

    def _pop_stdout_line(self) -> str | None:
        if b"\n" not in self._stdout_buffer:
            return None
        line, self._stdout_buffer = self._stdout_buffer.split(b"\n", 1)
        return (line + b"\n").decode(errors="replace")

    def _kill_process(self) -> None:
        proc = self._process
        if proc is None:
            return
        try:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)
        except Exception:
            logger.exception("Failed to kill OvRTX daemon subprocess")
        finally:
            self._process = None
            self._stdout_buffer = b""

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Gracefully shut down the daemon subprocess."""
        with self._lock:
            if not self._is_running():
                self._process = None
                return
            assert self._process is not None
            assert self._process.stdin is not None
            try:
                self._process.stdin.write(json.dumps({"command": "shutdown"}) + "\n")
                self._process.stdin.flush()
                self._process.wait(timeout=10)
            except (BrokenPipeError, OSError):
                pass
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=5)
            finally:
                logger.info("OvRTX daemon shut down")
                self._process = None


# Default latlong HDRI shipped with the world-understanding package —
# ``SmartMaterials_Environment_with_Lights.exr``, the same brightly-
# exposed env map Kit's ``stage_manager._create_default_light`` binds
# on lightless scenes. Shipping a bundled EXR gives deterministic
# rendering without depending on network / S3 asset-resolution and
# makes pixel-parity with the Kit renderer possible out of the box
# (~39.7 dB PSNR at 500 PT iterations on the kit golden scene).
_DEFAULT_HDRI_PATH = str(
    Path(__file__).resolve().parents[2]
    / "data"
    / "env_maps"
    / "SmartMaterials_Environment_with_Lights.exr"
)

# DomeLight ``intensity`` multiplier applied to the HDRI texture.
# Matches Kit's ``stage_manager._create_default_light`` exactly: the
# SmartMaterials EXR is a brightly-exposed HDR, so intensity=1 already
# produces a correctly-exposed scene. Operators pointing
# ``WU_OVRTX_DEFAULT_HDRI`` at a dimmer public HDRI (e.g. ovrtx's
# ``ZetoCGcom_ExhibitionHall_Interior1.hdr``) should tune via
# ``WU_OVRTX_DEFAULT_HDRI_INTENSITY`` (~600 matches Kit-ref brightness
# on the kit golden scene, see docs/developer/OVRTX_LIMITATIONS.md §6).
_DEFAULT_HDRI_INTENSITY = 1.0


def _stage_has_lights(stage: "Usd.Stage") -> bool:
    """True if ``stage`` already contains at least one UsdLux light prim."""
    from pxr import UsdLux

    return any(
        p.IsA(UsdLux.BoundableLightBase) or p.IsA(UsdLux.NonboundableLightBase)
        for p in stage.Traverse()
    )


def _resolve_default_hdri() -> str:
    """Return the HDRI asset path/URL to use for default DomeLight binding."""
    return os.environ.get("WU_OVRTX_DEFAULT_HDRI", "").strip() or _DEFAULT_HDRI_PATH


def _resolve_default_hdri_intensity() -> float:
    """Return the ``DomeLight.intensity`` to use for the default HDRI dome.

    Reads ``WU_OVRTX_DEFAULT_HDRI_INTENSITY`` env var (float); falls back
    to ``_DEFAULT_HDRI_INTENSITY`` (1.0, matching Kit when paired with
    the bundled SmartMaterials EXR).
    """
    val = os.environ.get("WU_OVRTX_DEFAULT_HDRI_INTENSITY", "").strip()
    if not val:
        return _DEFAULT_HDRI_INTENSITY
    try:
        return float(val)
    except ValueError:
        logger.warning(
            "Invalid WU_OVRTX_DEFAULT_HDRI_INTENSITY=%r, using default %g",
            val,
            _DEFAULT_HDRI_INTENSITY,
        )
        return _DEFAULT_HDRI_INTENSITY


def _build_default_lights_usda(hdri_asset: str, intensity: float) -> str:
    """Serialize a default HDRI DomeLight as a standalone USDA overlay.

    This is the *sublayer* form of the lights rig, used by
    ``render_all_cameras``. Mutating the loaded scene stage in-place and
    then ``Export()``-ing it sometimes rewrites ``@<url>@`` asset paths
    to resolved local paths that OvRTX can't resolve, producing a black
    render. Writing the dome to a separate sublayer file preserves the
    asset URL verbatim, matching the robot-ovrtx/planet-system reference
    scene structure and what the ovrtx examples render cleanly.
    """
    return f"""#usda 1.0
(
)

def "OvRTXDefaultLights" (
    hide_in_stage_window = true
    no_delete = true
)
{{
    def DomeLight "DomeLight"
    {{
        float inputs:intensity = {intensity}
        token inputs:texture:format = "latlong"
        asset inputs:texture:file = @{hdri_asset}@
        custom bool visibleInPrimaryRay = 0
    }}
}}
"""


def _ensure_lights(stage: "Usd.Stage") -> None:
    """Add a default HDRI DomeLight to the stage if none are present.

    Kept for stand-alone callers that mutate a ``Usd.Stage`` directly
    (tests, CLIs). The daemon pipeline in ``render_all_cameras`` uses
    ``_build_default_lights_usda`` as a sublayer overlay instead —
    see the helper's docstring for why.

    Args:
        stage: USD stage to check/modify (in-place).
    """
    from pxr import Sdf, UsdLux

    if _stage_has_lights(stage):
        return

    hdri = _resolve_default_hdri()
    logger.info("No lights in stage — adding HDRI DomeLight (%s)", hdri)

    dome = UsdLux.DomeLight.Define(stage, "/OvRTXDefaultLights/DomeLight")
    dome.CreateIntensityAttr(_resolve_default_hdri_intensity())
    dome.GetPrim().CreateAttribute(
        "inputs:texture:format", Sdf.ValueTypeNames.Token
    ).Set("latlong")
    dome.GetPrim().CreateAttribute("inputs:texture:file", Sdf.ValueTypeNames.Asset).Set(
        hdri
    )
    dome.GetPrim().CreateAttribute("visibleInPrimaryRay", Sdf.ValueTypeNames.Bool).Set(
        False
    )


def render_all_cameras(
    stage: "Usd.Stage",
    image_width: int = 1024,
    image_height: int = 1024,
    cameras: list[str] | None = None,
    frames: str = "0",
    sensors: list[str] | None = None,
    ovrtx_renderer: Any = None,
    log_level: str = "warn",
    ovrtx_venv_dir: Path | str | None = None,
    num_sensor_updates: int = DEFAULT_NUM_SENSOR_UPDATES,
    render_mode: str = DEFAULT_RENDER_MODE,
    daemon: _OvRTXDaemon | None = None,
) -> dict[str, Any]:
    """Render multiple cameras from a USD stage using OvRTX.

    This function exports the stage to a temp file, then launches an isolated
    subprocess using a separate ovrtx-only venv (without usd-core) that
    renders all cameras and saves images to a temp directory, which are then
    loaded back in the main process.

    Args:
        stage: USD stage to render.
        image_width: Output image width in pixels.
        image_height: Output image height in pixels.
        cameras: List of camera prim paths. If None, uses ["/Camera"].
        frames: Frame specification (e.g., "0", "0:10", "0,5,10").
        sensors: Optional sensor names (e.g., ["depth"]).
        ovrtx_renderer: Ignored (kept for API compatibility). Subprocess
            always creates its own renderer.
        log_level: OvRTX log level ("error", "warn", "info", "debug").
        ovrtx_venv_dir: Override directory for the isolated ovrtx venv.
            Defaults to ``~/.cache/wu/ovrtx_venv``.
        num_sensor_updates: Number of progressive
            ``renderer.step(delta_time=0)`` iterations to run per frame.
            Drives quality: more iterations = less path-trace noise, at
            linear wall-clock cost. Default ``500`` is the plateau where
            PT output converges within ~2 dB PSNR of Kit's reference
            render; lower values trade quality for latency. Note:
            OVRtx 0.2.0 silently ignores the ``omni:rtx:pt:samplesPerPixel``
            / ``omni:rtx:rt:accumulationLimit`` USD attributes, so this
            step loop is the only quality knob that actually takes effect.
        render_mode: ``rt1`` | ``rt2`` | ``pt``. Translates to
            ``omni:rtx:rendermode`` on the RenderProduct. Default ``pt``
            (PathTracing); ``rt2`` (``RealTimePathTracing``, Kit's
            default) caps at ~27 dB PSNR vs the Kit reference regardless
            of step count, so PT is the only mode that reaches
            Kit-parity quality. See ``docs/developer/OVRTX_LIMITATIONS.md``
            §5.
        daemon: Optional persistent daemon. When provided, the daemon's
            already-running ``ovrtx.Renderer()`` is reused, avoiding the
            ~5.5 s GPU init on every call.  When ``None`` (the default),
            falls back to a one-shot ``subprocess.run()`` worker.

    Returns:
        Dict matching RenderingBackend.render() contract with keys:
            total_cameras, successful_cameras, failed_cameras,
            total_render_time, results (list of per-camera dicts).
    """
    if cameras is None or len(cameras) == 0:
        cameras = ["/Camera"]

    frame_list = _parse_frames(frames)
    total_start_time = time.time()

    # Resolve the ovrtx venv Python (auto-provisions on first call)
    venv_path = Path(ovrtx_venv_dir) if ovrtx_venv_dir else None
    ovrtx_python = _get_ovrtx_python(venv_dir=venv_path)

    # Create temp directory for IPC (exported USD + rendered images)
    tmp_dir = tempfile.mkdtemp(prefix="ovrtx_render_")
    tmp_usd_path = os.path.join(tmp_dir, "stage.usdc")

    # Imports and state used by both the try and finally blocks. Keeping
    # them out here means an early exception inside the try doesn't trigger
    # a NameError in the finally cleanup that would mask the original error.
    from pxr import Usd
    from pxr import UsdGeom as _UsdGeom

    # Keys are stringified floats (e.g. "1.0", "1.5") so subframe time
    # samples survive both the dict lookup and the JSON serialization
    # round-trip into the renderer subprocess. Integer-frame consumers
    # build their lookup with `str(float(frame_num))` to match.
    visibility_schedule: dict[str, dict[str, str]] = {}
    stripped_prim_paths: list[str] = []
    # Captured default visibility per prim path. Restoring time samples
    # without restoring the default would silently flip prims that were
    # `invisible` by default to `inherited`. Populated at strip-time and
    # consumed in the finally block.
    stripped_default_vis: dict[str, str] = {}
    had_default_lights = True  # safe default — finally only removes if False

    try:
        # Pre-build the USDA and product paths (pure string operations, no ovrtx)
        usda_content, product_paths = _build_render_products_usda(
            cameras=cameras,
            image_width=image_width,
            image_height=image_height,
            sensors=sensors,
            render_mode=render_mode,
        )

        # OvRTX requires explicit lights — scenes without any get a
        # default HDRI DomeLight. We emit it as a *sublayer* overlay
        # rather than mutating the live stage + re-exporting, because
        # Export() sometimes rewrites asset URL paths in ways OvRTX
        # can't resolve (resulting in a black render). The sublayer
        # path preserves the URL verbatim. See _build_default_lights_usda.
        root_layer = stage.GetRootLayer()
        had_default_lights = bool(stage.GetPrimAtPath("/OvRTXDefaultLights"))
        default_lights_layer_path: str | None = None
        if not _stage_has_lights(stage):
            default_lights_layer_path = os.path.join(tmp_dir, "default_lights.usda")
            with open(default_lights_layer_path, "w", encoding="utf-8") as f:
                f.write(
                    _build_default_lights_usda(
                        _resolve_default_hdri(),
                        _resolve_default_hdri_intensity(),
                    )
                )
            logger.info(
                "Scene has no lights — overlaying default HDRI DomeLight sublayer (%s)",
                default_lights_layer_path,
            )

        # Extract time-sampled visibility schedule and strip it from the
        # stage.  OvRTX 0.2.0 segfaults when USD contains time-sampled
        # visibility attributes.  Instead, we pass the schedule to the
        # subprocess which uses renderer.write_attribute() to toggle
        # visibility per frame via the ovrtx API.
        for prim in stage.Traverse():
            vis_attr = _UsdGeom.Imageable(prim).GetVisibilityAttr()
            if not vis_attr or vis_attr.GetNumTimeSamples() == 0:
                continue
            prim_path_str = str(prim.GetPath())
            # Capture the prim's *default* visibility opinion before we
            # blow it away — Clear() drops both default and time samples,
            # and the restore path needs to put the default back so prims
            # that were invisible-by-default stay that way.
            default_val = vis_attr.Get(Usd.TimeCode.Default())
            if default_val is not None:
                stripped_default_vis[prim_path_str] = str(default_val)
            # Record schedule per time code (preserving subframes)
            for tc in vis_attr.GetTimeSamples():
                frame_key = str(float(tc))
                if frame_key not in visibility_schedule:
                    visibility_schedule[frame_key] = {}
                val = vis_attr.Get(Usd.TimeCode(tc))
                visibility_schedule[frame_key][prim_path_str] = str(val)
            # Clear time samples and set default to inherited (visible)
            vis_attr.Clear()
            vis_attr.Set(_UsdGeom.Tokens.inherited)
            stripped_prim_paths.append(prim_path_str)

        if stripped_prim_paths:
            logger.info(
                "Extracted visibility schedule for %d prims across %d frames "
                "(ovrtx write_attribute workaround)",
                len(stripped_prim_paths),
                len(visibility_schedule),
            )
        visibility_updates = _build_visibility_frame_updates(
            visibility_schedule, frame_list
        )
        if visibility_updates:
            update_count = sum(len(v) for v in visibility_updates.values())
            logger.info(
                "Prepared %d visibility write(s) across %d rendered frame(s) "
                "from %d scheduled frame(s)",
                update_count,
                len(visibility_updates),
                len(visibility_schedule),
            )

        # Export the stage to a temp file (without render products).
        t_export = time.time()
        if not root_layer.Export(tmp_usd_path):
            raise RuntimeError("Failed to export USD stage to temp file")
        copied_assets = _copy_exported_relative_assets(stage, Path(tmp_dir))
        if copied_assets:
            logger.info(
                "Copied %d local texture asset(s) next to exported OVRTX stage",
                copied_assets,
            )

        # Write render products + lights as a separate USDA overlay
        render_products_layer_path = os.path.join(tmp_dir, "render_products.usda")
        with open(render_products_layer_path, "w", encoding="utf-8") as f:
            f.write(usda_content)

        # Create a combined USDA that sublayers the scene + render products.
        # This matches the contents-claw pattern: ovrtx loads the combined
        # file and resolves both layers correctly (vs flattening which can
        # break ovrtx 0.2.0's render product discovery).
        from pxr import Sdf

        combined_path = os.path.join(tmp_dir, "combined.usda")
        combined = Sdf.Layer.CreateNew(combined_path)
        sublayers = [tmp_usd_path, render_products_layer_path]
        if default_lights_layer_path is not None:
            sublayers.append(default_lights_layer_path)
        combined.subLayerPaths = sublayers
        combined.Save()
        tmp_usd_path = combined_path
        logger.debug("Exported USD stage in %.2fs", time.time() - t_export)

        # Read timeCodesPerSecond so the subprocess converts frame
        # numbers to seconds for update_from_usd_time().
        fps = stage.GetTimeCodesPerSecond()  # defaults to 24.0

        # Build subprocess parameters
        params = {
            "usd_path": tmp_usd_path,
            "fps": fps,
            "cameras": cameras,
            "image_width": image_width,
            "image_height": image_height,
            "frames": frame_list,
            "sensors": sensors or [],
            "output_dir": tmp_dir,
            "product_paths": product_paths,
            "num_sensor_updates": num_sensor_updates,
            "visibility_updates": visibility_updates,
            "log_level": log_level,
        }

        if daemon is not None:
            # ---- Persistent daemon path (reuses Renderer across calls) ----
            logger.info(
                "Sending render request to OvRTX daemon: %d camera(s), %d frame(s)",
                len(cameras),
                len(frame_list),
            )
            daemon.ensure_running()
            t_render = time.time()
            manifest = daemon.render(params)
            logger.debug("Daemon render completed in %.2fs", time.time() - t_render)
        else:
            # ---- One-shot subprocess path (backward compatible) ----
            # Write worker script to temp file
            worker_path = os.path.join(tmp_dir, "_ovrtx_worker.py")
            with open(worker_path, "w", encoding="utf-8") as f:
                f.write(_WORKER_SCRIPT)

            # Launch subprocess using the isolated ovrtx venv Python.
            # Remove PYTHONPATH so the venv doesn't pick up the app's usd-core.
            # OvRTX requires Vulkan GPU access which needs a display server.
            env = os.environ.copy()
            env.pop("PYTHONPATH", None)
            if not env.get("DISPLAY"):
                env["DISPLAY"] = ":0"
            env["OVRTX_LOG_LEVEL"] = log_level

            logger.info(
                "Launching OvRTX subprocess for %d camera(s), %d frame(s)",
                len(cameras),
                len(frame_list),
            )
            proc = subprocess.run(
                [ovrtx_python, worker_path, json.dumps(params)],
                capture_output=True,
                text=True,
                env=env,
                check=False,
            )

            if proc.returncode != 0:
                error_msg = f"OvRTX subprocess failed (exit code {proc.returncode})"
                if proc.stdout:
                    error_msg += f"\n--- stdout (last 1000) ---\n{proc.stdout[-1000:]}"
                if proc.stderr:
                    error_msg += f"\n--- stderr (last 2000) ---\n{proc.stderr[-2000:]}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)

            # Read manifest from subprocess output
            manifest_path = os.path.join(tmp_dir, "manifest.json")
            if not os.path.exists(manifest_path):
                raise RuntimeError(
                    "OvRTX subprocess did not produce manifest. "
                    f"stdout: {proc.stdout[-500:]}, "
                    f"stderr: {proc.stderr[-500:]}"
                )

            with open(manifest_path, encoding="utf-8") as f:
                manifest = json.load(f)

        # Load images and sensor data back into memory
        t_load = time.time()
        results = []
        successful_cameras = 0
        failed_cameras = 0

        for cam_result in manifest:
            camera_images: list[Image.Image] = []
            camera_sensors: dict[str, dict[int, np.ndarray]] = {
                s: {} for s in (sensors or [])
            }

            # Load images
            for img_fname in cam_result["image_files"]:
                img_path = os.path.join(tmp_dir, img_fname)
                if os.path.exists(img_path):
                    camera_images.append(Image.open(img_path).copy())

            # Load sensor data
            for sensor_name, frame_files in cam_result["sensor_files"].items():
                for frame_num_str, npy_fname in frame_files.items():
                    npy_path = os.path.join(tmp_dir, npy_fname)
                    if os.path.exists(npy_path):
                        camera_sensors[sensor_name][int(frame_num_str)] = np.load(
                            npy_path
                        )

            if camera_images:
                successful_cameras += 1
                result: dict[str, Any] = {
                    "camera": cam_result["camera"],
                    "images": camera_images,
                    "sensors": camera_sensors,
                    "render_time": time.time() - total_start_time,
                    "frame_count": len(camera_images),
                }
            else:
                failed_cameras += 1
                result = {
                    "camera": cam_result["camera"],
                    "images": [],
                    "sensors": {},
                    "render_time": time.time() - total_start_time,
                    "frame_count": 0,
                    "error": "No images produced",
                }

            results.append(result)

        logger.debug(
            "Loaded %d images from disk in %.2fs",
            sum(len(r.get("images", [])) for r in results),
            time.time() - t_load,
        )

    finally:
        # Revert stage mutations so the caller's stage is unchanged
        # Restore visibility time samples and the prim's original default
        # opinion (the strip path captured the default before clearing).
        if stripped_prim_paths:
            for prim_path in stripped_prim_paths:
                prim = stage.GetPrimAtPath(prim_path)
                if not prim:
                    continue
                vis_attr = _UsdGeom.Imageable(prim).GetVisibilityAttr()
                vis_attr.Clear()
                # Restore the original default opinion BEFORE writing time
                # samples so prims that were invisible-by-default keep that
                # default after the render call returns.
                orig_default = stripped_default_vis.get(prim_path)
                if orig_default is not None:
                    default_token = (
                        _UsdGeom.Tokens.inherited
                        if orig_default == "inherited"
                        else _UsdGeom.Tokens.invisible
                    )
                    vis_attr.Set(default_token)
                for frame_key, vis_map in visibility_schedule.items():
                    if prim_path in vis_map:
                        val = vis_map[prim_path]
                        token = (
                            _UsdGeom.Tokens.inherited
                            if val == "inherited"
                            else _UsdGeom.Tokens.invisible
                        )
                        # frame_key is a stringified float ("1.0", "1.5", ...)
                        # so use float() to preserve subframe samples on
                        # round-trip back to the stage.
                        vis_attr.Set(token, time=Usd.TimeCode(float(frame_key)))

        if not had_default_lights:
            default_lights = stage.GetPrimAtPath("/OvRTXDefaultLights")
            if default_lights:
                stage.RemovePrim("/OvRTXDefaultLights")
        # Capture-and-replay debug hook: when set, the entire render
        # tmp_dir (combined.usda + scene export + render_products +
        # default_lights sublayer) is copied to this directory and the
        # combined.usda gets its sublayer paths rewritten to basenames
        # so the whole bundle is portable. Lets us diff the daemon's
        # output against a known-good pure-ovrtx scene to root-cause
        # rendering issues without rebuilding the container.
        dump_path = os.environ.get("WU_OVRTX_DUMP_COMBINED", "").strip()
        if dump_path:
            try:
                dump_dir = Path(dump_path).parent
                dump_dir.mkdir(parents=True, exist_ok=True)
                # Copy every file under tmp_dir to dump_dir (flat).
                for p in Path(tmp_dir).rglob("*"):
                    if p.is_file():
                        shutil.copy(p, dump_dir / p.name)
                # Rewrite the combined.usda sublayer paths to basenames.
                combined_src = dump_dir / Path(tmp_usd_path).name
                combined_dst = Path(dump_path)
                text = combined_src.read_text()
                text = text.replace(str(tmp_dir) + "/", "")
                combined_dst.write_text(text)
                if combined_dst != combined_src:
                    combined_src.unlink(missing_ok=True)
                logger.info(
                    "WU_OVRTX_DUMP_COMBINED: wrote %s (+ sublayers in %s)",
                    combined_dst,
                    dump_dir,
                )
            except Exception as _e:
                logger.warning("WU_OVRTX_DUMP_COMBINED copy failed: %s", _e)
        # Clean up temp directory
        try:
            shutil.rmtree(tmp_dir)
        except Exception:
            pass

    total_render_time = time.time() - total_start_time

    return {
        "total_cameras": len(cameras),
        "successful_cameras": successful_cameras,
        "failed_cameras": failed_cameras,
        "total_render_time": total_render_time,
        "results": results,
    }
