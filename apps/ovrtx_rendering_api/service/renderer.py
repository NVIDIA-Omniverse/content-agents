# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OVRTX rendering wrapper for the rendering API service.

Fetches USD from URL/data-URI, renders via OvRTXRenderingBackend,
and converts results to the V1 response format expected by the
material-agent-service client.
"""

from __future__ import annotations

import base64
import binascii
import io
import logging
import os
import re
import tempfile
import threading
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import requests
from PIL import Image

logger = logging.getLogger(__name__)

# USD layer extensions recognized inside ZIP bundles. Mirrors the client
# bundling in world_understanding.functions.graphics.render_nvcf, which
# packages a .usda root plus MDL/texture assets when the scene references
# local files.
_USD_EXTENSIONS = (".usd", ".usda", ".usdc")

# Denial-of-service guards for untrusted ZIP bundles. Real material bundles
# are well under these limits; the thresholds exist to stop classic
# amplification attacks (a few-KB bomb expanding to multiple GB).
_ZIP_MAX_FILES = 10_000
_ZIP_MAX_UNCOMPRESSED_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB

# Regex for S3 HTTPS URLs:
#   https://bucket.s3.region.amazonaws.com/key
#   https://s3.region.amazonaws.com/bucket/key
_S3_VHOST_RE = re.compile(
    r"^https?://(?P<bucket>[a-z0-9][a-z0-9.\-]+)\.s3[.\-](?P<region>[a-z0-9-]+)\.amazonaws\.com/(?P<key>.+)$"
)
_S3_PATH_RE = re.compile(
    r"^https?://s3[.\-](?P<region>[a-z0-9-]+)\.amazonaws\.com/(?P<bucket>[^/]+)/(?P<key>.+)$"
)

# Sensor name mapping: Kit API names -> OVRTX names
_SENSOR_KIT_TO_OVRTX: dict[str, str] = {
    "linear_depth": "depth",
    "depth": "depth",
}

# Sensors supported by OVRTX
_SUPPORTED_SENSORS = {"depth"}


class Renderer:
    """OVRTX-based USD renderer.

    Wraps OvRTXRenderingBackend from world_understanding with methods
    for fetching USD and converting results to the Kit API V1 format.
    """

    def __init__(
        self,
        log_level: str = "warn",
        num_sensor_updates: int = 500,
        render_mode: str = "pt",
    ) -> None:
        from world_understanding.functions.graphics.rendering import (
            OvRTXRenderingBackend,
        )

        self._backend = OvRTXRenderingBackend(
            log_level=log_level,
            num_sensor_updates=num_sensor_updates,
            render_mode=render_mode,
        )
        # Constructing OvRTXRenderingBackend builds the daemon IPC handle but
        # the actual ovrtx subprocess (and the GPU) doesn't start until the
        # first render request — see rendering.py "The daemon starts lazily
        # on first render(); GPU init cost is paid once." Leave `_initialized`
        # False until warm_up() actually exercises the GPU so /health does
        # not lie about readiness.
        self._initialized = False
        # The OVRTX daemon crashes when hit with concurrent renders.
        # Serialize all render calls through a lock so requests queue
        # instead of overlapping.
        self._render_lock = threading.Lock()
        logger.info(
            "OVRTX renderer constructed (num_sensor_updates=%d, render_mode=%s)",
            num_sensor_updates,
            render_mode,
        )

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def warm_up(self) -> bool:
        """Force lazy GPU init by rendering a tiny programmatic scene once.

        Builds an in-memory USD stage (cube + camera + distant light) and
        drives it straight through the backend. No fixture file needed —
        earlier versions shipped ``tests/renders/smoke_cube.usda`` and
        resolved it via ``__file__``, which silently broke on the wheel
        install path because the wheel doesn't ship ``tests/``.

        Returns True on success and flips ``is_initialized``. Returns
        False and leaves the renderer in the not-initialized state on
        failure so /health correctly reports the GPU is not ready.
        """
        try:
            stage = _build_smoke_stage()
            result = self._backend.render(
                stage=stage,
                cameras=["/World/Camera"],
                image_width=64,
                image_height=64,
                frames="0",
                sensors=None,
            )
        except Exception:
            logger.exception("OVRTX warm-up render failed; GPU not initialized")
            return False
        if not result.get("results"):
            logger.error("OVRTX warm-up render returned no results: %s", result)
            return False
        self._initialized = True
        logger.info("OVRTX renderer warmed up — GPU is ready")
        return True

    def shutdown(self) -> None:
        """Shut down the OVRTX daemon.

        TODO: switch to a public OvRTXRenderingBackend.shutdown() once the
        backend exposes one — we reach into ``_daemon`` directly today.
        """
        if hasattr(self._backend, "_daemon"):
            self._backend._daemon.shutdown()
        self._initialized = False

    def render(
        self,
        url: str,
        camera_paths: list[str],
        frame_start: int,
        frame_end: int,
        width: int,
        height: int,
        sensors: list[str] | None = None,
        num_sensor_updates: int | None = None,
        render_mode: str | None = None,
    ) -> dict[str, Any]:
        """Render a USD file and return V1-format response.

        Args:
            url: USD file URL (http/https) or data URI.
            camera_paths: Camera prim paths to render.
            frame_start: First frame to render.
            frame_end: Last frame to render (inclusive).
            width: Output image width.
            height: Output image height.
            sensors: Optional sensor names (Kit API names).
            num_sensor_updates: Per-request number of progressive
                ``renderer.step(dt=0)`` iterations per frame. ``None``
                falls back to the instance default
                (``OVRTX_NUM_SENSOR_UPDATES``, default 500 — the
                convergence plateau on the kit golden scene).
            render_mode: ``rt1`` | ``rt2`` | ``pt``. ``None`` falls back
                to the instance default (``OVRTX_RENDER_MODE``, default
                ``pt`` — the only mode that reaches Kit-parity quality).

        Returns:
            V1 response dict: {status, error, images}.
        """
        from pxr import Usd

        # Map requested sensors to OVRTX names, track unsupported
        ovrtx_sensors: list[str] = []
        unsupported_sensors: list[str] = []
        for s in sensors or []:
            mapped = _SENSOR_KIT_TO_OVRTX.get(s)
            if mapped and mapped in _SUPPORTED_SENSORS:
                if mapped not in ovrtx_sensors:
                    ovrtx_sensors.append(mapped)
            else:
                unsupported_sensors.append(s)

        if unsupported_sensors:
            logger.warning(
                "Unsupported sensors (will be empty in response): %s",
                unsupported_sensors,
            )

        # Fetch USD to temp file. Use the extension-agnostic `.usd` name so
        # USD's format sniffing picks the right reader for the content —
        # `.usda` ASCII and `.usdc` binary crate both work. Hardcoding
        # `.usdc` broke ASCII payloads ("Sdf crate bootstrap section corrupt").
        tmp_dir = tempfile.mkdtemp(prefix="render_api_")
        usd_path = os.path.join(tmp_dir, "input.usd")

        try:
            _fetch_usd(url, usd_path)

            # ZIP payloads include generic render_nvcf bundles and USDZ
            # packages. Extract both forms so relative texture files are real
            # files on disk before render_ovrtx re-exports the stage into its
            # daemon IPC directory. Opening a USDZ package directly keeps
            # textures behind ArPackageResolver; they are then lost on export.
            if zipfile.is_zipfile(usd_path):
                usd_path = _extract_zip_bundle(
                    usd_path,
                    tmp_dir,
                    prefer_first_usd=_is_usdz_payload(url, usd_path),
                )

            stage = Usd.Stage.Open(usd_path)
            if not stage:
                return _error_response(f"Failed to open USD stage from: {url}")

            # Build frames string
            if frame_start == frame_end:
                frames = str(frame_start)
            else:
                frames = f"{frame_start}:{frame_end}"

            # Serialize render calls — the OVRTX daemon crashes under
            # concurrent renders. Uvicorn dispatches sync endpoints to a
            # thread pool, so without this lock multiple requests would
            # call _backend.render() in parallel and kill the daemon.
            with self._render_lock:
                result = self._backend.render(
                    stage=stage,
                    cameras=camera_paths,
                    image_width=width,
                    image_height=height,
                    frames=frames,
                    sensors=ovrtx_sensors or None,
                    num_sensor_updates=num_sensor_updates,
                    render_mode=render_mode,
                )

            # Convert to V1 response format
            return _to_v1_response(result, sensors or [], ovrtx_sensors, frame_start)

        except Exception as e:
            logger.exception("Render failed")
            return _error_response(str(e))
        finally:
            # Clean up temp files
            import shutil

            shutil.rmtree(tmp_dir, ignore_errors=True)


def _build_smoke_stage() -> Any:
    """Build a tiny in-memory USD stage for warm-up rendering.

    Contains a single unit cube at the origin, a camera at (3,3,3)
    looking at the origin, and a distant key light. Kept deliberately
    minimal — the point is to exercise GPU init, not to render anything
    pretty.
    """
    from pxr import Gf, Usd, UsdGeom, UsdLux

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetDefaultPrim(stage.DefinePrim("/World", "Xform"))

    cube = UsdGeom.Cube.Define(stage, "/World/Cube")
    cube.CreateSizeAttr(1.0)
    cube.CreateDisplayColorAttr([Gf.Vec3f(0.6, 0.4, 0.2)])

    cam = UsdGeom.Camera.Define(stage, "/World/Camera")
    cam.CreateFocalLengthAttr(35.0)
    cam.CreateHorizontalApertureAttr(36.0)
    cam.CreateVerticalApertureAttr(36.0)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.1, 1000.0))
    cam_xform = UsdGeom.Xformable(cam)
    cam_xform.AddTranslateOp().Set(Gf.Vec3d(3.0, 3.0, 3.0))
    cam_xform.AddRotateXYZOp().Set(Gf.Vec3f(-30.0, 45.0, 0.0))

    light = UsdLux.DistantLight.Define(stage, "/World/KeyLight")
    light.CreateIntensityAttr(5000.0)
    UsdGeom.Xformable(light).AddRotateXYZOp().Set(Gf.Vec3f(-45.0, 30.0, 0.0))

    return stage


def _validate_url_target(url: str) -> None:
    """Block HTTP requests to cloud metadata, loopback, and private addresses."""
    import ipaddress
    from urllib.parse import urlparse

    parsed = urlparse(url)
    host = parsed.hostname or ""
    blocked_hosts = {
        "localhost",
        "169.254.169.254",
        "metadata.google.internal",
    }
    if host in blocked_hosts:
        raise ValueError(f"URL blocked (metadata/loopback): {url[:80]}")
    try:
        ip = ipaddress.ip_address(host)
        if ip.is_private or ip.is_loopback or ip.is_link_local:
            raise ValueError(f"URL blocked (private/link-local IP): {url[:80]}")
    except ValueError:
        pass  # Not an IP literal — hostname is fine


def _is_usdz_payload(url: str, zip_path: str) -> bool:
    """Return True if the ZIP should open as a native ``.usdz`` package.

    Primary signal is the URL path: a ``.usdz`` suffix on http/https/s3
    URLs is the caller's declaration that they want package semantics
    (default prim, internal ``package.usdz[inner.usda]`` references)
    preserved. For data URIs, which carry no path, fall back to the
    structural signature of the Pixar USDZ spec — all entries stored
    uncompressed with a USD layer as the first member. That shape is
    narrow enough that it only matches archives produced by the USDZ
    tooling (``usdzip`` and equivalents), not generic bundles.

    This keeps render_nvcf client bundles on the extraction path (they
    upload ``.zip`` with ``ZIP_DEFLATED``) while letting real USDZ
    assets reach Usd.Stage.Open intact.
    """
    from urllib.parse import urlparse

    if url.startswith("data:"):
        return _zip_matches_usdz_structure(zip_path)
    return urlparse(url).path.lower().endswith(".usdz")


def _zip_matches_usdz_structure(zip_path: str) -> bool:
    """Narrow check for the Pixar USDZ archive layout (stored + USD first)."""
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            infos = zf.infolist()
    except zipfile.BadZipFile:
        return False
    if not infos:
        return False
    if any(info.compress_type != zipfile.ZIP_STORED for info in infos):
        return False
    first = infos[0].filename.lower()
    return any(first.endswith(ext) for ext in _USD_EXTENSIONS)


def _extract_zip_bundle(
    zip_path: str,
    tmp_dir: str,
    prefer_first_usd: bool = False,
) -> str:
    """Extract a USD+assets bundle and return the path to the main USD layer.

    Selection priority follows kit-gen-ai-service/common/file_handler.py so the
    two services share behavior: ``main.*`` → ``scene.*`` → ``stage.*`` (the
    root name used by render_nvcf._bundle_stage_with_local_assets) → first
    match alphabetically. For USDZ payloads, ``prefer_first_usd`` selects the
    first USD layer in archive order because the USDZ spec uses that as the
    package root. Raises ValueError if no USD layer is present.
    """
    extract_dir = os.path.join(tmp_dir, "bundle")
    os.makedirs(extract_dir, exist_ok=True)
    extract_root = Path(extract_dir).resolve()
    with zipfile.ZipFile(zip_path, "r") as zf:
        infos = zf.infolist()

        # ZIP-bomb guard: refuse archives that would expand past our
        # per-request ceilings. Counts and sizes come from the ZIP's
        # central directory, so we reject *before* any data hits disk.
        if len(infos) > _ZIP_MAX_FILES:
            raise ValueError(
                f"ZIP bundle has too many entries: {len(infos)} "
                f"(limit {_ZIP_MAX_FILES})"
            )
        total_uncompressed = sum(info.file_size for info in infos)
        if total_uncompressed > _ZIP_MAX_UNCOMPRESSED_BYTES:
            raise ValueError(
                f"ZIP bundle uncompressed size too large: "
                f"{total_uncompressed} bytes (limit {_ZIP_MAX_UNCOMPRESSED_BYTES})"
            )

        # Python's zipfile already strips leading "/" and ".." components, but
        # this endpoint opens URLs/data URIs from external callers so we
        # belt-and-suspenders reject: (a) entries that resolve outside
        # extract_root after symlink expansion and (b) symlink entries that
        # extractall would otherwise materialize on disk (0xA000 = S_IFLNK).
        for info in infos:
            if (info.external_attr >> 16) & 0xF000 == 0xA000:
                raise ValueError(f"ZIP bundle contains symlink entry: {info.filename}")
            resolved = (extract_root / info.filename).resolve()
            if extract_root not in resolved.parents and resolved != extract_root:
                raise ValueError(
                    f"ZIP bundle contains unsafe entry path: {info.filename}"
                )
        zf.extractall(extract_dir)

    # Walk everything once and filter by a lowercased suffix so producers
    # that mint entries like ``MAIN.USDA`` are still recognized on
    # case-sensitive filesystems (Linux). ``Path.rglob(f"*{ext}")`` would
    # otherwise be case-sensitive and drop those archives.
    usd_files = sorted(
        p
        for p in Path(extract_dir).rglob("*")
        if p.is_file() and not p.is_symlink() and p.suffix.lower() in _USD_EXTENSIONS
    )
    if not usd_files:
        raise ValueError(
            f"No USD layer found in ZIP bundle (expected one of {_USD_EXTENSIONS})"
        )

    if prefer_first_usd:
        for info in infos:
            if Path(info.filename).suffix.lower() not in _USD_EXTENSIONS:
                continue
            first_usd = Path(extract_dir) / info.filename
            if first_usd.is_file() and not first_usd.is_symlink():
                return str(first_usd)

    for preferred in ("main", "scene", "stage"):
        for p in usd_files:
            if p.stem.lower() == preferred:
                return str(p)
    return str(usd_files[0])


def _fetch_usd(url: str, dest_path: str) -> None:
    """Fetch USD file from URL, data URI, or S3 to a local path."""
    from urllib.parse import urlparse

    if url.startswith("data:"):
        # data:application/octet-stream;base64,<data>
        parts = url.split(",", 1)
        if len(parts) != 2 or not parts[1]:
            raise ValueError(
                f"Malformed data URI: missing comma or payload: {url[:60]}"
            )
        try:
            raw = base64.b64decode(parts[1], validate=True)
        except (ValueError, binascii.Error) as e:
            raise ValueError(f"Malformed data URI: invalid base64: {e}") from e
        with open(dest_path, "wb") as f:
            f.write(raw)
        return

    scheme = urlparse(url).scheme
    if scheme == "s3":
        _download_s3(url, dest_path)
        return
    # Accept http/https here because material-agent-service and internal
    # tools call this service over the in-cluster plain-text network (no
    # TLS terminator between the pods). External callers are expected to
    # use https or s3://.
    _allowed_http_schemes = {"https", "http"}
    if scheme in _allowed_http_schemes:
        # Check if this is an S3 HTTPS URL (e.g. bucket.s3.region.amazonaws.com/key)
        s3_url = _https_to_s3(url)
        if s3_url:
            _download_s3(s3_url, dest_path)
            return
        # Block requests to cloud metadata and private/loopback addresses
        _validate_url_target(url)
        resp = requests.get(url, timeout=300, allow_redirects=False)
        resp.raise_for_status()
        with open(dest_path, "wb") as f:
            f.write(resp.content)
        return

    raise ValueError(f"Unsupported URL scheme: {url[:50]}")


def _https_to_s3(url: str) -> str | None:
    """Convert an S3 HTTPS URL to s3:// URI, or None if not an S3 URL."""
    m = _S3_VHOST_RE.match(url) or _S3_PATH_RE.match(url)
    if m:
        return f"s3://{m.group('bucket')}/{m.group('key')}"
    return None


def _download_s3(s3_url: str, dest_path: str) -> None:
    """Download from s3://bucket/key to a local path."""
    import boto3
    from botocore.exceptions import ClientError, ProfileNotFound

    parts = s3_url[5:].split("/", 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ""
    # Try AWS_PROFILE env var, then default credential chain
    profile = os.environ.get("AWS_PROFILE")
    if profile:
        session = boto3.Session(profile_name=profile)
    else:
        # Fall back to the "default" profile if it exists; otherwise let
        # boto3 use its standard credential chain (env vars, instance role).
        session = None
        try:
            candidate = boto3.Session(profile_name="default")
        except ProfileNotFound as e:
            logger.debug("S3 profile 'default' not found: %s", e)
        else:
            try:
                candidate.client("s3").head_bucket(Bucket=bucket)
            except ClientError as e:
                logger.debug(
                    "S3 profile 'default' cannot access bucket %s: %s", bucket, e
                )
            else:
                logger.debug("S3 profile 'default' resolved for bucket %s", bucket)
                session = candidate
        if session is None:
            logger.debug(
                "No named S3 profile worked for bucket %s, using default "
                "credential chain",
                bucket,
            )
            session = boto3.Session()
    s3 = session.client("s3")
    logger.info("Downloading s3://%s/%s", bucket, key)
    s3.download_file(bucket, key, dest_path)


def _to_v1_response(
    result: dict[str, Any],
    requested_sensors: list[str],
    ovrtx_sensors: list[str],
    frame_start: int,
) -> dict[str, Any]:
    """Convert OvRTXRenderingBackend output to V1 response format.

    V1 format: images[frame_str][camera_path][sensor_name] = base64 string

    ``frame_start`` is added to the 0-based enumerate index so that the
    ``v1_images`` keys are the *absolute* frame numbers the client asked
    for (not 0,1,2,...). Sensor lookups are keyed on the same absolute
    frame number because OVRTX returns sensor dicts keyed on the real
    frame index. Getting this wrong is silent — the main RGB image is
    keyed 0-based so looks correct, but sensor data comes back empty for
    any render where frame_start != 0.
    """
    v1_images: dict[str, dict[str, dict[str, str]]] = {}

    for cam_result in result.get("results", []):
        camera = cam_result["camera"]
        images: list[Image.Image] = cam_result.get("images", [])
        sensor_data: dict[str, dict[int, np.ndarray]] = cam_result.get("sensors", {})

        for frame_idx, img in enumerate(images):
            actual_frame = frame_start + frame_idx
            frame_str = str(actual_frame)
            if frame_str not in v1_images:
                v1_images[frame_str] = {}

            camera_data: dict[str, str] = {}

            # Main RGB image -> base64 PNG
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            camera_data["images"] = base64.b64encode(buf.getvalue()).decode()

            # Sensor data -> base64 raw bytes
            for req_sensor in requested_sensors:
                ovrtx_name = _SENSOR_KIT_TO_OVRTX.get(req_sensor)
                if ovrtx_name and ovrtx_name in sensor_data:
                    arr = sensor_data[ovrtx_name].get(actual_frame)
                    if arr is not None:
                        camera_data[req_sensor] = base64.b64encode(
                            arr.tobytes()
                        ).decode()
                    else:
                        camera_data[req_sensor] = ""
                else:
                    # Unsupported sensor -- return empty string
                    camera_data[req_sensor] = ""

            v1_images[frame_str][camera] = camera_data

    return {
        "status": "success",
        "error": None,
        "images": v1_images,
    }


def _error_response(message: str) -> dict[str, Any]:
    """Build an error response dict."""
    return {
        "status": "exception",
        "error": message,
        "images": {},
    }
