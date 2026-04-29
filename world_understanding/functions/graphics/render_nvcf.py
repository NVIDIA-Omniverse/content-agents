# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""USD rendering functions using NVIDIA Cloud Functions (NVCF) microservice."""

import io
import json
import logging
import os
import random
import tempfile
import time
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import requests
from PIL import Image
from requests.exceptions import ConnectionError, HTTPError, RequestException, Timeout

from world_understanding.config.s3 import WU_S3_BUCKET, WU_S3_PROFILE, WU_S3_REGION

if TYPE_CHECKING:
    from pxr import Usd

from world_understanding.utils.image_utils import (
    base64_to_image,
    base64_to_numpy,
    process_depth_map,
)
from world_understanding.utils.nvcf_utils import (
    create_nvcf_headers,
    get_base_url,
    get_nvcf_api_key,
    s3_uri_to_https_url,
)
from world_understanding.utils.s3_utils import delete_s3_path, upload_file_to_s3
from world_understanding.utils.usd.material import (
    get_local_mdl_assets,
    get_local_texture_file_assets,
)
from world_understanding.utils.usd.stage import create_data_uri_from_file

logger = logging.getLogger(__name__)


class RenderingStatus(StrEnum):
    """Status codes returned by NVCF rendering service."""

    empty_response = "empty_response"
    load_error = "load_error"
    exception = "exception"
    success = "success"


# Note: decode_base64_to_image and decode_base64_to_numpy have been moved to
# world_understanding.utils.image_utils for reusability

# Sensor name mapping from V2 to V1 format
_V2_SENSOR_TO_V1 = {
    "rgb": "images",
    "distance_to_image_plane": "linear_depth",
    "distance_to_camera": "linear_depth",
    "instance_segmentation": "instance_id_segmentation",
}


def _is_v2_response(result: dict[str, Any]) -> bool:
    """Check if a render response uses the V2 format."""
    return "rendered_data" in result and "total_cameras" in result


def _convert_v2_sensor(sensor_obj: dict[str, Any]) -> str | np.ndarray:
    """Convert a V2 sensor object {type, data, shape, dtype} to an array or passthrough string.

    Returns:
        np.ndarray if shape/dtype metadata is present (decoded raw array data).
        str (the original base64 string) if shape is missing or data is empty.
    """
    import base64

    data_b64 = sensor_obj.get("data", "")
    shape = sensor_obj.get("shape")
    dtype_str = sensor_obj.get("dtype", "uint8")

    if not data_b64 or not shape:
        return data_b64

    # Decode raw array data
    raw_bytes = base64.b64decode(data_b64)
    arr = np.frombuffer(raw_bytes, dtype=np.dtype(dtype_str)).reshape(shape)

    return arr


def _convert_v2_to_v1(result: dict[str, Any]) -> dict[str, Any]:
    """Convert a V2 render response to V1 format for backward compatibility.

    V2 structure: rendered_data[camera][frame] = {rgb: {type, data, shape, dtype}}
    V1 structure: images[frame][camera] = {images: base64_png, sensor: base64_raw}
    """
    import base64 as b64mod

    rendered_data = result.get("rendered_data", {})
    v1_images: dict[str, dict[str, dict[str, Any]]] = {}

    for camera_key, frames in rendered_data.items():
        for frame_str, sensors_dict in frames.items():
            if frame_str not in v1_images:
                v1_images[frame_str] = {}

            camera_data: dict[str, Any] = {}
            for sensor_name, sensor_obj in sensors_dict.items():
                v1_name = _V2_SENSOR_TO_V1.get(sensor_name, sensor_name)

                if not isinstance(sensor_obj, dict) or "data" not in sensor_obj:
                    camera_data[v1_name] = sensor_obj
                    continue

                arr = _convert_v2_sensor(sensor_obj)
                if isinstance(arr, np.ndarray) and v1_name == "images":
                    # Convert to PNG base64 for the main image
                    img = Image.fromarray(arr.astype(np.uint8))
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    camera_data[v1_name] = b64mod.b64encode(buf.getvalue()).decode()
                elif isinstance(arr, np.ndarray):
                    # Raw sensor data as base64 numpy
                    camera_data[v1_name] = b64mod.b64encode(arr.tobytes()).decode()
                else:
                    camera_data[v1_name] = arr

            v1_images[frame_str][camera_key] = camera_data

    logger.info(
        "Converted V2 response to V1 format: %d cameras, %d frames",
        result.get("total_cameras", 0),
        result.get("total_frames", 0),
    )

    return {
        "images": v1_images,
        "status": RenderingStatus.success,
        "error": None,
    }


def _export_stage_and_get_url(
    stage_path: str,
    use_data_uri: bool,
    s3_bucket: str,
    s3_profile: str | None,
    s3_region: str,
) -> tuple[str, str | None]:
    """Export USD stage and return URL and optional S3 URI for cleanup.

    Args:
        stage_path: Path to the USD stage
        use_data_uri: If True, use data URI encoding instead of S3 upload.
        s3_bucket: S3 bucket for stage upload (ignored if use_data_uri=True).
        s3_profile: AWS profile for S3 upload (ignored if use_data_uri=True).
        s3_region: AWS region where the S3 bucket is located (ignored if use_data_uri=True).

    Returns:
        Tuple containing the asset URL and optional S3 URI for cleanup.
    """
    if use_data_uri:
        asset_url = create_data_uri_from_file(stage_path)
        logger.info("Created data URI for stage")
        return asset_url, None
    else:
        unique_id = uuid.uuid4().hex
        # Preserve original file extension so renderers can detect format
        ext = Path(stage_path).suffix or ".usd"
        s3_key = f"nvcf-renders/{unique_id}/stage{ext}"
        s3_uri = upload_file_to_s3(
            file_path=stage_path,
            s3_path=f"s3://{s3_bucket}/{s3_key}",
            profile_name=s3_profile,
        )
        asset_url = s3_uri_to_https_url(s3_uri, s3_region)
        logger.info("Uploaded stage to S3: %s", asset_url)
        return asset_url, s3_uri


def _bundle_stage_with_local_assets(
    stage: "Usd.Stage",
    temp_dir: Path,
    base_dir: str | Path | None = None,
) -> tuple[Path | None, bool]:
    """Bundle USD stage with local MDL and texture assets into a ZIP archive.

    This function checks if the stage references any local MDL files or texture
    files (PNG, JPG, EXR, etc. from UsdPreviewSurface shaders). If so, it:
    1. Creates a directory structure with the USD, MDL, and texture files
    2. Updates asset paths in the USD to be relative
    3. Creates a ZIP archive containing everything

    Args:
        stage: USD stage to bundle
        temp_dir: Temporary directory for creating the bundle
        base_dir: Base directory for resolving relative texture paths. If None,
                 uses the stage's root layer directory.

    Returns:
        Tuple of (zip_path, was_bundled):
            - zip_path: Path to the created ZIP file, or None if no bundling needed
            - was_bundled: True if bundling occurred, False otherwise
    """
    import shutil

    from pxr import Sdf

    # Get all MDL assets from the stage
    mdl_assets = get_local_mdl_assets(stage, base_dir=base_dir)

    # Filter to only local, existing files
    local_assets = [a for a in mdl_assets if a["is_local"] and a["resolved_path"]]

    # Get all texture file assets from the stage
    texture_assets = get_local_texture_file_assets(stage, base_dir=base_dir)
    local_textures = [a for a in texture_assets if a["is_local"] and a["resolved_path"]]

    if not local_assets and not local_textures:
        logger.info("No local MDL or texture assets found, skipping bundling")
        return None, False

    logger.info(
        f"Found {len(local_assets)} local MDL assets and "
        f"{len(local_textures)} local texture files to bundle"
    )

    # Create bundle directory structure
    bundle_dir = temp_dir / "bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    # Track copied directories to avoid duplicates (for MDL)
    copied_dirs: dict[str, str] = {}  # original_dir -> relative_path_in_bundle

    # ---- Copy MDL files and their directories ----
    if local_assets:
        mdl_dir = bundle_dir / "mdl_materials"
        mdl_dir.mkdir(parents=True, exist_ok=True)

        for asset in local_assets:
            mdl_file = Path(asset["resolved_path"])
            mdl_parent = mdl_file.parent

            # Use directory name as unique identifier
            dir_name = mdl_parent.name

            if str(mdl_parent) not in copied_dirs:
                # Copy entire directory to preserve textures
                dest_dir = mdl_dir / dir_name

                # Handle duplicate directory names
                counter = 1
                while dest_dir.exists():
                    dest_dir = mdl_dir / f"{dir_name}_{counter}"
                    counter += 1

                try:
                    shutil.copytree(mdl_parent, dest_dir)
                    copied_dirs[str(mdl_parent)] = str(dest_dir.relative_to(bundle_dir))
                    logger.debug(f"Copied MDL directory: {mdl_parent} -> {dest_dir}")
                except Exception as e:
                    logger.warning(f"Failed to copy MDL directory {mdl_parent}: {e}")
                    continue

    # ---- Copy texture files ----
    # Track: resolved_path -> relative_path_in_bundle
    copied_textures: dict[str, str] = {}

    if local_textures:
        textures_dir = bundle_dir / "textures"
        textures_dir.mkdir(parents=True, exist_ok=True)

        seen_filenames: dict[str, int] = {}  # filename -> counter for collisions

        for tex in local_textures:
            resolved = tex["resolved_path"]
            if resolved in copied_textures:
                continue

            src_path = Path(resolved)
            filename = src_path.name

            # Handle filename collisions
            if filename in seen_filenames:
                seen_filenames[filename] += 1
                stem = src_path.stem
                suffix = src_path.suffix
                filename = f"{stem}_{seen_filenames[filename]}{suffix}"
            else:
                seen_filenames[filename] = 0

            dest_path = textures_dir / filename
            try:
                shutil.copy2(str(src_path), str(dest_path))
                rel_path = str(dest_path.relative_to(bundle_dir))
                copied_textures[resolved] = rel_path
                logger.debug(f"Copied texture: {src_path} -> {dest_path}")
            except Exception as e:
                logger.warning(f"Failed to copy texture {src_path}: {e}")

    if not copied_dirs and not copied_textures:
        logger.warning("No assets were copied, skipping bundling")
        return None, False

    # Export the stage and update paths
    root_layer = stage.GetRootLayer()

    # Create a copy of the layer to modify
    temp_usda = bundle_dir / "stage.usda"
    root_layer.Export(str(temp_usda))

    # Reopen the exported layer to update paths
    exported_layer = Sdf.Layer.FindOrOpen(str(temp_usda))
    if not exported_layer:
        logger.error("Failed to open exported layer for path updates")
        return None, False

    # Update asset paths in the exported layer
    def update_asset_paths_in_layer(layer: Sdf.Layer) -> int:
        """Recursively update MDL and texture asset paths in a layer."""
        updated_count = 0

        def process_prim_spec(prim_spec):
            nonlocal updated_count

            for attr_name in list(prim_spec.attributes.keys()):
                attr_spec = prim_spec.attributes[attr_name]
                value = attr_spec.default

                if value is None:
                    continue

                # Only process Sdf.AssetPath values
                if not isinstance(value, Sdf.AssetPath):
                    continue

                try:
                    asset_path = value.path if hasattr(value, "path") else str(value)
                except Exception:
                    continue

                if not asset_path:
                    continue

                # --- MDL path rewriting ---
                if attr_name == "info:mdl:sourceAsset":
                    for orig_dir, rel_bundle_path in copied_dirs.items():
                        if asset_path.startswith(orig_dir) or (
                            os.path.isabs(asset_path)
                            and str(Path(asset_path).parent) == orig_dir
                        ):
                            mdl_filename = Path(asset_path).name
                            new_path = f"{rel_bundle_path}/{mdl_filename}"
                            attr_spec.default = Sdf.AssetPath(new_path)
                            updated_count += 1
                            logger.debug(
                                f"Updated MDL path: {asset_path} -> {new_path}"
                            )
                            break
                    continue

                # --- Texture path rewriting ---
                if not copied_textures:
                    continue

                # Match by filename against copied textures
                asset_filename = Path(asset_path).name
                for resolved, rel_bundle_path in copied_textures.items():
                    if Path(resolved).name == asset_filename:
                        attr_spec.default = Sdf.AssetPath(rel_bundle_path)
                        updated_count += 1
                        logger.debug(
                            f"Updated texture path: {asset_path} -> {rel_bundle_path}"
                        )
                        break

            # Process child prims
            for child in prim_spec.nameChildren:
                process_prim_spec(child)

        for prim in layer.rootPrims:
            process_prim_spec(prim)

        return updated_count

    updated_count = update_asset_paths_in_layer(exported_layer)
    logger.info(f"Updated {updated_count} asset paths to relative paths")

    # Save the modified layer
    exported_layer.Save()

    # Create ZIP archive
    zip_path = temp_dir / "bundle.zip"

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in bundle_dir.rglob("*"):
            if file_path.is_file():
                arc_name = file_path.relative_to(bundle_dir)
                zf.write(file_path, arc_name)
                logger.debug(f"Added to ZIP: {arc_name}")

    zip_size = zip_path.stat().st_size
    logger.info(f"Created asset bundle: {zip_path} ({zip_size / 1024:.1f} KB)")

    return zip_path, True


def export_stage_to_s3(
    stage: "Usd.Stage",
    s3_bucket: str = WU_S3_BUCKET,
    s3_region: str = WU_S3_REGION,
    s3_profile: str = WU_S3_PROFILE,
    use_data_uri: bool = False,
    bundle_mdl_assets: bool = True,
    base_dir: str | Path | None = None,
) -> tuple[str, str | None]:
    """Export and upload a USD stage to S3, returning URL and S3 URI for cleanup.

    This function is useful for batch rendering workflows where the same stage
    needs to be rendered multiple times. Instead of uploading the stage repeatedly,
    upload it once and reuse the URL.

    If bundle_mdl_assets is True and the stage contains local MDL materials,
    the function will bundle the USD with all referenced MDL files into a ZIP
    archive and upload that instead.

    Args:
        stage: A Usd.Stage object from pxr package
        s3_bucket: S3 bucket for stage upload (ignored if use_data_uri=True).
                  Default: WU_S3_BUCKET env var (required for S3 mode).
        s3_region: AWS region (ignored if use_data_uri=True).
                  Default: WU_S3_REGION env var or "us-east-2"
        s3_profile: AWS profile for S3 upload (ignored if use_data_uri=True).
                   Default: WU_S3_PROFILE env var (required for S3 mode).
        use_data_uri: If True, use data URI encoding instead of S3 upload.
                     This embeds the USD file as base64 in the request.
                     Default: False
        bundle_mdl_assets: If True, attempt to bundle local MDL and texture assets
                          with the USD file into a ZIP archive. If no local assets
                          are found, falls back to uploading just the USD.
                          Default: True
        base_dir: Base directory for resolving relative texture paths. If None,
                 uses the stage's root layer directory.

    Returns:
        Tuple containing:
            - asset_url: The URL to access the stage (HTTPS URL or data URI)
            - s3_uri: The S3 URI for cleanup (None if use_data_uri=True)

    Example:
        >>> from pxr import Usd
        >>> stage = Usd.Stage.CreateInMemory()
        >>> # ... build scene ...
        >>> # Upload once
        >>> url, s3_uri = export_stage_to_s3(stage, s3_bucket="my-bucket")
        >>> # Use the URL for multiple render calls
        >>> result1 = render_single_camera_from_url(url, camera="/Camera1")
        >>> result2 = render_single_camera_from_url(url, camera="/Camera2")
        >>> # Clean up when done
        >>> if s3_uri:
        ...     delete_s3_path(s3_uri, profile_name="your-aws-profile")
    """
    # Try bundling MDL assets if requested
    temp_dir = None
    zip_path = None
    was_bundled = False

    if bundle_mdl_assets and not use_data_uri:
        try:
            temp_dir = Path(tempfile.mkdtemp(prefix="nvcf_bundle_"))
            zip_path, was_bundled = _bundle_stage_with_local_assets(
                stage, temp_dir, base_dir=base_dir
            )
        except Exception as e:
            logger.warning(f"MDL bundling failed, falling back to USD-only: {e}")
            was_bundled = False

    if was_bundled and zip_path:
        # Upload the ZIP bundle
        try:
            unique_id = uuid.uuid4().hex
            s3_key = f"nvcf-renders/{unique_id}/bundle.zip"
            s3_uri = upload_file_to_s3(
                file_path=str(zip_path),
                s3_path=f"s3://{s3_bucket}/{s3_key}",
                profile_name=s3_profile,
            )
            asset_url = s3_uri_to_https_url(s3_uri, s3_region)
            logger.info("Uploaded MDL bundle to S3: %s", asset_url)
            return asset_url, s3_uri
        finally:
            # Clean up temp directory
            if temp_dir and temp_dir.exists():
                try:
                    import shutil

                    shutil.rmtree(temp_dir)
                except Exception:
                    pass
    else:
        # Clean up temp directory if bundling was attempted but didn't produce a bundle
        if temp_dir and temp_dir.exists():
            try:
                import shutil

                shutil.rmtree(temp_dir)
            except Exception:
                pass

    # Fall back to original behavior: export USD only
    with tempfile.NamedTemporaryFile(suffix=".usdc", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        if not stage.GetRootLayer().Export(tmp_path):
            raise RuntimeError("Failed to export USD stage")

        asset_url, s3_uri = _export_stage_and_get_url(
            stage_path=tmp_path,
            use_data_uri=use_data_uri,
            s3_bucket=s3_bucket,
            s3_profile=s3_profile,
            s3_region=s3_region,
        )
        return asset_url, s3_uri
    finally:
        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def _parse_zip_response(zip_content: bytes) -> dict | None:
    """
    Parse a ZIP response from the rendering server.

    The server sometimes returns a ZIP file containing a single JSON file ({uuid}.response)
    with the complete rendering result, including base64-encoded image data.
    This function extracts and parses that JSON file.

    Args:
        zip_content: Raw ZIP file content as bytes

    Returns:
        dict | None: Parsed result with structure:
            {
                "images": {frame_num: {camera_path: {sensor_name: base64_data}}},
                "status": "success",
                "error": null
            }
            Returns None if parsing fails.
    """
    # Open ZIP from memory
    with zipfile.ZipFile(io.BytesIO(zip_content)) as zip_file:
        # List all files in the ZIP
        file_list = zip_file.namelist()
        logger.info("ZIP contains files: %s", file_list)

        # Find the .response file (contains JSON with base64-encoded data)
        response_file = None
        for filename in file_list:
            if filename.endswith(".response"):
                response_file = filename
                break

        if response_file is None:
            logger.error("No .response file found in ZIP. Files: %s", file_list)
            raise ValueError("No .response file found in ZIP")

        # Extract and parse the JSON response
        logger.info("Extracting response file: %s", response_file)
        with zip_file.open(response_file) as f:
            response_text = f.read().decode("utf-8")
            result = json.loads(response_text)

        # The result is already in the correct format:
        # {"images": {frame: {camera_path: {sensor: base64_data}}}, "status": "success", "error": null}
        logger.info(
            "Successfully parsed ZIP response: status=%s, frames=%d",
            result.get("status", "unknown"),
            len(result.get("images", {})),
        )
        return result


def render_single_camera(
    stage: "Usd.Stage",
    camera: str,
    image_width: int = 1024,
    image_height: int = 1024,
    frames: str = "0",
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: int = 3600,
    sensors: list[str] | None = None,
    apply_background_mask: bool = False,
    use_data_uri: bool = False,
    s3_bucket: str = WU_S3_BUCKET,
    s3_region: str = WU_S3_REGION,
    s3_profile: str = WU_S3_PROFILE,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    retry_backoff_factor: float = 2.0,
    retry_jitter: float = 0.1,
) -> dict[str, Any]:
    """
    Render a single camera view from an in-memory USD Stage using NVCF.

    This function exports the USD stage and uses the NVCF microservice to render
    images. It supports two modes: S3 upload (default) or data URI encoding.
    It returns PIL Image objects and optionally sensor data like depth and normals.

    Args:
        stage: A Usd.Stage object from pxr package
        camera: Camera path to use for rendering (e.g., "/Camera", "/World/Camera")
        image_width: Image width in pixels. Default: 1024
        image_height: Image height in pixels. Default: 1024
        frames: Frame(s) to render. Can be:
               - Single frame: "0", "42"
               - Frame range: "0:10", "5:15"
               Default: "0" (first frame)
        api_key: NVCF API key. If None, uses NGC_API_KEY env var
        base_url: NVCF base URL. If None, uses RENDER_ENDPOINT env var (or NVCF_RENDER_FUNCTION_ID fallback)
        timeout: Request timeout in seconds. Default: 3600
        sensors: Additional sensors to render (e.g., ["linear_depth", "instance_id_segmentation"])
        apply_background_mask: If True, apply background masking during rendering. Default: False
        use_data_uri: If True, use data URI encoding instead of S3 upload.
                     This embeds the USD file as base64 in the request.
                     Default: False
        s3_bucket: S3 bucket for stage upload (ignored if use_data_uri=True).
                  Default: WU_S3_BUCKET env var (required for S3 mode).
        s3_region: AWS region (ignored if use_data_uri=True).
                  Default: WU_S3_REGION env var or "us-east-2"
        s3_profile: AWS profile for S3 upload (ignored if use_data_uri=True).
                   Default: WU_S3_PROFILE env var (required for S3 mode).
        max_retries: Maximum number of retry attempts. Default: 3
        retry_delay: Initial delay between retries in seconds. Default: 1.0
        retry_backoff_factor: Factor to multiply delay by after each retry. Default: 2.0
        retry_jitter: Random jitter factor (0-1) to add to delays. Default: 0.1

    Returns:
        Dict containing:
            - camera: Camera path used for rendering
            - images: List of PIL Image objects (ordered by frame)
            - sensors: Dict of sensor_name -> frame_num -> numpy arrays (if sensors requested)
            - render_time: Total rendering time in seconds
            - frame_count: Number of frames rendered
            - status: Rendering status (success, load_error, etc.)
            - error: Error message if rendering failed (optional)

    Raises:
        ValueError: If input parameters are invalid
        RuntimeError: If rendering fails

    Example:
        >>> from pxr import Usd, UsdGeom
        >>> stage = Usd.Stage.CreateInMemory()
        >>> # ... build your USD scene ...
        >>> # Using S3 (default)
        >>> result = render_single_camera(
        ...     stage=stage,
        ...     camera="/Camera",
        ...     image_width=1920,
        ...     image_height=1080,
        ...     s3_bucket="my-bucket",
        ...     s3_region="us-west-2"
        ... )
        >>> # Using data URI (no S3 needed)
        >>> result = render_single_camera(
        ...     stage=stage,
        ...     camera="/Camera",
        ...     use_data_uri=True
        ... )
        >>> for i, img in enumerate(result['images']):
        ...     img.save(f"frame_{i}.png")
    """
    try:
        from pxr import Usd
    except ImportError as e:
        raise RuntimeError(
            "USD-Core is required for this function. Install with: pip install usd-core"
        ) from e

    if not isinstance(stage, Usd.Stage):
        raise ValueError(f"stage must be a Usd.Stage object. Got: {type(stage)}")

    # Export stage to temporary file (using binary format for smaller disk footprint)
    with tempfile.NamedTemporaryFile(suffix=".usdc", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        if not stage.GetRootLayer().Export(tmp_path):
            raise RuntimeError("Failed to export USD stage")

        asset_url, s3_uri = _export_stage_and_get_url(
            stage_path=tmp_path,
            use_data_uri=use_data_uri,
            s3_bucket=s3_bucket,
            s3_profile=s3_profile,
            s3_region=s3_region,
        )
    finally:
        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    # Render using the URL and clean up S3 file afterwards
    try:
        result = render_single_camera_from_url(
            usd_url=asset_url,
            camera=camera,
            image_width=image_width,
            image_height=image_height,
            frames=frames,
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            sensors=sensors,
            apply_background_mask=apply_background_mask,
            max_retries=max_retries,
            retry_delay=retry_delay,
            retry_backoff_factor=retry_backoff_factor,
            retry_jitter=retry_jitter,
        )
        return result
    finally:
        # Clean up S3 file if it was used
        if s3_uri:
            try:
                delete_s3_path(s3_uri, profile_name=s3_profile)
                logger.info("Cleaned up S3 file: %s", s3_uri)
            except Exception as e:
                logger.warning("Failed to clean up S3 file %s: %s", s3_uri, e)


def render_single_camera_from_url(
    usd_url: str,
    camera: str,
    image_width: int = 1024,
    image_height: int = 1024,
    frames: str = "0",
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: int = 3600,
    sensors: list[str] | None = None,
    apply_background_mask: bool = False,
    force_render: bool = True,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    retry_backoff_factor: float = 2.0,
    retry_jitter: float = 0.1,
) -> dict[str, Any]:
    """
    Render a single camera view from a USD file URL using NVCF.

    This function uses the NVCF microservice to render images from USD scenes
    accessible via URL (HTTP/HTTPS or S3). It supports single frames or frame
    ranges and can render additional sensor data.

    Args:
        usd_url: URL to the USD file (HTTP/HTTPS or S3 URL)
        camera: Camera path to use for rendering (e.g., "/Camera", "/World/Camera")
        image_width: Image width in pixels. Default: 1024
        image_height: Image height in pixels. Default: 1024
        frames: Frame(s) to render. Can be:
               - Single frame: "0", "42"
               - Frame range: "0:10", "5:15"
               Default: "0" (first frame)
        api_key: NVCF API key. If None, uses NGC_API_KEY env var
        base_url: NVCF base URL. If None, uses RENDER_ENDPOINT env var (or NVCF_RENDER_FUNCTION_ID fallback)
        timeout: Request timeout in seconds. Default: 3600
        sensors: Additional sensors to render (e.g., ["linear_depth", "instance_id_segmentation"])
        apply_background_mask: If True, apply background masking during rendering. Default: False
        force_render: Force re-rendering even if cached. Default: True
        max_retries: Maximum number of retry attempts. Default: 3
        retry_delay: Initial delay between retries in seconds. Default: 1.0
        retry_backoff_factor: Factor to multiply delay by after each retry. Default: 2.0
        retry_jitter: Random jitter factor (0-1) to add to delays. Default: 0.1

    Returns:
        Dict containing:
            - camera: Camera path used for rendering
            - images: List of PIL Image objects (ordered by frame)
            - sensors: Dict of sensor_name -> frame_num -> numpy arrays (if sensors requested)
            - render_time: Total rendering time in seconds
            - frame_count: Number of frames rendered
            - status: Rendering status (success, load_error, etc.)
            - error: Error message if rendering failed (optional)

    Raises:
        ValueError: If input parameters are invalid
        RuntimeError: If rendering fails

    Example:
        >>> result = render_single_camera_from_url(
        ...     usd_url="https://example.com/scene.usd",
        ...     camera="/Camera",
        ...     image_width=1920,
        ...     image_height=1080,
        ...     frames="0:10",
        ...     sensors=["linear_depth", "instance_id_segmentation"]
        ... )
        >>> print(f"Rendered {result['frame_count']} frames")
        >>> # Access sensor data
        >>> depth_data = result['sensors']['linear_depth'][0]  # Frame 0 depth
        >>> seg_data = result['sensors']['instance_id_segmentation'][0]  # Frame 0 segmentation
    """
    # Get API key and base URL using common utilities
    api_key = get_nvcf_api_key(api_key)
    base_url = get_base_url(base_url, "RENDER_ENDPOINT", "NVCF_RENDER_FUNCTION_ID")

    # Construct full URL with render endpoint
    full_url = f"{base_url.rstrip('/')}/render"

    # Parse frames parameter
    if ":" in frames:
        # Frame range
        start_str, end_str = frames.split(":")
        frame_start = int(start_str)
        frame_end = int(end_str)
    else:
        # Single frame
        frame_num = int(frames)
        frame_start = frame_num
        frame_end = frame_num

    # Build request parameters
    params = {
        "url": usd_url,
        "force_render": force_render,
        "render_settings": {
            "camera_paths": [camera],
            "frame_range": {"start": frame_start, "end": frame_end},
            "camera_parameters": {
                "width": image_width,
                "height": image_height,
            },
            "sensors": sensors,
            "apply_background_mask": apply_background_mask,
        },
    }

    # Create headers using common utility
    headers = create_nvcf_headers(api_key, timeout)

    # Truncate URL for logging (data URIs can be very long)
    logger.info(
        "Rendering camera %s with frames %s from %s", camera, frames, usd_url[:100]
    )
    start_time = time.time()

    # Retry logic for NVCF request
    last_error = None
    current_delay = retry_delay

    for attempt in range(max_retries + 1):
        try:
            if attempt > 0:
                # Add jitter to prevent thundering herd
                jittered_delay = current_delay * (
                    1 + random.uniform(-retry_jitter, retry_jitter)
                )
                logger.info(
                    "Retrying NVCF request (attempt %d/%d) after %.2fs delay",
                    attempt + 1,
                    max_retries + 1,
                    jittered_delay,
                )
                time.sleep(jittered_delay)
                current_delay *= retry_backoff_factor

            response = requests.post(
                full_url,
                headers=headers,
                json=params,
                timeout=timeout + 10,
                allow_redirects=True,
            )
            response.raise_for_status()

            # Check content type to handle both JSON and ZIP responses
            content_type = response.headers.get("Content-Type", "")

            if "application/json" in content_type:
                result = response.json()
            elif "application/zip" in content_type:
                logger.info(
                    "Received ZIP response for %s. Processing directly...",
                    usd_url[:100],
                )
                # Read the ZIP content
                zip_content = response.content

                # Parse ZIP and convert to expected result format
                result = _parse_zip_response(zip_content)
                if result is None:
                    logger.error("Failed to parse ZIP response for %s", usd_url[:100])
                    raise ValueError("Failed to parse ZIP response")
            else:
                logger.error(
                    "Unexpected content type '%s' for %s",
                    content_type,
                    usd_url[:100],
                )
                raise ValueError("Unexpected content type")

            # Success - break out of retry loop
            break

        except (ConnectionError, Timeout) as e:
            # Network errors - retry
            last_error = e
            logger.warning(
                "NVCF request attempt %d failed with network error: %s",
                attempt + 1,
                str(e),
            )
            if attempt == max_retries:
                error_msg = f"NVCF request failed after {max_retries + 1} attempts: {str(last_error)}"
                logger.error(error_msg)
                return {
                    "camera": camera,
                    "images": [],
                    "sensors": {},
                    "render_time": time.time() - start_time,
                    "frame_count": 0,
                    "status": RenderingStatus.exception,
                    "error": error_msg,
                }

        except HTTPError as e:
            # HTTP errors - check if we should retry
            # Retryable status codes:
            # - 408: Request Timeout (client timeout, can retry)
            # - 429: Too Many Requests (rate limiting, should retry with backoff)
            # - 502, 503, 504: Gateway/service errors (server issues)
            retryable_codes = [408, 429, 502, 503, 504]

            if e.response.status_code in retryable_codes:
                last_error = e
                logger.warning(
                    "NVCF request attempt %d failed with HTTP %d: %s",
                    attempt + 1,
                    e.response.status_code,
                    str(e),
                )
                if attempt == max_retries:
                    error_msg = f"NVCF request failed after {max_retries + 1} attempts: HTTP {e.response.status_code}"
                    logger.error(error_msg)
                    return {
                        "camera": camera,
                        "images": [],
                        "sensors": {},
                        "render_time": time.time() - start_time,
                        "frame_count": 0,
                        "status": RenderingStatus.exception,
                        "error": error_msg,
                    }
            else:
                # Non-retryable HTTP error (400, 401, 403, 404, etc.)
                error_msg = f"NVCF request failed with non-retryable HTTP {e.response.status_code}: {str(e)}"
                logger.error(
                    "Non-retryable error: HTTP %d. Will not retry. Error: %s",
                    e.response.status_code,
                    str(e),
                )
                return {
                    "camera": camera,
                    "images": [],
                    "sensors": {},
                    "render_time": time.time() - start_time,
                    "frame_count": 0,
                    "status": RenderingStatus.exception,
                    "error": error_msg,
                }

        except RequestException as e:
            # Other request exceptions - don't retry
            error_msg = f"NVCF request failed with non-retryable error: {str(e)}"
            logger.error(
                "Non-retryable request exception. Will not retry. Error: %s", str(e)
            )
            return {
                "camera": camera,
                "images": [],
                "sensors": {},
                "render_time": time.time() - start_time,
                "frame_count": 0,
                "status": RenderingStatus.exception,
                "error": error_msg,
            }

    render_time = time.time() - start_time
    logger.info("NVCF request completed in %.2fs", render_time)

    # Convert V2 response to V1 format if needed
    if _is_v2_response(result):
        result = _convert_v2_to_v1(result)

    # Check status
    status = result.get("status", RenderingStatus.exception)
    if status != RenderingStatus.success:
        error_msg = f"Rendering failed with status: {status}"
        logger.error(error_msg)
        return {
            "camera": camera,
            "images": [],
            "sensors": {},
            "render_time": render_time,
            "frame_count": 0,
            "status": status,
            "error": error_msg,
        }

    # Process results - convert dict to list for compatibility
    images = []
    sensor_data = {sensor: {} for sensor in (sensors or [])}

    # Sort frames by frame number to maintain order
    frame_items = sorted(result.get("images", {}).items(), key=lambda x: int(x[0]))
    for frame_num, frame_data in frame_items:
        frame_num_int = int(frame_num)

        # Get camera data (should only be one camera)
        for _camera_path, camera_data in frame_data.items():
            # Process main image
            if "images" in camera_data:
                try:
                    img = base64_to_image(camera_data["images"])
                    images.append(img)
                except Exception as e:
                    logger.warning(
                        "Failed to decode image for frame %s: %s", frame_num, e
                    )

            # Process sensor data
            for sensor_name in sensors or []:
                if sensor_name in camera_data:
                    try:
                        # Determine dtype based on sensor type
                        if sensor_name == "instance_id_segmentation":
                            # Segmentation uses uint32 for instance IDs (not uint8!)
                            # Using uint8 causes 4x data size and stride issues
                            dtype = np.uint32
                        else:
                            dtype = np.float32

                        data = base64_to_numpy(camera_data[sensor_name], dtype=dtype)
                        sensor_data[sensor_name][frame_num_int] = data
                    except Exception as e:
                        logger.warning(
                            "Failed to decode %s for frame %s: %s",
                            sensor_name,
                            frame_num,
                            e,
                        )

    frame_count = len(images)
    logger.info("Successfully rendered %s frames for camera %s", frame_count, camera)

    return {
        "camera": camera,
        "images": images,
        "sensors": sensor_data,
        "render_time": render_time,
        "frame_count": frame_count,
        "status": status,
    }


def render_all_cameras(
    stage: "Usd.Stage",
    image_width: int = 1024,
    image_height: int = 1024,
    cameras: list[str] | None = None,
    frames: str = "0",
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: int = 3600,
    sensors: list[str] | None = None,
    apply_background_mask: bool = False,
    use_data_uri: bool = False,
    s3_bucket: str = WU_S3_BUCKET,
    s3_region: str = WU_S3_REGION,
    s3_profile: str = WU_S3_PROFILE,
    max_workers: int = 8,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    retry_backoff_factor: float = 2.0,
    retry_jitter: float = 0.1,
    bundle_mdl_assets: bool = True,
) -> dict[str, Any]:
    """
    Render multiple cameras from an in-memory USD Stage using NVCF.

    This function renders multiple camera views from a USD Stage object.
    It supports two modes: S3 upload (default) or data URI encoding.
    If no cameras are specified, it uses a default camera named "/Camera".
    Multiple cameras can be rendered in parallel using threading.

    Args:
        stage: A Usd.Stage object from pxr package
        image_width: Image width in pixels. Default: 1024
        image_height: Image height in pixels. Default: 1024
        cameras: List of camera paths to render. If None, uses ["/Camera"]
        frames: Frame(s) to render. Default: "0"
        api_key: NVCF API key. If None, uses NGC_API_KEY env var
        base_url: NVCF base URL. If None, uses RENDER_ENDPOINT env var (or NVCF_RENDER_FUNCTION_ID fallback)
        timeout: Request timeout in seconds. Default: 3600
        sensors: Additional sensors to render (e.g., ["linear_depth", "instance_id_segmentation"])
        apply_background_mask: If True, apply background masking during rendering. Default: False
        use_data_uri: If True, use data URI encoding instead of S3 upload.
                     This embeds the USD file as base64 in the request.
                     Default: False
        s3_bucket: S3 bucket for stage upload (ignored if use_data_uri=True).
                  Default: WU_S3_BUCKET env var (required for S3 mode).
        s3_region: AWS region (ignored if use_data_uri=True).
                  Default: WU_S3_REGION env var or "us-east-2"
        s3_profile: AWS profile for S3 upload (ignored if use_data_uri=True).
                   Default: WU_S3_PROFILE env var (required for S3 mode).
        max_workers: Maximum number of parallel render threads. Default: 8
        max_retries: Maximum number of retry attempts. Default: 3
        retry_delay: Initial delay between retries in seconds. Default: 1.0
        retry_backoff_factor: Factor to multiply delay by after each retry. Default: 2.0
        retry_jitter: Random jitter factor (0-1) to add to delays. Default: 0.1
        bundle_mdl_assets: If True, attempt to bundle local MDL and texture assets
                          with the USD file into a ZIP archive. If no local assets
                          are found, falls back to uploading just the USD.
                          Default: True

    Returns:
        Dict containing:
            - total_cameras: Number of cameras to render
            - successful_cameras: Number of successfully rendered cameras
            - failed_cameras: Number of failed camera renders
            - total_render_time: Total time for all renders in seconds
            - results: List of individual camera render results

    Example:
        >>> from pxr import Usd
        >>> stage = Usd.Stage.CreateInMemory()
        >>> # ... build scene ...
        >>> # Using S3 (default)
        >>> result = render_all_cameras(
        ...     stage=stage,
        ...     image_width=1920,
        ...     image_height=1080,
        ...     cameras=["/Camera1", "/Camera2"],
        ...     s3_bucket="my-bucket",
        ...     s3_region="us-west-2"
        ... )
        >>> # Using data URI (no S3 needed)
        >>> result = render_all_cameras(
        ...     stage=stage,
        ...     cameras=["/Camera1", "/Camera2"],
        ...     use_data_uri=True
        ... )
        >>> print(f"Rendered {result['successful_cameras']} cameras")
    """
    # Default camera if none specified
    if cameras is None or len(cameras) == 0:
        cameras = ["/Camera"]

    results = []
    successful_cameras = 0
    failed_cameras = 0
    total_start_time = time.time()

    # Export and upload stage once for all cameras
    # If bundle_mdl_assets is True, will attempt to bundle local MDL files
    asset_url, s3_uri = export_stage_to_s3(
        stage=stage,
        s3_bucket=s3_bucket,
        s3_region=s3_region,
        s3_profile=s3_profile,
        use_data_uri=use_data_uri,
        bundle_mdl_assets=bundle_mdl_assets,
    )

    # Render cameras and clean up S3 file afterwards
    try:
        # Render cameras in parallel if max_workers > 1
        if max_workers > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        render_single_camera_from_url,
                        usd_url=asset_url,
                        camera=camera,
                        image_width=image_width,
                        image_height=image_height,
                        frames=frames,
                        api_key=api_key,
                        base_url=base_url,
                        timeout=timeout,
                        sensors=sensors,
                        apply_background_mask=apply_background_mask,
                        max_retries=max_retries,
                        retry_delay=retry_delay,
                        retry_backoff_factor=retry_backoff_factor,
                        retry_jitter=retry_jitter,
                    ): camera
                    for camera in cameras
                }

                for future in as_completed(futures):
                    camera = futures[future]
                    try:
                        result = future.result()
                        results.append(result)
                        if result.get("status") == RenderingStatus.success:
                            successful_cameras += 1
                        else:
                            failed_cameras += 1
                    except Exception as e:
                        failed_cameras += 1
                        error_result = {
                            "camera": camera,
                            "images": [],
                            "sensors": {},
                            "render_time": 0.0,
                            "frame_count": 0,
                            "status": RenderingStatus.exception,
                            "error": str(e),
                        }
                        results.append(error_result)
                        logger.exception("Failed to render camera %s", camera)
        else:
            # Sequential rendering
            for camera in cameras:
                try:
                    result = render_single_camera_from_url(
                        usd_url=asset_url,
                        camera=camera,
                        image_width=image_width,
                        image_height=image_height,
                        frames=frames,
                        api_key=api_key,
                        base_url=base_url,
                        timeout=timeout,
                        sensors=sensors,
                        apply_background_mask=apply_background_mask,
                        max_retries=max_retries,
                        retry_delay=retry_delay,
                        retry_backoff_factor=retry_backoff_factor,
                        retry_jitter=retry_jitter,
                    )
                    results.append(result)
                    if result.get("status") == RenderingStatus.success:
                        successful_cameras += 1
                    else:
                        failed_cameras += 1
                except Exception as e:
                    failed_cameras += 1
                    error_result = {
                        "camera": camera,
                        "images": [],
                        "sensors": {},
                        "render_time": 0.0,
                        "frame_count": 0,
                        "status": RenderingStatus.exception,
                        "error": str(e),
                    }
                    results.append(error_result)
                    logger.exception("Failed to render camera %s", camera)

        total_render_time = time.time() - total_start_time

        return {
            "total_cameras": len(cameras),
            "successful_cameras": successful_cameras,
            "failed_cameras": failed_cameras,
            "total_render_time": total_render_time,
            "results": results,
        }
    finally:
        # Clean up S3 file if it was used
        if s3_uri:
            try:
                delete_s3_path(s3_uri, profile_name=s3_profile)
                logger.info("Cleaned up S3 file: %s", s3_uri)
            except Exception as e:
                logger.warning("Failed to clean up S3 file %s: %s", s3_uri, e)


def render_all_cameras_from_url(
    usd_url: str,
    image_width: int = 1024,
    image_height: int = 1024,
    cameras: list[str] | None = None,
    frames: str = "0",
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: int = 3600,
    sensors: list[str] | None = None,
    apply_background_mask: bool = False,
    max_workers: int = 1,
) -> dict[str, Any]:
    """
    Render multiple cameras from a USD file URL using NVCF.

    This function renders multiple camera views from a USD file accessible
    via URL. Multiple cameras can be rendered in parallel using threading.

    Args:
        usd_url: URL to the USD file (HTTP/HTTPS or S3 URL)
        image_width: Image width in pixels. Default: 1024
        image_height: Image height in pixels. Default: 1024
        cameras: List of camera paths to render. If None, uses ["/Camera"]
        frames: Frame(s) to render. Default: "0"
        api_key: NVCF API key. If None, uses NGC_API_KEY env var
        base_url: NVCF base URL. If None, uses RENDER_ENDPOINT env var (or NVCF_RENDER_FUNCTION_ID fallback)
        timeout: Request timeout in seconds. Default: 3600
        sensors: Additional sensors to render (e.g., ["linear_depth", "instance_id_segmentation"])
        apply_background_mask: If True, apply background masking during rendering. Default: False
        max_workers: Maximum number of parallel render threads. Default: 1

    Returns:
        Dict containing:
            - total_cameras: Number of cameras to render
            - successful_cameras: Number of successfully rendered cameras
            - failed_cameras: Number of failed camera renders
            - total_render_time: Total time for all renders in seconds
            - results: List of individual camera render results

    Example:
        >>> result = render_all_cameras_from_url(
        ...     usd_url="https://example.com/scene.usd",
        ...     image_width=1920,
        ...     image_height=1080,
        ...     cameras=["/Camera1", "/Camera2"],
        ...     sensors=["depth"],
        ...     max_workers=2
        ... )
        >>> for cam_result in result['results']:
        ...     if cam_result['status'] == 'success':
        ...         print(f"Camera {cam_result['camera']}: {cam_result['frame_count']} frames")
    """
    # Default camera if none specified
    if cameras is None or len(cameras) == 0:
        cameras = ["/Camera"]

    results = []
    successful_cameras = 0
    failed_cameras = 0
    total_start_time = time.time()

    # Render cameras in parallel if max_workers > 1
    if max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    render_single_camera_from_url,
                    usd_url=usd_url,
                    camera=camera,
                    image_width=image_width,
                    image_height=image_height,
                    frames=frames,
                    api_key=api_key,
                    base_url=base_url,
                    timeout=timeout,
                    sensors=sensors,
                    apply_background_mask=apply_background_mask,
                ): camera
                for camera in cameras
            }

            for future in as_completed(futures):
                camera = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    if result.get("status") == RenderingStatus.success:
                        successful_cameras += 1
                    else:
                        failed_cameras += 1
                except Exception as e:
                    failed_cameras += 1
                    error_result = {
                        "camera": camera,
                        "images": [],
                        "sensors": {},
                        "render_time": 0.0,
                        "frame_count": 0,
                        "status": RenderingStatus.exception,
                        "error": str(e),
                    }
                    results.append(error_result)
                    logger.exception("Failed to render camera %s", camera)
    else:
        # Sequential rendering
        for camera in cameras:
            try:
                result = render_single_camera_from_url(
                    usd_url=usd_url,
                    camera=camera,
                    image_width=image_width,
                    image_height=image_height,
                    frames=frames,
                    api_key=api_key,
                    base_url=base_url,
                    timeout=timeout,
                    sensors=sensors,
                    apply_background_mask=apply_background_mask,
                )
                results.append(result)
                if result.get("status") == RenderingStatus.success:
                    successful_cameras += 1
                else:
                    failed_cameras += 1
            except Exception as e:
                failed_cameras += 1
                error_result = {
                    "camera": camera,
                    "images": [],
                    "sensors": {},
                    "render_time": 0.0,
                    "frame_count": 0,
                    "status": RenderingStatus.exception,
                    "error": str(e),
                }
                results.append(error_result)
                logger.exception("Failed to render camera %s", camera)

    total_render_time = time.time() - total_start_time

    return {
        "total_cameras": len(cameras),
        "successful_cameras": successful_cameras,
        "failed_cameras": failed_cameras,
        "total_render_time": total_render_time,
        "results": results,
    }


def batch_render_assets(
    asset_urls: list[str],
    cameras: list[str] | None = None,
    image_width: int = 1024,
    image_height: int = 1024,
    frames: str = "0",
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: int = 3600,
    sensors: list[str] | None = None,
    apply_background_mask: bool = False,
    max_workers: int = 32,
) -> dict[str, Any]:
    """
    Batch render multiple USD assets with specified cameras using NVCF.

    This function efficiently renders multiple USD files, each with multiple
    cameras, using parallel processing. It's optimized for high-throughput
    rendering of many assets.

    Args:
        asset_urls: List of USD file URLs to render
        cameras: List of camera paths to render for each asset
        image_width: Image width in pixels. Default: 1024
        image_height: Image height in pixels. Default: 1024
        frames: Frame(s) to render. Default: "0"
        api_key: NVCF API key. If None, uses NGC_API_KEY env var
        base_url: NVCF base URL. If None, uses RENDER_ENDPOINT env var (or NVCF_RENDER_FUNCTION_ID fallback)
        timeout: Request timeout in seconds. Default: 3600
        sensors: Additional sensors to render (e.g., ["linear_depth", "instance_id_segmentation"])
        apply_background_mask: If True, apply background masking during rendering. Default: False
        max_workers: Maximum number of parallel render threads. Default: 32

    Returns:
        Dict containing:
            - total_assets: Number of assets processed
            - successful_assets: Number of successfully rendered assets
            - failed_assets: Number of failed asset renders
            - total_render_time: Total time for all renders in seconds
            - asset_results: Dict mapping asset URL to render results

    Example:
        >>> asset_urls = [
        ...     "https://example.com/asset1.usd",
        ...     "https://example.com/asset2.usd",
        ...     "https://example.com/asset3.usd"
        ... ]
        >>> result = batch_render_assets(
        ...     asset_urls=asset_urls,
        ...     cameras=["/Camera"],
        ...     image_width=1920,
        ...     image_height=1080,
        ...     max_workers=8
        ... )
        >>> print(f"Rendered {result['successful_assets']}/{result['total_assets']} assets")
    """
    if not asset_urls:
        raise ValueError("No asset URLs provided")

    # Default camera if none specified
    if cameras is None or len(cameras) == 0:
        cameras = ["/Camera"]

    asset_results = {}
    successful_assets = 0
    failed_assets = 0
    total_start_time = time.time()

    logger.info(
        "Starting batch render of %s assets with %s workers",
        len(asset_urls),
        max_workers,
    )

    # Create tasks for all asset-camera combinations
    render_tasks = []
    for asset_url in asset_urls:
        for camera in cameras:
            render_tasks.append((asset_url, camera))

    # Process all tasks in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                render_single_camera_from_url,
                usd_url=asset_url,
                camera=camera,
                image_width=image_width,
                image_height=image_height,
                frames=frames,
                api_key=api_key,
                base_url=base_url,
                timeout=timeout,
                sensors=sensors,
                apply_background_mask=apply_background_mask,
            ): (asset_url, camera)
            for asset_url, camera in render_tasks
        }

        # Collect results
        for future in as_completed(futures):
            asset_url, camera = futures[future]

            # Initialize asset results if needed
            if asset_url not in asset_results:
                asset_results[asset_url] = {
                    "cameras": {},
                    "successful_cameras": 0,
                    "failed_cameras": 0,
                }

            try:
                result = future.result()
                asset_results[asset_url]["cameras"][camera] = result

                if result.get("status") == RenderingStatus.success:
                    asset_results[asset_url]["successful_cameras"] += 1
                else:
                    asset_results[asset_url]["failed_cameras"] += 1

            except Exception as e:
                asset_results[asset_url]["cameras"][camera] = {
                    "camera": camera,
                    "images": [],
                    "sensors": {},
                    "render_time": 0.0,
                    "frame_count": 0,
                    "status": RenderingStatus.exception,
                    "error": str(e),
                }
                asset_results[asset_url]["failed_cameras"] += 1
                # Truncate URL for logging (data URIs can be very long)
                logger.exception(
                    "Failed to render %s camera %s", asset_url[:100], camera
                )

    # Count successful assets (all cameras rendered successfully)
    for _asset_url, results in asset_results.items():
        if results["failed_cameras"] == 0:
            successful_assets += 1
        else:
            failed_assets += 1

    total_render_time = time.time() - total_start_time

    logger.info(
        "Batch render completed in %.2fs: %s/%s assets successful",
        total_render_time,
        successful_assets,
        len(asset_urls),
    )

    return {
        "total_assets": len(asset_urls),
        "successful_assets": successful_assets,
        "failed_assets": failed_assets,
        "total_render_time": total_render_time,
        "asset_results": asset_results,
    }


def save_render_results(
    result: dict[str, Any],
    output_dir: Path | str,
    file_name: str = "render",
    image_width: int = 1024,
    image_height: int = 1024,
    save_npy: bool = False,
) -> dict[str, int]:
    """Save render results to disk with proper processing for different sensor types.

    This function saves rendered images and sensor data to disk, handling different
    sensor types appropriately:
    - images: saved as PNG
    - instance_id_segmentation: saved as NPY (uint8) and PNG
    - depth/linear_depth: saved as NPY (float32) and processed PNG
    - other sensors: saved as NPY (float32)

    Args:
        result: Render result dictionary from NVCF render functions
        output_dir: Directory to save the output files
        file_name: Base name for output files
        image_width: Width of the rendered images
        image_height: Height of the rendered images
        save_npy: If True, save sensor data as NPY files. Default: False
    Returns:
        dict: Dictionary with counts:
            - total_count: Total number of files saved
            - success_count: Number of successfully saved files
            - error_count: Number of failed saves

    Example:
        >>> result = render_single_camera(
        ...     stage=stage,
        ...     camera="/Camera",
        ...     frames="0:2",
        ...     sensors=["depth", "instance_id_segmentation"]
        ... )
        >>> stats = save_render_results(
        ...     result=result,
        ...     output_dir="output",
        ...     file_name="scene",
        ...     image_width=1024,
        ...     image_height=1024
        ... )
        >>> print(f"Saved {stats['success_count']}/{stats['total_count']} files")
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_count = 0
    success_count = 0
    error_count = 0

    # Save main images
    if "images" in result and result["images"]:
        for frame_idx, img in enumerate(result["images"]):
            total_count += 1
            try:
                output_path = output_dir / f"{file_name}_f{frame_idx:04d}_images.png"
                img.save(output_path)
                success_count += 1
            except Exception as e:
                logger.warning("Failed to save image for frame %d: %s", frame_idx, e)
                error_count += 1

    # Save sensor data
    if "sensors" in result and result["sensors"]:
        for sensor_name, frame_data in result["sensors"].items():
            for frame_num, data in frame_data.items():
                total_count += 1
                try:
                    # Reshape data to proper dimensions
                    data = data.reshape(image_height, image_width, -1)

                    # Save as NPY
                    npy_path = (
                        output_dir / f"{file_name}_f{frame_num:04d}_{sensor_name}.npy"
                    )
                    if save_npy:
                        np.save(npy_path, data)

                    # Apply depth processing for depth sensors
                    # Note: linear_depth (distance_to_image_plane) is the standard Z-depth for computer vision
                    # depth (distance_to_camera) is radial distance from camera center
                    png_path = npy_path.with_suffix(".png")
                    if sensor_name in ("depth", "linear_depth"):
                        depth_map = process_depth_map(data)
                        depth_map = Image.fromarray(
                            (depth_map[..., 0] * 255.0).astype(np.uint8)
                        )
                        depth_map.save(png_path)
                    elif sensor_name == "instance_id_segmentation":
                        Image.fromarray(data.squeeze().astype(np.uint8)).convert(
                            "RGB"
                        ).save(png_path)

                    success_count += 1
                except Exception as e:
                    logger.warning(
                        "Failed to save %s for frame %d: %s",
                        sensor_name,
                        frame_num,
                        e,
                    )
                    error_count += 1

    logger.info(
        "Saved %d/%d files to %s (%d errors)",
        success_count,
        total_count,
        output_dir,
        error_count,
    )

    return {
        "total_count": total_count,
        "success_count": success_count,
        "error_count": error_count,
    }
