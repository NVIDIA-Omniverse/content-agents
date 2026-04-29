# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Async USD rendering functions using NVIDIA Cloud Functions (NVCF) microservice.

Provides async multi-camera rendering support, building on the existing
execute_nvcf_request_async() infrastructure in nvcf_utils.py.
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from world_understanding.functions.graphics.render_nvcf import (
    RenderingStatus,
    _convert_v2_to_v1,
    _is_v2_response,
)
from world_understanding.utils.image_utils import (
    base64_to_image,
    base64_to_numpy,
)
from world_understanding.utils.nvcf_utils import (
    create_nvcf_headers,
    execute_nvcf_request_async,
    get_base_url,
    get_nvcf_api_key,
)

logger = logging.getLogger(__name__)

MAX_RENDERINGS_PER_BATCH = 1024


def validate_batch_size(
    batch_size: int, num_cameras: int, num_sensors: int = 1
) -> None:
    """Validate that total renderings do not exceed the maximum batch size.

    Args:
        batch_size: Number of frames in the batch.
        num_cameras: Number of cameras to render.
        num_sensors: Number of sensors per camera (default 1 for the main image).

    Raises:
        ValueError: If total renderings exceed MAX_RENDERINGS_PER_BATCH.
    """
    total = batch_size * num_cameras * num_sensors
    if total > MAX_RENDERINGS_PER_BATCH:
        raise ValueError(
            f"Batch too large: {total} renderings "
            f"({batch_size} frames x {num_cameras} cameras x {num_sensors} sensors) "
            f"exceeds maximum of {MAX_RENDERINGS_PER_BATCH}"
        )


async def render_cameras_from_url(
    usd_url: str,
    cameras: list[str],
    image_width: int = 1024,
    image_height: int = 1024,
    frames: str = "0",
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: int = 3600,
    sensors: list[str] | None = None,
    apply_background_mask: bool = False,
    poll_seconds: int = 300,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    retry_backoff_factor: float = 2.0,
    semaphore: asyncio.Semaphore | None = None,
) -> dict[str, Any]:
    """Render all cameras in a single async NVCF request.

    Sends all cameras in one request to the NVCF rendering service and parses
    the multi-camera response. Uses execute_nvcf_request_async() which handles
    httpx.AsyncClient, 202 polling, and retries.

    Args:
        usd_url: URL to the USD file (HTTP/HTTPS or S3 URL).
        cameras: List of camera paths to render.
        image_width: Image width in pixels. Default: 1024.
        image_height: Image height in pixels. Default: 1024.
        frames: Frame(s) to render ("0", "0:10"). Default: "0".
        api_key: NVCF API key. If None, uses NGC_API_KEY env var.
        base_url: NVCF base URL. If None, uses NVCF_RENDER_FUNCTION_ID env var.
        timeout: Request timeout in seconds. Default: 3600.
        sensors: Additional sensors to render (e.g., ["linear_depth"]).
        apply_background_mask: Apply background masking. Default: False.
        poll_seconds: NVCF long-polling timeout in seconds. Default: 300.
        max_retries: Maximum retry attempts. Default: 3.
        retry_delay: Initial retry delay in seconds. Default: 1.0.
        retry_backoff_factor: Backoff multiplier per retry. Default: 2.0.
        semaphore: Optional semaphore to limit concurrent requests.

    Returns:
        Dict matching render_all_cameras_from_url format:
            - total_cameras: Number of cameras requested
            - successful_cameras: Number successfully rendered
            - failed_cameras: Number that failed
            - total_render_time: Elapsed time in seconds
            - results: List of per-camera result dicts
    """
    # Parse frames parameter
    if ":" in frames:
        start_str, end_str = frames.split(":")
        frame_start = int(start_str)
        frame_end = int(end_str)
    else:
        frame_start = int(frames)
        frame_end = frame_start

    # Validate batch size before making the request
    num_frames = frame_end - frame_start + 1
    num_sensors = len(sensors) + 1 if sensors else 1  # +1 for main image
    validate_batch_size(num_frames, len(cameras), num_sensors)

    api_key = get_nvcf_api_key(api_key)
    base_url = get_base_url(base_url, "RENDER_ENDPOINT", "NVCF_RENDER_FUNCTION_ID")
    full_url = f"{base_url.rstrip('/')}/render"

    # Build request parameters with all cameras in one request
    params: dict[str, Any] = {
        "url": usd_url,
        "force_render": True,
        "render_settings": {
            "camera_paths": cameras,
            "frame_range": {"start": frame_start, "end": frame_end},
            "camera_parameters": {
                "width": image_width,
                "height": image_height,
            },
            "sensors": sensors,
            "apply_background_mask": apply_background_mask,
        },
    }

    headers = create_nvcf_headers(api_key, timeout, poll_seconds=poll_seconds)

    logger.info(
        "Rendering %d cameras with frames %s from %s",
        len(cameras),
        frames,
        usd_url[:100],
    )
    start_time = time.time()

    try:
        if semaphore is not None:
            async with semaphore:
                result = await execute_nvcf_request_async(
                    url=full_url,
                    headers=headers,
                    params=params,
                    api_key=api_key,
                    timeout=timeout,
                    max_retries=max_retries,
                    retry_delay=retry_delay,
                    retry_backoff_factor=retry_backoff_factor,
                )
        else:
            result = await execute_nvcf_request_async(
                url=full_url,
                headers=headers,
                params=params,
                api_key=api_key,
                timeout=timeout,
                max_retries=max_retries,
                retry_delay=retry_delay,
                retry_backoff_factor=retry_backoff_factor,
            )
    except RuntimeError as e:
        elapsed = time.time() - start_time
        logger.error("NVCF request failed after %.2fs: %s", elapsed, e)
        error_results = []
        for camera in cameras:
            error_results.append(
                {
                    "camera": camera,
                    "images": [],
                    "sensors": {},
                    "render_time": elapsed,
                    "frame_count": 0,
                    "status": RenderingStatus.exception,
                    "error": str(e),
                }
            )
        return {
            "total_cameras": len(cameras),
            "successful_cameras": 0,
            "failed_cameras": len(cameras),
            "total_render_time": elapsed,
            "results": error_results,
        }

    render_time = time.time() - start_time
    logger.info("NVCF request completed in %.2fs", render_time)

    # Convert V2 response to V1 format if needed
    if _is_v2_response(result):
        result = _convert_v2_to_v1(result)

    # Check overall status
    status = result.get("status", RenderingStatus.exception)
    if status != RenderingStatus.success:
        error_msg = f"Rendering failed with status: {status}"
        logger.error(error_msg)
        error_results = []
        for camera in cameras:
            error_results.append(
                {
                    "camera": camera,
                    "images": [],
                    "sensors": {},
                    "render_time": render_time,
                    "frame_count": 0,
                    "status": status,
                    "error": error_msg,
                }
            )
        return {
            "total_cameras": len(cameras),
            "successful_cameras": 0,
            "failed_cameras": len(cameras),
            "total_render_time": render_time,
            "results": error_results,
        }

    # Parse multi-camera response
    # Response structure: result["images"][frame_num_str][camera_key]["images"]
    # Iterate response keys like sync version (render_nvcf.py:938) to avoid
    # camera path format mismatches between what we sent and what server returns.
    camera_results: list[dict[str, Any]] = []
    successful_cameras = 0
    failed_cameras = 0

    # First pass: collect all data by iterating response keys (like sync version).
    # Structure: per_camera_data[response_camera_key] = {images: [], sensors: {}}
    per_camera_data: dict[str, dict[str, Any]] = {}

    frame_items = sorted(result.get("images", {}).items(), key=lambda x: int(x[0]))

    # Log response camera keys from first frame for debugging
    if frame_items:
        first_frame_keys = list(frame_items[0][1].keys())
        logger.debug(
            "Response camera keys: %s, input cameras: %s",
            first_frame_keys,
            cameras,
        )

    for frame_num_str, frame_data in frame_items:
        frame_num_int = int(frame_num_str)

        # Iterate ALL camera keys in this frame (like sync render_nvcf.py:938)
        for response_camera_key, camera_data in frame_data.items():
            if response_camera_key not in per_camera_data:
                per_camera_data[response_camera_key] = {
                    "images": [],
                    "sensors": {s: {} for s in (sensors or [])},
                }
            cam_store = per_camera_data[response_camera_key]

            # Process main image
            if "images" in camera_data:
                try:
                    img = base64_to_image(camera_data["images"])
                    cam_store["images"].append(img)
                except Exception as e:
                    logger.warning(
                        "Failed to decode image for camera %s frame %s: %s",
                        response_camera_key,
                        frame_num_str,
                        e,
                    )

            # Process sensor data
            for sensor_name in sensors or []:
                if sensor_name in camera_data:
                    try:
                        dtype: np.dtype[Any] = (
                            np.dtype(np.uint32)
                            if sensor_name == "instance_id_segmentation"
                            else np.dtype(np.float32)
                        )
                        data = base64_to_numpy(camera_data[sensor_name], dtype=dtype)
                        cam_store["sensors"][sensor_name][frame_num_int] = data
                    except Exception as e:
                        logger.warning(
                            "Failed to decode %s for camera %s frame %s: %s",
                            sensor_name,
                            response_camera_key,
                            frame_num_str,
                            e,
                        )

    # Second pass: map response camera keys back to our input camera list.
    # Build mapping: input camera path -> response camera key
    response_keys = list(per_camera_data.keys())
    input_to_response: dict[str, str] = {}

    for input_cam in cameras:
        # Try exact match first
        if input_cam in per_camera_data:
            input_to_response[input_cam] = input_cam
            continue
        # Try stripping leading "/" from both sides
        input_stripped = input_cam.lstrip("/")
        matched = False
        for rk in response_keys:
            if rk.lstrip("/") == input_stripped:
                input_to_response[input_cam] = rk
                matched = True
                break
        if not matched:
            # Try matching by camera name (last path component)
            input_name = input_cam.rsplit("/", 1)[-1]
            for rk in response_keys:
                rk_name = rk.rsplit("/", 1)[-1]
                if input_name == rk_name:
                    input_to_response[input_cam] = rk
                    break

    if input_to_response:
        logger.debug("Camera path mapping: %s", input_to_response)

    # Build results in input camera order
    for camera_path in cameras:
        response_key = input_to_response.get(camera_path)

        if response_key and response_key in per_camera_data:
            cam_store = per_camera_data[response_key]
            camera_results.append(
                {
                    "camera": camera_path,
                    "images": cam_store["images"],
                    "sensors": cam_store["sensors"],
                    "render_time": render_time,
                    "frame_count": len(cam_store["images"]),
                    "status": RenderingStatus.success,
                }
            )
            successful_cameras += 1
            logger.info(
                "Camera %s: %d frames parsed", camera_path, len(cam_store["images"])
            )
        else:
            failed_cameras += 1
            camera_results.append(
                {
                    "camera": camera_path,
                    "images": [],
                    "sensors": {},
                    "render_time": render_time,
                    "frame_count": 0,
                    "status": RenderingStatus.exception,
                    "error": f"No response data for camera (tried key mapping from {response_keys})",
                }
            )
            logger.warning(
                "No response data for camera %s. Response keys: %s",
                camera_path,
                response_keys,
            )

    return {
        "total_cameras": len(cameras),
        "successful_cameras": successful_cameras,
        "failed_cameras": failed_cameras,
        "total_render_time": render_time,
        "results": camera_results,
    }


async def render_composition_from_url(
    highlight_url: str,
    plain_url: str,
    cameras: list[str],
    image_width: int = 1024,
    image_height: int = 1024,
    frames: str = "0",
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: int = 3600,
    single_camera_per_request: bool = True,
    sensors: list[str] | None = None,
    apply_background_mask: bool = False,
    poll_seconds: int = 300,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    retry_backoff_factor: float = 2.0,
    semaphore: asyncio.Semaphore | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Render highlight and plain USD compositions concurrently.

    Renders both the highlighted and plain versions of a scene in parallel.

    Args:
        highlight_url: URL to the highlighted USD file.
        plain_url: URL to the plain USD file.
        cameras: List of camera paths to render.
        image_width: Image width in pixels. Default: 1024.
        image_height: Image height in pixels. Default: 1024.
        frames: Frame(s) to render. Default: "0".
        api_key: NVCF API key. If None, uses NGC_API_KEY env var.
        base_url: NVCF base URL. If None, uses env var.
        timeout: Request timeout in seconds. Default: 3600.
        single_camera_per_request: If True, send 1 camera per NVCF request
            to work around a server-side bug where multi-camera requests
            produce incorrect visibility keyframe evaluation for later cameras.
            Set to False when the server is fixed. Default: True.
        sensors: Additional sensors to render.
        apply_background_mask: Apply background masking. Default: False.
        poll_seconds: NVCF long-polling timeout. Default: 300.
        max_retries: Maximum retry attempts. Default: 3.
        retry_delay: Initial retry delay in seconds. Default: 1.0.
        retry_backoff_factor: Backoff multiplier per retry. Default: 2.0.
        semaphore: Optional semaphore to limit concurrent requests.

    Returns:
        Tuple of (highlight_result, plain_result), each matching the
        render_cameras_from_url() return format.
    """
    common_kwargs: dict[str, Any] = {
        "image_width": image_width,
        "image_height": image_height,
        "frames": frames,
        "api_key": api_key,
        "base_url": base_url,
        "timeout": timeout,
        "sensors": sensors,
        "apply_background_mask": apply_background_mask,
        "poll_seconds": poll_seconds,
        "max_retries": max_retries,
        "retry_delay": retry_delay,
        "retry_backoff_factor": retry_backoff_factor,
        "semaphore": semaphore,
    }

    if single_camera_per_request and len(cameras) > 1:
        # Send 1 camera per request to avoid server-side rendering artifacts
        # when switching cameras within a single NVCF instance. Each camera
        # gets a fresh scene load, ensuring correct visibility/material state.
        # TODO: Set single_camera_per_request=False when server bug is fixed.
        coros = []
        for camera in cameras:
            coros.append(
                render_cameras_from_url(
                    usd_url=highlight_url, cameras=[camera], **common_kwargs
                )
            )
            coros.append(
                render_cameras_from_url(
                    usd_url=plain_url, cameras=[camera], **common_kwargs
                )
            )

        results = await asyncio.gather(*coros)

        # Merge per-camera results back into highlight/plain dicts
        # coros order: [highlight_cam0, plain_cam0, highlight_cam1, plain_cam1, ...]
        def _merge_camera_results(
            single_results: list[dict[str, Any]],
        ) -> dict[str, Any]:
            merged: dict[str, Any] = {
                "total_cameras": len(single_results),
                "successful_cameras": 0,
                "failed_cameras": 0,
                "total_render_time": 0.0,
                "results": [],
            }
            for r in single_results:
                merged["successful_cameras"] += r.get("successful_cameras", 0)
                merged["failed_cameras"] += r.get("failed_cameras", 0)
                merged["total_render_time"] = max(
                    merged["total_render_time"], r.get("total_render_time", 0.0)
                )
                merged["results"].extend(r.get("results", []))
            return merged

        highlight_singles = [results[i] for i in range(0, len(results), 2)]
        plain_singles = [results[i] for i in range(1, len(results), 2)]

        highlight_result = _merge_camera_results(highlight_singles)
        plain_result = _merge_camera_results(plain_singles)
    else:
        # Multi-camera: send all cameras in one request per stage (faster
        # but requires server to correctly evaluate visibility keyframes
        # across camera switches)
        highlight_result, plain_result = await asyncio.gather(
            render_cameras_from_url(
                usd_url=highlight_url, cameras=cameras, **common_kwargs
            ),
            render_cameras_from_url(
                usd_url=plain_url, cameras=cameras, **common_kwargs
            ),
        )

    return highlight_result, plain_result


def save_images_parallel(
    save_tasks: list[tuple[Image.Image, Path | str]],
    max_workers: int = 12,
) -> int:
    """Save multiple images in parallel using a thread pool.

    Args:
        save_tasks: List of (image, path) tuples to save.
        max_workers: Maximum number of concurrent save threads. Default: 12.

    Returns:
        Number of images successfully saved.
    """
    if not save_tasks:
        return 0

    def _save_one(task: tuple[Image.Image, Path | str]) -> bool:
        img, path = task
        try:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            img.save(path)
            return True
        except Exception as e:
            logger.warning("Failed to save image to %s: %s", path, e)
            return False

    saved = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_save_one, task) for task in save_tasks]
        for future in futures:
            try:
                if future.result():
                    saved += 1
            except Exception as e:
                logger.warning("Save task raised exception: %s", e)

    return saved
