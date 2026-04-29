# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Portable function for Grounding DINO object detection."""

import json
import mimetypes
import os
import shutil
import tempfile
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import requests
from PIL import Image

# NVIDIA API configuration
NVAI_URL = "https://ai.api.nvidia.com/v1/cv/nvidia/nv-grounding-dino"
NVAI_POLLING_URL = "https://api.nvcf.nvidia.com/v2/nvcf/pexec/status/"
UPLOAD_ASSET_TIMEOUT = 300  # Timeout (in secs) to upload asset
MAX_RETRIES = 5  # Max num of retries while polling
DELAY_BTW_RETRIES = 1  # adding 1s delay between each polls


class BoundingBox(TypedDict):
    """Bounding box detection result."""

    phrase: str
    bboxes: list[list[int]]  # [[x, y, width, height], ...]
    confidence: list[float]


def _upload_asset(
    input_data: bytes, description: str, content_type: str, api_key: str
) -> uuid.UUID:
    """Upload an asset to NVIDIA API.

    Args:
        input_data: Raw bytes of the asset to upload.
        description: Description metadata for the asset.
        content_type: MIME type of the asset.
        api_key: NVIDIA API key used for authorization.
    """
    assets_url = "https://api.nvcf.nvidia.com/v2/nvcf/assets"
    if not api_key:
        raise RuntimeError("NVIDIA API key is required.")
    header_auth = f"Bearer {api_key}"
    headers = {
        "Authorization": header_auth,
        "Content-Type": "application/json",
        "accept": "application/json",
    }

    s3_headers = {
        "x-amz-meta-nvcf-asset-description": description,
        "content-type": content_type,
    }

    payload = {"contentType": content_type, "description": description}

    response = requests.post(assets_url, headers=headers, json=payload, timeout=60)
    response.raise_for_status()

    asset_url = response.json()["uploadUrl"]
    asset_id = response.json()["assetId"]

    response = requests.put(
        asset_url,
        data=input_data,
        headers=s3_headers,
        timeout=UPLOAD_ASSET_TIMEOUT,
    )

    response.raise_for_status()
    return uuid.UUID(asset_id)


def _run_grounding_dino(
    image_path: str, prompt: str, threshold: float = 0.3, *, api_key: str
) -> list[BoundingBox]:
    """Run Grounding DINO on an image using NVIDIA API.

    Args:
        image_path: Path to the input image file.
        prompt: Text query for object grounding.
        threshold: Confidence threshold for detections.
        api_key: NVIDIA API key used for authorization.
    """
    if not api_key:
        raise RuntimeError("NVIDIA API key is required.")
    header_auth = f"Bearer {api_key}"

    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type:
        mime_type = "image/jpeg"

    with open(image_path, "rb") as f:
        asset_id = _upload_asset(f.read(), "Input Image", mime_type, api_key=api_key)

    inputs = {
        "model": "Grounding-Dino",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{prompt}"},
                    {
                        "type": "media_url",
                        "media_url": {"url": f"data:{mime_type};asset_id,{asset_id}"},
                    },
                ],
            }
        ],
        "threshold": threshold,
    }

    asset_list = f"{asset_id}"

    headers = {
        "Content-Type": "application/json",
        "NVCF-INPUT-ASSET-REFERENCES": asset_list,
        "NVCF-FUNCTION-ASSET-IDS": asset_list,
        "Authorization": header_auth,
    }

    tmp_dir = tempfile.mkdtemp(prefix="grounding_dino_")
    zip_path = os.path.join(tmp_dir, "result.zip")

    response = requests.post(NVAI_URL, headers=headers, json=inputs)

    if response.status_code == 200:
        with open(zip_path, "wb") as out:
            out.write(response.content)
    elif response.status_code == 202:
        nvcf_reqid = response.headers["NVCF-REQID"]
        polling_url = NVAI_POLLING_URL + nvcf_reqid

        retries_remaining = MAX_RETRIES
        while retries_remaining:
            headers_polling = {
                "accept": "application/json",
                "Authorization": header_auth,
            }
            response_polling = requests.get(polling_url, headers=headers_polling)
            if response_polling.status_code == 202:
                retries_remaining -= 1
                time.sleep(DELAY_BTW_RETRIES)
                continue
            elif response_polling.status_code == 200:
                with open(zip_path, "wb") as out:
                    out.write(response_polling.content)
                break
            else:
                raise RuntimeError(
                    f"Unexpected response status while polling: {response_polling.status_code}"
                )
        else:
            raise TimeoutError("Polling timed out waiting for Grounding DINO results.")
    else:
        raise RuntimeError(
            f"Unexpected response status: {response.status_code}, body: {response.text}"
        )

    # Create a separate temporary directory for extraction
    extract_dir = tempfile.mkdtemp(prefix="grounding_dino_extract_")

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)

    json_path = None
    for root, _dirs, files in os.walk(extract_dir):
        for name in files:
            if name.lower().endswith(".response"):
                json_path = os.path.join(root, name)
                break
        if json_path:
            break

    if not json_path:
        raise FileNotFoundError("No .response file found in extracted results.")

    with open(json_path, encoding="utf-8") as f:
        result = json.load(f)

    # Clean up temporary directories
    shutil.rmtree(tmp_dir)  # Remove the directory containing the zip file
    shutil.rmtree(extract_dir)  # Remove the extraction directory

    # Extract and return just the boundingBoxes part
    try:
        bounding_boxes = result["choices"][0]["message"]["content"]["boundingBoxes"]
        return bounding_boxes
    except (KeyError, IndexError) as e:
        raise ValueError(f"Unable to extract boundingBoxes from response: {e}") from e


def detect_objects_with_grounding_dino(
    image: str | Path | Image.Image | np.ndarray,
    prompt: str,
    threshold: float = 0.3,
    api_key: str | None = None,
) -> dict[str, Any]:
    """
    Detect objects in an image using Grounding DINO based on text prompts.

    Args:
        image: Input image (path, PIL Image, or numpy array)
        prompt: Text description of objects to detect (e.g., "red pot", "person wearing hat")
        threshold: Confidence threshold for detections (0.0-1.0)
        api_key: NVIDIA API key (uses NVIDIA_API_KEY env var if not provided)

    Returns:
        Dict containing:
            - detections: List of BoundingBox objects with detected objects
            - total_detections: Total number of objects detected
            - image_size: (width, height) of the processed image
    """
    # Handle different image input types
    if isinstance(image, str | Path):
        image_path = str(image)
        temp_file_created = False
    elif isinstance(image, Image.Image):
        # Save PIL image to temp file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            image.save(tmp.name)
            image_path = tmp.name
            temp_file_created = True
    elif isinstance(image, np.ndarray):
        # Convert numpy array to PIL and save
        pil_image = Image.fromarray(image)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            pil_image.save(tmp.name)
            image_path = tmp.name
            temp_file_created = True
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")

    try:
        # Call the internal function
        detections = _run_grounding_dino(
            image_path=image_path,
            prompt=prompt,
            threshold=threshold,
            api_key=api_key or os.getenv("NVIDIA_API_KEY", ""),
        )

        # Get image dimensions
        with Image.open(image_path) as img:
            width, height = img.size

        # Count total detections
        total_detections = sum(len(det["bboxes"]) for det in detections)

        return {
            "detections": detections,
            "total_detections": total_detections,
            "image_size": (width, height),
        }

    finally:
        # Clean up temp files if created
        if temp_file_created and os.path.exists(image_path):
            os.unlink(image_path)
