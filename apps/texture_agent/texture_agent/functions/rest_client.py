# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""REST client for the Texture Variation API service.

Calls the remote service at /v1/texture-variations endpoints.
Drop-in replacement for the local TextureVariationClient.

Usage:
    from texture_agent.functions.rest_client import RestTextureVariationClient

    client = RestTextureVariationClient("http://dt1:8000")
    status = client.generate(
        source_asset_uri="file:///path/to/asset.usd",
        conditioning=Conditioning(text_prompt="rusted metal"),
        config=TextureVariationConfig(strength=0.8),
    )
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

from texture_agent.functions.texture_generation import (
    Conditioning,
    GeneratedTextures,
    GenerationResult,
    JobStatus,
    TextureVariationConfig,
)

logger = logging.getLogger(__name__)


class RestTextureVariationClient:
    """REST client implementing the Texture Variation API contract.

    Talks to a remote service running the texture-editing pipeline
    (Step1X-3D + Material Anything) via the REST API.
    """

    def __init__(
        self,
        endpoint_url: str,
        api_key: str | None = None,
        timeout: float = 600.0,
    ) -> None:
        self._endpoint_url = endpoint_url.rstrip("/")
        self._headers: dict[str, str] = {}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"
        self._timeout = timeout

    def generate(
        self,
        source_asset_uri: str,
        conditioning: Conditioning,
        config: TextureVariationConfig | None = None,
        wait: bool = True,
        timeout_sec: int = 600,
    ) -> JobStatus:
        """Submit a texture variation job.

        Args:
            source_asset_uri: URI to the source USD asset.
            conditioning: Text prompt, reference images, etc.
            config: Generation configuration.
            wait: If True, poll until terminal status or timeout.
            timeout_sec: Max wait time in seconds.

        Returns:
            JobStatus with result on completion.
        """
        config = config or TextureVariationConfig()
        conditioning.validate()

        # Build request body matching the REST API spec
        body: dict[str, Any] = {
            "source_asset_uri": source_asset_uri,
            "conditioning": {
                "text_prompt": conditioning.text_prompt,
                "reference_image_uris": conditioning.reference_image_uris,
                "turntable_video_uri": conditioning.turntable_video_uri,
            },
            "configuration": {
                "strength": config.strength,
                "seed": config.seed,
                "variant_name": config.variant_name,
                "engine": config.engine,
                "custom_parameters": config.custom_parameters,
            },
        }

        url = f"{self._endpoint_url}/v1/texture-variations"
        logger.info(
            "POST %s (prompt='%s', strength=%.2f)",
            url,
            conditioning.text_prompt,
            config.strength,
        )

        with httpx.Client(timeout=self._timeout, headers=self._headers) as client:
            # Submit job
            resp = client.post(url, json=body)
            if resp.status_code not in (200, 201, 202):
                return JobStatus(
                    job_id="",
                    status="failed",
                    error_message=f"HTTP {resp.status_code}: {resp.text}",
                )

            status = self._parse_status(resp.json())
            logger.info("Job submitted: %s (status=%s)", status.job_id, status.status)

            if not wait:
                return status

            # Poll until terminal or timeout
            deadline = time.time() + timeout_sec
            poll_interval = 2.0

            while status.status in ("queued", "processing"):
                if time.time() > deadline:
                    logger.warning("Timeout waiting for job %s", status.job_id)
                    return status

                time.sleep(poll_interval)
                status = self.get_status(status.job_id, client=client)
                logger.info(
                    "Job %s: %s (%d%%) %s",
                    status.job_id,
                    status.status,
                    status.progress,
                    status.message or "",
                )

                # Back off gradually
                poll_interval = min(poll_interval * 1.5, 10.0)

            return status

    def get_status(
        self,
        job_id: str,
        client: httpx.Client | None = None,
    ) -> JobStatus:
        """Query job status."""
        url = f"{self._endpoint_url}/v1/texture-variations/{job_id}"

        if client:
            resp = client.get(url)
        else:
            with httpx.Client(timeout=self._timeout, headers=self._headers) as c:
                resp = c.get(url)

        if resp.status_code == 404:
            return JobStatus(
                job_id=job_id, status="failed", error_message="Job not found"
            )
        resp.raise_for_status()
        return self._parse_status(resp.json())

    def cancel(self, job_id: str) -> None:
        """Cancel a job."""
        url = f"{self._endpoint_url}/v1/texture-variations/{job_id}"
        with httpx.Client(timeout=30, headers=self._headers) as client:
            resp = client.delete(url)
            if resp.status_code == 409:
                raise ValueError("Job already in terminal state")
            resp.raise_for_status()

    @staticmethod
    def _parse_status(data: dict[str, Any]) -> JobStatus:
        """Parse a JSON response into a JobStatus."""
        result = None
        if data.get("result"):
            r = data["result"]
            gt = r.get("generated_textures", {})
            result = GenerationResult(
                variant_asset_uri=r.get("variant_asset_uri", ""),
                variant_name=r.get("variant_name", ""),
                generated_textures=GeneratedTextures(
                    albedo=gt.get("albedo", ""),
                    normal=gt.get("normal", ""),
                    orm=gt.get("orm", ""),
                ),
            )

        return JobStatus(
            job_id=data.get("job_id", ""),
            status=data.get("status", "failed"),
            progress=data.get("progress", 0),
            message=data.get("message"),
            result=result,
            error_message=data.get("error_message"),
        )
