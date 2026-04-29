"""Tests for GET /artifacts/{session_id}/output-usd.

Covers the service endpoint that serves the apply_physics step's
simulation-ready USD.
"""

import asyncio

import httpx
import pytest

from ..conftest import make_pipeline_files


async def _wait_for_completion(client: httpx.AsyncClient, session_id: str) -> None:
    """Poll status until the pipeline is completed; fail if it times out."""
    for _ in range(200):
        status_r = await client.get(f"/pipeline/{session_id}/status")
        if status_r.json()["status"] == "completed":
            return
        await asyncio.sleep(0.01)
    pytest.fail(f"Pipeline for {session_id} did not reach 'completed' in time")


@pytest.mark.api
class TestOutputUsdDownload:
    """Test the output-usd artifact endpoint."""

    async def test_download_returns_file_after_completion(
        self, client: httpx.AsyncClient
    ) -> None:
        """Once the pipeline completes, the USD file is served on GET."""
        create_r = await client.post("/pipeline", files=make_pipeline_files())
        session_id = create_r.json()["session_id"]

        await _wait_for_completion(client, session_id)

        r = await client.get(f"/artifacts/{session_id}/output-usd")

        assert r.status_code == 200
        # Stub writes a minimal `#usda 1.0` header.
        assert r.content.startswith(b"#usda")
        content_disposition = r.headers.get("content-disposition", "")
        assert "scene_physics.usda" in content_disposition

    async def test_returns_404_for_nonexistent_session(
        self, client: httpx.AsyncClient
    ) -> None:
        """Unknown session → 404."""
        r = await client.get(
            "/artifacts/00000000-0000-0000-0000-000000000000/output-usd"
        )
        assert r.status_code == 404

    async def test_results_response_includes_output_usd_link(
        self, client: httpx.AsyncClient
    ) -> None:
        """GET /pipeline/{id}/results advertises the output-usd download URL."""
        create_r = await client.post("/pipeline", files=make_pipeline_files())
        session_id = create_r.json()["session_id"]

        await _wait_for_completion(client, session_id)

        r = await client.get(f"/pipeline/{session_id}/results")

        assert r.status_code == 200
        download_urls = r.json()["download_urls"]
        assert "output_usd" in download_urls
        assert download_urls["output_usd"] == f"/artifacts/{session_id}/output-usd"
