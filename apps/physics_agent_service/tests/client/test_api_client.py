# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os

import pytest

from ...client.client import PhysicsAgentClient

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")


@pytest.fixture
def api_client() -> PhysicsAgentClient:
    client = PhysicsAgentClient(base_url=BASE_URL)
    return client


def test_start_pipeline_forwards_optimizer_flags(tmp_path):
    usd_path = tmp_path / "scene.usda"
    usd_path.write_text("#usda 1.0\n", encoding="utf-8")
    captured = {}

    class FakeResponse:
        ok = True
        status_code = 202
        text = ""

        def raise_for_status(self):
            return None

        def json(self):
            return {"session_id": "session-123"}

    class FakeSession:
        headers: dict[str, str] = {}

        def post(self, url, data=None, files=None, timeout=None):
            captured.update(
                {"url": url, "data": data, "files": files, "timeout": timeout}
            )
            return FakeResponse()

    client = PhysicsAgentClient(base_url="http://test", timeout_seconds=7)
    client._http = FakeSession()

    session_id = client.start_pipeline(
        usd_path=str(usd_path),
        optimize_usd=True,
        enable_deinstance=False,
        enable_split=True,
        enable_deduplicate=True,
    )

    assert session_id == "session-123"
    assert captured["url"] == "http://test/pipeline"
    assert captured["timeout"] == 7
    assert captured["data"]["optimize_usd"] == "true"
    assert captured["data"]["enable_deinstance"] == "false"
    assert captured["data"]["enable_split"] == "true"
    assert captured["data"]["enable_deduplicate"] == "true"


@pytest.mark.parametrize(
    "kwargs",
    [
        (
            {
                "usd_path": os.path.join(
                    os.path.dirname(__file__),
                    "../../../physics_agent/data/examples/Lightbulb01/light_bulb_01.usdz",
                ),
            }
        ),
        (
            {
                "usd_path": os.path.join(
                    os.path.dirname(__file__),
                    "../../../physics_agent/data/examples/Lightbulb01/light_bulb_01.usdz",
                ),
                "user_prompt": "Focus on identifying the glass and metal parts of the lightbulb.",
            }
        ),
    ],
)
@pytest.mark.skipif(
    os.getenv("RUN_CLIENT_TESTS", "false").lower() not in ["true", "1", "yes", "y"],
    reason="Skipping test in CI",
)
async def test_basic_pipeline(api_client: PhysicsAgentClient, kwargs: dict):
    session_id, status = api_client.run_and_monitor(**kwargs)
    assert status is not None
    assert status["status"] == "completed"

    # Check predictions artifact
    predictions = api_client.download_predictions(session_id)
    assert len(predictions) > 0
    print(f"Predictions downloaded: {len(predictions)} bytes")

    # Check report artifact
    report = api_client._http.get(
        f"{api_client.base_url}/artifacts/{session_id}/report"
    )
    assert report.status_code == 200
    report_text = report.text
    assert len(report_text) > 0
    print(f"Report downloaded: {len(report_text)} bytes")

    if "user_prompt" in kwargs:
        assert kwargs["user_prompt"] in report_text
        print(f"user_prompt: {kwargs['user_prompt']} found in report")
