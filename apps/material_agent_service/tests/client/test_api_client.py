# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import os

import pytest

from ...client.client import MaterialAgentClient

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")


class _FakeResponse:
    def __init__(self, payload: dict | None = None, status_code: int = 200):
        self._payload = payload or {}
        self.status_code = status_code

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise AssertionError(f"unexpected HTTP status {self.status_code}")


class _FakeSession:
    def __init__(self):
        self.posts: list[dict] = []
        self.heads: list[dict] = []
        self.head_responses: list[_FakeResponse] = []

    def post(self, url: str, **kwargs):
        self.posts.append({"url": url, **kwargs})
        return _FakeResponse(
            {
                "status": "ok",
                "reference_id": "ref-1",
                "image_url": "/assets/s/generated-ref/ref-1",
            }
        )

    def head(self, url: str, **kwargs):
        self.heads.append({"url": url, **kwargs})
        if self.head_responses:
            return self.head_responses.pop(0)
        return _FakeResponse(status_code=200)


@pytest.fixture
def api_client() -> MaterialAgentClient:
    client = MaterialAgentClient(base_url=BASE_URL)
    return client


def test_generate_reference_image_posts_prompt():
    client = MaterialAgentClient(base_url="http://service")
    fake_session = _FakeSession()
    client._http = fake_session  # type: ignore[assignment]

    result = client.generate_reference_image("session-1", "matte blue plastic")

    assert result["reference_id"] == "ref-1"
    assert result["image_url"] == "/assets/s/generated-ref/ref-1"
    assert fake_session.posts == [
        {
            "url": "http://service/pipeline/session-1/generate-reference-image",
            "data": {"prompt": "matte blue plastic"},
            "timeout": client.timeout_seconds,
        }
    ]


def test_wait_for_input_render_stops_on_terminal_failure():
    client = MaterialAgentClient(base_url="http://service")
    fake_session = _FakeSession()
    fake_session.head_responses.append(_FakeResponse(status_code=424))
    client._http = fake_session  # type: ignore[assignment]

    with pytest.raises(RuntimeError, match="Input preview render failed"):
        client.wait_for_input_render("session-1", poll_interval_seconds=0)


def test_wait_for_input_render_follows_presigned_redirects():
    client = MaterialAgentClient(base_url="http://service")
    fake_session = _FakeSession()
    fake_session.head_responses.append(_FakeResponse(status_code=302))
    client._http = fake_session  # type: ignore[assignment]

    client.wait_for_input_render("session-1", poll_interval_seconds=0)

    assert fake_session.heads == [
        {
            "url": "http://service/assets/session-1/input-render",
            "timeout": client.timeout_seconds,
            "allow_redirects": True,
        }
    ]


def test_run_and_monitor_generates_reference_before_pipeline(monkeypatch):
    client = MaterialAgentClient(base_url="http://service")
    calls: list[str] = []

    def upload_usd(usd_path: str) -> str:
        calls.append(f"upload:{usd_path}")
        return "session-1"

    def wait_for_input_render(session_id: str, timeout_seconds: int = 180, **_) -> None:
        calls.append(f"wait:{session_id}:{timeout_seconds}")

    def generate_reference_image(session_id: str, prompt: str) -> dict:
        calls.append(f"generate:{session_id}:{prompt}")
        return {"status": "ok", "reference_id": "ref-1"}

    def start_pipeline(**kwargs) -> str:
        calls.append(f"start:{kwargs['session_id']}:{kwargs['generated_reference_id']}")
        return kwargs["session_id"]

    monkeypatch.setattr(client, "upload_usd", upload_usd)
    monkeypatch.setattr(client, "wait_for_input_render", wait_for_input_render)
    monkeypatch.setattr(client, "generate_reference_image", generate_reference_image)
    monkeypatch.setattr(client, "start_pipeline", start_pipeline)
    monkeypatch.setattr(client, "stream_events", lambda _session_id: iter(()))
    monkeypatch.setattr(
        client, "get_status", lambda _session_id: {"status": "completed"}
    )

    session_id, status = client.run_and_monitor(
        usd_path="/tmp/scene.usd",
        generated_reference_prompt="matte blue plastic",
        preview_timeout_seconds=12,
        print_stream=False,
    )

    assert session_id == "session-1"
    assert status == {"status": "completed"}
    assert calls == [
        "upload:/tmp/scene.usd",
        "wait:session-1:12",
        "generate:session-1:matte blue plastic",
        "start:session-1:ref-1",
    ]


@pytest.mark.parametrize(
    "kwargs,not_found_in_report",
    [
        (
            {
                "usd_path": os.path.join(
                    os.path.dirname(__file__),
                    "../../../material_agent/data/examples/ladder/sources/usd/ladder.usd",
                ),
                "user_email": "test@nvidia.com",
            },
            None,
        ),
        (
            {
                "usd_path": os.path.join(
                    os.path.dirname(__file__),
                    "../../../material_agent/data/examples/ladder/sources/usd/ladder.usd",
                ),
                "user_email": "test@nvidia.com",
                "reference_images": [
                    os.path.join(
                        os.path.dirname(__file__),
                        "../../../material_agent/data/examples/ladder/sources/images/ladder_reference_1.jpeg",
                    ),
                    os.path.join(
                        os.path.dirname(__file__),
                        "../../../material_agent/data/examples/ladder/sources/images/ladder_reference_2.jpeg",
                    ),
                ],
            },
            None,
        ),
        (
            {
                "usd_path": os.path.join(
                    os.path.dirname(__file__),
                    "../../../material_agent/data/examples/ladder/sources/usd/ladder.usd",
                ),
                "user_email": "test@nvidia.com",
                "reference_images": [
                    os.path.join(
                        os.path.dirname(__file__),
                        "../../../material_agent/data/examples/ladder/sources/images/ladder_reference_1.jpeg",
                    ),
                    os.path.join(
                        os.path.dirname(__file__),
                        "../../../material_agent/data/examples/ladder/sources/images/ladder_reference_2.jpeg",
                    ),
                ],
                "reference_descriptions": [
                    "This is a reference image of the ladder (front view) that you can use to identify the material of the parts.",
                    "This is a reference image of the ladder (rear view) that you can use to identify the material of the parts.",
                ],
            },
            None,
        ),
        (
            {
                "usd_path": os.path.join(
                    os.path.dirname(__file__),
                    "../../../material_agent/data/examples/ladder/sources/usd/ladder.usd",
                ),
                "user_email": "test@nvidia.com",
                "user_prompt": "Identify what the object part is, and then select a material from the predefined list of materials for this highlighted object part. Provide the identified object part in the reasoning.",
            },
            None,
        ),
        (
            {
                "usd_path": os.path.join(
                    os.path.dirname(__file__),
                    "../../../material_agent/data/examples/ladder/sources/usd/ladder.usd",
                ),
                "user_email": "test@nvidia.com",
                "camera_views": "+x+y+z",
            },
            ["-x+y+z"],
        ),
        (
            {
                "usd_path": os.path.join(
                    os.path.dirname(__file__),
                    "../../../material_agent/data/examples/ladder/sources/usd/ladder.usd",
                ),
                "user_email": "test@nvidia.com",
                "optimize_usd": True,
            },
            None,
        ),
        (
            {
                "usd_path": os.path.join(
                    os.path.dirname(__file__),
                    "../../../material_agent/data/examples/ladder/sources/usd/ladder.usd",
                ),
                "user_email": "test@nvidia.com",
                "optimize_usd": False,
            },
            None,
        ),
    ],
)
@pytest.mark.skipif(
    os.getenv("RUN_CLIENT_TESTS", "false").lower() not in ["true", "1", "yes", "y"],
    reason="Skipping test in CI",
)
async def test_basic_pipeline(
    api_client: MaterialAgentClient, kwargs: dict, not_found_in_report: list[str] | None
):
    session_id, status = api_client.run_and_monitor(**kwargs)
    assert status is not None
    assert status["status"] == "completed"
    await asyncio.sleep(5)
    report = api_client._http.get(
        f"{api_client.base_url}/artifacts/{session_id}/report"
    )
    assert report.status_code == 200
    report_text = report.text
    if "reference_images" in kwargs:
        response = api_client._http.get(
            f"{api_client.base_url}/assets/{session_id}/references"
        )
        assert response.status_code == 200
        json_response = response.json()
        assert json_response is not None
        assert json_response["references"] is not None
        assert len(json_response["references"]) == len(kwargs["reference_images"])

        if "reference_descriptions" not in kwargs:
            # make sure reference images were used and are mentioned in the report
            for ind in range(len(json_response["references"])):
                assert (
                    f"This is reference image {ind + 1} of the asset you will match this look exactly"
                    in report_text
                )
                print("default reference descriptions found in report")
        else:
            assert len(kwargs["reference_descriptions"]) == len(
                json_response["references"]
            )
            print(
                f"reference_descriptions: {kwargs['reference_descriptions']} found in report"
            )
            for description in kwargs["reference_descriptions"]:
                assert description in report_text
                print(f"reference_description: {description} found in report")

    if "user_prompt" in kwargs:
        assert kwargs["user_prompt"] in report_text
        print(f"user_prompt: {kwargs['user_prompt']} found in report")

    if "camera_views" in kwargs:
        for camera_view in kwargs["camera_views"].split(","):
            assert camera_view.strip().lower() in report_text.lower()
            print(f"camera_view: {camera_view.strip().lower()} found in report")

    if "optimize_usd" in kwargs:
        optimization_report = api_client._http.get(
            f"{api_client.base_url}/artifacts/{session_id}/optimization-report"
        )
        # Check whether optimize_usd actually ran (it may be silently skipped
        # when the Scene Optimizer backend is unavailable in the test environment).
        completed_step_names = [s["name"] for s in status.get("completed_steps", [])]
        optimize_actually_ran = "optimize_usd" in completed_step_names

        if kwargs["optimize_usd"] and optimize_actually_ran:
            assert optimization_report.status_code == 200
            optimization_report_json = optimization_report.json()
            assert "report" in optimization_report_json
            assert "operations_executed" in optimization_report_json
            assert len(optimization_report_json["operations_executed"]) > 0
        else:
            assert optimization_report.status_code == 404

    if not_found_in_report:
        for not_found in not_found_in_report:
            assert not_found.strip().lower() not in report_text.lower()
            print(
                f"not_found_in_report: {not_found.strip().lower()} not found in report"
            )
