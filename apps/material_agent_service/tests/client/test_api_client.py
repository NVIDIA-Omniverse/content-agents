# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import json
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
        self.gets: list[dict] = []
        self.heads: list[dict] = []
        self.head_responses: list[_FakeResponse] = []

    def post(self, url: str, **kwargs):
        self.posts.append({"url": url, **kwargs})
        return _FakeResponse(
            {
                "status": "ok",
                "session_id": "session-1",
                "reference_id": "ref-1",
                "image_url": "/assets/s/generated-ref/ref-1",
            }
        )

    def get(self, url: str, **kwargs):
        self.gets.append({"url": url, **kwargs})
        return _FakeResponse({"events": [{"step": "predict"}], "total": 1})

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


def test_start_pipeline_posts_worker_overrides(tmp_path):
    client = MaterialAgentClient(base_url="http://service")
    fake_session = _FakeSession()
    client._http = fake_session  # type: ignore[assignment]
    usd_path = tmp_path / "scene.usda"
    usd_path.write_text("#usda 1.0\n")

    session_id = client.start_pipeline(
        usd_path=str(usd_path),
        user_email="test@example.com",
        vlm_max_workers=2,
        render_num_workers=1,
    )

    assert session_id == "session-1"
    assert fake_session.posts[0]["url"] == "http://service/pipeline"
    assert fake_session.posts[0]["data"]["vlm_max_workers"] == "2"
    assert fake_session.posts[0]["data"]["render_num_workers"] == "1"


def test_regenerate_posts_json_body():
    client = MaterialAgentClient(base_url="http://service")
    fake_session = _FakeSession()
    client._http = fake_session  # type: ignore[assignment]

    result = client.regenerate(
        "session-1",
        steps=["predict", "apply"],
        user_prompt="Prefer brushed aluminum",
        layer_only=True,
    )

    assert result["session_id"] == "session-1"
    assert fake_session.posts == [
        {
            "url": "http://service/pipeline/session-1/regenerate",
            "json": {
                "steps": ["predict", "apply"],
                "user_prompt": "Prefer brushed aluminum",
                "layer_only": True,
            },
            "timeout": client.timeout_seconds,
        }
    ]


def test_get_event_log_uses_persisted_history_endpoint():
    client = MaterialAgentClient(base_url="http://service")
    fake_session = _FakeSession()
    client._http = fake_session  # type: ignore[assignment]

    result = client.get_event_log("session-1")

    assert result == {"events": [{"step": "predict"}], "total": 1}
    assert fake_session.gets == [
        {
            "url": "http://service/pipeline/session-1/event-log",
            "timeout": client.timeout_seconds,
        }
    ]


def test_start_pipeline_allows_service_default_vlm_worker_cap(tmp_path):
    client = MaterialAgentClient(base_url="http://service")
    fake_session = _FakeSession()
    client._http = fake_session  # type: ignore[assignment]
    usd_path = tmp_path / "scene.usda"
    usd_path.write_text("#usda 1.0\n")

    session_id = client.start_pipeline(
        usd_path=str(usd_path),
        user_email="test@example.com",
        vlm_max_workers=64,
    )

    assert session_id == "session-1"
    assert fake_session.posts[0]["data"]["vlm_max_workers"] == "64"


def test_start_pipeline_posts_prim_clustering_overrides(tmp_path):
    client = MaterialAgentClient(base_url="http://service")
    fake_session = _FakeSession()
    client._http = fake_session  # type: ignore[assignment]
    usd_path = tmp_path / "scene.usda"
    usd_path.write_text("#usda 1.0\n")

    session_id = client.start_pipeline(
        usd_path=str(usd_path),
        user_email="test@example.com",
        enable_prim_clustering=True,
        cluster_min_prims=25,
        cluster_embedding_backend="nim",
        cluster_embedding_model="nvidia/llama-nemotron-embed-vl-1b-v2",
        cluster_embedding_base_url="http://embedding-nim:8000/v1",
        cluster_embedding_max_workers=2,
        cluster_embedding_batch_size=8,
        cluster_max_size=11,
        cluster_similarity_threshold_low=0.97,
        cluster_similarity_threshold_medium=0.94,
        cluster_similarity_threshold_high=0.88,
        cluster_report=False,
    )

    assert session_id == "session-1"
    data = fake_session.posts[0]["data"]
    assert data["enable_prim_clustering"] == "true"
    assert data["cluster_min_prims"] == "25"
    assert data["cluster_embedding_backend"] == "nim"
    assert data["cluster_embedding_model"] == "nvidia/llama-nemotron-embed-vl-1b-v2"
    assert data["cluster_embedding_base_url"] == "http://embedding-nim:8000/v1"
    assert data["cluster_embedding_max_workers"] == "2"
    assert data["cluster_embedding_batch_size"] == "8"
    assert data["cluster_max_size"] == "11"
    assert data["cluster_similarity_threshold_low"] == "0.97"
    assert data["cluster_similarity_threshold_medium"] == "0.94"
    assert data["cluster_similarity_threshold_high"] == "0.88"
    assert data["cluster_report"] == "false"


def test_start_pipeline_posts_large_scene_options(tmp_path):
    client = MaterialAgentClient(base_url="http://service")
    fake_session = _FakeSession()
    client._http = fake_session  # type: ignore[assignment]
    usd_path = tmp_path / "scene.usda"
    usd_path.write_text("#usda 1.0\n")

    session_id = client.start_pipeline(
        usd_path=str(usd_path),
        user_email="test@example.com",
        large_scene=True,
        scene_workers=2,
        scene_assets=["AssetA", "/World/AssetB"],
        scene_resume=True,
        scene_from_step="predict",
        scene_skip_existing=True,
        scene_no_render=True,
        scene_simulate=True,
        scene_simulate_mock_analyze=True,
        scene_fail_on_validation_error=True,
        scene_filters={"include_prim_paths": ["/World"]},
    )

    assert session_id == "session-1"
    data = fake_session.posts[0]["data"]
    assert data["large_scene"] == "true"
    assert data["scene_workers"] == "2"
    assert data["scene_assets"] == "AssetA,/World/AssetB"
    assert data["scene_resume"] == "true"
    assert data["scene_from_step"] == "predict"
    assert data["scene_skip_existing"] == "true"
    assert data["scene_no_render"] == "true"
    assert data["scene_simulate"] == "true"
    assert data["scene_simulate_mock_analyze"] == "true"
    assert data["scene_fail_on_validation_error"] == "true"
    assert json.loads(data["scene_filters"]) == {"include_prim_paths": ["/World"]}
    assert "scene_analyze_llm" not in data


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"cluster_min_prims": 0}, "cluster_min_prims must be at least 1"),
        (
            {"cluster_embedding_max_workers": 0},
            "cluster_embedding_max_workers must be at least 1",
        ),
        (
            {"cluster_embedding_batch_size": 0},
            "cluster_embedding_batch_size must be at least 1",
        ),
        ({"cluster_max_size": 0}, "cluster_max_size must be at least 1"),
        (
            {"cluster_similarity_threshold_low": -0.1},
            "cluster_similarity_threshold_low must be between 0.0 and 1.0",
        ),
        (
            {"cluster_similarity_threshold_medium": 1.1},
            "cluster_similarity_threshold_medium must be between 0.0 and 1.0",
        ),
    ],
)
def test_start_pipeline_rejects_bad_prim_clustering_overrides(tmp_path, kwargs, match):
    client = MaterialAgentClient(base_url="http://service")
    fake_session = _FakeSession()
    client._http = fake_session  # type: ignore[assignment]
    usd_path = tmp_path / "scene.usda"
    usd_path.write_text("#usda 1.0\n")

    with pytest.raises(ValueError, match=match):
        client.start_pipeline(
            usd_path=str(usd_path),
            user_email="test@example.com",
            enable_prim_clustering=True,
            **kwargs,
        )

    assert fake_session.posts == []


def test_start_pipeline_rejects_worker_overrides_above_client_cap(
    monkeypatch, tmp_path
):
    client = MaterialAgentClient(base_url="http://service")
    fake_session = _FakeSession()
    client._http = fake_session  # type: ignore[assignment]
    usd_path = tmp_path / "scene.usda"
    usd_path.write_text("#usda 1.0\n")
    monkeypatch.setenv("RENDER_NUM_WORKERS_MAX", "1")

    with pytest.raises(ValueError, match="render_num_workers must be between 1 and 1"):
        client.start_pipeline(
            usd_path=str(usd_path),
            user_email="test@example.com",
            render_num_workers=2,
        )

    assert fake_session.posts == []


def test_run_and_monitor_rejects_worker_overrides_before_upload(monkeypatch):
    client = MaterialAgentClient(base_url="http://service")
    monkeypatch.setenv("VLM_MAX_WORKERS_MAX", "1")

    def fail_upload(_usd_path: str) -> str:
        raise AssertionError("upload_usd should not be called")

    monkeypatch.setattr(client, "upload_usd", fail_upload)

    with pytest.raises(ValueError, match="vlm_max_workers must be between 1 and 1"):
        client.run_and_monitor(
            usd_path="/tmp/scene.usd",
            upload_first=True,
            vlm_max_workers=2,
            print_stream=False,
        )


def test_run_and_monitor_rejects_cluster_overrides_before_upload(monkeypatch):
    client = MaterialAgentClient(base_url="http://service")

    def fail_upload(_usd_path: str) -> str:
        raise AssertionError("upload_usd should not be called")

    monkeypatch.setattr(client, "upload_usd", fail_upload)

    with pytest.raises(ValueError, match="cluster_min_prims must be at least 1"):
        client.run_and_monitor(
            usd_path="/tmp/scene.usd",
            upload_first=True,
            enable_prim_clustering=True,
            cluster_min_prims=0,
            print_stream=False,
        )


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
    captured_start_kwargs: dict = {}

    def upload_usd(usd_path: str) -> str:
        calls.append(f"upload:{usd_path}")
        return "session-1"

    def wait_for_input_render(session_id: str, timeout_seconds: int = 180, **_) -> None:
        calls.append(f"wait:{session_id}:{timeout_seconds}")

    def generate_reference_image(session_id: str, prompt: str) -> dict:
        calls.append(f"generate:{session_id}:{prompt}")
        return {"status": "ok", "reference_id": "ref-1"}

    def start_pipeline(**kwargs) -> str:
        captured_start_kwargs.update(kwargs)
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
        vlm_max_workers=2,
        render_num_workers=1,
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
    assert captured_start_kwargs["vlm_max_workers"] == 2
    assert captured_start_kwargs["render_num_workers"] == 1


def test_run_and_monitor_passes_prim_clustering_overrides(monkeypatch):
    client = MaterialAgentClient(base_url="http://service")
    captured_start_kwargs: dict = {}

    def start_pipeline(**kwargs) -> str:
        captured_start_kwargs.update(kwargs)
        return "session-1"

    monkeypatch.setattr(client, "start_pipeline", start_pipeline)
    monkeypatch.setattr(client, "stream_events", lambda _session_id: iter(()))
    monkeypatch.setattr(
        client, "get_status", lambda _session_id: {"status": "completed"}
    )

    session_id, status = client.run_and_monitor(
        usd_path="/tmp/scene.usd",
        enable_prim_clustering=True,
        cluster_min_prims=25,
        cluster_embedding_backend="nim",
        cluster_embedding_model="nvidia/llama-nemotron-embed-vl-1b-v2",
        cluster_embedding_base_url="http://embedding-nim:8000/v1",
        cluster_embedding_max_workers=2,
        cluster_embedding_batch_size=8,
        cluster_max_size=11,
        cluster_similarity_threshold_low=0.97,
        cluster_similarity_threshold_medium=0.94,
        cluster_similarity_threshold_high=0.88,
        cluster_report=False,
        print_stream=False,
    )

    assert session_id == "session-1"
    assert status == {"status": "completed"}
    assert captured_start_kwargs["enable_prim_clustering"] is True
    assert captured_start_kwargs["cluster_min_prims"] == 25
    assert captured_start_kwargs["cluster_embedding_backend"] == "nim"
    assert (
        captured_start_kwargs["cluster_embedding_model"]
        == "nvidia/llama-nemotron-embed-vl-1b-v2"
    )
    assert (
        captured_start_kwargs["cluster_embedding_base_url"]
        == "http://embedding-nim:8000/v1"
    )
    assert captured_start_kwargs["cluster_embedding_max_workers"] == 2
    assert captured_start_kwargs["cluster_embedding_batch_size"] == 8
    assert captured_start_kwargs["cluster_max_size"] == 11
    assert captured_start_kwargs["cluster_similarity_threshold_low"] == 0.97
    assert captured_start_kwargs["cluster_similarity_threshold_medium"] == 0.94
    assert captured_start_kwargs["cluster_similarity_threshold_high"] == 0.88
    assert captured_start_kwargs["cluster_report"] is False


@pytest.mark.parametrize(
    "kwargs",
    [
        {"upload_first": True},
        {"generated_reference_prompt": "matte blue plastic"},
    ],
)
def test_run_and_monitor_rejects_preview_upload_modes_for_large_scene(
    monkeypatch,
    kwargs,
):
    client = MaterialAgentClient(base_url="http://service")

    def upload_usd(_usd_path: str) -> str:
        raise AssertionError("large_scene must not call upload_usd")

    monkeypatch.setattr(client, "upload_usd", upload_usd)

    with pytest.raises(ValueError, match="large_scene is not compatible"):
        client.run_and_monitor(
            usd_path="/tmp/scene.usda",
            large_scene=True,
            print_stream=False,
            **kwargs,
        )


def test_main_passes_worker_overrides(monkeypatch, tmp_path, capsys):
    from ...client import client as client_module

    captured_kwargs: dict = {}

    class FakeClient:
        def __init__(self, base_url: str, token: str | None = None):
            self.base_url = base_url

        def run_and_monitor(self, **kwargs):
            captured_kwargs.update(kwargs)
            return "session-1", {"status": "completed"}

    usd_path = tmp_path / "scene.usda"
    usd_path.write_text("#usda 1.0\n")

    monkeypatch.setattr(client_module, "MaterialAgentClient", FakeClient)

    exit_code = client_module.main(
        [
            "--base-url",
            "http://service",
            "--email",
            "test@example.com",
            "--vlm-max-workers",
            "2",
            "--render-num-workers",
            "1",
            "--quiet",
            str(usd_path),
        ]
    )

    assert exit_code == 0
    assert captured_kwargs["vlm_max_workers"] == 2
    assert captured_kwargs["render_num_workers"] == 1
    assert captured_kwargs["user_email"] == "test@example.com"
    assert captured_kwargs["enable_prim_clustering"] is None
    captured = capsys.readouterr()
    assert "Session: session-1" in captured.out


def test_main_email_is_optional(monkeypatch, tmp_path, capsys):
    from ...client import client as client_module

    captured_kwargs: dict = {}

    class FakeClient:
        def __init__(self, base_url: str, token: str | None = None):
            self.base_url = base_url

        def run_and_monitor(self, **kwargs):
            captured_kwargs.update(kwargs)
            return "session-1", {"status": "completed"}

    usd_path = tmp_path / "scene.usda"
    usd_path.write_text("#usda 1.0\n")

    monkeypatch.setattr(client_module, "MaterialAgentClient", FakeClient)

    exit_code = client_module.main(
        [
            "--base-url",
            "http://service",
            "--quiet",
            str(usd_path),
        ]
    )

    assert exit_code == 0
    assert captured_kwargs["user_email"] == ""
    captured = capsys.readouterr()
    assert "Session: session-1" in captured.out


def test_main_can_explicitly_disable_prim_clustering(monkeypatch, tmp_path, capsys):
    from ...client import client as client_module

    captured_kwargs: dict = {}

    class FakeClient:
        def __init__(self, base_url: str, token: str | None = None):
            self.base_url = base_url

        def run_and_monitor(self, **kwargs):
            captured_kwargs.update(kwargs)
            return "session-1", {"status": "completed"}

    usd_path = tmp_path / "scene.usda"
    usd_path.write_text("#usda 1.0\n")

    monkeypatch.setattr(client_module, "MaterialAgentClient", FakeClient)

    exit_code = client_module.main(
        [
            "--base-url",
            "http://service",
            "--email",
            "test@example.com",
            "--disable-prim-clustering",
            "--quiet",
            str(usd_path),
        ]
    )

    assert exit_code == 0
    assert captured_kwargs["enable_prim_clustering"] is False
    captured = capsys.readouterr()
    assert "Session: session-1" in captured.out


def test_main_passes_prim_clustering_overrides(monkeypatch, tmp_path, capsys):
    from ...client import client as client_module

    captured_kwargs: dict = {}

    class FakeClient:
        def __init__(self, base_url: str, token: str | None = None):
            self.base_url = base_url

        def run_and_monitor(self, **kwargs):
            captured_kwargs.update(kwargs)
            return "session-1", {"status": "completed"}

    usd_path = tmp_path / "scene.usda"
    usd_path.write_text("#usda 1.0\n")

    monkeypatch.setattr(client_module, "MaterialAgentClient", FakeClient)

    exit_code = client_module.main(
        [
            "--base-url",
            "http://service",
            "--email",
            "test@example.com",
            "--enable-prim-clustering",
            "--cluster-min-prims",
            "25",
            "--cluster-embedding-backend",
            "nim",
            "--cluster-embedding-model",
            "nvidia/llama-nemotron-embed-vl-1b-v2",
            "--cluster-embedding-base-url",
            "http://embedding-nim:8000/v1",
            "--cluster-embedding-max-workers",
            "2",
            "--cluster-embedding-batch-size",
            "8",
            "--cluster-max-size",
            "11",
            "--cluster-similarity-threshold-low",
            "0.97",
            "--cluster-similarity-threshold-medium",
            "0.94",
            "--cluster-similarity-threshold-high",
            "0.88",
            "--no-cluster-report",
            "--quiet",
            str(usd_path),
        ]
    )

    assert exit_code == 0
    assert captured_kwargs["enable_prim_clustering"] is True
    assert captured_kwargs["cluster_min_prims"] == 25
    assert captured_kwargs["cluster_embedding_backend"] == "nim"
    assert (
        captured_kwargs["cluster_embedding_model"]
        == "nvidia/llama-nemotron-embed-vl-1b-v2"
    )
    assert (
        captured_kwargs["cluster_embedding_base_url"] == "http://embedding-nim:8000/v1"
    )
    assert captured_kwargs["cluster_embedding_max_workers"] == 2
    assert captured_kwargs["cluster_embedding_batch_size"] == 8
    assert captured_kwargs["cluster_max_size"] == 11
    assert captured_kwargs["cluster_similarity_threshold_low"] == 0.97
    assert captured_kwargs["cluster_similarity_threshold_medium"] == 0.94
    assert captured_kwargs["cluster_similarity_threshold_high"] == 0.88
    assert captured_kwargs["cluster_report"] is False
    captured = capsys.readouterr()
    assert "Session: session-1" in captured.out


def test_main_passes_large_scene_options(monkeypatch, tmp_path, capsys):
    from ...client import client as client_module

    captured_kwargs: dict = {}

    class FakeClient:
        def __init__(self, base_url: str, token: str | None = None):
            self.base_url = base_url

        def run_and_monitor(self, **kwargs):
            captured_kwargs.update(kwargs)
            return "session-1", {"status": "completed"}

    usd_path = tmp_path / "scene.usda"
    usd_path.write_text("#usda 1.0\n")

    monkeypatch.setattr(client_module, "MaterialAgentClient", FakeClient)

    exit_code = client_module.main(
        [
            "--base-url",
            "http://service",
            "--email",
            "test@example.com",
            "--large-scene",
            "--scene-workers",
            "2",
            "--scene-assets",
            "AssetA,/World/AssetB",
            "--scene-resume",
            "--scene-from-step",
            "predict",
            "--scene-skip-existing",
            "--scene-no-render",
            "--scene-simulate",
            "--scene-simulate-mock-analyze",
            "--scene-fail-on-validation-error",
            "--scene-filters-json",
            '{"include_prim_paths": ["/World"]}',
            "--quiet",
            str(usd_path),
        ]
    )

    assert exit_code == 0
    assert captured_kwargs["large_scene"] is True
    assert captured_kwargs["scene_workers"] == 2
    assert captured_kwargs["scene_assets"] == "AssetA,/World/AssetB"
    assert captured_kwargs["scene_resume"] is True
    assert captured_kwargs["scene_from_step"] == "predict"
    assert captured_kwargs["scene_skip_existing"] is True
    assert captured_kwargs["scene_no_render"] is True
    assert captured_kwargs["scene_simulate"] is True
    assert captured_kwargs["scene_simulate_mock_analyze"] is True
    assert captured_kwargs["scene_fail_on_validation_error"] is True
    assert captured_kwargs["scene_filters"] == {"include_prim_paths": ["/World"]}
    captured = capsys.readouterr()
    assert "Scene manifest:" in captured.out
    assert "Predictions JSONL" not in captured.out
    assert "Report HTML" not in captured.out


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
