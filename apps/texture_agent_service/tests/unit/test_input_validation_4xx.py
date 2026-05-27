# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""4xx-hygiene regression tests for the texture-agent service (Category A).

Covers three NVBugs against the public 0.3.x release surface:

* **6127692 / OMPE-91855** -- ``/pipeline/{sid}/regenerate`` accepted requests
  for steps that are disabled in the loaded session config (no rendering
  backend in the default docker-compose deploy disables ``render`` and
  ``render_previews``). The workflow factory silently dropped those
  steps, so the API returned 202 with no real work performed.
* **6127699 / OMPE-91858** -- the same endpoint accepted ``{"steps": []}``
  with HTTP 202 and an empty ``"Regenerating steps: "`` message instead
  of 422.
* **6127700 / OMPE-91859** -- ``POST /pipeline`` returned a plain 400 (and
  in the QA repro, dropped the TCP connection mid-multipart) when
  ``material_textures_json`` was malformed, instead of a structured 422
  matching FastAPI's request-validation contract used elsewhere.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from ...service.routers import pipeline_router
from ...service.session.manager import SessionManager


class _ChunkedUpload:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = list(chunks)

    async def read(self, size: int) -> bytes:
        if not self._chunks:
            return b""
        return self._chunks.pop(0)


def _default_steps_disabling_render() -> dict[str, Any]:
    """Mirror of the defaults emitted by build_default_pipeline_config()."""
    return {
        "prepare_uvs": {"enabled": True},
        "discover_materials": {"enabled": True},
        "generate_prompts": {"enabled": True},
        "render_previews": {"enabled": False},
        "generate_textures": {"enabled": True},
        "blend_textures": {"enabled": True},
        "apply_textures": {"enabled": True},
        "render": {"enabled": False},
    }


async def test_stream_copy_removes_oversized_upload_after_close(
    tmp_path: Path,
) -> None:
    dest = tmp_path / "input.usda"
    upload = _ChunkedUpload([b"aaaa", b"bbbb"])

    with pytest.raises(HTTPException) as exc_info:
        await pipeline_router._stream_copy(
            upload,
            dest,
            chunk_size=4,
            max_bytes=5,
        )

    assert exc_info.value.status_code == 413
    assert not dest.exists()


def test_upload_usd_oversize_removes_created_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = SessionManager(tmp_path, ttl_hours=2)
    client = _build_test_client(manager)
    monkeypatch.setattr(pipeline_router.config, "max_upload_size_mb", 1)

    response = client.post(
        "/pipeline/upload-usd",
        files={
            "usd_file": (
                "scene.usd",
                b"#usda 1.0\n" + b"x" * (1024 * 1024),
                "application/octet-stream",
            )
        },
    )

    assert response.status_code == 413
    assert manager.list_sessions() == []


def _seed_completed_session(
    storage_path: Path,
    session_id: str,
    steps_cfg: dict[str, Any],
) -> SessionManager:
    """Create a session in 'completed' status with a config.yaml on disk."""
    manager = SessionManager(storage_path, ttl_hours=2)
    session_dir = manager.create_session(session_id)
    manager.update_session(session_id, {"status": "completed"})

    config_path = session_dir / "input" / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "project": {"session_id": session_id},
                "input": {"usd_path": "scene.usd"},
                "steps": steps_cfg,
            }
        )
    )
    return manager


def _build_test_client(manager: SessionManager) -> TestClient:
    """Build a TestClient with just the pipeline router wired up.

    Bypasses the full FastAPI lifespan (which would touch global config,
    NVIDIA API key checks, periodic cleanup tasks, etc.).
    """
    app = FastAPI()
    pipeline_router.set_session_manager(manager)
    app.include_router(pipeline_router.router)
    return TestClient(app)


def test_regenerate_rejects_disabled_render_step(tmp_path: Path) -> None:
    """Regenerate with a single disabled step returns 422 with a clear detail."""
    sid = "session-render-disabled"
    manager = _seed_completed_session(tmp_path, sid, _default_steps_disabling_render())
    client = _build_test_client(manager)

    response = client.post(f"/pipeline/{sid}/regenerate", json={"steps": ["render"]})

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "render" in detail
    assert "disabled" in detail.lower()


def test_default_pipeline_config_preserves_service_auto_prompting() -> None:
    """Service-created configs should continue auto-prompting missing materials."""
    config = pipeline_router.build_default_pipeline_config(
        session_id="session-auto-prompt",
        usd_path="/tmp/scene.usd",
        working_dir="/tmp/work",
        material_textures={"Steel": {"prompt": "brushed steel"}},
        user_prompt="aged",
    )

    assert config["auto_prompt"]["enabled"] is True
    assert config["auto_prompt"]["user_prompt"] == "aged"


def test_default_pipeline_config_can_disable_service_auto_prompting() -> None:
    """Explicit validation runs can request strict material_textures scope."""
    config = pipeline_router.build_default_pipeline_config(
        session_id="session-strict-scope",
        usd_path="/tmp/scene.usd",
        working_dir="/tmp/work",
        material_textures={"Steel": {"prompt": "brushed steel"}},
        user_prompt="aged",
        auto_prompt_enabled=False,
    )

    assert config["auto_prompt"]["enabled"] is False
    assert config["material_textures"] == {"Steel": {"prompt": "brushed steel"}}


def test_legacy_service_config_migration_preserves_auto_prompting() -> None:
    """Regenerate should keep auto-prompting for configs saved before enabled."""
    config = {"auto_prompt": {"user_prompt": "aged"}}

    pipeline_router._preserve_legacy_service_auto_prompting(config)

    assert config["auto_prompt"]["enabled"] is True


def test_regenerate_rejects_disabled_render_previews_step(tmp_path: Path) -> None:
    """render_previews disabled by default in service deploy must also be rejected."""
    sid = "session-render-previews-disabled"
    manager = _seed_completed_session(tmp_path, sid, _default_steps_disabling_render())
    client = _build_test_client(manager)

    response = client.post(
        f"/pipeline/{sid}/regenerate", json={"steps": ["render_previews"]}
    )

    assert response.status_code == 422
    assert "render_previews" in response.json()["detail"]


def test_regenerate_rejects_mixed_request_listing_all_disabled(tmp_path: Path) -> None:
    """A mixed request (one valid + two disabled) must list every offender."""
    sid = "session-mixed"
    manager = _seed_completed_session(tmp_path, sid, _default_steps_disabling_render())
    client = _build_test_client(manager)

    response = client.post(
        f"/pipeline/{sid}/regenerate",
        json={"steps": ["generate_textures", "render", "render_previews"]},
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    # Both disabled steps are surfaced; the enabled one is not flagged.
    assert "render" in detail
    assert "render_previews" in detail
    assert "generate_textures" not in detail


def test_regenerate_accepts_enabled_step_when_others_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sanity: requesting only an enabled step must NOT 422.

    We stub the job registry so the test does not actually launch the
    pipeline executor -- the validation guard must let the request through
    before registration happens.
    """
    sid = "session-enabled-only"
    manager = _seed_completed_session(tmp_path, sid, _default_steps_disabling_render())

    class _StubRegistry:
        async def register(
            self,
            session_id: str,
            coro: Any,
            *args: Any,
            on_finished: Any = None,
            **kwargs: Any,
        ) -> None:
            # Close the unawaited coroutine to avoid RuntimeWarning. The
            # production registry would schedule it; the test only cares
            # about the 202 acknowledgement.
            coro.close()
            if on_finished is not None:
                on_finished()

    class _StubBus:
        def clear_session_state(self, *args: Any, **kwargs: Any) -> None:
            return None

    monkeypatch.setattr(pipeline_router, "get_job_registry", lambda: _StubRegistry())
    monkeypatch.setattr(pipeline_router, "get_event_bus", lambda: _StubBus())

    client = _build_test_client(manager)
    response = client.post(
        f"/pipeline/{sid}/regenerate", json={"steps": ["generate_textures"]}
    )

    assert response.status_code == 202, response.text


def test_regenerate_hydrates_cache_for_incremental_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sid = "session-regenerate-hydrate-cache"
    manager = _seed_completed_session(tmp_path, sid, _default_steps_disabling_render())
    sync_prefixes: list[str] = []

    def recording_sync_from_store(session_id: str, prefix: str = "") -> int:
        sync_prefixes.append(prefix)
        return 0

    class _StubRegistry:
        async def register(
            self,
            session_id: str,
            coro: Any,
            *args: Any,
            on_finished: Any = None,
            **kwargs: Any,
        ) -> None:
            coro.close()
            if on_finished is not None:
                on_finished()

    class _StubBus:
        def clear_session_state(self, *args: Any, **kwargs: Any) -> None:
            return None

    monkeypatch.setattr(manager, "sync_from_store", recording_sync_from_store)
    monkeypatch.setattr(pipeline_router, "get_job_registry", lambda: _StubRegistry())
    monkeypatch.setattr(pipeline_router, "get_event_bus", lambda: _StubBus())

    client = _build_test_client(manager)
    response = client.post(
        f"/pipeline/{sid}/regenerate", json={"steps": ["generate_textures"]}
    )

    assert response.status_code == 202, response.text
    assert sync_prefixes == ["input/", "cache/"]


def test_regenerate_clears_stale_bus_state_before_register(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A new regenerate run must not let post-register cleanup erase new events."""
    sid = "session-regenerate-clear-before-register"
    manager = _seed_completed_session(tmp_path, sid, _default_steps_disabling_render())
    manager.update_session(
        sid,
        {
            "status": "failed",
            "error": "old failure",
            "failed_step": "generate_textures",
            "failed_step_stats": {"old": True},
            "partial_results": {"old": True},
        },
    )
    calls: list[str] = []

    class _StubBus:
        cleared = False

        def clear_session_state(self, session_id: str) -> None:
            assert session_id == sid
            calls.append("clear")
            self.cleared = True

    bus = _StubBus()

    class _StubRegistry:
        async def register(
            self,
            session_id: str,
            coro: Any,
            *args: Any,
            on_finished: Any = None,
            **kwargs: Any,
        ) -> None:
            calls.append("register")
            assert bus.cleared is True
            metadata = manager.get_session_metadata(session_id)
            assert metadata is not None
            assert metadata["status"] == "pending"
            assert metadata.get("error") is None
            assert metadata.get("failed_step") is None
            coro.close()
            if on_finished is not None:
                on_finished()

    monkeypatch.setattr(pipeline_router, "get_job_registry", lambda: _StubRegistry())
    monkeypatch.setattr(pipeline_router, "get_event_bus", lambda: bus)

    client = _build_test_client(manager)
    response = client.post(
        f"/pipeline/{sid}/regenerate", json={"steps": ["generate_textures"]}
    )

    assert response.status_code == 202, response.text
    assert calls == ["clear", "register"]


def test_regenerate_register_failure_restores_prior_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Clearing before register still rolls disk diagnostics back on failure."""
    sid = "session-regenerate-register-fails"
    manager = _seed_completed_session(tmp_path, sid, _default_steps_disabling_render())
    old_diagnostics = {
        "status": "failed",
        "error": "old failure",
        "failed_step": "generate_textures",
        "failed_step_stats": {"old": True},
        "failed_at": "2026-04-30T00:00:00+00:00",
        "partial_results": {"old": True},
    }
    manager.update_session(sid, old_diagnostics)

    class _StubBus:
        def clear_session_state(self, session_id: str) -> None:
            assert session_id == sid

    class _FailingRegistry:
        async def register(
            self,
            session_id: str,
            coro: Any,
            *args: Any,
            **kwargs: Any,
        ) -> None:
            coro.close()
            raise RuntimeError("synthetic register failure")

    monkeypatch.setattr(pipeline_router, "get_job_registry", lambda: _FailingRegistry())
    monkeypatch.setattr(pipeline_router, "get_event_bus", lambda: _StubBus())

    client = _build_test_client(manager)
    with pytest.raises(RuntimeError, match="synthetic register failure"):
        client.post(
            f"/pipeline/{sid}/regenerate", json={"steps": ["generate_textures"]}
        )

    metadata = manager.get_session_metadata(sid)
    assert metadata is not None
    for key, value in old_diagnostics.items():
        assert metadata[key] == value
    assert manager.is_worker_active(sid) is False


def test_create_existing_session_rejects_worker_lock(tmp_path: Path) -> None:
    """A draining worker lock blocks same-session pipeline restart."""
    sid = "session-worker-locked"
    manager = SessionManager(tmp_path, ttl_hours=2)
    manager.create_session(sid)
    client = _build_test_client(manager)

    with manager.worker_lock(sid):
        response = client.post("/pipeline", data={"session_id": sid})

    assert response.status_code == 409
    assert "worker" in response.json()["detail"].lower()


def test_create_existing_shared_session_defers_hydration_until_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Submitting a shared session should not download input before 202."""
    from ...service.storage import LocalSessionStore

    sid = "session-defer-hydration"
    shared_store = LocalSessionStore(str(tmp_path / "shared"))
    manager = SessionManager(tmp_path / "pod", ttl_hours=2, store=shared_store)
    manager.create_session(sid)
    manager.update_session(
        sid,
        {
            "config": {
                "has_usd_upload": True,
                "input_extension": ".usd",
                "original_filename": "scene.usd",
            }
        },
    )
    released: list[str] = []
    sync_called = False
    real_release_worker_lock = manager.release_worker_lock

    def failing_sync_from_store(session_id: str, prefix: str = "") -> int:
        nonlocal sync_called
        sync_called = True
        raise RuntimeError("synthetic hydration failure")

    def recording_release(worker_lock: Any, session_id: str) -> None:
        released.append(session_id)
        real_release_worker_lock(worker_lock, session_id)

    class _StubRegistry:
        def is_running(self, session_id: str) -> bool:
            return False

        async def register(
            self,
            session_id: str,
            coro: Any,
            *args: Any,
            on_finished: Any = None,
            **kwargs: Any,
        ) -> None:
            coro.close()
            if on_finished is not None:
                on_finished()

    monkeypatch.setattr(manager, "sync_from_store", failing_sync_from_store)
    monkeypatch.setattr(manager, "release_worker_lock", recording_release)
    monkeypatch.setattr(pipeline_router, "get_job_registry", lambda: _StubRegistry())

    client = _build_test_client(manager)
    response = client.post("/pipeline", data={"session_id": sid})

    assert response.status_code == 202, response.text
    assert sync_called is False
    assert released == [sid]


def test_create_existing_session_reserves_worker_lock_before_202(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An accepted existing-session job blocks cross-process deletion immediately."""
    sid = "session-reserve-before-ack"
    manager = SessionManager(tmp_path, ttl_hours=2)
    manager.create_session(sid)
    session_dir = manager.get_session_dir(sid)
    (session_dir / "input" / "scene.usd").write_text("#usda 1.0\n", encoding="utf-8")
    peer_manager = SessionManager(tmp_path, ttl_hours=2)
    observed: dict[str, bool] = {}
    real_find_input_usd = pipeline_router._find_input_usd

    def racing_find_input_usd(session_dir: Path) -> Path | None:
        observed["delete_blocked_before_read"] = (
            peer_manager.delete_session(sid) is False
        )
        return real_find_input_usd(session_dir)

    class _StubRegistry:
        def is_running(self, session_id: str) -> bool:
            return False

        async def register(
            self,
            session_id: str,
            coro: Any,
            *args: Any,
            on_finished: Any = None,
            **kwargs: Any,
        ) -> None:
            observed["delete_blocked"] = (
                peer_manager.delete_session(session_id) is False
            )
            coro.close()
            if on_finished is not None:
                on_finished()

    monkeypatch.setattr(pipeline_router, "get_job_registry", lambda: _StubRegistry())
    monkeypatch.setattr(pipeline_router, "_find_input_usd", racing_find_input_usd)

    client = _build_test_client(manager)
    response = client.post("/pipeline", data={"session_id": sid})

    assert response.status_code == 202, response.text
    assert observed["delete_blocked_before_read"] is True
    assert observed["delete_blocked"] is True
    with peer_manager.worker_lock(sid, timeout=0):
        pass
    assert manager.session_exists(sid) is True


def test_regenerate_reserves_worker_lock_before_session_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regeneration blocks deletion before reading metadata/config from disk."""
    sid = "session-regenerate-reserve-before-read"
    manager = _seed_completed_session(tmp_path, sid, _default_steps_disabling_render())
    peer_manager = SessionManager(tmp_path, ttl_hours=2)
    observed: dict[str, bool] = {}
    real_get_metadata = manager.get_session_metadata

    def racing_get_metadata(session_id: str) -> dict[str, Any] | None:
        if session_id == sid and "delete_blocked_before_read" not in observed:
            observed["delete_blocked_before_read"] = (
                peer_manager.delete_session(session_id) is False
            )
        return real_get_metadata(session_id)

    class _StubRegistry:
        async def register(
            self,
            session_id: str,
            coro: Any,
            *args: Any,
            on_finished: Any = None,
            **kwargs: Any,
        ) -> None:
            coro.close()
            if on_finished is not None:
                on_finished()

    class _StubBus:
        def clear_session_state(self, *args: Any, **kwargs: Any) -> None:
            return None

    monkeypatch.setattr(manager, "get_session_metadata", racing_get_metadata)
    monkeypatch.setattr(pipeline_router, "get_job_registry", lambda: _StubRegistry())
    monkeypatch.setattr(pipeline_router, "get_event_bus", lambda: _StubBus())

    client = _build_test_client(manager)
    response = client.post(
        f"/pipeline/{sid}/regenerate", json={"steps": ["generate_textures"]}
    )

    assert response.status_code == 202, response.text
    assert observed["delete_blocked_before_read"] is True
    with peer_manager.worker_lock(sid, timeout=0):
        pass
    assert manager.session_exists(sid) is True


def test_regenerate_rejects_worker_lock(tmp_path: Path) -> None:
    """A draining worker lock blocks same-session regeneration."""
    sid = "session-regenerate-worker-locked"
    manager = _seed_completed_session(tmp_path, sid, _default_steps_disabling_render())
    client = _build_test_client(manager)

    with manager.worker_lock(sid):
        response = client.post(
            f"/pipeline/{sid}/regenerate", json={"steps": ["generate_textures"]}
        )

    assert response.status_code == 409
    assert "worker" in response.json()["detail"].lower()


def test_regenerate_missing_session_does_not_create_session_dir(
    tmp_path: Path,
) -> None:
    """A 404 regenerate request must not leave hidden lock-only sessions."""
    sid = "missing-regenerate-session"
    manager = SessionManager(tmp_path, ttl_hours=2)
    client = _build_test_client(manager)

    response = client.post(
        f"/pipeline/{sid}/regenerate", json={"steps": ["generate_textures"]}
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Session not found"
    assert not (tmp_path / sid).exists()


# ---------------------------------------------------------------------------
# NVBug 6127699 / OMPE-91858 -- empty steps[] must be 422, not 202.
# ---------------------------------------------------------------------------


def test_regenerate_rejects_empty_steps_array(tmp_path: Path) -> None:
    """``{"steps": []}`` must hit pydantic's min_length validator and 422."""
    sid = "session-empty-steps"
    manager = _seed_completed_session(tmp_path, sid, _default_steps_disabling_render())
    client = _build_test_client(manager)

    response = client.post(f"/pipeline/{sid}/regenerate", json={"steps": []})

    assert response.status_code == 422
    detail = response.json()["detail"]
    # FastAPI/pydantic v2 emits a list of structured errors for body
    # validation failures. The min_length=1 violation is type=too_short.
    assert isinstance(detail, list)
    assert any(item.get("type") == "too_short" for item in detail)
    assert any(item.get("loc") == ["body", "steps"] for item in detail)


# ---------------------------------------------------------------------------
# NVBug 6127700 / OMPE-91859 -- malformed material_textures_json must be a
# structured 422, not a plain 400 (or worse, a connection drop).
# ---------------------------------------------------------------------------


def _make_minimal_usd_bytes() -> bytes:
    """Return a tiny but-valid .usda payload for multipart upload tests."""
    return b'#usda 1.0\n(\n    defaultPrim = "World"\n)\n\ndef Xform "World" {}\n'


def test_create_pipeline_rejects_malformed_material_textures_json(
    tmp_path: Path,
) -> None:
    """Malformed JSON in the form field returns a structured 422 detail list."""
    sid = "session-for-malformed-json"
    manager = _seed_completed_session(tmp_path, sid, _default_steps_disabling_render())
    # Reuse the existing session so the handler does not try to download
    # / persist a fresh upload through the broader filesystem path.
    client = _build_test_client(manager)

    response = client.post(
        "/pipeline",
        files={
            "usd_file": (
                "scene.usda",
                _make_minimal_usd_bytes(),
                "application/octet-stream",
            ),
        },
        data={
            "session_id": sid,
            "material_textures_json": "NOT_JSON_AT_ALL",
        },
    )

    assert response.status_code == 422, response.text
    detail = response.json()["detail"]
    assert isinstance(detail, list)
    assert any(item.get("type") == "json_invalid" for item in detail)
    assert any(item.get("loc") == ["form", "material_textures_json"] for item in detail)


def test_create_pipeline_accepts_empty_material_textures_json(
    tmp_path: Path,
) -> None:
    """Sanity: empty / whitespace-only material_textures_json must not 422.

    The existing parser intentionally treats an empty form value as "no
    overrides", and customers rely on that. The 6127700 fix tightens
    *malformed* JSON without regressing the empty-string allowance.
    """
    sid = "session-for-empty-json"
    manager = _seed_completed_session(tmp_path, sid, _default_steps_disabling_render())
    client = _build_test_client(manager)

    response = client.post(
        "/pipeline",
        files={
            "usd_file": (
                "scene.usda",
                _make_minimal_usd_bytes(),
                "application/octet-stream",
            ),
        },
        data={"session_id": sid, "material_textures_json": "   "},
    )

    # The handler accepts the request and proceeds toward pipeline
    # registration. We only assert that *parsing* did not 422 -- a
    # downstream failure (e.g., 409 because the executor stub is not
    # wired) is fine for this test's contract.
    assert response.status_code != 422, response.text


# ---------------------------------------------------------------------------
# NVBug 6127700 follow-up (Codex review of !415) -- syntactically valid but
# structurally invalid material_textures_json must also 422 at submit, not
# fail asynchronously inside the pipeline after the 202.
# ---------------------------------------------------------------------------


def _post_with_material_textures_json(
    client: TestClient, sid: str, payload: str
) -> Any:
    return client.post(
        "/pipeline",
        files={
            "usd_file": (
                "scene.usda",
                _make_minimal_usd_bytes(),
                "application/octet-stream",
            ),
        },
        data={"session_id": sid, "material_textures_json": payload},
    )


def test_create_pipeline_rejects_top_level_list_material_textures(
    tmp_path: Path,
) -> None:
    """``[]`` is valid JSON but the wire shape is dict[str, dict]."""
    sid = "session-mt-list"
    manager = _seed_completed_session(tmp_path, sid, _default_steps_disabling_render())
    client = _build_test_client(manager)

    response = _post_with_material_textures_json(client, sid, "[]")

    assert response.status_code == 422, response.text
    detail = response.json()["detail"]
    assert isinstance(detail, list)
    assert any(item.get("type") == "dict_type" for item in detail)
    assert any(item.get("loc") == ["form", "material_textures_json"] for item in detail)


def test_create_pipeline_rejects_top_level_scalar_material_textures(
    tmp_path: Path,
) -> None:
    """``42`` decodes to int -- not a dict, must 422."""
    sid = "session-mt-scalar"
    manager = _seed_completed_session(tmp_path, sid, _default_steps_disabling_render())
    client = _build_test_client(manager)

    response = _post_with_material_textures_json(client, sid, "42")

    assert response.status_code == 422, response.text
    detail = response.json()["detail"]
    assert any(item.get("type") == "dict_type" for item in detail)


def test_create_pipeline_rejects_dict_with_scalar_value_material_textures(
    tmp_path: Path,
) -> None:
    """``{"Steel":"rust"}`` -- top level is dict but the value is a str.

    Per-material overrides must themselves be objects (prompt/opacity
    fields). A scalar leaf value would crash the prompt-expansion code
    later in the pipeline.
    """
    sid = "session-mt-scalar-leaf"
    manager = _seed_completed_session(tmp_path, sid, _default_steps_disabling_render())
    client = _build_test_client(manager)

    response = _post_with_material_textures_json(
        client, sid, '{"Steel": "rust", "Wood": {"prompt": "ok"}}'
    )

    assert response.status_code == 422, response.text
    detail = response.json()["detail"]
    assert any(item.get("type") == "dict_type" for item in detail)
    # The offending key is named in the loc; the well-formed key is not.
    assert any(
        item.get("loc") == ["form", "material_textures_json", "Steel"]
        for item in detail
    )
    assert not any(
        item.get("loc") == ["form", "material_textures_json", "Wood"] for item in detail
    )


def test_create_pipeline_accepts_empty_dict_material_textures(
    tmp_path: Path,
) -> None:
    """``{}`` is the documented "no overrides" shape -- must not 422."""
    sid = "session-mt-empty-dict"
    manager = _seed_completed_session(tmp_path, sid, _default_steps_disabling_render())
    client = _build_test_client(manager)

    response = _post_with_material_textures_json(client, sid, "{}")

    assert response.status_code != 422, response.text


def test_create_pipeline_rejects_material_missing_prompt(
    tmp_path: Path,
) -> None:
    """An explicit material override needs a prompt before job acceptance."""
    sid = "session-mt-missing-prompt"
    manager = _seed_completed_session(tmp_path, sid, _default_steps_disabling_render())
    client = _build_test_client(manager)

    response = _post_with_material_textures_json(client, sid, '{"Steel": {}}')

    assert response.status_code == 422, response.text
    detail = response.json()["detail"]
    assert any(
        item.get("loc") == ["form", "material_textures_json", "Steel", "prompt"]
        and item.get("type") == "missing"
        for item in detail
    )


def test_create_pipeline_rejects_material_prompt_list(
    tmp_path: Path,
) -> None:
    """Prompt must be a non-empty string, not an arbitrary JSON value."""
    sid = "session-mt-list-prompt"
    manager = _seed_completed_session(tmp_path, sid, _default_steps_disabling_render())
    client = _build_test_client(manager)

    response = _post_with_material_textures_json(
        client, sid, '{"Steel": {"prompt": ["rust"]}}'
    )

    assert response.status_code == 422, response.text
    detail = response.json()["detail"]
    assert any(
        item.get("loc") == ["form", "material_textures_json", "Steel", "prompt"]
        for item in detail
    )


def test_create_pipeline_rejects_out_of_range_material_opacity(
    tmp_path: Path,
) -> None:
    """Opacity must be numeric and bounded before the job is registered."""
    sid = "session-mt-bad-opacity"
    manager = _seed_completed_session(tmp_path, sid, _default_steps_disabling_render())
    client = _build_test_client(manager)

    response = _post_with_material_textures_json(
        client, sid, '{"Steel": {"prompt": "rust", "opacity": 1.5}}'
    )

    assert response.status_code == 422, response.text
    detail = response.json()["detail"]
    assert any(
        item.get("loc") == ["form", "material_textures_json", "Steel", "opacity"]
        for item in detail
    )


def test_create_pipeline_rejects_unknown_material_override_field(
    tmp_path: Path,
) -> None:
    """Unknown material override fields are rejected before job acceptance."""
    sid = "session-mt-extra-field"
    manager = _seed_completed_session(tmp_path, sid, _default_steps_disabling_render())
    client = _build_test_client(manager)

    response = _post_with_material_textures_json(
        client, sid, '{"Steel": {"prompt": "rust", "roughness": 0.2}}'
    )

    assert response.status_code == 422, response.text
    detail = response.json()["detail"]
    assert any(
        item.get("loc") == ["form", "material_textures_json", "Steel", "roughness"]
        and item.get("type") == "extra_forbidden"
        for item in detail
    )


def test_create_pipeline_rejects_blank_material_key(
    tmp_path: Path,
) -> None:
    """Material override keys must name a real discovered material."""
    sid = "session-mt-blank-material"
    manager = _seed_completed_session(tmp_path, sid, _default_steps_disabling_render())
    client = _build_test_client(manager)

    response = _post_with_material_textures_json(
        client, sid, '{"   ": {"prompt": "rust"}}'
    )

    assert response.status_code == 422, response.text
    detail = response.json()["detail"]
    assert any(
        "Material override keys must be non-empty" in item.get("msg", "")
        for item in detail
    )


def test_create_pipeline_rejects_blank_per_prim_key(
    tmp_path: Path,
) -> None:
    """Per-prim override keys must identify a prim path or leaf name."""
    sid = "session-mt-blank-prim"
    manager = _seed_completed_session(tmp_path, sid, _default_steps_disabling_render())
    client = _build_test_client(manager)

    response = _post_with_material_textures_json(
        client,
        sid,
        '{"Steel": {"prompt": "rust", "per_prim": {"   ": {"opacity": 0.5}}}}',
    )

    assert response.status_code == 422, response.text
    detail = response.json()["detail"]
    assert any(
        "Per-prim override keys must be non-empty" in item.get("msg", "")
        for item in detail
    )


def test_create_pipeline_accepts_per_prim_and_enables_per_prim_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A documented per_prim override must switch the generated config mode."""
    sid = "session-mt-per-prim"
    manager = _seed_completed_session(tmp_path, sid, _default_steps_disabling_render())
    session_dir = manager.get_session_dir(sid)
    (session_dir / "input" / "scene.usd").write_text("#usda 1.0\n", encoding="utf-8")
    captured: dict[str, Any] = {}

    class _StubRegistry:
        def is_running(self, session_id: str) -> bool:
            return False

        async def register(
            self,
            session_id: str,
            coro: Any,
            *args: Any,
            on_finished: Any = None,
            **kwargs: Any,
        ) -> None:
            coro.close()
            if on_finished is not None:
                on_finished()

    def fake_execute_pipeline_async(**kwargs: Any) -> Any:
        captured["config"] = kwargs["config_dict"]

        async def noop() -> None:
            return None

        return noop()

    monkeypatch.setattr(pipeline_router, "get_job_registry", lambda: _StubRegistry())
    monkeypatch.setattr(
        pipeline_router, "execute_pipeline_async", fake_execute_pipeline_async
    )
    client = _build_test_client(manager)

    response = _post_with_material_textures_json(
        client,
        sid,
        (
            '{"Steel": {"prompt": "rust", "per_prim": '
            '{"/World/Rung_01": {"opacity": 0.65}}}}'
        ),
    )

    assert response.status_code == 202, response.text
    config = captured["config"]
    assert config["texture"]["mode"] == "per_prim"
    assert (
        config["material_textures"]["Steel"]["per_prim"]["/World/Rung_01"]["opacity"]
        == 0.65
    )


def test_create_pipeline_material_only_override_keeps_default_texture_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Material-only overrides should not opt a new run into per-prim mode."""
    sid = "session-mt-material-only"
    manager = _seed_completed_session(tmp_path, sid, _default_steps_disabling_render())
    session_dir = manager.get_session_dir(sid)
    (session_dir / "input" / "scene.usd").write_text("#usda 1.0\n", encoding="utf-8")
    captured: dict[str, Any] = {}

    class _StubRegistry:
        def is_running(self, session_id: str) -> bool:
            return False

        async def register(
            self,
            session_id: str,
            coro: Any,
            *args: Any,
            on_finished: Any = None,
            **kwargs: Any,
        ) -> None:
            coro.close()
            if on_finished is not None:
                on_finished()

    def fake_execute_pipeline_async(**kwargs: Any) -> Any:
        captured["config"] = kwargs["config_dict"]

        async def noop() -> None:
            return None

        return noop()

    monkeypatch.setattr(pipeline_router, "get_job_registry", lambda: _StubRegistry())
    monkeypatch.setattr(
        pipeline_router, "execute_pipeline_async", fake_execute_pipeline_async
    )
    client = _build_test_client(manager)

    response = _post_with_material_textures_json(
        client,
        sid,
        '{"Steel": {"prompt": "brushed steel", "opacity": 0.75}}',
    )

    assert response.status_code == 202, response.text
    assert "mode" not in captured["config"]["texture"]


@pytest.mark.parametrize(
    ("form_value", "expected_enabled"),
    [("false", False), ("true", True)],
)
def test_create_pipeline_auto_prompt_enabled_form_sets_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    form_value: str,
    expected_enabled: bool,
) -> None:
    """REST clients can explicitly choose strict or auto-prompting scope."""
    sid = f"session-mt-auto-prompt-{form_value}"
    manager = _seed_completed_session(tmp_path, sid, _default_steps_disabling_render())
    session_dir = manager.get_session_dir(sid)
    (session_dir / "input" / "scene.usd").write_text("#usda 1.0\n", encoding="utf-8")
    captured: dict[str, Any] = {}

    class _StubRegistry:
        def is_running(self, session_id: str) -> bool:
            return False

        async def register(
            self,
            session_id: str,
            coro: Any,
            *args: Any,
            on_finished: Any = None,
            **kwargs: Any,
        ) -> None:
            coro.close()
            if on_finished is not None:
                on_finished()

    def fake_execute_pipeline_async(**kwargs: Any) -> Any:
        captured["config"] = kwargs["config_dict"]

        async def noop() -> None:
            return None

        return noop()

    monkeypatch.setattr(pipeline_router, "get_job_registry", lambda: _StubRegistry())
    monkeypatch.setattr(
        pipeline_router, "execute_pipeline_async", fake_execute_pipeline_async
    )
    client = _build_test_client(manager)

    response = client.post(
        "/pipeline",
        files={
            "usd_file": (
                "scene.usda",
                _make_minimal_usd_bytes(),
                "application/octet-stream",
            ),
        },
        data={
            "session_id": sid,
            "material_textures_json": (
                '{"Aluminum_Matte": {"prompt": "weathered aluminum"}}'
            ),
            "auto_prompt_enabled": form_value,
        },
    )

    assert response.status_code == 202, response.text
    assert captured["config"]["auto_prompt"]["enabled"] is expected_enabled
    assert captured["config"]["material_textures"] == {
        "Aluminum_Matte": {"prompt": "weathered aluminum"}
    }


def test_regenerate_rejects_invalid_material_textures(
    tmp_path: Path,
) -> None:
    """Regenerate uses the same material override schema as POST /pipeline."""
    sid = "session-regenerate-mt-invalid"
    manager = _seed_completed_session(tmp_path, sid, _default_steps_disabling_render())
    client = _build_test_client(manager)

    response = client.post(
        f"/pipeline/{sid}/regenerate",
        json={
            "steps": ["generate_textures"],
            "material_textures": {"Steel": {"opacity": "opaque"}},
        },
    )

    assert response.status_code == 422, response.text
    detail = response.json()["detail"]
    assert any(
        item.get("loc") == ["body", "material_textures", "Steel", "prompt"]
        for item in detail
    )
    assert any(
        item.get("loc") == ["body", "material_textures", "Steel", "opacity"]
        for item in detail
    )


def test_regenerate_per_prim_override_enables_per_prim_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regenerate uses per_prim overrides to switch texture mode too."""
    sid = "session-regenerate-mt-per-prim"
    manager = _seed_completed_session(tmp_path, sid, _default_steps_disabling_render())
    captured: dict[str, Any] = {}

    class _StubRegistry:
        async def register(
            self,
            session_id: str,
            coro: Any,
            *args: Any,
            on_finished: Any = None,
            **kwargs: Any,
        ) -> None:
            coro.close()
            if on_finished is not None:
                on_finished()

    class _StubBus:
        def clear_session_state(self, *args: Any, **kwargs: Any) -> None:
            return None

    def fake_execute_pipeline_async(**kwargs: Any) -> Any:
        captured["config"] = kwargs["config_dict"]

        async def noop() -> None:
            return None

        return noop()

    monkeypatch.setattr(pipeline_router, "get_job_registry", lambda: _StubRegistry())
    monkeypatch.setattr(pipeline_router, "get_event_bus", lambda: _StubBus())
    monkeypatch.setattr(
        pipeline_router, "execute_pipeline_async", fake_execute_pipeline_async
    )
    client = _build_test_client(manager)

    response = client.post(
        f"/pipeline/{sid}/regenerate",
        json={
            "steps": ["generate_textures"],
            "material_textures": {
                "Steel": {
                    "prompt": "rust",
                    "per_prim": {"/World/Rung_01": {"opacity": 0.65}},
                }
            },
        },
    )

    assert response.status_code == 202, response.text
    assert captured["config"]["texture"]["mode"] == "per_prim"


def test_regenerate_material_only_override_preserves_per_prim_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regenerate should not downgrade a stored per-prim session config."""
    sid = "session-regenerate-mt-preserve-per-prim"
    manager = _seed_completed_session(tmp_path, sid, _default_steps_disabling_render())
    session_dir = manager.get_session_dir(sid)
    config_path = session_dir / "input" / "config.yaml"
    config_data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config_data["texture"] = {"mode": "per_prim", "backend": "mock"}
    config_path.write_text(yaml.safe_dump(config_data), encoding="utf-8")
    captured: dict[str, Any] = {}

    class _StubRegistry:
        async def register(
            self,
            session_id: str,
            coro: Any,
            *args: Any,
            on_finished: Any = None,
            **kwargs: Any,
        ) -> None:
            coro.close()
            if on_finished is not None:
                on_finished()

    class _StubBus:
        def clear_session_state(self, *args: Any, **kwargs: Any) -> None:
            return None

    def fake_execute_pipeline_async(**kwargs: Any) -> Any:
        captured["config"] = kwargs["config_dict"]

        async def noop() -> None:
            return None

        return noop()

    monkeypatch.setattr(pipeline_router, "get_job_registry", lambda: _StubRegistry())
    monkeypatch.setattr(pipeline_router, "get_event_bus", lambda: _StubBus())
    monkeypatch.setattr(
        pipeline_router, "execute_pipeline_async", fake_execute_pipeline_async
    )
    client = _build_test_client(manager)

    response = client.post(
        f"/pipeline/{sid}/regenerate",
        json={
            "steps": ["generate_textures"],
            "material_textures": {
                "Steel": {"prompt": "brushed steel", "opacity": 0.75}
            },
        },
    )

    assert response.status_code == 202, response.text
    assert captured["config"]["texture"]["mode"] == "per_prim"
    assert captured["config"]["material_textures"]["Steel"] == {
        "prompt": "brushed steel",
        "opacity": 0.75,
    }
