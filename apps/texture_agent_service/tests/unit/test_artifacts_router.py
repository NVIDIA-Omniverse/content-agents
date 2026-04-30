# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from ...service.routers import artifacts_router
from ...service.session.manager import SessionManager


def _build_artifact_app(tmp_path: Path) -> tuple[TestClient, str]:
    sid = "artifact-session"
    manager = SessionManager(tmp_path)
    session_dir = manager.create_session(sid)

    (session_dir / "cache" / "discovery" / "materials.json").write_text(
        "[]",
        encoding="utf-8",
    )
    (session_dir / "cache" / "textures" / "albedo.png").write_bytes(b"texture")
    (session_dir / "cache" / "output" / "textured_output.usdz").write_bytes(
        b"usdz",
    )
    (session_dir / "cache" / "renders" / "final.png").write_bytes(b"render")
    (session_dir / "preview" / "preview.png").write_bytes(b"preview")

    artifacts_router.set_session_manager(manager)
    app = FastAPI()
    app.include_router(artifacts_router.router)
    return TestClient(app), sid


def test_artifact_endpoints_return_documented_media_types(tmp_path: Path) -> None:
    client, sid = _build_artifact_app(tmp_path)

    expected = {
        f"/artifacts/{sid}/materials": "application/json",
        f"/artifacts/{sid}/textures": "application/zip",
        f"/artifacts/{sid}/textures/albedo.png": "image/png",
        f"/artifacts/{sid}/output": "model/vnd.usdz+zip",
        f"/artifacts/{sid}/renders": "application/zip",
        f"/artifacts/{sid}/renders/final.png": "image/png",
        f"/artifacts/{sid}/preview/preview.png": "image/png",
    }

    for path, media_type in expected.items():
        response = client.get(path)

        assert response.status_code == 200
        assert response.headers["content-type"] == media_type


def test_missing_artifacts_return_json_errors(tmp_path: Path) -> None:
    client, sid = _build_artifact_app(tmp_path)

    expected = {
        f"/artifacts/{sid}-missing/materials": (404, "Session not found"),
        f"/artifacts/{sid}/textures/missing.png": (
            404,
            "Texture file not found: missing.png",
        ),
        f"/artifacts/{sid}/renders/missing.png": (
            404,
            "Render file not found: missing.png",
        ),
        f"/artifacts/{sid}/preview/missing.png": (
            404,
            "Preview file not found: missing.png",
        ),
        f"/artifacts/{sid}/textures/%5Cescape.png": (400, "Invalid filename"),
        f"/artifacts/{sid}/renders/%5Cescape.png": (400, "Invalid filename"),
        f"/artifacts/{sid}/preview/%5Cescape.png": (400, "Invalid filename"),
    }

    for path, (status_code, detail) in expected.items():
        response = client.get(path)

        assert response.status_code == status_code
        assert response.headers["content-type"] == "application/json"
        assert response.json()["detail"] == detail


def test_invalid_session_id_artifact_route_returns_not_found(tmp_path: Path) -> None:
    client, _ = _build_artifact_app(tmp_path)

    response = client.get("/artifacts/%2E%2E/materials")

    assert response.status_code == 404
    assert response.headers["content-type"] == "application/json"
    assert response.json()["detail"] == "Session not found"
    assert tmp_path.exists()


def test_unavailable_artifact_collections_return_json_errors(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path)
    sid = "empty-artifact-session"
    session_dir = manager.create_session(sid)

    (session_dir / "cache" / "textures").rmdir()
    (session_dir / "cache" / "renders").rmdir()

    artifacts_router.set_session_manager(manager)
    app = FastAPI()
    app.include_router(artifacts_router.router)
    client = TestClient(app)

    expected = {
        f"/artifacts/{sid}/materials": "Materials data not available",
        f"/artifacts/{sid}/textures": "Textures not available",
        f"/artifacts/{sid}/textures/albedo.png": "Textures not available",
        f"/artifacts/{sid}/output": "Output USDZ not available",
        f"/artifacts/{sid}/renders": "Renders not available",
        f"/artifacts/{sid}/renders/final.png": "Renders not available",
    }

    for path, detail in expected.items():
        response = client.get(path)

        assert response.status_code == 404
        assert response.headers["content-type"] == "application/json"
        assert response.json()["detail"] == detail


def test_artifact_openapi_documents_binary_media_types(tmp_path: Path) -> None:
    client, _ = _build_artifact_app(tmp_path)
    paths = client.app.openapi()["paths"]

    expected = {
        "/artifacts/{session_id}/textures": "application/zip",
        "/artifacts/{session_id}/textures/{filename}": "image/png",
        "/artifacts/{session_id}/output": "model/vnd.usdz+zip",
        "/artifacts/{session_id}/renders": "application/zip",
        "/artifacts/{session_id}/renders/{filename}": "image/png",
        "/artifacts/{session_id}/preview/{filename}": "image/png",
    }

    for path, media_type in expected.items():
        content = paths[path]["get"]["responses"]["200"]["content"]

        assert list(content) == [media_type]
        assert content[media_type]["schema"] == {
            "type": "string",
            "format": "binary",
        }
