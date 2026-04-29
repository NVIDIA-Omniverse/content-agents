# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Smoke test for the real Material Agent Service executor path.

Runs the actual service worker and only fakes the material-agent pipeline API
underneath it. This gives coverage for the service's async job execution,
session lifecycle, status reporting, and artifact serving without pulling in
heavy pipeline dependencies.
"""

import asyncio
import json
from collections.abc import Generator
from pathlib import Path
from typing import Any

import httpx
import pytest
from material_agent.api.pipeline import PipelineInput, PipelineOutput


def _get_session_dir(event_listener: Any) -> Path:
    inner_listener = getattr(event_listener, "_inner", event_listener)
    session_dir = getattr(inner_listener, "session_dir", None)
    assert session_dir is not None
    return Path(session_dir)


async def _emit_step(listener: Any, step_name: str, *, progress_message: str) -> None:
    listener.event(
        "step.started",
        {"step_name": step_name, "message": f"Starting {step_name}"},
    )
    await asyncio.sleep(0.01)
    progress_event_type = (
        "prediction.completed" if step_name == "predict" else "step.progress"
    )
    listener.event(
        progress_event_type,
        {
            "step_name": step_name,
            "current": 1,
            "total": 1,
            "percent": 100,
            "message": progress_message,
        },
    )
    await asyncio.sleep(0.01)
    listener.event(
        "step.completed",
        {"step_name": step_name, "message": f"Completed {step_name}"},
    )
    await asyncio.sleep(0.01)


@pytest.fixture
def _reset_event_bus() -> Generator[None, None, None]:
    from ...service.runtime import bus as bus_module

    bus_module._event_bus = None
    yield
    bus_module._event_bus = None


@pytest.mark.api
@pytest.mark.real_executor
async def test_pipeline_uses_real_executor_with_mocked_material_agent(
    client: httpx.AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    _reset_event_bus: None,
) -> None:
    """Exercise the real worker path while stubbing only MAA's pipeline API."""
    from ...service.workers import executor as executor_module

    async def fake_arun_pipeline(params: PipelineInput) -> PipelineOutput:
        session_dir = _get_session_dir(params.event_listener)
        dataset_dir = session_dir / "cache" / "dataset"
        predictions_dir = session_dir / "cache" / "predictions"
        output_dir = session_dir / "output"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        predictions_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        (dataset_dir / "dataset.jsonl").write_text(
            "\n".join(
                [
                    json.dumps(
                        {"id": "/Root/Cube", "images": {"composition": "cube.png"}}
                    ),
                    json.dumps(
                        {"id": "/Root/Sphere", "images": {"composition": "sphere.png"}}
                    ),
                ]
            )
            + "\n"
        )
        (predictions_dir / "predictions.jsonl").write_text(
            "\n".join(
                [
                    json.dumps({"id": "/Root/Cube", "material": "Aluminum"}),
                    json.dumps({"id": "/Root/Sphere", "material": "Plastic"}),
                ]
            )
            + "\n"
        )
        (output_dir / "scene_with_materials.usd").write_text("#usda 1.0\n")

        await _emit_step(
            params.event_listener,
            "build_dataset_usd",
            progress_message="Rendered 2 views",
        )
        await _emit_step(
            params.event_listener,
            "predict",
            progress_message="Predicted 2 materials",
        )
        await _emit_step(
            params.event_listener,
            "apply",
            progress_message="Applied 2 materials",
        )
        params.event_listener.event(
            "workflow.completed",
            {
                "workflow_type": "pipeline",
                "completed_steps": ["build_dataset_usd", "predict", "apply"],
            },
        )
        await asyncio.sleep(0.01)

        return PipelineOutput(
            success=True,
            step_results={
                "build_dataset_usd": {"num_prims": 2, "num_images": 2},
                "predict": {"predictions_count": 2},
                "apply": {
                    "materials_applied": {
                        "Aluminum": ["/Root/Cube"],
                        "Plastic": ["/Root/Sphere"],
                    }
                },
            },
            completed_steps=["build_dataset_usd", "predict", "apply"],
            raw_result={
                "pipeline_results": {
                    "build_dataset_usd": {"num_prims": 2, "num_images": 2},
                    "predict": {"predictions_count": 2},
                    "apply": {
                        "materials_applied": {
                            "Aluminum": ["/Root/Cube"],
                            "Plastic": ["/Root/Sphere"],
                        }
                    },
                }
            },
        )

    monkeypatch.setattr(executor_module, "arun_pipeline", fake_arun_pipeline)

    response = await client.post(
        "/pipeline",
        files={"usd_file": ("scene.usda", b"#usda 1.0\n", "application/octet-stream")},
        data={"user_email": "test@example.com"},
    )

    assert response.status_code == 202
    session_id = response.json()["session_id"]

    seen_statuses: list[str] = []
    final_status = None
    for _ in range(200):
        status_r = await client.get(f"/pipeline/{session_id}/status")
        assert status_r.status_code == 200
        body = status_r.json()
        seen_statuses.append(body["status"])
        if body["status"] == "completed":
            final_status = body
            break
        await asyncio.sleep(0.01)

    assert final_status is not None
    assert "running" in seen_statuses
    assert final_status["overall_progress"]["percent"] == 100
    assert [step["name"] for step in final_status["completed_steps"]] == [
        "build_dataset_usd",
        "predict",
        "apply",
    ]

    results_r = await client.get(f"/pipeline/{session_id}/results")
    assert results_r.status_code == 200
    results = results_r.json()
    assert results["status"] == "completed"
    assert results["stats"]["original_prim_count"] == 2
    assert results["stats"]["prims_processed"] == 2
    assert results["stats"]["predictions_made"] == 2
    assert results["stats"]["materials_applied"] == 2

    predictions_r = await client.get(f"/artifacts/{session_id}/predictions")
    assert predictions_r.status_code == 200
    output_r = await client.get(f"/artifacts/{session_id}/output")
    assert output_r.status_code == 200
