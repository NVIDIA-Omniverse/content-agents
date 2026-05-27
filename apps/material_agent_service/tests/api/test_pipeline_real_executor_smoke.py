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
from material_agent.api.scene_pipeline import ScenePipelineInput, ScenePipelineOutput

_USD_WITH_DEFAULT_ROOT_PRIM = """#usda 1.0
(
    defaultPrim = "Root"
)

def Xform "Root"
{
}
"""


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


@pytest.mark.api
@pytest.mark.real_executor
async def test_large_scene_pipeline_uses_real_executor_with_mocked_scene_api(
    client: httpx.AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    _reset_event_bus: None,
) -> None:
    """Exercise the real scene worker while stubbing only the scene API."""
    from ...service.workers import executor as executor_module

    async def fake_arun_scene_pipeline(
        params: ScenePipelineInput,
    ) -> ScenePipelineOutput:
        assert params.simulate is True
        assert params.simulate_mock_analyze is True
        session_dir = _get_session_dir(params.event_listener)
        scene_dir = session_dir / "scene"
        output_dir = session_dir / "output"
        scene_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        manifest_path = scene_dir / "manifest.json"
        output_path = output_dir / "scene_with_materials.usd"
        flat_output_path = output_dir / "composed_scene_flat.usd"
        render_path = scene_dir / "output" / "composed_scene_front.png"
        asset_predictions_path = scene_dir / "asset_a_predictions.jsonl"
        payload_predictions_path = scene_dir / "payload_predictions.jsonl"
        asset_working_dir = scene_dir / ".asset_a"
        payload_working_dir = scene_dir / ".payload_a"
        asset_working_dir.mkdir(parents=True, exist_ok=True)
        payload_working_dir.mkdir(parents=True, exist_ok=True)
        render_path.parent.mkdir(parents=True, exist_ok=True)
        (asset_working_dir / ".pipeline_state.json").write_text(
            json.dumps({"step_outputs": {"build_dataset_usd": {"num_images": 4}}})
            + "\n"
        )
        (payload_working_dir / ".pipeline_state.json").write_text(
            json.dumps({"step_outputs": {"build_dataset_usd": {"num_images": 2}}})
            + "\n"
        )
        asset_predictions_path.write_text(
            json.dumps({"id": "/Root/AssetA", "material": "Steel"}) + "\n"
        )
        payload_predictions_path.write_text(
            json.dumps({"id": "/Root/PayloadMesh", "material": "Plastic"}) + "\n"
        )
        manifest_data = {
            "sub_assets": [
                {
                    "id": "asset-a",
                    "name": "AssetA",
                    "prim_path": "/Root/AssetA",
                    "predictions_path": str(asset_predictions_path),
                    "working_dir": str(asset_working_dir),
                    "status": "completed",
                }
            ],
            "payload_groups": [
                {
                    "id": "payload-a",
                    "group_name": "PayloadA",
                    "payload_file": str(scene_dir / "payload.usda"),
                    "predictions_path": str(payload_predictions_path),
                    "working_dir": str(payload_working_dir),
                    "status": "completed",
                }
            ],
        }
        manifest_path.write_text(json.dumps(manifest_data) + "\n")
        output_path.write_text("#usda 1.0\n# layered scene\n")
        flat_output_path.write_text("#usda 1.0\n# flattened scene\n")
        render_path.write_bytes(b"\x89PNG\r\n\x1a\nscene-render")

        await _emit_step(
            params.event_listener,
            "scene_analyze",
            progress_message="Detected 2 scene assets",
        )
        await _emit_step(
            params.event_listener,
            "scene_run_assets",
            progress_message="Completed 2 scene assets",
        )
        await _emit_step(
            params.event_listener,
            "scene_collect",
            progress_message="Composed scene output",
        )
        params.event_listener.event(
            "workflow.completed",
            {
                "workflow_type": "scene_pipeline",
                "manifest_path": str(manifest_path),
                "output_usd_path": str(output_path),
                "rendered_images": [str(render_path)],
            },
        )
        await asyncio.sleep(0.01)

        return ScenePipelineOutput(
            success=True,
            working_dir=str(scene_dir),
            manifest_path=str(manifest_path),
            output_usd_path=str(output_path),
            rendered_images=[str(render_path)],
            completed_assets=2,
            failed_assets=0,
            completed_payloads=1,
            failed_payloads=0,
            validation_passed=True,
            validation_report={"errors": [], "warnings": ["checked"]},
            raw_result={"sub_assets": 2, "payload_groups": 1},
        )

    monkeypatch.setattr(
        executor_module,
        "arun_scene_pipeline",
        fake_arun_scene_pipeline,
    )

    response = await client.post(
        "/pipeline",
        files={
            "usd_file": (
                "scene.usda",
                _USD_WITH_DEFAULT_ROOT_PRIM.encode("utf-8"),
                "application/octet-stream",
            )
        },
        data={
            "user_email": "test@example.com",
            "large_scene": "true",
            "scene_workers": "2",
            "scene_simulate": "true",
            "scene_simulate_mock_analyze": "true",
        },
    )

    assert response.status_code == 202
    session_id = response.json()["session_id"]

    final_status = None
    for _ in range(200):
        status_r = await client.get(f"/pipeline/{session_id}/status")
        assert status_r.status_code == 200
        body = status_r.json()
        if body["status"] == "completed":
            final_status = body
            break
        await asyncio.sleep(0.01)

    assert final_status is not None
    assert final_status["overall_progress"]["percent"] == 100
    assert final_status["overall_progress"]["total_steps"] == 9

    results_r = await client.get(f"/pipeline/{session_id}/results")
    assert results_r.status_code == 200
    results = results_r.json()
    assert results["status"] == "completed"
    assert results["stats"]["pipeline_type"] == "large_scene"
    assert results["stats"]["scene_sub_assets_detected"] == 2
    assert results["stats"]["scene_sub_assets_completed"] == 2
    assert results["stats"]["scene_payload_groups_completed"] == 1
    assert results["stats"]["prims_processed"] == 3
    assert results["stats"]["images_generated"] == 7
    assert results["stats"]["scene_asset_image_count"] == 6
    assert results["stats"]["scene_render_count"] == 1
    assert results["stats"]["scene_validation_passed"] is True
    assert results["stats"]["scene_validation_warnings"] == 1
    assert (
        results["download_urls"]["scene_manifest"]
        == f"/artifacts/{session_id}/scene-manifest"
    )
    assert (
        results["download_urls"]["scene_validation_report"]
        == f"/artifacts/{session_id}/scene-validation-report"
    )
    assert (
        results["download_urls"]["scene_predictions"]
        == f"/artifacts/{session_id}/scene-predictions"
    )
    assert (
        results["download_urls"]["final_render"]
        == f"/artifacts/{session_id}/final-render"
    )
    assert "predictions" not in results["download_urls"]
    assert "report" not in results["download_urls"]

    output_r = await client.get(f"/artifacts/{session_id}/output")
    assert output_r.status_code == 200
    assert output_r.content == b"#usda 1.0\n# flattened scene\n"
    final_render_r = await client.get(f"/artifacts/{session_id}/final-render")
    assert final_render_r.status_code == 200
    assert final_render_r.content == b"\x89PNG\r\n\x1a\nscene-render"
    manifest_r = await client.get(f"/artifacts/{session_id}/scene-manifest")
    assert manifest_r.status_code == 200
    manifest_body = manifest_r.json()
    assert manifest_body["sub_assets"][0]["name"] == "AssetA"
    assert manifest_body["payload_groups"][0]["group_name"] == "PayloadA"
    validation_r = await client.get(f"/artifacts/{session_id}/scene-validation-report")
    assert validation_r.status_code == 200
    assert validation_r.json() == {"errors": [], "warnings": ["checked"]}
    scene_predictions_r = await client.get(f"/artifacts/{session_id}/scene-predictions")
    assert scene_predictions_r.status_code == 200
    scene_prediction_records = [
        json.loads(line)
        for line in scene_predictions_r.text.splitlines()
        if line.strip()
    ]
    assert [record["source_type"] for record in scene_prediction_records] == [
        "sub_asset",
        "payload_group",
    ]
    assert scene_prediction_records[0]["prediction"]["material"] == "Steel"
    assert scene_prediction_records[1]["prediction"]["material"] == "Plastic"


@pytest.mark.api
@pytest.mark.real_executor
async def test_large_scene_validation_failure_keeps_report_and_partial_results(
    client: httpx.AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    _reset_event_bus: None,
) -> None:
    """Validation-gated failures should preserve stats and validation artifacts."""
    from ...service.workers import executor as executor_module

    async def fake_arun_scene_pipeline(
        params: ScenePipelineInput,
    ) -> ScenePipelineOutput:
        assert params.fail_on_validation_error is True
        session_dir = _get_session_dir(params.event_listener)
        scene_dir = session_dir / "scene"
        output_dir = session_dir / "output"
        scene_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        manifest_path = scene_dir / "manifest.json"
        output_path = output_dir / "scene_with_materials.usd"
        manifest_path.write_text(
            json.dumps(
                {
                    "sub_assets": [
                        {
                            "id": "asset-a",
                            "name": "AssetA",
                            "prim_path": "/Root/AssetA",
                            "status": "completed",
                        },
                        {
                            "id": "asset-b",
                            "name": "AssetB",
                            "prim_path": "/Root/AssetB",
                            "status": "failed",
                        },
                    ],
                    "payload_groups": [
                        {
                            "id": "payload-a",
                            "group_name": "PayloadA",
                            "payload_file": str(scene_dir / "payload.usda"),
                            "status": "failed",
                        }
                    ],
                }
            )
            + "\n"
        )
        output_path.write_text("#usda 1.0\n")

        await _emit_step(
            params.event_listener,
            "scene_analyze",
            progress_message="Detected 1 scene asset",
        )
        await _emit_step(
            params.event_listener,
            "scene_collect",
            progress_message="Composed scene output",
        )
        await _emit_step(
            params.event_listener,
            "scene_validate",
            progress_message="Validation found 1 error",
        )

        return ScenePipelineOutput(
            success=False,
            error="Scene validation failed",
            working_dir=str(scene_dir),
            manifest_path=str(manifest_path),
            output_usd_path=str(output_path),
            rendered_images=[],
            completed_assets=1,
            failed_assets=1,
            completed_payloads=0,
            failed_payloads=1,
            validation_passed=False,
            validation_report={"errors": ["missing binding"], "warnings": []},
            raw_result={"sub_assets": 2, "payload_groups": 1},
        )

    monkeypatch.setattr(
        executor_module,
        "arun_scene_pipeline",
        fake_arun_scene_pipeline,
    )

    response = await client.post(
        "/pipeline",
        files={
            "usd_file": (
                "scene.usda",
                _USD_WITH_DEFAULT_ROOT_PRIM.encode("utf-8"),
                "application/octet-stream",
            )
        },
        data={
            "user_email": "test@example.com",
            "large_scene": "true",
            "scene_no_render": "true",
            "scene_fail_on_validation_error": "true",
        },
    )

    assert response.status_code == 202
    session_id = response.json()["session_id"]

    final_status = None
    for _ in range(200):
        status_r = await client.get(f"/pipeline/{session_id}/status")
        assert status_r.status_code == 200
        body = status_r.json()
        if body["status"] == "failed":
            final_status = body
            break
        await asyncio.sleep(0.01)

    assert final_status is not None
    assert final_status["status"] == "failed"

    results_r = await client.get(f"/pipeline/{session_id}/results")
    assert results_r.status_code == 200
    results = results_r.json()
    assert results["status"] == "failed"
    assert results["error_message"] == "Scene validation failed"
    assert results["failed_step"] == "scene_validate"
    assert results["partial_results"]["scene_validation_passed"] is False
    assert results["partial_results"]["scene_validation_errors"] == 1
    failed_items = results["partial_results"]["scene_failed_items"]
    assert failed_items[0] == {
        "source_type": "sub_asset",
        "source_id": "asset-b",
        "source_name": "AssetB",
        "source_prim_path": "/Root/AssetB",
    }
    assert failed_items[1]["source_type"] == "payload_group"
    assert failed_items[1]["source_id"] == "payload-a"
    assert failed_items[1]["source_name"] == "PayloadA"
    assert failed_items[1]["source_payload_file"].endswith("payload.usda")

    output_r = await client.get(f"/artifacts/{session_id}/output")
    assert output_r.status_code == 200
    validation_r = await client.get(f"/artifacts/{session_id}/scene-validation-report")
    assert validation_r.status_code == 200
    assert validation_r.json() == {"errors": ["missing binding"], "warnings": []}
