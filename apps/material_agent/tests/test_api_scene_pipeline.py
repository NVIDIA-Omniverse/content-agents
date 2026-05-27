# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the public large-scene Python API."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
import yaml

from material_agent.api.scene_pipeline import ScenePipelineInput, run_scene_pipeline
from material_agent.scene.manifest import SceneManifest, SubAsset
from material_agent.scene.validate import SceneReport


class RecordingListener:
    """Minimal listener for API event assertions."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []
        self.warnings: list[str] = []

    def event(self, event_type: str, data: dict[str, Any], **kwargs: Any) -> None:
        self.events.append((event_type, data))

    def info(self, message: str, **kwargs: Any) -> None:
        pass

    def debug(self, message: str, **kwargs: Any) -> None:
        pass

    def warning(self, message: str, **kwargs: Any) -> None:
        self.warnings.append(message)

    def error(self, message: str, **kwargs: Any) -> None:
        pass


def _scene_config(tmp_path: Path) -> dict[str, Any]:
    usd_path = tmp_path / "scene.usda"
    usd_path.write_text(
        """#usda 1.0
(
    defaultPrim = "Root"
)

def Xform "Root"
{
}
"""
    )
    material_lib = tmp_path / "materials.usda"
    material_lib.write_text("#usda 1.0\n")

    return {
        "project": {
            "name": "scene_api",
            "working_dir": str(tmp_path / "scene_work"),
        },
        "input": {"usd_path": str(usd_path)},
        "materials": {
            "library_path": str(material_lib),
            "entries": [
                {
                    "name": "Steel",
                    "description": "Test steel",
                    "prim_path": "/World/Looks/Steel",
                }
            ],
        },
        "scene": {
            "extract": {"flatten": True, "max_workers": 1},
            "reconcile": {"enabled": False},
            "harmonize": {"enabled": False},
        },
        "steps": {
            "render": {"enabled": False},
        },
    }


def test_run_scene_pipeline_orchestrates_and_materializes_inline_materials(
    monkeypatch,
    tmp_path: Path,
) -> None:
    manifest = SceneManifest(
        scene_usd_path=str(tmp_path / "scene.usda"),
        sub_assets=[
            SubAsset(
                id="asset_a",
                name="AssetA",
                prim_path="/Root/AssetA",
                mesh_count=2,
            )
        ],
    )
    called: dict[str, Any] = {}

    def fake_analyze_scene(**kwargs: Any) -> SceneManifest:
        called["analyze"] = kwargs
        return manifest

    def fake_extract_all(**kwargs: Any) -> SceneManifest:
        called["extract"] = kwargs
        for sub_asset in manifest.sub_assets:
            sub_asset.extracted_usd = str(tmp_path / "asset.usda")
            sub_asset.status = "extracted"
        return manifest

    def fake_generate_all_configs(**kwargs: Any) -> SceneManifest:
        called["config_gen"] = kwargs
        config_path = tmp_path / "asset.yaml"
        config_path.write_text("project:\n  session_id: asset_a\n")
        for sub_asset in manifest.sub_assets:
            sub_asset.config_path = str(config_path)
            sub_asset.working_dir = str(tmp_path / ".asset_a")
        return manifest

    def fake_run_all(**kwargs: Any) -> SceneManifest:
        called["run_all"] = kwargs
        for sub_asset in manifest.sub_assets:
            sub_asset.status = "completed"
        progress_callback = kwargs.get("progress_callback")
        if progress_callback:
            progress_callback(
                {
                    "current": 1,
                    "total": 1,
                    "completed": 1,
                    "failed": 0,
                    "asset_id": "asset_a",
                    "asset_name": "AssetA",
                    "asset_status": "completed",
                }
            )
        return manifest

    def fake_apply_and_compose(**kwargs: Any) -> Path:
        called["collect"] = kwargs
        output_path = Path(kwargs["output_usd_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("#usda 1.0\n")
        return output_path

    monkeypatch.setattr(
        "material_agent.scene.analyze.analyze_scene",
        fake_analyze_scene,
    )
    monkeypatch.setattr(
        "material_agent.scene.extract.extract_all",
        fake_extract_all,
    )
    monkeypatch.setattr(
        "material_agent.scene.config_gen.generate_all_configs",
        fake_generate_all_configs,
    )
    monkeypatch.setattr(
        "material_agent.scene.run.run_all",
        fake_run_all,
    )
    monkeypatch.setattr(
        "material_agent.scene.collect.apply_and_compose",
        fake_apply_and_compose,
    )
    listener = RecordingListener()

    result = run_scene_pipeline(
        ScenePipelineInput(
            config=_scene_config(tmp_path),
            config_base_dir=tmp_path,
            no_render=True,
            validate_output=False,
            event_listener=listener,
        )
    )

    assert result.success
    assert result.completed_assets == 1
    assert result.failed_assets == 0
    assert Path(result.output_usd_path).exists()

    material_yaml = Path(called["collect"]["material_library_yaml"])
    material_data = yaml.safe_load(material_yaml.read_text())
    assert material_data["library_path"] == "../../materials.usda"
    assert material_data["entries"][0]["binding"] == "/World/Looks/Steel"
    assert "prim_path" in material_data["entries"][0]
    assert called["config_gen"]["scene_config"]["materials"] == {
        "path": str(material_yaml)
    }

    assert called["extract"]["flatten"] is True
    assert called["config_gen"]["scene_config_dir"] == tmp_path

    completed_scene_steps = [
        data["step_name"]
        for event_type, data in listener.events
        if event_type == "step.completed"
        and data.get("workflow_type") == "scene_pipeline"
    ]
    assert completed_scene_steps == [
        "scene_analyze",
        "scene_extract",
        "scene_run_assets",
        "scene_run_payloads",
        "scene_reconcile",
        "scene_harmonize",
        "scene_collect",
        "scene_render",
        "scene_validate",
    ]
    asset_progress_events = [
        data
        for event_type, data in listener.events
        if event_type == "step.progress" and data.get("step_name") == "scene_run_assets"
    ]
    assert asset_progress_events
    assert asset_progress_events[-1]["current"] == 1
    assert asset_progress_events[-1]["total"] == 1
    assert asset_progress_events[-1]["percent"] == 100
    assert asset_progress_events[-1]["asset_name"] == "AssetA"


def test_run_scene_pipeline_validates_explicit_output_paths(
    monkeypatch,
    tmp_path: Path,
) -> None:
    manifest = SceneManifest(
        scene_usd_path=str(tmp_path / "scene.usda"),
        sub_assets=[
            SubAsset(
                id="asset_a",
                name="AssetA",
                prim_path="/Root/AssetA",
                mesh_count=1,
            )
        ],
    )
    called: dict[str, Any] = {}

    monkeypatch.setattr(
        "material_agent.scene.analyze.analyze_scene",
        lambda **kwargs: manifest,
    )

    def fake_extract_all(**kwargs: Any) -> SceneManifest:
        for sub_asset in manifest.sub_assets:
            sub_asset.extracted_usd = str(tmp_path / "asset.usda")
            sub_asset.status = "extracted"
        return manifest

    def fake_generate_all_configs(**kwargs: Any) -> SceneManifest:
        config_path = tmp_path / "asset.yaml"
        config_path.write_text("project:\n  session_id: asset_a\n")
        for sub_asset in manifest.sub_assets:
            sub_asset.config_path = str(config_path)
            sub_asset.working_dir = str(tmp_path / ".asset_a")
        return manifest

    def fake_run_all(**kwargs: Any) -> SceneManifest:
        for sub_asset in manifest.sub_assets:
            sub_asset.status = "completed"
        return manifest

    def fake_apply_and_compose(**kwargs: Any) -> Path:
        output_path = Path(kwargs["output_usd_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("#usda 1.0\n")
        return output_path

    def fake_validate_scene_outputs(**kwargs: Any) -> SceneReport:
        called["validate"] = kwargs
        return SceneReport()

    monkeypatch.setattr("material_agent.scene.extract.extract_all", fake_extract_all)
    monkeypatch.setattr(
        "material_agent.scene.config_gen.generate_all_configs",
        fake_generate_all_configs,
    )
    monkeypatch.setattr("material_agent.scene.run.run_all", fake_run_all)
    monkeypatch.setattr(
        "material_agent.scene.collect.apply_and_compose",
        fake_apply_and_compose,
    )
    monkeypatch.setattr(
        "material_agent.scene.validate.validate_scene_outputs",
        fake_validate_scene_outputs,
    )

    result = run_scene_pipeline(
        ScenePipelineInput(
            config=_scene_config(tmp_path),
            config_base_dir=tmp_path,
            output_usd_path=Path("exports/scene_with_materials.usd"),
            no_render=True,
            validate_output=True,
        )
    )

    assert result.success
    assert result.validation_passed is True
    assert Path(called["validate"]["manifest_path"]) == Path(result.manifest_path)
    assert Path(called["validate"]["working_dir"]) == Path(result.working_dir)
    assert Path(called["validate"]["composed_scene_path"]) == Path(
        result.output_usd_path
    )
    assert Path(result.output_usd_path) == tmp_path / "exports" / (
        "scene_with_materials.usd"
    )


def test_run_scene_pipeline_rebases_relative_config_and_output_paths(
    monkeypatch,
    tmp_path: Path,
) -> None:
    base_dir = tmp_path / "project"
    (base_dir / "assets").mkdir(parents=True)
    (base_dir / "libs").mkdir()
    usd_path = base_dir / "assets" / "scene.usda"
    usd_path.write_text(
        """#usda 1.0
(
    defaultPrim = "Root"
)

def Xform "Root"
{
}
"""
    )
    material_lib = base_dir / "libs" / "materials.usda"
    material_lib.write_text("#usda 1.0\n")

    config = {
        "project": {"name": "relative_scene", "working_dir": "work"},
        "input": {"usd_path": "assets/scene.usda"},
        "materials": {
            "library_path": "libs/materials.usda",
            "entries": [
                {
                    "name": "Steel",
                    "description": "Test steel",
                    "prim_path": "/World/Looks/Steel",
                }
            ],
        },
        "scene": {
            "extract": {"flatten": True, "max_workers": 1},
            "reconcile": {"enabled": False},
            "harmonize": {"enabled": False},
        },
        "steps": {"render": {"enabled": False}},
    }
    manifest = SceneManifest(
        scene_usd_path=str(usd_path),
        sub_assets=[SubAsset(id="asset_a", name="AssetA", prim_path="/Root/AssetA")],
    )
    called: dict[str, Any] = {}

    def fake_analyze_scene(**kwargs: Any) -> SceneManifest:
        called["analyze"] = kwargs
        return manifest

    def fake_extract_all(**kwargs: Any) -> SceneManifest:
        called["extract"] = kwargs
        for sub_asset in manifest.sub_assets:
            sub_asset.extracted_usd = str(base_dir / "asset.usda")
            sub_asset.status = "extracted"
        return manifest

    def fake_generate_all_configs(**kwargs: Any) -> SceneManifest:
        called["config_gen"] = kwargs
        for sub_asset in manifest.sub_assets:
            sub_asset.config_path = str(base_dir / "work" / "configs" / "asset.yaml")
            sub_asset.working_dir = str(base_dir / "work" / "configs" / ".asset")
        return manifest

    def fake_run_all(**kwargs: Any) -> SceneManifest:
        called["run_all"] = kwargs
        for sub_asset in manifest.sub_assets:
            sub_asset.status = "completed"
        return manifest

    def fake_apply_and_compose(**kwargs: Any) -> Path:
        called["collect"] = kwargs
        output_path = Path(kwargs["output_usd_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("#usda 1.0\n")
        return output_path

    monkeypatch.setattr(
        "material_agent.scene.analyze.analyze_scene",
        fake_analyze_scene,
    )
    monkeypatch.setattr(
        "material_agent.scene.extract.extract_all",
        fake_extract_all,
    )
    monkeypatch.setattr(
        "material_agent.scene.config_gen.generate_all_configs",
        fake_generate_all_configs,
    )
    monkeypatch.setattr(
        "material_agent.scene.run.run_all",
        fake_run_all,
    )
    monkeypatch.setattr(
        "material_agent.scene.collect.apply_and_compose",
        fake_apply_and_compose,
    )

    result = run_scene_pipeline(
        ScenePipelineInput(
            config=config,
            config_base_dir=base_dir,
            output_usd_path=Path("exports/scene_with_materials.usd"),
            no_render=True,
            validate_output=False,
        )
    )

    assert result.success
    assert Path(called["analyze"]["scene_usd_path"]) == usd_path
    assert called["config_gen"]["scene_config_dir"] == base_dir
    assert Path(called["collect"]["output_usd_path"]) == (
        base_dir / "exports" / "scene_with_materials.usd"
    )
    assert Path(result.output_usd_path) == (
        base_dir / "exports" / "scene_with_materials.usd"
    )
    material_yaml = Path(called["collect"]["material_library_yaml"])
    material_data = yaml.safe_load(material_yaml.read_text())
    assert material_yaml == base_dir / "work" / "materials" / "materials.yaml"
    assert material_data["library_path"] == "../../libs/materials.usda"


def test_run_scene_pipeline_from_step_sets_resume_and_skip_steps(
    monkeypatch,
    tmp_path: Path,
) -> None:
    manifest = SceneManifest(
        sub_assets=[SubAsset(id="asset_a", name="AssetA", prim_path="/Root/AssetA")],
    )
    called: dict[str, Any] = {}

    monkeypatch.setattr(
        "material_agent.scene.analyze.analyze_scene",
        lambda **kwargs: manifest,
    )
    monkeypatch.setattr(
        "material_agent.scene.extract.extract_all",
        lambda **kwargs: manifest,
    )
    monkeypatch.setattr(
        "material_agent.scene.config_gen.generate_all_configs",
        lambda **kwargs: manifest,
    )

    def fake_run_all(**kwargs: Any) -> SceneManifest:
        called["run_all"] = kwargs
        for sub_asset in manifest.sub_assets:
            sub_asset.status = "completed"
        return manifest

    monkeypatch.setattr("material_agent.scene.run.run_all", fake_run_all)

    def fake_collect(**kwargs: Any) -> Path:
        output_path = Path(kwargs["output_usd_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("#usda 1.0\n")
        return output_path

    monkeypatch.setattr(
        "material_agent.scene.collect.apply_and_compose",
        fake_collect,
    )

    result = run_scene_pipeline(
        ScenePipelineInput(
            config=_scene_config(tmp_path),
            config_base_dir=tmp_path,
            from_step="predict",
            skip_steps=["render"],
            no_render=True,
            validate_output=False,
        )
    )

    assert result.success
    assert called["run_all"]["resume"] is True
    assert "build_dataset_prepare_dataset" in called["run_all"]["skip_steps"]
    assert "cluster_prims" in called["run_all"]["skip_steps"]
    assert "render" in called["run_all"]["skip_steps"]


def test_run_scene_pipeline_requires_default_root_prim(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = _scene_config(tmp_path)
    Path(config["input"]["usd_path"]).write_text("#usda 1.0\n")

    def fail_analyze_scene(**kwargs: Any) -> SceneManifest:
        raise AssertionError("analyze_scene should not run for invalid input")

    monkeypatch.setattr(
        "material_agent.scene.analyze.analyze_scene",
        fail_analyze_scene,
    )

    result = run_scene_pipeline(
        ScenePipelineInput(
            config=config,
            config_base_dir=tmp_path,
            no_render=True,
            validate_output=False,
        )
    )

    assert not result.success
    assert result.error is not None
    assert "default root prim" in result.error
    assert "collection of USD files" in result.error


def test_run_scene_pipeline_cancel_checker_stops_before_stages(
    tmp_path: Path,
) -> None:
    listener = RecordingListener()

    with pytest.raises(asyncio.CancelledError):
        run_scene_pipeline(
            ScenePipelineInput(
                config=_scene_config(tmp_path),
                config_base_dir=tmp_path,
                cancel_checker=lambda: True,
                no_render=True,
                validate_output=False,
                event_listener=listener,
            )
        )

    assert (
        "workflow.cancelled",
        {
            "workflow_type": "scene_pipeline",
            "step_name": "scene_pipeline",
            "message": "Scene pipeline cancellation requested",
        },
    ) in listener.events
