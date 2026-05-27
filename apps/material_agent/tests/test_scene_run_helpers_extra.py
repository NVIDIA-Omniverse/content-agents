# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Extra focused coverage for material_agent.scene.run helper functions."""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml
from pxr import Sdf, Usd, UsdGeom, UsdShade

from material_agent.scene.manifest import (
    InstanceGroup,
    PayloadGroup,
    SceneManifest,
    SubAsset,
)
from material_agent.scene.run import (
    _clean_working_dir_for_so_retry,
    _clear_pipeline_state_from_step,
    _copy_results_to_duplicates,
    _create_modified_parent_copy,
    _fix_output_material_scope,
    _fix_representative_sublayer,
    _generate_simulate_predictions,
    _run_payloads_parallel,
    _run_payloads_sequential,
    _run_sequential,
    _run_simulate,
    _update_output_paths,
    _update_payload_output_paths,
    run_all_payloads_bottomup,
    run_payload,
    run_sub_asset,
)


@dataclass
class FakePipelineOutput:
    success: bool
    error: str | None = None
    step_results: dict[str, dict[str, Any]] = field(default_factory=dict)
    completed_steps: list[str] = field(default_factory=list)
    skipped_steps: list[str] = field(default_factory=list)
    raw_result: dict[str, Any] | None = None


def _make_sub_asset(
    name: str = "asset_a",
    *,
    status: str = "pending",
    config_path: str | None = None,
    working_dir: str | None = None,
    instance_group: str | None = None,
) -> SubAsset:
    return SubAsset(
        id=str(uuid.uuid4()),
        name=name,
        prim_path=f"/World/{name}",
        status=status,
        config_path=config_path,
        working_dir=working_dir,
        instance_group=instance_group,
    )


def _make_payload_group(
    group_name: str = "payload_a",
    *,
    depth: int = 0,
    status: str = "pending",
    config_path: str | None = None,
    working_dir: str | None = None,
    representative_path: str | None = None,
) -> PayloadGroup:
    return PayloadGroup(
        id=str(uuid.uuid4()),
        group_name=group_name,
        payload_file=f"/tmp/{group_name}.usd",
        depth=depth,
        status=status,
        config_path=config_path,
        working_dir=working_dir,
        representative_path=representative_path,
    )


def _write_config(
    path: Path,
    *,
    session_id: str | None = "test_session",
    extra: dict[str, Any] | None = None,
) -> None:
    cfg: dict[str, Any] = {"project": {}, "steps": {}}
    if session_id is not None:
        cfg["project"]["session_id"] = session_id
    if extra:
        for key, value in extra.items():
            if isinstance(value, dict) and isinstance(cfg.get(key), dict):
                cfg[key].update(value)
            else:
                cfg[key] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")


def _touch(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _create_empty_layer(path: Path, *, sublayers: list[str] | None = None) -> Path:
    layer = Sdf.Layer.CreateNew(str(path))
    if sublayers is not None:
        layer.subLayerPaths = sublayers
    layer.Save()
    return path


def test_clean_working_dir_for_so_retry_removes_artifacts(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, session_id="asset-1")
    working_dir = tmp_path / ".asset-1"

    for dirname in [
        "optimized",
        "dataset",
        "predictions",
        "restored",
        ".pipeline_temp",
    ]:
        _touch(working_dir / dirname / "artifact.txt")
    _touch(working_dir / ".pipeline_state.json", "{}")

    _clean_working_dir_for_so_retry(config_path)

    for dirname in [
        "optimized",
        "dataset",
        "predictions",
        "restored",
        ".pipeline_temp",
    ]:
        assert not (working_dir / dirname).exists()
    assert not (working_dir / ".pipeline_state.json").exists()


def test_clear_pipeline_state_from_step_removes_downstream_steps(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, session_id="asset-2")
    state_file = tmp_path / ".asset-2" / ".pipeline_state.json"
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "completed_steps": [
            "validate_input",
            "optimize_usd",
            "build_dataset_usd",
            "predict",
            "apply",
        ],
        "failed_steps": ["apply"],
        "step_outputs": {
            "optimize_usd": {"optimized_usd_path": "optimized.usd"},
            "build_dataset_usd": {"output_dir": "dataset"},
            "predict": {"predictions_path": "predictions.jsonl"},
            "apply": {"output_usd_path": "output.usd"},
        },
    }
    state_file.write_text(json.dumps(state), encoding="utf-8")

    _clear_pipeline_state_from_step(config_path, "predict")

    updated = json.loads(state_file.read_text(encoding="utf-8"))
    assert updated["completed_steps"] == [
        "validate_input",
        "optimize_usd",
        "build_dataset_usd",
    ]
    assert updated["failed_steps"] == []
    assert "predict" not in updated["step_outputs"]
    assert "apply" not in updated["step_outputs"]
    assert "optimize_usd" in updated["step_outputs"]


def test_clear_pipeline_state_from_step_persists_step_error_cleanup(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, session_id="asset-errors")
    state_file = tmp_path / ".asset-errors" / ".pipeline_state.json"
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "completed_steps": ["validate_input", "optimize_usd"],
        "failed_steps": [],
        "step_outputs": {"optimize_usd": {"optimized_usd_path": "optimized.usd"}},
        "step_errors": {
            "optimize_usd": "keep this older error",
            "predict": "stale prediction failure",
            "apply": "stale apply failure",
        },
    }
    state_file.write_text(json.dumps(state), encoding="utf-8")

    _clear_pipeline_state_from_step(config_path, "predict")

    updated = json.loads(state_file.read_text(encoding="utf-8"))
    assert updated["completed_steps"] == ["validate_input", "optimize_usd"]
    assert updated["step_outputs"] == {
        "optimize_usd": {"optimized_usd_path": "optimized.usd"}
    }
    assert updated["step_errors"] == {"optimize_usd": "keep this older error"}


def test_copy_results_to_duplicates_copies_files_and_status(tmp_path: Path) -> None:
    rep_work = tmp_path / "rep"
    member_work = tmp_path / "member"
    predictions = _touch(rep_work / "predictions" / "predictions.jsonl", "predictions")
    material_layer = _touch(rep_work / "output" / "output.usd", "output")

    representative = _make_sub_asset(
        "rep",
        status="completed",
        working_dir=str(rep_work),
        instance_group="dup_group",
    )
    representative.predictions_path = str(predictions)
    representative.material_layer_path = str(material_layer)

    member = _make_sub_asset(
        "member",
        working_dir=str(member_work),
        instance_group="dup_group",
    )

    manifest = SceneManifest(
        sub_assets=[representative, member],
        instance_groups=[
            InstanceGroup(
                group_name="dup_group",
                representative_id=representative.id,
                member_paths=[representative.prim_path, member.prim_path],
            )
        ],
    )

    _copy_results_to_duplicates(manifest, [member])

    copied_predictions = member_work / "predictions" / predictions.name
    copied_output = member_work / "output" / material_layer.name
    assert copied_predictions.exists()
    assert copied_output.exists()
    assert member.predictions_path == str(copied_predictions)
    assert member.material_layer_path == str(copied_output)
    assert member.status == "completed"


def test_run_sequential_counts_completed_and_failures(tmp_path: Path) -> None:
    assets = [_make_sub_asset("a"), _make_sub_asset("b"), _make_sub_asset("c")]
    manifest = SceneManifest(sub_assets=assets)
    manifest.save = MagicMock()  # type: ignore[method-assign]

    def fake_run_sub_asset(sa, *args, **kwargs):
        if sa.name == "a":
            sa.status = "completed"
            return sa
        if sa.name == "b":
            sa.status = "failed"
            return sa
        raise RuntimeError("boom")

    with patch(
        "material_agent.scene.run.run_sub_asset", side_effect=fake_run_sub_asset
    ):
        completed, failed = _run_sequential(
            assets,
            manifest,
            tmp_path / "manifest.json",
            skip_steps=None,
            only_steps=None,
            verbose=False,
        )

    assert completed == 1
    assert failed == 2
    assert assets[2].status == "failed"
    assert manifest.save.call_count == 3


def test_run_sequential_cancel_checker_stops_between_assets(tmp_path: Path) -> None:
    assets = [_make_sub_asset("a"), _make_sub_asset("b")]
    manifest = SceneManifest(sub_assets=assets)
    manifest.save = MagicMock()  # type: ignore[method-assign]
    processed: list[str] = []

    def fake_run_sub_asset(sa, *args, **kwargs):
        processed.append(sa.name)
        sa.status = "completed"
        return sa

    with patch(
        "material_agent.scene.run.run_sub_asset", side_effect=fake_run_sub_asset
    ):
        with pytest.raises(asyncio.CancelledError):
            _run_sequential(
                assets,
                manifest,
                tmp_path / "manifest.json",
                skip_steps=None,
                only_steps=None,
                verbose=False,
                cancel_checker=lambda: bool(processed),
            )

    assert processed == ["a"]
    assert manifest.save.call_count == 1


def test_generate_simulate_predictions_prefers_optimized_usd(tmp_path: Path) -> None:
    config = {"input": {"usd_path": "scene.usd", "prim_path": "/Root"}}
    config_path = tmp_path / "config.yaml"
    working_dir = tmp_path / ".session"
    optimized = _touch(working_dir / "optimized" / "optimized_input.usd")

    with patch(
        "material_agent.scene.simulate.generate_mock_predictions", return_value=7
    ) as mock_generate:
        result = _generate_simulate_predictions(
            config,
            config_path,
            working_dir,
            ["Steel", "Plastic"],
        )

    assert result == 7
    kwargs = mock_generate.call_args.kwargs
    assert kwargs["usd_path"] == optimized
    assert kwargs["material_names"] == ["Steel", "Plastic"]
    assert kwargs["output_path"] == working_dir / "predictions" / "predictions.jsonl"
    assert kwargs["prim_path_scope"] == "/Root"


def test_update_output_paths_prefers_restored_predictions(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, session_id="asset-3")
    working_dir = tmp_path / ".asset-3"
    restored = _touch(working_dir / "restored" / "restored_predictions.jsonl")
    _touch(working_dir / "predictions" / "predictions.jsonl")
    output = _touch(working_dir / "output" / "output.usd")
    sub_asset = _make_sub_asset("Widget")

    _update_output_paths(sub_asset, config_path)

    assert sub_asset.working_dir == str(working_dir)
    assert sub_asset.predictions_path == str(restored)
    assert sub_asset.material_layer_path == str(output)


def test_update_output_paths_falls_back_to_safe_name_without_session(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, session_id=None)
    working_dir = tmp_path / ".my_asset"
    raw_predictions = _touch(working_dir / "predictions" / "predictions.jsonl")
    sub_asset = _make_sub_asset("My Asset")

    _update_output_paths(sub_asset, config_path)

    assert sub_asset.working_dir == str(working_dir)
    assert sub_asset.predictions_path == str(raw_predictions)


def test_run_simulate_short_circuits_when_no_predictions(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    _write_config(
        config_path,
        session_id="simulate-1",
        extra={
            "input": {"usd_path": "scene.usd"},
            "steps": {"optimize_usd": {"enabled": True}},
        },
    )

    with (
        patch(
            "material_agent.api.pipeline.run_pipeline",
            return_value=FakePipelineOutput(
                success=True, completed_steps=["optimize_usd"]
            ),
        ) as mock_run,
        patch(
            "material_agent.scene.run._generate_simulate_predictions", return_value=0
        ),
    ):
        result = _run_simulate(config_path, ["Steel"], verbose=False)

    assert result.success is True
    assert result.completed_steps == ["optimize_usd"]
    assert mock_run.call_count == 1
    first_input = mock_run.call_args_list[0].args[0]
    assert first_input.only_steps == ["optimize_usd"]
    marker = tmp_path / ".simulate-1" / ".simulate"
    assert marker.exists()


def test_run_simulate_skips_apply_when_apply_disabled(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    _write_config(
        config_path,
        session_id="simulate-2",
        extra={
            "input": {"usd_path": "scene.usd"},
            "steps": {"apply": {"enabled": False}},
        },
    )

    with (
        patch("material_agent.api.pipeline.run_pipeline") as mock_run,
        patch(
            "material_agent.scene.run._generate_simulate_predictions", return_value=3
        ),
    ):
        result = _run_simulate(config_path, ["Steel"], verbose=False)

    assert result.success is True
    assert result.completed_steps == ["predict"]
    assert result.step_results["predict"]["predictions_count"] == 3
    mock_run.assert_not_called()
    marker = tmp_path / ".simulate-2" / ".simulate"
    assert marker.exists()


def test_run_sub_asset_simulate_uses_fast_prediction_path(tmp_path: Path) -> None:
    config_path = tmp_path / "asset.yaml"
    _write_config(config_path, session_id="asset-sim")
    predictions = _touch(
        tmp_path / ".asset-sim" / "predictions" / "predictions.jsonl",
        text='{"prim_path": "/World/asset_1", "material": "Steel"}\n',
    )

    sub_asset = _make_sub_asset("asset_1", config_path=str(config_path))
    with (
        patch(
            "material_agent.scene.run._run_simulate",
            return_value=FakePipelineOutput(success=True, completed_steps=["predict"]),
        ) as mock_simulate,
        patch("material_agent.api.pipeline.run_pipeline") as mock_run,
    ):
        result = run_sub_asset(
            sub_asset,
            simulate=True,
            material_names=["Steel"],
        )

    assert result.status == "completed"
    assert result.predictions_path == str(predictions)
    mock_simulate.assert_called_once_with(
        config_path,
        ["Steel"],
        verbose=False,
        cancel_checker=None,
    )
    mock_run.assert_not_called()


def test_run_sub_asset_forwards_cancel_checker(tmp_path: Path) -> None:
    config_path = tmp_path / "asset.yaml"
    _write_config(config_path, session_id="asset-1")
    seen_inputs: list[Any] = []

    def checker() -> bool:
        return False

    def fake_run_pipeline(params):
        seen_inputs.append(params)
        return FakePipelineOutput(success=True, completed_steps=["predict"])

    sub_asset = _make_sub_asset("asset_1", config_path=str(config_path))
    with patch(
        "material_agent.api.pipeline.run_pipeline", side_effect=fake_run_pipeline
    ):
        result = run_sub_asset(sub_asset, cancel_checker=checker)

    assert result.status == "completed"
    assert seen_inputs[0].cancel_checker is checker


def test_run_payload_forwards_cancel_checker(tmp_path: Path) -> None:
    config_path = tmp_path / "payload.yaml"
    _write_config(config_path, session_id="payload-1")
    seen_inputs: list[Any] = []

    def checker() -> bool:
        return False

    def fake_run_pipeline(params):
        seen_inputs.append(params)
        return FakePipelineOutput(success=True, completed_steps=["predict"])

    payload = _make_payload_group("payload_1", config_path=str(config_path))
    with patch(
        "material_agent.api.pipeline.run_pipeline", side_effect=fake_run_pipeline
    ):
        result = run_payload(payload, cancel_checker=checker)

    assert result.status == "completed"
    assert seen_inputs[0].cancel_checker is checker


def test_fix_output_material_scope_moves_materials_under_default_prim(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "output.usda"
    stage = Usd.Stage.CreateNew(str(output_path))
    root = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(root.GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, "/Asset/Geom/Mesh").GetPrim()
    material = UsdShade.Material.Define(stage, "/World/Looks/TestMaterial")
    UsdShade.MaterialBindingAPI(mesh).Bind(material)
    stage.GetRootLayer().Save()

    payload_group = _make_payload_group("scope_fix")
    payload_group.output_usd_path = str(output_path)

    _fix_output_material_scope(payload_group)

    layer = Sdf.Layer.FindOrOpen(str(output_path))
    assert layer is not None
    assert layer.GetPrimAtPath("/Asset/Looks").typeName == "Scope"
    assert layer.GetPrimAtPath("/Asset/Looks/TestMaterial") is not None
    mesh_spec = layer.GetPrimAtPath("/Asset/Geom/Mesh")
    assert mesh_spec is not None
    targets = mesh_spec.relationships["material:binding"].targetPathList.explicitItems
    assert targets == [Sdf.Path("/Asset/Looks/TestMaterial")]
    assert layer.GetPrimAtPath("/World/Looks/TestMaterial") is None


def test_fix_representative_sublayer_swaps_to_original_payload(tmp_path: Path) -> None:
    original = _create_empty_layer(tmp_path / "original_payload.usda")
    representative = _create_empty_layer(tmp_path / "representative.usda")
    output_path = tmp_path / "out" / "output.usda"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    layer = Sdf.Layer.CreateNew(str(output_path))
    layer.subLayerPaths = [os.path.relpath(representative, output_path.parent)]
    layer.Save()

    payload_group = _make_payload_group("payload")
    payload_group.payload_file = str(original)
    payload_group.representative_path = str(representative)
    payload_group.output_usd_path = str(output_path)

    _fix_representative_sublayer(payload_group)

    updated = Sdf.Layer.FindOrOpen(str(output_path))
    assert updated is not None
    assert updated.subLayerPaths == [os.path.relpath(original, output_path.parent)]


def test_update_payload_output_paths_uses_group_name_when_session_missing(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "payload.yaml"
    _write_config(config_path, session_id=None)
    predictions = _touch(
        tmp_path / ".payload_a" / "predictions" / "predictions.jsonl", "predictions"
    )
    payload = _make_payload_group("payload_a")

    _update_payload_output_paths(payload, config_path)

    assert payload.working_dir == str(tmp_path / ".payload_a")
    assert payload.predictions_path == str(predictions)


def test_create_modified_parent_copy_rewrites_sublayers(tmp_path: Path) -> None:
    child_original = _create_empty_layer(tmp_path / "child.usda")
    child_output = _create_empty_layer(tmp_path / "child_output.usda")
    sublayer_original = _create_empty_layer(tmp_path / "parent_sub.usda")
    parent_original = _create_empty_layer(
        tmp_path / "parent.usda",
        sublayers=[os.path.relpath(sublayer_original, tmp_path)],
    )

    child_group = _make_payload_group("child", status="completed")
    child_group.payload_file = str(child_original)
    child_group.output_usd_path = str(child_output)

    parent_group = _make_payload_group("parent")
    parent_group.payload_file = str(parent_original)
    parent_group.child_payload_files = [str(child_original)]

    manifest = SceneManifest(payload_groups=[child_group, parent_group])

    def fake_rewrite_arcs_in_layer(layer, child_map, resolve_from):
        if Path(resolve_from) == parent_original:
            return 1
        if Path(resolve_from) == sublayer_original:
            return 1
        return 0

    with patch(
        "material_agent.scene.payload_dag_utils.rewrite_arcs_in_layer",
        side_effect=fake_rewrite_arcs_in_layer,
    ):
        _create_modified_parent_copy(parent_group, manifest, tmp_path / "work")

    modified = Path(parent_group.modified_input_path)
    copied_sublayer = modified.parent / sublayer_original.name
    assert modified.exists()
    assert copied_sublayer.exists()
    modified_layer = Sdf.Layer.FindOrOpen(str(modified))
    assert modified_layer is not None
    assert modified_layer.subLayerPaths == [str(copied_sublayer)]


def test_run_payloads_sequential_counts_and_saves(tmp_path: Path) -> None:
    payloads = [
        _make_payload_group("a"),
        _make_payload_group("b"),
        _make_payload_group("c"),
    ]
    manifest = SceneManifest(payload_groups=payloads)
    manifest.save = MagicMock()  # type: ignore[method-assign]

    def fake_run_payload(pg, *args, **kwargs):
        if pg.group_name == "a":
            pg.status = "completed"
            return pg
        if pg.group_name == "b":
            pg.status = "failed"
            return pg
        raise RuntimeError("payload boom")

    with patch("material_agent.scene.run.run_payload", side_effect=fake_run_payload):
        completed, failed = _run_payloads_sequential(
            payloads,
            manifest,
            tmp_path / "manifest.json",
            skip_steps=None,
            only_steps=None,
            verbose=False,
        )

    assert completed == 1
    assert failed == 2
    assert payloads[2].status == "failed"
    assert manifest.save.call_count == 3


def test_run_payloads_sequential_cancel_checker_stops_between_payloads(
    tmp_path: Path,
) -> None:
    payloads = [_make_payload_group("a"), _make_payload_group("b")]
    manifest = SceneManifest(payload_groups=payloads)
    manifest.save = MagicMock()  # type: ignore[method-assign]
    processed: list[str] = []

    def fake_run_payload(pg, *args, **kwargs):
        processed.append(pg.group_name)
        pg.status = "completed"
        return pg

    with patch("material_agent.scene.run.run_payload", side_effect=fake_run_payload):
        with pytest.raises(asyncio.CancelledError):
            _run_payloads_sequential(
                payloads,
                manifest,
                tmp_path / "manifest.json",
                skip_steps=None,
                only_steps=None,
                verbose=False,
                cancel_checker=lambda: bool(processed),
            )

    assert processed == ["a"]
    assert manifest.save.call_count == 1


def test_run_payloads_parallel_updates_manifest(tmp_path: Path) -> None:
    payload_a = _make_payload_group("a")
    payload_b = _make_payload_group("b")
    manifest = SceneManifest(payload_groups=[payload_a, payload_b])
    manifest.save = MagicMock()  # type: ignore[method-assign]

    def fake_worker(pg, *args, **kwargs):
        pg.status = "completed" if pg.group_name == "a" else "failed"
        return pg

    with patch("material_agent.scene.run._run_payload_worker", side_effect=fake_worker):
        completed, failed = _run_payloads_parallel(
            [payload_a, payload_b],
            manifest,
            tmp_path / "manifest.json",
            skip_steps=None,
            only_steps=None,
            verbose=False,
            max_workers=2,
        )

    assert completed == 1
    assert failed == 1
    assert manifest.payload_groups[0].status in {"completed", "failed"}
    assert manifest.payload_groups[1].status in {"completed", "failed"}
    assert manifest.save.call_count == 2


def test_run_all_payloads_bottomup_prepares_parent_depths(tmp_path: Path) -> None:
    configs_dir = tmp_path / "configs"
    leaf = _make_payload_group(
        "leaf",
        depth=0,
        config_path=str(configs_dir / "payloads" / "leaf.yaml"),
        working_dir=str(configs_dir / "payloads" / ".leaf"),
    )
    parent = _make_payload_group(
        "parent",
        depth=1,
        representative_path=str(tmp_path / "representative.usda"),
    )
    manifest = SceneManifest(payload_groups=[leaf, parent])
    manifest.save = MagicMock()  # type: ignore[method-assign]

    def fake_run_payloads_sequential(
        payloads, manifest, manifest_path, *args, **kwargs
    ):
        for pg in payloads:
            if not pg.working_dir:
                continue
            output = _touch(Path(pg.working_dir) / "output" / "output.usd")
            pg.output_usd_path = str(output)
            pg.status = "completed"
        return len(payloads), 0

    with (
        patch(
            "material_agent.scene.run._run_payloads_sequential",
            side_effect=fake_run_payloads_sequential,
        ) as mock_run,
        patch(
            "material_agent.scene.run._create_modified_parent_copy"
        ) as mock_create_parent,
        patch("material_agent.scene.run._fix_output_material_scope") as mock_fix_scope,
        patch("material_agent.scene.run._fix_representative_sublayer") as mock_fix_rep,
        patch(
            "material_agent.scene.config_gen.generate_payload_config"
        ) as mock_generate_config,
    ):
        result = run_all_payloads_bottomup(
            manifest,
            tmp_path / "manifest.json",
            scene_config={"project": {"name": "scene"}},
            configs_dir=configs_dir,
            max_workers=1,
        )

    assert result is manifest
    assert mock_run.call_count == 2
    mock_create_parent.assert_called_once()
    mock_generate_config.assert_called_once()
    assert parent.config_path == str(configs_dir / "payloads" / "parent.yaml")
    assert parent.working_dir == str(configs_dir / "payloads" / ".parent")
    assert leaf.output_usd_path is not None
    assert parent.output_usd_path is not None
    assert mock_fix_scope.call_count == 2
    mock_fix_rep.assert_called_once_with(parent)
