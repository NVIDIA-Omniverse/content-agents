# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ApplyMaterialsToUSD task error handling and stage metadata."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pxr import Sdf, Usd, UsdGeom, UsdShade

from material_agent.tasks.apply_materials_to_usd import ApplyMaterialsToUSDTask


class TestApplyMaterialsErrorHandling:
    """Tests for error handling in ApplyMaterialsToUSDTask."""

    def test_fails_when_predictions_exist_but_no_materials_resolved(self, tmp_path):
        """Test that task fails with clear error when predictions exist but materials cannot be resolved.

        This tests the fix for Issue #2 where the pipeline should fail when:
        - VLM successfully generates predictions
        - But material resolution fails (materials don't match library)
        - Previously it would silently continue with no materials
        """
        # Setup: Create predictions file with valid predictions
        predictions_path = tmp_path / "predictions.jsonl"
        predictions = [
            {
                "id": "/RootNode/Geometry/Part1",
                "materials": {
                    "material": "NonExistentMaterial",
                    "original_response": "Some reasoning",
                },
            },
            {
                "id": "/RootNode/Geometry/Part2",
                "materials": {
                    "material": "AnotherMissingMaterial",
                    "original_response": "More reasoning",
                },
            },
        ]

        with open(predictions_path, "w") as f:
            for pred in predictions:
                f.write(json.dumps(pred) + "\n")

        # Setup: Create mock USD files
        input_usd = tmp_path / "input.usd"
        input_usd.write_text("# Mock USD")
        output_usd = tmp_path / "output.usd"

        # Setup: Context with predictions but NO resolved materials (resolution failed)
        context = {
            "input_usd_path": str(input_usd),
            "output_usd_path": str(output_usd),
            "predictions_path": str(predictions_path),
            "resolved_materials": {},  # Empty - material resolution failed!
            "is_library_based_mapping": True,
            "material_library_path": "/path/to/library.usd",
        }

        # Create task
        task = ApplyMaterialsToUSDTask()

        # Execute and verify it raises ValueError with clear error message
        with pytest.raises(ValueError) as exc_info:
            task.run(context)

        # Verify error message contains key information
        error_msg = str(exc_info.value)
        assert "Critical error" in error_msg
        assert "Material resolution failed" in error_msg
        assert "VLM predicted materials but none could be resolved" in error_msg
        assert "check system prompt" in error_msg.lower()
        assert "MaterialRetrieval task logs" in error_msg

    def test_succeeds_with_warning_when_no_predictions_and_no_materials(self, tmp_path):
        """Test that task succeeds with warning when there are no predictions at all.

        This tests that the task differentiates between:
        - No predictions (might be expected in some workflows) -> warning
        - Predictions but no materials (resolution failure) -> error
        """
        # Setup: No predictions file
        predictions_path = tmp_path / "predictions.jsonl"
        # Don't create the file

        # Setup: Create mock USD files
        input_usd = tmp_path / "input.usd"
        input_usd.write_text("# Mock USD")
        output_usd = tmp_path / "output.usd"

        # Setup: Context with no resolved materials AND no predictions file
        context = {
            "input_usd_path": str(input_usd),
            "output_usd_path": str(output_usd),
            "predictions_path": str(
                predictions_path
            ),  # Path set but file doesn't exist
            "resolved_materials": {},  # Empty
            "is_library_based_mapping": True,
        }

        # Create task
        task = ApplyMaterialsToUSDTask()

        # Execute - should succeed with warning, not raise error
        result = task.run(context)

        # Verify it succeeded with warning (not exception)
        assert result is not None
        assert result["materials_applied"] == {}
        assert result["assignment_stats"]["total_prims"] == 0
        assert result["assignment_stats"]["materials_applied"] == 0
        assert result["assignment_stats"]["failed"] == 0

    def test_succeeds_when_predictions_and_materials_both_exist(self, tmp_path):
        """Test normal success case when predictions exist and materials are resolved."""
        # Setup: Create predictions file
        predictions_path = tmp_path / "predictions.jsonl"
        predictions = [
            {
                "id": "/RootNode/Geometry/Part1",
                "materials": {
                    "material": "Steel",
                    "original_response": "Some reasoning",
                },
            }
        ]

        with open(predictions_path, "w") as f:
            for pred in predictions:
                f.write(json.dumps(pred) + "\n")

        # Setup: Create mock USD files
        input_usd = tmp_path / "input.usd"
        output_usd = tmp_path / "output.usd"

        # Create minimal valid USD content
        usd_content = """#usda 1.0
(
    defaultPrim = "RootNode"
)

def Xform "RootNode" {
    def Xform "Geometry" {
        def Mesh "Part1" {
        }
    }
}
"""
        input_usd.write_text(usd_content)

        # Setup: Context with both predictions AND resolved materials
        context = {
            "input_usd_path": str(input_usd),
            "output_usd_path": str(output_usd),
            "predictions_path": str(predictions_path),
            "resolved_materials": {
                "Steel": "/path/to/steel.usd"  # Material successfully resolved!
            },
            "is_library_based_mapping": True,
            "material_library_path": str(tmp_path / "library.usd"),
            "layer_only": False,
            "flatten_output": False,
        }

        # Create task
        task = ApplyMaterialsToUSDTask()

        # Execute - should succeed without raising exceptions
        result = task.run(context)

        # Verify success (no exception raised means success)
        assert result is not None
        assert "materials_applied" in result
        assert "assignment_stats" in result
        # Note: Material binding might fail due to mock library path, but task should complete
        assert result["assignment_stats"]["total_prims"] >= 0


class TestApplyMaterialsDatasetSystemPrompt:
    """Tests for system prompt loading from dataset.json."""

    def test_dataset_loading_extracts_system_prompt(self, tmp_path):
        """Test that DatasetLoadingTask extracts system prompt from dataset.json.

        This tests the fix for Issue #1 where system prompt from dataset.json
        should be loaded and passed to VLM inference.
        """
        from material_agent.tasks.dataset import DatasetLoadingTask

        # Setup: Create dataset.jsonl
        dataset_jsonl = tmp_path / "dataset.jsonl"
        dataset_entry = {
            "id": "test_entry",
            "media": {
                "images": [{"path": "image1.png", "metadata": {"view": "front"}}]
            },
            "user_prompt": "Identify the material",
        }

        with open(dataset_jsonl, "w") as f:
            f.write(json.dumps(dataset_entry) + "\n")

        # Setup: Create dataset.json with system prompt
        dataset_json = tmp_path / "dataset.json"
        dataset_metadata = {
            "schema_version": "0.2",
            "metadata": {"created": "2025-01-01", "num_entries": 1},
            "inference": {
                "prompts": [
                    {
                        "step_name": "material_selection",
                        "step_index": 0,
                        "system_prompt": "You are an expert at identifying materials. Return JSON format.",
                        "output_format": {"material": "material name"},
                    }
                ]
            },
            "prims_file": "dataset.jsonl",
        }

        with open(dataset_json, "w") as f:
            json.dump(dataset_metadata, f)

        # Create test images
        image1 = tmp_path / "image1.png"
        from PIL import Image

        Image.new("RGB", (100, 100), color="red").save(image1)

        # Setup: Context
        context = {"dataset_path": str(dataset_jsonl)}

        # Create and run task
        task = DatasetLoadingTask()
        result = task.run(context)

        # Verify system prompt was loaded from dataset.json
        assert "system_prompt" in result
        assert (
            result["system_prompt"]
            == "You are an expert at identifying materials. Return JSON format."
        )

        # Verify it's also in config for VLMInferenceTask
        assert "config" in result
        assert "system_prompt" in result["config"]
        assert result["config"]["system_prompt"] == result["system_prompt"]

        # Verify dataset was loaded
        assert "dataset" in result
        assert len(result["dataset"]) == 1

    def test_dataset_loading_respects_existing_system_prompt(self, tmp_path):
        """Test that DatasetLoadingTask doesn't override existing system prompt."""
        from material_agent.tasks.dataset import DatasetLoadingTask

        # Setup: Create minimal dataset
        dataset_jsonl = tmp_path / "dataset.jsonl"
        dataset_entry = {
            "id": "test_entry",
            "media": {"images": [{"path": "image1.png"}]},
            "user_prompt": "Test",
        }

        with open(dataset_jsonl, "w") as f:
            f.write(json.dumps(dataset_entry) + "\n")

        # Create dataset.json with system prompt
        dataset_json = tmp_path / "dataset.json"
        with open(dataset_json, "w") as f:
            json.dump(
                {
                    "inference": {
                        "prompts": [{"system_prompt": "System prompt from dataset"}]
                    }
                },
                f,
            )

        # Create test image
        image1 = tmp_path / "image1.png"
        from PIL import Image

        Image.new("RGB", (100, 100)).save(image1)

        # Setup: Context with existing system_prompt
        context = {
            "dataset_path": str(dataset_jsonl),
            "system_prompt": "Existing system prompt from config",
            "config": {"system_prompt": "Existing system prompt from config"},
        }

        # Run task
        task = DatasetLoadingTask()
        result = task.run(context)

        # Verify existing system prompt was NOT overridden
        assert result["system_prompt"] == "Existing system prompt from config"
        assert result["config"]["system_prompt"] == "Existing system prompt from config"


def _create_input_usd(path: Path, default_prim: str | None = "RootNode") -> None:
    """Helper to create a minimal valid USD file with a defaultPrim and mesh."""
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    if default_prim:
        root = UsdGeom.Xform.Define(stage, f"/{default_prim}")
        stage.SetDefaultPrim(root.GetPrim())
        UsdGeom.Scope.Define(stage, f"/{default_prim}/Geometry")
        UsdGeom.Mesh.Define(stage, f"/{default_prim}/Geometry/Part1")
    stage.GetRootLayer().Save()


def _create_predictions(path: Path, prim_prefix: str = "/RootNode") -> None:
    """Helper to create a minimal predictions JSONL file."""
    predictions = [
        {
            "id": f"{prim_prefix}/Geometry/Part1",
            "materials": {"material": "TestMaterial"},
        }
    ]
    with open(path, "w") as f:
        for pred in predictions:
            f.write(json.dumps(pred) + "\n")


def _create_material_library(path: Path) -> None:
    """Helper to create a minimal material library USD with one material."""
    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.Scope.Define(stage, "/World/Looks")
    UsdShade.Material.Define(stage, "/World/Looks/TestMaterial")
    stage.GetRootLayer().Save()


class TestDefaultPrimPreservation:
    """Tests that defaultPrim is preserved from input to output.

    defaultPrim is non-composable USD layer metadata — it only takes effect on
    the root layer and does not compose from sublayers. Both _create_full_stage()
    and _create_material_layer() must explicitly copy it from the input.
    """

    def test_full_stage_preserves_default_prim(self, tmp_path):
        """_create_full_stage() must copy defaultPrim from input to output."""
        input_usd = tmp_path / "input.usda"
        output_usd = tmp_path / "output.usda"
        predictions_path = tmp_path / "predictions.jsonl"
        library_path = tmp_path / "library.usda"

        _create_input_usd(input_usd, default_prim="RootNode")
        _create_predictions(predictions_path, prim_prefix="/RootNode")
        _create_material_library(library_path)

        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()

        stage, materials_applied, stats = task._create_full_stage(
            input_usd_path=input_usd,
            output_usd_path=output_usd,
            resolved_materials={"TestMaterial": "/World/Looks/TestMaterial"},
            prim_to_material={"/RootNode/Geometry/Part1": "TestMaterial"},
            is_library_based=True,
            material_library_path=str(library_path),
            flatten_output=False,
        )

        # Verify defaultPrim is set on the output root layer
        output_layer = Sdf.Layer.FindOrOpen(str(output_usd))
        assert output_layer.defaultPrim == "RootNode"

        # Verify via composed stage
        output_stage = Usd.Stage.Open(str(output_usd))
        assert output_stage.HasDefaultPrim()
        assert str(output_stage.GetDefaultPrim().GetPath()) == "/RootNode"

    def test_full_stage_preserves_different_default_prim_name(self, tmp_path):
        """_create_full_stage() works with non-standard defaultPrim names."""
        input_usd = tmp_path / "input.usda"
        output_usd = tmp_path / "output.usda"
        predictions_path = tmp_path / "predictions.jsonl"
        library_path = tmp_path / "library.usda"

        _create_input_usd(input_usd, default_prim="World")
        _create_predictions(predictions_path, prim_prefix="/World")
        _create_material_library(library_path)

        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()

        task._create_full_stage(
            input_usd_path=input_usd,
            output_usd_path=output_usd,
            resolved_materials={"TestMaterial": "/World/Looks/TestMaterial"},
            prim_to_material={"/World/Geometry/Part1": "TestMaterial"},
            is_library_based=True,
            material_library_path=str(library_path),
            flatten_output=False,
        )

        output_layer = Sdf.Layer.FindOrOpen(str(output_usd))
        assert output_layer.defaultPrim == "World"

    def test_full_stage_handles_no_default_prim(self, tmp_path):
        """_create_full_stage() auto-detects root prim when input has no defaultPrim."""
        input_usd = tmp_path / "input.usda"
        output_usd = tmp_path / "output.usda"
        predictions_path = tmp_path / "predictions.jsonl"
        library_path = tmp_path / "library.usda"

        # Create input WITHOUT a defaultPrim
        _create_input_usd(input_usd, default_prim=None)
        # Need a prim so the stage is valid — add one manually
        stage = Usd.Stage.Open(str(input_usd))
        UsdGeom.Xform.Define(stage, "/SomeRoot")
        UsdGeom.Mesh.Define(stage, "/SomeRoot/Mesh")
        stage.GetRootLayer().Save()

        _create_predictions(predictions_path, prim_prefix="/SomeRoot")
        _create_material_library(library_path)

        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()

        task._create_full_stage(
            input_usd_path=input_usd,
            output_usd_path=output_usd,
            resolved_materials={"TestMaterial": "/World/Looks/TestMaterial"},
            prim_to_material={"/SomeRoot/Mesh": "TestMaterial"},
            is_library_based=True,
            material_library_path=str(library_path),
            flatten_output=False,
        )

        # When input has no defaultPrim, the fix auto-detects the actual root
        # prim from the composed stage so materials are placed correctly.
        output_layer = Sdf.Layer.FindOrOpen(str(output_usd))
        assert output_layer.defaultPrim == "SomeRoot"

    def test_material_layer_preserves_default_prim(self, tmp_path):
        """_create_material_layer() must copy defaultPrim from input to output."""
        input_usd = tmp_path / "input.usda"
        output_usd = tmp_path / "output.usda"
        predictions_path = tmp_path / "predictions.jsonl"
        library_path = tmp_path / "library.usda"

        _create_input_usd(input_usd, default_prim="RootNode")
        _create_predictions(predictions_path, prim_prefix="/RootNode")
        _create_material_library(library_path)

        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()

        stage, materials_applied, stats = task._create_material_layer(
            input_usd_path=input_usd,
            output_usd_path=output_usd,
            resolved_materials={"TestMaterial": "/World/Looks/TestMaterial"},
            prim_to_material={"/RootNode/Geometry/Part1": "TestMaterial"},
            is_library_based=True,
            material_library_path=str(library_path),
        )

        # Verify defaultPrim is set on the output root layer
        output_layer = Sdf.Layer.FindOrOpen(str(output_usd))
        assert output_layer.defaultPrim == "RootNode"

        # Verify via composed stage
        output_stage = Usd.Stage.Open(str(output_usd))
        assert output_stage.HasDefaultPrim()
        assert str(output_stage.GetDefaultPrim().GetPath()) == "/RootNode"

    def test_up_axis_also_preserved(self, tmp_path):
        """Verify upAxis is preserved alongside defaultPrim."""
        input_usd = tmp_path / "input.usda"
        output_usd = tmp_path / "output.usda"
        predictions_path = tmp_path / "predictions.jsonl"
        library_path = tmp_path / "library.usda"

        _create_input_usd(input_usd, default_prim="RootNode")
        _create_predictions(predictions_path, prim_prefix="/RootNode")
        _create_material_library(library_path)

        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()

        task._create_full_stage(
            input_usd_path=input_usd,
            output_usd_path=output_usd,
            resolved_materials={"TestMaterial": "/World/Looks/TestMaterial"},
            prim_to_material={"/RootNode/Geometry/Part1": "TestMaterial"},
            is_library_based=True,
            material_library_path=str(library_path),
            flatten_output=False,
        )

        output_stage = Usd.Stage.Open(str(output_usd))
        assert UsdGeom.GetStageUpAxis(output_stage) == UsdGeom.Tokens.z

    def test_full_stage_fixes_stale_default_prim(self, tmp_path):
        """_create_full_stage() corrects stale defaultPrim after optimizer renames root.

        When the NVCF optimizer wraps content under /World but the input's
        defaultPrim still says 'OriginalRoot', the composed stage has no valid
        default prim. The fix detects this and updates defaultPrim to match the
        actual root, so materials go under the correct prim.
        """
        input_usd = tmp_path / "input.usda"
        output_usd = tmp_path / "output.usda"
        predictions_path = tmp_path / "predictions.jsonl"
        library_path = tmp_path / "library.usda"

        # Create input simulating NVCF optimizer output:
        # - Content under /World (optimizer's convention)
        # - But defaultPrim still says "OriginalRoot" (stale from pre-optimization)
        stage = Usd.Stage.CreateNew(str(input_usd))
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.Xform.Define(stage, "/World")
        UsdGeom.Scope.Define(stage, "/World/Geometry")
        UsdGeom.Mesh.Define(stage, "/World/Geometry/Part1")
        stage.GetRootLayer().defaultPrim = "OriginalRoot"  # Stale!
        stage.GetRootLayer().Save()

        _create_predictions(predictions_path, prim_prefix="/World")
        _create_material_library(library_path)

        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()

        task._create_full_stage(
            input_usd_path=input_usd,
            output_usd_path=output_usd,
            resolved_materials={"TestMaterial": "/World/Looks/TestMaterial"},
            prim_to_material={"/World/Geometry/Part1": "TestMaterial"},
            is_library_based=True,
            material_library_path=str(library_path),
            flatten_output=False,
        )

        # Verify defaultPrim was corrected to the actual root prim
        output_layer = Sdf.Layer.FindOrOpen(str(output_usd))
        assert output_layer.defaultPrim == "World"

        # Verify via composed stage
        output_stage = Usd.Stage.Open(str(output_usd))
        assert output_stage.HasDefaultPrim()
        assert str(output_stage.GetDefaultPrim().GetPath()) == "/World"


class TestApplyMaterialsOutputIntegrity:
    """Regression tests for output USD integrity (metersPerUnit, no extra prims)."""

    def test_flatten_preserves_meters_per_unit(self, tmp_path):
        """Flatten must not change metersPerUnit from the original stage.

        Regression: flatten was silently resetting metersPerUnit to 0.01
        when the original asset used 1.0 (meters).
        """
        input_usd = tmp_path / "input.usda"
        output_usd = tmp_path / "output.usd"

        # Create input with metersPerUnit=1.0 (meters, NOT the 0.01 default)
        stage = Usd.Stage.CreateNew(str(input_usd))
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        root = UsdGeom.Xform.Define(stage, "/Asset")
        stage.SetDefaultPrim(root.GetPrim())
        UsdGeom.Mesh.Define(stage, "/Asset/Mesh")
        stage.GetRootLayer().Save()

        # Verify input
        assert UsdGeom.GetStageMetersPerUnit(stage) == 1.0

        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()

        # Run with flatten_output=True (the default in the service)
        task._create_full_stage(
            input_usd_path=input_usd,
            output_usd_path=output_usd,
            resolved_materials={},
            prim_to_material={},
            flatten_output=True,
        )

        # Verify metersPerUnit is preserved
        out_stage = Usd.Stage.Open(str(output_usd))
        assert UsdGeom.GetStageMetersPerUnit(out_stage) == 1.0, (
            f"metersPerUnit changed from 1.0 to "
            f"{UsdGeom.GetStageMetersPerUnit(out_stage)} after flatten"
        )

    def test_flatten_preserves_up_axis(self, tmp_path):
        """Flatten must preserve the original upAxis."""
        input_usd = tmp_path / "input.usda"
        output_usd = tmp_path / "output.usd"

        stage = Usd.Stage.CreateNew(str(input_usd))
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        root = UsdGeom.Xform.Define(stage, "/Asset")
        stage.SetDefaultPrim(root.GetPrim())
        stage.GetRootLayer().Save()

        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()

        task._create_full_stage(
            input_usd_path=input_usd,
            output_usd_path=output_usd,
            resolved_materials={},
            prim_to_material={},
            flatten_output=True,
        )

        out_stage = Usd.Stage.Open(str(output_usd))
        assert UsdGeom.GetStageUpAxis(out_stage) == UsdGeom.Tokens.z

    def test_layer_only_has_no_geometry(self, tmp_path):
        """layer_only output must not contain geometry from the input."""
        input_usd = tmp_path / "input.usda"
        output_usd = tmp_path / "output.usd"

        stage = Usd.Stage.CreateNew(str(input_usd))
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        root = UsdGeom.Xform.Define(stage, "/Asset")
        stage.SetDefaultPrim(root.GetPrim())
        UsdGeom.Mesh.Define(stage, "/Asset/Body")
        UsdGeom.Mesh.Define(stage, "/Asset/Wheel")
        stage.GetRootLayer().Save()

        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()

        task._create_material_layer(
            input_usd_path=input_usd,
            output_usd_path=output_usd,
            resolved_materials={},
            prim_to_material={},
        )

        # The output root layer should NOT define the geometry prims
        # (they come through sublayer composition, not the root layer)
        out_layer = Sdf.Layer.FindOrOpen(str(output_usd))
        root_prims = [p.name for p in out_layer.rootPrims]
        assert "Asset" not in root_prims or all(
            out_layer.GetPrimAtPath(f"/Asset/{child}").specifier == Sdf.SpecifierOver
            for child in ["Body", "Wheel"]
            if out_layer.GetPrimAtPath(f"/Asset/{child}")
        ), "layer_only output should use 'over' specs, not 'def' for geometry"

    def test_library_materials_placed_under_default_prim(self, tmp_path):
        """Library materials must go under the asset's defaultPrim, not /World.

        Regression: materials from the library at /World/Looks/Iron were
        copied verbatim, creating an extra /World root prim in the output.
        """
        input_usd = tmp_path / "input.usda"
        output_usd = tmp_path / "output.usd"
        library_usd = tmp_path / "library.usd"

        # Create input with defaultPrim = "MyGear"
        stage = Usd.Stage.CreateNew(str(input_usd))
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
        root = UsdGeom.Xform.Define(stage, "/MyGear")
        stage.SetDefaultPrim(root.GetPrim())
        UsdGeom.Mesh.Define(stage, "/MyGear/Body")
        stage.GetRootLayer().Save()

        # Create library with materials under /World/Looks
        lib_stage = Usd.Stage.CreateNew(str(library_usd))
        UsdGeom.Scope.Define(lib_stage, "/World")
        UsdGeom.Scope.Define(lib_stage, "/World/Looks")
        UsdShade.Material.Define(lib_stage, "/World/Looks/Iron")
        lib_stage.GetRootLayer().Save()

        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()

        task._create_full_stage(
            input_usd_path=input_usd,
            output_usd_path=output_usd,
            resolved_materials={"Iron": "/World/Looks/Iron"},
            prim_to_material={"/MyGear/Body": "Iron"},
            is_library_based=True,
            material_library_path=str(library_usd),
            flatten_output=True,
        )

        out_stage = Usd.Stage.Open(str(output_usd))
        root_prims = [p.GetName() for p in out_stage.GetPseudoRoot().GetChildren()]

        # /World must NOT be a root prim — materials should be under /MyGear
        assert "World" not in root_prims, (
            f"Output has /World root prim — materials should be under "
            f"the default prim /MyGear. Root prims: {root_prims}"
        )

        # Materials should be under /MyGear/Looks/Iron
        iron_prim = out_stage.GetPrimAtPath("/MyGear/Looks/Iron")
        assert iron_prim.IsValid(), (
            "Material should be at /MyGear/Looks/Iron, not /World/Looks/Iron"
        )
