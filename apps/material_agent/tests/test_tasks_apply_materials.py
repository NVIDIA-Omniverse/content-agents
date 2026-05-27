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

    def test_fails_clearly_when_all_predictions_are_unknown(self, tmp_path: Path):
        predictions_path = tmp_path / "predictions.jsonl"
        predictions_path.write_text(
            json.dumps(
                {
                    "id": "/RootNode/Geometry/Part1",
                    "materials": {
                        "material": "__UNKNOWN__",
                        "reason": "no visible geometry",
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        input_usd = tmp_path / "input.usd"
        input_usd.write_text("#usda 1.0\n")

        context = {
            "input_usd_path": str(input_usd),
            "output_usd_path": str(tmp_path / "output.usd"),
            "predictions_path": str(predictions_path),
            "resolved_materials": {},
            "is_library_based_mapping": True,
        }

        with pytest.raises(ValueError, match="classified as '__UNKNOWN__'"):
            ApplyMaterialsToUSDTask().run(context)

    def test_allows_all_unknown_predictions_when_empty_apply_is_allowed(
        self, tmp_path: Path
    ) -> None:
        predictions_path = tmp_path / "predictions.jsonl"
        predictions_path.write_text(
            json.dumps({"id": "/Hidden", "materials": {"material": "__UNKNOWN__"}})
            + "\n",
            encoding="utf-8",
        )
        input_usd = tmp_path / "input.usd"
        input_usd.write_text("#usda 1.0\n")

        context = {
            "input_usd_path": str(input_usd),
            "output_usd_path": str(tmp_path / "output.usd"),
            "predictions_path": str(predictions_path),
            "resolved_materials": {},
            "is_library_based_mapping": True,
            "allow_empty_predictions": True,
        }

        result = ApplyMaterialsToUSDTask().run(context)

        assert result["materials_applied"] == {}
        assert result["assignment_stats"]["unknown"] == 1
        assert result["assignment_stats"]["materials_applied"] == 0

    def test_load_mapping_skips_unknown_predictions(self, tmp_path: Path) -> None:
        predictions_path = tmp_path / "predictions.jsonl"
        predictions_path.write_text(
            json.dumps(
                {
                    "id": "/RootNode/Geometry/Hidden",
                    "materials": {"material": "__UNKNOWN__"},
                }
            )
            + "\n"
            + json.dumps(
                {
                    "id": "/RootNode/Geometry/Visible",
                    "materials": {"material": "Steel"},
                }
            )
            + "\n",
            encoding="utf-8",
        )

        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()

        mapping = task._load_prim_material_mapping(predictions_path)

        assert mapping == {"/RootNode/Geometry/Visible": "Steel"}

    def test_load_mapping_normalizes_material_names(self, tmp_path: Path) -> None:
        predictions_path = tmp_path / "predictions.jsonl"
        predictions_path.write_text(
            json.dumps(
                {
                    "id": "/RootNode/Geometry/Visible",
                    "materials": {"material": " Steel "},
                }
            )
            + "\n",
            encoding="utf-8",
        )

        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()

        mapping = task._load_prim_material_mapping(predictions_path)

        assert mapping == {"/RootNode/Geometry/Visible": "Steel"}

    def test_load_mapping_handles_predicted_material_field(
        self, tmp_path: Path
    ) -> None:
        predictions_path = tmp_path / "predictions.jsonl"
        predictions_path.write_text(
            json.dumps(
                {
                    "id": "/RootNode/Geometry/Visible",
                    "predicted_material": " Steel ",
                }
            )
            + "\n",
            encoding="utf-8",
        )

        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()

        mapping = task._load_prim_material_mapping(predictions_path)
        counts = task._count_prediction_materials(predictions_path)

        assert mapping == {"/RootNode/Geometry/Visible": "Steel"}
        assert counts["total"] == 1
        assert counts["actionable"] == 1
        assert counts["missing"] == 0

    def test_load_mapping_handles_nested_payloads(self, tmp_path: Path) -> None:
        predictions_path = tmp_path / "predictions.jsonl"
        predictions_path.write_text(
            json.dumps(
                {
                    "predictions": [
                        {
                            "id": "/RootNode/Geometry/Hidden",
                            "materials": {"material": "__UNKNOWN__"},
                        },
                        {
                            "id": "/RootNode/Geometry/Visible",
                            "materials": {"material": " Steel "},
                        },
                    ]
                }
            )
            + "\n"
            + json.dumps(
                {
                    "/MappedHidden": {
                        "materials": {"material": "__UNKNOWN__"},
                    },
                    "/MappedVisible": "Plastic",
                }
            )
            + "\n",
            encoding="utf-8",
        )

        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()

        mapping = task._load_prim_material_mapping(predictions_path)

        assert mapping == {
            "/RootNode/Geometry/Visible": "Steel",
            "/MappedVisible": "Plastic",
        }

    def test_load_mapping_propagates_parent_id_into_container_children(
        self, tmp_path: Path
    ) -> None:
        predictions_path = tmp_path / "predictions.jsonl"
        predictions_path.write_text(
            json.dumps(
                {
                    "id": "/RootNode/Geometry/Parent",
                    "predictions": [
                        {
                            "materials": {"material": " Steel "},
                        }
                    ],
                }
            )
            + "\n",
            encoding="utf-8",
        )

        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()

        mapping = task._load_prim_material_mapping(predictions_path)
        counts = task._count_prediction_materials(predictions_path)

        assert mapping == {"/RootNode/Geometry/Parent": "Steel"}
        assert counts["total"] == 1
        assert counts["actionable"] == 1

    def test_count_prediction_materials_classifies_jsonl_rows(
        self, tmp_path: Path
    ) -> None:
        predictions_path = tmp_path / "predictions.jsonl"
        predictions_path.write_text(
            json.dumps({"id": "/Hidden", "materials": {"material": "__UNKNOWN__"}})
            + "\n"
            + json.dumps({"id": "/Visible", "materials": {"material": "Steel"}})
            + "\n"
            + json.dumps({"id": "/NoMaterial", "materials": {}})
            + "\n"
            + "{invalid json}\n",
            encoding="utf-8",
        )

        counts = ApplyMaterialsToUSDTask()._count_prediction_materials(predictions_path)

        assert counts == {"total": 3, "actionable": 1, "unknown": 1, "missing": 1}

    def test_count_prediction_materials_classifies_nested_payloads(
        self, tmp_path: Path
    ) -> None:
        predictions_path = tmp_path / "predictions.jsonl"
        predictions_path.write_text(
            json.dumps(
                {
                    "predictions": [
                        {
                            "id": "/Hidden",
                            "materials": {"material": "__UNKNOWN__"},
                        },
                        {"id": "/Visible", "material": "Steel"},
                        {"id": "/NoMaterial", "materials": {}},
                    ]
                }
            )
            + "\n"
            + json.dumps(
                {
                    "/MappedHidden": {
                        "materials": {"material": "__UNKNOWN__"},
                    },
                    "/MappedVisible": "Plastic",
                }
            )
            + "\n",
            encoding="utf-8",
        )

        counts = ApplyMaterialsToUSDTask()._count_prediction_materials(predictions_path)

        assert counts == {"total": 5, "actionable": 2, "unknown": 2, "missing": 1}

    def test_path_keyed_peers_are_counted_when_top_level_material_exists(
        self, tmp_path: Path
    ) -> None:
        predictions_path = tmp_path / "predictions.jsonl"
        predictions_path.write_text(
            json.dumps(
                {
                    "id": "/Batch",
                    "material": "Batch Material",
                    "/MappedHidden": {
                        "material": "",
                        "validation_status": "disallowed_unknown",
                    },
                    "/MappedVisible": "Steel",
                }
            )
            + "\n",
            encoding="utf-8",
        )

        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()

        mapping = task._load_prim_material_mapping(predictions_path)
        counts = task._count_prediction_materials(predictions_path)

        assert mapping == {
            "/Batch": "Batch Material",
            "/MappedVisible": "Steel",
        }
        assert counts == {"total": 3, "actionable": 2, "unknown": 1, "missing": 0}

    def test_count_prediction_materials_ignores_metadata_only_path_peers(
        self, tmp_path: Path
    ) -> None:
        predictions_path = tmp_path / "predictions.jsonl"
        predictions_path.write_text(
            json.dumps(
                {
                    "id": "/Batch",
                    "material": "Batch Material",
                    "/MetadataOnly": {"validation_status": "valid"},
                }
            )
            + "\n",
            encoding="utf-8",
        )

        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()

        mapping = task._load_prim_material_mapping(predictions_path)
        counts = task._count_prediction_materials(predictions_path)

        assert mapping == {"/Batch": "Batch Material"}
        assert counts == {"total": 1, "actionable": 1, "unknown": 0, "missing": 0}

    def test_count_prediction_materials_ignores_bare_empty_dicts(
        self, tmp_path: Path
    ) -> None:
        predictions_path = tmp_path / "predictions.jsonl"
        predictions_path.write_text(
            json.dumps(
                [
                    {},
                    {"id": "/NoMaterial"},
                    {"materials": {}},
                    "Steel",
                    "__UNKNOWN__",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        counts = ApplyMaterialsToUSDTask()._count_prediction_materials(predictions_path)

        assert counts == {"total": 4, "actionable": 1, "unknown": 1, "missing": 2}

    def test_load_mapping_handles_string_items_in_parent_container(
        self, tmp_path: Path
    ) -> None:
        predictions_path = tmp_path / "predictions.jsonl"
        predictions_path.write_text(
            json.dumps({"id": "/Parent", "predictions": ["Steel", "__UNKNOWN__"]})
            + "\n",
            encoding="utf-8",
        )

        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()

        mapping = task._load_prim_material_mapping(predictions_path)
        counts = task._count_prediction_materials(predictions_path)

        assert mapping == {"/Parent": "Steel"}
        assert counts == {"total": 2, "actionable": 1, "unknown": 1, "missing": 0}

    def test_count_prediction_materials_warns_on_invalid_json(
        self, tmp_path: Path
    ) -> None:
        predictions_path = tmp_path / "predictions.jsonl"
        predictions_path.write_text("{invalid json}\n", encoding="utf-8")
        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()

        counts = task._count_prediction_materials(predictions_path)

        assert counts == {"total": 0, "actionable": 0, "unknown": 0, "missing": 0}
        task.listener.warning.assert_called_once()
        assert (
            task.listener.warning.call_args[0][0]
            == "Failed to parse prediction line while counting materials: "
            "Expecting property name enclosed in double quotes: line 1 column 2 "
            "(char 1)"
        )

    def test_strict_unknown_mode_fails_on_partial_unknown_predictions(
        self, tmp_path: Path
    ) -> None:
        predictions_path = tmp_path / "predictions.jsonl"
        predictions_path.write_text(
            json.dumps({"id": "/Hidden", "materials": {"material": "__UNKNOWN__"}})
            + "\n"
            + json.dumps({"id": "/Visible", "materials": {"material": "Steel"}})
            + "\n",
            encoding="utf-8",
        )
        input_usd = tmp_path / "input.usd"
        input_usd.write_text("#usda 1.0\n")

        context = {
            "input_usd_path": str(input_usd),
            "output_usd_path": str(tmp_path / "output.usd"),
            "predictions_path": str(predictions_path),
            "resolved_materials": {"Steel": "/path/to/steel.usd"},
            "is_library_based_mapping": True,
            "fail_on_unknown_material": True,
        }

        with pytest.raises(ValueError, match="fail_on_unknown_material=true"):
            ApplyMaterialsToUSDTask().run(context)

    def test_strict_unknown_mode_fails_on_nested_unknown_predictions(
        self, tmp_path: Path
    ) -> None:
        predictions_path = tmp_path / "predictions.jsonl"
        predictions_path.write_text(
            json.dumps(
                {
                    "predictions": [
                        {
                            "id": "/Hidden",
                            "materials": {"material": "__UNKNOWN__"},
                        },
                        {"id": "/Visible", "materials": {"material": "Steel"}},
                    ]
                }
            )
            + "\n",
            encoding="utf-8",
        )
        input_usd = tmp_path / "input.usd"
        input_usd.write_text("#usda 1.0\n")

        context = {
            "input_usd_path": str(input_usd),
            "output_usd_path": str(tmp_path / "output.usd"),
            "predictions_path": str(predictions_path),
            "resolved_materials": {"Steel": "/path/to/steel.usd"},
            "is_library_based_mapping": True,
            "fail_on_unknown_material": True,
        }

        with pytest.raises(ValueError, match="fail_on_unknown_material=true"):
            ApplyMaterialsToUSDTask().run(context)

    def test_strict_unknown_mode_fails_when_validation_cleared_sentinel(
        self, tmp_path: Path
    ) -> None:
        predictions_path = tmp_path / "predictions.jsonl"
        predictions_path.write_text(
            json.dumps({"id": "/Hidden", "materials": {"material": ""}}) + "\n",
            encoding="utf-8",
        )
        input_usd = tmp_path / "input.usd"
        input_usd.write_text("#usda 1.0\n")

        context = {
            "input_usd_path": str(input_usd),
            "output_usd_path": str(tmp_path / "output.usd"),
            "predictions_path": str(predictions_path),
            "resolved_materials": {"Steel": "/path/to/steel.usd"},
            "is_library_based_mapping": True,
            "fail_on_unknown_material": True,
            "unknown_material_predictions": 1,
        }

        with pytest.raises(ValueError) as exc_info:
            ApplyMaterialsToUSDTask().run(context)

        error_msg = str(exc_info.value)
        assert "fail_on_unknown_material=true" in error_msg
        assert "earlier validation steps" in error_msg

    def test_strict_unknown_mode_fails_on_durable_disallowed_unknown_marker(
        self, tmp_path: Path
    ) -> None:
        predictions_path = tmp_path / "predictions.jsonl"
        predictions_path.write_text(
            json.dumps(
                {
                    "id": "/Hidden",
                    "materials": {
                        "material": "",
                        "validation_status": "disallowed_unknown",
                    },
                }
            )
            + "\n"
            + json.dumps({"id": "/Visible", "materials": {"material": "Steel"}})
            + "\n",
            encoding="utf-8",
        )
        input_usd = tmp_path / "input.usd"
        input_usd.write_text("#usda 1.0\n")

        context = {
            "input_usd_path": str(input_usd),
            "output_usd_path": str(tmp_path / "output.usd"),
            "predictions_path": str(predictions_path),
            "resolved_materials": {"Steel": "/path/to/steel.usd"},
            "is_library_based_mapping": True,
            "fail_on_unknown_material": True,
        }

        task = ApplyMaterialsToUSDTask()
        counts = task._count_prediction_materials(predictions_path)

        assert counts == {"total": 2, "actionable": 1, "unknown": 1, "missing": 0}
        with pytest.raises(ValueError, match="fail_on_unknown_material=true"):
            task.run(context)

    def test_fails_clearly_when_predictions_have_only_missing_materials(
        self, tmp_path: Path
    ) -> None:
        predictions_path = tmp_path / "predictions.jsonl"
        predictions_path.write_text(
            json.dumps({"id": "/Hidden", "materials": {"material": ""}}) + "\n",
            encoding="utf-8",
        )
        input_usd = tmp_path / "input.usd"
        input_usd.write_text("#usda 1.0\n")

        context = {
            "input_usd_path": str(input_usd),
            "output_usd_path": str(tmp_path / "output.usd"),
            "predictions_path": str(predictions_path),
            "resolved_materials": {},
            "is_library_based_mapping": True,
        }

        with pytest.raises(
            ValueError,
            match="did not contain actionable material values",
        ):
            ApplyMaterialsToUSDTask().run(context)

    def test_fails_clearly_when_resolved_materials_but_predictions_are_unknown(
        self, tmp_path: Path
    ) -> None:
        predictions_path = tmp_path / "predictions.jsonl"
        predictions_path.write_text(
            json.dumps({"id": "/Hidden", "materials": {"material": "__UNKNOWN__"}})
            + "\n",
            encoding="utf-8",
        )
        input_usd = tmp_path / "input.usd"
        input_usd.write_text("#usda 1.0\n")

        context = {
            "input_usd_path": str(input_usd),
            "output_usd_path": str(tmp_path / "output.usd"),
            "predictions_path": str(predictions_path),
            "resolved_materials": {"Steel": "/path/to/steel.usd"},
            "is_library_based_mapping": True,
        }

        with pytest.raises(ValueError, match="classified as '__UNKNOWN__'"):
            ApplyMaterialsToUSDTask().run(context)

    def test_fails_when_no_predictions_and_no_materials_by_default(
        self, tmp_path: Path
    ) -> None:
        """No predictions should fail closed unless explicitly opted in."""
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

        with pytest.raises(ValueError, match="No material predictions"):
            task.run(context)

    def test_allows_no_predictions_and_no_materials_when_opted_in(
        self, tmp_path: Path
    ) -> None:
        """Intentional empty material application remains an explicit opt-in."""
        predictions_path = tmp_path / "predictions.jsonl"
        input_usd = tmp_path / "input.usd"
        input_usd.write_text("# Mock USD")
        output_usd = tmp_path / "output.usd"
        context = {
            "input_usd_path": str(input_usd),
            "output_usd_path": str(output_usd),
            "predictions_path": str(predictions_path),
            "resolved_materials": {},
            "is_library_based_mapping": True,
            "allow_empty_predictions": True,
        }

        result = ApplyMaterialsToUSDTask().run(context)

        assert result is not None
        assert result["materials_applied"] == {}
        assert result["assignment_stats"]["total_prims"] == 0
        assert result["assignment_stats"]["materials_applied"] == 0
        assert result["assignment_stats"]["failed"] == 0

    def test_fails_when_resolved_materials_have_empty_predictions(
        self, tmp_path: Path
    ) -> None:
        """Resolved materials without prediction bindings should fail closed."""
        predictions_path = tmp_path / "predictions.jsonl"
        predictions_path.write_text("", encoding="utf-8")
        input_usd = tmp_path / "input.usd"
        input_usd.write_text("#usda 1.0\n")

        context = {
            "input_usd_path": str(input_usd),
            "output_usd_path": str(tmp_path / "output.usd"),
            "predictions_path": str(predictions_path),
            "resolved_materials": {"Steel": "/path/to/steel.usd"},
            "is_library_based_mapping": True,
        }

        with pytest.raises(ValueError, match="No material predictions"):
            ApplyMaterialsToUSDTask().run(context)

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

    def test_raises_when_usd_export_fails(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Authoring failures should abort instead of reporting success."""
        predictions_path = tmp_path / "predictions.jsonl"
        predictions_path.write_text(
            json.dumps(
                {
                    "id": "/RootNode/Geometry/Part1",
                    "materials": {"material": "Steel"},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        input_usd = tmp_path / "input.usd"
        output_usd = tmp_path / "output.usd"
        input_usd.write_text("#usda 1.0\n")

        def fail_export(path: str) -> None:
            Path(path).write_text("#usda 1.0\n# partial output\n", encoding="utf-8")
            raise RuntimeError("export failed")

        root_layer = MagicMock()
        root_layer.Export.side_effect = fail_export
        stage = MagicMock()
        stage.GetRootLayer.return_value = root_layer

        def fake_create_full_stage(*args, **kwargs):
            return (
                stage,
                {"/RootNode/Geometry/Part1": "Steel"},
                {"materials_created": 1, "prims_with_materials": 1, "failed": 0},
            )

        monkeypatch.setattr(
            ApplyMaterialsToUSDTask,
            "_create_full_stage",
            fake_create_full_stage,
        )

        with pytest.raises(RuntimeError, match="export failed"):
            ApplyMaterialsToUSDTask().run(
                {
                    "input_usd_path": str(input_usd),
                    "output_usd_path": str(output_usd),
                    "predictions_path": str(predictions_path),
                    "resolved_materials": {"Steel": "/path/to/steel.usd"},
                    "is_library_based_mapping": True,
                    "layer_only": False,
                    "flatten_output": False,
                }
            )
        assert not output_usd.exists()


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

    def test_flatten_continues_when_stale_shader_cleanup_fails(
        self, monkeypatch, tmp_path
    ):
        """Flatten should still export if best-effort stale shader cleanup fails."""
        input_usd = tmp_path / "input.usda"
        output_usd = tmp_path / "output.usd"

        stage = Usd.Stage.CreateNew(str(input_usd))
        root = UsdGeom.Xform.Define(stage, "/Asset")
        stage.SetDefaultPrim(root.GetPrim())
        UsdGeom.Mesh.Define(stage, "/Asset/Mesh")
        stage.GetRootLayer().Save()

        def fail_cleanup(*args, **kwargs):
            raise RuntimeError("cleanup failed")

        monkeypatch.setattr(
            ApplyMaterialsToUSDTask,
            "_deactivate_unbound_unresolved_mdl_shaders",
            fail_cleanup,
        )

        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()

        task._create_full_stage(
            input_usd_path=input_usd,
            output_usd_path=output_usd,
            resolved_materials={},
            prim_to_material={},
            flatten_output=True,
        )

        assert output_usd.exists()
        warnings = [call.args[0] for call in task.listener.warning.call_args_list]
        assert any(
            "Failed to deactivate stale unresolved MDL shaders" in warning
            for warning in warnings
        )

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
        looks_prim = out_stage.GetPrimAtPath("/MyGear/Looks")
        assert looks_prim.IsA(UsdGeom.Scope)

    def test_library_copy_clears_color_space_on_empty_asset_inputs(self, tmp_path):
        """Empty texture slots copied from a library must not keep colorSpace."""
        input_usd = tmp_path / "input.usda"
        output_usd = tmp_path / "output.usd"
        library_usd = tmp_path / "library.usd"

        stage = Usd.Stage.CreateNew(str(input_usd))
        root = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(root.GetPrim())
        UsdGeom.Mesh.Define(stage, "/World/Body")
        stage.GetRootLayer().Save()

        lib_stage = Usd.Stage.CreateNew(str(library_usd))
        UsdGeom.Scope.Define(lib_stage, "/World")
        UsdGeom.Scope.Define(lib_stage, "/World/Looks")
        material = UsdShade.Material.Define(lib_stage, "/World/Looks/Steel")
        material_prim = material.GetPrim()
        empty_texture = material_prim.CreateAttribute(
            "inputs:base_color_texture_file",
            Sdf.ValueTypeNames.Asset,
        )
        empty_texture.Set(Sdf.AssetPath(""))
        empty_texture.SetColorSpace("sRGB")
        real_texture = material_prim.CreateAttribute(
            "inputs:geometry_normal_texture_file",
            Sdf.ValueTypeNames.Asset,
        )
        real_texture.Set(Sdf.AssetPath("textures/normal.png"))
        real_texture.SetColorSpace("raw")
        shader = UsdShade.Shader.Define(lib_stage, "/World/Looks/Steel/Shader")
        nested_empty_texture = shader.CreateInput(
            "emissive_texture_file",
            Sdf.ValueTypeNames.Asset,
        )
        nested_empty_texture.Set(Sdf.AssetPath(""))
        nested_empty_texture.GetAttr().SetColorSpace("sRGB")
        connected_source = material.CreateInput(
            "connected_texture_file",
            Sdf.ValueTypeNames.Asset,
        )
        connected_source.Set(Sdf.AssetPath("textures/connected.png"))
        connected_texture = shader.CreateInput(
            "connected_file", Sdf.ValueTypeNames.Asset
        )
        connected_texture.Set(Sdf.AssetPath(""))
        connected_texture.GetAttr().SetColorSpace("sRGB")
        connected_texture.ConnectToSource(connected_source)
        lib_stage.GetRootLayer().Save()

        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()

        task._create_full_stage(
            input_usd_path=input_usd,
            output_usd_path=output_usd,
            resolved_materials={"Steel": "/World/Looks/Steel"},
            prim_to_material={"/World/Body": "Steel"},
            is_library_based=True,
            material_library_path=str(library_usd),
            flatten_output=True,
        )

        out_stage = Usd.Stage.Open(str(output_usd))
        out_material = out_stage.GetPrimAtPath("/World/Looks/Steel")
        out_empty_texture = out_material.GetAttribute("inputs:base_color_texture_file")
        out_real_texture = out_material.GetAttribute(
            "inputs:geometry_normal_texture_file"
        )
        out_shader = out_stage.GetPrimAtPath("/World/Looks/Steel/Shader")
        out_nested_empty_texture = out_shader.GetAttribute(
            "inputs:emissive_texture_file"
        )
        out_connected_texture = out_shader.GetAttribute("inputs:connected_file")

        assert out_empty_texture.Get() == Sdf.AssetPath("")
        assert not out_empty_texture.HasColorSpace()
        assert out_real_texture.Get() == Sdf.AssetPath("textures/normal.png")
        assert out_real_texture.HasColorSpace()
        assert out_real_texture.GetColorSpace() == "raw"
        assert out_nested_empty_texture.Get() == Sdf.AssetPath("")
        assert not out_nested_empty_texture.HasColorSpace()
        assert out_connected_texture.Get() == Sdf.AssetPath("")
        assert out_connected_texture.HasColorSpace()
        assert out_connected_texture.GetColorSpace() == "sRGB"
        assert out_connected_texture.GetConnections()

    def test_flatten_removes_unbound_input_mdl_shader_with_unresolved_mdl(
        self, tmp_path
    ):
        """Flattened output should not keep stale unresolved input MDL shaders."""
        input_usd = tmp_path / "input.usda"
        output_usd = tmp_path / "output.usd"
        library_usd = tmp_path / "library.usd"

        stage = Usd.Stage.CreateNew(str(input_usd))
        root = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(root.GetPrim())
        mesh = UsdGeom.Mesh.Define(stage, "/World/Mesh")
        UsdGeom.Scope.Define(stage, "/World/Looks")
        default_mat = UsdShade.Material.Define(stage, "/World/Looks/DefaultMaterial")
        shader = UsdShade.Shader.Define(
            stage, "/World/Looks/DefaultMaterial/DefaultMaterial"
        )
        shader_prim = shader.GetPrim()
        shader_prim.CreateAttribute(
            "info:mdl:sourceAsset",
            Sdf.ValueTypeNames.Asset,
        ).Set(Sdf.AssetPath("OmniPBR.mdl"))
        shader_prim.CreateAttribute(
            "info:mdl:sourceAsset:subIdentifier",
            Sdf.ValueTypeNames.Token,
        ).Set("OmniPBR")
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(default_mat)
        stage.GetRootLayer().Save()

        lib_stage = Usd.Stage.CreateNew(str(library_usd))
        UsdGeom.Scope.Define(lib_stage, "/World")
        UsdGeom.Scope.Define(lib_stage, "/World/Looks")
        UsdShade.Material.Define(lib_stage, "/World/Looks/TestMaterial")
        lib_stage.GetRootLayer().Save()

        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()

        task._create_full_stage(
            input_usd_path=input_usd,
            output_usd_path=output_usd,
            resolved_materials={"TestMaterial": "/World/Looks/TestMaterial"},
            prim_to_material={"/World/Mesh": "TestMaterial"},
            is_library_based=True,
            material_library_path=str(library_usd),
            flatten_output=True,
        )

        out_stage = Usd.Stage.Open(str(output_usd))

        stale_shader = out_stage.GetPrimAtPath(
            "/World/Looks/DefaultMaterial/DefaultMaterial"
        )
        assert not stale_shader or not stale_shader.IsActive()
        material, _relationship = UsdShade.MaterialBindingAPI(
            out_stage.GetPrimAtPath("/World/Mesh")
        ).ComputeBoundMaterial()
        assert material
        assert str(material.GetPath()) == "/World/Looks/TestMaterial"

        mdl_paths = []
        for prim in out_stage.Traverse():
            attr = prim.GetAttribute("info:mdl:sourceAsset")
            if attr and attr.IsValid() and attr.Get():
                mdl_paths.append(attr.Get().path)
        assert "OmniPBR.mdl" not in mdl_paths

    def test_flatten_removes_instance_prototype_unresolved_mdl_shader(self, tmp_path):
        """Cleanup should also sanitize prototype contents materialized by flatten."""
        input_usd = tmp_path / "input.usda"
        external_usd = tmp_path / "external.usda"
        output_usd = tmp_path / "output.usd"

        external_stage = Usd.Stage.CreateNew(str(external_usd))
        external_root = UsdGeom.Xform.Define(external_stage, "/Asset")
        external_stage.SetDefaultPrim(external_root.GetPrim())
        UsdGeom.Xform.Define(external_stage, "/Asset/Prototype")
        UsdGeom.Mesh.Define(external_stage, "/Asset/Prototype/Mesh")
        UsdShade.Material.Define(
            external_stage,
            "/Asset/Prototype/Looks/StaleMaterial",
        )
        stale_shader = UsdShade.Shader.Define(
            external_stage,
            "/Asset/Prototype/Looks/StaleMaterial/MDLShader",
        )
        stale_shader.GetPrim().CreateAttribute(
            "info:mdl:sourceAsset",
            Sdf.ValueTypeNames.Asset,
        ).Set(Sdf.AssetPath("OmniPBR.mdl"))
        external_stage.GetRootLayer().Save()

        input_stage = Usd.Stage.CreateNew(str(input_usd))
        root = UsdGeom.Xform.Define(input_stage, "/World")
        input_stage.SetDefaultPrim(root.GetPrim())
        for name in ("InstA", "InstB"):
            instance = UsdGeom.Xform.Define(input_stage, f"/World/{name}")
            instance.GetPrim().GetReferences().AddReference(
                str(external_usd),
                "/Asset/Prototype",
            )
            instance.GetPrim().SetInstanceable(True)
        input_stage.GetRootLayer().Save()

        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()

        task._create_full_stage(
            input_usd_path=input_usd,
            output_usd_path=output_usd,
            resolved_materials={},
            prim_to_material={},
            flatten_output=True,
        )

        output_layer = Sdf.Layer.FindOrOpen(str(output_usd))
        assert output_layer
        assert "OmniPBR.mdl" not in output_layer.ExportToString()

    def test_bound_input_material_with_unresolved_mdl_stays_active(self, tmp_path):
        """Cleanup must not deactivate materials that are still bound."""
        input_usd = tmp_path / "input.usda"

        stage = Usd.Stage.CreateNew(str(input_usd))
        root = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(root.GetPrim())
        mesh = UsdGeom.Mesh.Define(stage, "/World/Mesh")
        material = UsdShade.Material.Define(stage, "/World/Looks/DefaultMaterial")
        shader = UsdShade.Shader.Define(
            stage, "/World/Looks/DefaultMaterial/DefaultMaterial"
        )
        shader.GetPrim().CreateAttribute(
            "info:mdl:sourceAsset",
            Sdf.ValueTypeNames.Asset,
        ).Set(Sdf.AssetPath("OmniPBR.mdl"))
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
        stage.GetRootLayer().Save()

        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()

        assert task._deactivate_unbound_unresolved_mdl_shaders(stage) == []
        assert stage.GetPrimAtPath("/World/Looks/DefaultMaterial").IsActive()
        assert stage.GetPrimAtPath(
            "/World/Looks/DefaultMaterial/DefaultMaterial"
        ).IsActive()

    def test_preview_bound_material_with_unresolved_mdl_stays_active(self, tmp_path):
        """Purpose-specific material bindings should protect material roots."""
        input_usd = tmp_path / "input.usda"

        stage = Usd.Stage.CreateNew(str(input_usd))
        root = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(root.GetPrim())
        mesh = UsdGeom.Mesh.Define(stage, "/World/Mesh")
        material = UsdShade.Material.Define(stage, "/World/Looks/PreviewMaterial")
        shader = UsdShade.Shader.Define(stage, "/World/Looks/PreviewMaterial/MDLShader")
        shader.GetPrim().CreateAttribute(
            "info:mdl:sourceAsset",
            Sdf.ValueTypeNames.Asset,
        ).Set(Sdf.AssetPath("OmniPBR.mdl"))
        shader_output = shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        material.CreateSurfaceOutput().ConnectToSource(shader_output)
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(
            material,
            UsdShade.Tokens.weakerThanDescendants,
            UsdShade.Tokens.preview,
        )
        stage.GetRootLayer().Save()

        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()

        assert task._collect_bound_material_paths(stage) == {
            "/World/Looks/PreviewMaterial"
        }
        assert task._deactivate_unbound_unresolved_mdl_shaders(stage) == []
        assert shader.GetPrim().IsActive()

    def test_cyclic_material_graph_cleanup_terminates(self, tmp_path):
        """Cyclic shader connections should not hang stale shader cleanup."""
        input_usd = tmp_path / "input.usda"

        stage = Usd.Stage.CreateNew(str(input_usd))
        root = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(root.GetPrim())
        mesh = UsdGeom.Mesh.Define(stage, "/World/Mesh")
        material = UsdShade.Material.Define(stage, "/World/Looks/CyclicMaterial")
        shader_a = UsdShade.Shader.Define(stage, "/World/Looks/CyclicMaterial/ShaderA")
        shader_b = UsdShade.Shader.Define(stage, "/World/Looks/CyclicMaterial/ShaderB")
        for shader in (shader_a, shader_b):
            shader.GetPrim().CreateAttribute(
                "info:mdl:sourceAsset",
                Sdf.ValueTypeNames.Asset,
            ).Set(Sdf.AssetPath("OmniPBR.mdl"))
        shader_a_output = shader_a.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        shader_b_output = shader_b.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        shader_a.CreateInput("cycle", Sdf.ValueTypeNames.Token).ConnectToSource(
            shader_b_output
        )
        shader_b.CreateInput("cycle", Sdf.ValueTypeNames.Token).ConnectToSource(
            shader_a_output
        )
        material.CreateSurfaceOutput().ConnectToSource(shader_a_output)
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
        stage.GetRootLayer().Save()

        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()

        assert task._collect_material_graph_prim_paths(stage, material.GetPrim()) == {
            "/World/Looks/CyclicMaterial",
            "/World/Looks/CyclicMaterial/ShaderA",
            "/World/Looks/CyclicMaterial/ShaderB",
        }
        assert task._deactivate_unbound_unresolved_mdl_shaders(stage) == []
        assert shader_a.GetPrim().IsActive()
        assert shader_b.GetPrim().IsActive()

    def test_protected_replacement_material_prunes_stale_child_shader(self, tmp_path):
        """Same-path replacement should remove obsolete unresolved child shaders."""
        input_usd = tmp_path / "input.usda"

        stage = Usd.Stage.CreateNew(str(input_usd))
        root = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(root.GetPrim())
        mesh = UsdGeom.Mesh.Define(stage, "/World/Mesh")
        material = UsdShade.Material.Define(stage, "/World/Looks/TestMaterial")
        old_shader = UsdShade.Shader.Define(
            stage, "/World/Looks/TestMaterial/OldShader"
        )
        old_shader.GetPrim().CreateAttribute(
            "info:mdl:sourceAsset",
            Sdf.ValueTypeNames.Asset,
        ).Set(Sdf.AssetPath("OmniPBR.mdl"))
        new_shader = UsdShade.Shader.Define(
            stage, "/World/Looks/TestMaterial/NewShader"
        )
        new_shader_output = new_shader.CreateOutput(
            "surface",
            Sdf.ValueTypeNames.Token,
        )
        material.CreateSurfaceOutput().ConnectToSource(new_shader_output)
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
        stage.GetRootLayer().Save()

        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()

        assert task._deactivate_unbound_unresolved_mdl_shaders(
            stage,
            protected_material_paths={"/World/Looks/TestMaterial"},
        ) == ["/World/Looks/TestMaterial/OldShader"]
        assert not old_shader.GetPrim().IsActive()
        assert new_shader.GetPrim().IsActive()
        assert "OmniPBR.mdl" not in stage.Flatten().ExportToString()

    def test_protected_material_keeps_connected_unresolved_mdl_shader(self, tmp_path):
        """Protected cleanup should keep shaders reached from material outputs."""
        input_usd = tmp_path / "input.usda"

        stage = Usd.Stage.CreateNew(str(input_usd))
        root = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(root.GetPrim())
        mesh = UsdGeom.Mesh.Define(stage, "/World/Mesh")
        material = UsdShade.Material.Define(stage, "/World/Looks/TestMaterial")
        shader = UsdShade.Shader.Define(stage, "/World/Looks/TestMaterial/MDLShader")
        shader.GetPrim().CreateAttribute(
            "info:mdl:sourceAsset",
            Sdf.ValueTypeNames.Asset,
        ).Set(Sdf.AssetPath("OmniPBR.mdl"))
        shader_output = shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        material.CreateSurfaceOutput().ConnectToSource(shader_output)
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
        stage.GetRootLayer().Save()

        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()

        assert (
            task._deactivate_unbound_unresolved_mdl_shaders(
                stage,
                protected_material_paths={"/World/Looks/TestMaterial"},
            )
            == []
        )
        assert shader.GetPrim().IsActive()

    def test_protected_replacement_ignores_stale_child_shader_connections(
        self, tmp_path
    ):
        """Unused old shaders should not protect other stale materials."""
        input_usd = tmp_path / "input.usda"

        stage = Usd.Stage.CreateNew(str(input_usd))
        root = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(root.GetPrim())
        mesh = UsdGeom.Mesh.Define(stage, "/World/Mesh")
        material = UsdShade.Material.Define(stage, "/World/Looks/TestMaterial")
        old_shader = UsdShade.Shader.Define(
            stage, "/World/Looks/TestMaterial/OldShader"
        )
        old_shader.GetPrim().CreateAttribute(
            "info:mdl:sourceAsset",
            Sdf.ValueTypeNames.Asset,
        ).Set(Sdf.AssetPath("OmniPBR.mdl"))
        UsdShade.Material.Define(stage, "/World/Looks/HelperMaterial")
        helper_shader = UsdShade.Shader.Define(
            stage, "/World/Looks/HelperMaterial/MDLShader"
        )
        helper_shader.GetPrim().CreateAttribute(
            "info:mdl:sourceAsset",
            Sdf.ValueTypeNames.Asset,
        ).Set(Sdf.AssetPath("OmniPBR.mdl"))
        helper_output = helper_shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        old_shader.CreateInput("unused", Sdf.ValueTypeNames.Token).ConnectToSource(
            helper_output
        )
        new_shader = UsdShade.Shader.Define(
            stage, "/World/Looks/TestMaterial/NewShader"
        )
        new_shader_output = new_shader.CreateOutput(
            "surface",
            Sdf.ValueTypeNames.Token,
        )
        material.CreateSurfaceOutput().ConnectToSource(new_shader_output)
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
        stage.GetRootLayer().Save()

        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()

        deactivated_paths = task._deactivate_unbound_unresolved_mdl_shaders(
            stage,
            protected_material_paths={"/World/Looks/TestMaterial"},
        )

        assert set(deactivated_paths) == {
            "/World/Looks/TestMaterial/OldShader",
            "/World/Looks/HelperMaterial/MDLShader",
        }
        assert not old_shader.GetPrim().IsActive()
        assert not helper_shader.GetPrim().IsActive()
        assert new_shader.GetPrim().IsActive()

    def test_remote_resolved_mdl_uri_is_not_treated_as_missing(self, tmp_path):
        """Remote resolved asset URIs should not go through local Path.exists()."""

        class AssetValue:
            path = "Materials/RemoteMaterial.mdl"
            resolvedPath = "omniverse://server.example/Materials/RemoteMaterial.mdl"

        task = ApplyMaterialsToUSDTask()

        assert not task._is_uri_asset_path("C:/Materials/RemoteMaterial.mdl")
        assert not task._is_uri_asset_path(r"C:\Materials\RemoteMaterial.mdl")
        assert not task._is_unresolved_local_asset_path(
            AssetValue(),
            AssetValue.path,
            tmp_path,
        )

    def test_authored_mdl_uri_is_treated_as_unsafe(self, tmp_path):
        """Authored resolver URIs must not survive into generated USD output."""

        class AssetValue:
            path = "https://metadata.example.invalid/Materials/Evil.mdl"
            resolvedPath = ""

        task = ApplyMaterialsToUSDTask()

        assert task._is_uri_asset_path(AssetValue.path)
        assert task._is_unresolved_local_asset_path(
            AssetValue(),
            AssetValue.path,
            tmp_path,
        )

    def test_make_path_relative_rejects_resolver_uri_material(self, tmp_path):
        """LLM-provided material URLs should fail before OVRTX can resolve them."""
        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()

        with pytest.raises(ValueError, match="resolver URI material path"):
            task._make_path_relative_to_usd(
                "https://metadata.example.invalid/Evil.mdl",
                tmp_path / "output.usda",
            )

        with pytest.raises(ValueError, match="resolver URI material path"):
            task._make_path_relative_to_usd(
                "file:///var/run/secrets/kubernetes.io/serviceaccount/token",
                tmp_path / "output.usda",
            )

    def test_remap_single_asset_path_clears_unsafe_resolver_paths(self, tmp_path):
        """Copied material libraries should not author URI or host-absolute assets."""
        source_dir = tmp_path / "library"
        target_dir = tmp_path / "out"
        source_dir.mkdir()
        target_dir.mkdir()
        local_asset = source_dir / "textures" / "albedo.png"
        local_asset.parent.mkdir()
        local_asset.write_bytes(b"png")

        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()

        assert (
            task._remap_single_asset_path(
                str(local_asset),
                source_dir,
                target_dir,
            )
            == "../library/textures/albedo.png"
        )
        for unsafe_path in (
            "https://metadata.example.invalid/albedo.png",
            "file:///etc/shadow",
            "/etc/shadow",
            "C:/Users/secret/material.mdl",
            "../outside/material.mdl",
        ):
            assert (
                task._remap_single_asset_path(
                    unsafe_path,
                    source_dir,
                    target_dir,
                )
                == ""
            )

    def test_windows_absolute_mdl_path_is_not_treated_as_uri(self, tmp_path):
        """Windows drive paths should still be checked as local asset paths."""

        class AssetValue:
            path = "C:/Materials/MissingMaterial.mdl"
            resolvedPath = ""

        task = ApplyMaterialsToUSDTask()

        assert task._is_absolute_asset_path(AssetValue.path)
        assert not task._is_uri_asset_path(AssetValue.path)
        assert task._is_unresolved_local_asset_path(
            AssetValue(),
            AssetValue.path,
            tmp_path,
        )

    def test_resolved_mdl_package_path_is_not_treated_as_missing(self, tmp_path):
        """Any non-empty resolver path should be trusted, including package paths."""

        class AssetValue:
            path = "Materials/PackagedMaterial.mdl"
            resolvedPath = "asset.usdz[Materials/PackagedMaterial.mdl]"

        task = ApplyMaterialsToUSDTask()

        assert not task._is_unresolved_local_asset_path(
            AssetValue(),
            AssetValue.path,
            tmp_path,
        )

    def test_layer_relative_asset_fallback_checks_authored_layer_dir(self, tmp_path):
        """Fallback local checks should honor the layer that authored the asset."""

        class AssetValue:
            path = "materials/LayerMaterial.mdl"
            resolvedPath = ""

        subdir = tmp_path / "layers"
        material_dir = subdir / "materials"
        material_dir.mkdir(parents=True)
        (material_dir / "LayerMaterial.mdl").write_text("mdl", encoding="utf-8")

        sublayer_path = subdir / "materials.usda"
        sublayer_stage = Usd.Stage.CreateNew(str(sublayer_path))
        material = UsdShade.Material.Define(
            sublayer_stage,
            "/World/Looks/LayerMaterial",
        )
        shader = UsdShade.Shader.Define(
            sublayer_stage,
            "/World/Looks/LayerMaterial/MDLShader",
        )
        shader.GetPrim().CreateAttribute(
            "info:mdl:sourceAsset",
            Sdf.ValueTypeNames.Asset,
        ).Set(Sdf.AssetPath(AssetValue.path))
        shader_output = shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        material.CreateSurfaceOutput().ConnectToSource(shader_output)
        sublayer_stage.GetRootLayer().Save()

        root_path = tmp_path / "root.usda"
        root_stage = Usd.Stage.CreateNew(str(root_path))
        root_stage.GetRootLayer().subLayerPaths.append(str(sublayer_path))
        root_stage.GetRootLayer().Save()
        stage = Usd.Stage.Open(str(root_path))
        attr = stage.GetPrimAtPath("/World/Looks/LayerMaterial/MDLShader").GetAttribute(
            "info:mdl:sourceAsset"
        )

        task = ApplyMaterialsToUSDTask()

        base_dirs = task._asset_base_dirs_for_attr(stage, attr)
        assert subdir in base_dirs
        assert not task._is_unresolved_local_asset_path(
            AssetValue(),
            AssetValue.path,
            base_dirs,
        )

    def test_unbound_unresolved_mdl_cleanup_preserves_fallback_shader(self, tmp_path):
        """Only stale unresolved MDL shaders should be hidden, not whole materials."""
        input_usd = tmp_path / "input.usda"

        stage = Usd.Stage.CreateNew(str(input_usd))
        root = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(root.GetPrim())
        material = UsdShade.Material.Define(stage, "/World/Looks/DefaultMaterial")
        mdl_shader = UsdShade.Shader.Define(
            stage, "/World/Looks/DefaultMaterial/MDLShader"
        )
        mdl_shader.GetPrim().CreateAttribute(
            "info:mdl:sourceAsset",
            Sdf.ValueTypeNames.Asset,
        ).Set(Sdf.AssetPath("OmniPBR.mdl"))
        preview_shader = UsdShade.Shader.Define(
            stage, "/World/Looks/DefaultMaterial/PreviewShader"
        )
        stage.GetRootLayer().Save()

        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()

        assert task._deactivate_unbound_unresolved_mdl_shaders(stage) == [
            "/World/Looks/DefaultMaterial/MDLShader"
        ]
        assert material.GetPrim().IsActive()
        assert not mdl_shader.GetPrim().IsActive()
        assert preview_shader.GetPrim().IsActive()

    def test_unbound_material_with_loose_shader_is_deactivated(self, tmp_path):
        """Stale material graphs can use sibling shaders outside the material."""
        input_usd = tmp_path / "input.usda"

        stage = Usd.Stage.CreateNew(str(input_usd))
        root = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(root.GetPrim())
        old_material = UsdShade.Material.Define(stage, "/World/Looks/OldMaterial")
        old_shader = UsdShade.Shader.Define(stage, "/World/Looks/OldMaterialShader")
        old_shader.GetPrim().CreateAttribute(
            "info:mdl:sourceAsset",
            Sdf.ValueTypeNames.Asset,
        ).Set(Sdf.AssetPath("OmniPBR.mdl"))
        old_shader_output = old_shader.CreateOutput(
            "surface",
            Sdf.ValueTypeNames.Token,
        )
        old_material.CreateSurfaceOutput().ConnectToSource(old_shader_output)
        stage.GetRootLayer().Save()

        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()

        assert task._deactivate_unbound_unresolved_mdl_shaders(stage) == [
            "/World/Looks/OldMaterialShader"
        ]
        assert not old_shader.GetPrim().IsActive()
        assert "OmniPBR.mdl" not in stage.Flatten().ExportToString()

    def test_bound_material_with_loose_shader_stays_active(self, tmp_path):
        """Loose shaders used by resolved bound materials remain protected."""
        input_usd = tmp_path / "input.usda"

        stage = Usd.Stage.CreateNew(str(input_usd))
        root = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(root.GetPrim())
        mesh = UsdGeom.Mesh.Define(stage, "/World/Mesh")
        bound_material = UsdShade.Material.Define(stage, "/World/Looks/BoundMaterial")
        stale_material = UsdShade.Material.Define(stage, "/World/Looks/StaleMaterial")
        shared_shader = UsdShade.Shader.Define(stage, "/World/Looks/SharedShader")
        shared_shader.GetPrim().CreateAttribute(
            "info:mdl:sourceAsset",
            Sdf.ValueTypeNames.Asset,
        ).Set(Sdf.AssetPath("OmniPBR.mdl"))
        shared_output = shared_shader.CreateOutput(
            "surface",
            Sdf.ValueTypeNames.Token,
        )
        bound_material.CreateSurfaceOutput().ConnectToSource(shared_output)
        stale_material.CreateSurfaceOutput().ConnectToSource(shared_output)
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(bound_material)
        stage.GetRootLayer().Save()

        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()

        assert task._deactivate_unbound_unresolved_mdl_shaders(stage) == []
        assert shared_shader.GetPrim().IsActive()

    def test_standalone_loose_unresolved_mdl_shader_is_deactivated(self, tmp_path):
        """Loose unresolved shaders outside any reachable graph should be hidden."""
        input_usd = tmp_path / "input.usda"

        stage = Usd.Stage.CreateNew(str(input_usd))
        root = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(root.GetPrim())
        loose_shader = UsdShade.Shader.Define(stage, "/World/Looks/LooseShader")
        loose_shader.GetPrim().CreateAttribute(
            "info:mdl:sourceAsset",
            Sdf.ValueTypeNames.Asset,
        ).Set(Sdf.AssetPath("OmniPBR.mdl"))
        stage.GetRootLayer().Save()

        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()

        assert task._deactivate_unbound_unresolved_mdl_shaders(stage) == [
            "/World/Looks/LooseShader"
        ]
        assert not loose_shader.GetPrim().IsActive()
        assert "OmniPBR.mdl" not in stage.Flatten().ExportToString()

    def test_composition_target_material_with_unresolved_mdl_is_protected(
        self, tmp_path
    ):
        """Materials used as inherit bases should not be treated as stale."""
        input_usd = tmp_path / "input.usda"

        stage = Usd.Stage.CreateNew(str(input_usd))
        root = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(root.GetPrim())
        mesh = UsdGeom.Mesh.Define(stage, "/World/Mesh")
        base_material = UsdShade.Material.Define(stage, "/World/Looks/BaseMaterial")
        base_shader = UsdShade.Shader.Define(
            stage, "/World/Looks/BaseMaterial/MDLShader"
        )
        base_shader.GetPrim().CreateAttribute(
            "info:mdl:sourceAsset",
            Sdf.ValueTypeNames.Asset,
        ).Set(Sdf.AssetPath("OmniPBR.mdl"))
        child_material = UsdShade.Material.Define(stage, "/World/Looks/ChildMaterial")
        child_material.GetPrim().GetInherits().AddInherit("/World/Looks/BaseMaterial")
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(child_material)
        stage.GetRootLayer().Save()

        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()

        assert task._deactivate_unbound_unresolved_mdl_shaders(stage) == []
        assert base_material.GetPrim().IsActive()
        assert base_shader.GetPrim().IsActive()

    def test_specializes_target_material_with_unresolved_mdl_is_protected(
        self, tmp_path
    ):
        """Materials used as specializes bases should not be treated as stale."""
        input_usd = tmp_path / "input.usda"

        stage = Usd.Stage.CreateNew(str(input_usd))
        root = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(root.GetPrim())
        mesh = UsdGeom.Mesh.Define(stage, "/World/Mesh")
        base_material = UsdShade.Material.Define(stage, "/World/Looks/BaseMaterial")
        base_shader = UsdShade.Shader.Define(
            stage, "/World/Looks/BaseMaterial/MDLShader"
        )
        base_shader.GetPrim().CreateAttribute(
            "info:mdl:sourceAsset",
            Sdf.ValueTypeNames.Asset,
        ).Set(Sdf.AssetPath("OmniPBR.mdl"))
        child_material = UsdShade.Material.Define(stage, "/World/Looks/ChildMaterial")
        child_material.GetPrim().GetSpecializes().AddSpecialize(
            "/World/Looks/BaseMaterial"
        )
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(child_material)
        stage.GetRootLayer().Save()

        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()

        assert task._deactivate_unbound_unresolved_mdl_shaders(stage) == []
        assert base_material.GetPrim().IsActive()
        assert base_shader.GetPrim().IsActive()

    def test_payload_target_material_with_unresolved_mdl_is_protected(self, tmp_path):
        """Materials used as payload bases should not be treated as stale."""
        input_usd = tmp_path / "input.usda"

        stage = Usd.Stage.CreateNew(str(input_usd))
        root = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(root.GetPrim())
        mesh = UsdGeom.Mesh.Define(stage, "/World/Mesh")
        base_material = UsdShade.Material.Define(stage, "/World/Looks/BaseMaterial")
        base_shader = UsdShade.Shader.Define(
            stage, "/World/Looks/BaseMaterial/MDLShader"
        )
        base_shader.GetPrim().CreateAttribute(
            "info:mdl:sourceAsset",
            Sdf.ValueTypeNames.Asset,
        ).Set(Sdf.AssetPath("OmniPBR.mdl"))
        child_material = UsdShade.Material.Define(stage, "/World/Looks/ChildMaterial")
        child_material.GetPrim().GetPayloads().AddInternalPayload(
            "/World/Looks/BaseMaterial"
        )
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(child_material)
        stage.GetRootLayer().Save()

        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()

        assert task._deactivate_unbound_unresolved_mdl_shaders(stage) == []
        assert base_material.GetPrim().IsActive()
        assert base_shader.GetPrim().IsActive()

    def test_shader_composition_target_material_with_unresolved_mdl_is_protected(
        self, tmp_path
    ):
        """Materials owning shader composition targets should not be stale."""
        input_usd = tmp_path / "input.usda"

        stage = Usd.Stage.CreateNew(str(input_usd))
        root = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(root.GetPrim())
        mesh = UsdGeom.Mesh.Define(stage, "/World/Mesh")
        base_material = UsdShade.Material.Define(stage, "/World/Looks/BaseMaterial")
        base_shader = UsdShade.Shader.Define(
            stage, "/World/Looks/BaseMaterial/MDLShader"
        )
        base_shader.GetPrim().CreateAttribute(
            "info:mdl:sourceAsset",
            Sdf.ValueTypeNames.Asset,
        ).Set(Sdf.AssetPath("OmniPBR.mdl"))
        child_material = UsdShade.Material.Define(stage, "/World/Looks/ChildMaterial")
        child_shader = UsdShade.Shader.Define(
            stage, "/World/Looks/ChildMaterial/MDLShader"
        )
        child_shader.GetPrim().GetInherits().AddInherit(
            "/World/Looks/BaseMaterial/MDLShader"
        )
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(child_material)
        stage.GetRootLayer().Save()

        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()

        assert task._deactivate_unbound_unresolved_mdl_shaders(stage) == []
        assert base_material.GetPrim().IsActive()
        assert base_shader.GetPrim().IsActive()

    def test_connected_shader_material_with_unresolved_mdl_is_protected(self, tmp_path):
        """Materials reached through bound material connections should be active."""
        input_usd = tmp_path / "input.usda"

        stage = Usd.Stage.CreateNew(str(input_usd))
        root = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(root.GetPrim())
        mesh = UsdGeom.Mesh.Define(stage, "/World/Mesh")
        bound_material = UsdShade.Material.Define(stage, "/World/Looks/BoundMaterial")
        helper_material = UsdShade.Material.Define(stage, "/World/Looks/HelperMaterial")
        helper_shader = UsdShade.Shader.Define(
            stage, "/World/Looks/HelperMaterial/MDLShader"
        )
        helper_shader.GetPrim().CreateAttribute(
            "info:mdl:sourceAsset",
            Sdf.ValueTypeNames.Asset,
        ).Set(Sdf.AssetPath("OmniPBR.mdl"))
        helper_output = helper_shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        bound_material.CreateSurfaceOutput().ConnectToSource(helper_output)
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(bound_material)
        stage.GetRootLayer().Save()

        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()

        assert task._deactivate_unbound_unresolved_mdl_shaders(stage) == []
        assert helper_material.GetPrim().IsActive()
        assert helper_shader.GetPrim().IsActive()

    def test_collection_binding_does_not_protect_unrelated_stale_material(
        self, tmp_path
    ):
        """Collection targets should not become material roots for reachability."""
        input_usd = tmp_path / "input.usda"

        stage = Usd.Stage.CreateNew(str(input_usd))
        root = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(root.GetPrim())
        mesh = UsdGeom.Mesh.Define(stage, "/World/Mesh")
        bound_material = UsdShade.Material.Define(stage, "/World/Looks/BoundMaterial")
        bound_shader = UsdShade.Shader.Define(
            stage, "/World/Looks/BoundMaterial/MDLShader"
        )
        bound_shader.GetPrim().CreateAttribute(
            "info:mdl:sourceAsset",
            Sdf.ValueTypeNames.Asset,
        ).Set(Sdf.AssetPath("OmniPBR.mdl"))
        bound_output = bound_shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        bound_material.CreateSurfaceOutput().ConnectToSource(bound_output)
        stale_material = UsdShade.Material.Define(stage, "/World/Looks/StaleMaterial")
        stale_shader = UsdShade.Shader.Define(
            stage, "/World/Looks/StaleMaterial/MDLShader"
        )
        stale_shader.GetPrim().CreateAttribute(
            "info:mdl:sourceAsset",
            Sdf.ValueTypeNames.Asset,
        ).Set(Sdf.AssetPath("OmniPBR.mdl"))
        stale_output = stale_shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        stale_material.CreateSurfaceOutput().ConnectToSource(stale_output)

        collection = Usd.CollectionAPI.Apply(root.GetPrim(), "all")
        collection.GetIncludesRel().AddTarget(mesh.GetPath())
        UsdShade.MaterialBindingAPI.Apply(root.GetPrim()).Bind(
            collection,
            bound_material,
        )
        stage.GetRootLayer().Save()

        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()

        assert task._deactivate_unbound_unresolved_mdl_shaders(stage) == [
            "/World/Looks/StaleMaterial/MDLShader"
        ]
        assert bound_shader.GetPrim().IsActive()
        assert not stale_shader.GetPrim().IsActive()

    def test_overridden_collection_binding_does_not_protect_stale_material(
        self, tmp_path
    ):
        """Only resolved material bindings should protect material roots."""
        input_usd = tmp_path / "input.usda"

        stage = Usd.Stage.CreateNew(str(input_usd))
        root = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(root.GetPrim())
        mesh = UsdGeom.Mesh.Define(stage, "/World/Mesh")
        stale_material = UsdShade.Material.Define(stage, "/World/Looks/StaleMaterial")
        stale_shader = UsdShade.Shader.Define(
            stage, "/World/Looks/StaleMaterial/MDLShader"
        )
        stale_shader.GetPrim().CreateAttribute(
            "info:mdl:sourceAsset",
            Sdf.ValueTypeNames.Asset,
        ).Set(Sdf.AssetPath("OmniPBR.mdl"))
        stale_output = stale_shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        stale_material.CreateSurfaceOutput().ConnectToSource(stale_output)
        bound_material = UsdShade.Material.Define(stage, "/World/Looks/BoundMaterial")
        bound_shader = UsdShade.Shader.Define(
            stage, "/World/Looks/BoundMaterial/MDLShader"
        )
        bound_shader.GetPrim().CreateAttribute(
            "info:mdl:sourceAsset",
            Sdf.ValueTypeNames.Asset,
        ).Set(Sdf.AssetPath("OmniPBR.mdl"))
        bound_output = bound_shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        bound_material.CreateSurfaceOutput().ConnectToSource(bound_output)

        collection = Usd.CollectionAPI.Apply(root.GetPrim(), "all")
        collection.GetIncludesRel().AddTarget(mesh.GetPath())
        UsdShade.MaterialBindingAPI.Apply(root.GetPrim()).Bind(
            collection,
            stale_material,
        )
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(bound_material)
        stage.GetRootLayer().Save()

        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()

        assert task._collect_bound_material_paths(stage) == {
            "/World/Looks/BoundMaterial"
        }
        assert task._deactivate_unbound_unresolved_mdl_shaders(stage) == [
            "/World/Looks/StaleMaterial/MDLShader"
        ]
        assert bound_shader.GetPrim().IsActive()
        assert not stale_shader.GetPrim().IsActive()

    def test_empty_material_binding_relationship_is_ignored(self, tmp_path):
        """Cleared bindings should not be resolved as material targets."""
        input_usd = tmp_path / "input.usda"

        stage = Usd.Stage.CreateNew(str(input_usd))
        root = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(root.GetPrim())
        mesh = UsdGeom.Mesh.Define(stage, "/World/Mesh")
        mesh.GetPrim().CreateRelationship("material:binding").SetTargets([])
        UsdShade.Material.Define(stage, "/World/Looks/StaleMaterial")
        stale_shader = UsdShade.Shader.Define(
            stage, "/World/Looks/StaleMaterial/MDLShader"
        )
        stale_shader.GetPrim().CreateAttribute(
            "info:mdl:sourceAsset",
            Sdf.ValueTypeNames.Asset,
        ).Set(Sdf.AssetPath("OmniPBR.mdl"))
        stage.GetRootLayer().Save()

        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()

        assert task._collect_bound_material_paths(stage) == set()
        assert task._deactivate_unbound_unresolved_mdl_shaders(stage) == [
            "/World/Looks/StaleMaterial/MDLShader"
        ]
        assert not stale_shader.GetPrim().IsActive()

    def test_unbound_class_material_shader_with_unresolved_mdl_is_deactivated(
        self, tmp_path
    ):
        """Cleanup should include abstract class material templates."""
        input_usd = tmp_path / "input.usda"

        stage = Usd.Stage.CreateNew(str(input_usd))
        root = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(root.GetPrim())
        class_material = UsdShade.Material.Define(stage, "/World/Looks/ClassMaterial")
        class_material.GetPrim().SetSpecifier(Sdf.SpecifierClass)
        class_shader = UsdShade.Shader.Define(
            stage, "/World/Looks/ClassMaterial/MDLShader"
        )
        class_shader.GetPrim().SetSpecifier(Sdf.SpecifierClass)
        class_shader.GetPrim().CreateAttribute(
            "info:mdl:sourceAsset",
            Sdf.ValueTypeNames.Asset,
        ).Set(Sdf.AssetPath("OmniPBR.mdl"))
        stage.GetRootLayer().Save()

        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()

        assert class_material.GetPrim().IsAbstract()
        assert class_shader.GetPrim().IsAbstract()
        assert task._deactivate_unbound_unresolved_mdl_shaders(stage) == [
            "/World/Looks/ClassMaterial/MDLShader"
        ]
        assert not class_shader.GetPrim().IsActive()
        assert "OmniPBR.mdl" not in stage.Flatten().ExportToString()

    def test_stale_composition_base_with_unresolved_mdl_is_not_protected(
        self, tmp_path
    ):
        """Unbound material composition graphs should not protect stale bases."""
        input_usd = tmp_path / "input.usda"

        stage = Usd.Stage.CreateNew(str(input_usd))
        root = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(root.GetPrim())
        mesh = UsdGeom.Mesh.Define(stage, "/World/Mesh")
        bound_material = UsdShade.Material.Define(stage, "/World/Looks/BoundMaterial")
        UsdShade.Material.Define(stage, "/World/Looks/BaseMaterial")
        base_shader = UsdShade.Shader.Define(
            stage, "/World/Looks/BaseMaterial/MDLShader"
        )
        base_shader.GetPrim().CreateAttribute(
            "info:mdl:sourceAsset",
            Sdf.ValueTypeNames.Asset,
        ).Set(Sdf.AssetPath("OmniPBR.mdl"))
        stale_child = UsdShade.Material.Define(stage, "/World/Looks/StaleChild")
        stale_child.GetPrim().GetInherits().AddInherit("/World/Looks/BaseMaterial")
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(bound_material)
        stage.GetRootLayer().Save()

        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()

        assert task._deactivate_unbound_unresolved_mdl_shaders(stage) == [
            "/World/Looks/BaseMaterial/MDLShader"
        ]
        assert not base_shader.GetPrim().IsActive()

    def test_external_reference_prim_path_does_not_protect_local_stale_material(
        self, tmp_path
    ):
        """External reference prim paths should not resolve in the current stage."""
        input_usd = tmp_path / "input.usda"
        external_usd = tmp_path / "external.usda"

        external_stage = Usd.Stage.CreateNew(str(external_usd))
        external_root = UsdGeom.Xform.Define(external_stage, "/World")
        external_stage.SetDefaultPrim(external_root.GetPrim())
        UsdShade.Material.Define(external_stage, "/World/Looks/BaseMaterial")
        external_stage.GetRootLayer().Save()

        stage = Usd.Stage.CreateNew(str(input_usd))
        root = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(root.GetPrim())
        mesh = UsdGeom.Mesh.Define(stage, "/World/Mesh")
        bound_material = UsdShade.Material.Define(stage, "/World/Looks/BoundMaterial")
        bound_material.GetPrim().GetReferences().AddReference(
            str(external_usd),
            "/World/Looks/BaseMaterial",
        )
        UsdShade.Material.Define(stage, "/World/Looks/BaseMaterial")
        stale_shader = UsdShade.Shader.Define(
            stage, "/World/Looks/BaseMaterial/MDLShader"
        )
        stale_shader.GetPrim().CreateAttribute(
            "info:mdl:sourceAsset",
            Sdf.ValueTypeNames.Asset,
        ).Set(Sdf.AssetPath("OmniPBR.mdl"))
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(bound_material)
        stage.GetRootLayer().Save()

        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()

        assert task._deactivate_unbound_unresolved_mdl_shaders(stage) == [
            "/World/Looks/BaseMaterial/MDLShader"
        ]
        assert not stale_shader.GetPrim().IsActive()

    def test_nested_bound_material_shader_is_not_deactivated_by_stale_parent(
        self, tmp_path
    ):
        """Cleanup should not cross nested material ownership boundaries."""
        input_usd = tmp_path / "input.usda"

        stage = Usd.Stage.CreateNew(str(input_usd))
        root = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(root.GetPrim())
        mesh = UsdGeom.Mesh.Define(stage, "/World/Mesh")
        parent_material = UsdShade.Material.Define(stage, "/World/Looks/ParentMaterial")
        parent_shader = UsdShade.Shader.Define(
            stage, "/World/Looks/ParentMaterial/MDLShader"
        )
        parent_shader.GetPrim().CreateAttribute(
            "info:mdl:sourceAsset",
            Sdf.ValueTypeNames.Asset,
        ).Set(Sdf.AssetPath("OmniPBR.mdl"))
        child_material = UsdShade.Material.Define(
            stage, "/World/Looks/ParentMaterial/ChildMaterial"
        )
        child_shader = UsdShade.Shader.Define(
            stage, "/World/Looks/ParentMaterial/ChildMaterial/MDLShader"
        )
        child_shader.GetPrim().CreateAttribute(
            "info:mdl:sourceAsset",
            Sdf.ValueTypeNames.Asset,
        ).Set(Sdf.AssetPath("OmniPBR.mdl"))
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(child_material)
        stage.GetRootLayer().Save()

        task = ApplyMaterialsToUSDTask()
        task.listener = MagicMock()

        assert task._deactivate_unbound_unresolved_mdl_shaders(stage) == [
            "/World/Looks/ParentMaterial/MDLShader"
        ]
        assert parent_material.GetPrim().IsActive()
        assert not parent_shader.GetPrim().IsActive()
        assert child_material.GetPrim().IsActive()
        assert child_shader.GetPrim().IsActive()
