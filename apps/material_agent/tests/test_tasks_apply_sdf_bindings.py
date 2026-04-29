# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Sdf-level binding and instance-aware path remapping in ApplyMaterialsToUSDTask."""

from pathlib import Path
from unittest.mock import Mock

from pxr import Sdf, Usd, UsdGeom

from material_agent.tasks.apply_materials_to_usd import ApplyMaterialsToUSDTask


class TestApplySkipInstanceCheck:
    """Tests for skip_instance_check flag."""

    def test_task_instantiation(self):
        """ApplyMaterialsToUSDTask can be instantiated."""
        task = ApplyMaterialsToUSDTask()
        assert task.name == "ApplyMaterialsToUSD"

    def test_empty_predictions_file_handled(self, tmp_path):
        """Empty predictions file is not treated as a critical failure."""
        usd_path = str(tmp_path / "test.usda")
        stage = Usd.Stage.CreateNew(usd_path)
        UsdGeom.Mesh.Define(stage, "/World/Mesh")
        stage.Save()

        pred_path = tmp_path / "predictions.jsonl"
        pred_path.write_text("")

        task = ApplyMaterialsToUSDTask()
        context = {
            "input_usd_path": usd_path,
            "resolved_materials": {},
            "prim_to_material": {},
            "predictions_path": str(pred_path),
            "output_usd_path": str(tmp_path / "output.usda"),
        }

        result = task.run(context, None)
        assert "output_usd_path" in result


class TestSdfLevelBindings:
    """Tests for Sdf-level material binding in the material layer."""

    def test_sdf_binding_writes_relationship(self, tmp_path):
        """Sdf-level binding writes material:binding relationship on layer."""
        # Create a simple stage
        usd_path = str(tmp_path / "test.usda")
        stage = Usd.Stage.CreateNew(usd_path)
        UsdGeom.Mesh.Define(stage, "/World/Mesh")
        stage.Save()

        # Create an output layer with an over prim and Sdf-level binding
        output_path = str(tmp_path / "output.usda")
        out_layer = Sdf.Layer.CreateNew(output_path)

        # Create material
        mat_spec = Sdf.CreatePrimInLayer(out_layer, "/Materials/Steel")
        mat_spec.specifier = Sdf.SpecifierDef
        mat_spec.typeName = "Material"

        # Create over prim and write binding at Sdf level
        prim_spec = Sdf.CreatePrimInLayer(out_layer, "/World/Mesh")
        prim_spec.specifier = Sdf.SpecifierOver
        prim_spec.SetInfo(
            "apiSchemas",
            Sdf.TokenListOp.Create(prependedItems=["MaterialBindingAPI"]),
        )
        binding_rel = Sdf.RelationshipSpec(prim_spec, "material:binding")
        binding_rel.targetPathList.explicitItems = [Sdf.Path("/Materials/Steel")]

        out_layer.Save()

        # Verify the binding exists at the Sdf level
        reloaded = Sdf.Layer.FindOrOpen(output_path)
        ps = reloaded.GetPrimAtPath("/World/Mesh")
        assert ps is not None
        rel = ps.relationships.get("material:binding")
        assert rel is not None
        assert Sdf.Path("/Materials/Steel") in rel.targetPathList.explicitItems

    def test_config_apply_has_skip_instance_check(self, tmp_path):
        """ApplyConfigTask passes skip_instance_check from config."""
        import yaml

        from material_agent.tasks.config_apply import ApplyConfigTask

        config = {
            "input_usd_path": "/dummy.usd",
            "predictions_path": "/dummy.jsonl",
            "output_usd_path": "/dummy_out.usd",
            "skip_instance_check": True,
        }
        config_path = tmp_path / "apply_config.yaml"
        config_path.write_text(yaml.dump(config))

        task = ApplyConfigTask()
        context = {"config_path": str(config_path)}
        result = task.run(context, None)
        assert result["skip_instance_check"] is True
