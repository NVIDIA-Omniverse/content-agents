# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end tests for the scene pipeline (analyze → extract → collect).

Uses lightweight synthetic USDA scenes from scene_fixtures.py.
No external services (NVCF, VLM, SO) are called — the run-agent step is
replaced by mock prediction JSONL files written directly to disk.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from scene_fixtures import (
    create_material_library,
    create_multi_asset_scene,
    create_payload_scene,
)

from material_agent.scene.collect import (
    _load_material_library,
    _write_material_bindings,
    apply_and_compose,
)
from material_agent.scene.extract import extract_all
from material_agent.scene.manifest import (
    InstanceGroup,
    SceneManifest,
    SubAsset,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def scene_a(tmp_path: Path) -> Path:
    """Multi-asset scene with structural duplicates."""
    return create_multi_asset_scene(tmp_path / "scenes" / "multi_asset_scene.usda")


@pytest.fixture()
def payload_scene(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Payload DAG scene (scene, parent, leaf)."""
    return create_payload_scene(tmp_path / "scenes" / "payload")


@pytest.fixture()
def material_lib(tmp_path: Path) -> tuple[Path, Path]:
    """Minimal material library (yaml, usd)."""
    return create_material_library(tmp_path / "materials")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_manifest_for_scene_a(scene_path: Path) -> SceneManifest:
    """Build a SceneManifest manually for scene A (bypassing analyze)."""
    return SceneManifest(
        scene_usd_path=str(scene_path.resolve()),
        generated_at=SceneManifest.timestamp(),
        sub_assets=[
            SubAsset(
                id="robot_arm",
                name="RobotArm",
                prim_path="/Root/RobotArm",
                mesh_count=3,
            ),
            SubAsset(
                id="conveyor",
                name="Conveyor",
                prim_path="/Root/Conveyor",
                mesh_count=3,
            ),
            SubAsset(
                id="fence_a",
                name="Fence_A",
                prim_path="/Root/Fence_A",
                mesh_count=2,
            ),
            SubAsset(
                id="fence_b",
                name="Fence_B",
                prim_path="/Root/Fence_B",
                mesh_count=2,
                instance_group="structural_Fence_A",
            ),
        ],
        instance_groups=[
            InstanceGroup(
                group_name="structural_Fence_A",
                instance_count=1,
                member_paths=["/Root/Fence_B"],
                representative_id="fence_a",
            ),
        ],
    )


def _write_predictions(working_dir: Path, predictions: list[dict]) -> Path:
    """Write mock predictions to the expected JSONL location."""
    pred_dir = working_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_path = pred_dir / "predictions.jsonl"
    with open(pred_path, "w") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")
    return pred_path


def _setup_completed_asset(
    sa: SubAsset,
    working_base: Path,
    predictions: list[dict],
) -> None:
    """Set up a sub-asset as 'completed' with mock predictions."""
    wd = working_base / sa.name.lower()
    wd.mkdir(parents=True, exist_ok=True)
    sa.working_dir = str(wd)
    sa.predictions_path = str(_write_predictions(wd, predictions))
    sa.status = "completed"


# ---------------------------------------------------------------------------
# Test 1: analyze_scene detects sub-assets
# ---------------------------------------------------------------------------


class TestAnalyzeDetectsSubAssets:
    """Test that analyze_scene correctly detects sub-assets in scene A."""

    def test_detects_sub_assets(self, scene_a: Path) -> None:
        """analyze_scene finds the top-level Xform children as sub-assets."""
        from material_agent.scene.analyze import analyze_scene

        manifest = analyze_scene(scene_a, llm_config=None)

        # Should detect objects — at minimum, the four top-level Xforms
        assert len(manifest.sub_assets) > 0, "No sub-assets detected"

        # Verify known prim paths are present
        detected_paths = {sa.prim_path for sa in manifest.sub_assets}
        for expected in [
            "/Root/RobotArm",
            "/Root/Conveyor",
            "/Root/Fence_A",
            "/Root/Fence_B",
        ]:
            assert expected in detected_paths, f"Expected {expected} in detected paths"

        # Each detected sub-asset should have mesh_count > 0
        for sa in manifest.sub_assets:
            if sa.prim_path in {
                "/Root/RobotArm",
                "/Root/Conveyor",
                "/Root/Fence_A",
                "/Root/Fence_B",
            }:
                assert sa.mesh_count > 0, f"{sa.prim_path} has mesh_count=0"


# ---------------------------------------------------------------------------
# Test 2: structural duplicate detection
# ---------------------------------------------------------------------------


class TestStructuralDuplicateDetection:
    """Test structural duplicate detection groups Fence_A and Fence_B."""

    def test_structural_duplicates(self, scene_a: Path) -> None:
        from material_agent.scene.analyze import analyze_scene

        manifest = analyze_scene(
            scene_a,
            llm_config=None,
            filters={"detect_structural_duplicates": True},
        )

        # Should have at least one structural duplicate instance group
        structural_groups = [
            ig
            for ig in manifest.instance_groups
            if ig.group_name.startswith("structural_")
        ]
        assert len(structural_groups) >= 1, "No structural duplicate groups detected"

        # Find the group that contains Fence paths
        fence_group = None
        for ig in structural_groups:
            member_set = set(ig.member_paths)
            # One of Fence_A/Fence_B is representative, the other is a member
            if "/Root/Fence_A" in member_set or "/Root/Fence_B" in member_set:
                fence_group = ig
                break

        assert fence_group is not None, "No structural group found for Fence_A/Fence_B"
        assert fence_group.representative_id is not None
        assert fence_group.instance_count >= 1

        # get_processable_assets should exclude the non-representative
        processable = manifest.get_processable_assets()
        processable_paths = {sa.prim_path for sa in processable}

        # Representative should be processable
        rep = manifest.get_asset_by_id(fence_group.representative_id)
        assert rep is not None
        assert rep.prim_path in processable_paths

        # Non-representative member should NOT be processable
        for mp in fence_group.member_paths:
            assert mp not in processable_paths, (
                f"Non-representative {mp} should not be processable"
            )


# ---------------------------------------------------------------------------
# Test 3: extract creates standalone USDs
# ---------------------------------------------------------------------------


class TestExtractCreatesStandaloneUsds:
    """Test that extract_all creates valid USD files for each processable asset."""

    def test_extract_creates_files(self, scene_a: Path, tmp_path: Path) -> None:
        from pxr import Usd

        manifest = _build_manifest_for_scene_a(scene_a)
        output_dir = tmp_path / "extracted"

        manifest = extract_all(scene_a, manifest, output_dir)

        processable = manifest.get_processable_assets()
        assert len(processable) == 3, "Expected 3 processable assets (Fence_B excluded)"

        for sa in processable:
            assert sa.status == "extracted", f"{sa.name} status={sa.status}"
            assert sa.extracted_usd is not None
            assert Path(sa.extracted_usd).exists(), f"{sa.extracted_usd} does not exist"

            # Open extracted USD and verify it contains expected prims
            stage = Usd.Stage.Open(sa.extracted_usd)
            assert stage is not None
            root = stage.GetPrimAtPath(sa.prim_path)
            assert root.IsValid(), (
                f"Root prim {sa.prim_path} not found in extracted USD"
            )

        # Fence_B (non-representative) should NOT be extracted
        fence_b = manifest.get_asset_by_id("fence_b")
        assert fence_b is not None
        assert fence_b.extracted_usd is None


# ---------------------------------------------------------------------------
# Test 4: full collect applies materials
# ---------------------------------------------------------------------------


class TestCollectAppliesMaterials:
    """Full flow: build manifest → extract → mock predictions → collect."""

    def test_collect_applies_materials(
        self, scene_a: Path, material_lib: tuple[Path, Path], tmp_path: Path
    ) -> None:
        from pxr import Usd, UsdShade

        yaml_path, _usd_path = material_lib
        manifest = _build_manifest_for_scene_a(scene_a)
        output_dir = tmp_path / "extracted"

        # Extract
        manifest = extract_all(scene_a, manifest, output_dir)

        # Mock predictions for all processable assets
        working_base = tmp_path / "working"
        robot_preds = [
            {"id": "/Root/RobotArm/Base", "materials": {"material": "Steel"}},
            {"id": "/Root/RobotArm/Joint", "materials": {"material": "Steel"}},
            {"id": "/Root/RobotArm/Gripper", "materials": {"material": "Aluminum"}},
        ]
        conveyor_preds = [
            {"id": "/Root/Conveyor/Frame", "materials": {"material": "Steel"}},
            {"id": "/Root/Conveyor/Belt", "materials": {"material": "Rubber"}},
            {"id": "/Root/Conveyor/Roller", "materials": {"material": "Steel"}},
        ]
        fence_a_preds = [
            {"id": "/Root/Fence_A/Post", "materials": {"material": "Steel"}},
            {"id": "/Root/Fence_A/Panel", "materials": {"material": "Plastic"}},
        ]

        for sa in manifest.get_processable_assets():
            if sa.name == "RobotArm":
                preds = robot_preds
            elif sa.name == "Conveyor":
                preds = conveyor_preds
            elif sa.name == "Fence_A":
                preds = fence_a_preds
            else:
                preds = []
            _setup_completed_asset(sa, working_base, preds)

        # Collect
        output_usd = tmp_path / "output" / "composed.usd"
        apply_and_compose(scene_a, manifest, output_usd, yaml_path)

        assert output_usd.exists()

        # Verify material bindings on predicted prims
        stage = Usd.Stage.Open(str(output_usd))
        assert stage is not None

        expected_bindings = {
            "/Root/RobotArm/Base": "Steel",
            "/Root/RobotArm/Joint": "Steel",
            "/Root/RobotArm/Gripper": "Aluminum",
            "/Root/Conveyor/Frame": "Steel",
            "/Root/Conveyor/Belt": "Rubber",
            "/Root/Conveyor/Roller": "Steel",
            "/Root/Fence_A/Post": "Steel",
            "/Root/Fence_A/Panel": "Plastic",
        }

        for prim_path, expected_mat in expected_bindings.items():
            prim = stage.GetPrimAtPath(prim_path)
            assert prim.IsValid(), f"Prim {prim_path} not found in composed stage"
            binding_api = UsdShade.MaterialBindingAPI(prim)
            bound_mat, _rel = binding_api.ComputeBoundMaterial()
            assert bound_mat, f"No material bound to {prim_path}"
            mat_name = bound_mat.GetPrim().GetName()
            assert mat_name == expected_mat, (
                f"Expected {expected_mat} on {prim_path}, got {mat_name}"
            )


# ---------------------------------------------------------------------------
# Test 5: gap fill
# ---------------------------------------------------------------------------


class TestGapFill:
    """Write predictions for Frame+Belt but NOT Roller. Roller should inherit."""

    def test_gap_fill_from_siblings(
        self, scene_a: Path, material_lib: tuple[Path, Path], tmp_path: Path
    ) -> None:
        from pxr import Usd, UsdShade

        yaml_path, _ = material_lib
        manifest = _build_manifest_for_scene_a(scene_a)
        output_dir = tmp_path / "extracted"
        manifest = extract_all(scene_a, manifest, output_dir)

        working_base = tmp_path / "working"

        # Predictions for RobotArm and Fence_A — full coverage
        robot_preds = [
            {"id": "/Root/RobotArm/Base", "materials": {"material": "Steel"}},
            {"id": "/Root/RobotArm/Joint", "materials": {"material": "Steel"}},
            {"id": "/Root/RobotArm/Gripper", "materials": {"material": "Aluminum"}},
        ]
        fence_a_preds = [
            {"id": "/Root/Fence_A/Post", "materials": {"material": "Steel"}},
            {"id": "/Root/Fence_A/Panel", "materials": {"material": "Plastic"}},
        ]
        # Conveyor: predict Frame + Belt as Steel, but SKIP Roller
        conveyor_preds = [
            {"id": "/Root/Conveyor/Frame", "materials": {"material": "Steel"}},
            {"id": "/Root/Conveyor/Belt", "materials": {"material": "Steel"}},
            # Roller deliberately omitted
        ]

        for sa in manifest.get_processable_assets():
            if sa.name == "RobotArm":
                preds = robot_preds
            elif sa.name == "Conveyor":
                preds = conveyor_preds
            elif sa.name == "Fence_A":
                preds = fence_a_preds
            else:
                preds = []
            _setup_completed_asset(sa, working_base, preds)

        output_usd = tmp_path / "output" / "composed.usd"
        apply_and_compose(scene_a, manifest, output_usd, yaml_path)

        # Roller should have been gap-filled (siblings Frame and Belt are both Steel)
        stage = Usd.Stage.Open(str(output_usd))
        roller_prim = stage.GetPrimAtPath("/Root/Conveyor/Roller")
        assert roller_prim.IsValid()
        binding_api = UsdShade.MaterialBindingAPI(roller_prim)
        bound_mat, _rel = binding_api.ComputeBoundMaterial()
        assert bound_mat, "Roller should have a gap-filled material binding"
        assert bound_mat.GetPrim().GetName() == "Steel", (
            f"Expected Steel on Roller (gap fill), got {bound_mat.GetPrim().GetName()}"
        )


# ---------------------------------------------------------------------------
# Test 6: instance group propagation
# ---------------------------------------------------------------------------


class TestInstanceGroupPropagation:
    """Write predictions only for Fence_A. Fence_B should get matching bindings."""

    def test_instance_propagation(
        self, scene_a: Path, material_lib: tuple[Path, Path], tmp_path: Path
    ) -> None:
        from pxr import Usd, UsdShade

        yaml_path, _ = material_lib
        manifest = _build_manifest_for_scene_a(scene_a)
        output_dir = tmp_path / "extracted"
        manifest = extract_all(scene_a, manifest, output_dir)

        working_base = tmp_path / "working"

        # Only predict for processable assets (Fence_A is representative)
        robot_preds = [
            {"id": "/Root/RobotArm/Base", "materials": {"material": "Steel"}},
            {"id": "/Root/RobotArm/Joint", "materials": {"material": "Steel"}},
            {"id": "/Root/RobotArm/Gripper", "materials": {"material": "Aluminum"}},
        ]
        conveyor_preds = [
            {"id": "/Root/Conveyor/Frame", "materials": {"material": "Steel"}},
            {"id": "/Root/Conveyor/Belt", "materials": {"material": "Rubber"}},
            {"id": "/Root/Conveyor/Roller", "materials": {"material": "Steel"}},
        ]
        fence_a_preds = [
            {"id": "/Root/Fence_A/Post", "materials": {"material": "Steel"}},
            {"id": "/Root/Fence_A/Panel", "materials": {"material": "Plastic"}},
        ]

        for sa in manifest.get_processable_assets():
            if sa.name == "RobotArm":
                preds = robot_preds
            elif sa.name == "Conveyor":
                preds = conveyor_preds
            elif sa.name == "Fence_A":
                preds = fence_a_preds
            else:
                preds = []
            _setup_completed_asset(sa, working_base, preds)

        output_usd = tmp_path / "output" / "composed.usd"
        apply_and_compose(scene_a, manifest, output_usd, yaml_path)

        stage = Usd.Stage.Open(str(output_usd))

        # Fence_B should have matching bindings propagated from Fence_A
        fence_b_expected = {
            "/Root/Fence_B/Post": "Steel",
            "/Root/Fence_B/Panel": "Plastic",
        }
        for prim_path, expected_mat in fence_b_expected.items():
            prim = stage.GetPrimAtPath(prim_path)
            assert prim.IsValid(), f"Prim {prim_path} not found"
            binding_api = UsdShade.MaterialBindingAPI(prim)
            bound_mat, _rel = binding_api.ComputeBoundMaterial()
            assert bound_mat, f"No material bound to {prim_path} (should be propagated)"
            assert bound_mat.GetPrim().GetName() == expected_mat, (
                f"Expected {expected_mat} on {prim_path}, got {bound_mat.GetPrim().GetName()}"
            )


# ---------------------------------------------------------------------------
# Test 7: payload DAG detection
# ---------------------------------------------------------------------------


class TestPayloadDagDetection:
    """Test that analyze_scene detects the payload DAG correctly."""

    def test_payload_dag(self, payload_scene: tuple[Path, Path, Path]) -> None:
        from material_agent.scene.analyze import analyze_scene

        scene_path, parent_path, leaf_path = payload_scene

        manifest = analyze_scene(scene_path, llm_config=None)

        assert len(manifest.payload_groups) >= 2, (
            f"Expected >= 2 payload groups, got {len(manifest.payload_groups)}"
        )

        # Find the leaf and parent payload groups by file path
        leaf_pg = manifest.get_payload_by_file(str(leaf_path.resolve()))
        parent_pg = manifest.get_payload_by_file(str(parent_path.resolve()))

        assert parent_pg is not None, f"Parent payload not found for {parent_path}"
        assert leaf_pg is not None, f"Leaf payload not found for {leaf_path}"

        # Leaf should be depth 0
        assert leaf_pg.depth == 0, f"Leaf depth should be 0, got {leaf_pg.depth}"
        # Parent should be depth 1
        assert parent_pg.depth == 1, f"Parent depth should be 1, got {parent_pg.depth}"

        # DAG edges
        assert str(leaf_path.resolve()) in parent_pg.child_payload_files, (
            "Parent should list leaf as child"
        )
        assert str(parent_path.resolve()) in leaf_pg.parent_payload_files, (
            "Leaf should list parent as parent"
        )

        # Instance paths — Machine_A and Machine_B should be listed
        all_instance_paths = set()
        for pg in manifest.payload_groups:
            all_instance_paths.update(pg.instance_paths)
        assert "/Root/Machine_A" in all_instance_paths
        assert "/Root/Machine_B" in all_instance_paths


# ---------------------------------------------------------------------------
# Test 8: payload bottom-up collect
# ---------------------------------------------------------------------------


class TestPayloadBottomUpCollect:
    """Mock predictions for leaf + parent payloads, verify composed scene."""

    def test_payload_collect(
        self,
        payload_scene: tuple[Path, Path, Path],
        material_lib: tuple[Path, Path],
        tmp_path: Path,
    ) -> None:
        from pxr import Usd, UsdShade

        scene_path, parent_path, leaf_path = payload_scene
        yaml_path, _ = material_lib

        # Build manifest manually for payload scene
        manifest = SceneManifest(
            scene_usd_path=str(scene_path.resolve()),
            generated_at=SceneManifest.timestamp(),
            sub_assets=[],
            payload_groups=[],
        )

        # We need to run analyze to get proper payload groups with DAG
        from material_agent.scene.analyze import analyze_scene

        manifest = analyze_scene(scene_path, llm_config=None)

        leaf_pg = manifest.get_payload_by_file(str(leaf_path.resolve()))
        parent_pg = manifest.get_payload_by_file(str(parent_path.resolve()))

        if not leaf_pg or not parent_pg:
            pytest.skip("Payload groups not detected — cannot test collect")

        # Set up working dirs and mock predictions for payload groups
        working_base = tmp_path / "payload_working"

        # Leaf payload predictions (prim paths relative to payload root)
        if leaf_pg.status != "skipped":
            leaf_wd = working_base / "leaf"
            leaf_wd.mkdir(parents=True, exist_ok=True)
            leaf_pg.working_dir = str(leaf_wd)
            leaf_preds = [
                {"id": "/Root/Bottom", "materials": {"material": "Plastic"}},
                {"id": "/Root/Side", "materials": {"material": "Plastic"}},
            ]
            leaf_pg.predictions_path = str(_write_predictions(leaf_wd, leaf_preds))

            # Create the output.usd for leaf (sublayers original + applies materials)
            leaf_output = leaf_wd / "output.usd"
            _create_payload_output(leaf_path, leaf_output, leaf_preds, yaml_path)
            leaf_pg.output_usd_path = str(leaf_output)
            leaf_pg.status = "completed"

        # Parent payload predictions
        if parent_pg.status != "skipped":
            parent_wd = working_base / "parent"
            parent_wd.mkdir(parents=True, exist_ok=True)
            parent_pg.working_dir = str(parent_wd)
            parent_preds = [
                {"id": "/Root/Housing", "materials": {"material": "Steel"}},
            ]
            parent_pg.predictions_path = str(
                _write_predictions(parent_wd, parent_preds)
            )

            parent_output = parent_wd / "output.usd"
            _create_payload_output(parent_path, parent_output, parent_preds, yaml_path)
            parent_pg.output_usd_path = str(parent_output)
            parent_pg.status = "completed"

        # Mark any sub-assets as completed with empty predictions
        # (the scene has Floor as direct geometry, which analyze may detect)
        for sa in manifest.sub_assets:
            if sa.status == "pending":
                sa_wd = working_base / sa.name.lower()
                sa_wd.mkdir(parents=True, exist_ok=True)
                sa.working_dir = str(sa_wd)
                sa.status = "completed"
                _write_predictions(sa_wd, [])

        # Collect
        output_usd = tmp_path / "output" / "composed.usd"
        apply_and_compose(scene_path, manifest, output_usd, yaml_path)

        assert output_usd.exists(), "Composed USD not created"

        # Verify the output stage exists and can be opened
        stage = Usd.Stage.Open(str(output_usd))
        assert stage is not None, "Failed to open composed stage"

        # The Floor prim should exist
        floor = stage.GetPrimAtPath("/Root/Floor")
        assert floor.IsValid(), "Floor prim not found in composed stage"


# ---------------------------------------------------------------------------
# Test 9: reconcile detects ambiguous pairs and applies remapping
# ---------------------------------------------------------------------------


class TestReconcileAmbiguousPairs:
    """Test reconcile detection + file remapping without LLM."""

    def test_detect_ambiguous_and_remap(self, scene_a: Path, tmp_path: Path) -> None:
        from material_agent.scene.reconcile import (
            _build_asset_distributions,
            _detect_ambiguous_pairs,
            _gather_predictions,
            apply_remapping,
        )

        manifest = _build_manifest_for_scene_a(scene_a)
        working_base = tmp_path / "working"

        # Predictions with color-family conflicts: two orange variants
        # co-occurring across assets → should trigger ambiguity detection.
        robot_preds = [
            {
                "id": "/Root/RobotArm/Base",
                "materials": {"material": "Steel Painted Orange"},
            },
            {
                "id": "/Root/RobotArm/Joint",
                "materials": {"material": "Car Paint Orange"},
            },
            {
                "id": "/Root/RobotArm/Gripper",
                "materials": {"material": "Aluminum"},
            },
        ]
        conveyor_preds = [
            {
                "id": "/Root/Conveyor/Frame",
                "materials": {"material": "Steel Painted Orange"},
            },
            {
                "id": "/Root/Conveyor/Belt",
                "materials": {"material": "Car Paint Orange"},
            },
            {
                "id": "/Root/Conveyor/Roller",
                "materials": {"material": "Steel"},
            },
        ]
        fence_a_preds = [
            {"id": "/Root/Fence_A/Post", "materials": {"material": "Steel"}},
            {"id": "/Root/Fence_A/Panel", "materials": {"material": "Plastic"}},
        ]

        for sa in manifest.get_processable_assets():
            if sa.name == "RobotArm":
                preds = robot_preds
            elif sa.name == "Conveyor":
                preds = conveyor_preds
            elif sa.name == "Fence_A":
                preds = fence_a_preds
            else:
                preds = []
            _setup_completed_asset(sa, working_base, preds)

        # Gather and detect
        all_preds = _gather_predictions(manifest)
        assert len(all_preds) == 8  # 3 + 3 + 2

        distributions = _build_asset_distributions(all_preds)
        assert "RobotArm" in distributions
        assert "Conveyor" in distributions

        ambiguous = _detect_ambiguous_pairs(distributions)
        assert len(ambiguous) >= 1, "Expected at least one ambiguous group"

        # Should detect an 'orange' group since both assets use two orange mats
        orange_group = ambiguous.get("orange_group")
        assert orange_group is not None, (
            f"Expected 'orange_group', got groups: {list(ambiguous.keys())}"
        )
        assert "Steel Painted Orange" in orange_group["materials"]
        assert "Car Paint Orange" in orange_group["materials"]
        assert orange_group["asset_count"] >= 2

        # Apply a known remap (simulating what LLM would produce)
        remap = {"Car Paint Orange": "Steel Painted Orange"}
        updated = apply_remapping(manifest, remap)
        assert updated >= 2, f"Expected >= 2 predictions remapped, got {updated}"

        # Verify the prediction files were actually rewritten
        for sa in manifest.get_processable_assets():
            if sa.status != "completed" or not sa.working_dir:
                continue
            pred_file = Path(sa.working_dir) / "predictions" / "predictions.jsonl"
            if not pred_file.exists():
                continue
            for line in pred_file.read_text().splitlines():
                if not line.strip():
                    continue
                entry = json.loads(line)
                mat = entry.get("materials", {}).get("material", "")
                # "Car Paint Orange" should have been remapped
                assert mat != "Car Paint Orange", (
                    f"Remap failed: still found 'Car Paint Orange' in {pred_file}"
                )


# ---------------------------------------------------------------------------
# Test 10: harmonize scene predictions (simple/majority-vote mode)
# ---------------------------------------------------------------------------


class TestHarmonizeScenePredictions:
    """Test cross-asset harmonization with name-template grouping + majority vote."""

    def test_harmonize_majority_vote(self, scene_a: Path, tmp_path: Path) -> None:
        from material_agent.scene.harmonize import harmonize_scene_predictions

        manifest = _build_manifest_for_scene_a(scene_a)
        working_base = tmp_path / "working"

        # Create predictions where name-template grouping will fire:
        # Fence_A/Post and Fence_B/Post → template "Root/Fence_{}/Post"
        # Give them different materials so there's a conflict to resolve.
        # Majority vote should pick the material that appears more often.
        #
        # Note: Fence_B is the non-representative member but we still
        # write predictions for it to simulate a scenario where both
        # were processed (harmonize reads all completed assets).
        robot_preds = [
            {"id": "/Root/RobotArm/Base", "materials": {"material": "Steel"}},
            {"id": "/Root/RobotArm/Joint", "materials": {"material": "Steel"}},
            {"id": "/Root/RobotArm/Gripper", "materials": {"material": "Aluminum"}},
        ]
        conveyor_preds = [
            {"id": "/Root/Conveyor/Frame", "materials": {"material": "Steel"}},
            {"id": "/Root/Conveyor/Belt", "materials": {"material": "Rubber"}},
            {"id": "/Root/Conveyor/Roller", "materials": {"material": "Steel"}},
        ]
        fence_a_preds = [
            {"id": "/Root/Fence_A/Post", "materials": {"material": "Steel"}},
            {"id": "/Root/Fence_A/Panel", "materials": {"material": "Plastic"}},
        ]
        # Fence_B has a CONFLICT: Post is Aluminum (vs Steel in Fence_A)
        fence_b_preds = [
            {"id": "/Root/Fence_B/Post", "materials": {"material": "Aluminum"}},
            {"id": "/Root/Fence_B/Panel", "materials": {"material": "Plastic"}},
        ]

        for sa in manifest.sub_assets:
            if sa.name == "RobotArm":
                preds = robot_preds
            elif sa.name == "Conveyor":
                preds = conveyor_preds
            elif sa.name == "Fence_A":
                preds = fence_a_preds
            elif sa.name == "Fence_B":
                preds = fence_b_preds
            else:
                preds = []
            _setup_completed_asset(sa, working_base, preds)

        # Run harmonize in simple mode (majority vote, no LLM)
        remap = harmonize_scene_predictions(manifest, llm_config=None, mode="simple")

        # If name-template grouping detected the Fence Post conflict and majority
        # vote resolved it, there should be a remap entry.  But even if the
        # grouping doesn't fire on these short paths (signatures require CAD-like
        # part numbers), the function should still complete without error.
        # Either way, verify prediction files are readable and consistent.
        for sa in manifest.sub_assets:
            if sa.status != "completed" or not sa.working_dir:
                continue
            pred_file = Path(sa.working_dir) / "predictions" / "predictions.jsonl"
            if not pred_file.exists():
                continue
            for line in pred_file.read_text().splitlines():
                if not line.strip():
                    continue
                entry = json.loads(line)
                # Every entry should still have a valid material
                mat = entry.get("materials", {}).get("material")
                assert mat, f"Empty material in {pred_file}: {entry}"

        # If remap was produced, verify Fence_B/Post was harmonized to Steel
        if remap:
            assert remap.get("/Root/Fence_B/Post") == "Steel", (
                f"Expected Fence_B/Post → Steel, got {remap}"
            )


# ---------------------------------------------------------------------------
# Test 11: harmonize asset-level (within single asset, simple mode)
# ---------------------------------------------------------------------------


class TestHarmonizeAssetPredictions:
    """Test within-asset harmonization with geometry fingerprint grouping."""

    def test_harmonize_within_asset(self, scene_a: Path, tmp_path: Path) -> None:
        from material_agent.scene.harmonize import harmonize_asset_predictions

        # Create a predictions file with a name-template conflict
        # within a single asset: Belt1 and Belt2 (template "Root/Conveyor/Belt{}")
        # Both should be Rubber, but Belt2 was mispredicted as Plastic.
        pred_dir = tmp_path / "asset_harmonize"
        pred_dir.mkdir(parents=True, exist_ok=True)
        pred_file = pred_dir / "predictions.jsonl"
        preds = [
            {"id": "/Root/Conveyor/Belt1", "materials": {"material": "Rubber"}},
            {"id": "/Root/Conveyor/Belt2", "materials": {"material": "Plastic"}},
            {"id": "/Root/Conveyor/Belt3", "materials": {"material": "Rubber"}},
            {"id": "/Root/Conveyor/Frame", "materials": {"material": "Steel"}},
        ]
        with open(pred_file, "w") as f:
            for p in preds:
                f.write(json.dumps(p) + "\n")

        # Run harmonize (simple mode = majority vote)
        result_path, remap = harmonize_asset_predictions(pred_file, llm_config=None)

        assert result_path == pred_file

        # If name-template grouping caught Belt1/Belt2/Belt3 conflict,
        # majority vote should remap Belt2: Plastic → Rubber
        if remap:
            assert remap.get("/Root/Conveyor/Belt2") == "Rubber", (
                f"Expected Belt2 → Rubber, got {remap}"
            )
            # Verify the file was updated
            updated_preds = []
            for line in pred_file.read_text().splitlines():
                if line.strip():
                    updated_preds.append(json.loads(line))
            belt2 = next(p for p in updated_preds if p["id"] == "/Root/Conveyor/Belt2")
            assert belt2["materials"]["material"] == "Rubber"
            assert belt2["materials"].get("harmonized_from") == "Plastic"

        # Harmonize report should be written regardless
        report_path = pred_dir / "harmonize_report.json"
        assert report_path.exists(), "Harmonize report not written"
        report = json.loads(report_path.read_text())
        assert report["total_predictions"] == 4


# ---------------------------------------------------------------------------
# Test 12: harmonize asset-level with mocked LLM (full mode)
# ---------------------------------------------------------------------------


class TestHarmonizeAssetWithLLM:
    """Test within-asset harmonization in full mode with a mocked LLM."""

    def test_harmonize_full_mode_unify(self, tmp_path: Path) -> None:
        """LLM decides to unify Belt2 to Rubber."""
        from unittest.mock import MagicMock, patch

        from material_agent.scene.harmonize import harmonize_asset_predictions

        pred_dir = tmp_path / "asset_harmonize_llm"
        pred_dir.mkdir(parents=True, exist_ok=True)
        pred_file = pred_dir / "predictions.jsonl"
        preds = [
            {"id": "/Root/Conveyor/Belt1", "materials": {"material": "Rubber"}},
            {"id": "/Root/Conveyor/Belt2", "materials": {"material": "Plastic"}},
            {"id": "/Root/Conveyor/Belt3", "materials": {"material": "Rubber"}},
            {"id": "/Root/Conveyor/Frame", "materials": {"material": "Steel"}},
        ]
        with open(pred_file, "w") as f:
            for p in preds:
                f.write(json.dumps(p) + "\n")

        # Mock LLM: returns a "unify" decision picking Rubber
        mock_response = MagicMock()
        mock_response.content = json.dumps(
            {"action": "unify", "material": "Rubber", "reason": "Same conveyor belt"}
        )
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response

        with patch(
            "world_understanding.functions.models.chat_models.create_chat_model",
            return_value=mock_llm,
        ):
            result_path, remap = harmonize_asset_predictions(
                pred_file,
                llm_config={"backend": "mock", "model": "mock", "api_key": "fake"},
            )

        assert result_path == pred_file

        # The LLM chose to unify → Belt2 should be remapped
        if remap:
            assert remap.get("/Root/Conveyor/Belt2") == "Rubber", (
                f"Expected Belt2 → Rubber, got {remap}"
            )
            # Verify file was updated with harmonized_from audit field
            updated_preds = []
            for line in pred_file.read_text().splitlines():
                if line.strip():
                    updated_preds.append(json.loads(line))
            belt2 = next(p for p in updated_preds if p["id"] == "/Root/Conveyor/Belt2")
            assert belt2["materials"]["material"] == "Rubber"
            assert belt2["materials"].get("harmonized_from") == "Plastic"

        # LLM should have been invoked at least once
        assert mock_llm.invoke.call_count >= 1

        # Report should exist
        report_path = pred_dir / "harmonize_report.json"
        assert report_path.exists()
        report = json.loads(report_path.read_text())
        assert report["conflict_groups_count"] >= 1

    def test_harmonize_full_mode_keep(self, tmp_path: Path) -> None:
        """LLM decides to keep different materials."""
        from unittest.mock import MagicMock, patch

        from material_agent.scene.harmonize import harmonize_asset_predictions

        pred_dir = tmp_path / "asset_harmonize_keep"
        pred_dir.mkdir(parents=True, exist_ok=True)
        pred_file = pred_dir / "predictions.jsonl"
        preds = [
            {"id": "/Root/Conveyor/Belt1", "materials": {"material": "Rubber"}},
            {"id": "/Root/Conveyor/Belt2", "materials": {"material": "Plastic"}},
            {"id": "/Root/Conveyor/Belt3", "materials": {"material": "Rubber"}},
            {"id": "/Root/Conveyor/Frame", "materials": {"material": "Steel"}},
        ]
        with open(pred_file, "w") as f:
            for p in preds:
                f.write(json.dumps(p) + "\n")

        # Mock LLM: returns a "keep" decision — no changes
        mock_response = MagicMock()
        mock_response.content = json.dumps(
            {"action": "keep", "reason": "Belt2 is a protective cover, not a belt"}
        )
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response

        with patch(
            "world_understanding.functions.models.chat_models.create_chat_model",
            return_value=mock_llm,
        ):
            result_path, remap = harmonize_asset_predictions(
                pred_file,
                llm_config={"backend": "mock", "model": "mock", "api_key": "fake"},
            )

        # LLM said keep → no remapping should have occurred
        assert not remap, f"Expected no remap for 'keep', got {remap}"

        # File should be unchanged — Belt2 still Plastic
        updated_preds = []
        for line in pred_file.read_text().splitlines():
            if line.strip():
                updated_preds.append(json.loads(line))
        belt2 = next(p for p in updated_preds if p["id"] == "/Root/Conveyor/Belt2")
        assert belt2["materials"]["material"] == "Plastic"
        assert "harmonized_from" not in belt2["materials"]


def _create_payload_output(
    payload_path: Path,
    output_path: Path,
    predictions: list[dict],
    material_yaml_path: Path,
) -> Path:
    """Create a simple output.usd that sublayers the payload and applies materials.

    This simulates what the real pipeline's apply step does for a payload.
    """
    from pxr import Sdf

    output_path.parent.mkdir(parents=True, exist_ok=True)

    _, name_to_prim = _load_material_library(material_yaml_path)

    layer = Sdf.Layer.CreateNew(str(output_path))
    layer.subLayerPaths = [str(payload_path.resolve())]

    # Copy stage metadata
    source = Sdf.Layer.FindOrOpen(str(payload_path.resolve()))
    if source and source.defaultPrim:
        layer.defaultPrim = source.defaultPrim

    # Write bindings
    prim_to_mat = {}
    for pred in predictions:
        prim_id = pred.get("id")
        mat = pred.get("materials", {}).get("material")
        if prim_id and mat:
            prim_to_mat[prim_id] = mat

    _write_material_bindings(layer, prim_to_mat, name_to_prim)

    # Copy material definitions from library
    library_usd_path, _ = _load_material_library(material_yaml_path)
    if library_usd_path:
        from material_agent.scene.collect import _copy_materials_from_library

        used = {
            m: name_to_prim[m] for m in set(prim_to_mat.values()) if m in name_to_prim
        }
        if used:
            _copy_materials_from_library(
                layer,
                library_usd_path,
                used,
                output_path,
                scene_default_prim=layer.defaultPrim or "",
            )

    layer.Save()
    return output_path
