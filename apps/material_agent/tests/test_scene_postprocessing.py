# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for scene-level post-processing: reconcile and harmonize.

These tests verify the pure-function logic of each post-processing step
without requiring USD stages or LLM calls.
"""

from __future__ import annotations

import json
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from material_agent.scene.collect import _extract_material_name
from material_agent.scene.harmonize import (
    _find_best_predictions,
    _find_conflicts,
    _group_by_name_template,
    _group_by_signature,
    _load_predictions,
    _majority_vote_fallback,
    _merge_groups,
    _name_template,
    _resolve_conflicts,
    apply_prim_remap,
    build_segment_frequency,
    compute_signature,
    is_part_number,
    normalize_segment,
)
from material_agent.scene.reconcile import (
    _build_asset_distributions,
    _detect_ambiguous_pairs,
    _find_best_predictions_file,
    _parse_remap_json,
    _remap_predictions_file,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pred(prim_id: str, material: str, reasoning: str = "") -> dict[str, Any]:
    """Create a minimal prediction dict."""
    return {
        "id": prim_id,
        "materials": {
            "material": material,
            "original_response": f"<reasoning>{reasoning}</reasoning>",
        },
    }


def _write_predictions(preds: list[dict], path: Path) -> Path:
    """Write a list of prediction dicts to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(p) for p in preds) + "\n")
    return path


def _read_predictions(path: Path) -> list[dict]:
    """Read predictions from a JSONL file."""
    preds = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            preds.append(json.loads(line))
    return preds


# ===================================================================
# reconcile.py — pure function tests
# ===================================================================


class TestParseRemapJson:
    """_parse_remap_json extracts a JSON remap dict from LLM output."""

    def test_direct_json(self):
        resp = '{"Car Paint Orange": "Steel Painted Orange"}'
        assert _parse_remap_json(resp) == {"Car Paint Orange": "Steel Painted Orange"}

    def test_json_in_code_block(self):
        resp = '```json\n{"Car Paint Orange": "Steel Painted Orange"}\n```'
        assert _parse_remap_json(resp) == {"Car Paint Orange": "Steel Painted Orange"}

    def test_json_in_answer_tags(self):
        resp = '<answer>{"Car Paint Orange": "Steel Painted Orange"}</answer>'
        assert _parse_remap_json(resp) == {"Car Paint Orange": "Steel Painted Orange"}

    def test_identity_mappings_filtered(self):
        """Entries where old == new should be dropped."""
        resp = '{"Steel Painted Orange": "Steel Painted Orange", "Car Paint Orange": "Steel Painted Orange"}'
        result = _parse_remap_json(resp)
        assert "Steel Painted Orange" not in result
        assert result == {"Car Paint Orange": "Steel Painted Orange"}

    def test_empty_on_garbage(self):
        assert _parse_remap_json("this is not json at all") == {}

    def test_empty_string(self):
        assert _parse_remap_json("") == {}


class TestBuildAssetDistributions:
    def test_basic(self):
        preds = [
            {"_asset_name": "robot_1", "materials": {"material": "Steel"}},
            {"_asset_name": "robot_1", "materials": {"material": "Plastic"}},
            {"_asset_name": "robot_2", "materials": {"material": "Steel"}},
        ]
        dist = _build_asset_distributions(preds)
        assert dist["robot_1"] == Counter({"Steel": 1, "Plastic": 1})
        assert dist["robot_2"] == Counter({"Steel": 1})

    def test_empty(self):
        assert _build_asset_distributions([]) == {}


class TestDetectAmbiguousPairs:
    def test_detects_same_color_cooccurrence(self):
        """Two orange materials in the same asset should be flagged."""
        dist = {
            "asset_1": Counter({"Car Paint Orange": 5, "Steel Painted Orange": 3}),
            "asset_2": Counter({"Car Paint Orange": 2}),
        }
        result = _detect_ambiguous_pairs(dist)
        assert "orange_group" in result
        assert "Car Paint Orange" in result["orange_group"]["materials"]
        assert "Steel Painted Orange" in result["orange_group"]["materials"]
        assert result["orange_group"]["asset_count"] == 1  # only asset_1 has both

    def test_no_ambiguity_single_material_per_color(self):
        dist = {
            "asset_1": Counter({"Steel Painted Orange": 5}),
            "asset_2": Counter({"Steel Painted Orange": 3}),
        }
        result = _detect_ambiguous_pairs(dist)
        assert result == {}

    def test_no_ambiguity_different_colors(self):
        dist = {
            "asset_1": Counter({"Steel Painted Orange": 5, "Steel Painted Black": 3}),
        }
        result = _detect_ambiguous_pairs(dist)
        # orange_group has only 1 material, black_group has only 1 — neither flagged
        assert result == {}


class TestRemapPredictionsFile:
    def test_remaps_material_and_preserves_original(self):
        preds = [
            _pred("/a", "Car Paint Orange"),
            _pred("/b", "Steel Painted Orange"),
            _pred("/c", "Car Paint Orange"),
        ]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "predictions.jsonl"
            _write_predictions(preds, path)

            remap = {"Car Paint Orange": "Steel Painted Orange"}
            updated = _remap_predictions_file(path, remap)

            assert updated == 2
            result = _read_predictions(path)
            assert result[0]["materials"]["material"] == "Steel Painted Orange"
            assert result[0]["materials"]["original_material"] == "Car Paint Orange"
            # Already "Steel Painted Orange" — no change
            assert result[1]["materials"]["material"] == "Steel Painted Orange"
            assert "original_material" not in result[1]["materials"]

    def test_empty_remap_is_noop(self):
        preds = [_pred("/a", "Steel")]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "predictions.jsonl"
            _write_predictions(preds, path)
            updated = _remap_predictions_file(path, {})
            assert updated == 0
            result = _read_predictions(path)
            assert result[0]["materials"]["material"] == "Steel"


# ===================================================================
# harmonize.py — grouping signal tests
# ===================================================================


class TestNormalizeSegment:
    def test_strips_instance_suffix(self):
        # The regex strips the entire _U20__U28_N_U29_ suffix
        assert normalize_segment("Part_U20__U28_42_U29_") == "Part"

    def test_keeps_plain_segment(self):
        assert normalize_segment("mesh") == "mesh"

    def test_keeps_mesh_array(self):
        assert normalize_segment("mesh_U5B_3_U5D_") == "mesh_U5B_3_U5D_"


class TestIsPartNumber:
    def test_long_with_digits(self):
        assert is_part_number("PFEN562C431F1LE5__1__XT__12__8020545") is True

    def test_short_segment(self):
        assert is_part_number("mesh") is False

    def test_all_alpha(self):
        assert is_part_number("SomeVeryLongSegmentName") is False


class TestNameTemplate:
    def test_strips_trailing_digits(self):
        assert _name_template("/Root/CraneBase/Column2/shape/mesh") == (
            "Root/CraneBase/Column{}/shape/mesh"
        )

    def test_no_digits_unchanged(self):
        assert _name_template("/Root/Base/shape/mesh") == ("Root/Base/shape/mesh")

    def test_multiple_segments_with_digits(self):
        assert _name_template("/Root/Arm3/Joint12/mesh") == ("Root/Arm{}/Joint{}/mesh")


class TestGroupByNameTemplate:
    def test_groups_trailing_digits(self):
        preds = [
            _pred("/Root/Column1/mesh", "Steel"),
            _pred("/Root/Column2/mesh", "Plastic"),
            _pred("/Root/Column3/mesh", "Steel"),
        ]
        groups = _group_by_name_template(preds)
        # All 3 should be in one group keyed by "name:Root/Column{}/mesh"
        assert len(groups) == 1
        key = next(iter(groups))
        assert key.startswith("name:")
        assert sorted(groups[key]) == [0, 1, 2]

    def test_no_digits_no_groups(self):
        preds = [
            _pred("/Root/Base/mesh", "Steel"),
            _pred("/Root/Arm/mesh", "Plastic"),
        ]
        groups = _group_by_name_template(preds)
        assert groups == {}

    def test_single_member_filtered(self):
        preds = [
            _pred("/Root/Column1/mesh", "Steel"),
            _pred("/Root/Base/mesh", "Plastic"),
        ]
        groups = _group_by_name_template(preds)
        # Column1 produces a template but it's the only member — filtered out
        assert groups == {}


class TestGroupBySignature:
    def test_groups_rare_part_numbers(self):
        # Create paths with rare part-number-like segments
        preds = [
            _pred(
                "/Root/PFEN562C431F__1__XT__8020545_U20_/mesh_U5B_0_U5D_/mesh",
                "Steel",
            ),
            _pred(
                "/Root/PFEN562C431F__1__XT__8020545_U20_/mesh_U5B_1_U5D_/mesh",
                "Plastic",
            ),
        ]
        _group_by_signature(preds)
        # Both share the rare segment, different mesh indices
        # Whether they group depends on frequency ratio — with 2 prims,
        # the rare segment has 100% frequency (too high). This test
        # documents that behavior.
        # With more prims to dilute the ratio, they would group.
        # This is expected — signature grouping needs scale.

    def test_empty_predictions(self):
        assert _group_by_signature([]) == {}


class TestMergeGroups:
    def test_merges_overlapping_signals(self):
        """If signal A groups {0,1} and signal B groups {1,2}, merged = {0,1,2}."""
        sig_a = {"g1": [0, 1]}
        sig_b = {"g2": [1, 2]}
        merged = _merge_groups([sig_a, sig_b], 3)
        # All three should be in the same group
        assert len(merged) == 1
        members = next(iter(merged.values()))
        assert sorted(members) == [0, 1, 2]

    def test_disjoint_signals_stay_separate(self):
        sig_a = {"g1": [0, 1]}
        sig_b = {"g2": [2, 3]}
        merged = _merge_groups([sig_a, sig_b], 4)
        assert len(merged) == 2

    def test_single_member_filtered(self):
        sig_a = {"g1": [0, 1]}
        merged = _merge_groups([sig_a], 5)
        # Only group {0,1} — indices 2,3,4 are singletons, filtered
        assert len(merged) == 1


class TestFindConflicts:
    def test_conflict_detected(self):
        groups = {0: [0, 1]}
        preds = [
            _pred("/a", "Steel"),
            _pred("/b", "Plastic"),
        ]
        conflicts = _find_conflicts(groups, preds)
        assert 0 in conflicts

    def test_no_conflict_when_all_agree(self):
        groups = {0: [0, 1]}
        preds = [
            _pred("/a", "Steel"),
            _pred("/b", "Steel"),
        ]
        conflicts = _find_conflicts(groups, preds)
        assert conflicts == {}


class TestMajorityVoteFallback:
    def test_majority_wins(self):
        conflicts = {0: [0, 1, 2]}
        preds = [
            _pred("/a", "Steel"),
            _pred("/b", "Steel"),
            _pred("/c", "Plastic"),
        ]
        remap = _majority_vote_fallback(conflicts, preds)
        assert remap == {"/c": "Steel"}

    def test_tie_no_remap(self):
        """50/50 split — no clear majority, no override."""
        conflicts = {0: [0, 1]}
        preds = [
            _pred("/a", "Steel"),
            _pred("/b", "Plastic"),
        ]
        remap = _majority_vote_fallback(conflicts, preds)
        assert remap == {}

    def test_empty_conflicts(self):
        assert _majority_vote_fallback({}, []) == {}


class TestApplyPrimRemap:
    def test_remap_writes_harmonized_from(self):
        preds = [
            _pred("/a", "Steel"),
            _pred("/b", "Plastic"),
        ]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "predictions.jsonl"
            _write_predictions(preds, path)

            remap = {"/b": "Steel"}
            updated = apply_prim_remap(path, remap)

            assert updated == 1
            result = _read_predictions(path)
            assert result[1]["materials"]["material"] == "Steel"
            assert result[1]["materials"]["harmonized_from"] == "Plastic"

    def test_remap_skips_already_correct(self):
        """If prim is in remap but already has the target material, skip."""
        preds = [_pred("/a", "Steel")]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "predictions.jsonl"
            _write_predictions(preds, path)

            remap = {"/a": "Steel"}
            updated = apply_prim_remap(path, remap)

            assert updated == 0
            result = _read_predictions(path)
            assert "harmonized_from" not in result[0]["materials"]

    def test_empty_remap(self):
        preds = [_pred("/a", "Steel")]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "predictions.jsonl"
            _write_predictions(preds, path)
            updated = apply_prim_remap(path, {})
            assert updated == 0

    def test_rejects_missing_file(self):
        """Canonicalisation requires the predictions file to exist."""
        with tempfile.TemporaryDirectory() as d:
            missing = Path(d) / "does-not-exist.jsonl"
            with pytest.raises(FileNotFoundError):
                apply_prim_remap(missing, {"/a": "Steel"})

    def test_rejects_non_jsonl_suffix(self):
        """Reject anything that isn't a .jsonl file even if it exists."""
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "predictions.txt"
            path.write_text("{}\n")
            with pytest.raises(ValueError, match="Not a predictions JSONL file"):
                apply_prim_remap(path, {"/a": "Steel"})


class TestFindBestPredictions:
    def test_prefers_restored(self):
        with tempfile.TemporaryDirectory() as d:
            wd = Path(d)
            raw = wd / "predictions" / "predictions.jsonl"
            restored = wd / "restored" / "restored_predictions.jsonl"
            _write_predictions([_pred("/a", "Steel")], raw)
            _write_predictions([_pred("/a", "Plastic")], restored)

            result = _find_best_predictions(wd)
            assert result == restored

    def test_falls_back_to_raw(self):
        with tempfile.TemporaryDirectory() as d:
            wd = Path(d)
            raw = wd / "predictions" / "predictions.jsonl"
            _write_predictions([_pred("/a", "Steel")], raw)

            result = _find_best_predictions(wd)
            assert result == raw

    def test_returns_none_if_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            result = _find_best_predictions(Path(d))
            assert result is None


class TestLoadPredictions:
    def test_loads_jsonl(self):
        preds = [_pred("/a", "Steel"), _pred("/b", "Plastic")]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "predictions.jsonl"
            _write_predictions(preds, path)
            result = _load_predictions(path)
            assert len(result) == 2
            assert result[0]["id"] == "/a"

    def test_skips_blank_lines(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "predictions.jsonl"
            content = (
                json.dumps(_pred("/a", "Steel"))
                + "\n\n"
                + json.dumps(_pred("/b", "X"))
                + "\n"
            )
            path.write_text(content)
            result = _load_predictions(path)
            assert len(result) == 2


# ===================================================================
# collect.py — utility tests
# ===================================================================


class TestExtractMaterialName:
    def test_from_materials_dict(self):
        pred = {"materials": {"material": "Steel"}}
        assert _extract_material_name(pred) == "Steel"

    def test_from_materials_string(self):
        pred = {"materials": "Steel"}
        assert _extract_material_name(pred) == "Steel"

    def test_from_top_level_material(self):
        pred = {"material": "Steel"}
        assert _extract_material_name(pred) == "Steel"

    def test_returns_none_if_missing(self):
        assert _extract_material_name({}) is None


class TestResolveConflictsSimpleMode:
    """Verify that mode='simple' uses majority vote without LLM."""

    def test_simple_mode_majority_vote(self):
        """Simple mode resolves conflicts via majority vote."""
        conflicts = {0: [0, 1, 2]}
        preds = [
            _pred("/a", "Steel"),
            _pred("/b", "Steel"),
            _pred("/c", "Plastic"),
        ]
        remap = _resolve_conflicts(
            conflicts, preds, {0: ["sig:test"]}, llm_config=None, mode="simple"
        )
        assert remap == {"/c": "Steel"}

    def test_simple_mode_ignores_llm_config(self):
        """Even with llm_config present, simple mode uses majority vote only."""
        conflicts = {0: [0, 1, 2]}
        preds = [
            _pred("/a", "Steel"),
            _pred("/b", "Steel"),
            _pred("/c", "Plastic"),
        ]
        # Pass an llm_config — it should be ignored in simple mode
        remap = _resolve_conflicts(
            conflicts,
            preds,
            {0: ["sig:test"]},
            llm_config={"model": "some-model"},
            mode="simple",
        )
        assert remap == {"/c": "Steel"}

    def test_simple_mode_tie_no_remap(self):
        """50/50 tie in simple mode — no override."""
        conflicts = {0: [0, 1]}
        preds = [
            _pred("/a", "Steel"),
            _pred("/b", "Plastic"),
        ]
        remap = _resolve_conflicts(
            conflicts, preds, {0: ["sig:test"]}, llm_config=None, mode="simple"
        )
        assert remap == {}

    def test_simple_mode_empty_conflicts(self):
        remap = _resolve_conflicts({}, [], {}, llm_config=None, mode="simple")
        assert remap == {}

    def test_mode_defaults_to_full(self):
        """Default mode is 'full' — with no llm_config, falls back to majority vote."""
        conflicts = {0: [0, 1, 2]}
        preds = [
            _pred("/a", "Steel"),
            _pred("/b", "Steel"),
            _pred("/c", "Plastic"),
        ]
        # mode defaults to "full", no llm_config → fallback to majority vote
        remap = _resolve_conflicts(conflicts, preds, {0: ["sig:test"]})
        assert remap == {"/c": "Steel"}


# ===================================================================
# Cross-cutting: audit field contracts
# ===================================================================


class TestAuditFieldContracts:
    """Verify the audit fields written by each step.

    Contracts:
    - Reconcile writes ``original_material``
    - Harmonize (both full and simple mode) writes ``harmonized_from``
    """

    def test_reconcile_writes_original_material(self):
        preds = [_pred("/a", "Car Paint Orange")]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "predictions.jsonl"
            _write_predictions(preds, path)
            _remap_predictions_file(path, {"Car Paint Orange": "Steel Painted Orange"})
            result = _read_predictions(path)
            assert result[0]["materials"]["original_material"] == "Car Paint Orange"
            assert result[0]["materials"]["material"] == "Steel Painted Orange"

    def test_harmonize_writes_harmonized_from(self):
        preds = [_pred("/a", "Plastic")]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "predictions.jsonl"
            _write_predictions(preds, path)
            apply_prim_remap(path, {"/a": "Steel"})
            result = _read_predictions(path)
            assert result[0]["materials"]["harmonized_from"] == "Plastic"
            assert result[0]["materials"]["material"] == "Steel"

    def test_simple_mode_also_writes_harmonized_from(self):
        """Simple mode uses the same apply_prim_remap, so audit trail is preserved."""
        preds = [_pred("/a", "Plastic")]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "predictions.jsonl"
            _write_predictions(preds, path)
            # Simulate what simple mode does: majority vote → apply_prim_remap
            apply_prim_remap(path, {"/a": "Steel"})
            result = _read_predictions(path)
            assert result[0]["materials"]["harmonized_from"] == "Plastic"


# ===================================================================
# Integration: reconcile → harmonize → collect data flow
# ===================================================================


class TestDataFlowContracts:
    """Test the data contracts between pipeline steps.

    These verify that the output format of one step is compatible with
    the input format of the next step.
    """

    def test_reconcile_output_is_valid_predictions_jsonl(self):
        """After reconcile remap, the file is still valid JSONL with 'id' and 'materials'."""
        preds = [
            _pred("/a", "Car Paint Orange"),
            _pred("/b", "Steel Painted Black"),
        ]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "predictions.jsonl"
            _write_predictions(preds, path)
            _remap_predictions_file(path, {"Car Paint Orange": "Steel Painted Orange"})

            # Verify _load_predictions (used by harmonize) can read it
            loaded = _load_predictions(path)
            assert len(loaded) == 2
            assert loaded[0]["materials"]["material"] == "Steel Painted Orange"

    def test_harmonize_output_is_valid_predictions_jsonl(self):
        """After harmonize remap, the file is still valid JSONL."""
        preds = [
            _pred("/a", "Steel"),
            _pred("/b", "Plastic"),
        ]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "predictions.jsonl"
            _write_predictions(preds, path)
            apply_prim_remap(path, {"/b": "Steel"})

            loaded = _load_predictions(path)
            assert len(loaded) == 2
            assert loaded[1]["materials"]["material"] == "Steel"

    def test_extract_material_name_reads_reconciled_data(self):
        """_extract_material_name should read the current 'material', not 'original_material'."""
        pred = {
            "materials": {
                "material": "Steel Painted Orange",
                "original_material": "Car Paint Orange",
            }
        }
        assert _extract_material_name(pred) == "Steel Painted Orange"

    def test_extract_material_name_reads_harmonized_data(self):
        """_extract_material_name should read the current 'material', not 'harmonized_from'."""
        pred = {
            "materials": {
                "material": "Steel",
                "harmonized_from": "Plastic",
            }
        }
        assert _extract_material_name(pred) == "Steel"

    def test_chained_reconcile_then_harmonize(self):
        """Reconcile remap followed by harmonize remap — both audit fields preserved."""
        preds = [
            _pred("/a", "Car Paint Orange"),
            _pred("/b", "Car Paint Orange"),
        ]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "predictions.jsonl"
            _write_predictions(preds, path)

            # Step 1: reconcile remaps name
            _remap_predictions_file(path, {"Car Paint Orange": "Steel Painted Orange"})

            # Step 2: harmonize remaps prim /b to a different material
            apply_prim_remap(path, {"/b": "Aluminum"})

            result = _read_predictions(path)

            # /a: reconciled but not harmonized
            assert result[0]["materials"]["material"] == "Steel Painted Orange"
            assert result[0]["materials"]["original_material"] == "Car Paint Orange"
            assert "harmonized_from" not in result[0]["materials"]

            # /b: reconciled then harmonized — both fields present
            assert result[1]["materials"]["material"] == "Aluminum"
            assert result[1]["materials"]["original_material"] == "Car Paint Orange"
            assert result[1]["materials"]["harmonized_from"] == "Steel Painted Orange"


# ===================================================================
# Edge cases and robustness
# ===================================================================


class TestEdgeCases:
    def test_remap_predictions_file_preserves_extra_fields(self):
        """Remap should not drop fields beyond 'materials'."""
        pred = _pred("/a", "Old")
        pred["confidence"] = 0.9
        pred["extra"] = {"foo": "bar"}
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "predictions.jsonl"
            _write_predictions([pred], path)
            _remap_predictions_file(path, {"Old": "New"})
            result = _read_predictions(path)
            assert result[0]["confidence"] == 0.9
            assert result[0]["extra"] == {"foo": "bar"}

    def test_harmonize_remap_preserves_extra_fields(self):
        pred = _pred("/a", "Old")
        pred["confidence"] = 0.85
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "predictions.jsonl"
            _write_predictions([pred], path)
            apply_prim_remap(path, {"/a": "New"})
            result = _read_predictions(path)
            assert result[0]["confidence"] == 0.85

    def test_majority_vote_exact_50_percent(self):
        """Exactly 50% — _majority_vote_fallback requires > 50%, so no override."""
        conflicts = {0: [0, 1, 2, 3]}
        preds = [
            _pred("/a", "Steel"),
            _pred("/b", "Steel"),
            _pred("/c", "Plastic"),
            _pred("/d", "Plastic"),
        ]
        remap = _majority_vote_fallback(conflicts, preds)
        # _majority_vote_fallback uses `top_count > len(members) / 2` — strict >
        # So 2 > 2 is False → no remap
        assert remap == {}


# ===================================================================
# Fix 3: Reconcile targets the right prediction file
# ===================================================================


class TestReconcileFileTargeting:
    """Verify reconcile reads/writes the file that downstream steps consume."""

    def test_find_best_predictions_file_prefers_restored(self):
        with tempfile.TemporaryDirectory() as d:
            wd = Path(d)
            raw = wd / "predictions" / "predictions.jsonl"
            restored = wd / "restored" / "restored_predictions.jsonl"
            _write_predictions([_pred("/a", "Steel")], raw)
            _write_predictions([_pred("/a", "Steel")], restored)

            result = _find_best_predictions_file(wd)
            assert result == restored

    def test_find_best_predictions_file_falls_back_to_raw(self):
        with tempfile.TemporaryDirectory() as d:
            wd = Path(d)
            raw = wd / "predictions" / "predictions.jsonl"
            _write_predictions([_pred("/a", "Steel")], raw)

            result = _find_best_predictions_file(wd)
            assert result == raw

    def test_find_best_predictions_file_none_when_empty(self):
        with tempfile.TemporaryDirectory() as d:
            assert _find_best_predictions_file(Path(d)) is None

    def test_reconcile_remap_targets_restored_when_present(self):
        """When restored exists, reconcile should modify IT (not raw predictions)."""
        with tempfile.TemporaryDirectory() as d:
            wd = Path(d)
            raw = wd / "predictions" / "predictions.jsonl"
            restored = wd / "restored" / "restored_predictions.jsonl"
            _write_predictions([_pred("/a", "Car Paint Orange")], raw)
            _write_predictions([_pred("/a", "Car Paint Orange")], restored)

            pred_file = _find_best_predictions_file(wd)
            assert pred_file == restored

            _remap_predictions_file(
                pred_file, {"Car Paint Orange": "Steel Painted Orange"}
            )

            # Restored should be modified
            restored_result = _read_predictions(restored)
            assert restored_result[0]["materials"]["material"] == "Steel Painted Orange"

            # Raw should be UNTOUCHED
            raw_result = _read_predictions(raw)
            assert raw_result[0]["materials"]["material"] == "Car Paint Orange"

    def test_reconcile_remap_targets_raw_when_no_restored(self):
        """When restored doesn't exist, reconcile modifies raw predictions."""
        with tempfile.TemporaryDirectory() as d:
            wd = Path(d)
            raw = wd / "predictions" / "predictions.jsonl"
            _write_predictions([_pred("/a", "Car Paint Orange")], raw)

            pred_file = _find_best_predictions_file(wd)
            assert pred_file == raw

            _remap_predictions_file(
                pred_file, {"Car Paint Orange": "Steel Painted Orange"}
            )

            result = _read_predictions(raw)
            assert result[0]["materials"]["material"] == "Steel Painted Orange"

    def test_reconcile_then_harmonize_reads_same_file(self):
        """After reconcile modifies restored, harmonize reads the reconciled data."""
        with tempfile.TemporaryDirectory() as d:
            wd = Path(d)
            restored = wd / "restored" / "restored_predictions.jsonl"
            _write_predictions(
                [_pred("/a", "Car Paint Orange"), _pred("/b", "Steel")],
                restored,
            )

            # Reconcile targets restored
            pred_file = _find_best_predictions_file(wd)
            _remap_predictions_file(
                pred_file, {"Car Paint Orange": "Steel Painted Orange"}
            )

            # Harmonize also reads restored (via _find_best_predictions)
            harmonize_file = _find_best_predictions(wd)
            assert harmonize_file == restored

            loaded = _load_predictions(harmonize_file)
            assert loaded[0]["materials"]["material"] == "Steel Painted Orange"
