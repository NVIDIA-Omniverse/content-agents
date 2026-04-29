# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from material_agent.scene.config_gen import (
    _rebase_paths,
    _sanitize_name,
    _unique_safe_names,
    generate_all_configs,
    generate_all_payload_configs,
    generate_payload_config,
    generate_sub_asset_config,
)
from material_agent.scene.manifest import PayloadGroup, SceneManifest, SubAsset

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base_scene_config() -> dict:
    """Minimal scene config template used across tests."""
    return {
        "project": {"name": "test"},
        "input": {"usd_path": "scene.usda"},
        "output": {"format": "usda"},
        "steps": {
            "apply": {"enabled": True},
            "render": {"enabled": True},
            "restore_usd": {"enabled": False},
        },
        "scene": {"analyze": {"some_key": True}},
    }


def _make_sub_asset(
    *,
    id: str = "sa1",
    name: str = "Ladder",
    prim_path: str = "/Root/Ladder",
    **kwargs,
) -> SubAsset:
    return SubAsset(id=id, name=name, prim_path=prim_path, **kwargs)


def _make_payload_group(
    *,
    id: str = "pg1",
    group_name: str = "Tray",
    payload_file: str = "/assets/Tray/Tray.usd",
    **kwargs,
) -> PayloadGroup:
    return PayloadGroup(
        id=id, group_name=group_name, payload_file=payload_file, **kwargs
    )


def _write_yaml(path: Path, data: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f)
    return path


def _read_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# _sanitize_name
# ---------------------------------------------------------------------------


class TestSanitizeName:
    def test_simple_name(self):
        assert _sanitize_name("Ladder") == "ladder"

    def test_special_characters_stripped(self):
        assert _sanitize_name("My Object!@#$%") == "my_object"

    def test_spaces_become_underscores(self):
        assert _sanitize_name("hello world") == "hello_world"

    def test_consecutive_underscores_collapsed(self):
        assert _sanitize_name("a___b") == "a_b"

    def test_leading_trailing_underscores_stripped(self):
        assert _sanitize_name("__foo__") == "foo"

    def test_empty_string_returns_unnamed(self):
        assert _sanitize_name("") == "unnamed"

    def test_only_special_chars_returns_unnamed(self):
        assert _sanitize_name("@#$") == "unnamed"

    def test_hyphens_preserved(self):
        assert _sanitize_name("UR-5e") == "ur-5e"

    def test_digits_preserved(self):
        assert _sanitize_name("Part_007") == "part_007"

    def test_slash_replaced(self):
        result = _sanitize_name("/Root/Obj")
        assert "/" not in result


# ---------------------------------------------------------------------------
# _unique_safe_names
# ---------------------------------------------------------------------------


class TestUniqueSafeNames:
    def test_no_collisions(self):
        assets = [
            _make_sub_asset(id="a1", name="Alpha"),
            _make_sub_asset(id="a2", name="Beta"),
        ]
        result = _unique_safe_names(assets)
        assert result == {"a1": "alpha", "a2": "beta"}

    def test_collisions_get_id_suffix(self):
        assets = [
            _make_sub_asset(id="id_100", name="Widget"),
            _make_sub_asset(id="id_200", name="Widget"),
        ]
        result = _unique_safe_names(assets)
        # Both should have unique names with ID suffix
        assert result["id_100"] != result["id_200"]
        assert result["id_100"].startswith("widget_")
        assert result["id_200"].startswith("widget_")

    def test_unique_names_no_suffix(self):
        assets = [_make_sub_asset(id="x", name="Solo")]
        result = _unique_safe_names(assets)
        assert result["x"] == "solo"


# ---------------------------------------------------------------------------
# _rebase_paths
# ---------------------------------------------------------------------------


class TestRebasePaths:
    def test_relative_path_rebased(self, tmp_path: Path):
        old_base = tmp_path / "configs"
        new_base = tmp_path / "configs" / "sub"
        old_base.mkdir()
        new_base.mkdir()
        config = {"input": {"usd_path": "scene.usda"}}
        _rebase_paths(config, old_base, new_base)
        # From new_base, we need to go up one level to reach old_base/scene.usda
        assert config["input"]["usd_path"] == str(Path("..") / "scene.usda")

    def test_absolute_path_unchanged(self, tmp_path: Path):
        config = {"input": {"usd_path": "/absolute/scene.usda"}}
        _rebase_paths(config, tmp_path, tmp_path / "sub")
        assert config["input"]["usd_path"] == "/absolute/scene.usda"

    def test_nested_dict_rebased(self, tmp_path: Path):
        old_base = tmp_path / "a"
        new_base = tmp_path / "a" / "b"
        old_base.mkdir()
        new_base.mkdir()
        config = {"steps": {"optimize_usd": {"path": "data/model.usd"}}}
        _rebase_paths(config, old_base, new_base)
        assert config["steps"]["optimize_usd"]["path"] == str(
            Path("..") / "data" / "model.usd"
        )

    def test_path_list_keys_rebased(self, tmp_path: Path):
        old_base = tmp_path / "a"
        new_base = tmp_path / "a" / "b"
        old_base.mkdir()
        new_base.mkdir()
        config = {"reference_images": ["img1.png", "img2.png"]}
        _rebase_paths(config, old_base, new_base)
        for val in config["reference_images"]:
            assert val.startswith("..")

    def test_non_path_keys_untouched(self, tmp_path: Path):
        config = {"input": {"description": "relative/looking/string"}}
        _rebase_paths(config, tmp_path, tmp_path / "sub")
        assert config["input"]["description"] == "relative/looking/string"

    def test_working_dir_rebased(self, tmp_path: Path):
        old_base = tmp_path / "root"
        new_base = tmp_path / "root" / "sub"
        old_base.mkdir()
        new_base.mkdir()
        config = {"project": {"working_dir": ".workdir"}}
        _rebase_paths(config, old_base, new_base)
        assert ".." in config["project"]["working_dir"]


# ---------------------------------------------------------------------------
# generate_sub_asset_config
# ---------------------------------------------------------------------------


class TestGenerateSubAssetConfig:
    def test_basic_generation(self, tmp_path: Path):
        sa = _make_sub_asset()
        config = _base_scene_config()
        out = tmp_path / "configs" / "ladder.yaml"

        result = generate_sub_asset_config(sa, config, out)
        assert result == out
        assert out.exists()

        data = _read_yaml(out)
        # scene section removed
        assert "scene" not in data
        # prim_path set
        assert data["input"]["prim_path"] == "/Root/Ladder"
        # layer_only forced
        assert data["output"]["layer_only"] is True
        assert data["output"]["flatten_output"] is False
        # apply and render disabled
        assert data["steps"]["apply"]["enabled"] is False
        assert data["steps"]["render"]["enabled"] is False
        # restore_usd enabled
        assert data["steps"]["restore_usd"]["enabled"] is True

    def test_project_name_sanitized(self, tmp_path: Path):
        sa = _make_sub_asset(name="My Object!!")
        config = _base_scene_config()
        out = tmp_path / "out.yaml"

        generate_sub_asset_config(sa, config, out)
        data = _read_yaml(out)
        assert data["project"]["name"] == "my_object"
        assert data["project"]["session_id"] == "my_object"

    def test_session_id_override(self, tmp_path: Path):
        sa = _make_sub_asset()
        config = _base_scene_config()
        out = tmp_path / "out.yaml"

        generate_sub_asset_config(sa, config, out, session_id="custom_session")
        data = _read_yaml(out)
        assert data["project"]["name"] == "custom_session"
        assert data["project"]["session_id"] == "custom_session"

    def test_path_rebasing(self, tmp_path: Path):
        scene_dir = tmp_path / "scene"
        scene_dir.mkdir()
        configs_dir = tmp_path / "scene" / "configs"
        configs_dir.mkdir()

        sa = _make_sub_asset()
        config = {"input": {"usd_path": "model.usda"}, "steps": {}}
        out = configs_dir / "asset.yaml"

        generate_sub_asset_config(sa, config, out, scene_config_dir=scene_dir)
        data = _read_yaml(out)
        # usd_path should be rebased: from configs/, go up to scene/
        assert data["input"]["usd_path"] == str(Path("..") / "model.usda")

    def test_does_not_mutate_original_config(self, tmp_path: Path):
        sa = _make_sub_asset()
        config = _base_scene_config()
        original_keys = set(config.keys())
        out = tmp_path / "out.yaml"

        generate_sub_asset_config(sa, config, out)
        # Original config should still have "scene" key
        assert "scene" in config
        assert set(config.keys()) == original_keys

    def test_extracted_usd_used_when_exists(self, tmp_path: Path):
        extracted = tmp_path / "extracted" / "ladder.usda"
        extracted.parent.mkdir()
        extracted.write_text("# extracted")

        sa = _make_sub_asset(extracted_usd=str(extracted))
        config = _base_scene_config()
        out = tmp_path / "configs" / "ladder.yaml"

        generate_sub_asset_config(sa, config, out)
        data = _read_yaml(out)
        # Should use a relative path to the extracted USD
        assert "extracted" in data["input"]["usd_path"]

    def test_split_context_injected(self, tmp_path: Path):
        sa = _make_sub_asset(
            split_context={
                "parent_name": "BigMachine",
                "sibling_names": ["Arm", "Ladder"],
                "ancestors": ["Factory", "Line_01"],
            }
        )
        config = _base_scene_config()
        config["steps"]["build_dataset_prepare_dataset"] = {
            "prompts": {"vlm_system": "You are a material expert."}
        }
        out = tmp_path / "out.yaml"

        generate_sub_asset_config(sa, config, out)
        data = _read_yaml(out)
        vlm_system = data["steps"]["build_dataset_prepare_dataset"]["prompts"][
            "vlm_system"
        ]
        assert "extracted from a larger structure" in vlm_system
        assert "Factory" in vlm_system

    def test_output_is_valid_yaml(self, tmp_path: Path):
        sa = _make_sub_asset()
        config = _base_scene_config()
        out = tmp_path / "out.yaml"

        generate_sub_asset_config(sa, config, out)
        # Should not raise
        data = _read_yaml(out)
        assert isinstance(data, dict)


# ---------------------------------------------------------------------------
# generate_all_configs
# ---------------------------------------------------------------------------


class TestGenerateAllConfigs:
    def test_generates_configs_for_all_assets(self, tmp_path: Path):
        manifest = SceneManifest(
            sub_assets=[
                _make_sub_asset(id="a1", name="Alpha", prim_path="/Root/Alpha"),
                _make_sub_asset(id="a2", name="Beta", prim_path="/Root/Beta"),
            ]
        )
        config = _base_scene_config()
        configs_dir = tmp_path / "configs"

        result = generate_all_configs(manifest, config, configs_dir)
        assert len(list(configs_dir.glob("*.yaml"))) == 2

        # Each asset should have config_path and working_dir set
        for sa in result.sub_assets:
            assert sa.config_path is not None
            assert sa.working_dir is not None

    def test_names_filter_limits_output(self, tmp_path: Path):
        manifest = SceneManifest(
            sub_assets=[
                _make_sub_asset(id="a1", name="Alpha", prim_path="/Root/Alpha"),
                _make_sub_asset(id="a2", name="Beta", prim_path="/Root/Beta"),
            ]
        )
        config = _base_scene_config()
        configs_dir = tmp_path / "configs"

        generate_all_configs(manifest, config, configs_dir, names_filter=["Alpha"])
        # Only Alpha should have a config
        assert manifest.sub_assets[0].config_path is not None
        assert manifest.sub_assets[1].config_path is None

    def test_collision_handling(self, tmp_path: Path):
        manifest = SceneManifest(
            sub_assets=[
                _make_sub_asset(id="id_1", name="Widget", prim_path="/Root/W1"),
                _make_sub_asset(id="id_2", name="Widget", prim_path="/Root/W2"),
            ]
        )
        config = _base_scene_config()
        configs_dir = tmp_path / "configs"

        generate_all_configs(manifest, config, configs_dir)
        files = list(configs_dir.glob("*.yaml"))
        assert len(files) == 2
        # File names should be different
        names = {f.stem for f in files}
        assert len(names) == 2


# ---------------------------------------------------------------------------
# generate_payload_config
# ---------------------------------------------------------------------------


class TestGeneratePayloadConfig:
    def test_basic_generation(self, tmp_path: Path):
        payload_file = tmp_path / "assets" / "Tray.usd"
        payload_file.parent.mkdir(parents=True)
        payload_file.write_text("# payload")

        pg = _make_payload_group(payload_file=str(payload_file))
        config = _base_scene_config()
        out = tmp_path / "configs" / "tray.yaml"

        result = generate_payload_config(pg, config, out)
        assert result == out
        assert out.exists()

        data = _read_yaml(out)
        # scene section removed
        assert "scene" not in data
        # prim_path removed (no scoping for payloads)
        assert "prim_path" not in data.get("input", {})
        # layer_only forced
        assert data["output"]["layer_only"] is True
        # apply enabled with layer_only
        assert data["steps"]["apply"]["enabled"] is True
        assert data["steps"]["apply"]["layer_only"] is True
        assert data["steps"]["apply"]["skip_instance_check"] is True
        # render disabled
        assert data["steps"]["render"]["enabled"] is False
        # restore_usd enabled
        assert data["steps"]["restore_usd"]["enabled"] is True

    def test_project_identity_set(self, tmp_path: Path):
        payload_file = tmp_path / "Tray.usd"
        payload_file.write_text("# payload")

        pg = _make_payload_group(group_name="my_tray", payload_file=str(payload_file))
        config = _base_scene_config()
        out = tmp_path / "out.yaml"

        generate_payload_config(pg, config, out)
        data = _read_yaml(out)
        assert data["project"]["name"] == "my_tray"
        assert data["project"]["session_id"] == "my_tray"

    def test_container_payload_disables_so(self, tmp_path: Path):
        payload_file = tmp_path / "Parent.usd"
        payload_file.write_text("# parent")

        pg = _make_payload_group(
            payload_file=str(payload_file),
            child_payload_files=["/child1.usd", "/child2.usd"],
        )
        config = _base_scene_config()
        out = tmp_path / "out.yaml"

        generate_payload_config(pg, config, out)
        data = _read_yaml(out)
        assert data["steps"]["optimize_usd"]["enabled"] is False

    def test_representative_payload_sets_so_options(self, tmp_path: Path):
        payload_file = tmp_path / "Tray.usd"
        payload_file.write_text("# payload")
        rep_file = tmp_path / "Tray_rep.usd"
        rep_file.write_text("# representative")

        pg = _make_payload_group(
            payload_file=str(payload_file),
            representative_path=str(rep_file),
        )
        config = _base_scene_config()
        out = tmp_path / "out.yaml"

        generate_payload_config(pg, config, out)
        data = _read_yaml(out)
        so_settings = data["steps"]["optimize_usd"]["scene_optimizer_settings"]
        assert so_settings["enableDeinstance"] is False
        assert so_settings["enableSplitMeshes"] is True
        assert so_settings["enableDeduplicate"] is False
        # Should use representative path as input
        assert "Tray_rep" in data["input"]["usd_path"]
        # Should store original payload path
        assert "_original_payload_file" in data

    def test_modified_input_path_used(self, tmp_path: Path):
        payload_file = tmp_path / "Original.usd"
        payload_file.write_text("# orig")
        modified_file = tmp_path / "Modified.usd"
        modified_file.write_text("# modified")

        pg = _make_payload_group(
            payload_file=str(payload_file),
            modified_input_path=str(modified_file),
        )
        config = _base_scene_config()
        out = tmp_path / "out.yaml"

        generate_payload_config(pg, config, out)
        data = _read_yaml(out)
        assert "Modified" in data["input"]["usd_path"]

    def test_payload_context_injected(self, tmp_path: Path):
        payload_file = (
            tmp_path / "Assets" / "Phase_01" / "Machine" / "Tray" / "Tray.usd"
        )
        payload_file.parent.mkdir(parents=True)
        payload_file.write_text("# tray")

        pg = _make_payload_group(payload_file=str(payload_file), group_name="Tray")
        config = _base_scene_config()
        config["steps"]["build_dataset_prepare_dataset"] = {
            "prompts": {"vlm_system": "You are a material expert."}
        }
        out = tmp_path / "out.yaml"

        generate_payload_config(pg, config, out)
        data = _read_yaml(out)
        vlm = data["steps"]["build_dataset_prepare_dataset"]["prompts"]["vlm_system"]
        assert "industrial/warehouse" in vlm

    def test_does_not_mutate_original_config(self, tmp_path: Path):
        payload_file = tmp_path / "Tray.usd"
        payload_file.write_text("# payload")

        pg = _make_payload_group(payload_file=str(payload_file))
        config = _base_scene_config()
        out = tmp_path / "out.yaml"

        generate_payload_config(pg, config, out)
        assert "scene" in config


# ---------------------------------------------------------------------------
# generate_all_payload_configs
# ---------------------------------------------------------------------------


class TestGenerateAllPayloadConfigs:
    def test_generates_configs_for_all_payloads(self, tmp_path: Path):
        pf1 = tmp_path / "A.usd"
        pf2 = tmp_path / "B.usd"
        pf1.write_text("# a")
        pf2.write_text("# b")

        manifest = SceneManifest(
            payload_groups=[
                _make_payload_group(id="p1", group_name="A", payload_file=str(pf1)),
                _make_payload_group(id="p2", group_name="B", payload_file=str(pf2)),
            ]
        )
        config = _base_scene_config()
        configs_dir = tmp_path / "configs"

        result = generate_all_payload_configs(manifest, config, configs_dir)
        payload_dir = configs_dir / "payloads"
        assert len(list(payload_dir.glob("*.yaml"))) == 2

        for pg in result.payload_groups:
            assert pg.config_path is not None
            assert pg.working_dir is not None

    def test_empty_payloads_returns_manifest(self, tmp_path: Path):
        manifest = SceneManifest(payload_groups=[])
        config = _base_scene_config()

        result = generate_all_payload_configs(manifest, config, tmp_path / "c")
        assert result is manifest

    def test_skipped_payloads_excluded(self, tmp_path: Path):
        pf = tmp_path / "A.usd"
        pf.write_text("# a")

        manifest = SceneManifest(
            payload_groups=[
                _make_payload_group(id="p1", group_name="A", payload_file=str(pf)),
                PayloadGroup(
                    id="p2",
                    group_name="B",
                    payload_file=str(pf),
                    status="skipped",
                ),
            ]
        )
        config = _base_scene_config()
        configs_dir = tmp_path / "configs"

        generate_all_payload_configs(manifest, config, configs_dir)
        payload_dir = configs_dir / "payloads"
        files = list(payload_dir.glob("*.yaml"))
        assert len(files) == 1
        assert files[0].stem == "A"
