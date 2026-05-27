# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the large-scene manifest model."""

from pathlib import Path

from material_agent.scene.manifest import SceneManifest, SubAsset


def test_scene_manifest_save_uses_atomic_replacement(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("previous")

    manifest = SceneManifest(
        scene_usd_path="/scene.usda",
        sub_assets=[
            SubAsset(
                id="asset_a",
                name="AssetA",
                prim_path="/Root/AssetA",
            )
        ],
    )

    manifest.save(manifest_path)

    loaded = SceneManifest.load(manifest_path)
    assert loaded.scene_usd_path == "/scene.usda"
    assert loaded.sub_assets[0].id == "asset_a"
    assert not list(tmp_path.glob(".manifest.json.*.tmp"))
