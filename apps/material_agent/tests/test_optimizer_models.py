# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for optimizer models with new structure."""

import pytest
from pydantic import ValidationError

from material_agent.tasks.optimizer_models import (
    DeduplicateConfig,
    DeinstanceConfig,
    SceneOptimizerSettings,
    SplitMeshesConfig,
)


def test_deinstance_config_defaults():
    """Test DeinstanceConfig with defaults."""
    config = DeinstanceConfig()
    assert config.prim_paths == []


def test_deinstance_config_with_paths():
    """Test DeinstanceConfig with prim paths."""
    config = DeinstanceConfig(prim_paths=["/World/Mesh1", "/World/Mesh2"])
    assert len(config.prim_paths) == 2


def test_split_meshes_config_defaults():
    """Test SplitMeshesConfig with defaults."""
    config = SplitMeshesConfig()
    assert config.paths == []


def test_deduplicate_config_defaults():
    """Test DeduplicateConfig with defaults (deep transforms mode, not fuzzy)."""
    config = DeduplicateConfig()
    assert config.tolerance == 0.001
    assert config.consider_deep_transforms is True
    assert config.fuzzy is False
    assert config.use_gpu is False
    assert config.allow_scaling is False
    assert config.ignore_attributes == []


def test_deduplicate_config_custom_values():
    """Test DeduplicateConfig with custom values (fuzzy mode)."""
    config = DeduplicateConfig(
        tolerance=0.01,
        consider_deep_transforms=False,
        fuzzy=True,
        use_gpu=True,
        ignore_attributes=["normals"],
    )
    assert config.tolerance == 0.01
    assert config.consider_deep_transforms is False
    assert config.fuzzy is True
    assert config.use_gpu is True
    assert config.ignore_attributes == ["normals"]


def test_deduplicate_config_rejects_both_deep_transforms_and_fuzzy():
    """Test that enabling both consider_deep_transforms and fuzzy raises ValueError."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        DeduplicateConfig(
            consider_deep_transforms=True,
            fuzzy=True,
        )


def test_scene_optimizer_settings_defaults():
    """Test SceneOptimizerSettings with all defaults."""
    settings = SceneOptimizerSettings()

    # Check enable flags default to True
    assert settings.enable_deinstance is True
    assert settings.enable_split_meshes is True
    assert settings.enable_deduplicate is True

    # Check nested configs are instantiated with correct defaults
    assert isinstance(settings.deinstance, DeinstanceConfig)
    assert isinstance(settings.split_meshes, SplitMeshesConfig)
    assert isinstance(settings.deduplicate, DeduplicateConfig)
    # Deduplicate defaults: deep transforms mode (not fuzzy)
    assert settings.deduplicate.consider_deep_transforms is True
    assert settings.deduplicate.fuzzy is False

    # Check global settings
    assert settings.generate_report is True
    assert settings.capture_stats is True
    assert settings.verbose is False
    assert settings.wait_for_assets is False
    assert settings.stage_timeout == 180.0
    assert settings.output_format == "usdc"
    assert settings.extract_geom_subset_indices is True


def test_scene_optimizer_settings_disable_operations():
    """Test disabling operations via enable flags."""
    settings = SceneOptimizerSettings(
        enable_deinstance=False,
        enable_split_meshes=False,
        enable_deduplicate=True,
    )

    assert settings.enable_deinstance is False
    assert settings.enable_split_meshes is False
    assert settings.enable_deduplicate is True


def test_scene_optimizer_settings_nested_configs():
    """Test nested operation configs."""
    settings = SceneOptimizerSettings(
        deinstance={"prim_paths": ["/World/Mesh1"]},
        split_meshes={"paths": ["/World/Mesh2"]},
        deduplicate={
            "tolerance": 0.01,
            "consider_deep_transforms": False,
            "fuzzy": True,
        },
    )

    assert settings.deinstance.prim_paths == ["/World/Mesh1"]
    assert settings.split_meshes.paths == ["/World/Mesh2"]
    assert settings.deduplicate.tolerance == 0.01
    assert settings.deduplicate.fuzzy is True


def test_scene_optimizer_settings_camel_case_aliases():
    """Test that camelCase aliases work correctly."""
    # Test with camelCase input
    settings = SceneOptimizerSettings(
        enableDeinstance=False,
        enableSplitMeshes=True,
        enableDeduplicate=False,
        generateReport=False,
        captureStats=False,
        waitForAssets=True,
        stageTimeout=300.0,
        outputFormat="usda",
        extractGeomSubsetIndices=False,
    )

    assert settings.enable_deinstance is False
    assert settings.enable_split_meshes is True
    assert settings.enable_deduplicate is False
    assert settings.generate_report is False
    assert settings.capture_stats is False
    assert settings.wait_for_assets is True
    assert settings.stage_timeout == 300.0
    assert settings.output_format == "usda"
    assert settings.extract_geom_subset_indices is False


def test_scene_optimizer_settings_serialization():
    """Test that settings serialize correctly with aliases."""
    settings = SceneOptimizerSettings(
        enable_deinstance=False,
        generate_report=False,
    )

    # Serialize with aliases (camelCase for API)
    data = settings.model_dump(by_alias=True, exclude_none=True)

    # Check keys are in camelCase
    assert "enableDeinstance" in data
    assert "enableSplitMeshes" in data
    assert "enableDeduplicate" in data
    assert "generateReport" in data
    assert "captureStats" in data

    # Check values
    assert data["enableDeinstance"] is False
    assert data["enableSplitMeshes"] is True  # default
    assert data["generateReport"] is False


def test_build_enabled_operations_list():
    """Test building list of enabled operations (matching client pattern)."""
    settings = SceneOptimizerSettings(
        enable_deinstance=True,
        enable_split_meshes=False,
        enable_deduplicate=True,
    )

    data = settings.model_dump(by_alias=True)

    # Build enabled ops list (matches client lines 744-750)
    enabled_ops = []
    if data.get("enableDeinstance", True):
        enabled_ops.append("deinstance")
    if data.get("enableSplitMeshes", True):
        enabled_ops.append("split")
    if data.get("enableDeduplicate", True):
        enabled_ops.append("deduplicate")

    assert enabled_ops == ["deinstance", "deduplicate"]
    assert " -> ".join(enabled_ops) == "deinstance -> deduplicate"


def test_validation_all_operations_disabled():
    """Test that we can detect when all operations are disabled."""
    settings = SceneOptimizerSettings(
        enable_deinstance=False,
        enable_split_meshes=False,
        enable_deduplicate=False,
    )

    data = settings.model_dump(by_alias=True)

    # Build enabled ops list
    enabled_ops = []
    if data.get("enableDeinstance", True):
        enabled_ops.append("deinstance")
    if data.get("enableSplitMeshes", True):
        enabled_ops.append("split")
    if data.get("enableDeduplicate", True):
        enabled_ops.append("deduplicate")

    # Should be empty - this should trigger validation error in config task
    assert enabled_ops == []


def test_nvcf_headers_conditional_polling():
    """Test that NVCF headers conditionally include polling headers."""
    from world_understanding.utils.nvcf_utils import create_nvcf_headers

    # Test with poll_seconds=None (no polling headers)
    headers = create_nvcf_headers("test-api-key", 600, poll_seconds=None)
    assert "Authorization" in headers
    assert "Content-Type" in headers
    assert "NVCF-POLL-SECONDS" not in headers
    assert "nvcf-feature-enable-gateway-timeout" not in headers

    # Test with poll_seconds=300 (include polling headers)
    headers_with_poll = create_nvcf_headers("test-api-key", 600, poll_seconds=300)
    assert "Authorization" in headers_with_poll
    assert "Content-Type" in headers_with_poll
    assert headers_with_poll["NVCF-POLL-SECONDS"] == "300"
    assert headers_with_poll["nvcf-feature-enable-gateway-timeout"] == "true"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
