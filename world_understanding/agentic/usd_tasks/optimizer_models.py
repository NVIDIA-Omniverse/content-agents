# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pydantic models for Scene Optimizer API parameters.

These models mirror the API schema and provide validation for optimization settings.
Users can specify parameters in either snake_case or camelCase in YAML configs.
"""

from enum import IntEnum, StrEnum

from pydantic import BaseModel, Field, model_validator


class UsdFormat(StrEnum):
    """USD file format options."""

    USD = "usd"  # Unspecified format
    USDA = "usda"  # ASCII text format
    USDC = "usdc"  # Binary format (default)


class SplitOnMethod(IntEnum):
    """Method for detecting disjoint meshes (matches C++ SplitMeshesSplitOn enum)"""

    VERTICES = 0  # Split on topology
    GEOM_SUBSETS = 1  # Split on UsdGeom Subsets


class SplitMeshesMethod(IntEnum):
    """How to express mesh subsets in the stage (matches C++ SplitMeshesMethod enum)"""

    GEOM_SUBSET = 0  # Create UsdGeom Subsets
    MESH_PRIM = 1  # Create UsdGeom Mesh Prims


class OriginalGeomOption(IntEnum):
    """How to handle original geometry after operations (matches C++ RemoveMethod enum)"""

    IGNORE = 0  # Leave original as is
    DELETE = 1  # Remove the original prim
    DEACTIVATE = 2  # Deactivate the original prim
    HIDE = 3  # Hide the original prim


class SpatialMode(IntEnum):
    """Spatial clustering mode (matches C++ ClusterMode enum)"""

    NONE = 0
    BOUNDING_BOX = 1
    VERTEX_COUNT = 2


class MergeBoundary(IntEnum):
    """Merge boundary options (matches C++ MergePointOption enum)"""

    STAGE = 0  # eDefault - Use Pseudo root prim
    PARENT_XFORM = 1  # eXform - Use the first xformable parent
    KIND_ASSEMBLY = 2  # eKindAssembly - Use the first parent of kind assembly
    KIND_GROUP = 3  # eKindGroup - Use the first parent of kind group
    KIND_COMPONENT = 4  # eKindComponent - Use the first parent of kind component
    KIND_MODEL = 5  # eKindModel - Use the first parent of kind model
    KIND_SUBCOMPONENT = (
        6  # eKindSubcomponent - Use the first parent of kind subcomponent
    )
    ROOT_PRIM = 7  # eRootPrim - Use root prims
    PARENT_PRIM = 8  # eParentPrim - Use the first parent
    ORIGINAL_PRIM = (
        9  # eOriginalPrim - Use the original prim that meshes have been split from
    )


class DuplicateMethod(IntEnum):
    """Method for deduplication (matches C++ DuplicateOption enum)"""

    COPY_VALUES = 0
    REFERENCE = 1
    INSTANCEABLE_REFERENCE = 2
    SET_ATTRIBUTE = 3


class DeinstanceConfig(BaseModel):
    """Deinstance operation settings.

    Matches client_scene_optimizer.py lines 137-139.
    """

    prim_paths: list[str] = Field(default_factory=list, alias="primPaths")

    class Config:
        populate_by_name = True


class SplitMeshesConfig(BaseModel):
    """Split meshes operation settings.

    Matches client_scene_optimizer.py lines 140-142.
    """

    paths: list[str] = Field(default_factory=list)

    class Config:
        populate_by_name = True


class DeduplicateConfig(BaseModel):
    """Deduplicate geometry operation settings.

    Matches client_scene_optimizer.py lines 143-151.
    """

    mesh_prim_paths: list[str] = Field(default_factory=list, alias="meshPrimPaths")
    tolerance: float = Field(default=0.001)
    consider_deep_transforms: bool = Field(
        default=True, alias="considerDeepTransforms"
    )  # default True: use deep transforms mode (cannot be combined with fuzzy)
    fuzzy: bool = Field(
        default=False
    )  # default False: deep transforms mode (fuzzy not compatible with consider_deep_transforms)
    use_gpu: bool = Field(default=False, alias="useGpu")
    allow_scaling: bool = Field(default=False, alias="allowScaling")
    ignore_attributes: list[str] = Field(default_factory=list, alias="ignoreAttributes")

    @model_validator(mode="after")
    def _consider_deep_transforms_and_fuzzy_mutually_exclusive(
        self,
    ) -> "DeduplicateConfig":
        if self.consider_deep_transforms and self.fuzzy:
            raise ValueError(
                "consider_deep_transforms and fuzzy are mutually exclusive; "
                "set one to False when enabling the other."
            )
        return self

    class Config:
        populate_by_name = True


class SceneOptimizerSettings(BaseModel):
    """Scene optimizer settings matching client_scene_optimizer.py structure."""

    # Operation enable flags (lines 134-136)
    enable_deinstance: bool = Field(default=True, alias="enableDeinstance")
    enable_split_meshes: bool = Field(default=True, alias="enableSplitMeshes")
    enable_deduplicate: bool = Field(default=True, alias="enableDeduplicate")

    # Nested operation configs (lines 137-151)
    deinstance: DeinstanceConfig = Field(default_factory=DeinstanceConfig)
    split_meshes: SplitMeshesConfig = Field(
        default_factory=SplitMeshesConfig, alias="splitMeshes"
    )
    deduplicate: DeduplicateConfig = Field(default_factory=DeduplicateConfig)

    # Global settings (lines 152-157)
    generate_report: bool = Field(default=True, alias="generateReport")
    capture_stats: bool = Field(default=True, alias="captureStats")
    verbose: bool = Field(default=False)
    wait_for_assets: bool = Field(default=False, alias="waitForAssets")
    stage_timeout: float = Field(default=180.0, alias="stageTimeout")
    output_format: UsdFormat = Field(default=UsdFormat.USDC, alias="outputFormat")
    extract_geom_subset_indices: bool = Field(
        default=True, alias="extractGeomSubsetIndices"
    )

    class Config:
        populate_by_name = True
