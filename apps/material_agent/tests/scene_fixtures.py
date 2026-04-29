# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Synthetic USD scene and material library generators for scene pipeline E2E tests.

Programmatically creates lightweight USDA scenes that exercise all key
pipeline paths (multi-asset, structural duplicates, gap fill, instances,
payloads) without requiring real 100MB+ assets or external services.
"""

from __future__ import annotations

from pathlib import Path


def create_multi_asset_scene(output_path: Path) -> Path:
    """Create Scene A: multi-asset scene with structural duplicates.

    Hierarchy::

        /Root (Xform, defaultPrim)
          /Root/RobotArm   (Xform, 3 meshes)
          /Root/Conveyor    (Xform, 3 meshes — Roller is gap fill target)
          /Root/Fence_A     (Xform, 2 meshes — structural dup representative)
          /Root/Fence_B     (Xform, 2 meshes — structural dup of Fence_A)
    """
    usda = """\
#usda 1.0
(
    defaultPrim = "Root"
    upAxis = "Y"
)

def Xform "Root"
{
    def Xform "RobotArm"
    {
        def Mesh "Base"
        {
            float3[] extent = [(-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)]
            int[] faceVertexCounts = [4, 4, 4, 4, 4, 4]
            int[] faceVertexIndices = [0,1,3,2, 4,6,7,5, 0,4,5,1, 2,3,7,6, 0,2,6,4, 1,5,7,3]
            point3f[] points = [(-0.5,-0.5,0.5),(0.5,-0.5,0.5),(-0.5,0.5,0.5),(0.5,0.5,0.5),(-0.5,-0.5,-0.5),(0.5,-0.5,-0.5),(-0.5,0.5,-0.5),(0.5,0.5,-0.5)]
        }

        def Mesh "Joint"
        {
            float3[] extent = [(-0.25, 0.5, -0.25), (0.25, 1.5, 0.25)]
            int[] faceVertexCounts = [4, 4, 4, 4, 4, 4]
            int[] faceVertexIndices = [0,1,3,2, 4,6,7,5, 0,4,5,1, 2,3,7,6, 0,2,6,4, 1,5,7,3]
            point3f[] points = [(-0.25,0.5,0.25),(0.25,0.5,0.25),(-0.25,1.5,0.25),(0.25,1.5,0.25),(-0.25,0.5,-0.25),(0.25,0.5,-0.25),(-0.25,1.5,-0.25),(0.25,1.5,-0.25)]
        }

        def Mesh "Gripper"
        {
            float3[] extent = [(-0.1, 1.5, -0.1), (0.1, 2.0, 0.1)]
            int[] faceVertexCounts = [4, 4, 4, 4, 4, 4]
            int[] faceVertexIndices = [0,1,3,2, 4,6,7,5, 0,4,5,1, 2,3,7,6, 0,2,6,4, 1,5,7,3]
            point3f[] points = [(-0.1,1.5,0.1),(0.1,1.5,0.1),(-0.1,2.0,0.1),(0.1,2.0,0.1),(-0.1,1.5,-0.1),(0.1,1.5,-0.1),(-0.1,2.0,-0.1),(0.1,2.0,-0.1)]
        }
    }

    def Xform "Conveyor"
    {
        def Mesh "Frame"
        {
            float3[] extent = [(-2, 0, -0.5), (2, 0.3, 0.5)]
            int[] faceVertexCounts = [4, 4, 4, 4, 4, 4]
            int[] faceVertexIndices = [0,1,3,2, 4,6,7,5, 0,4,5,1, 2,3,7,6, 0,2,6,4, 1,5,7,3]
            point3f[] points = [(-2,0,0.5),(2,0,0.5),(-2,0.3,0.5),(2,0.3,0.5),(-2,0,-0.5),(2,0,-0.5),(-2,0.3,-0.5),(2,0.3,-0.5)]
        }

        def Mesh "Belt"
        {
            float3[] extent = [(-1.8, 0.3, -0.4), (1.8, 0.35, 0.4)]
            int[] faceVertexCounts = [4, 4, 4, 4, 4, 4]
            int[] faceVertexIndices = [0,1,3,2, 4,6,7,5, 0,4,5,1, 2,3,7,6, 0,2,6,4, 1,5,7,3]
            point3f[] points = [(-1.8,0.3,0.4),(1.8,0.3,0.4),(-1.8,0.35,0.4),(1.8,0.35,0.4),(-1.8,0.3,-0.4),(1.8,0.3,-0.4),(-1.8,0.35,-0.4),(1.8,0.35,-0.4)]
        }

        def Mesh "Roller"
        {
            float3[] extent = [(-0.15, 0.05, -0.4), (0.15, 0.25, 0.4)]
            int[] faceVertexCounts = [4, 4, 4, 4, 4, 4]
            int[] faceVertexIndices = [0,1,3,2, 4,6,7,5, 0,4,5,1, 2,3,7,6, 0,2,6,4, 1,5,7,3]
            point3f[] points = [(-0.15,0.05,0.4),(0.15,0.05,0.4),(-0.15,0.25,0.4),(0.15,0.25,0.4),(-0.15,0.05,-0.4),(0.15,0.05,-0.4),(-0.15,0.25,-0.4),(0.15,0.25,-0.4)]
        }
    }

    def Xform "Fence_A"
    {
        def Mesh "Post"
        {
            float3[] extent = [(-0.05, 0, -0.05), (0.05, 1.2, 0.05)]
            int[] faceVertexCounts = [4, 4, 4, 4, 4, 4]
            int[] faceVertexIndices = [0,1,3,2, 4,6,7,5, 0,4,5,1, 2,3,7,6, 0,2,6,4, 1,5,7,3]
            point3f[] points = [(-0.05,0,0.05),(0.05,0,0.05),(-0.05,1.2,0.05),(0.05,1.2,0.05),(-0.05,0,-0.05),(0.05,0,-0.05),(-0.05,1.2,-0.05),(0.05,1.2,-0.05)]
        }

        def Mesh "Panel"
        {
            float3[] extent = [(0.05, 0.2, -0.01), (1.0, 1.0, 0.01)]
            int[] faceVertexCounts = [4, 4, 4, 4, 4, 4]
            int[] faceVertexIndices = [0,1,3,2, 4,6,7,5, 0,4,5,1, 2,3,7,6, 0,2,6,4, 1,5,7,3]
            point3f[] points = [(0.05,0.2,0.01),(1.0,0.2,0.01),(0.05,1.0,0.01),(1.0,1.0,0.01),(0.05,0.2,-0.01),(1.0,0.2,-0.01),(0.05,1.0,-0.01),(1.0,1.0,-0.01)]
        }
    }

    def Xform "Fence_B"
    {
        def Mesh "Post"
        {
            float3[] extent = [(-0.05, 0, -0.05), (0.05, 1.2, 0.05)]
            int[] faceVertexCounts = [4, 4, 4, 4, 4, 4]
            int[] faceVertexIndices = [0,1,3,2, 4,6,7,5, 0,4,5,1, 2,3,7,6, 0,2,6,4, 1,5,7,3]
            point3f[] points = [(-0.05,0,0.05),(0.05,0,0.05),(-0.05,1.2,0.05),(0.05,1.2,0.05),(-0.05,0,-0.05),(0.05,0,-0.05),(-0.05,1.2,-0.05),(0.05,1.2,-0.05)]
        }

        def Mesh "Panel"
        {
            float3[] extent = [(0.05, 0.2, -0.01), (1.0, 1.0, 0.01)]
            int[] faceVertexCounts = [4, 4, 4, 4, 4, 4]
            int[] faceVertexIndices = [0,1,3,2, 4,6,7,5, 0,4,5,1, 2,3,7,6, 0,2,6,4, 1,5,7,3]
            point3f[] points = [(0.05,0.2,0.01),(1.0,0.2,0.01),(0.05,1.0,0.01),(1.0,1.0,0.01),(0.05,0.2,-0.01),(1.0,0.2,-0.01),(0.05,1.0,-0.01),(1.0,1.0,-0.01)]
        }
    }
}
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(usda)
    return output_path


def create_payload_scene(output_dir: Path) -> tuple[Path, Path, Path]:
    """Create Scene C: payload DAG scene with leaf and parent payloads.

    Returns (scene_path, parent_payload_path, leaf_payload_path).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- payload_leaf.usda ---
    leaf_usda = """\
#usda 1.0
(
    defaultPrim = "Root"
    upAxis = "Y"
)

def Xform "Root"
{
    def Mesh "Bottom"
    {
        float3[] extent = [(-0.3, 0, -0.3), (0.3, 0.05, 0.3)]
        int[] faceVertexCounts = [4, 4, 4, 4, 4, 4]
        int[] faceVertexIndices = [0,1,3,2, 4,6,7,5, 0,4,5,1, 2,3,7,6, 0,2,6,4, 1,5,7,3]
        point3f[] points = [(-0.3,0,0.3),(0.3,0,0.3),(-0.3,0.05,0.3),(0.3,0.05,0.3),(-0.3,0,-0.3),(0.3,0,-0.3),(-0.3,0.05,-0.3),(0.3,0.05,-0.3)]
    }

    def Mesh "Side"
    {
        float3[] extent = [(-0.3, 0, -0.3), (-0.25, 0.2, 0.3)]
        int[] faceVertexCounts = [4, 4, 4, 4, 4, 4]
        int[] faceVertexIndices = [0,1,3,2, 4,6,7,5, 0,4,5,1, 2,3,7,6, 0,2,6,4, 1,5,7,3]
        point3f[] points = [(-0.3,0,0.3),(-0.25,0,0.3),(-0.3,0.2,0.3),(-0.25,0.2,0.3),(-0.3,0,-0.3),(-0.25,0,-0.3),(-0.3,0.2,-0.3),(-0.25,0.2,-0.3)]
    }
}
"""
    leaf_path = output_dir / "payload_leaf.usda"
    leaf_path.write_text(leaf_usda)

    # --- payload_parent.usda (references leaf via payload) ---
    parent_usda = f"""\
#usda 1.0
(
    defaultPrim = "Root"
    upAxis = "Y"
)

def Xform "Root"
{{
    def Mesh "Housing"
    {{
        float3[] extent = [(-0.5, 0, -0.5), (0.5, 0.8, 0.5)]
        int[] faceVertexCounts = [4, 4, 4, 4, 4, 4]
        int[] faceVertexIndices = [0,1,3,2, 4,6,7,5, 0,4,5,1, 2,3,7,6, 0,2,6,4, 1,5,7,3]
        point3f[] points = [(-0.5,0,0.5),(0.5,0,0.5),(-0.5,0.8,0.5),(0.5,0.8,0.5),(-0.5,0,-0.5),(0.5,0,-0.5),(-0.5,0.8,-0.5),(0.5,0.8,-0.5)]
    }}

    def Xform "TraySlot" (
        prepend payload = @{leaf_path.resolve()}@
    )
    {{
    }}
}}
"""
    parent_path = output_dir / "payload_parent.usda"
    parent_path.write_text(parent_usda)

    # --- payload_scene.usda (top-level scene, instances reference parent) ---
    scene_usda = f"""\
#usda 1.0
(
    defaultPrim = "Root"
    upAxis = "Y"
)

def Xform "Root"
{{
    def Mesh "Floor"
    {{
        float3[] extent = [(-5, -0.1, -5), (5, 0, 5)]
        int[] faceVertexCounts = [4, 4, 4, 4, 4, 4]
        int[] faceVertexIndices = [0,1,3,2, 4,6,7,5, 0,4,5,1, 2,3,7,6, 0,2,6,4, 1,5,7,3]
        point3f[] points = [(-5,-0.1,5),(5,-0.1,5),(-5,0,5),(5,0,5),(-5,-0.1,-5),(5,-0.1,-5),(-5,0,-5),(5,0,-5)]
    }}

    def Xform "Machine_A" (
        instanceable = true
        prepend payload = @{parent_path.resolve()}@
    )
    {{
    }}

    def Xform "Machine_B" (
        instanceable = true
        prepend payload = @{parent_path.resolve()}@
    )
    {{
    }}
}}
"""
    scene_path = output_dir / "payload_scene.usda"
    scene_path.write_text(scene_usda)

    return scene_path, parent_path, leaf_path


def create_material_library(output_dir: Path) -> tuple[Path, Path]:
    """Create a minimal material library with 4 entries.

    Returns (yaml_path, usd_path).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- test_materials.usd ---
    mat_usda = """\
#usda 1.0
(
    defaultPrim = "World"
    upAxis = "Y"
)

def Xform "World"
{
    def Scope "Looks"
    {
        def Material "Steel"
        {
            token outputs:surface.connect = </World/Looks/Steel/PBR.outputs:surface>

            def Shader "PBR"
            {
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor = (0.6, 0.6, 0.65)
                float inputs:metallic = 1.0
                float inputs:roughness = 0.3
                token outputs:surface
            }
        }

        def Material "Rubber"
        {
            token outputs:surface.connect = </World/Looks/Rubber/PBR.outputs:surface>

            def Shader "PBR"
            {
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor = (0.1, 0.1, 0.1)
                float inputs:metallic = 0.0
                float inputs:roughness = 0.9
                token outputs:surface
            }
        }

        def Material "Plastic"
        {
            token outputs:surface.connect = </World/Looks/Plastic/PBR.outputs:surface>

            def Shader "PBR"
            {
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor = (0.2, 0.4, 0.8)
                float inputs:metallic = 0.0
                float inputs:roughness = 0.5
                token outputs:surface
            }
        }

        def Material "Aluminum"
        {
            token outputs:surface.connect = </World/Looks/Aluminum/PBR.outputs:surface>

            def Shader "PBR"
            {
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor = (0.8, 0.8, 0.85)
                float inputs:metallic = 1.0
                float inputs:roughness = 0.2
                token outputs:surface
            }
        }
    }
}
"""
    usd_path = output_dir / "test_materials.usd"
    usd_path.write_text(mat_usda)

    # --- test_materials.yaml ---
    mat_yaml = f"""\
library_path: "{usd_path.name}"

entries:
  - name: "Steel"
    description: "Smooth metallic steel surface"
    binding: "/World/Looks/Steel"

  - name: "Rubber"
    description: "Dark matte rubber surface"
    binding: "/World/Looks/Rubber"

  - name: "Plastic"
    description: "Blue plastic surface"
    binding: "/World/Looks/Plastic"

  - name: "Aluminum"
    description: "Shiny aluminum surface"
    binding: "/World/Looks/Aluminum"
"""
    yaml_path = output_dir / "test_materials.yaml"
    yaml_path.write_text(mat_yaml)

    return yaml_path, usd_path
