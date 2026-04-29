# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""USD Material utilities for creating and binding MDL materials."""

import logging
import os
import re
from pathlib import Path

from pxr import Sdf, Usd, UsdGeom, UsdShade

logger = logging.getLogger(__name__)


def get_local_mdl_assets(
    stage: Usd.Stage, base_dir: str | Path | None = None
) -> list[dict]:
    """Get all local MDL sourceAsset paths from the stage.

    This function traverses the stage to find all Shader prims with MDL
    sourceAsset attributes and returns information about each one. It
    resolves paths to determine which are local files that need bundling.

    Args:
        stage: USD stage to scan for MDL materials
        base_dir: Base directory for resolving relative paths. If None,
                 uses the stage's root layer directory.

    Returns:
        List of dicts, each containing:
            - shader_path: SdfPath string to the shader prim
            - mdl_path: Original MDL path as stored in the attribute
            - resolved_path: Resolved absolute path to MDL file, or None if:
                - Path is a remote URL (http/https)
                - File doesn't exist locally
            - is_local: True if the file exists locally
    """
    if base_dir is None:
        # Use the root layer's directory as base
        root_layer = stage.GetRootLayer()
        if root_layer.realPath:
            base_dir = Path(root_layer.realPath).parent
        else:
            base_dir = Path.cwd()
    else:
        base_dir = Path(base_dir)

    mdl_assets = []

    for prim in stage.Traverse():
        # Check if it's a Shader prim
        if not prim.IsA(UsdShade.Shader):
            continue

        # Look for MDL sourceAsset attribute
        mdl_attr = prim.GetAttribute("info:mdl:sourceAsset")
        if not mdl_attr or not mdl_attr.IsValid():
            continue

        asset_val = mdl_attr.Get()
        if asset_val is None:
            continue

        # Get the path from Sdf.AssetPath
        try:
            mdl_path = asset_val.path if hasattr(asset_val, "path") else str(asset_val)
        except Exception:
            mdl_path = str(asset_val)

        if not mdl_path:
            continue

        # Check if it's a remote URL - skip these
        if mdl_path.startswith(("http://", "https://")):
            mdl_assets.append(
                {
                    "shader_path": str(prim.GetPath()),
                    "mdl_path": mdl_path,
                    "resolved_path": None,
                    "is_local": False,
                }
            )
            continue

        # Resolve the path
        resolved_path = None
        is_local = False

        if os.path.isabs(mdl_path):
            # Absolute path - check if file exists
            if os.path.exists(mdl_path):
                resolved_path = mdl_path
                is_local = True
        else:
            # Relative path - resolve from base_dir
            candidate = base_dir / mdl_path
            if candidate.exists():
                resolved_path = str(candidate.resolve())
                is_local = True

        mdl_assets.append(
            {
                "shader_path": str(prim.GetPath()),
                "mdl_path": mdl_path,
                "resolved_path": resolved_path,
                "is_local": is_local,
            }
        )

    return mdl_assets


def get_unique_mdl_directories(mdl_assets: list[dict]) -> list[Path]:
    """Get unique directories containing local MDL files.

    MDL materials often have texture files in the same directory,
    so we need to copy the entire directory, not just the MDL file.

    Args:
        mdl_assets: List of MDL asset dicts from get_local_mdl_assets()

    Returns:
        List of unique directory Paths containing local MDL files
    """
    directories = set()

    for asset in mdl_assets:
        if asset["is_local"] and asset["resolved_path"]:
            mdl_file = Path(asset["resolved_path"])
            directories.add(mdl_file.parent)

    return list(directories)


# Image file extensions recognized as texture files
_TEXTURE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".exr", ".tga", ".hdr", ".bmp"}


def get_local_texture_file_assets(
    stage: Usd.Stage, base_dir: str | Path | None = None
) -> list[dict]:
    """Get all local texture file asset paths from the stage.

    This function traverses the stage to find all prims with Sdf.AssetPath-typed
    attributes pointing to image files (PNG, JPG, EXR, TGA, HDR, BMP). It catches
    both direct ``inputs:file`` on UsdUVTexture shaders and texture paths on
    Material prims (e.g. ``inputs:DiffuseTexture``) — important because after
    ``duplicate_stage()`` flattening, paths may live on Material prims.

    Args:
        stage: USD stage to scan for texture references
        base_dir: Base directory for resolving relative paths. If None,
                 uses the stage's root layer directory.

    Returns:
        List of dicts (deduplicated by resolved_path), each containing:
            - prim_path: SdfPath string to the prim
            - attr_name: Name of the attribute containing the texture path
            - file_path: Original file path as stored in the attribute
            - resolved_path: Resolved absolute path to the texture file, or None
            - is_local: True if the file exists locally
    """
    if base_dir is None:
        root_layer = stage.GetRootLayer()
        if root_layer.realPath:
            base_dir = Path(root_layer.realPath).parent
        else:
            base_dir = Path.cwd()
    else:
        base_dir = Path(base_dir)

    texture_assets: list[dict] = []
    seen_resolved: set[str] = set()

    for prim in stage.Traverse():
        for attr in prim.GetAttributes():
            type_name = attr.GetTypeName()
            if type_name.type.typeName != "SdfAssetPath":
                continue

            asset_val = attr.Get()
            if asset_val is None:
                continue

            try:
                file_path = (
                    asset_val.path if hasattr(asset_val, "path") else str(asset_val)
                )
            except Exception:
                file_path = str(asset_val)

            if not file_path:
                continue

            # Check if extension is a known texture format
            ext = Path(file_path).suffix.lower()
            if ext not in _TEXTURE_EXTENSIONS:
                continue

            # Skip remote URLs
            if file_path.startswith(("http://", "https://")):
                texture_assets.append(
                    {
                        "prim_path": str(prim.GetPath()),
                        "attr_name": attr.GetName(),
                        "file_path": file_path,
                        "resolved_path": None,
                        "is_local": False,
                    }
                )
                continue

            # Resolve the path
            resolved_path = None
            is_local = False

            if os.path.isabs(file_path):
                if os.path.exists(file_path):
                    resolved_path = str(Path(file_path).resolve())
                    is_local = True
            else:
                candidate = base_dir / file_path
                if candidate.exists():
                    resolved_path = str(candidate.resolve())
                    is_local = True

            # Deduplicate by resolved_path
            if resolved_path and resolved_path in seen_resolved:
                continue
            if resolved_path:
                seen_resolved.add(resolved_path)

            texture_assets.append(
                {
                    "prim_path": str(prim.GetPath()),
                    "attr_name": attr.GetName(),
                    "file_path": file_path,
                    "resolved_path": resolved_path,
                    "is_local": is_local,
                }
            )

    return texture_assets


def add_mdl_material(
    stage: Usd.Stage,
    material_name: str,
    source_asset_path: str,
    sub_identifier: str = "OmniSurface",
    path_prefix: str = None,
    color: str | None = None,
) -> tuple[Usd.Stage, str]:
    """Add MDL material to a USD stage.

    Args:
        stage: The USD stage to add the material to
        material_name: Name for the material prim (should be sanitized for use as USD prim name,
                      with spaces, slashes, and dashes replaced with underscores)
        source_asset_path: Path to the MDL source asset
        sub_identifier: MDL subidentifier (typically the material name within the MDL)
        path_prefix: Optional path prefix for the material location (defaults to DefaultPrim/Looks)
        color: Optional hex color value for material modification (not yet implemented)

    Returns:
        Tuple of (updated stage, material_path)
    """
    if not path_prefix:
        default_prim = stage.GetDefaultPrim()
        if default_prim.IsValid():
            path_prefix = str(default_prim.GetPath())
        else:
            # Default prim is invalid (not set or stale after optimization).
            # Fall back to the first root-level prim so materials are created
            # under the actual scene root instead of at the stage root.
            root_children = list(stage.GetPseudoRoot().GetChildren())
            if root_children:
                path_prefix = str(root_children[0].GetPath())
                logger.warning(
                    f"Default prim is invalid, using root prim "
                    f"'{root_children[0].GetName()}' for material placement"
                )
            else:
                path_prefix = ""
                logger.warning(
                    "No default prim or root prims found, "
                    "creating materials at stage root"
                )
    path_prefix += "/Looks"

    UsdGeom.Scope.Define(stage, path_prefix)
    material_path = path_prefix + "/" + material_name
    material = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, material_path + "/Shader")

    # Apply NodeDefAPI schema to the shader prim for proper Omniverse compatibility
    shader_prim = shader.GetPrim()
    node_def_api = UsdShade.NodeDefAPI.Apply(shader_prim)

    # Set the implementation source and MDL asset information
    node_def_api.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
    node_def_api.SetSourceAsset(Sdf.AssetPath(source_asset_path), "mdl")
    node_def_api.SetSourceAssetSubIdentifier(sub_identifier, "mdl")

    # Connect shader to material outputs
    material.CreateSurfaceOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
    material.CreateDisplacementOutput("mdl").ConnectToSource(
        shader.ConnectableAPI(), "out"
    )
    material.CreateVolumeOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")

    # TODO: implement something that modifies the material based on hex color value
    if color is not None:
        pass

    return stage, material_path


def bind_material_to_prim(
    stage: Usd.Stage,
    material_path: str,
    prim_path: str,
    binding_strength: UsdShade.Tokens = UsdShade.Tokens.weakerThanDescendants,
) -> Usd.Stage:
    """Bind material to a prim.

    Args:
        stage: The USD stage
        material_path: Path to the material prim
        prim_path: Path to the prim to assign the material to
        binding_strength: Material binding strength (default: weakerThanDescendants)

    Returns:
        Updated stage

    Raises:
        ValueError: If the prim is an instance proxy (read-only)
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        logger.warning(f"Prim not found at path: {prim_path}")
        return stage

    # Instance proxies are READ-ONLY in USD - cannot author properties to them
    # Skip with a warning rather than failing the entire operation
    if prim.IsInstanceProxy():
        raise ValueError(
            f"Cannot bind material to instance proxy at {prim_path}. "
            "Instance proxies are read-only. Apply materials to the prototype instead."
        )

    material = UsdShade.Material(stage.GetPrimAtPath(material_path))

    try:
        # CRITICAL: Apply the MaterialBindingAPI schema to the prim before binding
        # This ensures the binding relationship is properly authored with the schema applied
        binding_api = UsdShade.MaterialBindingAPI.Apply(prim)
        binding_api.Bind(material, bindingStrength=binding_strength)
    except Exception as e:
        # Error: authoring to an instance proxy is not allowed
        logger.warning(f"Binding materials failed for {prim_path}: {e}")

    return stage


# Regex to strip triplanar channel suffix (_a, _b, _c) from input names
_TRIPLANAR_SUFFIX_RE = re.compile(r"_[abc]$")


def convert_custom_mdl_to_builtin(stage: Usd.Stage) -> None:
    """Replace custom MDL shader references with built-in equivalents.

    The NVCF renderer cannot load custom MDL modules. This converts:
    - CreativePBRTriplanar.mdl -> OmniPBR.mdl (with input name remapping)
    - ./Material/OmniPBR.mdl  -> OmniPBR.mdl  (fix relative path)

    Args:
        stage: USD stage to modify in-place.
    """
    for prim in stage.Traverse():
        if not prim.IsA(UsdShade.Shader):
            continue

        mdl_attr = prim.GetAttribute("info:mdl:sourceAsset")
        if not mdl_attr or not mdl_attr.IsValid():
            continue

        mdl_val = mdl_attr.Get()
        if mdl_val is None:
            continue
        mdl_path = mdl_val.path

        # Fix local OmniPBR path -> bare name
        if mdl_path.endswith("/OmniPBR.mdl") or mdl_path.endswith("\\OmniPBR.mdl"):
            mdl_attr.Set(Sdf.AssetPath("OmniPBR.mdl"))
            continue

        # CreativePBRTriplanar -> OmniPBR
        if "CreativePBRTriplanar" not in mdl_path:
            continue

        mdl_attr.Set(Sdf.AssetPath("OmniPBR.mdl"))
        sub_attr = prim.GetAttribute("info:mdl:sourceAsset:subIdentifier")
        if sub_attr and sub_attr.IsValid():
            sub_attr.Set("OmniPBR")

        # Remap inputs: strip the triplanar channel suffix (_a, _b, _c)
        shader = UsdShade.Shader(prim)
        for inp in shader.GetInputs():
            old_name = inp.GetBaseName()
            new_name = _TRIPLANAR_SUFFIX_RE.sub("", old_name)
            if new_name == old_name:
                continue

            val = inp.Get()
            if val is None:
                continue
            new_inp = shader.GetInput(new_name)
            if not new_inp:
                new_inp = shader.CreateInput(new_name, inp.GetTypeName())
            new_inp.Set(val)
