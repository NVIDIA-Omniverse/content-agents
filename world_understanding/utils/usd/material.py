# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""USD Material utilities for creating and binding MDL materials."""

import logging
import os
import re
from pathlib import Path
from urllib.parse import urlparse

from pxr import Sdf, Usd, UsdGeom, UsdShade

logger = logging.getLogger(__name__)
_NON_LOCAL_ASSET_SCHEMES = frozenset({"http", "https", "data"})
_OVRTX_PREVIEW_FALLBACK_SHADER_NAME = "OVRTXPreviewSurface"
_MATERIALX_OPENPBR_SHADER_ID = "ND_open_pbr_surface_surfaceshader"
_OPENPBR_DEFAULT_BASE_COLOR = (0.8, 0.8, 0.8)


def _output_has_connected_source(output: UsdShade.Output) -> bool:
    try:
        sources, _ = output.GetConnectedSources()
    except Exception:
        return False
    return bool(sources)


def _material_has_connected_surface(
    material: UsdShade.Material,
    render_context: str = "",
) -> bool:
    output = material.GetSurfaceOutput(render_context)
    return bool(output and _output_has_connected_source(output))


def _connected_materialx_openpbr_surface(
    material: UsdShade.Material,
) -> UsdShade.Shader | None:
    output = material.GetSurfaceOutput("mtlx")
    if not output:
        return None

    try:
        sources, _ = output.GetConnectedSources()
    except Exception:
        return None

    for source_info in sources:
        source = source_info.source
        if not source:
            continue
        shader = UsdShade.Shader(source.GetPrim())
        if not shader:
            continue
        shader_id_attr = shader.GetIdAttr()
        if shader_id_attr and shader_id_attr.Get() == _MATERIALX_OPENPBR_SHADER_ID:
            return shader
    return None


def _float_material_input(
    material_prim: Usd.Prim,
    input_name: str,
    default: float,
) -> float:
    attr = material_prim.GetAttribute(f"inputs:{input_name}")
    if not attr:
        return default
    value = attr.Get()
    if value is None:
        return default
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _material_input_value(material_prim: Usd.Prim, input_name: str) -> object | None:
    attr = material_prim.GetAttribute(f"inputs:{input_name}")
    if not attr:
        return None
    value: object | None = attr.Get()
    return value


def _iter_materialx_openpbr_fallback_prims(stage: Usd.Stage) -> list[Usd.Prim]:
    fallback_prims: list[Usd.Prim] = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdShade.Material):
            continue

        material = UsdShade.Material(prim)
        if _material_has_connected_surface(material) or _material_has_connected_surface(
            material,
            "mdl",
        ):
            continue
        if _connected_materialx_openpbr_surface(material) is None:
            continue
        fallback_prims.append(prim)
    return fallback_prims


def _iter_materialx_openpbr_surface_prims(stage: Usd.Stage) -> list[Usd.Prim]:
    surface_prims: list[Usd.Prim] = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdShade.Material):
            continue
        if _connected_materialx_openpbr_surface(UsdShade.Material(prim)) is None:
            continue
        surface_prims.append(prim)
    return surface_prims


def _prepare_material_for_surface_authoring(material: UsdShade.Material) -> bool:
    prim = material.GetPrim()
    if not prim or not prim.IsValid():
        return False
    if prim.IsInstanceProxy():
        return False
    if prim.IsInstance() or prim.IsInstanceable():
        prim.SetInstanceable(False)
    return True


def _suppress_materialx_surface(material: UsdShade.Material) -> bool:
    if not _prepare_material_for_surface_authoring(material):
        return False
    material.CreateSurfaceOutput("mtlx").GetAttr().SetConnections([])
    return True


def _author_ovrtx_preview_fallback(
    target_stage: Usd.Stage,
    material_path: str,
    source_material_prim: Usd.Prim,
    *,
    suppress_materialx_surface: bool = False,
) -> bool:
    source_is_instance = (
        source_material_prim.IsInstance() or source_material_prim.IsInstanceable()
    )
    target_prim = target_stage.GetPrimAtPath(material_path)
    if target_prim.IsValid():
        if target_prim.IsInstanceProxy():
            return False
        if target_prim.IsInstance() or target_prim.IsInstanceable():
            target_prim.SetInstanceable(False)

    material = UsdShade.Material.Define(target_stage, material_path)
    if source_is_instance:
        material.GetPrim().SetInstanceable(False)
    if not _prepare_material_for_surface_authoring(material):
        return False

    shader = UsdShade.Shader.Define(
        target_stage,
        f"{material_path}/{_OVRTX_PREVIEW_FALLBACK_SHADER_NAME}",
    )
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(0)

    base_color = _material_input_value(source_material_prim, "base_color")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        base_color if base_color is not None else _OPENPBR_DEFAULT_BASE_COLOR,
    )

    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(
        _float_material_input(source_material_prim, "base_metalness", 0.0),
    )
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(
        _float_material_input(source_material_prim, "specular_roughness", 0.5),
    )
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(
        _float_material_input(source_material_prim, "geometry_opacity", 1.0),
    )

    shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(
        shader.ConnectableAPI(),
        "surface",
    )

    if suppress_materialx_surface:
        # OVRTX can prefer the MaterialX render context over the universal
        # UsdPreviewSurface output and fall back to its red error shader.
        _suppress_materialx_surface(material)
    return True


def add_ovrtx_preview_fallbacks_for_materialx_openpbr(
    stage: Usd.Stage,
    *,
    suppress_materialx_surface: bool = False,
) -> int:
    """Add temporary UsdPreviewSurface fallbacks for OVRTX MaterialX rendering.

    OVRTX can resolve ordinary universal surface outputs and MDL render
    contexts, but the bundled OpenPBR MaterialX library only authors
    ``outputs:mtlx:surface``. For render-only exports, synthesize a lightweight
    preview shader from the material's direct OpenPBR constants and connect it
    to ``outputs:surface``. The original MaterialX network remains intact unless
    ``suppress_materialx_surface`` is enabled for a render-only export.

    Returns the number of Material prims that were updated.
    """
    updated = 0
    fallback_prims = _iter_materialx_openpbr_fallback_prims(stage)
    fallback_paths = {str(prim.GetPath()) for prim in fallback_prims}

    for prim in fallback_prims:
        if _author_ovrtx_preview_fallback(
            stage,
            str(prim.GetPath()),
            prim,
            suppress_materialx_surface=suppress_materialx_surface,
        ):
            updated += 1

    if suppress_materialx_surface:
        for prim in _iter_materialx_openpbr_surface_prims(stage):
            if str(prim.GetPath()) in fallback_paths:
                continue

            material = UsdShade.Material(prim)
            if not _material_has_connected_surface(material):
                continue

            if _suppress_materialx_surface(material):
                updated += 1

    return updated


def write_ovrtx_preview_fallback_overlay_for_materialx_openpbr(
    stage: Usd.Stage,
    overlay_path: str | Path,
    *,
    suppress_materialx_surface: bool = True,
) -> int:
    """Write a stronger overlay with OVRTX preview fallbacks for a composed stage.

    This covers materials that live in sublayers or references while keeping the
    source stage and its authored material libraries untouched. By default the
    overlay also blocks the MaterialX surface connection so OVRTX must use the
    preview fallback instead of its red error shader path.
    """
    fallback_prims = _iter_materialx_openpbr_fallback_prims(stage)
    fallback_paths = {str(prim.GetPath()) for prim in fallback_prims}
    suppress_only_prims: list[Usd.Prim] = []
    if suppress_materialx_surface:
        for prim in _iter_materialx_openpbr_surface_prims(stage):
            if str(prim.GetPath()) in fallback_paths:
                continue

            material = UsdShade.Material(prim)
            if not _material_has_connected_surface(material):
                continue

            suppress_only_prims.append(prim)

    if not fallback_prims and not suppress_only_prims:
        return 0

    path = Path(overlay_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    overlay_stage = Usd.Stage.CreateNew(str(path))
    updated = 0
    for prim in fallback_prims:
        if _author_ovrtx_preview_fallback(
            overlay_stage,
            str(prim.GetPath()),
            prim,
            suppress_materialx_surface=suppress_materialx_surface,
        ):
            updated += 1
    for prim in suppress_only_prims:
        material = UsdShade.Material.Define(overlay_stage, str(prim.GetPath()))
        if prim.IsInstance() or prim.IsInstanceable():
            material.GetPrim().SetInstanceable(False)
        if _suppress_materialx_surface(material):
            updated += 1
    overlay_stage.GetRootLayer().Save()
    return updated


def add_ovrtx_preview_fallbacks_to_stage_file(
    stage_path: str | Path,
    *,
    suppress_materialx_surface: bool = True,
) -> int:
    """Open a USD file, add OVRTX preview fallbacks, and save it if changed."""
    stage = Usd.Stage.Open(str(stage_path))
    if stage is None:
        return 0

    added = add_ovrtx_preview_fallbacks_for_materialx_openpbr(
        stage,
        suppress_materialx_surface=suppress_materialx_surface,
    )
    if added:
        stage.GetRootLayer().Save()
    return added


def _path_leaf_name(path: str | Sdf.Path) -> str:
    return Sdf.Path(str(path)).name


def ensure_looks_scope_spec(
    layer: Sdf.Layer,
    prim_path: str | Sdf.Path,
    *,
    allow_over: bool = False,
) -> None:
    """Type an existing untyped ``Looks`` prim spec as ``Scope``.

    The caller owns creating the prim spec and choosing its specifier
    (``def`` vs ``over``); this only authors the schema type when missing and
    does not repair a non-empty non-``Scope`` type. ``over`` specs require
    explicit opt-in because an ``over Scope`` can override a composed type.
    """
    if _path_leaf_name(prim_path) != "Looks":
        return

    prim_spec = layer.GetPrimAtPath(str(prim_path))
    if (
        prim_spec
        and not prim_spec.typeName
        and (allow_over or prim_spec.specifier != Sdf.SpecifierOver)
    ):
        prim_spec.typeName = "Scope"


def _author_looks_scope_type(stage: Usd.Stage, prim_path: Sdf.Path) -> None:
    """Author a ``Scope`` type opinion without changing the prim specifier."""
    layer = stage.GetEditTarget().GetLayer()
    if layer.GetPrimAtPath(str(prim_path)) is None:
        Sdf.CreatePrimInLayer(layer, prim_path)
    ensure_looks_scope_spec(layer, prim_path, allow_over=True)


def ensure_looks_scope(stage: Usd.Stage, material_path: str | Sdf.Path) -> None:
    """Type an untyped ``Looks`` ancestor of a material path as ``Scope``.

    This intentionally normalizes only the conventional material container,
    not arbitrary intermediate grouping prims. Missing ``Looks`` containers are
    created as ``def Scope``; existing untyped prims only receive a ``Scope``
    type opinion in the current edit layer.
    """
    material_path_str = str(material_path)
    if not material_path_str:
        return

    path = Sdf.Path(material_path_str)
    if not path.IsAbsolutePath():
        return

    parent_path = path.GetParentPath()
    while parent_path != Sdf.Path.absoluteRootPath:
        if _path_leaf_name(parent_path) == "Looks":
            parent_prim = stage.GetPrimAtPath(parent_path)
            if not parent_prim.IsValid():
                UsdGeom.Scope.Define(stage, parent_path)
            elif parent_prim.GetTypeName() == "":
                _author_looks_scope_type(stage, parent_path)
        parent_path = parent_path.GetParentPath()


def _resolve_local_asset_path(
    asset_val: object,
    authored_path: str,
    base_dir: Path,
) -> tuple[str | None, bool]:
    """Resolve a local USD asset path, preferring USD's authored-layer result."""
    resolved_path = str(getattr(asset_val, "resolvedPath", "") or "")
    if resolved_path and _safe_exists(resolved_path):
        return str(Path(resolved_path).resolve()), True

    if os.path.isabs(authored_path):
        if _safe_exists(authored_path):
            return str(Path(authored_path).resolve()), True
        return None, False

    candidate = base_dir / authored_path
    if _safe_exists(candidate):
        return str(candidate.resolve()), True
    return None, False


def _safe_exists(path: str | Path) -> bool:
    """Return whether a path exists without surfacing invalid path errors."""
    try:
        return Path(path).exists()
    except (OSError, ValueError):
        return False


def _is_non_local_asset_uri(asset_path: str) -> bool:
    """Return true for asset URI values that should not be treated as files."""
    return urlparse(asset_path).scheme.lower() in _NON_LOCAL_ASSET_SCHEMES


def get_local_mdl_assets(
    stage: Usd.Stage, base_dir: str | Path | None = None
) -> list[dict]:
    """Get all local MDL sourceAsset paths from the stage.

    This function traverses the stage to find all Shader prims with MDL
    sourceAsset attributes and returns information about each one. It
    resolves paths to determine which are local files that need bundling.

    Args:
        stage: USD stage to scan for MDL materials
        base_dir: Fallback base directory for resolving relative paths when
            USD does not provide ``Sdf.AssetPath.resolvedPath``. If None, uses
            the stage's root layer directory.

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

        # Check if it's a remote or embedded URI - skip these
        if _is_non_local_asset_uri(mdl_path):
            mdl_assets.append(
                {
                    "shader_path": str(prim.GetPath()),
                    "mdl_path": mdl_path,
                    "resolved_path": None,
                    "is_local": False,
                }
            )
            continue

        resolved_path, is_local = _resolve_local_asset_path(
            asset_val,
            mdl_path,
            base_dir,
        )

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
        base_dir: Fallback base directory for resolving relative paths when
            USD does not provide ``Sdf.AssetPath.resolvedPath``. If None, uses
            the stage's root layer directory.

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

            # Skip remote or embedded URIs before treating the value as a path.
            if _is_non_local_asset_uri(file_path):
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

            # Check if extension is a known texture format
            ext = Path(file_path).suffix.lower()
            if ext not in _TEXTURE_EXTENSIONS:
                continue

            resolved_path, is_local = _resolve_local_asset_path(
                asset_val,
                file_path,
                base_dir,
            )

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
