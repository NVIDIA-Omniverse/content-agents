# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task: Apply PBR textures to materials in USD.

Supports per-material mode (shared texture) and per-prim mode (unique
texture per geometry prim via material cloning).
"""

from __future__ import annotations

import logging
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

from pxr import Sdf, Usd, UsdGeom, UsdShade
from world_understanding.agentic.tasks import Task

from texture_agent.functions.material_discovery import PrimTextureUnit
from texture_agent.tasks.blend_textures import BlendedTextures

logger = logging.getLogger(__name__)


def _cached_blended_texture_set(out_dir: Path, key: str) -> BlendedTextures | None:
    albedo = out_dir / f"{key}_albedo.png"
    normal = out_dir / f"{key}_normal.png"
    orm = out_dir / f"{key}_orm.png"
    if albedo.exists() and normal.exists() and orm.exists():
        return BlendedTextures(albedo=str(albedo), normal=str(normal), orm=str(orm))
    return None


def _load_cached_blended_textures(
    working_dir: Path,
    units: list[PrimTextureUnit],
) -> dict[str, BlendedTextures]:
    out_dir = working_dir / "textures"
    cached: dict[str, BlendedTextures] = {}
    for unit in units:
        textures = _cached_blended_texture_set(out_dir, unit.key)
        if textures:
            cached[unit.key] = textures
    return cached


def _clone_material(
    stage: Usd.Stage,
    source_mat_path: str,
    clone_name: str,
) -> str:
    """Clone a material prim (deep copy of entire shader subtree).

    Args:
        stage: The USD stage.
        source_mat_path: Path to the source material prim.
        clone_name: Name for the cloned material.

    Returns:
        Path to the cloned material prim.
    """
    parent_path = str(Sdf.Path(source_mat_path).GetParentPath())
    clone_path = f"{parent_path}/{clone_name}"

    layer = stage.GetRootLayer()
    Sdf.CopySpec(layer, source_mat_path, layer, clone_path)

    logger.debug("Cloned material: %s -> %s", source_mat_path, clone_path)
    return clone_path


def _set_texture_attr(
    prim: Usd.Prim,
    attr_name: str,
    texture_path: str,
) -> None:
    """Set an asset path attribute on a prim, creating if needed."""
    attr = prim.GetAttribute(attr_name)
    if attr and attr.IsValid():
        attr.Set(Sdf.AssetPath(texture_path))
    else:
        prim.CreateAttribute(attr_name, Sdf.ValueTypeNames.Asset).Set(
            Sdf.AssetPath(texture_path)
        )


def _set_tiledimage_file_input(
    stage: Usd.Stage,
    mat_path: str,
    shader_name: str,
    texture_path: str,
) -> None:
    """Set the concrete tiledimage shader input used by NVCF/OpenPBR."""
    shader_prim = stage.GetPrimAtPath(f"{mat_path}/{shader_name}")
    if not shader_prim.IsValid():
        logger.debug(
            "OpenPBR tiledimage shader not found: %s/%s", mat_path, shader_name
        )
        return

    if not shader_prim.IsA(UsdShade.Shader):
        logger.debug("Prim is not a UsdShade shader: %s", shader_prim.GetPath())
        return

    shader = UsdShade.Shader(shader_prim)
    file_input = shader.GetInput("file")
    if file_input:
        file_input.Set(Sdf.AssetPath(texture_path))
    else:
        shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
            Sdf.AssetPath(texture_path)
        )


# SimReady/OmniPBR MDL texture-input names → channel of the BlendedTextures bundle.
# Keys are lowercased so we can match case-insensitively (e.g. SimReady's
# "ORM_texture" alongside OmniPBR's "ORM_texture" and OmniPBR-derived
# "diffuse_texture").
_MDL_TEXTURE_INPUT_MAP = {
    "diffuse_texture": "albedo",
    "albedo_texture": "albedo",
    "base_color_texture": "albedo",
    "diffuse_color_texture": "albedo",
    "normalmap_texture": "normal",
    "normal_texture": "normal",
    "normal_map_texture": "normal",
    "orm_texture": "orm",
    "reflectionroughness_texture": "roughness",
    "roughness_texture": "roughness",
    "specular_roughness_texture": "roughness",
    "metallic_texture": "metalness",
    "metalness_texture": "metalness",
}


def _is_mdl_shader(prim: Usd.Prim) -> bool:
    if not prim.IsA(UsdShade.Shader):
        return False
    attr = prim.GetAttribute("info:mdl:sourceAsset")
    return bool(attr and attr.IsValid() and attr.HasAuthoredValue())


_UNBUNDLEABLE_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://")

# Channels whose generated PNG is also referenced via an `Sdf.AssetPath`-typed
# attribute on the Material prim (the OpenPBR write path). USDZ packaging only
# follows asset-typed dependencies; channels in this set are guaranteed to be
# bundled regardless of how the MDL Shader's input is typed. Channels NOT in
# this set (today: only ``orm`` — the packed ORM is not duplicated to an
# OpenPBR Asset attr) cannot be safely written into a string/token-typed MDL
# input, since the packager would rewrite the path but the file would never
# enter the downloaded archive.
_USDZ_BUNDLED_CHANNELS = frozenset({"albedo", "normal", "roughness", "metalness"})

# The MDL `*_texture` input types we know how to round-trip safely. Anything
# else (e.g. AssetArray, StringArray, custom typedefs) is left untouched —
# we'd rather skip a rare schema than emit a corrupted value.
_SUPPORTED_TEXTURE_INPUT_TYPES = frozenset(
    {Sdf.ValueTypeNames.Asset, Sdf.ValueTypeNames.String, Sdf.ValueTypeNames.Token}
)


def _is_unbundleable_asset_path(path: str) -> bool:
    """A texture path the public bundle cannot resolve at render time.

    Anything carrying a URI scheme (`omniverse://`, `http://`, `https://`, …)
    falls in this bucket — only callers with the matching asset resolver can
    fetch it, and the service's USDZ packager rewrites every `*.png` asset
    path to `../textures/<basename>` regardless of source, which would point
    at a file the bundle does not ship. Local relative or absolute paths are
    left alone — they're either already packageable or were placed there
    intentionally by the asset author.
    """
    if not path:
        return False
    return bool(_UNBUNDLEABLE_SCHEME_RE.match(path))


def _resolve_layer_anchored_path(
    attr: Usd.Attribute,
    raw: str,
    fallback_anchor: Path,
) -> Path | None:
    """Resolve a relative MDL asset path against the layer that authored it.

    Composed USDs can author shader inputs in a referenced or sublayered file,
    where ``@./opacity.png@`` is relative to *that* layer, not the root. Using
    the root USD's directory as the anchor (Codex round-5 finding) silently
    drops legitimate textures from referenced material libraries.

    Resolution order:

    1. Prefer the asset resolver's already-resolved path
       (``Sdf.AssetPath.resolvedPath``) when USD has populated it.
    2. Fall back to anchoring on the strongest authoring layer's directory
       (from the property stack).
    3. Fall back to ``fallback_anchor`` (the root USD's directory) when no
       layer-on-disk anchor is available (anonymous layers, in-memory stages).

    Errors during ``Path.resolve()`` (NUL bytes, invalid UTF-8, …) are caught
    so a malicious USD can't crash apply_textures.
    """
    val = attr.Get()
    if val is None:
        return None
    resolved = getattr(val, "resolvedPath", "") or ""
    if resolved:
        try:
            return Path(resolved).resolve()
        except (OSError, ValueError):
            return None

    anchor = fallback_anchor
    try:
        prop_stack = attr.GetPropertyStack(Usd.TimeCode.Default())
    except Exception:
        prop_stack = []
    if prop_stack:
        layer = prop_stack[0].layer
        layer_path = getattr(layer, "realPath", "") if layer else ""
        if layer_path:
            anchor = Path(layer_path).parent

    try:
        return (anchor / raw).resolve()
    except (OSError, ValueError):
        return None


def _localize_asset(
    candidate: Path,
    upload_root: Path,
    tex_dir: Path,
    mat_name: str,
    input_name: str,
) -> str | None:
    """Copy an already-resolved local asset into the bundle textures dir.

    Security: USD content can come from untrusted uploads, so we refuse to
    localize anything that resolves outside the upload root. Without this
    scope check a crafted MDL input like ``inputs:leak_texture = @/etc/passwd@``
    would copy a host file into ``working_dir/textures/`` — which the service
    exposes as a downloadable artifact. We additionally require the file to
    carry a (case-insensitive) ``.png`` suffix so a path with no extension or
    a non-PNG image can't slip into the bundle: the service packager and the
    textures-artifact ZIP only handle ``*.png`` (case-sensitive ``endswith``),
    so anything else would be silently dropped or inconsistently rewritten.

    The caller is responsible for resolving the raw asset path (including
    layer-anchored relative resolution); this function only enforces the
    security boundary and the copy.

    Returns the new local path inside ``tex_dir`` on success, or ``None`` if
    the source could not be resolved or copied — caller should fall back to
    clearing.
    """
    try:
        if not candidate.is_file():
            return None
    except (OSError, ValueError):
        return None

    # Reject anything outside the upload root — this is the trust boundary in
    # the service path. ``resolve()`` already followed symlinks before we got
    # here, so an in-upload symlink pointing at ``/etc/passwd`` would resolve
    # outside ``upload_root`` and be rejected here.
    try:
        upload_resolved = upload_root.resolve()
    except (OSError, ValueError):
        return None
    try:
        candidate.relative_to(upload_resolved)
    except ValueError:
        return None

    # Only PNG (case-insensitive). Non-PNG suffixes would not be rewritten by
    # the service packager (which matches lower-case ``.png``) and would not
    # make it into the textures-artifact ZIP (which globs ``*.png``), so
    # accepting them creates inconsistent bundles.
    if candidate.suffix.lower() != ".png":
        return None

    # Already inside the bundle textures dir → nothing to do.
    try:
        if candidate.parent.samefile(tex_dir):
            return str(candidate)
    except OSError:
        pass

    # Prefix with material+input to avoid collisions across materials sharing
    # a basename (`opacity.png`). Always emit lower-case ``.png`` so the
    # packager's ``endswith(".png")`` match (case-sensitive) succeeds.
    safe_mat = mat_name.replace("/", "_").lstrip("_") or "mat"
    target = tex_dir / f"{safe_mat}__{input_name}.png"

    tex_dir.mkdir(parents=True, exist_ok=True)
    try:
        if not target.exists() or not target.samefile(candidate):
            shutil.copyfile(candidate, target)
    except OSError as err:
        logger.warning(
            "Failed to localize MDL asset %s -> %s: %s", candidate, target, err
        )
        return None
    return str(target)


def _override_mdl_texture_inputs(
    stage: Usd.Stage,
    mat_path: str,
    channel_paths: dict[str, str],
    usd_path: str,
    working_dir: Path,
) -> tuple[int, list[str], list[str]]:
    """Overwrite MDL shader texture inputs in-place with bundle-local paths.

    SimReady/OmniPBR-style materials carry a child Shader with an `info:mdl:sourceAsset`
    and texture inputs like `inputs:normalmap_texture` / `inputs:ORM_texture`. The
    OpenPBR-style attributes the agent writes on the Material prim are not consumed
    by the MDL shader, so without this override the freshly generated textures are
    silently ignored at render time and the original (often Nucleus-hosted) refs
    survive into the output bundle.

    For unmapped authored `*_texture` inputs (e.g. `opacity_texture`,
    `emissive_color_texture`, `displacement_texture`) the rule is:

    * **URI-scheme paths** (`omniverse://...`, `http(s)://...`) are unbundleable
      — the public bundle's asset resolver cannot satisfy them and the service
      packager's `../textures/<basename>` rewrite would dangle. → cleared.
    * **Local paths that resolve to an existing file on disk** (relative to the
      input USD or absolute) are copied into ``working_dir/textures`` under a
      `<material>__<input>.<ext>` filename so the service packager's rewrite
      step finds them, and the input is rewritten to that local copy. →
      localized.
    * **Local paths that do not resolve** (the asset author's reference is
      already broken) are cleared.

    Clearing an unbundleable path drops back to the MDL's constant default,
    which renders correctly everywhere.

    Returns:
        (overridden_count, cleared_input_names, localized_input_names)
    """
    mat_prim = stage.GetPrimAtPath(mat_path)
    if not mat_prim.IsValid():
        return 0, [], []

    upload_root = Path(usd_path).resolve().parent
    tex_dir = working_dir / "textures"

    overridden = 0
    cleared: list[str] = []
    localized: list[str] = []
    mat_name = Path(mat_path).name
    for child in mat_prim.GetChildren():
        if not _is_mdl_shader(child):
            continue
        shader = UsdShade.Shader(child)
        for inp in shader.GetInputs():
            base = inp.GetBaseName()
            # MDL shaders can legally author ``inputs:*_texture`` as ``asset``,
            # ``string`` or ``token`` (Codex round-6/7 findings). Read the
            # current value as a plain string regardless, then write back using
            # the input's native type via ``_safe_set_typed_value`` so we
            # don't crash on a string-typed input nor silently leave a
            # Nucleus URL pointing at an unbundleable file.
            type_name = inp.GetTypeName()
            existing = _read_texture_input_string(inp, type_name)
            if existing is None:
                continue
            channel = _MDL_TEXTURE_INPUT_MAP.get(base.lower())

            if channel is not None:
                new_path = channel_paths.get(channel)
                if not new_path:
                    continue
                # Asset-typed mapped inputs always override. String/token
                # mapped inputs only override for channels that already have
                # a parallel Asset-typed dep on the Material — otherwise
                # USDZ packaging won't bundle the file (Codex round-9
                # finding: packed ORM is the canonical un-bundled channel).
                if (
                    type_name != Sdf.ValueTypeNames.Asset
                    and channel not in _USDZ_BUNDLED_CHANNELS
                ):
                    if _safe_set_typed_value(inp, type_name, ""):
                        cleared.append(f"{mat_path}:{base}")
                    continue
                if _safe_set_typed_value(inp, type_name, new_path):
                    overridden += 1
                continue

            if not base.lower().endswith("_texture"):
                continue
            if not existing:
                continue
            if _is_unbundleable_asset_path(existing):
                if _safe_set_typed_value(inp, type_name, ""):
                    cleared.append(f"{mat_path}:{base}")
                continue
            # Localization writes a copy into ``working_dir/textures`` and
            # rewrites the input to point at it. USDZ packaging only follows
            # ``Sdf.AssetPath``-typed dependencies, so localizing a
            # string/token-typed unmapped input would put the path in the
            # USD but never include the file in the downloaded archive
            # (Codex round-8 finding). For string/token unmapped inputs we
            # therefore clear instead — the MDL drops back to its constant
            # default, which renders correctly. Mapped channels above are
            # always safe because the OpenPBR Material attribute references
            # the same generated PNG via an Asset-typed dep that USDZ does
            # bundle.
            if type_name != Sdf.ValueTypeNames.Asset:
                if _safe_set_typed_value(inp, type_name, ""):
                    cleared.append(f"{mat_path}:{base}")
                continue
            candidate = _resolve_layer_anchored_path(
                inp.GetAttr(), existing, upload_root
            )
            copied = (
                _localize_asset(candidate, upload_root, tex_dir, mat_name, base)
                if candidate is not None
                else None
            )
            if copied is None:
                if _safe_set_typed_value(inp, type_name, ""):
                    cleared.append(f"{mat_path}:{base}")
            else:
                if _safe_set_typed_value(inp, type_name, copied):
                    localized.append(f"{mat_path}:{base}")

    return overridden, cleared, localized


def _read_texture_input_string(
    inp: UsdShade.Input, type_name: Sdf.ValueTypeName
) -> str | None:
    """Read an MDL texture input as a plain string, regardless of authored type.

    Returns the value's string form for ``Asset``/``String``/``Token``-typed
    inputs (the only types we know how to safely round-trip), or ``None`` for
    unauthored values, unsupported types, or array variants. ``None`` means
    "skip this input" — neither an override candidate nor a clear/localize
    candidate.
    """
    if type_name not in _SUPPORTED_TEXTURE_INPUT_TYPES:
        return None
    val = inp.Get()
    if val is None:
        return None
    if type_name == Sdf.ValueTypeNames.Asset:
        return val.path if hasattr(val, "path") else str(val)
    return str(val)


def _safe_set_typed_value(
    inp: UsdShade.Input, type_name: Sdf.ValueTypeName, value: str
) -> bool:
    """Write a string back into an MDL texture input using its authored type.

    USD content can come from untrusted uploads; an in-pipeline `Set` raising
    ``pxr.Tf.ErrorException`` would tear down the whole apply_textures step
    instead of skipping a single input. We log and return ``False`` on
    failure so the caller does not record the input in its stat list.

    Only the three texture-input types listed in
    ``_SUPPORTED_TEXTURE_INPUT_TYPES`` are accepted; anything else is a no-op
    and returns ``False``.
    """
    if type_name not in _SUPPORTED_TEXTURE_INPUT_TYPES:
        return False
    try:
        if type_name == Sdf.ValueTypeNames.Asset:
            inp.Set(Sdf.AssetPath(value))
        else:
            inp.Set(value)
        return True
    except Exception as err:
        logger.warning(
            "Failed to set MDL texture input %s = %r: %s",
            inp.GetAttr().GetPath(),
            value,
            err,
        )
        return False


def _apply_pbr_textures(
    stage: Usd.Stage,
    mat_path: str,
    textures: BlendedTextures,
    working_dir: Path,
    key: str,
    usd_path: str,
) -> tuple[int, list[str], list[str]]:
    """Apply albedo, normal, and ORM textures to a material prim.

    Returns:
        (mdl_inputs_overridden, mdl_inputs_cleared, mdl_inputs_localized)
    """
    prim = stage.GetPrimAtPath(mat_path)
    if not prim.IsValid():
        logger.warning("Material prim not found: %s", mat_path)
        return 0, [], []

    # Ensure parent Looks scope is defined for NVCF traversal
    parent = prim.GetParent()
    if parent.IsValid() and not parent.IsDefined():
        UsdGeom.Scope.Define(stage, parent.GetPath())

    channel_paths: dict[str, str] = {"albedo": textures.albedo}

    # Albedo
    _set_texture_attr(prim, "inputs:base_color_texture_file", textures.albedo)
    _set_tiledimage_file_input(
        stage,
        mat_path,
        "tiledimage_base_color",
        textures.albedo,
    )

    # Normal
    if textures.normal and Path(textures.normal).exists():
        _set_texture_attr(prim, "inputs:geometry_normal_texture_file", textures.normal)
        _set_tiledimage_file_input(
            stage,
            mat_path,
            "tiledimage_geometry_normal",
            textures.normal,
        )
        channel_paths["normal"] = textures.normal

    # ORM → unpack into roughness + metalness (and keep packed for MDL ORM_texture)
    if textures.orm and Path(textures.orm).exists():
        import numpy as np
        from PIL import Image

        channel_paths["orm"] = textures.orm

        orm_img = Image.open(textures.orm)
        orm_arr = np.array(orm_img)
        tex_dir = working_dir / "textures"

        roughness_arr = orm_arr[:, :, 1]
        roughness_path = tex_dir / f"{key}_roughness.png"
        Image.fromarray(roughness_arr).save(str(roughness_path))
        _set_texture_attr(
            prim, "inputs:specular_roughness_texture_file", str(roughness_path)
        )
        _set_tiledimage_file_input(
            stage,
            mat_path,
            "tiledimage_specular_roughness",
            str(roughness_path),
        )
        channel_paths["roughness"] = str(roughness_path)

        metalness_arr = orm_arr[:, :, 2]
        metalness_path = tex_dir / f"{key}_metalness.png"
        Image.fromarray(metalness_arr).save(str(metalness_path))
        _set_texture_attr(
            prim, "inputs:base_metalness_texture_file", str(metalness_path)
        )
        _set_tiledimage_file_input(
            stage,
            mat_path,
            "tiledimage_base_metalness",
            str(metalness_path),
        )
        channel_paths["metalness"] = str(metalness_path)

    return _override_mdl_texture_inputs(
        stage, mat_path, channel_paths, usd_path, working_dir
    )


class ApplyTexturesTask(Task):
    """Set PBR texture file paths on OpenPBR materials in the USD stage.

    In per-material mode: applies textures directly to shared materials.
    In per-prim mode: clones materials so each prim gets its own texture,
    then re-binds each geometry prim to its cloned material.

    For materials whose Material prim has an MDL Shader child (SimReady /
    OmniPBR), the task also overrides the well-known MDL texture inputs
    (`diffuse_texture`, `normalmap_texture`, `ORM_texture`,
    `reflectionroughness_texture`, `metallic_texture`, plus aliases) with
    the freshly generated local textures, and clears any unmapped
    `*_texture` input that points at an unbundleable URI (`omniverse://`,
    `http(s)://`, …) so the output USD does not survive into the
    downloaded bundle with refs the asset resolver cannot satisfy. Local
    relative/absolute paths on unmapped inputs are preserved.

    Context keys read:
        usd_path (str): Input USD file path.
        blended_textures (dict[str, BlendedTextures]): From BlendTexturesTask.
        prim_texture_units (list[PrimTextureUnit]): From DiscoverMaterialsTask.
        working_dir (str): Working directory.

    Context keys written:
        output_usd_paths (list[str]): Paths to output USD files.
        apply_textures_stats (dict): Summary of MDL-override activity:
            ``applied_count`` (int), ``mdl_inputs_overridden`` (int),
            ``mdl_inputs_cleared`` (list of ``"<mat_path>:<input_name>"``
            strings — unbundleable URI paths or unresolvable local refs that
            were blanked), and ``mdl_inputs_localized`` (list of strings —
            resolvable local refs that were copied into
            ``working_dir/textures`` so the bundle's path-rewrite step
            keeps them packageable). Consumed by the texture-agent service
            to surface a per-step warning in ``/status`` / ``/results``.
    """

    def __init__(self) -> None:
        self.name = "ApplyTextures"
        self.description = "Apply PBR texture maps to materials in USD"

    def run(self, context: dict[str, Any], object_store: Any = None) -> dict[str, Any]:
        usd_path = context["usd_path"]
        blended: dict[str, BlendedTextures] = context.get("blended_textures", {})
        units: list[PrimTextureUnit] = context.get("prim_texture_units", [])
        working_dir = Path(context["working_dir"])

        if not blended and context.get("resume"):
            blended = _load_cached_blended_textures(working_dir, units)
            if blended:
                logger.info(
                    "Loaded %d cached blended texture sets from %s",
                    len(blended),
                    working_dir / "textures",
                )
                context["blended_textures"] = blended

        if not blended:
            logger.info("No blended textures to apply")
            context["output_usd_paths"] = []
            return context

        out_dir = working_dir / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_usd_path = out_dir / "textured_output.usd"

        stage = Usd.Stage.Open(str(usd_path))
        if not stage:
            raise FileNotFoundError(f"Failed to open USD stage: {usd_path}")

        # Group units by material for cloning decisions
        units_by_material: dict[str, list[PrimTextureUnit]] = defaultdict(list)
        for unit in units:
            if unit.key in blended:
                units_by_material[unit.material_info.name].append(unit)

        applied_count = 0
        mdl_inputs_overridden = 0
        mdl_inputs_cleared: list[str] = []
        mdl_inputs_localized: list[str] = []

        for _mat_name, mat_units in units_by_material.items():
            mat = mat_units[0].material_info

            if len(mat_units) == 1 and not mat_units[0].prim_path:
                # Per-material mode (or single prim): apply directly
                unit = mat_units[0]
                overridden, cleared, localized = _apply_pbr_textures(
                    stage,
                    mat.prim_path,
                    blended[unit.key],
                    working_dir,
                    unit.key,
                    usd_path,
                )
                mdl_inputs_overridden += overridden
                mdl_inputs_cleared.extend(cleared)
                mdl_inputs_localized.extend(localized)
                logger.info("Applied textures to %s (direct)", unit.key)
                applied_count += 1

            else:
                # Per-prim mode: clone material for each prim
                for unit in mat_units:
                    clone_name = unit.key
                    clone_path = _clone_material(stage, mat.prim_path, clone_name)

                    # Apply textures to the clone
                    overridden, cleared, localized = _apply_pbr_textures(
                        stage,
                        clone_path,
                        blended[unit.key],
                        working_dir,
                        unit.key,
                        usd_path,
                    )
                    mdl_inputs_overridden += overridden
                    mdl_inputs_cleared.extend(cleared)
                    mdl_inputs_localized.extend(localized)

                    # Re-bind the geometry prim to the cloned material
                    if unit.prim_path:
                        geom_prim = stage.GetPrimAtPath(unit.prim_path)
                        if geom_prim.IsValid():
                            binding_api = UsdShade.MaterialBindingAPI.Apply(geom_prim)
                            cloned_mat = UsdShade.Material(
                                stage.GetPrimAtPath(clone_path)
                            )
                            binding_api.Bind(cloned_mat)
                            logger.info(
                                "Applied textures to %s (cloned, bound %s)",
                                unit.key,
                                unit.prim_path,
                            )
                        else:
                            logger.warning(
                                "Prim not found for rebinding: %s",
                                unit.prim_path,
                            )

                    applied_count += 1

        stage.GetRootLayer().Export(str(output_usd_path))
        logger.info(
            "Applied PBR textures to %d units, saved to %s",
            applied_count,
            output_usd_path,
        )
        if mdl_inputs_overridden:
            logger.info(
                "Overrode %d pre-baked MDL texture inputs with new local textures",
                mdl_inputs_overridden,
            )
        if mdl_inputs_cleared:
            logger.warning(
                "Cleared %d MDL texture inputs that had no matching generated "
                "channel (would have produced broken refs after bundle "
                "rewriting): %s",
                len(mdl_inputs_cleared),
                ", ".join(mdl_inputs_cleared),
            )
        if mdl_inputs_localized:
            logger.info(
                "Localized %d MDL texture inputs (copied existing local refs into "
                "the bundle textures dir): %s",
                len(mdl_inputs_localized),
                ", ".join(mdl_inputs_localized),
            )

        context["output_usd_paths"] = [str(output_usd_path)]
        context["apply_textures_stats"] = {
            "applied_count": applied_count,
            "mdl_inputs_overridden": mdl_inputs_overridden,
            "mdl_inputs_cleared": mdl_inputs_cleared,
            "mdl_inputs_localized": mdl_inputs_localized,
        }
        return context
