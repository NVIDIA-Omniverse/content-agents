# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-asset config generation for large-scene pipeline.

Deep-copies the scene config template, injects prim_path scoping,
and forces layer_only output for each sub-asset.
"""

from __future__ import annotations

import copy
import logging
import os
import re
from pathlib import Path

import yaml

from .manifest import PayloadGroup, SceneManifest, SubAsset

logger = logging.getLogger(__name__)


def generate_sub_asset_config(
    sub_asset: SubAsset,
    scene_config: dict,
    output_path: Path,
    scene_config_dir: Path | None = None,
    session_id: str | None = None,
) -> Path:
    """Generate a per-asset config from the scene config template.

    The generated config:
    - Rebases all relative paths so they resolve correctly from output_path
    - Sets input.prim_path to scope the pipeline to the sub-asset
    - Forces output.layer_only = true and output.flatten_output = false
    - Sets project.name and session_id to a sanitized sub-asset name
    - Removes the scene section (not needed for per-asset runs)

    Args:
        sub_asset: The sub-asset to generate config for.
        scene_config: The scene-level config dict (will be deep-copied).
        output_path: Where to write the generated YAML.
        scene_config_dir: Directory of the original scene config (for rebasing
            relative paths). If None, paths are kept as-is.

    Returns:
        Path to the generated config file.
    """
    config = copy.deepcopy(scene_config)

    # Remove scene section (not relevant for per-asset pipeline)
    config.pop("scene", None)

    # Set project identity — use caller-supplied session_id if provided
    # (ensures uniqueness when multiple sub-assets share the same name)
    safe_name = session_id if session_id else _sanitize_name(sub_asset.name)
    project = config.setdefault("project", {})
    project["name"] = safe_name
    project["session_id"] = safe_name

    # Rebase relative paths from scene config dir to generated config dir
    if scene_config_dir is not None:
        output_dir = output_path.parent
        _rebase_paths(config, scene_config_dir.resolve(), output_dir.resolve())

    input_section = config.setdefault("input", {})

    # Use extracted USD if available (much smaller than full scene).
    # The extracted USD already contains only this sub-asset's subtree,
    # so prim_path scoping is still needed for the pipeline to find
    # the correct prims within the preserved hierarchy.
    if sub_asset.extracted_usd:
        extracted_path = Path(sub_asset.extracted_usd)
        if extracted_path.exists():
            try:
                rel = os.path.relpath(
                    extracted_path.resolve(), output_path.parent.resolve()
                )
            except ValueError:
                rel = str(extracted_path.resolve())
            input_section["usd_path"] = rel
            logger.info(f"  Using extracted USD: {rel} (instead of full scene)")

    # Set prim_path scoping — needed even with extracted USD because
    # the extracted file preserves the full prim hierarchy
    input_section["prim_path"] = sub_asset.prim_path

    # Force layer-only output (material layer for composition)
    output_section = config.setdefault("output", {})
    output_section["layer_only"] = True
    output_section["flatten_output"] = False

    # Configure per-asset steps for scene mode.
    # The collect step handles unified apply against the master scene,
    # so per-asset apply and render are disabled.
    steps = config.get("steps", {})

    # Disable apply (collect step handles unified apply)
    apply_config = steps.get("apply", {})
    apply_config["enabled"] = False
    apply_config["layer_only"] = True
    apply_config["flatten_output"] = False
    steps["apply"] = apply_config

    # Disable render (collect step handles scene-level render)
    render_config = steps.get("render", {})
    render_config["enabled"] = False
    steps["render"] = render_config

    # Enable restore_usd so predictions use original topology paths (not
    # the SO-optimized ones).  This ensures prim paths in predictions match
    # the base scene and resolve correctly in the unified collect apply.
    restore_config = steps.get("restore_usd", {})
    restore_config["enabled"] = True
    steps["restore_usd"] = restore_config

    # Inject split context into VLM prompt if this asset was produced by
    # splitting a larger container — gives the VLM global context about
    # where this asset fits in the scene hierarchy.
    if sub_asset.split_context:
        _inject_split_context(config, sub_asset)

    # Write config
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Generated config for '{sub_asset.name}': {output_path}")
    return output_path


# Path keys in the config that contain file/directory paths needing rebasing
_PATH_KEYS = frozenset(
    {
        "usd_path",
        "path",
        "working_dir",
    }
)

_PATH_LIST_KEYS = frozenset(
    {
        "reference_images",
        "reference_pdfs",
    }
)


def _rebase_paths(config: dict, old_base: Path, new_base: Path) -> None:
    """Rebase relative paths in config from old_base to new_base (in-place).

    Walks the config dict recursively. For known path keys, converts
    relative paths so they resolve to the same absolute location from
    the new base directory.
    """
    for key, value in config.items():
        if isinstance(value, dict):
            _rebase_paths(value, old_base, new_base)
        elif isinstance(value, list) and key in _PATH_LIST_KEYS:
            for i, item in enumerate(value):
                if isinstance(item, str) and not Path(item).is_absolute():
                    abs_path = (old_base / item).resolve()
                    try:
                        value[i] = str(os.path.relpath(abs_path, new_base))
                    except ValueError:
                        # Cross-drive on Windows; fall back to absolute
                        value[i] = str(abs_path)
        elif isinstance(value, str) and key in _PATH_KEYS:
            if not Path(value).is_absolute():
                abs_path = (old_base / value).resolve()
                try:
                    config[key] = str(os.path.relpath(abs_path, new_base))
                except ValueError:
                    config[key] = str(abs_path)


def generate_all_configs(
    manifest: SceneManifest,
    scene_config: dict,
    configs_dir: Path,
    scene_config_dir: Path | None = None,
    names_filter: list[str] | None = None,
) -> SceneManifest:
    """Generate per-asset configs for all processable assets.

    Updates the manifest with config paths and working directories.

    Args:
        manifest: Scene manifest with detected sub-assets.
        scene_config: The scene-level config dict.
        configs_dir: Directory to write generated configs.
        scene_config_dir: Directory of the original scene config (for rebasing
            relative paths).
        names_filter: Optional name/path filter for assets.

    Returns:
        Updated SceneManifest.
    """
    assets = manifest.get_processable_assets(names_filter)
    logger.info(f"Generating configs for {len(assets)} sub-assets")

    # Build unique safe names: append ID suffix when names collide
    safe_names = _unique_safe_names(assets)

    for i, sa in enumerate(assets, 1):
        safe_name = safe_names[sa.id]
        config_path = configs_dir / f"{safe_name}.yaml"

        logger.info(f"[{i}/{len(assets)}] Generating config for '{sa.name}'")
        try:
            generate_sub_asset_config(
                sa,
                scene_config,
                config_path,
                scene_config_dir=scene_config_dir,
                session_id=safe_name,
            )
            sa.config_path = str(config_path)
            # Working dir will be relative to config file location:
            # configs_dir/.{safe_name}
            sa.working_dir = str(configs_dir / f".{safe_name}")
        except Exception:
            logger.exception(f"Failed to generate config for '{sa.name}'")
            sa.status = "failed"

    return manifest


def generate_payload_config(
    payload_group: PayloadGroup,
    scene_config: dict,
    output_path: Path,
    scene_config_dir: Path | None = None,
    sibling_names: list[str] | None = None,
) -> Path:
    """Generate a per-payload config from the scene config template.

    The generated config:
    - Sets input.usd_path to the payload file path (no prim_path scoping)
    - Disables: optimize_usd, restore_usd, apply, render
    - Enables: build_dataset_usd, build_dataset_prepare_dataset, predict

    Args:
        payload_group: The payload group to generate config for.
        scene_config: The scene-level config dict (will be deep-copied).
        output_path: Where to write the generated YAML.
        scene_config_dir: Directory of the original scene config (for rebasing
            relative paths). If None, paths are kept as-is.

    Returns:
        Path to the generated config file.
    """
    config = copy.deepcopy(scene_config)

    # Remove scene section
    config.pop("scene", None)

    # Set project identity
    safe_name = payload_group.group_name
    project = config.setdefault("project", {})
    project["name"] = safe_name
    project["session_id"] = safe_name

    # Rebase relative paths from scene config dir to generated config dir
    if scene_config_dir is not None:
        output_dir = output_path.parent
        _rebase_paths(config, scene_config_dir.resolve(), output_dir.resolve())

    input_section = config.setdefault("input", {})

    # For large payloads with a representative file, use that for
    # SO/render/predict (much smaller). Otherwise use modified copy
    # (parents) or original payload file (leaves).
    if payload_group.representative_path:
        input_file = payload_group.representative_path
    else:
        input_file = payload_group.modified_input_path or payload_group.payload_file
    payload_path = Path(input_file)
    try:
        rel = os.path.relpath(payload_path.resolve(), output_path.parent.resolve())
    except ValueError:
        rel = str(payload_path.resolve())
    input_section["usd_path"] = rel

    # No prim_path scoping — process the entire payload file
    input_section.pop("prim_path", None)

    # Force layer-only output
    output_section = config.setdefault("output", {})
    output_section["layer_only"] = True
    output_section["flatten_output"] = False

    # Configure steps for payload mode
    steps = config.get("steps", {})

    # Enable restore_usd so predictions use original topology paths.
    # This ensures the output.usd sublayers the original payload
    # (not the SO-optimized file), preserving the drop-in replacement chain.
    restore_config = steps.get("restore_usd", {})
    restore_config["enabled"] = True
    steps["restore_usd"] = restore_config

    # Enable apply — the output.usd IS the "new version" of this payload.
    # layer_only=True means it sublayers the input (drop-in replacement).
    # skip_instance_check=True because payload instances inherit materials
    # via USD composition — no need to traverse all scene instances.
    apply_config = steps.get("apply", {})
    apply_config["enabled"] = True
    apply_config["layer_only"] = True
    apply_config["flatten_output"] = False
    apply_config["skip_instance_check"] = True
    steps["apply"] = apply_config

    # Disable render
    render_config = steps.get("render", {})
    render_config["enabled"] = False
    steps["render"] = render_config

    # Container payloads (parent with children) have no direct meshes —
    # they are pure assembly layers that stitch child outputs together.
    # Disable SO entirely: the flatten step would resolve all child payload
    # arcs and load the entire hierarchy into memory, potentially OOMing.
    optimize_config = steps.get("optimize_usd", {})
    if payload_group.child_payload_files:
        optimize_config["enabled"] = False
        steps["optimize_usd"] = optimize_config
        logger.info("  Container payload: SO disabled (no direct meshes)")
    elif payload_group.representative_path:
        # Representative payloads: use split-only SO (no deinstance, no dedupe).
        # The representative file contains only prototype source prims, so
        # de-instancing is unnecessary and deduplication is counterproductive.
        so_settings = optimize_config.setdefault("scene_optimizer_settings", {})
        so_settings["enableDeinstance"] = False
        so_settings["enableSplitMeshes"] = True
        so_settings["enableDeduplicate"] = False
        steps["optimize_usd"] = optimize_config
        # Store original payload path so the runner can fix output.usd sublayer
        config["_original_payload_file"] = str(
            Path(payload_group.payload_file).resolve()
        )
        logger.info("  Representative mode: SO split-only (no deinstance, no dedupe)")

    # Inject payload context into VLM system prompt so the VLM knows
    # what kind of object it's looking at (critical for simple payloads
    # like a lone tray or carton that lack visual context in isolation)
    _inject_payload_context(config, payload_group, sibling_names=sibling_names)

    # Write config
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    logger.info(
        f"Generated payload config for '{payload_group.group_name}': {output_path}"
    )
    return output_path


def generate_all_payload_configs(
    manifest: SceneManifest,
    scene_config: dict,
    configs_dir: Path,
    scene_config_dir: Path | None = None,
) -> SceneManifest:
    """Generate per-payload configs for all processable payload groups.

    Updates the manifest with config paths and working directories.

    Args:
        manifest: Scene manifest with detected payload groups.
        scene_config: The scene-level config dict.
        configs_dir: Directory to write generated configs.
        scene_config_dir: Directory of the original scene config (for rebasing
            relative paths).

    Returns:
        Updated SceneManifest.
    """
    payloads = manifest.get_processable_payloads()
    if not payloads:
        logger.info("No payload groups to generate configs for")
        return manifest

    logger.info(f"Generating configs for {len(payloads)} payload groups")

    # Build sibling map from the DAG: for each payload, find siblings
    # (other children of the same parent payload). This gives the VLM
    # context about neighboring components in the same system.
    sibling_map: dict[str, list[str]] = {}
    all_pgs = manifest.payload_groups
    file_to_name: dict[str, str] = {}
    for pg in all_pgs:
        resolved = str(Path(pg.payload_file).resolve()) if pg.payload_file else ""
        if resolved:
            file_to_name[resolved] = pg.group_name
    for pg in all_pgs:
        if pg.child_payload_files:
            child_names = []
            for cf in pg.child_payload_files:
                resolved_cf = str(Path(cf).resolve())
                name = file_to_name.get(resolved_cf)
                if name:
                    child_names.append(name)
            for name in child_names:
                sibling_map[name] = child_names

    # Use a subdirectory for payload configs to keep them separate
    payload_configs_dir = configs_dir / "payloads"

    for i, pg in enumerate(payloads, 1):
        config_path = payload_configs_dir / f"{pg.group_name}.yaml"

        logger.info(
            f"[{i}/{len(payloads)}] Generating config for payload '{pg.group_name}'"
        )
        try:
            generate_payload_config(
                pg,
                scene_config,
                config_path,
                scene_config_dir=scene_config_dir,
                sibling_names=sibling_map.get(pg.group_name),
            )
            pg.config_path = str(config_path)
            pg.working_dir = str(payload_configs_dir / f".{pg.group_name}")
        except Exception:
            logger.exception(f"Failed to generate config for payload '{pg.group_name}'")
            pg.status = "failed"

    return manifest


def _inject_payload_context(
    config: dict,
    payload_group: PayloadGroup,
    sibling_names: list[str] | None = None,
) -> None:
    """Append contextual information to the VLM system prompt.

    When a payload is processed in isolation, the VLM loses the spatial
    context it would have in the full scene. For simple objects (a tray,
    a carton, a bracket) this can lead to wrong material guesses.

    This function appends a context block to the VLM system prompt with:
    - The human-readable payload name (e.g., "Tray", "Conveyor_09")
    - The parent system derived from the file path (e.g., "DMS_Shuttle_System")
    - Sibling payload names if available (from DAG parent)

    Args:
        config: The per-payload config dict (modified in place).
        payload_group: The payload group being configured.
        sibling_names: Optional list of sibling payload names (other children
            of the same parent payload in the DAG).
    """
    # Derive human-readable name and parent context from the file path
    payload_path = Path(payload_group.payload_file)
    asset_name = payload_path.stem.replace("_", " ")

    # Walk up the directory tree to find meaningful parent context
    # e.g., .../Assets/Phase_01/DMS_Shuttle_System/Tray/Tray.usd
    #        → parent system: "DMS Shuttle System", category: "Phase 01"
    parent_parts = []
    for parent in payload_path.parents:
        name = parent.name
        if not name or name.lower() in ("assets", "subusd", "subusds", "collected"):
            break
        parent_parts.append(name.replace("_", " "))
    parent_parts.reverse()

    parent_context = ""
    if parent_parts:
        parent_context = " > ".join(parent_parts)

    context_block = (
        "\n\n"
        "IMPORTANT CONTEXT: This object is a component from an industrial/warehouse scene.\n"
    )
    if parent_context:
        context_block += f"Asset hierarchy: {parent_context}\n"
    if sibling_names:
        others = [
            s.replace("_", " ") for s in sibling_names if s != payload_group.group_name
        ]
        if others:
            context_block += f"Sibling components in same system: {', '.join(others)}\n"
    context_block += (
        f'Asset name: "{asset_name}"\n'
        "Use this context to inform your material choices — "
        "industrial/warehouse materials are expected "
        "(e.g., painted metal, powder-coated steel, rubber, plastic).\n"
    )

    # Append to the VLM system prompt
    steps = config.setdefault("steps", {})
    prepare = steps.setdefault("build_dataset_prepare_dataset", {})
    prompts = prepare.setdefault("prompts", {})

    existing_system = prompts.get("vlm_system", "")
    if existing_system:
        prompts["vlm_system"] = existing_system.rstrip() + context_block

    logger.debug(
        f"Injected VLM context for payload '{payload_group.group_name}': "
        f"name='{asset_name}', parent='{parent_context}', "
        f"siblings={len(sibling_names) if sibling_names else 0}"
    )


def _inject_split_context(config: dict, sub_asset: SubAsset) -> None:
    """Append split context to the VLM system prompt.

    When an asset was produced by splitting a larger container, the VLM
    loses the broader context of the parent structure. This injects a
    context block with parent name, ancestor chain, and sibling names
    so the VLM can make informed material choices.

    Args:
        config: The per-asset config dict (modified in place).
        sub_asset: The sub-asset with split_context.
    """
    ctx = sub_asset.split_context
    if not ctx:
        return

    parent_name = ctx.get("parent_name", "").replace("_", " ")
    siblings = ctx.get("sibling_names", [])
    ancestors = ctx.get("ancestors", [])

    # Build human-readable hierarchy
    hierarchy = " > ".join(a.replace("_", " ") for a in ancestors)

    sibling_list = ", ".join(
        s.replace("_", " ") for s in siblings if s != sub_asset.name
    )

    context_block = (
        "\n\nIMPORTANT CONTEXT: This object was extracted from a larger structure.\n"
    )
    if hierarchy:
        context_block += f"Parent hierarchy: {hierarchy}\n"
    if sibling_list:
        context_block += f"Sibling components: {sibling_list}\n"
    context_block += (
        f'This component: "{sub_asset.name.replace("_", " ")}"\n'
        "Use this context to inform your material choices — "
        "materials should be consistent with the parent structure "
        "and neighboring components.\n"
    )

    # Append to the VLM system prompt
    steps = config.setdefault("steps", {})
    prepare = steps.setdefault("build_dataset_prepare_dataset", {})
    prompts = prepare.setdefault("prompts", {})

    existing_system = prompts.get("vlm_system", "")
    if existing_system:
        prompts["vlm_system"] = existing_system.rstrip() + context_block

    logger.debug(
        f"Injected split context for '{sub_asset.name}': "
        f"parent='{parent_name}', siblings={len(siblings)}"
    )


def _sanitize_name(name: str) -> str:
    """Sanitize a name for use as directory/file names and session IDs."""
    safe = re.sub(r"[^\w\-]", "_", name)
    safe = re.sub(r"_+", "_", safe).strip("_")
    return safe.lower() if safe else "unnamed"


def _unique_safe_names(assets: list) -> dict[str, str]:
    """Build a mapping of asset ID → unique safe name.

    When multiple assets share the same sanitized name, appends the asset ID
    as a suffix (e.g. ``default_obj_230``) to disambiguate.
    """
    from collections import Counter

    name_counts = Counter(_sanitize_name(sa.name) for sa in assets)
    result: dict[str, str] = {}
    for sa in assets:
        safe = _sanitize_name(sa.name)
        if name_counts[safe] > 1:
            suffix = _sanitize_name(sa.id)
            safe = f"{safe}_{suffix}"
        result[sa.id] = safe
    return result
