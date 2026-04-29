# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bundle a composed scene into a self-contained directory.

Flattens the composed scene USD, copies the material library alongside it,
and rewrites all asset paths to be relative so the directory can be moved
to any machine with Kit for rendering.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def create_bundle(
    composed_scene_path: Path,
    material_library_dir: Path,
    bundle_dir: Path,
    output_format: str = ".usdc",
) -> dict[str, Any]:
    """Create a self-contained bundle from a composed scene.

    Args:
        composed_scene_path: Path to the composed_scene.usd.
        material_library_dir: Path to the material library root directory
            (the directory containing the ``Library/`` folder and the
            material USD file).
        bundle_dir: Output directory for the bundle.
        output_format: ``.usdc`` (binary, default) or ``.usda`` (text).

    Returns:
        Dict with bundle metadata: usd_file, usd_size_mb, library_files,
        total_size_mb, verified_paths, missing_paths.
    """
    from pxr import Sdf, Usd, UsdGeom, UsdShade

    # Clean and create bundle dir
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True)

    # --- Step 1: Flatten ---
    logger.info("Flattening composed scene: %s", composed_scene_path)
    stage = Usd.Stage.Open(str(composed_scene_path))
    if stage is None:
        raise RuntimeError(f"Failed to open USD stage: {composed_scene_path}")
    up_axis = UsdGeom.GetStageUpAxis(stage)
    meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)
    flat_layer = stage.Flatten()

    # Export as USDA first (we need text for path rewriting)
    usda_path = bundle_dir / "scene_flat.usda"
    flat_layer.Export(str(usda_path))

    # Set stage metadata on exported file
    flat_stage = Usd.Stage.Open(str(usda_path))
    if flat_stage is None:
        raise RuntimeError(f"Failed to open flattened USD stage: {usda_path}")
    UsdGeom.SetStageUpAxis(flat_stage, up_axis)
    UsdGeom.SetStageMetersPerUnit(flat_stage, meters_per_unit)
    flat_stage.GetRootLayer().Save()
    del flat_stage

    logger.info(
        "Flattened to %s (%.0f MB)",
        usda_path.name,
        usda_path.stat().st_size / 1024 / 1024,
    )

    # --- Step 2: Copy material library ---
    lib_src = material_library_dir / "Library"
    lib_dest = bundle_dir / "Library"
    if lib_src.exists():
        logger.info("Copying material library from %s", lib_src)
        shutil.copytree(str(lib_src), str(lib_dest))
    else:
        logger.warning("No Library/ directory found in %s", material_library_dir)

    # --- Step 3: Rewrite asset paths ---
    mat_lib_resolved = str(material_library_dir.resolve())
    text = usda_path.read_text()

    # Replace absolute resolved paths → relative ./Library/...
    # After flatten, USD resolves all asset paths to absolute.
    old_lib = mat_lib_resolved + "/Library/"
    new_lib = "./Library/"
    count = text.count(old_lib)
    text = text.replace(old_lib, new_lib)

    # Also catch any paths relative to the mat lib root
    old_root = mat_lib_resolved + "/"
    count2 = text.count(old_root)
    if count2:
        text = text.replace(old_root, "./")
        count += count2

    logger.info("Rewrote %d asset paths", count)
    usda_path.write_text(text)

    # --- Step 4: Convert to final format ---
    if output_format == ".usdc":
        usdc_path = bundle_dir / "scene_flat.usdc"
        logger.info("Converting to binary USDC...")
        final_stage = Usd.Stage.Open(str(usda_path))
        if final_stage is None:
            raise RuntimeError(f"Failed to open USD stage for USDC export: {usda_path}")
        final_stage.GetRootLayer().Export(str(usdc_path))
        usda_path.unlink()
        final_path = usdc_path
    else:
        final_path = usda_path

    # --- Step 5: Verify ---
    logger.info("Verifying asset paths...")
    test_stage = Usd.Stage.Open(str(final_path))
    if test_stage is None:
        raise RuntimeError(
            f"Failed to open bundled USD stage for verification: {final_path}"
        )
    missing = 0
    total = 0
    for prim in test_stage.Traverse():
        if prim.IsA(UsdShade.Shader):
            for attr in prim.GetAttributes():
                if attr.HasValue():
                    val = attr.Get()
                    if isinstance(val, Sdf.AssetPath) and val.path:
                        if val.path.startswith("http"):
                            continue
                        total += 1
                        resolved = val.resolvedPath
                        if not resolved or not Path(resolved).exists():
                            missing += 1
                            logger.warning(
                                "Unresolved: %s %s: %s",
                                prim.GetPath(),
                                attr.GetName(),
                                val.path,
                            )

    if missing:
        logger.warning("%d of %d asset paths unresolved", missing, total)
    else:
        logger.info("All %d asset paths resolve correctly", total)

    # --- Report ---
    bundle_size = sum(f.stat().st_size for f in bundle_dir.rglob("*") if f.is_file())
    lib_file_count = (
        sum(1 for _ in lib_dest.rglob("*") if _.is_file()) if lib_dest.exists() else 0
    )

    return {
        "usd_file": final_path,
        "usd_size_mb": final_path.stat().st_size / 1024 / 1024,
        "library_files": lib_file_count,
        "total_size_mb": bundle_size / 1024 / 1024,
        "verified_paths": total,
        "missing_paths": missing,
    }
