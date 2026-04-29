# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task: Prepare UV coordinates for texture generation.

Automatically detects and fixes UV issues:
1. Generates UVs for meshes that have none (box or planar projection)
2. Fixes 'constant' interpolation to 'faceVarying'
3. Normalizes out-of-range UVs to [0, 1]

Saves a prepared copy of the input USD so the original is not modified.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pxr import Usd
from world_understanding.agentic.tasks import Task
from world_understanding.functions.graphics.uv_generation import (
    ProjectionType,
    generate_projection_uvs,
)

from texture_agent.functions.uv_generation import (
    UVProjectionMode,
    fix_uv_interpolation,
    generate_uvs_for_stage,
    normalize_uvs,
)

logger = logging.getLogger(__name__)


_SO_PROJECTION_TYPES = {
    "planar": ProjectionType.PLANAR,
    "spherical": ProjectionType.SPHERICAL,
    "cylindrical": ProjectionType.CYLINDRICAL,
    "triplanar": ProjectionType.TRIPLANAR,
    "cube": ProjectionType.CUBE,
    "box": ProjectionType.CUBE,
}


def _prepare_with_python_uvs(
    usd_path: str,
    working_dir: Path,
    uv_mode: UVProjectionMode,
) -> tuple[str, dict[str, Any]]:
    stage = Usd.Stage.Open(str(usd_path))
    if not stage:
        raise FileNotFoundError(f"Failed to open: {usd_path}")

    flat_layer = stage.Flatten()
    flat_stage = Usd.Stage.Open(flat_layer)

    generated = generate_uvs_for_stage(flat_stage, mode=uv_mode)
    fixed_interp = fix_uv_interpolation(flat_stage)
    normalized = normalize_uvs(flat_stage)

    total_fixes = generated + fixed_interp + normalized
    summary: dict[str, Any] = {
        "generated": generated,
        "fixed_interpolation": fixed_interp,
        "normalized": normalized,
    }

    if total_fixes == 0:
        logger.info("No UV fixes needed")
        return usd_path, summary

    prep_dir = working_dir / "prepared"
    prep_dir.mkdir(parents=True, exist_ok=True)
    prepared_path = prep_dir / "prepared_input.usd"
    flat_stage.GetRootLayer().Export(str(prepared_path))
    return str(prepared_path), summary


class PrepareUVsTask(Task):
    """Prepare UV coordinates for all meshes in the input USD.

    Detects meshes without UVs, with wrong interpolation, or with
    out-of-range coordinates, and fixes them automatically. Saves
    a prepared copy to the working directory if fixes are applied.

    Context keys read:
        usd_path (str): Path to the input USD file.
        working_dir (str): Working directory.
        texture_config (dict): May contain uv_mode ("box" or "planar") and
            uv_backend ("python" or "scene_optimizer").

    Context keys written:
        usd_path (str): Updated to point to the prepared USD copy.
        uv_preparation (dict): Summary of fixes applied.
    """

    def __init__(self) -> None:
        self.name = "PrepareUVs"
        self.description = "Generate and fix UV coordinates"

    def run(self, context: dict[str, Any], object_store: Any = None) -> dict[str, Any]:
        usd_path = context["usd_path"]
        working_dir = Path(context["working_dir"])
        texture_config = context.get("texture_config", {})

        uv_mode_str = texture_config.get("uv_mode", "box")
        uv_backend = texture_config.get("uv_backend", "python")
        try:
            uv_mode = UVProjectionMode(uv_mode_str)
        except ValueError:
            valid = [m.value for m in UVProjectionMode]
            raise ValueError(
                f"Invalid UV projection mode '{uv_mode_str}'. Valid modes: {valid}"
            ) from None

        logger.info(
            "Preparing UVs for %s (backend=%s, mode=%s)",
            usd_path,
            uv_backend,
            uv_mode.value,
        )

        fallback_uv_mode = uv_mode
        if uv_backend in ("scene_optimizer", "so"):
            prep_dir = working_dir / "prepared"
            prep_dir.mkdir(parents=True, exist_ok=True)
            flat_input_path = prep_dir / "prepared_input_flat.usd"
            prepared_path = prep_dir / "prepared_input.usd"

            stage = Usd.Stage.Open(str(usd_path))
            if not stage:
                raise FileNotFoundError(f"Failed to open: {usd_path}")
            flat_stage = Usd.Stage.Open(stage.Flatten())
            flat_stage.GetRootLayer().Export(str(flat_input_path))

            so_projection = texture_config.get("uv_projection", uv_mode.value)
            projection_type = _SO_PROJECTION_TYPES.get(str(so_projection))
            if projection_type is None:
                valid = ", ".join(sorted(_SO_PROJECTION_TYPES))
                raise ValueError(
                    f"Invalid Scene Optimizer UV projection '{so_projection}'. "
                    f"Valid modes: {valid}"
                )

            try:
                result = generate_projection_uvs(
                    flat_input_path,
                    prepared_path,
                    projection_type=projection_type,
                    backend=texture_config.get("uv_so_backend", "local"),
                    allow_remote_fallback=texture_config.get(
                        "uv_allow_remote_fallback", False
                    ),
                    overwrite_existing=texture_config.get(
                        "uv_overwrite_existing", True
                    ),
                    scale_factor=texture_config.get("uv_scale_factor", 0.01),
                    scale_units=texture_config.get("uv_scale_units", 0.0),
                    timeout=texture_config.get("uv_timeout", 600),
                )
                prepared_stage = Usd.Stage.Open(str(prepared_path))
                if not prepared_stage:
                    raise RuntimeError(
                        f"Scene Optimizer UV output could not be opened: {prepared_path}"
                    )
                fixed_interp = fix_uv_interpolation(prepared_stage)
                normalized = normalize_uvs(prepared_stage)
                prepared_stage.GetRootLayer().Export(str(prepared_path))

                context["usd_path"] = str(prepared_path)
                context["uv_preparation"] = {
                    "backend": "scene_optimizer",
                    "projection": projection_type.name.lower(),
                    "generated": int(result.get("meshes_with_uvs", 0)),
                    "fixed_interpolation": fixed_interp,
                    "normalized": normalized,
                    "so_result": result,
                }
                logger.info(
                    "UV preparation complete via Scene Optimizer: %s",
                    prepared_path,
                )
                return context
            except Exception as err:
                fallback_uv_mode = UVProjectionMode.BOX
                logger.warning(
                    "Scene Optimizer UV projection failed (%s); falling back to "
                    "Python box projection",
                    err,
                )

        prepared_usd_path, summary = _prepare_with_python_uvs(
            usd_path,
            working_dir,
            fallback_uv_mode,
        )

        if prepared_usd_path == usd_path:
            context["uv_preparation"] = summary
            return context

        context["usd_path"] = prepared_usd_path
        context["uv_preparation"] = summary

        logger.info(
            "UV preparation complete: %d generated, %d interp fixed, "
            "%d normalized. Saved: %s",
            summary["generated"],
            summary["fixed_interpolation"],
            summary["normalized"],
            prepared_usd_path,
        )

        return context
