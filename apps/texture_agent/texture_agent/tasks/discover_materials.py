# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task: Discover materials in a USD stage."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from world_understanding.agentic.tasks import Task

from texture_agent.functions.material_discovery import discover_materials_from_file

logger = logging.getLogger(__name__)


class DiscoverMaterialsTask(Task):
    """Discover all OpenPBR materials in a USD file.

    Reads the input USD, traverses material prims, and extracts their
    OpenPBR properties (base_color, texture slots, metalness, roughness).

    Context keys read:
        usd_path (str): Path to the input USD file.
        prim_paths (list[str], optional): Restrict to specific material prims.
        working_dir (str): Working directory for output.

    Context keys written:
        discovered_materials (list[MaterialInfo]): Discovered materials.
    """

    def __init__(self) -> None:
        self.name = "DiscoverMaterials"
        self.description = "Discover OpenPBR materials in the USD stage"

    def run(self, context: dict[str, Any], object_store: Any = None) -> dict[str, Any]:
        usd_path = context["usd_path"]
        prim_paths = context.get("prim_paths")

        logger.info("Discovering materials in %s", usd_path)
        materials = discover_materials_from_file(usd_path, prim_paths)

        context["discovered_materials"] = materials

        # Save discovery results to working dir
        working_dir = context.get("working_dir")
        if working_dir:
            out_dir = Path(working_dir) / "discovery"
            out_dir.mkdir(parents=True, exist_ok=True)
            summary = [
                {
                    "name": m.name,
                    "prim_path": m.prim_path,
                    "base_color": list(m.base_color),
                    "has_existing_texture": m.has_existing_texture,
                    "bound_prims": len(m.bound_prim_paths),
                }
                for m in materials
            ]
            (out_dir / "materials.json").write_text(json.dumps(summary, indent=2))

        # Log summary table
        logger.info("Discovered %d materials:", len(materials))
        for m in materials:
            logger.info(
                "  %-30s base_color=(%.2f, %.2f, %.2f) texture=%s prims=%d",
                m.name,
                *m.base_color,
                "yes" if m.has_existing_texture else "no",
                len(m.bound_prim_paths),
            )

        return context
