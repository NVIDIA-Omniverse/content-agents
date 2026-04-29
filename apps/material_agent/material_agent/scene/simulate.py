# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Simulate mode: generate mock predictions without VLM/render calls.

Produces valid ``predictions.jsonl`` by enumerating geometry prims in a
USD file and assigning materials round-robin from the material library.
This lets the full pipeline structure (SO, arc rewriting, collect) be
tested in minutes without any NVCF rendering or VLM inference.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

# Prim types that the pipeline considers for material assignment
_GEOMETRY_TYPE_NAMES = frozenset(
    {
        "Mesh",
        "GeomSubset",
        "Cube",
        "Cylinder",
        "Capsule",
        "Sphere",
        "Cone",
    }
)


def generate_mock_predictions(
    usd_path: str | Path,
    material_names: list[str],
    output_path: str | Path,
    prim_path_scope: str | None = None,
) -> int:
    """Generate mock predictions for geometry prims in a USD file.

    Opens the USD stage, traverses geometry prims (optionally scoped to
    *prim_path_scope*), and assigns materials round-robin from
    *material_names*.  Writes a valid ``predictions.jsonl`` file.

    Args:
        usd_path: Path to the USD file to enumerate prims from.
        material_names: List of material names from the library.
        output_path: Path to write the predictions JSONL.
        prim_path_scope: Optional prim path prefix to scope traversal.

    Returns:
        Number of predictions written.
    """
    from pxr import Usd

    usd_path = Path(usd_path)
    output_path = Path(output_path)

    if not material_names:
        raise ValueError("material_names must not be empty for simulate mode")

    stage = Usd.Stage.Open(str(usd_path), Usd.Stage.LoadAll)
    if not stage:
        raise RuntimeError(f"Cannot open USD stage: {usd_path}")

    # Collect geometry prim paths, including instance proxies so that
    # simulate mode covers exactly the same prims as the real pipeline.
    prim_paths: list[str] = []
    for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
        if prim.GetTypeName() not in _GEOMETRY_TYPE_NAMES:
            continue
        prim_path = str(prim.GetPath())
        if prim_path_scope and not prim_path.startswith(prim_path_scope):
            continue
        prim_paths.append(prim_path)

    if not prim_paths:
        logger.warning(f"No geometry prims found in {usd_path}")

    # Assign materials round-robin
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(output_path, "w") as f:
        for i, prim_path in enumerate(prim_paths):
            material = material_names[i % len(material_names)]
            prediction = {
                "id": prim_path,
                "materials": {"material": material},
            }
            f.write(json.dumps(prediction) + "\n")
            count += 1

    logger.info(
        f"simulate: wrote {count} mock predictions to {output_path} "
        f"(round-robin over {len(material_names)} materials)"
    )
    return count


def generate_mock_predictions_append(
    usd_path: str | Path,
    material_names: list[str],
    output_path: str | Path,
    prim_path_scope: str | None = None,
) -> int:
    """Append mock predictions for prims not already in *output_path*.

    Same as :func:`generate_mock_predictions` but reads existing predictions
    first and only writes entries for prim paths that are missing.  This is
    used to fill in parent Mesh prims after SO has split them into
    per-GeomSubset meshes.

    Returns:
        Number of new predictions appended.
    """
    from pxr import Usd

    usd_path = Path(usd_path)
    output_path = Path(output_path)

    if not material_names:
        return 0

    # Load existing prediction IDs
    existing_ids: set[str] = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    existing_ids.add(json.loads(line).get("id", ""))

    stage = Usd.Stage.Open(str(usd_path), Usd.Stage.LoadAll)
    if not stage:
        return 0

    # Collect new prim paths
    new_paths: list[str] = []
    for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
        if prim.GetTypeName() not in _GEOMETRY_TYPE_NAMES:
            continue
        prim_path = str(prim.GetPath())
        if prim_path_scope and not prim_path.startswith(prim_path_scope):
            continue
        if prim_path not in existing_ids:
            new_paths.append(prim_path)

    if not new_paths:
        return 0

    count = 0
    with open(output_path, "a") as f:
        for i, prim_path in enumerate(new_paths):
            material = material_names[i % len(material_names)]
            prediction = {
                "id": prim_path,
                "materials": {"material": material},
            }
            f.write(json.dumps(prediction) + "\n")
            count += 1

    return count


def load_material_names_from_config(
    scene_config: dict,
    config_path: Path,
) -> list[str]:
    """Load material names from the material library YAML referenced in config.

    Resolves ``materials.path`` relative to *config_path*, loads the YAML,
    and extracts ``entries[].name``.

    Args:
        scene_config: The scene-level config dict.
        config_path: Path to the scene config file (for resolving relative paths).

    Returns:
        List of material name strings.

    Raises:
        FileNotFoundError: If the materials YAML cannot be found.
        ValueError: If no material entries are found.
    """
    materials_section = scene_config.get("materials", {})
    mat_path_str = materials_section.get("path")
    if not mat_path_str:
        raise ValueError("No materials.path configured")

    mat_path = Path(mat_path_str)
    if not mat_path.is_absolute():
        mat_path = (config_path.parent / mat_path).resolve()

    if not mat_path.exists():
        raise FileNotFoundError(f"Materials YAML not found: {mat_path}")

    with open(mat_path) as f:
        mat_data = yaml.safe_load(f)

    entries = mat_data.get("entries", [])
    names = [e["name"] for e in entries if e.get("name")]

    if not names:
        raise ValueError(f"No material entries found in {mat_path}")

    logger.info(f"simulate: loaded {len(names)} material names from {mat_path}")
    return names
