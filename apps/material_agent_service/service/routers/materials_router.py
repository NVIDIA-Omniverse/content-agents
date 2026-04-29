# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Materials API endpoints - Materials catalog."""

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from ..config import config

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/materials", tags=["materials"])


def _serialize_material_entry(
    entry: dict[str, str], *, icon_url_prefix: str | None = None
) -> dict[str, str | None]:
    """Return a stable API payload for a material entry.

    Default libraries may omit thumbnails entirely. In that case we still list
    the material, but return null icon fields instead of pointing clients at a
    guaranteed 404.
    """
    name = entry.get("name", "")
    icon_path = entry.get("icon")

    return {
        "name": name,
        "description": entry.get("description", ""),
        "binding": entry.get("binding", ""),
        "icon_url": (
            f"{icon_url_prefix}/{name}" if icon_url_prefix and icon_path else None
        ),
        "icon_path": icon_path,
    }


# ── Per-library endpoints ────────────────────────────────────────────────────


@router.get("/libraries")
async def list_libraries():
    """List all available material libraries.

    Returns:
        List of {id, name, material_count}
    """
    libraries = []
    for lib_id, lib in config.material_libraries.items():
        libraries.append(
            {
                "id": lib.id,
                "name": lib.name,
                "material_count": len(lib.entries),
            }
        )

    # Sort: default first, then alphabetically
    libraries.sort(
        key=lambda x: (0 if x["id"] == config.default_library_id else 1, x["name"])
    )

    return {"libraries": libraries, "total": len(libraries)}


@router.get("/libraries/{library_id}")
async def get_library_materials(library_id: str):
    """Get materials list for a specific library.

    Args:
        library_id: Library identifier (directory name)

    Returns:
        List of materials with icon URLs
    """
    lib = config.get_library(library_id)
    if not lib:
        raise HTTPException(status_code=404, detail=f"Library not found: {library_id}")

    materials = []
    for entry in lib.entries:
        materials.append(
            _serialize_material_entry(
                entry,
                icon_url_prefix=f"/materials/libraries/{library_id}/icon",
            )
        )

    materials.sort(key=lambda x: x["name"])

    return {
        "library_id": library_id,
        "library_name": lib.name,
        "materials": materials,
        "total": len(materials),
    }


@router.get("/libraries/{library_id}/icon/{material_name:path}")
async def get_library_material_icon(library_id: str, material_name: str):
    """Get material icon from a specific library.

    Args:
        library_id: Library identifier
        material_name: Material name or direct icon path

    Returns:
        PNG image file
    """
    from urllib.parse import unquote, unquote_plus

    lib = config.get_library(library_id)
    if not lib:
        raise HTTPException(status_code=404, detail=f"Library not found: {library_id}")

    decoded_name = unquote_plus(unquote(material_name))

    # Look up icon path by material name
    icon_relative_path = lib.icons.get(decoded_name)

    if not icon_relative_path:
        # Try as direct icon path fallback
        icon_relative_path = decoded_name

    base_dir = Path(lib.base_dir).resolve()
    icon_path = (base_dir / icon_relative_path).resolve()

    # Security: validate resolved path is inside the library directory
    try:
        icon_path.relative_to(base_dir)
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")

    if not icon_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Icon not found for '{decoded_name}' in library '{library_id}'",
        )

    return FileResponse(icon_path, media_type="image/png")


# ── Legacy endpoints (delegate to default library) ──────────────────────────


@router.get("/icon/{material_name:path}")
async def get_material_icon(material_name: str):
    """Get material icon/thumbnail image by material name.

    Uses the default library. For per-library icons, use
    /materials/libraries/{library_id}/icon/{material_name}.

    Args:
        material_name: Material name (e.g., "Glass Clear Saturated Red")
                      or direct icon path (e.g., "thumbs/256x256/Glass_Clear_Saturated_Red.png")

    Returns:
        PNG image file
    """
    return await get_library_material_icon(config.default_library_id, material_name)


@router.get("")
async def list_materials():
    """List all available materials from the default library.

    Returns:
        List of material metadata from the default library.
    """
    materials = []
    default_lib = config.material_libraries.get(config.default_library_id)

    if default_lib:
        for entry in default_lib.entries:
            materials.append(
                _serialize_material_entry(entry, icon_url_prefix="/materials/icon")
            )
    else:
        for entry in config.materials:
            materials.append(
                _serialize_material_entry(entry, icon_url_prefix="/materials/icon")
            )

    # Sort alphabetically
    materials.sort(key=lambda x: x["name"])

    return {"materials": materials, "total": len(materials)}


@router.get("/template")
async def download_materials_template():
    """Download the default materials template ZIP.

    Users can download this as a starting point for creating custom materials.
    The ZIP contains:
    - materials.yaml: Material definitions with name, description, and binding
    - materials_libs.usda: USD material library with shader definitions
    - Optional thumbnails if the source library provides them

    Returns:
        ZIP file download
    """
    # Find the template zip in the materials directory
    template_path = (
        Path(__file__).parent.parent.parent
        / "materials"
        / "default"
        / "default_materials.zip"
    )

    if not template_path.exists():
        logger.error(f"Materials template not found at: {template_path}")
        raise HTTPException(
            status_code=404,
            detail="Materials template not found",
        )

    logger.info(f"Serving materials template: {template_path}")
    return FileResponse(
        template_path,
        media_type="application/zip",
        filename="default_materials.zip",
    )
