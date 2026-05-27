# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Artifact manifest and portability helpers for Texture Agent runs."""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from pxr import Sdf, Usd, UsdShade

ARTIFACTS_MANIFEST_SCHEMA_VERSION = "texture-agent-artifacts.v1"
DIAGNOSTIC_SCHEMA_VERSION = "texture-agent-diagnostic.v1"

_URI_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://")


def make_diagnostic(
    code: str,
    *,
    severity: str,
    stage: str,
    message: str,
    recommended_action: str = "",
    prim_path: str | None = None,
    material_name: str | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a versioned diagnostic payload."""
    payload: dict[str, Any] = {
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "code": code,
        "severity": severity,
        "stage": stage,
        "prim_path": prim_path,
        "material_name": material_name,
        "message": message,
        "recommended_action": recommended_action,
        "details": details or {},
    }
    return payload


def _working_root(context: dict[str, Any]) -> Path:
    working_dir = Path(context["working_dir"]).resolve()
    return working_dir.parent


def _display_path(path: str | Path | None, root: Path) -> str | None:
    if path is None:
        return None
    raw = str(path)
    if not raw:
        return raw
    if _URI_SCHEME_RE.match(raw):
        return raw
    try:
        return Path(os.path.relpath(Path(raw).resolve(), root)).as_posix()
    except (OSError, ValueError):
        return raw


def _path_entry(path: str | Path | None, root: Path) -> dict[str, Any] | None:
    if path is None:
        return None
    raw = str(path)
    entry: dict[str, Any] = {"path": _display_path(raw, root)}
    if not raw or _URI_SCHEME_RE.match(raw):
        entry["exists"] = False
        return entry
    try:
        local = Path(raw)
        entry["exists"] = local.exists()
        if local.is_file():
            entry["size_bytes"] = local.stat().st_size
    except (OSError, ValueError):
        entry["exists"] = False
    return entry


def _image_info(path: str | Path | None, root: Path) -> dict[str, Any] | None:
    entry = _path_entry(path, root)
    if entry is None:
        return None
    raw = str(path)
    if not raw or _URI_SCHEME_RE.match(raw):
        return entry
    try:
        from PIL import Image

        with Image.open(raw) as img:
            entry["width"] = img.width
            entry["height"] = img.height
            entry["mode"] = img.mode
            entry["nonblank"] = img.getbbox() is not None
    except Exception as err:
        entry["open_error"] = str(err)
    return entry


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple | set):
        return [_jsonable(v) for v in value]
    if isinstance(value, int | float | str | bool) or value is None:
        return value
    return str(value)


def _redact_sensitive(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            key_str = str(key)
            if any(
                token in key_str.lower()
                for token in ("api_key", "apikey", "token", "password", "secret")
            ):
                redacted[key_str] = "<redacted>"
            else:
                redacted[key_str] = _redact_sensitive(item)
        return redacted
    if isinstance(value, list):
        return [_redact_sensitive(item) for item in value]
    return _jsonable(value)


def _config_summary(context: dict[str, Any]) -> dict[str, Any]:
    config = context.get("config") or {}
    project = config.get("project") or {}
    texture = context.get("texture_config") or {}
    auto_prompt = context.get("auto_prompt_config") or {}
    return {
        "project_name": project.get("name"),
        "session_id": project.get("session_id"),
        "texture": _redact_sensitive(texture),
        "auto_prompt": _redact_sensitive(auto_prompt),
        "steps": _redact_sensitive(config.get("steps") or {}),
    }


def _read_uv_report(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    try:
        with Path(path).open(encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _material_entries(context: dict[str, Any]) -> list[dict[str, Any]]:
    materials = context.get("discovered_materials") or []
    return [_jsonable(material) for material in materials]


def _selected_materials(context: dict[str, Any]) -> list[dict[str, Any]]:
    units = context.get("prim_texture_units") or []
    selected: list[dict[str, Any]] = []
    for unit in units:
        material = getattr(unit, "material_info", None)
        selected.append(
            {
                "key": getattr(unit, "key", ""),
                "material_name": getattr(material, "name", ""),
                "material_path": getattr(material, "prim_path", ""),
                "prim_path": getattr(unit, "prim_path", ""),
                "prompt": getattr(unit, "prompt", ""),
                "opacity": getattr(unit, "opacity", None),
                "seed": getattr(unit, "seed", None),
            }
        )
    return selected


def _texture_set_entries(
    textures: dict[str, Any],
    root: Path,
) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    for key, bundle in textures.items():
        entries[key] = {
            "albedo": _image_info(getattr(bundle, "albedo", None), root),
            "normal": _image_info(getattr(bundle, "normal", None), root),
            "orm": _image_info(getattr(bundle, "orm", None), root),
        }
    return entries


def _output_texture_references(output_usd: Path) -> list[dict[str, Any]] | None:
    stage = Usd.Stage.Open(str(output_usd))
    if not stage:
        return None

    refs: list[dict[str, Any]] = []
    for prim in stage.Traverse():
        is_shader = prim.IsA(UsdShade.Shader)
        for attr in prim.GetAttributes():
            val = attr.Get()
            path: str | None = None
            value_type = ""
            if isinstance(val, Sdf.AssetPath) and val.path:
                path = val.path
                value_type = "asset"
            elif isinstance(val, str) and val and is_shader:
                attr_name = attr.GetName()
                if attr_name.startswith("inputs:") and attr_name.endswith("_texture"):
                    path = val
                    value_type = "string"

            if path and path.lower().endswith(".png"):
                refs.append(
                    {
                        "prim_path": str(prim.GetPath()),
                        "attribute": attr.GetName(),
                        "value_type": value_type,
                        "path": path,
                    }
                )
    return refs


def validate_output_texture_portability(
    output_usd_path: str | Path,
    *,
    bundle_root: str | Path | None = None,
) -> dict[str, Any]:
    """Validate texture refs in an output USD are relative and resolvable."""
    output_usd = Path(output_usd_path)
    diagnostics: list[dict[str, Any]] = []

    if not output_usd.is_file():
        diagnostics.append(
            make_diagnostic(
                "PACKAGE_MISSING_ARTIFACT",
                severity="error",
                stage="package",
                message="Output USD is not present.",
                recommended_action=(
                    "Inspect the apply_textures step and download available "
                    "individual artifacts instead of the package."
                ),
                details={"path": str(output_usd)},
            )
        )
        return {
            "portable": False,
            "texture_reference_count": 0,
            "non_relative_texture_paths": [],
            "missing_texture_paths": [str(output_usd)],
            "diagnostics": diagnostics,
        }

    references = _output_texture_references(output_usd)
    if references is None:
        diagnostics.append(
            make_diagnostic(
                "PACKAGE_MISSING_ARTIFACT",
                severity="error",
                stage="package",
                message="Output USD could not be opened for portability validation.",
                recommended_action=(
                    "Inspect the output USD and download individual artifacts "
                    "instead of the package."
                ),
                details={"path": str(output_usd)},
            )
        )
        return {
            "portable": False,
            "texture_reference_count": 0,
            "non_relative_texture_paths": [],
            "missing_texture_paths": [str(output_usd)],
            "diagnostics": diagnostics,
        }

    missing: list[str] = []
    non_relative: list[str] = []
    resolved_bundle_root = (
        Path(bundle_root).resolve()
        if bundle_root
        else output_usd.parent.parent.resolve()
    )

    for ref in references:
        path = str(ref["path"])
        details = {
            "attribute": ref["attribute"],
            "path": path,
            "value_type": ref["value_type"],
        }
        if _URI_SCHEME_RE.match(path) or Path(path).is_absolute():
            non_relative.append(path)
            diagnostics.append(
                make_diagnostic(
                    "PACKAGE_ABSOLUTE_TEXTURE_PATH",
                    severity="error",
                    stage="package",
                    prim_path=ref["prim_path"],
                    message="Output USD texture reference is not sibling-relative.",
                    recommended_action=(
                        "Rewrite texture references under the output directory "
                        "before packaging or download."
                    ),
                    details=details,
                )
            )
            continue

        resolved = (output_usd.parent / path).resolve()
        try:
            resolved.relative_to(resolved_bundle_root)
        except ValueError:
            non_relative.append(path)
            diagnostics.append(
                make_diagnostic(
                    "PACKAGE_ABSOLUTE_TEXTURE_PATH",
                    severity="error",
                    stage="package",
                    prim_path=ref["prim_path"],
                    message=(
                        "Output USD texture reference leaves the result directory."
                    ),
                    recommended_action=(
                        "Copy textures into the run textures directory and author "
                        "paths relative to the output USD."
                    ),
                    details={
                        **details,
                        "resolved_path": str(resolved),
                        "bundle_root": str(resolved_bundle_root),
                    },
                )
            )
            continue

        if not resolved.is_file():
            missing.append(path)
            diagnostics.append(
                make_diagnostic(
                    "PACKAGE_MISSING_ARTIFACT",
                    severity="error",
                    stage="package",
                    prim_path=ref["prim_path"],
                    message="Output USD references a texture file that is not present.",
                    recommended_action=(
                        "Ensure generated/localized textures are copied next to "
                        "the output USD before packaging."
                    ),
                    details={**details, "resolved_path": str(resolved)},
                )
            )

    return {
        "portable": not diagnostics,
        "texture_reference_count": len(references),
        "non_relative_texture_paths": sorted(set(non_relative)),
        "missing_texture_paths": sorted(set(missing)),
        "diagnostics": diagnostics,
    }


def validate_artifacts_manifest_schema(manifest: dict[str, Any]) -> list[str]:
    """Return schema-contract errors for a texture-agent artifact manifest."""
    errors: list[str] = []
    if manifest.get("schema_version") != ARTIFACTS_MANIFEST_SCHEMA_VERSION:
        errors.append("schema_version must be texture-agent-artifacts.v1")

    required_sections = (
        "input",
        "prepared",
        "materials",
        "prompts",
        "textures",
        "outputs",
        "renders",
        "backend",
        "status",
    )
    for section in required_sections:
        if not isinstance(manifest.get(section), dict):
            errors.append(f"{section} section must be present")

    outputs = manifest.get("outputs") or {}
    portability = outputs.get("portability") or {}
    for key in ("portable", "texture_reference_count", "diagnostics"):
        if key not in portability:
            errors.append(f"outputs.portability.{key} is required")

    status = manifest.get("status") or {}
    for key in ("state", "warnings", "errors", "diagnostics", "service_urls"):
        if key not in status:
            errors.append(f"status.{key} is required")

    for i, diagnostic in enumerate(status.get("diagnostics") or []):
        if not isinstance(diagnostic, dict):
            errors.append(f"status.diagnostics[{i}] must be an object")
            continue
        for key in (
            "schema_version",
            "code",
            "severity",
            "stage",
            "prim_path",
            "material_name",
            "message",
            "recommended_action",
            "details",
        ):
            if key not in diagnostic:
                errors.append(f"status.diagnostics[{i}].{key} is required")
        if diagnostic.get("schema_version") != DIAGNOSTIC_SCHEMA_VERSION:
            errors.append(
                f"status.diagnostics[{i}].schema_version must be "
                "texture-agent-diagnostic.v1"
            )

    return errors


def _dedupe_diagnostics(diagnostics: list[Any]) -> list[Any]:
    deduped: list[Any] = []
    seen: set[str] = set()
    for diagnostic in diagnostics:
        key = json.dumps(_jsonable(diagnostic), sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(diagnostic)
    return deduped


def _backend_section(context: dict[str, Any]) -> dict[str, Any]:
    texture_config = context.get("texture_config") or {}
    image_gen = texture_config.get("image_gen") or {}
    return {
        "backend": texture_config.get("backend", "simple_image_gen"),
        "model": image_gen.get("model") or texture_config.get("model"),
        "endpoint_type": "service"
        if texture_config.get("backend") == "service"
        else image_gen.get("backend", "nim"),
        "endpoint": "<configured>" if texture_config.get("endpoint") else None,
        "conditioning_support": texture_config.get("conditioning_support"),
        "seed": (context.get("variations_config") or {}).get("seed"),
        "texture_size": texture_config.get("size"),
        "custom_parameters": _redact_sensitive(
            texture_config.get("custom_parameters", {})
        ),
    }


def build_artifacts_manifest(
    context: dict[str, Any],
    *,
    status: str,
    service_urls: dict[str, str] | None = None,
    duration_seconds: int | None = None,
) -> dict[str, Any]:
    """Build the schema-versioned artifacts manifest payload."""
    root = _working_root(context)
    uv_prep = context.get("uv_preparation") or {}
    uv_report_path = uv_prep.get("uv_report_path")
    uv_report = _read_uv_report(uv_report_path)

    output_paths = [str(p) for p in context.get("output_usd_paths", [])]
    output_portability = context.get("output_portability")
    if output_portability is None and output_paths:
        output_portability = validate_output_texture_portability(output_paths[0])

    package_diagnostics = context.get("package_diagnostics") or []
    all_diagnostics = [*_jsonable(package_diagnostics)]
    if output_portability:
        all_diagnostics.extend(output_portability.get("diagnostics", []))
    all_diagnostics = _dedupe_diagnostics(all_diagnostics)

    generated = context.get("generated_textures") or {}
    blended = context.get("blended_textures") or {}
    render_paths = [str(p) for p in context.get("rendered_image_paths", [])]

    return {
        "schema_version": ARTIFACTS_MANIFEST_SCHEMA_VERSION,
        "input": {
            "usd": _path_entry(
                (context.get("config") or {}).get("input", {}).get("usd_path")
                or context.get("usd_path"),
                root,
            ),
            "current_usd": _path_entry(context.get("usd_path"), root),
            "config": _config_summary(context),
            "requested_material_scope": sorted(
                (context.get("material_textures") or {}).keys()
            ),
            "requested_prim_scope": context.get("prim_paths") or [],
        },
        "prepared": {
            "prepared_usd": _path_entry(
                uv_report.get("prepared_usd") if uv_report else context.get("usd_path"),
                root,
            ),
            "uv_report": _path_entry(uv_report_path, root),
            "uv_summary": (uv_report or {}).get("summary", {}),
            "uv_actions": _jsonable(uv_prep),
        },
        "materials": {
            "discovered": _material_entries(context),
            "selected": _selected_materials(context),
            "auto_prompt_additions": _jsonable(
                context.get("auto_prompt_additions", [])
            ),
        },
        "prompts": {
            "units": _selected_materials(context),
            "prompt_source": "auto_prompt"
            if context.get("auto_prompt_additions")
            else "material_textures",
        },
        "textures": {
            "generated": _texture_set_entries(generated, root),
            "blended": _texture_set_entries(blended, root),
            "generated_count": len(generated),
            "blended_count": len(blended),
            "generation_errors": _jsonable(context.get("generate_textures_errors", [])),
            "blend_errors": _jsonable(context.get("blend_textures_errors", [])),
        },
        "outputs": {
            "output_usd": [_path_entry(path, root) for path in output_paths],
            "output_usdz": _path_entry(context.get("output_usdz_path"), root),
            "portability": output_portability
            or {
                "portable": False,
                "texture_reference_count": 0,
                "diagnostics": [],
            },
        },
        "renders": {
            "render_available": bool(render_paths),
            "final": [_path_entry(path, root) for path in render_paths],
            "diagnostics": _jsonable(context.get("render_diagnostics", [])),
        },
        "backend": _backend_section(context),
        "status": {
            "state": status,
            "duration_seconds": duration_seconds,
            "timings": _jsonable(context.get("timings", {})),
            "warnings": _jsonable(context.get("warnings", [])),
            "errors": {
                "generate_textures": _jsonable(
                    context.get("generate_textures_errors", [])
                ),
                "blend_textures": _jsonable(context.get("blend_textures_errors", [])),
                "package": _jsonable(package_diagnostics),
            },
            "diagnostics": all_diagnostics,
            "service_urls": service_urls or {},
        },
    }


def write_artifacts_manifest(
    context: dict[str, Any],
    *,
    status: str = "completed",
    service_urls: dict[str, str] | None = None,
    duration_seconds: int | None = None,
    payload: dict[str, Any] | None = None,
) -> Path:
    """Write ``artifacts_manifest.json`` into the run working directory."""
    working_dir = Path(context["working_dir"])
    working_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = working_dir / "artifacts_manifest.json"
    manifest = payload or build_artifacts_manifest(
        context,
        status=status,
        service_urls=service_urls,
        duration_seconds=duration_seconds,
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    context["artifacts_manifest_path"] = str(manifest_path)
    return manifest_path
