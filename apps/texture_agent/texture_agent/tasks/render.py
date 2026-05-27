# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task: Render the final textured USD."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from world_understanding.agentic.tasks import Task

logger = logging.getLogger(__name__)

_DIAGNOSTIC_SCHEMA_VERSION = "texture-agent-diagnostic.v1"


def _diagnostic(
    code: str,
    message: str,
    *,
    severity: str = "error",
    usd_path: str | None = None,
    camera_path: str | None = None,
    recommended_action: str = "",
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    diagnostic: dict[str, Any] = {
        "schema_version": _DIAGNOSTIC_SCHEMA_VERSION,
        "code": code,
        "severity": severity,
        "stage": "render",
        "message": message,
        "recommended_action": recommended_action,
        "details": details or {},
    }
    if usd_path:
        diagnostic["usd_path"] = usd_path
    if camera_path:
        diagnostic["camera_path"] = camera_path
    return diagnostic


def _render_result_items(results: Any) -> list[dict[str, Any]]:
    """Return per-camera renderer results from supported renderer shapes."""
    if isinstance(results, dict):
        items = results.get("results")
        if isinstance(items, list) and all(isinstance(item, dict) for item in items):
            return items
        raise ValueError(
            "render_all_cameras returned a dict without a list-valued 'results' key"
        )

    if isinstance(results, list) and all(isinstance(item, dict) for item in results):
        return results

    raise TypeError(
        "render_all_cameras returned unsupported result shape "
        f"{type(results).__name__}; expected dict['results'] or list[dict]"
    )


def _as_path_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list | tuple):
        return [str(item) for item in value if item]
    return []


def _configured_camera_paths(config: dict[str, Any]) -> list[str]:
    for key in ("camera_paths", "cameras", "camera_path"):
        camera_paths = _as_path_list(config.get(key))
        if camera_paths:
            return camera_paths
    return []


def _config_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)


def _focus_cameras_enabled(config: dict[str, Any]) -> bool:
    if "focus_cameras" in config:
        return _config_bool(config.get("focus_cameras"), default=True)
    return _config_bool(config.get("render_focus_cameras"), default=True)


def _stage_camera_paths(stage: Any) -> list[str]:
    from pxr import UsdGeom

    return [
        str(prim.GetPath()) for prim in stage.Traverse() if prim.IsA(UsdGeom.Camera)
    ]


def _selected_prim_paths(context: dict[str, Any], config: dict[str, Any]) -> list[str]:
    explicit = _as_path_list(
        config.get("focus_prim_paths") or config.get("focus_prim_path")
    )
    if explicit:
        return explicit

    selected: list[str] = []
    units = context.get("prim_texture_units") or []
    for unit in units:
        unit_path = getattr(unit, "prim_path", None)
        if unit_path:
            selected.append(str(unit_path))
            continue
        material = getattr(unit, "material_info", None)
        selected.extend(str(path) for path in getattr(material, "bound_prim_paths", []))

    if selected:
        return list(dict.fromkeys(selected))

    material_textures = context.get("material_textures") or {}
    selected_materials = set(material_textures)
    if not selected_materials:
        return []

    for material in context.get("discovered_materials", []) or []:
        if getattr(material, "name", None) in selected_materials:
            selected.extend(
                str(path) for path in getattr(material, "bound_prim_paths", [])
            )

    return list(dict.fromkeys(selected))


def _add_default_camera(stage: Any, config: dict[str, Any]) -> str:
    from world_understanding.utils.usd.camera import add_corner_view_camera

    camera_path = str(config.get("camera_path") or "/Cameras/TextureAgentFinal")
    add_corner_view_camera(
        stage,
        camera_path=camera_path,
        direction=str(config.get("camera_direction", "+x+y+z")),
        margin=float(config.get("camera_margin", 1.25)),
        focal_length=float(config.get("camera_focal_length", 60.0)),
    )
    return camera_path


def _add_focus_cameras(
    stage: Any,
    context: dict[str, Any],
    config: dict[str, Any],
    output_index: int,
) -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]]]:
    if not _focus_cameras_enabled(config):
        return [], [], []

    from world_understanding.utils.usd.camera import add_focused_corner_view_camera

    camera_paths: list[str] = []
    focus_stats: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    focus_prim_paths = _selected_prim_paths(context, config)
    max_cameras = max(0, int(config.get("max_focus_cameras", 1)))
    if max_cameras == 0:
        return [], [], []

    threshold = float(config.get("target_frame_coverage_threshold", 0.2))
    margin = float(config.get("focus_camera_margin", 1.15))

    for focus_index, prim_path in enumerate(focus_prim_paths):
        prim = stage.GetPrimAtPath(prim_path)
        if not prim:
            diagnostics.append(
                _diagnostic(
                    "RENDER_FOCUS_PRIM_MISSING",
                    f"Focused render prim does not exist: {prim_path}",
                    severity="warning",
                    recommended_action=(
                        "Use steps.render.focus_prim_paths with geometry prim "
                        "paths present in the output USD."
                    ),
                    details={"prim_path": prim_path},
                )
            )
            continue

        camera_path = f"/Cameras/TextureAgentFocus_{output_index}_{focus_index}"
        try:
            add_focused_corner_view_camera(
                prim,
                camera_path=camera_path,
                direction=str(
                    config.get(
                        "focus_camera_direction",
                        config.get("camera_direction", "+x+y+z"),
                    )
                ),
                margin=margin,
                focal_length=float(
                    config.get(
                        "focus_camera_focal_length",
                        config.get("camera_focal_length", 60.0),
                    )
                ),
            )
        except Exception as exc:
            diagnostics.append(
                _diagnostic(
                    "RENDER_FOCUS_CAMERA_FAILED",
                    f"Failed to add focused render camera for {prim_path}: {exc}",
                    severity="warning",
                    camera_path=camera_path,
                    recommended_action=(
                        "Use steps.render.focus_prim_paths with imageable "
                        "geometry prims that have valid bounds."
                    ),
                    details={
                        "prim_path": prim_path,
                        "exception_type": type(exc).__name__,
                    },
                )
            )
            logger.warning(
                "Failed to add focused render camera for %s", prim_path, exc_info=True
            )
            continue

        camera_paths.append(camera_path)
        coverage_estimate = min(1.0, 1.0 / max(margin * margin, 1e-6))
        focus_stat = {
            "prim_path": prim_path,
            "camera_path": camera_path,
            "target_frame_coverage_threshold": threshold,
            "target_frame_coverage_heuristic": coverage_estimate,
            "coverage_metric_source": "focus_camera_bbox_margin_heuristic",
            "coverage_is_estimate": True,
            "meets_target_frame_coverage": coverage_estimate >= threshold,
        }
        focus_stats.append(focus_stat)
        if not focus_stat["meets_target_frame_coverage"]:
            diagnostics.append(
                _diagnostic(
                    "RENDER_FRAME_TOO_WIDE",
                    (
                        "Focused render framing estimate is below the target "
                        f"coverage threshold for {prim_path}."
                    ),
                    severity="warning",
                    camera_path=camera_path,
                    recommended_action=(
                        "Reduce steps.render.focus_camera_margin or lower "
                        "steps.render.target_frame_coverage_threshold."
                    ),
                    details=focus_stat,
                )
            )
        if len(camera_paths) >= max_cameras:
            break

    return camera_paths, focus_stats, diagnostics


def _status_is_failure(result: dict[str, Any]) -> bool:
    if not result.get("images"):
        return True
    status = result.get("status")
    if status is None:
        return False
    return str(status) != "success"


def _result_camera_path(
    result: dict[str, Any],
    result_index: int,
    requested_camera_paths: list[str],
) -> str:
    if result.get("camera"):
        return str(result["camera"])
    if result_index < len(requested_camera_paths):
        return requested_camera_paths[result_index]
    return f"camera_{result_index}"


def _extend_unique(target: list[str], values: list[str]) -> None:
    for value in values:
        if value not in target:
            target.append(value)


class RenderOutputTask(Task):
    """Render the final textured USD for visual verification.

    Context keys read:
        output_usd_paths (list[str]): From ApplyTexturesTask.
        render_config (dict): backend, image_width, image_height, etc.
        working_dir (str): Working directory.

    Context keys written:
        rendered_image_paths (list[str]): Paths to rendered images.
        render_diagnostics (list[dict]): Structured render warnings and errors.
        render_errors (list[dict]): Error-severity render diagnostics.
        render_stats (dict): Render availability, cameras, focus cameras, and count.
    """

    def __init__(self) -> None:
        self.name = "RenderOutput"
        self.description = "Render the final textured USD"

    def run(self, context: dict[str, Any], object_store: Any = None) -> dict[str, Any]:
        output_usd_paths: list[str] = context.get("output_usd_paths", [])
        config: dict[str, Any] = context.get("render_config", {})
        working_dir = Path(context["working_dir"])
        diagnostics: list[dict[str, Any]] = []
        render_stats: dict[str, Any] = {
            "camera_paths": [],
            "focus_cameras": [],
            "renders_count": 0,
            "render_available": False,
        }

        if not output_usd_paths:
            logger.info("No output USDs to render")
            context["rendered_image_paths"] = []
            context["render_diagnostics"] = diagnostics
            context["render_errors"] = []
            context["render_stats"] = render_stats
            return context

        image_width = config.get("image_width", 1024)
        image_height = config.get("image_height", image_width)

        out_dir = working_dir / "renders"
        out_dir.mkdir(parents=True, exist_ok=True)

        from pxr import Usd
        from world_understanding.functions.graphics.render_remote import (
            render_all_cameras,
        )
        from world_understanding.utils.usd.material import (
            convert_custom_mdl_to_builtin,
        )

        rendered: list[str] = []

        for output_index, usd_path in enumerate(output_usd_paths):
            logger.info("Rendering %s", usd_path)
            try:
                try:
                    stage = Usd.Stage.Open(str(usd_path))
                except Exception as exc:
                    diagnostics.append(
                        _diagnostic(
                            "RENDER_OUTPUT_USD_OPEN_FAILED",
                            f"Failed to open output USD for rendering: {usd_path}",
                            usd_path=str(usd_path),
                            recommended_action=(
                                "Check that apply_textures produced a valid USD "
                                "path before the render step runs."
                            ),
                            details={
                                "exception_type": type(exc).__name__,
                                "error": str(exc),
                            },
                        )
                    )
                    logger.exception("Failed to open stage: %s", usd_path)
                    continue

                if not stage:
                    diagnostics.append(
                        _diagnostic(
                            "RENDER_OUTPUT_USD_OPEN_FAILED",
                            f"Failed to open output USD for rendering: {usd_path}",
                            usd_path=str(usd_path),
                            recommended_action=(
                                "Check that apply_textures produced a valid USD "
                                "path before the render step runs."
                            ),
                            details={"reason": "stage_returned_none"},
                        )
                    )
                    logger.warning("Failed to open stage: %s", usd_path)
                    continue

                # Flatten for remote rendering.
                flat_layer = stage.Flatten()
                flat_stage = Usd.Stage.Open(flat_layer)
                convert_custom_mdl_to_builtin(flat_stage)

                camera_paths = _configured_camera_paths(config)
                if not camera_paths:
                    camera_paths = _stage_camera_paths(flat_stage)

                if not camera_paths:
                    camera_path = _add_default_camera(flat_stage, config)
                    camera_paths = [camera_path]
                    diagnostics.append(
                        _diagnostic(
                            "RENDER_NO_CAMERA",
                            (
                                "Output USD has no authored camera; added a "
                                f"default final render camera at {camera_path}."
                            ),
                            severity="warning",
                            usd_path=str(usd_path),
                            camera_path=camera_path,
                            recommended_action=(
                                "Author a camera in the source USD or set "
                                "steps.render.camera_paths for deterministic "
                                "final renders."
                            ),
                        )
                    )

                focus_cameras, focus_stats, focus_diagnostics = _add_focus_cameras(
                    flat_stage, context, config, output_index
                )
                for camera_path in focus_cameras:
                    if camera_path not in camera_paths:
                        camera_paths.append(camera_path)
                render_stats["focus_cameras"].extend(focus_stats)
                diagnostics.extend(focus_diagnostics)
                _extend_unique(render_stats["camera_paths"], camera_paths)

                results = render_all_cameras(
                    stage=flat_stage,
                    image_width=image_width,
                    image_height=image_height,
                    cameras=camera_paths,
                )

                try:
                    render_items = _render_result_items(results)
                except (TypeError, ValueError) as exc:
                    diagnostics.append(
                        _diagnostic(
                            "RENDER_RESULT_PARSE_ERROR",
                            f"Renderer returned an unsupported result shape: {exc}",
                            usd_path=str(usd_path),
                            recommended_action=(
                                "Update the render task contract or renderer "
                                "mock to return {'results': [...]}."
                            ),
                            details={"exception_type": type(exc).__name__},
                        )
                    )
                    logger.exception("Failed to parse render results for %s", usd_path)
                    continue

                for i, result in enumerate(render_items):
                    images = result.get("images", [])
                    if _status_is_failure(result):
                        camera_path = _result_camera_path(result, i, camera_paths)
                        status = result.get("status")
                        failed_status = status is not None and str(status) != "success"
                        code = (
                            "RENDER_PER_CAMERA_FAILURE"
                            if failed_status
                            else "RENDER_EMPTY_RESULT"
                        )
                        if code == "RENDER_EMPTY_RESULT":
                            message = str(
                                result.get("error") or "Renderer returned no images"
                            )
                        else:
                            message = str(
                                result.get("error")
                                or status
                                or "Renderer returned no images"
                            )
                        diagnostics.append(
                            _diagnostic(
                                code,
                                f"Render failed for {camera_path}: {message}",
                                usd_path=str(usd_path),
                                camera_path=camera_path,
                                recommended_action=(
                                    "Check render service logs, camera paths, "
                                    "and output USD asset dependencies."
                                ),
                                details={
                                    "status": status,
                                    "images_count": len(images),
                                },
                            )
                        )
                        logger.warning("Render failed for %s: %s", camera_path, message)
                        if failed_status:
                            continue

                    for j, img in enumerate(images):
                        out_name = f"render_{output_index}_{i}_{j}.png"
                        out_path = out_dir / out_name
                        img.save(str(out_path))
                        rendered.append(str(out_path))
                        logger.info("  Saved render: %s", out_path)

            except Exception as exc:
                message = f"Failed to render {usd_path}: {exc}"
                diagnostics.append(
                    _diagnostic(
                        "RENDER_UNEXPECTED_ERROR",
                        message,
                        usd_path=str(usd_path),
                        recommended_action=(
                            "Check renderer result shape, output USD validity, "
                            "camera paths, and render service connectivity."
                        ),
                        details={"exception_type": type(exc).__name__},
                    )
                )
                logger.exception("Failed to render %s", usd_path)

        context["rendered_image_paths"] = rendered
        render_stats["renders_count"] = len(rendered)
        render_stats["render_available"] = bool(rendered)
        context["render_stats"] = render_stats
        context["render_diagnostics"] = diagnostics
        context["render_errors"] = [
            item for item in diagnostics if item.get("severity") == "error"
        ]
        logger.info("Rendered %d images", len(rendered))
        return context
