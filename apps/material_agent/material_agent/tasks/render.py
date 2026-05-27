# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unified task for rendering USD with optional flattening.

This task provides a flexible render capability that:
1. Takes an arbitrary USD file (from apply step or any source)
2. Optionally flattens it for rendering
3. Renders it to specified output path(s)
4. Supports both standalone and workflow-integrated usage patterns
"""

import base64
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import Any, TypedDict, cast

from PIL import Image
from pxr import Usd, UsdGeom
from world_understanding.agentic.events import get_listener
from world_understanding.agentic.tasks import Task
from world_understanding.functions.graphics.rendering import (
    CameraFocusMode,
    CameraViewType,
    OvRTXRenderingBackend,
    RemoteRenderingBackend,
    RenderingConfig,
    format_direction_for_filename,
)
from world_understanding.utils.image_blankness import analyze_image_blankness
from world_understanding.utils.image_utils import paste_on_background
from world_understanding.utils.usd.camera import add_corner_view_camera
from world_understanding.utils.usd.stage import prepare_stage_for_render

logger = logging.getLogger(__name__)


class BlankRenderStatsDict(TypedDict):
    """JSON-safe shape emitted by ImageBlanknessStats.to_dict()."""

    blank: bool
    reason: str | None
    width: int
    height: int
    mode: str
    sampled_pixels: int
    unique_colors: int
    dominant_color_ratio: float
    luma_std: float
    luma_dynamic_range: float
    rgb_dynamic_range: float
    strong_minority_pixel_ratio: float
    alpha_visible_ratio: float | None


class RenderTask(Task):
    """Unified task to flatten and render a USD file.

    This task handles multiple usage patterns:
    - Standalone rendering with explicit paths
    - Workflow-integrated rendering with flexible context keys
    - Optional rendering that can be skipped via flag
    - Optional flattening before rendering

    Input context keys (flexible):
        Path inputs (in priority order):
        - output_usd_path: USD file with materials applied (preferred)
        - render_usd_path: Pre-flattened USD for rendering (workflow usage)
        - input_usd_path: Original input USD file path (fallback)

        Output directory (in priority order):
        - output_base_path: Explicit output directory
        - render_output_dir: Alternative output directory name
        - (defaults to input USD parent directory)

        Configuration:
        - render_enabled: Whether to perform rendering (default: True)
        - flatten_before_render: Whether to flatten USD before rendering (default: True)
        - render_config: Rendering configuration dictionary with:
            - backend: Rendering backend ("remote" or "ovrtx", default: "remote")
            - image_width: Image width in pixels (default: 1024)
            - image_height: Image height in pixels (default: image_width)
            - camera_corners: List of camera corners to render from (default: ["+x+y+z"])
            - camera_corner: Alternative single corner specification
            - camera_margin: Camera margin multiplier (default: 1.2)
            - background_color: Background color as [R, G, B] in 0-1 range (default: [1.0, 1.0, 1.0])
            - max_retries: REST renderer retry count (optional)
            - retry_delay: REST renderer retry delay (optional)
            - retry_backoff_factor: REST renderer backoff factor (optional)
            - retry_jitter: REST renderer jitter (optional)
            OvRTX-specific keys (backend == "ovrtx"):
            - log_level: Logging verbosity for OvRTX subprocess (str, default: "warn")
            - ovrtx_venv_dir: Path to the OvRTX virtual environment directory
                (str, optional; defaults to ~/.cache/wu/ovrtx_venv)
            - num_sensor_updates: Progressive path-tracer step iterations
                per frame (int, default: 500). Sized for ``render_mode="pt"``;
                rt2 caps quality regardless of step count, so paired with
                ``pt`` is the configuration used here.
            - render_mode: OvRTX render mode (``rt1`` | ``rt2`` | ``pt``,
                default ``"pt"``). The material-agent default is ``pt``
                (Kit's ground-truth mode) because final renders here are
                presentation-quality. The 500-step ``num_sensor_updates``
                default is sized for the ``pt`` convergence plateau; ``rt2``
                caps at ~27 dB PSNR regardless of step count and is the
                fast-iteration default of the underlying backend, which is
                not what this task needs.

    Output context keys:
        - flattened_usd_path: Path to flattened USD (if flattening was done)
        - rendered_image_paths: List of all rendered image paths
        - rendered_image_path: Path to the first rendered image (backward compatibility)
        - rendering_skipped: Boolean indicating if rendering was skipped
        - rendering_stats: Dictionary with rendering statistics
    """

    def __init__(self):
        """Initialize the render task."""
        self.name = "Render"
        self.description = "Flatten and render USD file"

    def run(self, context: dict[str, Any], object_store=None) -> dict[str, Any]:
        """Flatten and render the USD file.

        Args:
            context: Workflow context
            object_store: Optional object store (not used)

        Returns:
            Updated context with rendering results
        """
        # Get event listener (or logger fallback)
        listener = get_listener(context, logger_name=__name__)

        # Test listener immediately
        listener.info("🎬 Render task starting...")

        # Check if rendering is enabled (default True for backward compatibility)
        render_enabled = context.get("render_enabled", True)
        if not render_enabled:
            listener.info(
                "Rendering is disabled (render_enabled=False), skipping render task"
            )
            context["rendering_skipped"] = True
            return context

        # Get USD file path - support multiple context key patterns
        # Priority: output_usd_path > render_usd_path > input_usd_path
        # Note: output_usd_path takes priority because it represents the freshly
        # created USD file with materials applied (e.g., from apply/assign steps)
        input_usd_path = (
            context.get("output_usd_path")
            or context.get("render_usd_path")
            or context.get("input_usd_path")
        )

        if not input_usd_path:
            raise ValueError(
                "No USD file path found. Provide one of: output_usd_path, render_usd_path, or input_usd_path"
            )

        input_usd_path = Path(input_usd_path)
        if not input_usd_path.exists():
            listener.warning(
                f"USD file not found: {input_usd_path}, skipping rendering"
            )
            context["rendering_skipped"] = True
            return context

        # Get output directory - support multiple context key patterns
        # Priority: output_base_path > render_output_dir > input USD parent dir
        output_base_path = context.get("output_base_path") or context.get(
            "render_output_dir"
        )
        if not output_base_path:
            output_base_path = input_usd_path.parent
            listener.info(
                f"No output directory specified, using input USD parent: {output_base_path}"
            )

        output_base_path = Path(output_base_path)
        output_base_path.mkdir(parents=True, exist_ok=True)

        render_config = context.get("render_config", {})
        flatten_before_render = context.get("flatten_before_render", True)

        # Track which USD path to use for output naming (before potential flattening)
        original_usd_path = input_usd_path

        listener.info(f"Rendering USD: {input_usd_path}")
        listener.info(f"Output directory: {output_base_path}")
        listener.info(f"Flatten before render: {flatten_before_render}")

        # Step 1: Prepare the USD stage for rendering
        stage = Usd.Stage.Open(str(input_usd_path))
        if not stage:
            raise RuntimeError(f"Failed to open USD stage: {input_usd_path}")

        prepared_stage, preparation_metadata = prepare_stage_for_render(
            stage,
            flatten=flatten_before_render,
            normalize_materials=True,
        )
        render_asset_base_dir = preparation_metadata.get("asset_base_dir")
        listener.info(f"Render stage preparation: {preparation_metadata}")

        # Export the prepared stage. Flattening produces a self-contained stage;
        # the non-flatten path still writes a converted temp layer so the
        # original USD is not mutated.
        if flatten_before_render:
            flattened_usd_path = output_base_path / f"{input_usd_path.stem}_flat.usd"
            listener.info(f"Flattening USD to: {flattened_usd_path}")

            try:
                # Save the flattened stage
                prepared_stage.GetRootLayer().Export(str(flattened_usd_path))

                listener.info(f"✓ Flattened USD saved to: {flattened_usd_path}")
                context["flattened_usd_path"] = str(flattened_usd_path)

                # Use flattened USD for rendering
                usd_to_render = flattened_usd_path

            except Exception as e:
                listener.error(f"Failed to flatten USD: {e}")
                raise RuntimeError(f"USD flattening failed: {e}") from e
        else:
            converted_path = output_base_path / f"{input_usd_path.stem}_converted.usda"
            prepared_stage.GetRootLayer().Export(str(converted_path))
            usd_to_render = converted_path
            listener.info(f"Converted MDL shaders, rendering from: {converted_path}")

        # Step 2: Render the USD
        listener.info(f"Starting rendering from: {usd_to_render}")

        # Extract render settings
        backend_type = render_config.get("backend", "remote")
        image_width = render_config.get("image_width", 1024)
        image_height = render_config.get("image_height", image_width)

        # Support both single corner (string) and multiple corners (list)
        camera_corners_config = render_config.get(
            "camera_corners"
        ) or render_config.get("camera_corner", "+x+y+z")
        if isinstance(camera_corners_config, str):
            camera_corners = [camera_corners_config]
        else:
            camera_corners = camera_corners_config

        camera_margin = render_config.get("camera_margin", 1.2)

        # Background color: config uses 0-1 range, convert to 0-255 for PIL
        bg_color_normalized = render_config.get("background_color", [1.0, 1.0, 1.0])
        background_color = tuple(int(c * 255) for c in bg_color_normalized)

        listener.info("Rendering configuration:")
        listener.info(f"  Backend: {backend_type}")
        listener.info(f"  Image size: {image_width}x{image_height}")
        listener.info(
            f"  Camera corners: {', '.join(camera_corners)} ({len(camera_corners)} views)"
        )
        listener.info(f"  Camera margin: {camera_margin}")
        listener.info(f"  Background color: {background_color}")

        # Open the USD stage for rendering
        stage = Usd.Stage.Open(str(usd_to_render))
        if not stage:
            raise RuntimeError(
                f"Failed to open USD stage for rendering: {usd_to_render}"
            )

        # Calculate apertures based on desired aspect ratio
        aspect_ratio = image_width / image_height
        if aspect_ratio >= 1.0:
            # Landscape or square: keep horizontal at 36, adjust vertical
            horizontal_aperture = 36.0
            vertical_aperture = 36.0 / aspect_ratio
        else:
            # Portrait: keep vertical at 36, adjust horizontal
            vertical_aperture = 36.0
            horizontal_aperture = 36.0 * aspect_ratio

        listener.info(
            f"Camera apertures: {horizontal_aperture:.2f}mm x {vertical_aperture:.2f}mm "
            f"(aspect ratio: {aspect_ratio:.2f})"
        )

        # Clear existing material bindings before rendering if requested.
        # This shows only the newly-assigned materials from the pipeline,
        # making it easier to verify predictions against a neutral surface.
        clear_materials = render_config.get("clear_materials", False)
        if clear_materials:
            from world_understanding.utils.usd.prim import nullify_materials

            listener.info("Clearing original material bindings (clear_materials=True)")
            nullify_materials(stage)

        # Scope to prim_path: hide everything outside the subtree and
        # focus the camera on the target prim only.
        prim_path = render_config.get("prim_path")
        focus_prim = stage.GetPrimAtPath(prim_path) if prim_path else None
        if focus_prim and focus_prim.IsValid():
            listener.info(f"Isolating prim for render: {prim_path}")
            from world_understanding.functions.graphics.rendering import (
                hide_prims_outside_subtree,
            )

            hide_prims_outside_subtree(stage, prim_path)
            listener.info(f"Hidden prims outside {prim_path} subtree")
        elif prim_path:
            listener.warning(f"Prim '{prim_path}' not found, rendering full scene")
            prim_path = None
            focus_prim = None

        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]
        )
        if focus_prim:
            scene_bbox = bbox_cache.ComputeWorldBound(focus_prim)
        else:
            scene_bbox = bbox_cache.ComputeWorldBound(stage.GetPseudoRoot())
        aligned_range = scene_bbox.ComputeAlignedRange()
        bbox_min = aligned_range.GetMin()
        bbox_max = aligned_range.GetMax()

        scene_size_x = bbox_max[0] - bbox_min[0]
        scene_size_y = bbox_max[1] - bbox_min[1]
        scene_size_z = bbox_max[2] - bbox_min[2]

        listener.info(
            f"Scene bounding box: [{bbox_min[0]:.2f}, {bbox_min[1]:.2f}, {bbox_min[2]:.2f}] to "
            f"[{bbox_max[0]:.2f}, {bbox_max[1]:.2f}, {bbox_max[2]:.2f}]"
        )
        listener.info(
            f"Scene dimensions: {scene_size_x:.2f} × {scene_size_y:.2f} × {scene_size_z:.2f}"
        )

        # Set up rendering backend
        if backend_type == "remote":
            import os

            api_key = os.environ.get("NGC_API_KEY")
            remote_kwargs = {"api_key": api_key}

            for key in (
                "base_url",
                "s3_bucket",
                "s3_region",
                "s3_profile",
                "timeout",
                "max_retries",
                "retry_delay",
                "retry_backoff_factor",
                "retry_jitter",
                "bundle_mdl_assets",
                "use_data_uri",
            ):
                if key in render_config:
                    remote_kwargs[key] = render_config[key]

            rendering_backend = RemoteRenderingBackend(**remote_kwargs)
            listener.info(
                f"Using remote REST renderer with retry config: max_retries={remote_kwargs.get('max_retries', 3)}, "
                f"retry_delay={remote_kwargs.get('retry_delay', 1.0)}"
            )
        elif backend_type == "ovrtx":
            # Pin render_mode to ``pt`` because num_sensor_updates=500 is
            # sized for the PT convergence plateau. The shared backend's
            # default flipped to ``rt2`` for fast iteration, which caps
            # quality regardless of step count — pairing 500 steps with
            # rt2 would pay the latency without gaining quality.
            ovrtx_kwargs: dict[str, Any] = {
                "log_level": render_config.get("log_level", "warn"),
                "ovrtx_venv_dir": render_config.get("ovrtx_venv_dir"),
                "num_sensor_updates": render_config.get("num_sensor_updates", 500),
                "render_mode": render_config.get("render_mode", "pt"),
            }
            rendering_backend = OvRTXRenderingBackend(**ovrtx_kwargs)
            listener.info(
                f"Using OvRTX backend with log_level={ovrtx_kwargs['log_level']}"
            )
        else:
            listener.error(f"Unknown rendering backend: {backend_type}")
            context["rendering_skipped"] = True
            return context
        # Set up rendering configuration
        rendering_config = RenderingConfig(
            image_width=image_width,
            cull_style="back",
            # For final render, don't modify materials or colors
            should_reset_materials=False,
            should_highlight_prim=False,
            should_assign_random_colors=False,
            # Use white background by default for clean presentation
            background_color=background_color,
            use_background_color=True,
            # Use lights if available in the scene
            use_lights=True,
            # Focus on the entire stage
            camera_focus_mode=CameraFocusMode.STAGE,
            camera_view_type=CameraViewType.CORNER,
        )

        # Create all cameras
        camera_infos = []
        listener.info(f"Creating {len(camera_corners)} camera(s)...")

        if focus_prim:
            from world_understanding.utils.usd.camera import (
                add_focused_corner_view_camera,
            )

        for i, camera_corner in enumerate(camera_corners):
            camera_path = (
                f"/RenderCamera_{i}" if len(camera_corners) > 1 else "/RenderCamera"
            )

            listener.info(
                f"  [{i + 1}/{len(camera_corners)}] Creating camera at {camera_path} (direction: {camera_corner})"
            )

            # Add corner view camera (scoped to prim bbox when set)
            if focus_prim:
                add_focused_corner_view_camera(
                    prim_to_focus=focus_prim,
                    camera_path=camera_path,
                    direction=camera_corner,
                    margin=camera_margin,
                    min_distance=0,
                    focal_length=50.0,
                    horizontal_aperture=horizontal_aperture,
                    vertical_aperture=vertical_aperture,
                    near_clip_margin=0.1,
                    far_clip_margin=0.1,
                )
            else:
                add_corner_view_camera(
                    stage,
                    margin=camera_margin,
                    camera_path=camera_path,
                    direction=camera_corner,
                    focal_length=50.0,
                    horizontal_aperture=horizontal_aperture,
                    vertical_aperture=vertical_aperture,
                    near_clip_margin=0.1,
                    far_clip_margin=0.1,
                )

            # Generate output filename with corner suffix if multiple cameras
            # Use the original USD path for naming (before flattening)
            base_name = original_usd_path.stem

            if len(camera_corners) > 1:
                # Use standard direction formatting: "+x+y+z" -> "posx_posy_posz"
                corner_suffix = format_direction_for_filename(camera_corner)
                output_filename = f"{base_name}_{corner_suffix}.png"
            else:
                output_filename = f"{base_name}.png"

            output_image_path = output_base_path / output_filename

            camera_infos.append(
                {
                    "camera_path": camera_path,
                    "camera_corner": camera_corner,
                    "output_path": output_image_path,
                    "index": i,
                }
            )

        # Save the stage with all cameras
        stage.Save()

        # Define rendering function for parallel execution
        def render_single_camera(camera_info: dict) -> dict:
            """Render a single camera view."""
            camera_path = camera_info["camera_path"]
            output_path = camera_info["output_path"]
            corner = camera_info["camera_corner"]
            index = camera_info["index"]

            listener.info(
                f"[{index + 1}/{len(camera_corners)}] Rendering {corner} to {output_path.name}"
            )

            try:
                # Remote render functions can occasionally return HTTP 200
                # with body {"status": "exception"} on single full-scene renders
                # (seen on the final post-apply render step in CI). Retry a
                # couple of times before giving up on the camera.
                max_attempts = 3 if backend_type == "remote" else 1
                render_result = None
                for attempt in range(max_attempts):
                    render_result = rendering_backend.render(
                        stage=stage,
                        cameras=[camera_path],
                        image_width=image_width,
                        image_height=image_height,
                        cull_style=rendering_config.cull_style,
                        frames="0",  # Single frame render
                        base_dir=render_asset_base_dir,
                    )
                    if render_result and render_result.get("successful_cameras", 0) > 0:
                        break
                    if attempt < max_attempts - 1:
                        attempt_error = "No successful renders returned"
                        if render_result and "results" in render_result:
                            for r in render_result["results"]:
                                if "error" in r:
                                    attempt_error = r["error"]
                                    break
                        listener.warning(
                            f"Render {corner} attempt {attempt + 1}/{max_attempts} failed: {attempt_error}; retrying"
                        )
                        time.sleep(2 * (attempt + 1))

                # Check if rendering was successful
                if not (
                    render_result
                    and render_result.get("successful_cameras", 0) > 0
                    and "results" in render_result
                    and len(render_result["results"]) > 0
                ):
                    error_msg = "No successful renders returned"
                    if render_result and "results" in render_result:
                        for result in render_result["results"]:
                            if "error" in result:
                                error_msg = result["error"]
                                break
                    return {
                        "success": False,
                        "error": error_msg,
                        "camera_path": camera_path,
                        "corner": corner,
                    }

                # Get the first camera result
                camera_result = render_result["results"][0]

                # Save the image
                if not ("images" in camera_result and camera_result["images"]):
                    return {
                        "success": False,
                        "error": "No image data in result",
                        "camera_path": camera_path,
                        "corner": corner,
                    }

                # Get the first image (we only rendered one frame)
                image = camera_result["images"][0]

                # Check if it's a PIL Image or raw data
                if hasattr(image, "save"):
                    # It's a PIL Image, use it directly
                    pass
                elif isinstance(image, dict) and "image" in image:
                    # For remote REST backends, image data might be in a dict.
                    if isinstance(image["image"], bytes):
                        img_bytes = image["image"]
                    else:
                        # Decode base64 if needed
                        img_bytes = base64.b64decode(image["image"])
                    image = Image.open(BytesIO(img_bytes))
                else:
                    return {
                        "success": False,
                        "error": "Unexpected image format",
                        "camera_path": camera_path,
                        "corner": corner,
                    }

                blank_stats = analyze_image_blankness(image)
                if blank_stats.blank:
                    return {
                        "success": False,
                        "error": _blank_final_render_error(
                            str(output_path),
                            cast(BlankRenderStatsDict, blank_stats.to_dict()),
                        ),
                        "blank_render": True,
                        "blank_stats": blank_stats.to_dict(),
                        "camera_path": camera_path,
                        "corner": corner,
                    }

                # Apply background color if specified
                if rendering_config.use_background_color:
                    # Convert to RGBA if needed
                    if image.mode != "RGBA":
                        image = image.convert("RGBA")

                    # Apply background color
                    image = paste_on_background(image, background_color)

                # Save the final image
                image.save(str(output_path))

                listener.info(f"✓ Successfully rendered {corner} to {output_path.name}")

                # Emit per-camera rendering event
                try:
                    listener.event(
                        "rendering.completed",
                        {
                            "camera_corner": corner,
                            "output_path": str(output_path),
                            "image_width": image_width,
                            "image_height": image_height,
                            "backend": backend_type,
                        },
                    )
                except Exception as e:
                    logger.warning(f"Failed to emit rendering event for {corner}: {e}")

                return {
                    "success": True,
                    "output_path": str(output_path),
                    "camera_path": camera_path,
                    "corner": corner,
                }

            except Exception as e:
                listener.error(f"Rendering from {corner} failed: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "camera_path": camera_path,
                    "corner": corner,
                }

        # Render all cameras in parallel
        listener.info(f"Rendering {len(camera_infos)} view(s) in parallel...")
        rendered_image_paths = []
        failed_renders = []

        # Determine max workers based on number of cameras and backend
        max_workers = min(len(camera_infos), 4 if backend_type == "remote" else 2)
        listener.info(f"Using {max_workers} parallel workers")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all rendering tasks
            future_to_camera = {
                executor.submit(render_single_camera, cam_info): cam_info
                for cam_info in camera_infos
            }

            # Process results as they complete
            for future in as_completed(future_to_camera):
                cam_info = future_to_camera[future]
                try:
                    result = future.result()
                    if result["success"]:
                        rendered_image_paths.append(result["output_path"])
                    else:
                        failed_renders.append(result)
                        listener.error(
                            f"Failed to render {result['corner']}: {result.get('error', 'Unknown error')}"
                        )
                except Exception as e:
                    listener.error(
                        f"Exception rendering {cam_info['camera_corner']}: {e}"
                    )
                    failed_renders.append(
                        {
                            "success": False,
                            "error": str(e),
                            "corner": cam_info["camera_corner"],
                        }
                    )

        # Remove render cameras from the stage so they don't pollute the
        # output USD served to users (the flat file is also the download).
        for cam_info in camera_infos:
            cam_path = cam_info["camera_path"]
            if stage.GetPrimAtPath(cam_path):
                stage.RemovePrim(cam_path)
        stage.Save()
        listener.info("Cleaned up render camera prims from output USD")

        # Check if any renders failed
        if failed_renders:
            blank_renders = [
                result for result in failed_renders if result.get("blank_render")
            ]
            if blank_renders:
                raise RuntimeError(
                    "One or more final renders are blank or near-blank. "
                    f"First error: {blank_renders[0].get('error', 'Unknown')}"
                )
            listener.warning(
                f"{len(failed_renders)}/{len(camera_infos)} renders failed"
            )
            if len(rendered_image_paths) == 0:
                # All renders failed
                raise RuntimeError(
                    f"All {len(camera_infos)} camera renders failed. First error: {failed_renders[0].get('error', 'Unknown')}"
                )

        # Update context with results
        if rendered_image_paths:
            context["rendered_image_paths"] = rendered_image_paths
            context["rendered_image_path"] = rendered_image_paths[
                0
            ]  # Backward compatibility
            context["rendering_skipped"] = False
            context["rendering_stats"] = {
                "total_images": len(rendered_image_paths),
                "failed_renders": len(failed_renders),
                "image_width": image_width,
                "image_height": image_height,
                "backend": backend_type,
            }

            listener.info("✓ Rendering complete:")
            listener.info(f"  • Total images rendered: {len(rendered_image_paths)}")
            listener.info(f"  • Failed renders: {len(failed_renders)}")
            listener.info(f"  • Image size: {image_width}x{image_height}")
            for img_path in rendered_image_paths:
                listener.info(f"  • {img_path}")

            # Emit overall rendering completion event
            try:
                listener.event(
                    "rendering.all_completed",
                    {
                        "total_images": len(rendered_image_paths),
                        "failed_renders": len(failed_renders),
                        "image_width": image_width,
                        "image_height": image_height,
                        "backend": backend_type,
                        "rendered_image_paths": rendered_image_paths,
                        "camera_corners": camera_corners,
                    },
                )
            except Exception as e:
                logger.warning(f"Failed to emit overall rendering event: {e}")
        else:
            context["rendering_skipped"] = True

        return context


def _blank_final_render_error(
    output_path: str,
    stats: BlankRenderStatsDict,
) -> str:
    """Format final-render blankness metrics into a user-facing error."""
    dominant_color_ratio = float(stats.get("dominant_color_ratio", 0.0))
    luma_std = float(stats.get("luma_std", 0.0))
    return (
        f"Final render output is blank or near-blank: {output_path}. "
        f"reason={stats.get('reason')}, unique_colors={stats.get('unique_colors')}, "
        f"dominant_color_ratio={dominant_color_ratio:.3f}, "
        f"luma_std={luma_std:.3f}. Check the rendering endpoint logs "
        "and any HDRI / dome-light configuration "
        "(WU_OVRTX_DEFAULT_HDRI, WU_OVRTX_DEFAULT_HDRI_INTENSITY)."
    )
