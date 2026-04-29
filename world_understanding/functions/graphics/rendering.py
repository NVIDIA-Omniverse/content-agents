# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import logging
import os
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, ClassVar

from pxr import Usd, UsdGeom

from world_understanding.config.s3 import WU_S3_BUCKET, WU_S3_PROFILE, WU_S3_REGION
from world_understanding.functions.graphics import render_nvcf
from world_understanding.utils.data_uri import should_use_data_uri
from world_understanding.utils.image_utils import (
    draw_bounding_box_on_red,
    extract_non_black_outline,
    extract_red_outline,
    paste_on_background,
    paste_outline_to_image,
)
from world_understanding.utils.usd.camera import (
    DEFAULT_CAMERA_ORDERING,
    add_corner_view_camera,
    add_focused_corner_view_camera,
    add_focused_side_view_camera,
    add_side_view_camera,
)
from world_understanding.utils.usd.prim import (
    disable_visibility_for_all_mesh_prims,
    nullify_materials,
    remove_all_lights,
    set_mesh_display_color,
    traverse_meshes,
)

logger = logging.getLogger(__name__)


def format_direction_for_filename(direction: str) -> str:
    """Convert direction string to filename-safe format.

    Transforms '+x' to 'posx' and '-x' to 'negx' for filesystem compatibility.

    Args:
        direction: Direction string (e.g., "+x", "-x+y", "+x+y+z")

    Returns:
        Filename-safe version (e.g., "posx", "negx_posy", "posx_posy_posz")

    Examples:
        >>> format_direction_for_filename("+x")
        'posx'
        >>> format_direction_for_filename("-x+y-z")
        'negx_posy_negz'
        >>> format_direction_for_filename("+x-y")
        'posx_negy'
    """
    # Replace axis directions: +x → posx, -x → negx
    result = direction.replace("+x", "posx").replace("-x", "negx")
    result = result.replace("+y", "_posy").replace("-y", "_negy")
    result = result.replace("+z", "_posz").replace("-z", "_negz")

    # Remove any leading underscore if it exists (shouldn't happen, but just in case)
    return result.lstrip("_")


def parse_camera_angle_from_view_name(view_name: str) -> str:
    """Convert view name back to human-readable camera angle.

    Transforms 'posx' to '+X', 'negx_posy_posz' to '+X+Y+Z', etc.

    Args:
        view_name: View name from filename (e.g., "posx", "negx_posy_negz")

    Returns:
        Human-readable camera angle (e.g., "+X", "-X+Y-Z")

    Examples:
        >>> parse_camera_angle_from_view_name("posx")
        '+X'
        >>> parse_camera_angle_from_view_name("negx_posy_negz")
        '-X+Y-Z'
        >>> parse_camera_angle_from_view_name("posx_negy")
        '+X-Y'
    """
    # Replace filename-safe format back to direction format
    result = view_name.replace("posx", "+X").replace("negx", "-X")
    result = result.replace("_posy", "+Y").replace("_negy", "-Y")
    result = result.replace("_posz", "+Z").replace("_negz", "-Z")
    return result


class CameraViewType(Enum):
    """Enum for camera view types."""

    CORNER = "corner"  # Uses add_focused_corner_view_camera for 8 corner views
    SIDE = "side"  # Uses add_focused_side_view_camera for 6 cardinal views


class CameraFocusMode(Enum):
    """Enum for camera focus modes."""

    PRIM = "prim"  # Focus on individual prims
    STAGE = "stage"  # Focus on the entire stage


@dataclass
class CameraSpec:
    """Specification for a single camera view.

    This class represents a camera configuration with all its parameters.
    It supports the new multi-view camera configuration system.
    """

    # Required
    direction: str  # e.g., "+x+y+z", "-z"

    # Optional - inferred or use defaults if not specified
    view_type: CameraViewType | None = None  # Inferred from direction if None
    margin: float | None = None
    focal_length: float | None = None
    horizontal_aperture: float | None = None
    vertical_aperture: float | None = None
    near_clip_margin: float | None = None
    far_clip_margin: float | None = None

    # Optional metadata
    name: str | None = None  # User-friendly name for this camera

    def __post_init__(self):
        """Infer view_type from direction if not specified."""
        if self.view_type is None:
            # Infer from direction: corner views have 3 axes, side views have 1
            num_axes = sum(1 for c in self.direction if c in "xyz")
            self.view_type = (
                CameraViewType.CORNER if num_axes >= 2 else CameraViewType.SIDE
            )

    def to_camera_ordering_format(self) -> str:
        """Convert to legacy camera_ordering format (just the direction string)."""
        return self.direction

    def merge_with_defaults(
        self,
        default_margin: float = 1.0,
        default_focal_length: float = 100.0,
        default_horizontal_aperture: float = 1.0,
        default_vertical_aperture: float = 1.0,
        default_near_clip_margin: float = 0.1,
        default_far_clip_margin: float = 0.1,
    ) -> "CameraSpec":
        """Return a new CameraSpec with defaults filled in for None values."""
        return CameraSpec(
            direction=self.direction,
            view_type=self.view_type,
            margin=self.margin if self.margin is not None else default_margin,
            focal_length=(
                self.focal_length
                if self.focal_length is not None
                else default_focal_length
            ),
            horizontal_aperture=(
                self.horizontal_aperture
                if self.horizontal_aperture is not None
                else default_horizontal_aperture
            ),
            vertical_aperture=(
                self.vertical_aperture
                if self.vertical_aperture is not None
                else default_vertical_aperture
            ),
            near_clip_margin=(
                self.near_clip_margin
                if self.near_clip_margin is not None
                else default_near_clip_margin
            ),
            far_clip_margin=(
                self.far_clip_margin
                if self.far_clip_margin is not None
                else default_far_clip_margin
            ),
            name=self.name,
        )


@dataclass
class RenderingConfig:
    """Configuration for rendering USD stages."""

    # General rendering options
    cull_style: str = "back"
    image_width: int = 1024
    image_border_size: int = 0
    use_lights: bool = False
    strip_existing_animation: bool = True  # Remove time-sampled attrs before rendering

    # Camera options
    camera_view_type: CameraViewType = CameraViewType.CORNER
    camera_focus_mode: CameraFocusMode = CameraFocusMode.PRIM
    camera_name_prefix: str = "SideViewCamera"
    camera_focal_length: float = 100.0
    camera_horizontal_aperture: float = 1.0
    camera_vertical_aperture: float = 1.0
    camera_ordering: list[str] = field(default_factory=lambda: DEFAULT_CAMERA_ORDERING)
    camera_prim_focus_margin: float = 1.1
    camera_prim_with_stage_margin: float = 3.0  # Higher margin for prim_with_stage mode
    camera_composition_margin: float = 3.0  # Camera margin for composition mode
    near_clip_margin: float = 0.1
    far_clip_margin: float = 0.1

    # NEW: Per-mode camera specifications
    # Dict mapping render_mode -> list of CameraSpec
    # Special key "__all__" means applies to all modes
    camera_specs: dict[str, list[CameraSpec]] = field(default_factory=dict)

    # Background options
    background_color: tuple[int, int, int] = (0, 0, 0)
    use_background_color: bool = False

    # Prim-focused rendering options
    should_reset_materials: bool = True
    should_render_prim_only: bool = True
    should_highlight_prim: bool = False
    should_assign_random_colors: bool = True
    highlight_color: tuple[float, float, float] = (1.0, 0.0, 0.0)
    other_color_range: tuple[float, float] = (0.35, 0.35)

    # Composition options
    composition_show_full_scene: bool = True  # Show all prims in plain stage
    enable_contour: bool = True
    contour_color: tuple[float, float, float] = (1.0, 0.5, 0.0)  # RGB 0.0-1.0
    contour_method: str = "red"  # "red" or "non_black" - method for contour detection
    contour_black_threshold: int = (
        20  # Threshold for non_black method (pixels <= this are black)
    )
    enable_bbox: bool = False
    bbox_color: tuple[float, float, float] = (0.0, 1.0, 0.0)  # RGB 0.0-1.0

    # Occlusion detection options (global defaults)
    skip_occluded_images: bool = False  # Skip images where prim is completely occluded
    occlusion_pixel_threshold: int = (
        10  # Minimum visible pixels to consider prim not occluded
    )

    # Per-mode occlusion settings
    # Dict mapping render_mode -> skip_occluded_images bool
    # If a mode is not in this dict, uses the global skip_occluded_images setting
    per_mode_skip_occluded: dict[str, bool] = field(default_factory=dict)

    # Per-mode camera focus settings
    # Dict mapping render_mode -> camera_focus_mode
    # If a mode is not in this dict, uses the global camera_focus_mode setting
    per_mode_focus_mode: dict[str, CameraFocusMode] = field(default_factory=dict)

    # Per-mode original material settings
    # Dict mapping render_mode -> use_original_materials bool
    per_mode_use_original_materials: dict[str, bool] = field(default_factory=dict)

    # Per-mode base mode mapping (custom name -> core mode type)
    # Dict mapping render_mode -> base_mode (e.g., "prim_only_original" -> "prim_only")
    per_mode_base_mode: dict[str, str] = field(default_factory=dict)

    # Root prim path for scoping camera and visibility to a subtree
    # When set, STAGE focus mode computes bbox from this prim instead of
    # the pseudo-root, and composition mode hides prims outside this subtree.
    root_prim_path: str | None = None

    def should_use_original_materials_for_mode(self, render_mode: str) -> bool:
        """Get use_original_materials setting for a specific render mode.

        Args:
            render_mode: The rendering mode (e.g., "prim_only", "prim_only_original")

        Returns:
            True if original materials should be preserved for this mode
        """
        return self.per_mode_use_original_materials.get(render_mode, False)

    def get_base_mode(self, render_mode: str) -> str:
        """Get the base mode for a render mode.

        Custom mode names (e.g., "prim_only_original") map back to core mode types
        (e.g., "prim_only") for dispatch. If no mapping exists, the mode name is
        returned as-is.

        Args:
            render_mode: The rendering mode name

        Returns:
            The core mode type for dispatch
        """
        return self.per_mode_base_mode.get(render_mode, render_mode)

    def get_focus_mode_for_mode(self, render_mode: str) -> CameraFocusMode:
        """Get camera_focus_mode for a specific render mode.

        Args:
            render_mode: The rendering mode (e.g., "prim_only", "prim_with_stage", "composition")

        Returns:
            The camera focus mode for this render mode
        """
        # Check for mode-specific setting first
        if render_mode in self.per_mode_focus_mode:
            return self.per_mode_focus_mode[render_mode]
        # Fall back to global setting
        return self.camera_focus_mode

    def should_skip_occluded_for_mode(self, render_mode: str) -> bool:
        """Get skip_occluded_images setting for a specific render mode.

        Args:
            render_mode: The rendering mode (e.g., "prim_only", "prim_with_stage", "composition")

        Returns:
            True if occluded images should be skipped for this mode
        """
        # Check for mode-specific setting first
        if render_mode in self.per_mode_skip_occluded:
            return self.per_mode_skip_occluded[render_mode]
        # Fall back to global setting
        return self.skip_occluded_images

    def get_cameras_for_mode(self, render_mode: str) -> list[CameraSpec]:
        """Get camera specifications for a specific render mode.

        This method supports the new multi-view camera configuration system.
        It returns per-mode cameras if configured, falls back to global cameras,
        or builds from legacy fields.

        Args:
            render_mode: The rendering mode (e.g., "prim_only", "prim_with_stage")

        Returns:
            List of CameraSpec objects for the given render mode
        """
        # Check for mode-specific cameras
        if render_mode in self.camera_specs:
            return self._fill_camera_defaults(
                self.camera_specs[render_mode], render_mode
            )

        # Check for global cameras (applies to all modes)
        if "__all__" in self.camera_specs:
            return self._fill_camera_defaults(self.camera_specs["__all__"], render_mode)

        # Fallback: Build from legacy fields
        return self._legacy_cameras(render_mode)

    def _fill_camera_defaults(
        self, cameras: list[CameraSpec], render_mode: str
    ) -> list[CameraSpec]:
        """Fill in None values in CameraSpecs with appropriate defaults."""
        # Determine margin based on render mode if not specified
        default_margin = self._get_default_margin_for_mode(render_mode)

        return [
            cam.merge_with_defaults(
                default_margin=default_margin,
                default_focal_length=self.camera_focal_length,
                default_horizontal_aperture=self.camera_horizontal_aperture,
                default_vertical_aperture=self.camera_vertical_aperture,
                default_near_clip_margin=self.near_clip_margin,
                default_far_clip_margin=self.far_clip_margin,
            )
            for cam in cameras
        ]

    def _get_default_margin_for_mode(self, render_mode: str) -> float:
        """Get the default margin for a render mode based on legacy config."""
        if render_mode == "prim_with_stage":
            return self.camera_prim_with_stage_margin
        elif render_mode == "composition":
            return self.camera_composition_margin
        else:  # prim_only, sensors, etc.
            return self.camera_prim_focus_margin

    def _legacy_cameras(self, render_mode: str) -> list[CameraSpec]:
        """Build CameraSpec list from legacy fields for backward compatibility."""
        margin = self._get_default_margin_for_mode(render_mode)

        return [
            CameraSpec(
                direction=direction,
                view_type=self.camera_view_type,
                margin=margin,
                focal_length=self.camera_focal_length,
                horizontal_aperture=self.camera_horizontal_aperture,
                vertical_aperture=self.camera_vertical_aperture,
                near_clip_margin=self.near_clip_margin,
                far_clip_margin=self.far_clip_margin,
            )
            for direction in self.camera_ordering
        ]


class RenderingBackend(ABC):
    """Abstract base class for USD rendering backends."""

    def supports_sensors(self) -> bool:
        """Return True if this backend supports sensor rendering modes.

        Returns:
            bool: True if backend can render sensor modes (depth, segmentation, etc.)
        """
        return False

    def get_supported_sensor_modes(self) -> list[str]:
        """Return list of sensor rendering modes supported by this backend.

        Returns:
            list[str]: List of supported sensor mode names (e.g., ["linear_depth", "instance_id_segmentation"])
        """
        return []

    @abstractmethod
    def render(
        self,
        stage: Usd.Stage,
        cameras: list[str] | None = None,
        image_width: int = 1024,
        image_height: int | None = None,
        cull_style: str = "back",  # "back", "front", "none"
        frames: str = "0",
        renderer: str = "GL",
        sensors: list[str] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Render multiple cameras from a USD stage and return the rendered images and metadata.

        This method renders one or more camera views from a USD stage. If no cameras are
        specified, a default camera will be used (e.g., ["Camera"] or ["/Camera"]).

        Args:
            stage: USD stage to render
            cameras: List of camera paths to render. If None, uses default camera list.
                Examples: ["Camera"], ["/World/Camera"], ["Camera1", "Camera2"]
            image_width: Output image width in pixels. Default: 1024
            image_height: Output image height in pixels. If None, calculated from
                camera aspect ratio (for backends that support it). Default: None
            cull_style: Face culling style - "none", "back", or "front".
                Default: "back"
            frames: Frame(s) to render. Can be:
                - Single frame: "0", "42"
                - Frame range: "0:10", "5:15"
                - Comma-separated: "0,5,10"
                Default: "0" (first frame)
            renderer: Renderer to use - "GL" (OpenGL) or "RenderMan".
                Default: "GL"
            sensors: Additional sensors to render (e.g., ["normal", "depth", "linear_depth"]).
                Only supported by some backends. Default: None
            **kwargs: Additional backend-specific parameters

        Returns:
            Dict containing:
                - total_cameras: Number of cameras rendered
                - successful_cameras: Number of successfully rendered cameras
                - failed_cameras: Number of failed camera renders
                - total_render_time: Total time for all renders in seconds
                - results: List of individual camera render results, each containing:
                    - camera: Camera name used for rendering
                    - images: List of PIL Image objects (or dict for NVCF backend)
                    - render_time: Rendering time in seconds
                    - frame_count: Number of frames rendered
                    - return_code: Process return code (0 for success, if applicable)
                    - command: Full command executed (for debugging, if applicable)
                    - sensors: Dict of sensor data (if sensors requested and supported)
                    - status: Rendering status (for NVCF backend)
                    - error: Error message if rendering failed (optional)

        Raises:
            ValueError: If input parameters are invalid
            RuntimeError: If rendering fails

        Example:
            >>> from pxr import Usd, UsdGeom
            >>> stage = Usd.Stage.CreateInMemory()
            >>> # ... build your USD scene ...
            >>>
            >>> # Single camera (as a list)
            >>> result = backend.render(
            ...     stage=stage,
            ...     cameras=["Camera"],
            ...     image_width=1920
            ... )
            >>> print(f"Rendered {result['successful_cameras']} cameras")
            >>>
            >>> # Multiple cameras
            >>> result = backend.render(
            ...     stage=stage,
            ...     cameras=["Camera1", "Camera2", "Camera3"],
            ...     image_width=1920,
            ...     frames="0:10"
            ... )
            >>> for cam_result in result['results']:
            ...     if 'error' not in cam_result:
            ...         print(f"Camera {cam_result['camera']}: {cam_result['frame_count']} frames")
        """
        pass


class NVCFRenderingBackend(RenderingBackend):
    """USD rendering backend using NVIDIA Cloud Functions (NVCF) microservice.

    This backend uploads USD stages to S3 and uses NVCF for cloud-based rendering.
    It supports additional sensor outputs including linear_depth and instance_id_segmentation.
    """

    # Supported sensor modes for NVCF backend
    SUPPORTED_SENSOR_MODES: ClassVar[list[str]] = [
        "linear_depth",
        "depth",
        "instance_id_segmentation",
    ]

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        s3_bucket: str | None = None,
        s3_region: str | None = None,
        s3_profile: str | None = None,
        timeout: int = 3600,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        retry_backoff_factor: float = 2.0,
        retry_jitter: float = 0.1,
        bundle_mdl_assets: bool = True,
        use_data_uri: bool | None = None,
    ):
        """Initialize the NVCF rendering backend.

        Args:
            api_key: NVCF API key. If None, uses NGC_API_KEY env var
            base_url: NVCF base URL. If None, uses RENDER_ENDPOINT env var (or NVCF_RENDER_FUNCTION_ID fallback)
            s3_bucket: S3 bucket for stage upload. Explicit kwargs take
                      precedence. If unset, reads the current WU_S3_BUCKET env
                      var at backend instantiation time, then falls back to the
                      import-time config default.
            s3_region: AWS region where the S3 bucket is located.
                      Explicit kwargs take precedence. If unset, reads the
                      current WU_S3_REGION env var at backend instantiation
                      time, then falls back to the import-time config default.
            s3_profile: AWS profile for S3 upload. Explicit kwargs take
                       precedence. If unset, reads the current WU_S3_PROFILE
                       env var at backend instantiation time, then falls back
                       to the import-time config default.
            timeout: Request timeout in seconds. Default: 3600
            max_retries: Maximum number of retry attempts. Default: 3
            retry_delay: Initial delay between retries in seconds. Default: 1.0
            retry_backoff_factor: Factor to multiply delay by after each retry. Default: 2.0
            retry_jitter: Random jitter factor (0-1) to add to delays. Default: 0.1
            bundle_mdl_assets: If True, bundle local MDL assets with the USD file
                              into a ZIP archive for upload. Default: True
            use_data_uri: If True, base64-encode USD in request body instead of
                         uploading to S3. Default: reads MA_RENDERING_USE_DATA_URI env.
        """
        self.api_key = api_key
        self.base_url = base_url
        self.s3_bucket = s3_bucket or os.environ.get("WU_S3_BUCKET") or WU_S3_BUCKET
        self.s3_region = s3_region or os.environ.get("WU_S3_REGION") or WU_S3_REGION
        self.s3_profile = s3_profile or os.environ.get("WU_S3_PROFILE") or WU_S3_PROFILE
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.retry_backoff_factor = retry_backoff_factor
        self.retry_jitter = retry_jitter
        self.bundle_mdl_assets = bundle_mdl_assets
        self.use_data_uri = should_use_data_uri(use_data_uri)

    def supports_sensors(self) -> bool:
        """NVCF backend supports sensor rendering modes."""
        return True

    def get_supported_sensor_modes(self) -> list[str]:
        """Return list of sensor modes supported by NVCF backend."""
        return self.SUPPORTED_SENSOR_MODES.copy()

    def render(
        self,
        stage: Usd.Stage,
        cameras: list[str] | None = None,
        image_width: int = 1024,
        image_height: int | None = None,
        cull_style: str = "back",  # "back", "front", "none"
        frames: str = "0",
        renderer: str = "GL",
        sensors: list[str] | None = None,
        apply_background_mask: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        """Render multiple cameras from a USD stage using NVCF.

        Note: cull_style and renderer parameters are ignored as NVCF uses its own settings.

        Args:
            stage: USD stage to render
            cameras: List of camera paths to render. If None, uses ["/Camera"]
            image_width: Output image width in pixels
            image_height: Output image height in pixels. If None, defaults to image_width
            cull_style: Ignored (NVCF uses its own settings)
            frames: Frame(s) to render (e.g., "0", "0:10")
            renderer: Ignored (NVCF uses its own renderer)
            sensors: Additional sensors to render (e.g., ["linear_depth", "instance_id_segmentation"])
            apply_background_mask: If True, apply background masking during rendering. Default: False
            **kwargs: Additional parameters (ignored)

        Returns:
            Dict with rendering results matching the base class specification
        """
        # NVCF uses image_height, default to square if not specified
        if image_height is None:
            image_height = image_width

        # Note: NVCF doesn't use cull_style or renderer parameters
        # Could log warnings here if needed
        return render_nvcf.render_all_cameras(
            stage=stage,
            image_width=image_width,
            image_height=image_height,
            cameras=cameras,
            frames=frames,
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            sensors=sensors,
            apply_background_mask=apply_background_mask,
            s3_bucket=self.s3_bucket,
            s3_region=self.s3_region,
            s3_profile=self.s3_profile,
            max_workers=1,  # Disable per-camera parallelism
            max_retries=self.max_retries,
            retry_delay=self.retry_delay,
            retry_backoff_factor=self.retry_backoff_factor,
            retry_jitter=self.retry_jitter,
            bundle_mdl_assets=self.bundle_mdl_assets,
            use_data_uri=self.use_data_uri,
        )


class OvRTXRenderingBackend(RenderingBackend):
    """USD rendering backend using OvRTX local RTX renderer.

    This backend uses the ovrtx library for local RTX rendering. Because
    ovrtx bundles its own USD C libraries which conflict with pxr at the
    native level, rendering runs in an isolated subprocess using a separate
    virtual environment that has ovrtx installed without usd-core.

    The isolated venv is auto-provisioned on first use at
    ``~/.cache/wu/ovrtx_venv`` (override via ``ovrtx_venv_dir``).

    Requires: ovrtx >= 0.1.0
    """

    SUPPORTED_SENSOR_MODES: ClassVar[list[str]] = ["depth"]

    def __init__(
        self,
        log_level: str = "warn",
        ovrtx_venv_dir: str | None = None,
        num_sensor_updates: int = 500,
        render_mode: str = "pt",
    ):
        """Initialize the OvRTX rendering backend.

        Args:
            log_level: OvRTX log verbosity ("error", "warn", "info", "debug").
                Default: "warn"
            ovrtx_venv_dir: Override directory for the isolated ovrtx venv.
                Defaults to ``~/.cache/wu/ovrtx_venv``.
            num_sensor_updates: Number of progressive ``renderer.step(dt=0)``
                iterations per frame. This is the only quality knob
                OVRtx 0.2.0 actually honors — the bundled
                ``omni:rtx:pt:samplesPerPixel`` /
                ``omni:rtx:rt:accumulationLimit`` schema attributes are
                silently ignored. Default 500 is the convergence plateau
                on the kit golden scene (~39.7 dB PSNR vs Kit reference,
                see ``/tmp/ovrtx_cap.py``).
            render_mode: One of ``rt1``/``rt2``/``pt``. Translates to
                ``omni:rtx:rendermode`` on the RenderProduct. Default
                ``pt`` (Kit's ground-truth mode) is the only mode that
                reaches Kit-equivalent quality; rt2 caps at ~27 dB
                regardless of step count.
        """
        import os
        from pathlib import Path

        from world_understanding.functions.graphics import render_ovrtx

        # Eagerly provision the ovrtx venv so errors surface at init time
        venv_dir = Path(ovrtx_venv_dir) if ovrtx_venv_dir else None
        self._ovrtx_python = render_ovrtx._get_ovrtx_python(venv_dir=venv_dir)
        self.log_level = log_level
        self._ovrtx_venv_dir = ovrtx_venv_dir
        self._num_sensor_updates = num_sensor_updates
        self._render_mode = render_mode

        # Write daemon script next to the venv and create daemon handle.
        # The daemon starts lazily on first render(); GPU init cost is paid once.
        effective_venv_dir = venv_dir or render_ovrtx._OVRTX_VENV_DIR
        daemon_script_path = os.path.join(str(effective_venv_dir), "_ovrtx_daemon.py")
        with open(daemon_script_path, "w", encoding="utf-8") as f:
            f.write(render_ovrtx._DAEMON_SCRIPT)
        self._daemon = render_ovrtx._OvRTXDaemon(
            ovrtx_python=self._ovrtx_python,
            daemon_script_path=daemon_script_path,
            log_level=log_level,
        )

    def supports_sensors(self) -> bool:
        """OvRTX backend supports sensor rendering modes."""
        return True

    def get_supported_sensor_modes(self) -> list[str]:
        """Return list of sensor modes supported by OvRTX backend."""
        return self.SUPPORTED_SENSOR_MODES.copy()

    def render(
        self,
        stage: Usd.Stage,
        cameras: list[str] | None = None,
        image_width: int = 1024,
        image_height: int | None = None,
        cull_style: str = "back",
        frames: str = "0",
        renderer: str = "GL",
        sensors: list[str] | None = None,
        num_sensor_updates: int | None = None,
        render_mode: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Render multiple cameras from a USD stage using OvRTX.

        Note: cull_style and renderer parameters are ignored as OvRTX uses
        its own RTX rendering pipeline.

        Args:
            stage: USD stage to render.
            cameras: List of camera paths to render. If None, uses ["/Camera"].
            image_width: Output image width in pixels.
            image_height: Output image height in pixels. If None, defaults to
                image_width (square).
            cull_style: Ignored (OvRTX uses its own settings).
            frames: Frame(s) to render (e.g., "0", "0:10", "0,5,10").
            renderer: Ignored (OvRTX uses RTX).
            sensors: Additional sensors to render (e.g., ["depth"]).
            num_sensor_updates: Samples per pixel for this call. ``None`` uses
                the instance default (ctor argument).
            render_mode: ``rt1``/``rt2``/``pt`` for this call. ``None``
                uses the instance default (ctor argument).
            **kwargs: Additional parameters (ignored).

        Returns:
            Dict with rendering results matching the base class specification.
        """
        from world_understanding.functions.graphics import render_ovrtx

        if image_height is None:
            image_height = image_width

        effective_updates = (
            num_sensor_updates
            if num_sensor_updates is not None
            else self._num_sensor_updates
        )
        effective_mode = render_mode if render_mode is not None else self._render_mode

        return render_ovrtx.render_all_cameras(
            stage=stage,
            image_width=image_width,
            image_height=image_height,
            cameras=cameras,
            frames=frames,
            sensors=sensors,
            log_level=self.log_level,
            ovrtx_venv_dir=self._ovrtx_venv_dir,
            num_sensor_updates=effective_updates,
            render_mode=effective_mode,
            daemon=self._daemon,
        )

    def __del__(self) -> None:
        """Shut down the persistent OvRTX daemon on garbage collection."""
        if hasattr(self, "_daemon") and self._daemon is not None:
            self._daemon.shutdown()


class WarpRenderingBackend(RenderingBackend):
    """USD rendering backend using NVIDIA Warp GPU raytracer.

    This backend uses the warp-lang library's CUDA-based raytracer from
    the Newton physics project for in-process GPU rendering. Unlike OvRTX,
    it requires no Vulkan display server, no subprocess isolation, and no
    separate virtual environment — rendering runs directly in the current
    Python process on any CUDA-capable GPU.

    The raytracer uses diffuse-only shading with configurable color boosting
    to compensate for the lack of PBR materials.

    Requires: warp-lang, Newton warp_raytrace module on sys.path
    """

    SUPPORTED_SENSOR_MODES: ClassVar[list[str]] = ["depth", "normal"]

    def __init__(
        self,
        device: str = "cuda:0",
        color_boost: float = 3.0,
        enable_shadows: bool = True,
        enable_backface_culling: bool = True,
    ):
        """Initialize the Warp rendering backend.

        Args:
            device: CUDA device string (e.g., "cuda:0"). Default: "cuda:0".
            color_boost: Multiplier for displayColor to compensate for
                diffuse-only shading vs PBR. Default: 3.0.
            enable_shadows: Whether to cast shadow rays. Default: True.
            enable_backface_culling: Whether to enable backface culling.
                Default: True.
        """
        self._device = device
        self._color_boost = color_boost
        self._enable_shadows = enable_shadows
        self._enable_backface_culling = enable_backface_culling

    def supports_sensors(self) -> bool:
        """Warp backend supports sensor rendering modes."""
        return True

    def get_supported_sensor_modes(self) -> list[str]:
        """Return list of sensor modes supported by Warp backend."""
        return self.SUPPORTED_SENSOR_MODES.copy()

    def render(
        self,
        stage: Usd.Stage,
        cameras: list[str] | None = None,
        image_width: int = 1024,
        image_height: int | None = None,
        cull_style: str = "back",
        frames: str = "0",
        renderer: str = "GL",
        sensors: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Render multiple cameras from a USD stage using Warp.

        Note: renderer parameter is ignored as Warp uses its own
        CUDA-based raytracing pipeline.

        Args:
            stage: USD stage to render.
            cameras: List of camera paths to render. If None, uses ["/Camera"].
            image_width: Output image width in pixels.
            image_height: Output image height in pixels. If None, defaults to
                image_width (square).
            cull_style: Face culling style - "none", "back", or "front".
                Maps to enable_backface_culling option.
            frames: Frame(s) to render (e.g., "0", "0:10", "0,5,10").
            renderer: Ignored (Warp uses CUDA raytracing).
            sensors: Additional sensors to render (e.g., ["depth", "normal"]).
            **kwargs: Additional parameters (ignored).

        Returns:
            Dict with rendering results matching the base class specification.
        """
        from world_understanding.functions.graphics import render_warp

        if image_height is None:
            image_height = image_width

        # Map cull_style to backface culling flag
        backface_culling = self._enable_backface_culling
        if cull_style == "none":
            backface_culling = False
        elif cull_style == "front":
            logger.warning(
                "Warp does not support front-face culling; "
                "falling back to backface culling"
            )
            backface_culling = True
        elif cull_style == "back":
            backface_culling = True

        return render_warp.render_all_cameras(
            stage=stage,
            image_width=image_width,
            image_height=image_height,
            cameras=cameras,
            frames=frames,
            sensors=sensors,
            device=self._device,
            color_boost=self._color_boost,
            enable_shadows=self._enable_shadows,
            enable_backface_culling=backface_culling,
        )


def prepare_render_prims(
    stage: Usd.Stage,
    prim_paths: list[str],
    config: RenderingConfig | None = None,
    render_mode: str | None = None,
) -> tuple[Usd.Stage, list[str], int]:
    """Prepare the stage for rendering by modifying materials, lights, cameras, and visibility.

    This function performs all the setup needed for rendering but does not invoke the
    rendering backend. This allows for easier customization and parallelization.

    Side Effects:
        When config.strip_existing_animation is True (default), all time-sampled
        attributes on the stage are converted to static values sampled at time 0.
        This prevents conflicts with the time-coded camera/visibility keyframes
        used for per-prim rendering. Set strip_existing_animation=False in
        RenderingConfig to preserve existing animation.

    Args:
        stage: The stage to prepare for rendering.
        prim_paths: The paths to the primitives to render.
        config: The rendering configuration.
        render_mode: The rendering mode (e.g., "prim_only", "prim_with_stage").
            If not specified, uses legacy behavior with should_render_prim_only flag.

    Returns:
        A tuple containing:
        - The modified stage
        - List of camera paths created
        - Number of frames to render
    """
    if config is None:
        config = RenderingConfig()

    # Optionally remove existing animation to prevent conflicts with our time-coded
    # camera/visibility keyframes. This converts animated attributes to static
    # values sampled at time 0.
    if config.strip_existing_animation:
        from world_understanding.utils.usd.stage import remove_animation

        num_removed = remove_animation(stage, reference_time=Usd.TimeCode(0))
        if num_removed > 0:
            logger.debug(f"Removed animation from {num_removed} attributes")

    if config.should_reset_materials:
        nullify_materials(stage)

    if not config.use_lights:
        remove_all_lights(stage)

    # Calculate the bounding box — scope to root_prim_path when set
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    bbox_root = stage.GetPseudoRoot()
    if config.root_prim_path:
        rp = stage.GetPrimAtPath(config.root_prim_path)
        if rp and rp.IsValid():
            bbox_root = rp
    scene_bbox = bbox_cache.ComputeWorldBound(bbox_root)
    aligned_range = scene_bbox.ComputeAlignedRange()
    bbox_min = aligned_range.GetMin()
    bbox_max = aligned_range.GetMax()

    # Calculate the maximum dimension of the stage
    stage_size_x = bbox_max[0] - bbox_min[0]
    stage_size_y = bbox_max[1] - bbox_min[1]
    stage_size_z = bbox_max[2] - bbox_min[2]
    max_stage_size = max(stage_size_x, stage_size_y, stage_size_z)

    # Make multi-view cameras for the given prim for
    # the primitive identification.
    # Then render the scene from each camera.
    camera_paths = []
    frames = 0
    camera_root = "/Cameras"

    # Get camera specifications for the current render mode
    # If render_mode is not specified, infer from should_render_prim_only flag
    if render_mode is None:
        render_mode = (
            "prim_only" if config.should_render_prim_only else "prim_with_stage"
        )

    camera_specs = config.get_cameras_for_mode(render_mode)

    # Get focus mode for this render mode (per-mode or global)
    focus_mode = (
        config.get_focus_mode_for_mode(render_mode)
        if render_mode
        else config.camera_focus_mode
    )

    # If using stage focus mode, create cameras once for the entire stage
    if focus_mode == CameraFocusMode.STAGE:
        for camera_spec in camera_specs:
            # Camera specs from get_cameras_for_mode are guaranteed to have all values filled
            assert camera_spec.margin is not None
            assert camera_spec.focal_length is not None
            assert camera_spec.horizontal_aperture is not None
            assert camera_spec.vertical_aperture is not None
            assert camera_spec.near_clip_margin is not None
            assert camera_spec.far_clip_margin is not None

            dir_suffix = format_direction_for_filename(camera_spec.direction)
            camera_path = f"{camera_root}/{config.camera_name_prefix}_{dir_suffix}"

            # Choose camera function based on camera_view_type
            if camera_spec.view_type == CameraViewType.CORNER:
                add_corner_view_camera(
                    stage,
                    margin=camera_spec.margin,
                    camera_path=camera_path,
                    direction=camera_spec.direction,
                    focal_length=camera_spec.focal_length,
                    horizontal_aperture=camera_spec.horizontal_aperture,
                    vertical_aperture=camera_spec.vertical_aperture,
                    near_clip_margin=camera_spec.near_clip_margin,
                    far_clip_margin=camera_spec.far_clip_margin,
                    max_scene_size=max_stage_size,
                    time=Usd.TimeCode.Default(),
                )
            else:  # CameraViewType.SIDE
                add_side_view_camera(
                    stage,
                    margin=camera_spec.margin,
                    camera_path=camera_path,
                    direction=camera_spec.direction,
                    focal_length=camera_spec.focal_length,
                    horizontal_aperture=camera_spec.horizontal_aperture,
                    vertical_aperture=camera_spec.vertical_aperture,
                    near_clip_margin=camera_spec.near_clip_margin,
                    far_clip_margin=camera_spec.far_clip_margin,
                    max_scene_size=max_stage_size,
                    time=Usd.TimeCode.Default(),
                )
            camera_paths.append(camera_path)

    # Cache all mesh prims and prepare baseline state (O(N) setup for O(1) per-frame ops)
    mesh_colors: dict[str, tuple[float, float, float]] = {}

    for cached_prim_path in prim_paths:
        cached_prim = stage.GetPrimAtPath(cached_prim_path)
        cached_mesh = UsdGeom.Mesh(cached_prim)
        cached_path = cached_prim.GetPath().pathString

        # Generate and store random color
        if config.should_assign_random_colors:
            mesh_colors[cached_path] = (
                random.uniform(
                    config.other_color_range[0], config.other_color_range[1]
                ),
                random.uniform(
                    config.other_color_range[0], config.other_color_range[1]
                ),
                random.uniform(
                    config.other_color_range[0], config.other_color_range[1]
                ),
            )
            set_mesh_display_color(
                cached_mesh, mesh_colors[cached_path], time=Usd.TimeCode(0)
            )

    if config.should_render_prim_only:
        disable_visibility_for_all_mesh_prims(stage, time=Usd.TimeCode(0))

    for i, prim_path in enumerate(prim_paths):
        prim = stage.GetPrimAtPath(prim_path)

        if prim.IsInstance():
            prim.SetInstanceable(False)

        mesh = UsdGeom.Mesh(prim)

        # If using prim focus mode, create cameras for each prim
        if focus_mode == CameraFocusMode.PRIM:
            for camera_spec in camera_specs:
                # Camera specs from get_cameras_for_mode are guaranteed to have all values filled
                assert camera_spec.margin is not None
                assert camera_spec.focal_length is not None
                assert camera_spec.horizontal_aperture is not None
                assert camera_spec.vertical_aperture is not None
                assert camera_spec.near_clip_margin is not None
                assert camera_spec.far_clip_margin is not None

                dir_suffix = format_direction_for_filename(camera_spec.direction)
                camera_path = f"{camera_root}/{config.camera_name_prefix}_{dir_suffix}"

                # Choose camera function based on camera_view_type
                if camera_spec.view_type == CameraViewType.CORNER:
                    add_focused_corner_view_camera(
                        prim,
                        margin=camera_spec.margin,
                        camera_path=camera_path,
                        direction=camera_spec.direction,
                        focal_length=camera_spec.focal_length,
                        horizontal_aperture=camera_spec.horizontal_aperture,
                        vertical_aperture=camera_spec.vertical_aperture,
                        near_clip_margin=camera_spec.near_clip_margin,
                        far_clip_margin=camera_spec.far_clip_margin,
                        max_scene_size=max_stage_size,
                        time=Usd.TimeCode(i),
                    )
                else:  # CameraViewType.SIDE
                    add_focused_side_view_camera(
                        prim,
                        margin=camera_spec.margin,
                        camera_path=camera_path,
                        direction=camera_spec.direction,
                        focal_length=camera_spec.focal_length,
                        horizontal_aperture=camera_spec.horizontal_aperture,
                        vertical_aperture=camera_spec.vertical_aperture,
                        near_clip_margin=camera_spec.near_clip_margin,
                        far_clip_margin=camera_spec.far_clip_margin,
                        max_scene_size=max_stage_size,
                        time=Usd.TimeCode(i),
                    )
                if i == 0:
                    camera_paths.append(camera_path)

        # Highlight current prim at frame i (O(1) keyframe operation)
        if config.should_highlight_prim and not prim.IsInstanceProxy():
            set_mesh_display_color(mesh, config.highlight_color, time=Usd.TimeCode(i))

            # Reset highlight to random color at next and previous frame (to avoid interpolation artifacts)
            if config.should_assign_random_colors:
                if i > 0:
                    set_mesh_display_color(
                        mesh, mesh_colors[prim_path], time=Usd.TimeCode(i - 1)
                    )

                if i + 1 < len(prim_paths):
                    set_mesh_display_color(
                        mesh, mesh_colors[prim_path], time=Usd.TimeCode(i + 1)
                    )

        # Set visibility for current prim (O(1) keyframe operation)
        if config.should_render_prim_only and not prim.IsInstanceProxy():
            mesh.GetVisibilityAttr().Set(UsdGeom.Tokens.inherited, time=Usd.TimeCode(i))

            # Hide at next frame
            if i + 1 < len(prim_paths):
                mesh.GetVisibilityAttr().Set(
                    UsdGeom.Tokens.invisible, time=Usd.TimeCode(i + 1)
                )

        frames += 1

    return stage, camera_paths, frames


def render_prims(
    rendering_backend: RenderingBackend,
    stage: Usd.Stage,
    prim_paths: list[str],
    config: RenderingConfig | None = None,
) -> dict[str, Any]:
    """Render all primitives in the stage.

    Args:
        rendering_backend: The rendering backend.
        stage: The stage to render.
        prim_paths: The paths to the primitives to render.
        config: The rendering configuration.

    Returns:
        The rendering result.
    """
    if config is None:
        config = RenderingConfig()

    # Prepare the stage for rendering
    stage, camera_paths, num_frames = prepare_render_prims(stage, prim_paths, config)

    # Format frames string for rendering backend
    frames = "0:" + str(num_frames - 1) if num_frames > 1 else "0"

    # Invoke the rendering backend
    render_results = rendering_backend.render(
        stage,
        cameras=camera_paths,
        image_width=config.image_width,
        cull_style=config.cull_style,
        frames=frames,
        apply_background_mask=config.use_background_color,
    )

    # render_results["results"] is a list of dicts, each containing:
    # - camera: The camera name
    # - images: A list of PIL Image objects
    # - other metadata
    # We now want to turn the list of images into a dict of prim_path -> list of images
    for i, result in enumerate(render_results["results"]):
        prim_to_images = {}
        for j, image in enumerate(result["images"]):
            prim_path = str(prim_paths[j])

            # Apply background color if specified
            if config.use_background_color:
                image = paste_on_background(image, config.background_color)
                result["images"][j] = image

            prim_to_images[prim_path] = image
        render_results["results"][i]["prim_to_images"] = prim_to_images

    return render_results


def render_all_prims(
    rendering_backend: RenderingBackend,
    stage: Usd.Stage,
    config: RenderingConfig | None = None,
) -> dict[str, Any]:
    """Render all primitives in the stage.

    Args:
        rendering_backend: The rendering backend.
        stage: The stage to render.
        config: The rendering configuration.

    Returns:
        The rendering result.
    """
    if config is None:
        config = RenderingConfig()

    prim_paths = []
    for mesh in traverse_meshes(stage):
        prim = mesh.GetPrim()
        prim_paths.append(prim.GetPath().pathString)

    return render_prims(
        rendering_backend,
        stage,
        prim_paths,
        config,
    )


def hide_prims_outside_subtree(stage: Usd.Stage, root_prim_path: str) -> None:
    """Hide all prims outside the given subtree by setting visibility.

    Walks the hierarchy and makes invisible any prim that is not an
    ancestor of, or a descendant of, *root_prim_path*.
    """
    target = stage.GetPrimAtPath(root_prim_path)
    if not target or not target.IsValid():
        return

    ancestor_paths: set[str] = set()
    cur = target
    while cur and cur.IsValid() and not cur.IsPseudoRoot():
        ancestor_paths.add(str(cur.GetPath()))
        cur = cur.GetParent()

    def _recurse(prim: Usd.Prim) -> None:
        path = str(prim.GetPath())
        if path == root_prim_path or path.startswith(root_prim_path + "/"):
            return  # inside target subtree — keep visible
        if path in ancestor_paths:
            for child in prim.GetChildren():
                _recurse(child)
            return  # ancestor — recurse but don't hide
        imageable = UsdGeom.Imageable(prim)
        if imageable:
            imageable.MakeInvisible()

    for child in stage.GetPseudoRoot().GetChildren():
        _recurse(child)


def prepare_prims_with_composition(
    stage: Usd.Stage,
    prim_paths: list[str],
    config: RenderingConfig | None = None,
    render_mode: str = "composition",
) -> tuple[
    tuple[Usd.Stage, list[str], int],
    tuple[Usd.Stage, list[str], int],
]:
    """Prepare two stages for composition rendering.

    Creates two prepared stages:
    1. Highlight stage: Renders ONLY the focused prim (isolated,
       no occlusion) in red for extracting clear contours/bboxes
    2. Plain stage: Renders the FULL scene with original materials
       as the composition background, so the highlighted prim is
       shown in context of surrounding prims

    This function allows for customization and parallelization of the rendering process.

    IMPORTANT Overrides:
    - Highlight color: Always uses red (1.0, 0.0, 0.0) ignoring user config to ensure
      extract_red_outline() can reliably detect the highlighted regions
    - Highlight stage visibility: Renders only the focused prim
      (should_render_prim_only=True) for clean contour extraction
    - Plain stage visibility: Renders full scene (should_render_prim_only=False)
      so the composition shows the prim in context
    - Camera margin: Both stages use camera_composition_margin (default: 3.0) for
      consistent framing, overriding camera_prim_focus_margin and camera_prim_with_stage_margin

    Args:
        stage: The original stage to prepare.
        prim_paths: The paths to the primitives to render.
        config: The rendering configuration.

    Returns:
        A tuple containing two tuples:
        - First tuple: (highlight_stage, camera_paths, num_frames) for highlighted rendering
        - Second tuple: (plain_stage, camera_paths, num_frames) for plain rendering
    """
    if config is None:
        config = RenderingConfig()

    # Create two copies of the stage for independent preparation
    from world_understanding.utils.usd.stage import duplicate_stage

    logger.info("      Duplicating stage for highlight rendering...")
    highlight_stage = duplicate_stage(stage)
    logger.info("      Duplicating stage for plain rendering...")
    plain_stage = duplicate_stage(stage)

    # When root_prim_path is set, hide everything outside the subtree
    # on both stages so composition renders only show the target object.
    if config.root_prim_path:
        for s in [highlight_stage, plain_stage]:
            hide_prims_outside_subtree(s, config.root_prim_path)

    # Prepare stage with highlighted prims
    # IMPORTANT: Force red highlight color for contour extraction reliability
    # The user's highlight_color setting is ignored for composition mode
    # Render only the focused prim (should_render_prim_only=True) to avoid occlusion
    # but use wider camera margin (camera_composition_margin) for consistent framing
    logger.info(f"      Preparing highlight stage for {len(prim_paths)} prims...")
    config_with_highlight = replace(
        config,
        should_highlight_prim=True,
        highlight_color=(1.0, 0.0, 0.0),  # Always use red for contour detection
        should_render_prim_only=True,  # Isolate prim for clear contour extraction
        should_reset_materials=True,  # Always reset materials so red highlight is visible
        should_assign_random_colors=True,  # Ensure highlight color stands out
        camera_prim_focus_margin=config.camera_composition_margin,
        camera_prim_with_stage_margin=config.camera_composition_margin,
    )
    highlight_result = prepare_render_prims(
        highlight_stage, prim_paths, config_with_highlight, render_mode=render_mode
    )

    # Prepare stage with original materials for the background of the composition.
    # By default, isolate the same prim as highlight (should_render_prim_only=True)
    # so the orange contour aligns with the visible prim. This is critical for assets
    # with multiple co-located instances (e.g., SkelRoot robots sharing the same XForm
    # but different skeleton poses) where showing all prims would cause displacement.
    # When composition_show_full_scene=True, show all prims so the VLM can see the
    # component in context of the full assembly.
    render_prim_only = not config.composition_show_full_scene
    logger.info(f"      Preparing plain stage for {len(prim_paths)} prims...")
    config_plain = replace(
        config,
        should_render_prim_only=render_prim_only,
        should_highlight_prim=False,  # No highlight color — show original appearance
        camera_prim_focus_margin=config.camera_composition_margin,
        camera_prim_with_stage_margin=config.camera_composition_margin,
    )
    plain_result = prepare_render_prims(
        plain_stage, prim_paths, config_plain, render_mode=render_mode
    )

    logger.info("      ✓ Both stages prepared successfully")
    return highlight_result, plain_result


def render_from_prepared_prims(
    rendering_backend: RenderingBackend,
    prepared_stage: Usd.Stage,
    camera_paths: list[str],
    num_frames: int,
    prim_paths: list[str],
    config: RenderingConfig,
    frame_range: tuple[int, int] | None = None,
    sensors: list[str] | None = None,
    image_height: int | None = None,
    stage_url: str | None = None,
    render_mode: str | None = None,
) -> dict[str, Any]:
    """Render from a prepared stage.

    Args:
        rendering_backend: The rendering backend.
        prepared_stage: The prepared stage to render.
        camera_paths: List of camera paths in the prepared stage.
        num_frames: Total number of frames in the prepared stage.
        prim_paths: List of prim paths being rendered.
        config: The rendering configuration.
        frame_range: Optional (start, end) frame indices to render. If None, renders all frames.
        sensors: Optional list of sensor modes to render (e.g., ["linear_depth", "instance_id_segmentation"]).
        stage_url: Optional pre-uploaded URL for NVCF rendering. If provided with NVCF backend,
                  skips stage upload for better performance.
        render_mode: The rendering mode (e.g., "prim_only", "prim_with_stage", "composition").
                    Used to determine per-mode occlusion settings.

    Returns:
        The rendering result.
    """
    # Format frames string
    if frame_range is not None:
        start, end = frame_range
        frames = f"{start}:{end}" if end > start else str(start)
    else:
        frames = "0:" + str(num_frames - 1) if num_frames > 1 else "0"

    # Check if we can use pre-uploaded URL for NVCF backend
    if stage_url and isinstance(rendering_backend, NVCFRenderingBackend):
        # Use URL-based rendering to avoid re-uploading the stage
        render_results = render_nvcf.render_all_cameras_from_url(
            usd_url=stage_url,
            image_width=config.image_width,
            image_height=image_height if image_height else config.image_width,
            cameras=camera_paths,
            frames=frames,
            api_key=rendering_backend.api_key,
            base_url=rendering_backend.base_url,
            timeout=rendering_backend.timeout,
            sensors=sensors,
            apply_background_mask=config.use_background_color,
            max_workers=1,  # Disable per-camera parallelism (matches original behavior)
        )
    else:
        # Use standard backend.render() which uploads the stage
        render_results = rendering_backend.render(
            prepared_stage,
            cameras=camera_paths,
            image_width=config.image_width,
            image_height=image_height,
            cull_style=config.cull_style,
            frames=frames,
            apply_background_mask=config.use_background_color,
            sensors=sensors,
        )

    # Determine if we should skip occluded images for this mode
    skip_occluded = (
        config.should_skip_occluded_for_mode(render_mode)
        if render_mode
        else config.skip_occluded_images
    )

    # Import occlusion detection if needed
    if skip_occluded and config.should_highlight_prim:
        from world_understanding.utils.image_utils import is_prim_visible_in_image

    # Process results and add prim_to_images mapping
    for i, result in enumerate(render_results["results"]):
        prim_to_images = {}
        prim_occlusion = {}

        for j, image in enumerate(result["images"]):
            prim_path = str(prim_paths[j])

            # Apply background color if specified
            if config.use_background_color:
                image = paste_on_background(image, config.background_color)
                result["images"][j] = image

            # Check for occlusion if skip_occluded_images is enabled and highlighting is on
            is_occluded = False
            if (
                skip_occluded
                and config.should_highlight_prim
                and not config.should_render_prim_only
            ):
                # For prim_with_stage mode, check if highlight is visible in the scene
                is_visible = is_prim_visible_in_image(
                    image,
                    contour_method="red",  # Always use red for highlight detection
                    pixel_threshold=config.occlusion_pixel_threshold,
                )
                is_occluded = not is_visible
                prim_occlusion[prim_path] = is_occluded

                if is_occluded:
                    # Mark as None to skip saving
                    prim_to_images[prim_path] = None
                else:
                    prim_to_images[prim_path] = image
            else:
                prim_to_images[prim_path] = image

        render_results["results"][i]["prim_to_images"] = prim_to_images
        if prim_occlusion:
            render_results["results"][i]["prim_occlusion"] = prim_occlusion

    return render_results


def render_from_prepared_composition(
    rendering_backend: RenderingBackend,
    highlight_stage: Usd.Stage,
    highlight_cameras: list[str],
    plain_stage: Usd.Stage,
    plain_cameras: list[str],
    frames: int,
    prim_paths: list[str],
    config: RenderingConfig,
    frame_range: tuple[int, int] | None = None,
    sensors: list[str] | None = None,
    image_height: int | None = None,
    highlight_url: str | None = None,
    plain_url: str | None = None,
) -> dict[str, Any]:
    """Render from prepared composition stages.

    Args:
        rendering_backend: The rendering backend.
        highlight_stage: The prepared highlight stage.
        highlight_cameras: Camera paths for highlight stage.
        plain_stage: The prepared plain stage.
        plain_cameras: Camera paths for plain stage.
        frames: Number of frames in both stages.
        prim_paths: List of prim paths being rendered.
        config: The rendering configuration.
        frame_range: Optional (start, end) frame indices to render. If None, renders all frames.
        sensors: Optional list of sensor modes to render (e.g., ["linear_depth", "instance_id_segmentation"]).
        highlight_url: Optional pre-uploaded URL for NVCF rendering of highlight stage.
        plain_url: Optional pre-uploaded URL for NVCF rendering of plain stage.

    Returns:
        The rendering result with composed images.
    """
    # Format frames string
    if frame_range is not None:
        start, end = frame_range
        frames_str = f"{start}:{end}" if end > start else str(start)
    else:
        frames_str = "0:" + str(frames - 1) if frames > 1 else "0"

    # Render both stages - use pre-uploaded URLs if available for NVCF
    if (
        highlight_url
        and plain_url
        and isinstance(rendering_backend, NVCFRenderingBackend)
    ):
        # Use URL-based rendering for highlight stage
        prim_images_with_highlight = render_nvcf.render_all_cameras_from_url(
            usd_url=highlight_url,
            image_width=config.image_width,
            image_height=image_height if image_height else config.image_width,
            cameras=highlight_cameras,
            frames=frames_str,
            api_key=rendering_backend.api_key,
            base_url=rendering_backend.base_url,
            timeout=rendering_backend.timeout,
            sensors=sensors,
            apply_background_mask=config.use_background_color,
            max_workers=1,  # Disable per-camera parallelism (matches original behavior)
        )

        # Use URL-based rendering for plain stage
        prim_images_plain = render_nvcf.render_all_cameras_from_url(
            usd_url=plain_url,
            image_width=config.image_width,
            image_height=image_height if image_height else config.image_width,
            cameras=plain_cameras,
            frames=frames_str,
            api_key=rendering_backend.api_key,
            base_url=rendering_backend.base_url,
            timeout=rendering_backend.timeout,
            sensors=sensors,
            apply_background_mask=config.use_background_color,
            max_workers=1,  # Disable per-camera parallelism (matches original behavior)
        )
    else:
        # Use standard backend.render() which uploads the stages
        prim_images_with_highlight = rendering_backend.render(
            highlight_stage,
            cameras=highlight_cameras,
            image_width=config.image_width,
            image_height=image_height,
            cull_style=config.cull_style,
            frames=frames_str,
            apply_background_mask=config.use_background_color,
            sensors=sensors,
        )

        prim_images_plain = rendering_backend.render(
            plain_stage,
            cameras=plain_cameras,
            image_width=config.image_width,
            image_height=image_height,
            cull_style=config.cull_style,
            frames=frames_str,
            apply_background_mask=config.use_background_color,
            sensors=sensors,
        )

    # Process results for highlight rendering
    for i, result in enumerate(prim_images_with_highlight["results"]):
        prim_to_images = {}
        for j, image in enumerate(result["images"]):
            prim_path = str(prim_paths[j])
            if config.use_background_color:
                image = paste_on_background(image, config.background_color)
                result["images"][j] = image
            prim_to_images[prim_path] = image
        prim_images_with_highlight["results"][i]["prim_to_images"] = prim_to_images

    # Process results for plain rendering
    for i, result in enumerate(prim_images_plain["results"]):
        prim_to_images = {}
        for j, image in enumerate(result["images"]):
            prim_path = str(prim_paths[j])
            if config.use_background_color:
                image = paste_on_background(image, config.background_color)
                result["images"][j] = image
            prim_to_images[prim_path] = image
        prim_images_plain["results"][i]["prim_to_images"] = prim_to_images

    # Compose the images
    def compose_image(highlight_img, plain_img, config):
        final_image = plain_img.copy()
        if config.enable_contour:
            # Use the configured contour method
            if config.contour_method == "non_black":
                outline_img = extract_non_black_outline(
                    highlight_img,
                    black_threshold=config.contour_black_threshold,
                    thickness=3,
                )
            else:  # Default to red method
                outline_img = extract_red_outline(highlight_img, thickness=3)

            # Convert 0-1 range to 0-255 for PIL
            contour_color_255 = tuple(int(c * 255) for c in config.contour_color)
            final_image = paste_outline_to_image(
                final_image, outline_img, contour_color_255
            )
        if config.enable_bbox:
            bbox_img = draw_bounding_box_on_red(highlight_img, box_width=3)
            # Convert 0-1 range to 0-255 for PIL
            bbox_color_255 = tuple(int(c * 255) for c in config.bbox_color)
            final_image = paste_outline_to_image(final_image, bbox_img, bbox_color_255)
        return final_image

    # Create mapping of camera names
    highlight_cameras_dict = {
        r["camera"]: r for r in prim_images_with_highlight["results"]
    }
    plain_cameras_dict = {r["camera"]: r for r in prim_images_plain["results"]}

    # Compose images and detect occlusion
    # Import occlusion detection if needed
    if config.skip_occluded_images:
        from world_understanding.utils.image_utils import is_prim_visible_in_image

    for camera_name in highlight_cameras_dict.keys():
        results_with_highlight = highlight_cameras_dict[camera_name]
        results_plain = plain_cameras_dict[camera_name]

        # Add occlusion metadata field if not present
        if "prim_occlusion" not in results_with_highlight:
            results_with_highlight["prim_occlusion"] = {}

        for prim_path, image_with_highlight in results_with_highlight[
            "prim_to_images"
        ].items():
            image_plain = results_plain["prim_to_images"][prim_path]

            # Check for occlusion before composition if skip_occluded_images is enabled
            is_occluded = False
            if config.skip_occluded_images:
                is_visible = is_prim_visible_in_image(
                    image_with_highlight,
                    contour_method=config.contour_method,
                    pixel_threshold=config.occlusion_pixel_threshold,
                    black_threshold=config.contour_black_threshold,
                )
                is_occluded = not is_visible
                results_with_highlight["prim_occlusion"][prim_path] = is_occluded

                # Log occlusion detection for debugging
                import logging

                logger = logging.getLogger(__name__)
                if is_occluded:
                    logger.info(
                        f"Prim {prim_path} is occluded in camera {camera_name} - skipping image"
                    )
                else:
                    logger.info(f"Prim {prim_path} is visible in camera {camera_name}")

            # Only compose if not occluded (or if we're keeping occluded images)
            if not is_occluded or not config.skip_occluded_images:
                image_composition = compose_image(
                    image_with_highlight, image_plain, config
                )
                results_with_highlight["prim_to_images"][prim_path] = image_composition
            else:
                # Mark image as None to skip saving later
                results_with_highlight["prim_to_images"][prim_path] = None

    return prim_images_with_highlight


def render_prims_with_composition(
    rendering_backend: RenderingBackend,
    stage: Usd.Stage,
    prim_paths: list[str],
    config: RenderingConfig | None = None,
) -> dict[str, Any]:
    """Render all primitives in the stage with composition."""
    if config is None:
        config = RenderingConfig()

    # Prepare both stages for composition
    (
        (highlight_stage, highlight_camera_paths, highlight_frames),
        (
            plain_stage,
            plain_camera_paths,
            plain_frames,
        ),
    ) = prepare_prims_with_composition(stage, prim_paths, config)

    # Render the highlighted stage
    highlight_frames_str = (
        "0:" + str(highlight_frames - 1) if highlight_frames > 1 else "0"
    )
    prim_images_with_highlight = rendering_backend.render(
        highlight_stage,
        cameras=highlight_camera_paths,
        image_width=config.image_width,
        cull_style=config.cull_style,
        frames=highlight_frames_str,
        apply_background_mask=config.use_background_color,
    )

    # Render the plain stage
    plain_frames_str = "0:" + str(plain_frames - 1) if plain_frames > 1 else "0"
    prim_images_plain = rendering_backend.render(
        plain_stage,
        cameras=plain_camera_paths,
        image_width=config.image_width,
        cull_style=config.cull_style,
        frames=plain_frames_str,
        apply_background_mask=config.use_background_color,
    )

    # Process results for highlight rendering
    for i, result in enumerate(prim_images_with_highlight["results"]):
        prim_to_images = {}
        for j, image in enumerate(result["images"]):
            prim_path = str(prim_paths[j])
            if config.use_background_color:
                image = paste_on_background(image, config.background_color)
                result["images"][j] = image
            prim_to_images[prim_path] = image
        prim_images_with_highlight["results"][i]["prim_to_images"] = prim_to_images

    # Process results for plain rendering
    for i, result in enumerate(prim_images_plain["results"]):
        prim_to_images = {}
        for j, image in enumerate(result["images"]):
            prim_path = str(prim_paths[j])
            if config.use_background_color:
                image = paste_on_background(image, config.background_color)
                result["images"][j] = image
            prim_to_images[prim_path] = image
        prim_images_plain["results"][i]["prim_to_images"] = prim_to_images

    def compose_image(highlight_img, plain_img, config):
        # Start with the plain image
        final_image = plain_img.copy()

        # Add contour if enabled
        if config.enable_contour:
            # Use the configured contour method
            if config.contour_method == "non_black":
                outline_img = extract_non_black_outline(
                    highlight_img,
                    black_threshold=config.contour_black_threshold,
                    thickness=3,
                )
            else:  # Default to red method
                outline_img = extract_red_outline(highlight_img, thickness=3)

            # Convert 0-1 range to 0-255 for PIL
            contour_color_255 = tuple(int(c * 255) for c in config.contour_color)
            final_image = paste_outline_to_image(
                final_image, outline_img, contour_color_255
            )

        # Add bounding box if enabled
        if config.enable_bbox:
            bbox_img = draw_bounding_box_on_red(highlight_img, box_width=3)
            # Convert 0-1 range to 0-255 for PIL
            bbox_color_255 = tuple(int(c * 255) for c in config.bbox_color)
            final_image = paste_outline_to_image(final_image, bbox_img, bbox_color_255)

        return final_image

    # Iterate over the images and add the composition
    if len(prim_images_with_highlight["results"]) != len(prim_images_plain["results"]):
        raise ValueError(
            "The number of results from the highlighted and plain renderings do not match"
        )

    # Create a mapping of camera names to results for more robust comparison
    highlight_cameras = {
        result["camera"]: result for result in prim_images_with_highlight["results"]
    }
    plain_cameras = {
        result["camera"]: result for result in prim_images_plain["results"]
    }

    if set(highlight_cameras.keys()) != set(plain_cameras.keys()):
        raise ValueError(
            f"The camera names from the highlighted and plain renderings do not match. "
            f"Highlighted: {sorted(highlight_cameras.keys())}, "
            f"Plain: {sorted(plain_cameras.keys())}"
        )

    for camera_name in highlight_cameras.keys():
        results_with_highlight = highlight_cameras[camera_name]
        results_plain = plain_cameras[camera_name]

        for prim_path, image_with_highlight in results_with_highlight[
            "prim_to_images"
        ].items():
            if prim_path not in results_plain["prim_to_images"]:
                raise ValueError(
                    f"The prim path {prim_path} is not in the plain rendering"
                )

            image_plain = results_plain["prim_to_images"][prim_path]
            image_composition = compose_image(image_with_highlight, image_plain, config)
            results_with_highlight["prim_to_images"][prim_path] = image_composition

    return prim_images_with_highlight


def render_all_prims_with_composition(
    rendering_backend: RenderingBackend,
    stage: Usd.Stage,
    config: RenderingConfig | None = None,
) -> dict[str, Any]:
    """Render all primitives in the stage with composition."""
    if config is None:
        config = RenderingConfig()

    prim_paths = []
    for mesh in traverse_meshes(stage):
        prim = mesh.GetPrim()
        prim_paths.append(prim.GetPath().pathString)

    return render_prims_with_composition(
        rendering_backend,
        stage,
        prim_paths,
        config,
    )
