# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import logging
from pathlib import Path
from typing import Any

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


# USD Load Stage Configuration
class LoadUsdStageConfig(FunctionBaseConfig, name="load_usd_stage"):  # type: ignore[call-arg]
    pass


@register_function(config_type=LoadUsdStageConfig)  # type: ignore[misc]
async def load_usd_stage(config: LoadUsdStageConfig, builder: Builder) -> Any:
    from world_understanding.utils.usd import stage

    async def _load_usd_stage(file_path: str) -> dict[str, Any]:
        """Load a USD stage from a file.

        Args:
            file_path: Path to the USD file to load

        Returns:
            Dictionary containing loaded stage info and metadata
        """
        try:
            # Load the stage using the function
            usd_stage = await asyncio.to_thread(stage.load_stage, file_path)

            # Get metadata about the stage
            stage_info = await asyncio.to_thread(stage.get_stage_info, usd_stage)

            return {
                "success": True,
                "file_path": file_path,
                "prim_count": stage_info["prim_count"],
                "has_default_prim": stage_info["default_prim"] is not None,
                "up_axis": stage_info["up_axis"],
                "meters_per_unit": stage_info["meters_per_unit"],
                "time_codes_per_second": stage_info["time_codes_per_second"],
                "start_time_code": stage_info["start_time_code"],
                "end_time_code": stage_info["end_time_code"],
                "error": None,
            }
        except FileNotFoundError as e:
            return {
                "success": False,
                "file_path": file_path,
                "prim_count": 0,
                "has_default_prim": False,
                "up_axis": None,
                "meters_per_unit": None,
                "time_codes_per_second": None,
                "start_time_code": None,
                "end_time_code": None,
                "error": f"File not found: {e}",
            }
        except Exception as e:
            return {
                "success": False,
                "file_path": file_path,
                "prim_count": 0,
                "has_default_prim": False,
                "up_axis": None,
                "meters_per_unit": None,
                "time_codes_per_second": None,
                "start_time_code": None,
                "end_time_code": None,
                "error": f"Failed to load stage: {e}",
            }

    # Create a Generic NAT tool that can be used with any supported
    # LLM framework
    yield FunctionInfo.from_fn(
        _load_usd_stage,
        description=(
            "Load a USD stage from a file. Supports .usd, .usda, .usdc, "
            "and .usdz formats."
        ),
    )


# USD Save Stage Configuration
class SaveUsdStageConfig(FunctionBaseConfig, name="save_usd_stage"):  # type: ignore[call-arg]
    pass


@register_function(config_type=SaveUsdStageConfig)  # type: ignore[misc]
async def save_usd_stage(config: SaveUsdStageConfig, builder: Builder) -> Any:
    from world_understanding.utils.usd import stage

    async def _save_usd_stage(
        source_file: str,
        output_file: str,
        create_directories: bool = True,
    ) -> dict[str, Any]:
        """Save a USD stage to a file.

        Args:
            source_file: Path to the source USD file
            output_file: Path where to save the stage
            create_directories: Whether to create parent directories

        Returns:
            Dictionary containing save status and file info
        """
        try:
            # Create output directories if requested
            if create_directories:
                Path(output_file).parent.mkdir(parents=True, exist_ok=True)

            # Validate source file exists
            if not Path(source_file).exists():
                raise FileNotFoundError(f"Source file not found: {source_file}")

            # Load the stage from source file
            usd_stage = await asyncio.to_thread(stage.load_stage, source_file)

            # Save to output file
            saved_path = await asyncio.to_thread(
                stage.save_stage, usd_stage, output_file
            )

            # Get file size
            file_size = Path(saved_path).stat().st_size

            return {
                "success": True,
                "file_path": saved_path,
                "file_size": file_size,
                "error": None,
            }
        except Exception as e:
            return {
                "success": False,
                "file_path": output_file,
                "file_size": 0,
                "error": str(e),
            }

    # Create a Generic NAT tool that can be used with any supported
    # LLM framework
    yield FunctionInfo.from_fn(
        _save_usd_stage,
        description=(
            "Save a USD stage to a file. The file extension determines "
            "the format: .usda for ASCII, .usdc for binary."
        ),
    )


# USD Render Single Camera Configuration
class RenderSingleCameraConfig(FunctionBaseConfig, name="render_single_camera"):  # type: ignore[call-arg]
    pass


@register_function(config_type=RenderSingleCameraConfig)  # type: ignore[misc]
async def render_single_camera(
    config: RenderSingleCameraConfig, builder: Builder
) -> Any:
    async def _render_single_camera(
        usd_path: str,
        camera: str,
        output_dir: str = "./renders",
        image_width: int = 1920,
        frames: str = "0",
    ) -> dict[str, Any]:
        """Render a single camera view from a USD file.

        Args:
            usd_path: Path to the USD file
            camera: Camera name to render from
            output_dir: Directory to save rendered images
            image_width: Width of rendered images in pixels
            frames: Frame(s) to render (e.g., "0", "0:10", "0,5,10")

        Returns:
            Dictionary containing render results
        """
        try:
            from pxr import Usd

            from world_understanding.functions.graphics.rendering import (
                RemoteRenderingBackend,
            )

            stage = await asyncio.to_thread(Usd.Stage.Open, usd_path)
            backend = RemoteRenderingBackend()
            result = await asyncio.to_thread(
                backend.render,
                stage,
                cameras=[camera],
                image_width=image_width,
                frames=frames,
            )

            # Save rendered images
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            from world_understanding.utils.usd import stage as stage_utils

            saved_images = []
            camera_name = stage_utils.sanitize_name_for_filesystem(camera)
            cam_result = (result.get("results") or [{}])[0]
            images = cam_result.get("images", [])
            usd_stem = Path(usd_path).stem
            for idx, img in enumerate(images):
                if len(images) == 1:
                    filename = f"{usd_stem}_{camera_name}.png"
                else:
                    filename = f"{usd_stem}_{camera_name}_frame{idx}.png"

                img_path = output_path / filename
                img.save(img_path)
                saved_images.append(str(img_path))

            return {
                "success": True,
                "camera": cam_result.get("camera", camera),
                "images": saved_images,
                "render_time": result.get("total_render_time", 0),
                "frame_count": cam_result.get("frame_count", len(images)),
                "error": cam_result.get("error"),
            }
        except Exception as e:
            return {
                "success": False,
                "camera": camera,
                "images": [],
                "render_time": 0,
                "frame_count": 0,
                "error": str(e),
            }

    # Create a Generic NAT tool that can be used with any supported
    # LLM framework
    yield FunctionInfo.from_fn(
        _render_single_camera,
        description=(
            "Render a single camera view from a USD file. "
            "Saves rendered images to the specified output directory."
        ),
    )


# USD Render All Cameras Configuration
class RenderAllCamerasConfig(FunctionBaseConfig, name="render_all_cameras"):  # type: ignore[call-arg]
    pass


@register_function(config_type=RenderAllCamerasConfig)  # type: ignore[misc]
async def render_all_cameras(config: RenderAllCamerasConfig, builder: Builder) -> Any:
    async def _render_all_cameras(
        usd_path: str,
        output_dir: str = "./renders",
        image_width: int = 1920,
        frames: str = "0",
    ) -> dict[str, Any]:
        """Render all camera views from a USD file.

        Args:
            usd_path: Path to the USD file
            output_dir: Directory to save rendered images
            image_width: Width of rendered images in pixels
            frames: Frame(s) to render (e.g., "0", "0:10", "0,5,10")

        Returns:
            Dictionary containing render results for all cameras
        """
        try:
            from pxr import Usd

            from world_understanding.functions.graphics.rendering import (
                RemoteRenderingBackend,
            )

            stage = await asyncio.to_thread(Usd.Stage.Open, usd_path)
            backend = RemoteRenderingBackend()
            result = await asyncio.to_thread(
                backend.render,
                stage,
                cameras=None,
                image_width=image_width,
                frames=frames,
            )

            # Save rendered images
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            from world_understanding.utils.usd import stage as stage_utils

            cameras_data = []
            usd_stem = Path(usd_path).stem
            for cam_result in result.get("results", []):
                if not cam_result.get("error"):
                    saved_images = []
                    camera_name = stage_utils.sanitize_name_for_filesystem(
                        cam_result["camera"]
                    )
                    cam_images = cam_result["images"]
                    for idx, img in enumerate(cam_images):
                        if len(cam_images) == 1:
                            filename = f"{usd_stem}_{camera_name}.png"
                        else:
                            filename = f"{usd_stem}_{camera_name}_frame{idx}.png"

                        img_path = output_path / filename
                        img.save(img_path)
                        saved_images.append(str(img_path))

                    cameras_data.append(
                        {
                            "camera": cam_result["camera"],
                            "images": saved_images,
                            "frame_count": cam_result["frame_count"],
                        }
                    )

            return {
                "success": True,
                "cameras": cameras_data,
                "total_cameras": result["total_cameras"],
                "successful_cameras": result["successful_cameras"],
                "failed_cameras": result["failed_cameras"],
                "total_render_time": result["total_render_time"],
                "error": None,
            }
        except Exception as e:
            return {
                "success": False,
                "cameras": [],
                "total_cameras": 0,
                "successful_cameras": 0,
                "failed_cameras": 0,
                "total_render_time": 0,
                "error": str(e),
            }

    # Create a Generic NAT tool that can be used with any supported
    # LLM framework
    yield FunctionInfo.from_fn(
        _render_all_cameras,
        description=(
            "Render all camera views from a USD file. "
            "Automatically detects all cameras in the scene and renders "
            "each one."
        ),
    )
