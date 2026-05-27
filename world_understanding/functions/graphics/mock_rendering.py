# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mock rendering backend for simulate mode.

Generates solid-color PIL images instantly, with no network calls or GPU.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any

from PIL import Image as PILImage

from world_understanding.functions.graphics.rendering import RenderingBackend


class MockRenderingBackend(RenderingBackend):
    """Rendering backend that produces deterministic solid-color images.

    Each camera gets a slightly different colored dot at the centre for
    visual distinction.  Useful for ``--simulate`` pipeline runs.
    """

    def supports_sensors(self) -> bool:
        return False

    def render(
        self,
        stage: Any,
        cameras: list[str] | None = None,
        image_width: int = 512,
        image_height: int | None = None,
        cull_style: str = "back",
        frames: str = "0",
        renderer: str = "GL",
        sensors: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if image_height is None:
            image_height = image_width

        if cameras is None:
            cameras = ["default"]

        # Parse frames spec to determine how many images to produce.
        # The real renderer produces one image per frame; we must match that
        # so the pipeline maps each frame to the correct prim.
        num_frames = self._parse_frame_count(frames)

        results: list[dict[str, Any]] = []
        t0 = time.monotonic()

        for cam in cameras:
            images = []
            for frame_idx in range(num_frames):
                img = PILImage.new(
                    "RGBA", (image_width, image_height), color=(180, 180, 180, 255)
                )
                # Draw a small coloured dot for visual distinction
                dot_color = self._camera_color(f"{cam}_{frame_idx}")
                cx, cy = image_width // 2, image_height // 2
                for dx in range(-4, 5):
                    for dy in range(-4, 5):
                        if dx * dx + dy * dy <= 16:
                            img.putpixel((cx + dx, cy + dy), dot_color)
                images.append(img)

            results.append(
                {
                    "camera": cam,
                    "images": images,
                    "render_time": 0.001,
                    "frame_count": num_frames,
                    "return_code": 0,
                    "status": "success",
                }
            )

        elapsed = time.monotonic() - t0
        return {
            "total_cameras": len(cameras),
            "successful_cameras": len(cameras),
            "failed_cameras": 0,
            "total_render_time": elapsed,
            "results": results,
        }

    @staticmethod
    def _parse_frame_count(frames: str) -> int:
        """Parse a frames spec into a count.

        Supported formats (matching real NVCF renderer):
          - ``"0"``      → 1 frame
          - ``"0:7"``    → 8 frames (colon-separated range, used by pipeline)
          - ``"0-7"``    → 8 frames (dash-separated range)
          - ``"0,1,2"``  → 3 frames (comma-separated list)
        """
        frames = frames.strip()
        # Colon range (primary format used by render_from_prepared_prims)
        if ":" in frames:
            parts = frames.split(":", 1)
            try:
                return int(parts[1]) - int(parts[0]) + 1
            except (ValueError, IndexError):
                return 1
        # Dash range
        if "-" in frames:
            parts = frames.split("-", 1)
            try:
                return int(parts[1]) - int(parts[0]) + 1
            except (ValueError, IndexError):
                return 1
        # Comma-separated list (e.g. "0,1,2")
        if "," in frames:
            return len(frames.split(","))
        return 1

    @staticmethod
    def _camera_color(cam_name: str) -> tuple[int, int, int]:
        h = int(hashlib.blake2s(cam_name.encode(), digest_size=16).hexdigest(), 16)
        return (h % 200 + 55, (h >> 8) % 200 + 55, (h >> 16) % 200 + 55)
