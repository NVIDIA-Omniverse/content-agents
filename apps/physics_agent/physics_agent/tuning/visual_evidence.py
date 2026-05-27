# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reference-media preparation and visual judge evidence helpers."""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
_VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".webm", ".avi", ".mkv"}


@dataclass(frozen=True)
class JudgeVisualEvidence:
    """Image evidence consumed by the physics tune judge."""

    reference_image_caption_pairs: tuple[tuple[str, Path], ...] = ()
    generated_image_paths: tuple[Path, ...] = ()
    comparison_image_path: Path | None = None
    reference_error: str | None = None
    generated_error: str | None = None
    comparison_error: str | None = None

    @property
    def has_reference_media(self) -> bool:
        return bool(self.reference_image_caption_pairs) or bool(self.reference_error)

    def with_generated_images(
        self,
        generated_image_paths: list[Path] | tuple[Path, ...],
        *,
        generated_error: str | None = None,
    ) -> JudgeVisualEvidence:
        return JudgeVisualEvidence(
            reference_image_caption_pairs=self.reference_image_caption_pairs,
            generated_image_paths=tuple(Path(p) for p in generated_image_paths),
            comparison_image_path=self.comparison_image_path,
            reference_error=self.reference_error,
            generated_error=generated_error,
            comparison_error=self.comparison_error,
        )

    def with_comparison_image(
        self,
        comparison_image_path: Path | None,
        *,
        comparison_error: str | None = None,
    ) -> JudgeVisualEvidence:
        return JudgeVisualEvidence(
            reference_image_caption_pairs=self.reference_image_caption_pairs,
            generated_image_paths=self.generated_image_paths,
            comparison_image_path=(
                Path(comparison_image_path)
                if comparison_image_path is not None
                else None
            ),
            reference_error=self.reference_error,
            generated_error=self.generated_error,
            comparison_error=comparison_error,
        )

    def to_metadata(self) -> dict[str, Any]:
        """Return JSON-safe evidence metadata for audit artifacts."""
        return {
            "reference_images": [
                {"caption": caption, "path": str(path)}
                for caption, path in self.reference_image_caption_pairs
            ],
            "generated_images": [
                {"caption": generated_frame_caption(idx, path), "path": str(path)}
                for idx, path in enumerate(self.generated_image_paths, 1)
            ],
            "comparison_image": (
                str(self.comparison_image_path)
                if self.comparison_image_path is not None
                else None
            ),
            "reference_error": self.reference_error,
            "generated_error": self.generated_error,
            "comparison_error": self.comparison_error,
        }


def has_reference_media(
    *,
    reference_images: list[Path] | tuple[Path, ...] | None = None,
    reference_videos: list[Path] | tuple[Path, ...] | None = None,
) -> bool:
    """Return True when callers supplied any reference images or videos."""
    return bool(reference_images) or bool(reference_videos)


def prepare_reference_media(
    *,
    reference_images: list[Path] | tuple[Path, ...] | None = None,
    reference_videos: list[Path] | tuple[Path, ...] | None = None,
    reference_descriptions: list[str] | tuple[str, ...] | None = None,
    reference_video_descriptions: list[str] | tuple[str, ...] | None = None,
    output_dir: Path,
    frames_per_video: int = 8,
) -> JudgeVisualEvidence:
    """Copy reference images/videos and extract video frames for VLM judging.

    The helper owns ``reference_media/images``, ``reference_media/videos``,
    and ``reference_frames`` below ``output_dir``; each call replaces those
    managed subdirectories so stale media from a previous run cannot leak
    into the current judge request.
    """
    image_paths = _coerce_paths(reference_images)
    video_paths = _coerce_paths(reference_videos)
    image_descriptions = _coerce_descriptions(
        "reference_descriptions", reference_descriptions, len(image_paths)
    )
    video_descriptions = _coerce_descriptions(
        "reference_video_descriptions",
        reference_video_descriptions,
        len(video_paths),
    )
    if not image_paths and not video_paths:
        return JudgeVisualEvidence()

    ref_root = Path(output_dir) / "reference_media"
    image_out = ref_root / "images"
    video_out = ref_root / "videos"
    frame_root = Path(output_dir) / "reference_frames"
    for media_dir in (image_out, video_out, frame_root):
        if media_dir.exists():
            shutil.rmtree(media_dir)
    pairs: list[tuple[str, Path]] = []

    for idx, src in enumerate(image_paths, 1):
        _validate_file(src, _IMAGE_EXTENSIONS, "reference image")
        dest = image_out / f"reference_image_{idx:02d}{src.suffix.lower()}"
        _copy_file(src, dest)
        caption = _caption("Reference Image", idx, image_descriptions[idx - 1])
        pairs.append((caption, dest))

    if video_paths:
        from world_understanding.functions.cv.video_frames import extract_frames

        for idx, src in enumerate(video_paths, 1):
            _validate_file(src, _VIDEO_EXTENSIONS, "reference video")
            dest = video_out / f"reference_video_{idx:02d}{src.suffix.lower()}"
            _copy_file(src, dest)
            frames = extract_frames(
                dest,
                frame_root / f"video_{idx:02d}",
                n=frames_per_video,
            )
            description = video_descriptions[idx - 1]
            for frame_idx, frame_path in enumerate(frames, 1):
                label = f"Reference Video {idx} - Frame {frame_idx}"
                timestamp = _frame_timestamp_label(frame_path)
                if timestamp:
                    label += f" ({timestamp})"
                caption = _caption_label(label, description)
                pairs.append((caption, frame_path))

    return JudgeVisualEvidence(reference_image_caption_pairs=tuple(pairs))


def generated_frame_caption(index: int, frame_path: str | Path) -> str:
    """Caption a generated simulation frame for the VLM judge."""
    label = f"Generated Physics Output - Frame {index}"
    timestamp = _frame_timestamp_label(Path(frame_path))
    if timestamp:
        label += f" ({timestamp})"
    return f"{label}:"


def write_comparison_contact_sheet(
    evidence: JudgeVisualEvidence,
    output_path: Path,
    *,
    max_reference_images: int = 4,
    max_generated_images: int = 8,
) -> tuple[Path | None, str | None]:
    """Write a compact PNG contact sheet of reference and generated evidence."""
    reference_items = list(evidence.reference_image_caption_pairs)[
        :max_reference_images
    ]
    generated_items = [
        (generated_frame_caption(idx, path), path)
        for idx, path in enumerate(evidence.generated_image_paths, 1)
    ][:max_generated_images]
    if not reference_items or not generated_items:
        return None, None

    try:
        from PIL import Image, ImageDraw, ImageOps
    except ImportError as exc:  # pragma: no cover - depends on optional env
        return None, f"PIL unavailable: {exc}"

    tile_w = 240
    tile_h = 210
    thumb_w = 220
    thumb_h = 150
    pad = 12
    cols = min(4, max(1, len(reference_items), len(generated_items)))
    rows = _chunk_count(len(reference_items), cols) + _chunk_count(
        len(generated_items), cols
    )
    sheet = Image.new("RGB", (cols * tile_w, rows * tile_h), "white")
    draw = ImageDraw.Draw(sheet)

    errors: list[str] = []
    row_offset = 0
    row_offset = _paste_section(
        sheet=sheet,
        draw=draw,
        items=reference_items,
        section_label="Reference",
        row_offset=row_offset,
        cols=cols,
        tile_w=tile_w,
        tile_h=tile_h,
        thumb_w=thumb_w,
        thumb_h=thumb_h,
        pad=pad,
        image_module=Image,
        image_ops=ImageOps,
        errors=errors,
    )
    _ = _paste_section(
        sheet=sheet,
        draw=draw,
        items=generated_items,
        section_label="Generated",
        row_offset=row_offset,
        cols=cols,
        tile_w=tile_w,
        tile_h=tile_h,
        thumb_w=thumb_w,
        thumb_h=thumb_h,
        pad=pad,
        image_module=Image,
        image_ops=ImageOps,
        errors=errors,
    )

    if len(errors) == len(reference_items) + len(generated_items):
        return None, "; ".join(errors)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    return output_path, "; ".join(errors) if errors else None


def resolve_default_judge_vlm() -> Any:
    """Construct the default physics-agent VLM for visual judging."""
    from world_understanding.agentic.config import get_api_key_for_model_config
    from world_understanding.functions.models.vision_language_models import (
        create_vlm,
    )
    from world_understanding.utils.credentials import apply_vlm_nim_env_override

    from physics_agent.api.defaults import (
        DEFAULT_VLM_BACKEND,
        DEFAULT_VLM_LLMGATEWAY_CONFIG,
        DEFAULT_VLM_MAX_TOKENS,
        DEFAULT_VLM_MODEL,
        DEFAULT_VLM_REASONING_EFFORT,
        DEFAULT_VLM_TEMPERATURE,
    )

    vlm_config = apply_vlm_nim_env_override(
        {
            "backend": DEFAULT_VLM_BACKEND,
            "model": DEFAULT_VLM_MODEL,
            "temperature": DEFAULT_VLM_TEMPERATURE,
            "max_tokens": DEFAULT_VLM_MAX_TOKENS,
            "reasoning_effort": DEFAULT_VLM_REASONING_EFFORT,
            "llmgateway": DEFAULT_VLM_LLMGATEWAY_CONFIG,
        }
    )
    backend = str(vlm_config.get("backend") or vlm_config.get("provider") or "")
    kwargs: dict[str, Any] = {
        "backend": backend,
        "model": vlm_config["model"],
        "temperature": vlm_config["temperature"],
        "max_tokens": vlm_config["max_tokens"],
    }
    if vlm_config.get("base_url"):
        kwargs["base_url"] = vlm_config["base_url"]
    if backend_supports_reasoning_effort(backend):
        kwargs["reasoning_effort"] = vlm_config["reasoning_effort"]
    if vlm_config.get("llmgateway") and "llmgateway" in backend:
        kwargs["llmgateway"] = vlm_config["llmgateway"]
    api_key = get_api_key_for_model_config(backend, vlm_config, "VLM")
    if api_key:
        kwargs["api_key"] = api_key
    return create_vlm(**kwargs)


def _coerce_paths(value: list[Path] | tuple[Path, ...] | None) -> list[Path]:
    if not value:
        return []
    return [Path(p) for p in value]


def backend_supports_reasoning_effort(backend: str) -> bool:
    """Return whether a VLM backend accepts a reasoning_effort kwarg."""
    normalized = str(backend).lower()
    return normalized == "openai" or "llmgateway" in normalized


def _coerce_descriptions(
    field_name: str,
    value: list[str] | tuple[str, ...] | None,
    expected_len: int,
) -> list[str]:
    if value is None:
        return [""] * expected_len
    out = [str(v) for v in value]
    if len(out) != expected_len:
        raise ValueError(
            f"{field_name} must have {expected_len} item(s), got {len(out)}"
        )
    return out


def _validate_file(path: Path, extensions: set[str], label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    if not path.is_file():
        raise ValueError(f"{label} must be a file, got directory: {path}")
    if path.suffix.lower() not in extensions:
        raise ValueError(
            f"Unsupported {label} extension {path.suffix!r}; "
            f"expected one of {sorted(extensions)}"
        )


def _copy_file(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() == dest.resolve():
        return
    shutil.copy2(src, dest)


def _caption(prefix: str, index: int, description: str) -> str:
    return _caption_label(f"{prefix} {index}", description)


def _caption_label(label: str, description: str) -> str:
    clean = _clean_caption_text(description)
    if clean:
        return f"{label}: {clean}"
    return f"{label}:"


def _clean_caption_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def _frame_timestamp_label(path: Path) -> str | None:
    match = re.search(r"__t(\d+)", path.stem)
    if match is None:
        return None
    seconds = int(match.group(1)) / 1000.0
    return f"t={seconds:.3f}s"


def _chunk_count(n_items: int, cols: int) -> int:
    return (n_items + cols - 1) // cols


def _short_label(label: str, max_chars: int = 58) -> str:
    clean = _clean_caption_text(label)
    if len(clean) <= max_chars:
        return clean
    return clean[: max_chars - 3] + "..."


def _paste_section(
    *,
    sheet: Any,
    draw: Any,
    items: list[tuple[str, Path]],
    section_label: str,
    row_offset: int,
    cols: int,
    tile_w: int,
    tile_h: int,
    thumb_w: int,
    thumb_h: int,
    pad: int,
    image_module: Any,
    image_ops: Any,
    errors: list[str],
) -> int:
    for local_idx, (caption, path) in enumerate(items):
        row = row_offset + local_idx // cols
        col = local_idx % cols
        x = col * tile_w
        y = row * tile_h
        draw.rectangle((x, y, x + tile_w - 1, y + tile_h - 1), outline="#dddddd")
        draw.text(
            (x + pad, y + 8),
            f"{section_label}: {_short_label(caption)}",
            fill="black",
        )
        try:
            with image_module.open(path) as img:
                thumb = image_ops.contain(img.convert("RGB"), (thumb_w, thumb_h))
                bg = image_module.new("RGB", (thumb_w, thumb_h), "#f7f7f7")
                px = (thumb_w - thumb.width) // 2
                py = (thumb_h - thumb.height) // 2
                bg.paste(thumb, (px, py))
        except Exception as exc:  # noqa: BLE001 - comparison image is best-effort
            errors.append(f"{path}: {type(exc).__name__}: {exc}")
            continue
        sheet.paste(bg, (x + pad, y + 46))
    return row_offset + _chunk_count(len(items), cols)


__all__ = [
    "JudgeVisualEvidence",
    "generated_frame_caption",
    "has_reference_media",
    "prepare_reference_media",
    "resolve_default_judge_vlm",
    "write_comparison_contact_sheet",
]
