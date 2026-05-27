# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Contract-neutral input inventory helpers for validation workflows."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

InputKind = Literal["usd", "image", "video", "render_bundle"]
InputPath = str | Path
InputPathSpec = InputPath | Sequence[InputPath]
FocusPrimPathSpec = str | Sequence[str] | None

USD_EXTENSIONS = frozenset({".usd", ".usda", ".usdc", ".usdz"})
IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp"})
VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".webm"})


class InputResolutionError(ValueError):
    """Raised when validation inputs cannot be resolved into an inventory."""


@dataclass(frozen=True)
class ResolvedInput:
    """One user-supplied input path after resolution and classification."""

    original: str
    path: Path
    kind: InputKind
    extension: str | None = None
    image_paths: tuple[Path, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""
        return {
            "original": self.original,
            "path": str(self.path),
            "kind": self.kind,
            "extension": self.extension,
            "image_paths": [str(path) for path in self.image_paths],
        }


@dataclass(frozen=True)
class InputInventory:
    """Grouped validation inputs for planner/template consumption."""

    items: tuple[ResolvedInput, ...]
    usd_paths: tuple[Path, ...]
    image_paths: tuple[Path, ...]
    video_paths: tuple[Path, ...]
    render_bundle_dirs: tuple[Path, ...]
    render_bundle_image_paths: tuple[Path, ...]
    focus_prim_paths: tuple[str, ...] = ()
    working_dir: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation for later model-layer copying."""
        return {
            "items": [item.to_dict() for item in self.items],
            "usd_paths": [str(path) for path in self.usd_paths],
            "image_paths": [str(path) for path in self.image_paths],
            "video_paths": [str(path) for path in self.video_paths],
            "render_bundle_dirs": [str(path) for path in self.render_bundle_dirs],
            "render_bundle_image_paths": [
                str(path) for path in self.render_bundle_image_paths
            ],
            "focus_prim_paths": list(self.focus_prim_paths),
            "working_dir": str(self.working_dir) if self.working_dir else None,
        }


class InputResolver:
    """Resolve user input paths into a contract-neutral inventory.

    Relative input and working-directory paths are resolved against ``base_dir``,
    or the current working directory when no ``base_dir`` is provided. Directory
    inputs are treated as render bundles and scan image-like files only.
    """

    def __init__(
        self,
        *,
        base_dir: str | Path | None = None,
        working_dir: str | Path | None = None,
        create_working_dir: bool = False,
    ) -> None:
        self.base_dir = _resolve_base_dir(base_dir)
        self.working_dir = _resolve_optional_path(working_dir, self.base_dir)
        self.create_working_dir = create_working_dir

    def resolve(
        self,
        inputs: InputPathSpec,
        *,
        focus_prim_paths: FocusPrimPathSpec = None,
    ) -> InputInventory:
        """Resolve and classify validation inputs.

        Args:
            inputs: File or directory inputs supplied by a config or CLI.
            focus_prim_paths: Manual focus prims to pass through unchanged,
                without validating USD prim syntax or existence.

        This resolver intentionally validates path existence and file extensions
        only. It does not open files or verify image/video/USD readability.

        Returns:
            Grouped input inventory.

        Raises:
            InputResolutionError: If paths are missing, unsupported, or malformed.
        """
        input_paths = _normalize_input_paths(inputs)
        if not input_paths:
            raise InputResolutionError("At least one input path is required")

        items = tuple(self._resolve_input(path) for path in input_paths)
        working_dir = self._prepare_working_dir()
        usd_paths = tuple(item.path for item in items if item.kind == "usd")
        image_paths = tuple(item.path for item in items if item.kind == "image")
        video_paths = tuple(item.path for item in items if item.kind == "video")
        render_bundle_dirs = tuple(
            item.path for item in items if item.kind == "render_bundle"
        )
        render_bundle_image_paths = tuple(
            image_path
            for item in items
            if item.kind == "render_bundle"
            for image_path in item.image_paths
        )

        return InputInventory(
            items=items,
            usd_paths=usd_paths,
            image_paths=image_paths,
            video_paths=video_paths,
            render_bundle_dirs=render_bundle_dirs,
            render_bundle_image_paths=render_bundle_image_paths,
            focus_prim_paths=_normalize_focus_prim_paths(focus_prim_paths),
            working_dir=working_dir,
        )

    def _prepare_working_dir(self) -> Path | None:
        if self.working_dir is None:
            return None
        if self.working_dir.exists() and not self.working_dir.is_dir():
            raise InputResolutionError(
                f"Working directory is not a directory: {self.working_dir}"
            )
        if self.create_working_dir:
            self.working_dir.mkdir(parents=True, exist_ok=True)
        elif not self.working_dir.exists():
            raise InputResolutionError(
                "Working directory does not exist: "
                f"{self.working_dir}. Set create_working_dir=True to create it."
            )
        return self.working_dir

    def _resolve_input(self, raw_path: str | Path) -> ResolvedInput:
        original = str(raw_path)
        path = _resolve_required_path(raw_path, self.base_dir)
        if path.is_dir():
            image_paths = _find_image_files(path)
            if not image_paths:
                raise InputResolutionError(
                    "Input directory does not contain image-like files: "
                    f"{original} (resolved to {path})"
                )
            return ResolvedInput(
                original=original,
                path=path,
                kind="render_bundle",
                image_paths=image_paths,
            )

        suffix = path.suffix.lower()
        if suffix in USD_EXTENSIONS:
            return ResolvedInput(original, path, "usd", suffix)
        if suffix in IMAGE_EXTENSIONS:
            return ResolvedInput(original, path, "image", suffix)
        if suffix in VIDEO_EXTENSIONS:
            return ResolvedInput(original, path, "video", suffix)

        raise InputResolutionError(
            f"Unsupported input type for {original} (resolved to {path}, "
            f"extension={suffix or '<none>'}). Supported files: "
            f"USD {sorted(USD_EXTENSIONS)}, images {sorted(IMAGE_EXTENSIONS)}, "
            f"videos {sorted(VIDEO_EXTENSIONS)}, or directories containing images."
        )


def resolve_input_inventory(
    inputs: InputPathSpec,
    *,
    base_dir: str | Path | None = None,
    focus_prim_paths: FocusPrimPathSpec = None,
    working_dir: str | Path | None = None,
    create_working_dir: bool = False,
) -> InputInventory:
    """Resolve inputs without explicitly constructing an ``InputResolver``."""
    return InputResolver(
        base_dir=base_dir,
        working_dir=working_dir,
        create_working_dir=create_working_dir,
    ).resolve(inputs, focus_prim_paths=focus_prim_paths)


def _normalize_input_paths(inputs: InputPathSpec) -> tuple[InputPath, ...]:
    if isinstance(inputs, str | Path):
        return (inputs,)
    return tuple(inputs)


def _normalize_focus_prim_paths(
    focus_prim_paths: FocusPrimPathSpec,
) -> tuple[str, ...]:
    if focus_prim_paths is None:
        return ()
    if isinstance(focus_prim_paths, str):
        return (focus_prim_paths,)
    return tuple(focus_prim_paths)


def _resolve_base_dir(base_dir: str | Path | None) -> Path:
    if base_dir is None:
        return Path.cwd().resolve()
    path = Path(base_dir).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    resolved = path.resolve(strict=False)
    if not resolved.exists():
        raise InputResolutionError(f"Base directory does not exist: {resolved}")
    if not resolved.is_dir():
        raise InputResolutionError(f"Base path is not a directory: {resolved}")
    return resolved


def _resolve_optional_path(path: str | Path | None, base_dir: Path) -> Path | None:
    if path is None:
        return None
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = base_dir / resolved
    return resolved.resolve(strict=False)


def _resolve_required_path(path: str | Path, base_dir: Path) -> Path:
    resolved = _resolve_optional_path(path, base_dir)
    if resolved is None:
        raise InputResolutionError("Input path cannot be None")
    if not resolved.exists():
        raise InputResolutionError(
            f"Input path does not exist: {path} (resolved to {resolved})"
        )
    return resolved


def _find_image_files(directory: Path) -> tuple[Path, ...]:
    image_paths: list[Path] = []
    for root, dirnames, filenames in os.walk(directory, followlinks=False):
        root_path = Path(root)
        dirnames[:] = sorted(
            name for name in dirnames if not (root_path / name).is_symlink()
        )
        for filename in sorted(filenames):
            path = root_path / filename
            if (
                not path.is_symlink()
                and path.is_file()
                and path.suffix.lower() in IMAGE_EXTENSIONS
            ):
                image_paths.append(path.resolve(strict=False))
    return tuple(sorted(image_paths, key=lambda path: str(path)))
