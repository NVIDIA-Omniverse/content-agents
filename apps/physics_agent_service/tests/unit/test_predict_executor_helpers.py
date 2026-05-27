# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for predict_executor helpers.

The router and worker both depend on
``_dataset_jsonl_has_resolvable_images`` to decide whether a session-staged
``dataset.jsonl`` is actually runnable. The smoke tests stub out the full
executor, so without these direct tests the helper has zero coverage of:

* the v0.2 schema (``media.images[].path``) emitted by ``prepare_dataset.py``
* the legacy / test-stub schema (``images: {kind: file}`` / list / string)
* the missing-images branch that triggers Mode-B fallback in
  ``detect_predict_mode``
* the empty / unparseable JSONL branch that must not silently win Mode A
"""

from __future__ import annotations

import json
from pathlib import Path

from ...service.workers.predict_executor import (
    _dataset_jsonl_has_resolvable_images,
    _extract_image_paths,
    detect_predict_mode,
)


def _write_jsonl(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")


# ---------------------------------------------------------------------------
# _extract_image_paths
# ---------------------------------------------------------------------------


def test_extract_image_paths_v02_media_images() -> None:
    """v0.2 schema (`prepare_dataset.py` output): paths under media.images[]."""
    entry = {
        "id": "/p0",
        "media": {
            "images": [
                {"path": "model_a/render_001.png", "type": "render"},
                {"path": "model_a/render_002.png", "type": "render"},
            ]
        },
    }
    assert _extract_image_paths(entry) == [
        "model_a/render_001.png",
        "model_a/render_002.png",
    ]


def test_extract_image_paths_legacy_images_dict() -> None:
    """Legacy / test-stub schema: top-level images dict."""
    entry = {"id": "/p0", "images": {"prim_only": "img_0.png"}}
    assert _extract_image_paths(entry) == ["img_0.png"]


def test_extract_image_paths_legacy_images_list() -> None:
    entry = {"id": "/p0", "images": ["a.png", "b.png"]}
    assert _extract_image_paths(entry) == ["a.png", "b.png"]


def test_extract_image_paths_legacy_images_string() -> None:
    entry = {"id": "/p0", "images": "single.png"}
    assert _extract_image_paths(entry) == ["single.png"]


def test_extract_image_paths_no_images() -> None:
    assert _extract_image_paths({"id": "/p0"}) == []


def test_extract_image_paths_hybrid_schema() -> None:
    """Pathological hybrid: both media.images and top-level images."""
    entry = {
        "id": "/p0",
        "media": {"images": [{"path": "a.png"}]},
        "images": ["b.png"],
    }
    assert "a.png" in _extract_image_paths(entry)
    assert "b.png" in _extract_image_paths(entry)


# ---------------------------------------------------------------------------
# _dataset_jsonl_has_resolvable_images
# ---------------------------------------------------------------------------


def test_resolvable_v02_with_images_present(tmp_path: Path) -> None:
    img_dir = tmp_path / "model_a"
    img_dir.mkdir()
    (img_dir / "render_001.png").write_text("fake")

    jsonl = tmp_path / "dataset.jsonl"
    _write_jsonl(
        jsonl,
        [
            {
                "id": "/p0",
                "media": {
                    "images": [{"path": "model_a/render_001.png", "type": "render"}]
                },
            }
        ],
    )

    assert _dataset_jsonl_has_resolvable_images(jsonl) is True


def test_resolvable_v02_with_images_missing(tmp_path: Path) -> None:
    """Mirrors the externally-staged case: JSONL was copied but per-prim
    PNGs live next to the original dataset_path, not in this directory."""
    jsonl = tmp_path / "dataset.jsonl"
    _write_jsonl(
        jsonl,
        [
            {
                "id": "/p0",
                "media": {
                    "images": [{"path": "model_a/render_001.png", "type": "render"}]
                },
            }
        ],
    )
    assert _dataset_jsonl_has_resolvable_images(jsonl) is False


def test_resolvable_legacy_dict_present(tmp_path: Path) -> None:
    (tmp_path / "img_0.png").write_text("fake")
    jsonl = tmp_path / "dataset.jsonl"
    _write_jsonl(jsonl, [{"id": "/p0", "images": {"prim_only": "img_0.png"}}])
    assert _dataset_jsonl_has_resolvable_images(jsonl) is True


def test_resolvable_legacy_dict_missing(tmp_path: Path) -> None:
    jsonl = tmp_path / "dataset.jsonl"
    _write_jsonl(jsonl, [{"id": "/p0", "images": {"prim_only": "img_0.png"}}])
    assert _dataset_jsonl_has_resolvable_images(jsonl) is False


def test_resolvable_empty_jsonl_returns_false(tmp_path: Path) -> None:
    jsonl = tmp_path / "dataset.jsonl"
    jsonl.write_text("")
    assert _dataset_jsonl_has_resolvable_images(jsonl) is False


def test_resolvable_only_blank_lines_returns_false(tmp_path: Path) -> None:
    jsonl = tmp_path / "dataset.jsonl"
    jsonl.write_text("\n\n\n")
    assert _dataset_jsonl_has_resolvable_images(jsonl) is False


def test_resolvable_only_image_less_entries_returns_false(tmp_path: Path) -> None:
    """An entry with no image paths is skipped, not treated as resolvable.
    Otherwise a JSONL whose entries are all metadata-only would falsely
    trigger Mode A."""
    jsonl = tmp_path / "dataset.jsonl"
    _write_jsonl(jsonl, [{"id": "/p0", "type": "Mesh"}, {"id": "/p1"}])
    assert _dataset_jsonl_has_resolvable_images(jsonl) is False


def test_resolvable_skips_blank_lines_then_finds_runnable_entry(
    tmp_path: Path,
) -> None:
    """A JSONL whose first lines are blank / image-less must still consider
    later entries — the cap is on image-bearing *entries seen*, not on raw
    lines, so a sparsely-formatted file isn't penalised."""
    (tmp_path / "img.png").write_text("fake")
    jsonl = tmp_path / "dataset.jsonl"
    with jsonl.open("w") as f:
        for _ in range(5):
            f.write("\n")
        f.write(json.dumps({"id": "/p0", "type": "Mesh"}) + "\n")
        f.write("\n")
        f.write(json.dumps({"id": "/p1", "images": {"prim_only": "img.png"}}) + "\n")
    assert _dataset_jsonl_has_resolvable_images(jsonl) is True


def test_resolvable_handles_unparseable_jsonl_lines(tmp_path: Path) -> None:
    (tmp_path / "img.png").write_text("fake")
    jsonl = tmp_path / "dataset.jsonl"
    with jsonl.open("w") as f:
        f.write("not json at all\n")
        f.write("{broken json\n")
        f.write(json.dumps({"id": "/p0", "images": {"prim_only": "img.png"}}) + "\n")
    assert _dataset_jsonl_has_resolvable_images(jsonl) is True


def test_resolvable_partial_missing_image_in_entry_falls_through(
    tmp_path: Path,
) -> None:
    """An entry where ONE of N referenced images is missing must NOT count as
    resolvable for that entry. The all-resolvable bar prevents predict from
    silently dropping VLM views."""
    (tmp_path / "a.png").write_text("fake")
    # b.png intentionally absent
    jsonl = tmp_path / "dataset.jsonl"
    _write_jsonl(jsonl, [{"id": "/p0", "images": ["a.png", "b.png"]}])
    assert _dataset_jsonl_has_resolvable_images(jsonl) is False


def test_resolvable_absolute_image_paths(tmp_path: Path) -> None:
    img = tmp_path / "absolute.png"
    img.write_text("fake")
    jsonl = tmp_path / "dataset.jsonl"
    _write_jsonl(jsonl, [{"id": "/p0", "images": [str(img)]}])
    assert _dataset_jsonl_has_resolvable_images(jsonl) is True


def test_resolvable_returns_false_when_path_does_not_exist(tmp_path: Path) -> None:
    """Not a unit test of the entry-path traversal but of the exception
    handler: open() on a missing file must surface as False, not crash."""
    missing = tmp_path / "does_not_exist.jsonl"
    assert _dataset_jsonl_has_resolvable_images(missing) is False


# ---------------------------------------------------------------------------
# detect_predict_mode integration with _dataset_jsonl_has_resolvable_images
# ---------------------------------------------------------------------------


def test_detect_mode_falls_back_to_mode_b_when_images_missing(
    tmp_path: Path,
) -> None:
    """The Mode-A fast path keys off the session_dataset existing AND its
    images being resolvable — when the JSONL was staged but its images
    aren't co-located, the executor must rebuild via Mode B."""
    session_dir = tmp_path
    cache = session_dir / "cache" / "dataset"
    cache.mkdir(parents=True)
    _write_jsonl(
        cache / "dataset.jsonl",
        [{"id": "/p0", "images": ["lost.png"]}],
    )
    mode, resolved = detect_predict_mode(session_dir=session_dir, dataset_path=None)
    assert mode == "full_predict"
    assert resolved is None


def test_detect_mode_picks_a_when_images_present(tmp_path: Path) -> None:
    session_dir = tmp_path
    cache = session_dir / "cache" / "dataset"
    cache.mkdir(parents=True)
    (cache / "img.png").write_text("fake")
    _write_jsonl(
        cache / "dataset.jsonl",
        [{"id": "/p0", "images": ["img.png"]}],
    )
    mode, resolved = detect_predict_mode(session_dir=session_dir, dataset_path=None)
    assert mode == "dataset_only"
    assert resolved == cache / "dataset.jsonl"


def test_detect_mode_explicit_dataset_path_wins(tmp_path: Path) -> None:
    """An explicit, readable dataset_path takes precedence even when the
    session has its own staged JSONL."""
    explicit = tmp_path / "external" / "dataset.jsonl"
    explicit.parent.mkdir()
    _write_jsonl(explicit, [{"id": "/p0", "images": ["x.png"]}])
    session_dir = tmp_path / "session"
    cache = session_dir / "cache" / "dataset"
    cache.mkdir(parents=True)
    _write_jsonl(
        cache / "dataset.jsonl",
        [{"id": "/q0", "images": ["y.png"]}],
    )
    mode, resolved = detect_predict_mode(session_dir=session_dir, dataset_path=explicit)
    assert mode == "dataset_only"
    assert resolved == explicit


def test_detect_mode_no_dataset_at_all_picks_b(tmp_path: Path) -> None:
    mode, resolved = detect_predict_mode(session_dir=tmp_path, dataset_path=None)
    assert mode == "full_predict"
    assert resolved is None
