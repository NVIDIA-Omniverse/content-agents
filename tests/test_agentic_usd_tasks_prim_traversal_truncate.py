# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for path truncation and mapping in prim_traversal."""

import json

from world_understanding.agentic.usd_tasks.prim_traversal import (
    _MAX_SEGMENT_LEN,
    MAX_PATH_COMPONENT_LEN,
    _record_path_mapping,
    _truncate_segment,
    prim_path_to_directory_structure,
)


class TestTruncateSegment:
    """Tests for _truncate_segment."""

    def test_short_name_unchanged(self):
        """Names within limit are returned as-is."""
        assert _truncate_segment("short_name") == "short_name"

    def test_exactly_at_limit(self):
        """Name exactly at limit is returned as-is."""
        name = "a" * 80
        assert _truncate_segment(name) == name

    def test_over_limit_truncated(self):
        """Name over limit is truncated with hash suffix."""
        name = "a" * 100
        result = _truncate_segment(name)
        assert len(result) == 80
        assert result[-9] == "_"  # underscore before 8-char hash
        assert result[:-9] == name[:71]

    def test_truncation_is_deterministic(self):
        """Same input always produces same output."""
        name = "x" * 100
        assert _truncate_segment(name) == _truncate_segment(name)

    def test_different_names_different_hashes(self):
        """Different long names produce different truncated results."""
        name1 = "a" * 100
        name2 = "b" * 100
        assert _truncate_segment(name1) != _truncate_segment(name2)


class TestRecordPathMapping:
    """Tests for _record_path_mapping."""

    def test_creates_mapping_file(self, tmp_path):
        """Creates path_mapping.json when it doesn't exist."""
        _record_path_mapping(tmp_path, "original_name", "trunc_name")
        mapping_file = tmp_path / "path_mapping.json"
        assert mapping_file.exists()
        data = json.loads(mapping_file.read_text())
        assert data["trunc_name"] == "original_name"

    def test_appends_to_existing(self, tmp_path):
        """Appends to existing path_mapping.json."""
        _record_path_mapping(tmp_path, "orig1", "trunc1")
        _record_path_mapping(tmp_path, "orig2", "trunc2")
        data = json.loads((tmp_path / "path_mapping.json").read_text())
        assert data["trunc1"] == "orig1"
        assert data["trunc2"] == "orig2"

    def test_no_duplicate_entries(self, tmp_path):
        """Writing same mapping twice doesn't duplicate."""
        _record_path_mapping(tmp_path, "orig", "trunc")
        _record_path_mapping(tmp_path, "orig", "trunc")
        data = json.loads((tmp_path / "path_mapping.json").read_text())
        assert len(data) == 1


class TestPrimPathToDirectoryStructureTruncation:
    """Tests for truncation in prim_path_to_directory_structure."""

    def test_long_segment_truncated(self, tmp_path):
        """Long prim path segments are truncated."""
        long_name = "A" * 100
        result = prim_path_to_directory_structure(
            f"/World/{long_name}/mesh", tmp_path, "render.png"
        )
        # Directory segments use shorten_for_filesystem (MAX_PATH_COMPONENT_LEN=96),
        # filenames use _truncate_segment (_MAX_SEGMENT_LEN=80)
        parts = result.relative_to(tmp_path).parts
        assert all(len(p) <= MAX_PATH_COMPONENT_LEN for p in parts[:-1])
        assert len(parts[-1]) <= _MAX_SEGMENT_LEN

    def test_long_filename_truncated(self, tmp_path):
        """Long filenames are truncated preserving extension."""
        long_stem = "B" * 100
        result = prim_path_to_directory_structure(
            "/World/Mesh", tmp_path, f"{long_stem}.png"
        )
        assert result.suffix == ".png"
        assert len(result.stem) <= 80
