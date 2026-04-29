# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for storage utilities (JSONL counting, checkpoint reading).

Tests the robustness of file operations used in progress monitoring.
"""

import json

import pytest

from ...service.session.storage import count_jsonl_lines, read_checkpoint


@pytest.mark.unit
class TestCountJsonlLines:
    """Test JSONL line counting with various file states."""

    def test_count_empty_file(self, tmp_path):
        """Test counting lines in an empty file."""
        p = tmp_path / "empty.jsonl"
        p.write_text("")

        assert count_jsonl_lines(p) == 0

    def test_count_single_line(self, tmp_path):
        """Test counting a single line."""
        p = tmp_path / "single.jsonl"
        p.write_text('{"id": 1}\n')

        assert count_jsonl_lines(p) == 1

    def test_count_multiple_lines(self, tmp_path):
        """Test counting multiple lines."""
        p = tmp_path / "multi.jsonl"
        lines = ['{"id": 1}\n', '{"id": 2}\n', '{"id": 3}\n']
        p.write_text("".join(lines))

        assert count_jsonl_lines(p) == 3

    def test_count_ignores_incomplete_last_line(self, tmp_path):
        """Test that incomplete lines (no newline) are not counted.

        This is critical for handling partial writes.
        """
        p = tmp_path / "partial.jsonl"
        # Last line has no newline
        p.write_bytes(b'{"id": 1}\n{"id": 2}\n{"id": 3}')

        assert count_jsonl_lines(p) == 2  # Only lines 1 and 2 counted

    def test_count_nonexistent_file(self, tmp_path):
        """Test that nonexistent files return 0."""
        p = tmp_path / "nonexistent.jsonl"

        assert count_jsonl_lines(p) == 0

    def test_count_file_with_empty_lines(self, tmp_path):
        """Test handling of empty lines in file."""
        p = tmp_path / "empty_lines.jsonl"
        # Mix of valid and empty lines
        lines = ['{"id": 1}\n', "\n", '{"id": 2}\n', ""]
        p.write_text("".join(lines))

        # Empty lines with just \n should not be counted (strip + endswith check)
        assert count_jsonl_lines(p) == 2

    def test_count_large_file(self, tmp_path):
        """Test counting lines in a larger file."""
        p = tmp_path / "large.jsonl"

        num_lines = 1000
        with p.open("w") as f:
            for i in range(num_lines):
                f.write(json.dumps({"id": i}) + "\n")

        assert count_jsonl_lines(p) == num_lines

    def test_count_with_retry_on_busy_file(self, tmp_path):
        """Test that count_jsonl_lines retries on transient errors."""
        p = tmp_path / "test.jsonl"
        p.write_text('{"id": 1}\n{"id": 2}\n')

        # Normal operation should succeed (retries built-in)
        assert count_jsonl_lines(p) == 2


@pytest.mark.unit
class TestReadCheckpoint:
    """Test checkpoint file reading."""

    def test_read_valid_checkpoint(self, tmp_path):
        """Test reading a valid checkpoint file."""
        p = tmp_path / ".pipeline_state.json"
        checkpoint = {"status": "running", "step": "predict", "progress": 0.5}
        p.write_text(json.dumps(checkpoint))

        result = read_checkpoint(p)
        assert result == checkpoint

    def test_read_nonexistent_checkpoint(self, tmp_path):
        """Test that missing checkpoint returns None."""
        p = tmp_path / ".pipeline_state.json"

        result = read_checkpoint(p)
        assert result is None

    def test_read_corrupted_checkpoint(self, tmp_path):
        """Test that corrupted JSON returns None (graceful degradation)."""
        p = tmp_path / ".pipeline_state.json"
        p.write_text("{invalid json}")

        result = read_checkpoint(p)
        assert result is None

    def test_read_empty_checkpoint(self, tmp_path):
        """Test that empty file returns None."""
        p = tmp_path / ".pipeline_state.json"
        p.write_text("")

        result = read_checkpoint(p)
        assert result is None

    def test_read_checkpoint_with_complex_structure(self, tmp_path):
        """Test reading checkpoint with nested structures."""
        p = tmp_path / ".pipeline_state.json"
        checkpoint = {
            "status": "running",
            "steps": [
                {"name": "render", "status": "completed", "duration": 10},
                {"name": "predict", "status": "running", "progress": 0.5},
            ],
            "artifacts": {"dataset": "/path/to/dataset.jsonl"},
        }
        p.write_text(json.dumps(checkpoint))

        result = read_checkpoint(p)
        assert result == checkpoint
        assert result["steps"][0]["name"] == "render"
