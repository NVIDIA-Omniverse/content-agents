# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Storage utilities for session management."""

import json
from pathlib import Path


def count_jsonl_lines(file_path: Path, retries: int = 3) -> int:
    """Count lines in a JSONL file with retry logic for robustness.

    Args:
        file_path: Path to JSONL file
        retries: Number of retry attempts for file access issues

    Returns:
        Number of lines in file
    """
    if not file_path.exists():
        return 0

    last_error = None
    for attempt in range(retries):
        try:
            # Open in binary mode to bypass text buffering and see actual file content
            # This helps when another process is writing to the file
            with open(file_path, "rb") as f:
                count = 0
                for line in f:
                    # Only count complete lines (ending with newline)
                    # This avoids counting partial writes
                    if line.strip() and line.endswith(b"\n"):
                        count += 1
                return count
        except OSError as e:
            # File might be locked or temporarily unavailable (common on Windows/network drives)
            last_error = e
            if attempt < retries - 1:
                import time

                time.sleep(0.1)  # Wait 100ms before retry
                continue
        except Exception as e:
            # Unexpected error, don't retry
            last_error = e
            break

    # All retries failed, log warning and return 0
    if last_error:
        import logging

        logger = logging.getLogger(__name__)
        logger.debug(
            f"Failed to read {file_path} after {retries} attempts: {last_error}"
        )
    return 0


def read_checkpoint(checkpoint_path: Path) -> dict | None:
    """Read pipeline checkpoint file.

    Args:
        checkpoint_path: Path to .pipeline_state.json

    Returns:
        Checkpoint data or None
    """
    if not checkpoint_path.exists():
        return None

    try:
        with open(checkpoint_path) as f:
            return json.load(f)
    except Exception:
        return None
