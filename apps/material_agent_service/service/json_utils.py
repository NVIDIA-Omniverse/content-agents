# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""JSON normalization helpers for service metadata and events."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any


def to_json_safe(obj: Any) -> Any:
    """Convert common Python objects to JSON-serializable values."""
    if obj is None or isinstance(obj, bool | int | float | str):
        return obj
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, datetime | date):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {
            str(to_json_safe(key)): to_json_safe(value) for key, value in obj.items()
        }
    if isinstance(obj, list | tuple):
        return [to_json_safe(value) for value in obj]
    if isinstance(obj, set):
        return [to_json_safe(value) for value in sorted(obj, key=str)]
    if is_dataclass(obj) and not isinstance(obj, type):
        return to_json_safe(asdict(obj))
    if hasattr(obj, "model_dump"):
        return to_json_safe(obj.model_dump())
    return str(obj)
