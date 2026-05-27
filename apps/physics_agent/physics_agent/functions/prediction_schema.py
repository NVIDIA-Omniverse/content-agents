# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Helpers for the Physics Agent prediction JSONL contract."""

from __future__ import annotations

from typing import Any

_PRESERVED_WRAPPER_METADATA_KEYS = ("original_response",)


def unwrap_output_key_payload(payload: Any, output_key: str) -> Any:
    """Return the value that should live under ``output_key``.

    Some VLM/parser responses already include the configured output wrapper,
    for example ``{"classification": {"physical_properties": ...}}``. The
    predictions JSONL schema stores that parsed payload under ``output_key``;
    without normalization it becomes ``classification.classification`` and
    downstream ``apply_physics`` misses the physical properties.
    """

    if not isinstance(payload, dict):
        return payload

    nested = payload.get(output_key)
    if isinstance(nested, dict) and "physical_properties" not in payload:
        normalized = dict(nested)
        for key in _PRESERVED_WRAPPER_METADATA_KEYS:
            if key in payload and key not in normalized:
                normalized[key] = payload[key]
        return normalized

    return payload
