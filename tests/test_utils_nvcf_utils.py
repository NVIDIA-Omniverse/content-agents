# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

from world_understanding.utils.nvcf_utils import get_base_url


def test_get_base_url_resolves_render_function_id_when_endpoint_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RENDER_ENDPOINT", raising=False)
    monkeypatch.setenv(
        "NVCF_RENDER_FUNCTION_ID",
        "12345678-1234-1234-1234-123456789abc",
    )

    assert (
        get_base_url(None, "RENDER_ENDPOINT", "NVCF_RENDER_FUNCTION_ID")
        == "https://12345678-1234-1234-1234-123456789abc.invocation.api.nvcf.nvidia.com"
    )


def test_get_base_url_prefers_endpoint_over_function_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RENDER_ENDPOINT", "http://renderer.local:8001")
    monkeypatch.setenv(
        "NVCF_RENDER_FUNCTION_ID",
        "12345678-1234-1234-1234-123456789abc",
    )

    assert (
        get_base_url(None, "RENDER_ENDPOINT", "NVCF_RENDER_FUNCTION_ID")
        == "http://renderer.local:8001"
    )


def test_get_base_url_accepts_explicit_function_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RENDER_ENDPOINT", raising=False)
    monkeypatch.delenv("NVCF_RENDER_FUNCTION_ID", raising=False)

    assert (
        get_base_url(
            "12345678-1234-1234-1234-123456789abc",
            "RENDER_ENDPOINT",
            "NVCF_RENDER_FUNCTION_ID",
        )
        == "https://12345678-1234-1234-1234-123456789abc.invocation.api.nvcf.nvidia.com"
    )


def test_get_base_url_requires_endpoint_or_function_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RENDER_ENDPOINT", raising=False)
    monkeypatch.delenv("NVCF_RENDER_FUNCTION_ID", raising=False)

    with pytest.raises(ValueError, match="RENDER_ENDPOINT"):
        get_base_url(None, "RENDER_ENDPOINT", "NVCF_RENDER_FUNCTION_ID")
