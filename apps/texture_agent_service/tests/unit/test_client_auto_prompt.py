# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

import pytest

from ...client.client import TextureAgentClient, build_arg_parser


class _Response:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, str]:
        return {"session_id": "session-client-strict"}


class _FakeHttp:
    def __init__(self) -> None:
        self.headers: dict[str, str] = {}
        self.posts: list[dict[str, Any]] = []

    def post(self, url: str, **kwargs: Any) -> _Response:
        self.posts.append({"url": url, **kwargs})
        return _Response()


@pytest.mark.parametrize(
    ("auto_prompt_enabled", "expected"),
    [
        (False, "false"),
        (True, "true"),
    ],
)
def test_client_start_pipeline_serializes_auto_prompting(
    auto_prompt_enabled: bool,
    expected: str,
) -> None:
    client = TextureAgentClient("http://texture.test")
    fake_http = _FakeHttp()
    client._http = fake_http

    session_id = client.start_pipeline(
        session_id="uploaded-session",
        material_textures={"Aluminum_Matte": {"prompt": "weathered aluminum"}},
        auto_prompt_enabled=auto_prompt_enabled,
    )

    assert session_id == "session-client-strict"
    assert fake_http.posts[0]["data"]["session_id"] == "uploaded-session"
    assert fake_http.posts[0]["data"]["auto_prompt_enabled"] == expected
    assert "Aluminum_Matte" in fake_http.posts[0]["data"]["material_textures_json"]


def test_client_start_pipeline_omits_auto_prompting_when_defaulting() -> None:
    client = TextureAgentClient("http://texture.test")
    fake_http = _FakeHttp()
    client._http = fake_http

    client.start_pipeline(session_id="uploaded-session")

    assert "auto_prompt_enabled" not in fake_http.posts[0]["data"]


def test_client_cli_flag_disables_auto_prompting() -> None:
    args = build_arg_parser().parse_args(
        ["--disable-auto-prompt", "--quiet", "scene.usd"]
    )

    assert args.disable_auto_prompt is True
