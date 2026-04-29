# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path

from ...service.workers.executor import (
    _extract_final_stats,
    _extract_step_stats,
    _prepare_config_and_context,
    _task_to_step_name,
)


class PrepareUVsTask:
    pass


class _UnknownTask:
    pass


def test_task_to_step_name_maps_known_and_unknown_classes() -> None:
    assert _task_to_step_name(PrepareUVsTask()) == "prepare_uvs"
    assert _task_to_step_name(_UnknownTask()) == "_UnknownTask"


def test_prepare_config_and_context_applies_defaults_and_creates_dirs(
    tmp_path: Path,
) -> None:
    session_dir = tmp_path / "session"
    config, context = _prepare_config_and_context(
        {"input": {"usd_path": "/tmp/input.usd"}},
        session_dir,
    )

    working_dir = session_dir / "cache"
    assert config["project"]["working_dir"] == str(working_dir)
    assert context["working_dir"] == str(working_dir)
    assert context["usd_path"] == "/tmp/input.usd"
    assert (working_dir / "prepared").is_dir()
    assert (working_dir / "renders").is_dir()
    assert context["render_preview_config"]["image_width"] == 512
    assert context["render_config"]["image_width"] == 1024


def test_extract_step_stats_and_final_stats_fall_back_to_files(tmp_path: Path) -> None:
    session_dir = tmp_path / "session"
    (session_dir / "cache" / "textures").mkdir(parents=True)
    (session_dir / "cache" / "output").mkdir(parents=True)
    (session_dir / "cache" / "renders").mkdir(parents=True)
    (session_dir / "cache" / "textures" / "one.png").write_text("x", encoding="utf-8")
    (session_dir / "cache" / "textures" / "two.png").write_text("x", encoding="utf-8")
    (session_dir / "cache" / "output" / "a.usd").write_text(
        "#usda 1.0\n", encoding="utf-8"
    )
    (session_dir / "cache" / "renders" / "final.png").write_text(
        "png", encoding="utf-8"
    )

    assert _extract_step_stats(
        "discover_materials", {"discovered_materials": [1, 2]}
    ) == {"materials_found": 2}
    assert _extract_step_stats(
        "generate_textures", {"generated_textures": {"a": 1}}
    ) == {"textures_generated": 1}

    stats = _extract_final_stats({}, session_dir)

    assert stats == {
        "materials_found": 0,
        "textures_generated": 2,
        "output_usd_count": 1,
        "renders_count": 1,
    }
