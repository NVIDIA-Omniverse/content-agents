# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from texture_agent.config.schema import DEFAULTS, STEP_OUTPUT_DIRS
from texture_agent.config.unified_config import config_to_context, load_config


def test_load_config_applies_defaults_and_resolves_paths(tmp_path: Path) -> None:
    usd_path = tmp_path / "input.usda"
    usd_path.write_text("#usda 1.0\n", encoding="utf-8")
    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "project": {"working_dir": "runs/demo"},
                "input": {"usd_path": "input.usda"},
                "steps": {"render": {"image_width": 256}},
            }
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config["project"]["name"] == "pipeline"
    assert config["project"]["session_id"] == "pipeline"
    assert config["project"]["working_dir"] == str(
        (tmp_path / "runs" / "demo").resolve()
    )
    assert config["input"]["usd_path"] == str(usd_path.resolve())
    assert config["texture"] == DEFAULTS["texture"]
    assert config["variations"] == DEFAULTS["variations"]
    for step_name, dir_name in STEP_OUTPUT_DIRS.items():
        assert (Path(config["project"]["working_dir"]) / dir_name).is_dir(), step_name
    assert config["steps"]["render"]["image_width"] == 256
    assert config["steps"]["render"]["image_height"] == 1024


def test_load_config_uses_hidden_session_dir_when_working_dir_missing(
    tmp_path: Path,
) -> None:
    usd_path = tmp_path / "input.usda"
    usd_path.write_text("#usda 1.0\n", encoding="utf-8")
    config_path = tmp_path / "demo.yaml"
    config_path.write_text(
        yaml.safe_dump({"input": {"usd_path": str(usd_path)}}),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config["project"]["working_dir"] == str(tmp_path / ".demo")


def test_load_config_rejects_non_mapping_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "bad.yaml"
    config_path.write_text("- not-a-mapping\n", encoding="utf-8")

    with pytest.raises(ValueError, match="YAML mapping"):
        load_config(config_path)


def test_load_config_requires_input_usd_path(tmp_path: Path) -> None:
    config_path = tmp_path / "bad.yaml"
    config_path.write_text(yaml.safe_dump({"project": {"name": "x"}}), encoding="utf-8")

    with pytest.raises(ValueError, match="input.usd_path"):
        load_config(config_path)


def test_config_to_context_maps_expected_sections() -> None:
    config = {
        "input": {"usd_path": "/tmp/scene.usd", "prim_paths": ["/Root/Mesh"]},
        "project": {"working_dir": "/tmp/work"},
        "texture": {"backend": "simple_image_gen"},
        "material_textures": {"Steel": {"prompt": "brushed steel"}},
        "steps": {
            "blend_textures": {"default_opacity": 0.7},
            "render_previews": {"image_width": 512},
            "render": {"image_width": 1024},
        },
        "auto_prompt": {"user_prompt": "aged"},
        "variations": {"count": 2},
    }

    context = config_to_context(config)

    assert context["usd_path"] == "/tmp/scene.usd"
    assert context["prim_paths"] == ["/Root/Mesh"]
    assert context["working_dir"] == "/tmp/work"
    assert context["texture_config"] == {"backend": "simple_image_gen"}
    assert context["material_textures"] == {"Steel": {"prompt": "brushed steel"}}
    assert context["blend_config"] == {"default_opacity": 0.7}
    assert context["render_preview_config"] == {"image_width": 512}
    assert context["render_config"] == {"image_width": 1024}
    assert context["auto_prompt_config"] == {"user_prompt": "aged"}
    assert context["variations_config"] == {"count": 2}
    assert context["config"] is config
