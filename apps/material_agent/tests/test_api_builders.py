# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for material_agent.api.builders."""

from __future__ import annotations

import pytest

from material_agent.api.builders import (
    build_apply_config,
    build_benchmark_config,
    build_cluster_prims_config,
    build_predict_config,
    build_unified_pipeline_config,
    build_vlm_config,
    get_required_fields,
)


def test_build_vlm_config_handles_optional_fields_and_llmgateway_default() -> None:
    config = build_vlm_config(
        backend="llmgateway_azure_openai",
        model="gpt-x",
        temperature=None,
        max_tokens=None,
        top_p=0.5,
    )

    assert config["backend"] == "llmgateway_azure_openai"
    assert config["model"] == "gpt-x"
    assert "temperature" not in config
    assert "max_tokens" not in config
    assert "llmgateway" in config
    assert config["top_p"] == 0.5


def test_build_predict_config_requires_dataset_and_serializes_paths() -> None:
    with pytest.raises(ValueError, match="dataset_path is required"):
        build_predict_config("")

    config = build_predict_config(
        "dataset.jsonl",
        output_dir="out",
        system_prompt="prompt",
        system_prompt_file="prompt.txt",
        temperature=0.2,
        max_tokens=128,
        extra="value",
    )

    assert config["dataset"] == "dataset.jsonl"
    assert config["output_dir"] == "out"
    assert config["system_prompt"] == "prompt"
    assert config["system_prompt_file"] == "prompt.txt"
    assert config["vlm"]["temperature"] == 0.2
    assert config["llm"]["max_tokens"] == 128
    assert config["extra"] == "value"


def test_build_benchmark_config_defaults_judge_and_llm_from_vlm() -> None:
    config = build_benchmark_config("dataset.jsonl", vlm_backend="nim", vlm_model="m1")

    assert config["dataset"] == "dataset.jsonl"
    assert config["vlm"]["backend"] == "nim"
    assert config["llm"]["model"] == "m1"
    assert config["judge"]["backend"] == "nim"
    assert config["judge"]["model"] == "m1"


def test_build_apply_config_adds_optional_render_section() -> None:
    config = build_apply_config(
        input_usd_path="input.usd",
        predictions_path="predictions.jsonl",
        output_usd_path="output.usd",
        materials_library_path="materials.usd",
        materials_entries=[{"name": "Steel", "prim_path": "/Looks/Steel"}],
        layer_only=True,
        flatten=False,
        render_enabled=True,
        extra="value",
    )

    assert config["input_usd_path"] == "input.usd"
    assert config["materials"]["library_path"] == "materials.usd"
    assert config["materials"]["entries"][0]["name"] == "Steel"
    assert config["layer_only"] is True
    assert config["flatten"] is False
    assert config["render"] == {"enabled": True}
    assert config["extra"] == "value"


def test_build_cluster_prims_config_defaults_to_nim_nemotron_vl_embed() -> None:
    config = build_cluster_prims_config()

    assert config["enabled"] is True
    assert config["embedding_service"] == "nim"
    assert config["embedding_model"] == "nvidia/llama-nemotron-embed-vl-1b-v2"
    assert config["min_prims_to_activate"] == 50
    assert config["max_cluster_size"] == 25
    assert config["report"] == {"enabled": True}


def test_build_cluster_prims_config_defaults_nim_model() -> None:
    config = build_cluster_prims_config(embedding_service="nim")

    assert config["embedding_service"] == "nim"
    assert config["embedding_model"] == "nvidia/llama-nemotron-embed-vl-1b-v2"


def test_build_cluster_prims_config_allows_max_cluster_size_override() -> None:
    config = build_cluster_prims_config(max_cluster_size=12)

    assert config["max_cluster_size"] == 12


def test_build_cluster_prims_config_copies_default_thresholds() -> None:
    first = build_cluster_prims_config()
    first["complexity_thresholds"]["low"][2] = 0.1

    second = build_cluster_prims_config()

    assert second["complexity_thresholds"]["low"][2] == 0.98


def test_build_unified_pipeline_config_sets_steps_and_filters_path_like_keys() -> None:
    config = build_unified_pipeline_config(
        project_name="demo",
        input_usd_path="input.usd",
        materials_library_path="materials.usd",
        materials_entries=[{"name": "Steel", "prim_path": "/Looks/Steel"}],
        enabled_steps=["build_dataset_prepare_dataset", "predict", "apply"],
        session_id="session-1",
        working_dir=".work",
        output_usd_path="custom.usd",
        user_prompt="Use this prompt",
        extra="value",
    )

    assert config["project"] == {
        "name": "demo",
        "session_id": "session-1",
        "working_dir": ".work",
    }
    assert config["output"]["usd_path"] == "custom.usd"
    assert config["materials"]["library_path"] == "materials.usd"
    assert config["steps"]["predict"]["enabled"] is True
    assert config["steps"]["predict"]["vlm"]["backend"]
    assert config["steps"]["predict"]["llm"]["backend"]
    assert config["steps"]["build_dataset_prepare_dataset"]["prompts"]["vlm_user"] == (
        "Use this prompt"
    )
    assert "dataset" not in config["steps"]["predict"]
    assert "output_dir" not in config["steps"]["apply"]
    assert config["extra"] == "value"


def test_build_unified_pipeline_config_can_insert_cluster_prims() -> None:
    config = build_unified_pipeline_config(
        project_name="demo",
        input_usd_path="input.usd",
        materials_library_path="materials.usd",
        materials_entries=[],
        enabled_steps=["build_dataset_prepare_dataset", "predict", "apply"],
        enable_prim_clustering=True,
        cluster_prims_config={"min_prims_to_activate": 10},
    )

    assert list(config["steps"]) == [
        "build_dataset_prepare_dataset",
        "cluster_prims",
        "predict",
        "apply",
    ]
    assert config["steps"]["cluster_prims"]["enabled"] is True
    assert config["steps"]["cluster_prims"]["min_prims_to_activate"] == 10


def test_build_unified_pipeline_config_uses_default_steps_and_empty_output() -> None:
    config = build_unified_pipeline_config(
        project_name="demo",
        input_usd_path="input.usd",
        materials_library_path="materials.usd",
        materials_entries=[],
    )

    assert config["output"] == {}
    assert list(config["steps"]) == [
        "build_dataset_usd",
        "build_dataset_prepare_dataset",
        "predict",
        "apply",
    ]


def test_get_required_fields_returns_known_and_unknown_shapes() -> None:
    pipeline_fields = get_required_fields("pipeline")
    assert "project.name" in pipeline_fields["required"]
    assert "steps.*" in pipeline_fields["optional"]

    assert get_required_fields("unknown") == {"required": [], "optional": []}
