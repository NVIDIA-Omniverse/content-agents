# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Additional tests for legacy config loader tasks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
import yaml
from pxr import Usd, UsdGeom

import material_agent.tasks.config_benchmark as benchmark_mod
import material_agent.tasks.config_evaluate as evaluate_mod
import material_agent.tasks.config_pdf_vectorstore as pdf_mod
import material_agent.tasks.config_predict as predict_mod
import material_agent.tasks.config_prepare_dataset as prepare_mod
import material_agent.tasks.generate_ref_image_config as gen_ref_mod
import material_agent.tasks.render_config as render_mod
import material_agent.tasks.render_preview_config as preview_mod
from material_agent.tasks import ModelProvisioningTask
from material_agent.tasks.config_benchmark import BenchmarkConfigTask
from material_agent.tasks.config_evaluate import EvaluateConfigTask
from material_agent.tasks.config_pdf_vectorstore import PDFVectorstoreConfigTask
from material_agent.tasks.config_predict import PredictConfigTask
from material_agent.tasks.config_prepare_dataset import PrepareDatasetConfigTask
from material_agent.tasks.generate_ref_image_config import GenerateRefImageConfigTask
from material_agent.tasks.render_config import RenderConfigTask
from material_agent.tasks.render_preview_config import RenderPreviewConfigTask


def _write_yaml(path: Path, data: dict) -> Path:
    path.write_text(yaml.safe_dump(data))
    return path


def _make_usd(path: Path) -> Path:
    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(root.GetPrim())
    stage.GetRootLayer().Save()
    return path


def _patch_listener(monkeypatch: pytest.MonkeyPatch, module: object) -> Mock:
    listener = Mock()
    monkeypatch.setattr(
        module,
        "get_listener",
        lambda context, logger_name=None: listener,
    )
    return listener


@pytest.mark.parametrize(
    ("task_cls", "filename", "empty_message"),
    [
        (PredictConfigTask, "predict.yaml", "Configuration file is empty"),
        (BenchmarkConfigTask, "benchmark.yaml", "Configuration file is empty"),
        (EvaluateConfigTask, "evaluate.yaml", "Configuration file is empty"),
        (PrepareDatasetConfigTask, "prepare.yaml", "Configuration file is empty"),
    ],
)
def test_basic_config_loader_validation_errors(
    task_cls: type,
    filename: str,
    empty_message: str,
    tmp_path: Path,
) -> None:
    task = task_cls()

    with pytest.raises(ValueError, match="config_path"):
        task.run({})

    with pytest.raises(FileNotFoundError):
        task.run({"config_path": str(tmp_path / filename)})

    config_path = tmp_path / filename
    config_path.write_text("")
    with pytest.raises(ValueError, match=empty_message):
        task.run({"config_path": str(config_path)})


def test_predict_config_task_loads_dataset_prompt_and_nim_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    listener = _patch_listener(monkeypatch, predict_mod)
    monkeypatch.setenv("MA_VLM_NIM_BASE_URL", "http://nim")

    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "dataset.jsonl").write_text("")
    (dataset_dir / "dataset.json").write_text(
        json.dumps(
            {"inference": {"prompts": [{"system_prompt": "Prompt from dataset.json"}]}}
        )
    )
    config_path = _write_yaml(
        tmp_path / "predict.yaml",
        {
            "dataset": str(dataset_dir / "dataset.jsonl"),
            "output_dir": "predictions",
            "vlm": {"backend": "openai", "model": "gpt"},
            "llm": {"backend": "nim"},
            "max_workers": 8,
            "prediction_batch_size": 2,
            "report": {
                "image_max_size": 512,
                "image_format": "jpeg",
                "image_quality": 80,
            },
        },
    )

    context = PredictConfigTask().run({"config_path": str(config_path)})

    assert context["dataset_path"] == str(dataset_dir / "dataset.jsonl")
    assert context["output_dir"] == "predictions"
    assert context["vlm_config"]["backend"] == "nim"
    assert context["vlm_config"]["base_url"] == "http://nim"
    assert context["config"]["vlm"]["base_url"] == "http://nim"
    assert context["llm_config"]["backend"] == "nim"
    assert context["llm_config"]["base_url"] == "http://nim"
    assert context["llm_config"]["model"] == "gpt"
    assert context["config"]["llm"]["base_url"] == "http://nim"
    assert context["system_prompt"] == "Prompt from dataset.json"
    assert context["max_workers"] == 8
    assert context["prediction_batch_size"] == 2
    assert context["report_image_max_size"] == 512
    assert context["report_image_format"] == "jpeg"
    assert context["report_image_quality"] == 80
    listener.info.assert_any_call(
        "Loaded system prompt from dataset.json (v0.2 format)"
    )


def test_predict_config_task_prefers_llm_nim_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_listener(monkeypatch, predict_mod)
    monkeypatch.setenv("MA_VLM_NIM_BASE_URL", "http://vlm-nim:8000/v1")
    monkeypatch.setenv("MA_LLM_NIM_BASE_URL", "http://llm-nim:8000/v1")

    config_path = _write_yaml(
        tmp_path / "predict.yaml",
        {
            "dataset": str(tmp_path / "dataset.jsonl"),
            "output_dir": "predictions",
            "vlm": {"backend": "openai", "model": "vlm-model"},
            "llm": {"backend": "openai", "model": "llm-model"},
        },
    )

    context = PredictConfigTask().run({"config_path": str(config_path)})

    assert context["vlm_config"]["backend"] == "nim"
    assert context["vlm_config"]["base_url"] == "http://vlm-nim:8000/v1"
    assert context["llm_config"]["backend"] == "nim"
    assert context["llm_config"]["base_url"] == "http://llm-nim:8000/v1"
    assert context["llm_config"]["model"] == "llm-model"


def test_predict_config_task_nim_override_drops_stale_provider_api_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_listener(monkeypatch, predict_mod)
    monkeypatch.setenv("MA_VLM_NIM_BASE_URL", "http://vlm-nim:8000/v1")
    monkeypatch.setenv("MA_NIM_API_KEY", "not-used")
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)

    config_path = _write_yaml(
        tmp_path / "predict.yaml",
        {
            "dataset": str(tmp_path / "dataset.jsonl"),
            "output_dir": "predictions",
            "vlm": {
                "backend": "openai",
                "model": "vlm-model",
                "api_key": "hosted-openai-key",
            },
        },
    )

    context = PredictConfigTask().run({"config_path": str(config_path)})

    assert context["vlm_config"]["backend"] == "nim"
    assert context["vlm_config"]["base_url"] == "http://vlm-nim:8000/v1"
    assert "api_key" not in context["vlm_config"]


def test_predict_config_task_nim_override_drops_stale_existing_nim_keys(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_listener(monkeypatch, predict_mod)
    monkeypatch.setenv("MA_VLM_NIM_BASE_URL", "http://vlm-nim:8000/v1")
    monkeypatch.setenv("MA_LLM_NIM_BASE_URL", "http://llm-nim:8000/v1")
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)

    config_path = _write_yaml(
        tmp_path / "predict.yaml",
        {
            "dataset": str(tmp_path / "dataset.jsonl"),
            "output_dir": "predictions",
            "vlm": {
                "backend": "nim",
                "model": "hosted-vlm",
                "base_url": "https://integrate.api.nvidia.com/v1",
                "api_key": "hosted-nim-vlm-key",
            },
            "llm": {
                "backend": "nim",
                "model": "hosted-llm",
                "base_url": "https://integrate.api.nvidia.com/v1",
                "api_key": "hosted-nim-llm-key",
            },
        },
    )

    context = PredictConfigTask().run({"config_path": str(config_path)})

    assert context["vlm_config"]["backend"] == "nim"
    assert context["vlm_config"]["base_url"] == "http://vlm-nim:8000/v1"
    assert "api_key" not in context["vlm_config"]
    assert context["llm_config"]["backend"] == "nim"
    assert context["llm_config"]["base_url"] == "http://llm-nim:8000/v1"
    assert "api_key" not in context["llm_config"]


def test_predict_config_task_nim_override_forwards_local_placeholder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_listener(monkeypatch, predict_mod)
    monkeypatch.setenv("MA_VLM_NIM_BASE_URL", "http://vlm-nim:8000/v1")
    monkeypatch.setenv("MA_NIM_API_KEY", "not-used")
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)

    config_path = _write_yaml(
        tmp_path / "predict.yaml",
        {
            "dataset": str(tmp_path / "dataset.jsonl"),
            "output_dir": "predictions",
            "vlm": {
                "backend": "openai",
                "model": "vlm-model",
                "api_key": "hosted-openai-key",
            },
        },
    )
    context = PredictConfigTask().run({"config_path": str(config_path)})

    captured: dict[str, Any] = {}

    def fake_create_vlm(**kwargs: Any) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        "world_understanding.agentic.domain_tasks.model_provisioning.create_vlm",
        fake_create_vlm,
    )

    ModelProvisioningTask().run({"config": {"vlm": context["vlm_config"]}}, None)

    assert captured["api_key"] == "not-used"
    assert captured["base_url"] == "http://vlm-nim:8000/v1"


def test_predict_config_task_falls_back_to_prompt_file_and_warns(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    listener = _patch_listener(monkeypatch, predict_mod)

    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "dataset.jsonl").write_text("")
    (dataset_dir / "dataset.json").write_text("{bad json")
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("Prompt from file")

    config_path = _write_yaml(
        tmp_path / "predict.yaml",
        {
            "dataset": str(dataset_dir / "dataset.jsonl"),
            "system_prompt_file": str(prompt_file),
        },
    )

    context = PredictConfigTask().run({"config_path": str(config_path)})
    assert context["system_prompt"] == "Prompt from file"
    assert context["config"]["system_prompt"] == "Prompt from file"
    assert listener.warning.call_count == 1

    missing_prompt = _write_yaml(
        tmp_path / "predict-missing.yaml",
        {
            "dataset": str(dataset_dir / "dataset.jsonl"),
            "system_prompt_file": str(tmp_path / "missing.txt"),
        },
    )
    context = PredictConfigTask().run({"config_path": str(missing_prompt)})
    assert context["system_prompt"] is None
    assert listener.warning.call_count >= 2


def test_benchmark_config_task_loads_or_warns_for_system_prompt_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    listener = _patch_listener(monkeypatch, benchmark_mod)
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("benchmark prompt")

    config_path = _write_yaml(
        tmp_path / "benchmark.yaml",
        {
            "dataset": "dataset.jsonl",
            "output_dir": "out",
            "vlm": {"backend": "nim"},
            "llm": {"backend": "nim"},
            "llm_judge": {"backend": "nim"},
            "max_workers": 9,
            "system_prompt_file": str(prompt_file),
        },
    )

    context = BenchmarkConfigTask().run({"config_path": str(config_path)})
    assert context["system_prompt"] == "benchmark prompt"
    assert context["config"]["system_prompt"] == "benchmark prompt"
    assert context["max_workers"] == 9

    missing_config = _write_yaml(
        tmp_path / "benchmark-missing.yaml",
        {
            "dataset": "dataset.jsonl",
            "system_prompt_file": str(tmp_path / "missing.txt"),
        },
    )
    context = BenchmarkConfigTask().run({"config_path": str(missing_config)})
    assert context["system_prompt"] is None
    listener.warning.assert_called()


def test_evaluate_config_task_resolves_paths_from_cwd_and_config_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_listener(monkeypatch, evaluate_mod)
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text("")
    dataset = config_dir / "dataset.jsonl"
    dataset.write_text("")

    monkeypatch.chdir(tmp_path)
    config_path = _write_yaml(
        config_dir / "evaluate.yaml",
        {
            "predictions_path": "predictions.jsonl",
            "dataset_path": "dataset.jsonl",
            "output_dir": "reports",
            "llm_judge": {"backend": "nim"},
        },
    )

    context = EvaluateConfigTask().run({"config_path": str(config_path)})

    assert context["predictions_path"] == predictions.resolve()
    assert context["dataset_path"] == dataset.resolve()
    assert context["output_dir"] == (config_dir / "reports").resolve()
    assert context["llm_judge_config"] == {"backend": "nim"}


def test_prepare_dataset_config_task_uses_config_models_and_discovers_from_usd_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    listener = _patch_listener(monkeypatch, prepare_mod)
    usd_dir = tmp_path / "usd"
    usd_dir.mkdir()
    for model_name in ["model_b", "model_a"]:
        model_dir = usd_dir / model_name
        model_dir.mkdir()
        for filename in ["dataset.json", "prims.jsonl", "usd_model.json"]:
            (model_dir / filename).write_text("{}")
    incomplete_dir = usd_dir / "ignore_me"
    incomplete_dir.mkdir()
    (incomplete_dir / "dataset.json").write_text("{}")

    with_models = _write_yaml(
        tmp_path / "prepare-with-models.yaml",
        {
            "usd_dir": str(usd_dir),
            "vector_store": "store",
            "dataset": "dataset.jsonl",
            "models": ["configured-model"],
        },
    )
    context = PrepareDatasetConfigTask().run({"config_path": str(with_models)})
    assert context["models"] == ["configured-model"]
    assert context["vector_store_path"] == Path("store")
    assert context["dataset_path"] == Path("dataset.jsonl")

    discovered = _write_yaml(
        tmp_path / "prepare-discover.yaml", {"usd_dir": str(usd_dir)}
    )
    context = PrepareDatasetConfigTask().run({"config_path": str(discovered)})
    assert context["models"] == ["model_a", "model_b"]
    listener.info.assert_any_call("Discovered 2 models from usd_dir")

    missing = _write_yaml(tmp_path / "prepare-missing.yaml", {"usd_dir": "missing"})
    context = PrepareDatasetConfigTask().run({"config_path": str(missing)})
    assert context["models"] == []
    listener.warning.assert_called_with("No models found - usd_dir doesn't exist")


def test_pdf_vectorstore_config_task_supports_dicts_paths_and_validation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_listener(monkeypatch, pdf_mod)
    override_source = tmp_path / "override.pdf"
    override_source.write_text("pdf")

    context = PDFVectorstoreConfigTask().run(
        {
            "config_dict": {
                "source": "ignored.pdf",
                "output_dir": "ignored",
                "embedding": {"service": "svc", "model": "mdl"},
                "chunk_size": 128,
                "chunk_overlap": 12,
                "image_embedding_type": "image",
                "include_filename_metadata": False,
            },
            "source_override": str(override_source),
            "output_dir_override": str(tmp_path / "out"),
        }
    )
    assert context["source_path"] == str(override_source)
    assert context["output_dir"] == str(tmp_path / "out")
    assert context["embedding_model"] == "svc/mdl"
    assert context["chunk_size"] == 128
    assert context["chunk_overlap"] == 12
    assert context["image_embedding_type"] == "image"
    assert context["include_filename_metadata"] is False

    source = tmp_path / "doc.pdf"
    source.write_text("pdf")
    config_path = _write_yaml(
        tmp_path / "pdf.yaml",
        {"source": "doc.pdf", "output_dir": "vector", "embedding": {}},
    )
    context = PDFVectorstoreConfigTask().run({"config_path": str(config_path)})
    assert context["source_path"] == str(source)
    assert context["output_dir"] == str(tmp_path / "vector")
    assert context["embedding_model"] is None

    with pytest.raises(ValueError, match="Either config_path or config_dict"):
        PDFVectorstoreConfigTask().run({})

    with pytest.raises(ValueError, match="source not specified"):
        PDFVectorstoreConfigTask().run({"config_dict": {"output_dir": "x"}})

    with pytest.raises(ValueError, match="output_dir not specified"):
        PDFVectorstoreConfigTask().run({"config_dict": {"source": "x"}})


def test_render_config_task_supports_direct_and_unified_configs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_listener(monkeypatch, render_mod)
    input_usd = _make_usd(tmp_path / "input.usd")

    direct_config = _write_yaml(
        tmp_path / "render-direct.yaml",
        {
            "enabled": True,
            "backend": "ovrtx",
            "input_usd_path": "input.usd",
            "output_path": "renders",
            "image_width": 640,
            "camera_corners": "+x",
            "camera_margin": 1.5,
            "background_color": [0.0, 0.0, 0.0],
            "flatten_before_render": False,
            "prim_path": "/World/Mesh",
            "clear_materials": True,
        },
    )
    context = RenderConfigTask().run({"config_path": str(direct_config)})
    assert context["input_usd_path"] == str(input_usd)
    assert context["output_base_path"] == str(tmp_path / "renders")
    assert context["render_config"]["camera_corners"] == ["+x"]
    assert context["render_config"]["prim_path"] == "/World/Mesh"
    assert context["render_config"]["clear_materials"] is True
    assert context["flatten_before_render"] is False

    override_input = _make_usd(tmp_path / "override.usd")
    unified_config = _write_yaml(
        tmp_path / "render-unified.yaml",
        {
            "project": {"working_dir": "work"},
            "output": {"usd_path": "input.usd"},
            "steps": {"render": {"enabled": True}},
        },
    )
    context = RenderConfigTask().run(
        {
            "config_path": str(unified_config),
            "input_usd_override": str(override_input),
            "output_path_override": str(tmp_path / "override-renders"),
        }
    )
    assert context["input_usd_path"] == str(override_input)
    assert context["output_base_path"] == str(tmp_path / "override-renders")
    assert context["flatten_before_render"] is True

    bad_config = _write_yaml(tmp_path / "render-bad.yaml", {"other": {}})
    with pytest.raises(ValueError, match="No 'render' configuration found"):
        RenderConfigTask().run({"config_path": str(bad_config)})


def test_render_preview_and_generate_ref_image_config_tasks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_listener(monkeypatch, preview_mod)
    _patch_listener(monkeypatch, gen_ref_mod)

    usd_path = _make_usd(tmp_path / "scene.usd")
    preview_config = _write_yaml(
        tmp_path / "preview.yaml",
        {
            "usd_path": "scene.usd",
            "backend": "remote",
            "cameras": ["+x", "-x"],
            "prim_filters": {"types": ["Mesh"]},
        },
    )
    preview_context = RenderPreviewConfigTask().run(
        {"config_path": str(preview_config)}
    )
    assert preview_context["usd_path"] == str(usd_path)
    assert preview_context["output_dir"] == str(tmp_path / "preview")
    assert preview_context["render_config"]["flatten_before_render"] is False
    assert preview_context["prim_filters"] == {"types": ["Mesh"]}

    missing_usd = _write_yaml(tmp_path / "preview-missing.yaml", {"backend": "remote"})
    with pytest.raises(ValueError, match="usd_path"):
        RenderPreviewConfigTask().run({"config_path": str(missing_usd)})

    generated = GenerateRefImageConfigTask().run(
        {
            "config_path": str(
                _write_yaml(
                    tmp_path / "generate.yaml",
                    {
                        "rendered_preview_paths": ["a.png", "b.png"],
                        "prompt": "make it metallic",
                        "image_gen": {"backend": "nvidia", "model": "model-x"},
                        "output_dir": "refs",
                        "num_images": 3,
                        "reference_images": ["ref.png"],
                        "identification": {"category": "chair"},
                        "additional_prompt": "keep it clean",
                    },
                )
            )
        }
    )
    assert generated["rendered_preview_paths"] == ["a.png", "b.png"]
    assert generated["image_gen_prompt"] == "make it metallic"
    assert generated["num_images"] == 3
    assert generated["output_dir"] == str((tmp_path / "refs").resolve())
    assert generated["reference_images"] == ["ref.png"]
    assert generated["identification"] == {"category": "chair"}
    assert generated["additional_prompt"] == "keep it clean"

    auto_prompt = GenerateRefImageConfigTask().run(
        {
            "config_path": str(
                _write_yaml(
                    tmp_path / "generate-auto.yaml",
                    {
                        "rendered_preview_paths": ["a.png"],
                        "identification": {"category": "table"},
                    },
                )
            )
        }
    )
    assert "image_gen_prompt" not in auto_prompt
    assert auto_prompt["identification"] == {"category": "table"}

    with pytest.raises(ValueError, match="rendered_preview_paths"):
        GenerateRefImageConfigTask().run(
            {
                "config_path": str(
                    _write_yaml(tmp_path / "generate-no-previews.yaml", {"prompt": "x"})
                )
            }
        )

    with pytest.raises(ValueError, match="prompt is required"):
        GenerateRefImageConfigTask().run(
            {
                "config_path": str(
                    _write_yaml(
                        tmp_path / "generate-no-prompt.yaml",
                        {"rendered_preview_paths": ["a.png"]},
                    )
                )
            }
        )

    with pytest.raises(TypeError, match="identification must be a dict"):
        GenerateRefImageConfigTask().run(
            {
                "config_path": str(
                    _write_yaml(
                        tmp_path / "generate-bad-identification.yaml",
                        {
                            "rendered_preview_paths": ["a.png"],
                            "prompt": "x",
                            "identification": "bad",
                        },
                    )
                )
            }
        )
