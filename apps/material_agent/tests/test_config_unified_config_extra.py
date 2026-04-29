# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused coverage for unified configuration helper branches."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest


class _Resolver:
    def __init__(self, tmp_path: Path) -> None:
        self.base = tmp_path.resolve()
        self.working_dir = self.base / "work"
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = "session-1"
        self.input_usd = self.base / "input.usd"
        self.input_usd.touch()
        self.output_usd = self.base / "output" / "output.usd"
        self.output_usd.parent.mkdir(parents=True, exist_ok=True)
        self.layer_only = True
        self.flatten_output = False
        self.prim_path = "/Root/Part"

        refs_dir = self.base / "refs"
        refs_dir.mkdir(exist_ok=True)
        self.reference_images = [refs_dir / "ref.png"]
        self.reference_images[0].touch()
        self.reference_pdfs = [refs_dir / "spec.pdf"]
        self.reference_pdfs[0].touch()

    def _resolve_path(self, value: str | Path | None) -> Path | None:
        if value is None:
            return None
        path = Path(value)
        return path if path.is_absolute() else (self.base / path).resolve()

    def get_step_output_dir(self, step_name: str) -> Path:
        path = self.working_dir / step_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_usd_dataset_dir(self) -> Path:
        return self.get_step_output_dir("build_dataset_usd")

    def get_vectorstore_dir(self) -> Path:
        return self.get_step_output_dir("build_dataset_pdf_vectorstore")

    def get_dataset_dir(self) -> Path:
        return self.get_step_output_dir("build_dataset_prepare_dataset")

    def get_predictions_dir(self) -> Path:
        return self.get_step_output_dir("predict")

    def get_step_dataset_file(self, step_name: str) -> Path:
        return self.get_step_output_dir(step_name) / "dataset.jsonl"

    def get_step_predictions_file(self, step_name: str = "predict") -> Path:
        return self.get_step_output_dir(step_name) / "predictions.jsonl"


def _materials_data() -> dict[str, Any]:
    return {
        "library_path": "/materials/library.usd",
        "entries": [
            {
                "name": "Steel",
                "binding": "/World/Looks/Steel",
                "description": "Brushed metal",
            },
            {
                "name": "Wood",
                "binding": "/World/Looks/Wood",
            },
        ],
    }


def _load_unified_config():
    try:
        import material_agent.config.unified_config as unified_config
    except ImportError:
        import material_agent.config.unified_config as unified_config

    return unified_config


def test_run_loads_config_from_file_and_injects_session_id(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    UnifiedPipelineConfigTask = _load_unified_config().UnifiedPipelineConfigTask
    config_file = tmp_path / "config.yaml"
    (tmp_path / "input.usd").touch()
    config_file.write_text(
        """
project:
  name: demo
input:
  usd_path: input.usd
steps:
  predict:
    enabled: true
""".strip(),
        encoding="utf-8",
    )

    task = UnifiedPipelineConfigTask()
    monkeypatch.setattr(task.validator, "validate", lambda config: None)
    monkeypatch.setattr(task, "_parse_materials", lambda *args: None)
    monkeypatch.setattr(task, "_determine_steps", lambda *args: ["predict"])
    monkeypatch.setattr(task, "_build_step_configs", lambda *args: {"predict": {}})
    monkeypatch.setattr(task, "_log_summary", lambda *args: None)

    result = task.run({"config_path": str(config_file), "session_id": "override-1"})

    assert result["config_path"] == config_file
    assert result["config"]["project"]["session_id"] == "override-1"


def test_run_uses_config_dict_with_original_config_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    UnifiedPipelineConfigTask = _load_unified_config().UnifiedPipelineConfigTask
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "input.usd").touch()

    task = UnifiedPipelineConfigTask()
    monkeypatch.setattr(task.validator, "validate", lambda config: None)
    monkeypatch.setattr(task, "_parse_materials", lambda *args: None)
    monkeypatch.setattr(task, "_determine_steps", lambda *args: ["predict"])
    monkeypatch.setattr(task, "_build_step_configs", lambda *args: {"predict": {}})
    monkeypatch.setattr(task, "_log_summary", lambda *args: None)

    result = task.run(
        {
            "config_dict": {
                "project": {"name": "demo"},
                "input": {"usd_path": "input.usd"},
                "steps": {"predict": {"enabled": True}},
            },
            "config_path": str(config_dir / "config.yaml"),
        }
    )

    assert result["path_resolver"].input_usd == (config_dir / "input.usd").resolve()


def test_run_wraps_yaml_parse_errors(tmp_path: Path) -> None:
    UnifiedPipelineConfigTask = _load_unified_config().UnifiedPipelineConfigTask
    config_file = tmp_path / "bad.yaml"
    config_file.write_text("[", encoding="utf-8")

    with pytest.raises(ValueError, match="Failed to parse YAML configuration"):
        UnifiedPipelineConfigTask().run({"config_path": str(config_file)})


def test_merge_with_defaults_handles_external_materials_and_missing_steps() -> None:
    UnifiedPipelineConfigTask = _load_unified_config().UnifiedPipelineConfigTask
    task = UnifiedPipelineConfigTask()

    merged = task._merge_with_defaults({"materials": {"path": "materials.yaml"}})

    assert merged["materials"]["path"] == "materials.yaml"
    assert merged["materials"]["library_path"] is None
    assert "entries" not in merged["materials"]
    assert merged["steps"] == {}


def test_parse_materials_handles_none_inline_and_external_sources(
    tmp_path: Path,
) -> None:
    UnifiedPipelineConfigTask = _load_unified_config().UnifiedPipelineConfigTask
    task = UnifiedPipelineConfigTask()
    resolver = _Resolver(tmp_path)

    assert task._parse_materials({}, resolver) is None

    inline = task._parse_materials(
        {
            "materials": {
                "library_path": "libs/materials.usd",
                "entries": [{"name": "Steel", "binding": "/Steel"}],
            }
        },
        resolver,
    )
    assert inline["library_path"] == str(
        (tmp_path / "libs" / "materials.usd").resolve()
    )

    materials_file = tmp_path / "materials.yaml"
    materials_file.write_text(
        """
library_path: library/materials.usd
entries:
  - name: Steel
    binding: /World/Looks/Steel
""".strip(),
        encoding="utf-8",
    )
    external = task._parse_materials(
        {"materials": {"path": "materials.yaml"}}, resolver
    )
    assert external["library_path"] == str(
        (tmp_path / "library" / "materials.usd").resolve()
    )
    assert external["entries"][0]["name"] == "Steel"


@pytest.mark.parametrize(
    ("filename", "contents", "error_type", "message"),
    [
        ("missing.yaml", None, FileNotFoundError, "Materials file not found"),
        ("broken.yaml", "[", ValueError, "Failed to parse materials file"),
        ("empty.yaml", "", ValueError, "Materials file is empty"),
    ],
)
def test_parse_materials_error_paths(
    tmp_path: Path,
    filename: str,
    contents: str | None,
    error_type: type[Exception],
    message: str,
) -> None:
    UnifiedPipelineConfigTask = _load_unified_config().UnifiedPipelineConfigTask
    task = UnifiedPipelineConfigTask()
    resolver = _Resolver(tmp_path)
    target = tmp_path / filename
    if contents is not None:
        target.write_text(contents, encoding="utf-8")

    with pytest.raises(error_type, match=message):
        task._parse_materials({"materials": {"path": filename}}, resolver)


def test_determine_steps_applies_skip_and_only_filters() -> None:
    UnifiedPipelineConfigTask = _load_unified_config().UnifiedPipelineConfigTask
    task = UnifiedPipelineConfigTask()
    config = {
        "steps": {
            "build_dataset_usd": {"enabled": True},
            "predict": {"enabled": True},
            "apply": {"enabled": True},
        }
    }

    steps = task._determine_steps(
        config,
        {"skip_steps": ["build_dataset_usd"], "only_steps": ["predict"]},
    )

    assert steps == ["predict"]


def test_build_step_configs_updates_resolver_after_optimize(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    UnifiedPipelineConfigTask = _load_unified_config().UnifiedPipelineConfigTask
    task = UnifiedPipelineConfigTask()
    resolver = _Resolver(tmp_path)
    task.validator = type(
        "_Validator",
        (),
        {"validate_step_requirements": staticmethod(lambda *args: None)},
    )()

    monkeypatch.setattr(task, "_merge_step_config", lambda step_name, user_config: {})

    def fake_autowire(
        step_name, step_config, path_resolver, materials_data, full_config
    ):
        if step_name == "optimize_usd":
            return {
                "output_usd_path": str(
                    path_resolver.get_step_output_dir("optimize_usd")
                    / "optimized_input.usd"
                )
            }
        return {"observed_input_usd": str(path_resolver.input_usd)}

    monkeypatch.setattr(task, "_autowire_paths", fake_autowire)

    built = task._build_step_configs(
        ["optimize_usd", "build_dataset_usd"],
        {"steps": {}},
        resolver,
        None,
    )

    assert built["build_dataset_usd"]["observed_input_usd"].endswith(
        "optimized_input.usd"
    )
    assert resolver.input_usd.name == "optimized_input.usd"


def test_deep_merge_recurses_nested_dictionaries() -> None:
    UnifiedPipelineConfigTask = _load_unified_config().UnifiedPipelineConfigTask
    task = UnifiedPipelineConfigTask()

    merged = task._deep_merge(
        {"predict": {"vlm": {"model": "base", "temperature": 1.0}}, "enabled": True},
        {"predict": {"vlm": {"model": "override"}}},
    )

    assert merged["predict"]["vlm"] == {"model": "override", "temperature": 1.0}
    assert merged["enabled"] is True


def test_autowire_validation_and_basic_setup_steps(tmp_path: Path) -> None:
    UnifiedPipelineConfigTask = _load_unified_config().UnifiedPipelineConfigTask
    task = UnifiedPipelineConfigTask()
    resolver = _Resolver(tmp_path)

    validate_input = task._autowire_paths("validate_input", {}, resolver, None, {})
    assert validate_input["input_usd_path"] == str(resolver.input_usd)
    assert validate_input["validation_config"] == {}

    validate_output = task._autowire_paths("validate_output", {}, resolver, None, {})
    assert validate_output["input_usd_path"] == str(resolver.output_usd)
    assert validate_output["validation_config"] == {}

    with pytest.raises(ValueError, match="not supported for validate_output"):
        task._autowire_paths(
            "validate_output", {"on_failure": "fix"}, resolver, None, {}
        )

    optimize = task._autowire_paths("optimize_usd", {}, resolver, None, {})
    assert optimize["input_usd_path"] == str(resolver.input_usd)
    assert optimize["optimization_config"] == {}

    preview = task._autowire_paths("render_preview", {}, resolver, None, {})
    assert preview["usd_path"] == str(resolver.input_usd)

    resolver.reference_images = [resolver.reference_images[0]]
    reference = task._autowire_paths("generate_reference_image", {}, resolver, None, {})
    assert reference["reference_images"] == [str(resolver.reference_images[0])]


def test_autowire_build_dataset_steps(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    unified_config = _load_unified_config()
    UnifiedPipelineConfigTask = unified_config.UnifiedPipelineConfigTask
    task = UnifiedPipelineConfigTask()
    resolver = _Resolver(tmp_path)

    class _RendererConfig:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        def get_rendering_modes_config(
            self, rendering_modes_raw: dict[str, Any]
        ) -> dict[str, Any]:
            assert "beauty" in rendering_modes_raw
            return {"beauty": {}, "linear_depth": {}}

    monkeypatch.setattr(unified_config, "RendererConfig", _RendererConfig)

    usd_cfg = task._autowire_paths(
        "build_dataset_usd",
        {"renderer": {"rendering_modes": {"beauty": {}, "linear_depth": {}}}},
        resolver,
        None,
        {},
    )
    assert usd_cfg["prim_filters"]["root_prim"] == resolver.prim_path
    assert usd_cfg["renderer"]["rgb_rendering_modes"] == ["beauty"]
    assert usd_cfg["renderer"]["sensor_rendering_modes"] == ["linear_depth"]

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    pdf_cfg = task._autowire_paths(
        "build_dataset_pdf_vectorstore",
        {"source": "docs"},
        resolver,
        None,
        {},
    )
    assert pdf_cfg["source"] == str(docs_dir.resolve())
    assert pdf_cfg["output_dir"] == str(resolver.get_vectorstore_dir())

    prep_cfg = task._autowire_paths(
        "build_dataset_prepare_dataset",
        {"prompts": {}},
        resolver,
        _materials_data(),
        {"steps": {"build_dataset_pdf_vectorstore": {"enabled": True}}},
    )
    assert prep_cfg["usd_dir"] == str(resolver.get_usd_dataset_dir())
    assert prep_cfg["dataset"] == str(resolver.get_dataset_dir())
    assert prep_cfg["models"] == ["."]
    assert prep_cfg["vector_store"].endswith("vector_store")
    assert prep_cfg["reference_images"] == [str(resolver.reference_images[0])]
    assert prep_cfg["reference_pdfs"] == [str(resolver.reference_pdfs[0])]
    assert prep_cfg["materials_list"] == ["Steel", "Wood"]
    assert "Brushed metal" in prep_cfg["_materials_formatted"]
    assert "**Material name**: Wood" in prep_cfg["_materials_formatted"]


def test_autowire_prediction_validation_and_apply_steps(tmp_path: Path) -> None:
    UnifiedPipelineConfigTask = _load_unified_config().UnifiedPipelineConfigTask
    task = UnifiedPipelineConfigTask()
    resolver = _Resolver(tmp_path)
    materials_data = _materials_data()

    predict_cfg = task._autowire_paths("predict", {}, resolver, None, {})
    assert predict_cfg["dataset"].endswith("dataset.jsonl")
    assert predict_cfg["output_dir"] == str(resolver.get_predictions_dir())

    validate_cfg = task._autowire_paths(
        "validate_predictions", {}, resolver, materials_data, {}
    )
    assert validate_cfg["material_names"] == ["Steel", "Wood"]

    harmonize_cfg = task._autowire_paths(
        "harmonize_predictions", {}, resolver, materials_data, {}
    )
    assert harmonize_cfg["material_names"] == ["Steel", "Wood"]

    apply_cfg = task._autowire_paths("apply", {}, resolver, materials_data, {})
    assert apply_cfg["input_usd_path"] == str(resolver.input_usd)
    assert apply_cfg["output_usd_path"] == str(resolver.output_usd)
    assert apply_cfg["layer_only"] is True
    assert apply_cfg["flatten_output"] is False
    assert (
        apply_cfg["materials_mapping"]["material_library_path"]
        == "/materials/library.usd"
    )
    assert apply_cfg["materials_mapping"]["Steel"] == "/World/Looks/Steel"


def test_autowire_refine_with_minimal_defaults(tmp_path: Path) -> None:
    UnifiedPipelineConfigTask = _load_unified_config().UnifiedPipelineConfigTask
    task = UnifiedPipelineConfigTask()
    resolver = _Resolver(tmp_path)
    resolver.reference_images = []

    refine_cfg = task._autowire_paths("refine", {}, resolver, _materials_data(), {})

    assert refine_cfg["dataset"].endswith("dataset.jsonl")
    assert "system_prompt_file" not in refine_cfg["predict"]
    assert "vlm" in refine_cfg["predict"]
    assert refine_cfg["judge"]["backend"]
    assert refine_cfg["llm_judge"] == refine_cfg["judge"]
    assert refine_cfg["apply"]["materials_mapping"]["Wood"] == "/World/Looks/Wood"


def test_autowire_refine_with_custom_predict_and_judge(tmp_path: Path) -> None:
    UnifiedPipelineConfigTask = _load_unified_config().UnifiedPipelineConfigTask
    task = UnifiedPipelineConfigTask()
    resolver = _Resolver(tmp_path)

    refine_cfg = task._autowire_paths(
        "refine",
        {
            "predict": {
                "vlm": {"backend": "custom-vlm"},
                "max_workers": 7,
                "system_prompt_file": "legacy.txt",
            },
            "iteration": {"max_iterations": 5, "save_intermediate": False},
            "judge": {"vlm": {"backend": "judge-vlm"}},
        },
        resolver,
        _materials_data(),
        {},
    )

    assert "system_prompt_file" not in refine_cfg["predict"]
    assert refine_cfg["vlm"] == {"backend": "custom-vlm"}
    assert refine_cfg["max_workers"] == 7
    assert refine_cfg["max_iterations"] == 5
    assert refine_cfg["save_intermediate"] is False
    assert refine_cfg["vlm_judge"] == {"backend": "judge-vlm"}
    assert refine_cfg["judge"]["reference_images"] == [
        str(resolver.reference_images[0])
    ]


def test_autowire_restore_and_render_steps(tmp_path: Path) -> None:
    UnifiedPipelineConfigTask = _load_unified_config().UnifiedPipelineConfigTask
    task = UnifiedPipelineConfigTask()
    resolver = _Resolver(tmp_path)

    restore_cfg = task._autowire_paths("restore_usd", {}, resolver, None, {})
    assert restore_cfg["input_usd_path"] == str(resolver.output_usd)
    assert restore_cfg["output_usd_path"].endswith("restored_output.usd")
    assert restore_cfg["restore_config"] == {}

    render_cfg = task._autowire_paths("render", {}, resolver, None, {})
    assert render_cfg["input_usd_path"] == str(resolver.output_usd)
    assert render_cfg["prim_path"] == resolver.prim_path
    assert render_cfg["output_path"] == str(resolver.output_usd.parent)

    custom_render_cfg = task._autowire_paths(
        "render",
        {"input_usd_path": "custom/input.usd", "output_path": "renders"},
        resolver,
        None,
        {},
    )
    assert custom_render_cfg["input_usd_path"] == str(
        (tmp_path / "custom" / "input.usd").resolve()
    )
    assert custom_render_cfg["output_path"] == str((tmp_path / "renders").resolve())


def test_log_summary_includes_optional_description_and_library(tmp_path: Path) -> None:
    unified_config = _load_unified_config()
    UnifiedPipelineConfigTask = unified_config.UnifiedPipelineConfigTask
    task = UnifiedPipelineConfigTask()
    resolver = _Resolver(tmp_path)

    with patch.object(unified_config.logger, "info") as info:
        task._log_summary(
            {"project": {"name": "demo", "description": "Detailed project"}},
            resolver,
            _materials_data(),
            ["predict", "apply"],
        )

    logged = " ".join(" ".join(map(str, call.args)) for call in info.call_args_list)
    assert "Detailed project" in logged
    assert "/materials/library.usd" in logged
