# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Additional CLI coverage for configure, run, manifest, and helpers."""

from __future__ import annotations

import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import typer

import material_agent.api as api
import material_agent.cli as cli
import material_agent.manifest as manifest


class _SpanContextManager:
    def __init__(self, span: Mock):
        self._span = span

    def __enter__(self) -> Mock:
        return self._span

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


def _patch_cli_common(monkeypatch: pytest.MonkeyPatch) -> tuple[Mock, list[str]]:
    logger = Mock()
    printed: list[str] = []
    monkeypatch.setattr(cli, "setup_logging", lambda **kwargs: logger)
    monkeypatch.setattr(
        cli.console,
        "print",
        lambda *args, **kwargs: printed.append(" ".join(map(str, args))),
    )
    monkeypatch.setattr(cli.console, "print_exception", lambda *args, **kwargs: None)
    return logger, printed


def _write_config(tmp_path: Path, contents: str) -> Path:
    config = tmp_path / "config.yaml"
    config.write_text(contents, encoding="utf-8")
    return config


def test_cli_helpers_strip_email_generate_session_and_print_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MA_USER_EMAIL", " user@example.com ")
    assert cli._get_cli_user_email() == "user@example.com"

    monkeypatch.setenv("MA_USER_EMAIL", "   ")
    assert cli._get_cli_user_email() is None

    monkeypatch.setattr(
        cli.uuid,
        "uuid4",
        lambda: uuid.UUID("12345678-1234-5678-1234-567812345678"),
    )
    assert (
        cli._get_cli_telemetry_session_id(None)
        == "12345678-1234-5678-1234-567812345678"
    )
    assert cli._get_cli_telemetry_session_id("session-1") == "session-1"

    rich_print = Mock()
    monkeypatch.setattr(cli, "print", rich_print)
    with pytest.raises(typer.Exit):
        cli.version_callback(True)
    rich_print.assert_called_once()


def test_main_handles_failed_telemetry_initialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger = Mock()
    monkeypatch.setattr(cli, "setup_logging", lambda **kwargs: logger)
    monkeypatch.setattr(
        cli,
        "TelemetryConfig",
        lambda: SimpleNamespace(enabled=True, service_name="svc", exporters=["otlp"]),
    )
    monkeypatch.setattr(cli, "initialize_telemetry", lambda config: None)

    cli.main()

    logger.warning.assert_called_once()
    assert cli.app.state["logger"] is logger


def test_main_registers_telemetry_shutdown_and_verbose_logging(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    logger = Mock()
    register = Mock()
    monkeypatch.setattr(cli, "setup_logging", lambda **kwargs: logger)
    monkeypatch.setattr(
        cli,
        "TelemetryConfig",
        lambda: SimpleNamespace(enabled=True, service_name="svc", exporters=["otlp"]),
    )
    monkeypatch.setattr(cli, "initialize_telemetry", lambda config: object())
    monkeypatch.setattr(cli.atexit, "register", register)

    log_file = tmp_path / "material.log"
    cli.main(verbose=True, log_file=log_file, log_level="DEBUG")

    register.assert_called_once_with(cli.shutdown_telemetry)
    logger.debug.assert_any_call("Verbose mode enabled")
    logger.debug.assert_any_call("Log level: DEBUG")
    logger.debug.assert_any_call(f"Logging to file: {log_file}")


def test_run_rejects_missing_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_cli_common(monkeypatch)
    monkeypatch.setattr(
        cli,
        "get_listener",
        lambda *args, **kwargs: SimpleNamespace(event=lambda *a, **k: None),
    )

    with pytest.raises(typer.Exit) as exc:
        cli.run(config=tmp_path / "missing.yaml")

    assert exc.value.exit_code == 1


def test_run_supports_unified_and_legacy_dry_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_cli_common(monkeypatch)
    events: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        cli,
        "get_listener",
        lambda *args, **kwargs: SimpleNamespace(
            event=lambda name, payload: events.append((name, payload))
        ),
    )

    unified = _write_config(
        tmp_path,
        """
project:
  name: demo
  working_dir: .demo
input:
  usd_path: input.usd
output:
  usd_path: output.usd
steps:
  build_dataset_usd: {}
  predict:
    enabled: true
  apply:
    enabled: false
""".strip(),
    )
    cli.run(
        config=unified,
        skip="predict",
        only="build_dataset_usd",
        dry_run=True,
    )

    legacy = _write_config(
        tmp_path,
        """
build_dataset_usd: {}
predict: {}
""".strip(),
    )
    cli.run(config=legacy, dry_run=True)

    assert events[0][0] == "pipeline.config.display"
    assert events[0][1]["skip_steps"] == ["predict"]
    assert events[0][1]["only_steps"] == ["build_dataset_usd"]


def test_run_dry_run_wraps_yaml_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_cli_common(monkeypatch)
    monkeypatch.setattr(
        cli,
        "get_listener",
        lambda *args, **kwargs: SimpleNamespace(event=lambda *a, **k: None),
    )
    config = _write_config(tmp_path, "[")

    with pytest.raises(typer.Exit) as exc:
        cli.run(config=config, dry_run=True)

    assert exc.value.exit_code == 1


def test_run_sets_failed_status_when_pipeline_execution_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    logger, printed = _patch_cli_common(monkeypatch)
    monkeypatch.setenv("MA_USER_EMAIL", "user@example.com")
    monkeypatch.setattr(
        cli,
        "get_listener",
        lambda *args, **kwargs: SimpleNamespace(event=lambda *a, **k: None),
    )

    config = _write_config(tmp_path, "project:\n  name: demo\n")
    span = Mock()
    tracer = Mock()
    tracer.start_as_current_span.return_value = _SpanContextManager(span)
    monkeypatch.setattr(cli, "get_tracer", Mock(return_value=tracer))
    monkeypatch.setattr(api, "CLIEventListener", lambda **kwargs: object())

    class _PipelineInput:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(api, "PipelineInput", _PipelineInput)
    monkeypatch.setattr(api, "run_pipeline", Mock(side_effect=RuntimeError("boom")))

    with pytest.raises(typer.Exit) as exc:
        cli.run(config=config, session_id="session-123", resume=True)

    assert exc.value.exit_code == 1
    span.set_attribute.assert_any_call("maa.pipeline.status", "failed")
    assert any("Pipeline checkpoint saved" in line for line in printed)
    logger.error.assert_called()


def test_pipeline_alias_forwards_all_options(monkeypatch: pytest.MonkeyPatch) -> None:
    _, printed = _patch_cli_common(monkeypatch)
    run_mock = Mock()
    monkeypatch.setattr(cli, "run", run_mock)

    cli.pipeline(
        config=Path("config.yaml"),
        skip="a",
        only="b",
        resume=True,
        dry_run=True,
        clean=True,
        verbose=True,
        log_file=Path("material.log"),
        log_level="DEBUG",
    )

    assert any("deprecated" in line.lower() for line in printed)
    run_mock.assert_called_once_with(
        config=Path("config.yaml"),
        skip="a",
        only="b",
        resume=True,
        dry_run=True,
        clean=True,
        verbose=True,
        log_file=Path("material.log"),
        log_level="DEBUG",
    )


def test_configure_success_builds_api_input(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_cli_common(monkeypatch)
    captured: dict[str, object] = {}

    class _ConfigureInput:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(api, "ConfigureInput", _ConfigureInput)
    monkeypatch.setattr(
        api,
        "run_configure",
        lambda params: SimpleNamespace(
            success=True,
            config_path=tmp_path / "pipeline.yaml",
            pipeline_name="demo",
            input_usd_path="scene.usd",
            materials_library_path="materials.usd",
        ),
    )

    cli.configure(
        output_config=tmp_path / "pipeline.yaml",
        materials_manifest=tmp_path / "materials.yaml",
        reference_images=[tmp_path / "ref1.png", tmp_path / "ref2.png"],
        force=True,
    )

    assert captured["output_config_path"] == tmp_path / "pipeline.yaml"
    assert captured["materials_manifest"] == tmp_path / "materials.yaml"
    assert captured["reference_images"] == [
        str(tmp_path / "ref1.png"),
        str(tmp_path / "ref2.png"),
    ]
    assert captured["force"] is True


def test_configure_wraps_file_exists_and_generic_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_cli_common(monkeypatch)
    monkeypatch.setattr(
        api, "ConfigureInput", lambda **kwargs: SimpleNamespace(**kwargs)
    )

    monkeypatch.setattr(
        api, "run_configure", Mock(side_effect=FileExistsError("exists"))
    )
    with pytest.raises(typer.Exit) as file_exists_exc:
        cli.configure(output_config=tmp_path / "pipeline.yaml")
    assert file_exists_exc.value.exit_code == 1

    monkeypatch.setattr(api, "run_configure", Mock(side_effect=RuntimeError("boom")))
    with pytest.raises(typer.Exit) as generic_exc:
        cli.configure(output_config=tmp_path / "pipeline.yaml", force=True)
    assert generic_exc.value.exit_code == 1


def test_generate_manifest_lists_materials_and_uses_template(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_cli_common(monkeypatch)
    captured: dict[str, object] = {}

    class _GenerateManifestInput:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(manifest, "GenerateManifestInput", _GenerateManifestInput)
    monkeypatch.setattr(
        manifest,
        "run_generate_manifest",
        lambda params: SimpleNamespace(
            success=True,
            materials_count=2,
            material_paths=["/Root/MatA", "/Root/MatB"],
        ),
    )
    monkeypatch.setattr(
        manifest, "prim_path_to_name", lambda prim_path: prim_path.split("/")[-1]
    )

    cli.generate_manifest(
        usd_file=tmp_path / "materials.usd",
        output_dir=tmp_path / "out",
        template=tmp_path / "template.usd",
        list_materials=True,
        verbose=True,
    )

    assert captured["template"] == tmp_path / "template.usd"
    assert captured["list_materials"] is True
    assert captured["verbose"] is True


def test_generate_manifest_summary_and_failures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_cli_common(monkeypatch)
    monkeypatch.setattr(
        manifest,
        "GenerateManifestInput",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )

    monkeypatch.setattr(
        manifest,
        "run_generate_manifest",
        lambda params: SimpleNamespace(
            success=True,
            materials_count=3,
            thumbnails_count=2,
            descriptions_count=1,
            yaml_path=tmp_path / "out" / "materials.yaml",
        ),
    )
    cli.generate_manifest(
        usd_file=tmp_path / "materials.usd",
        output_dir=tmp_path / "out",
    )

    monkeypatch.setattr(
        manifest,
        "run_generate_manifest",
        lambda params: SimpleNamespace(success=False, error="bad manifest"),
    )
    with pytest.raises(typer.Exit) as failed_exc:
        cli.generate_manifest(
            usd_file=tmp_path / "materials.usd",
            output_dir=tmp_path / "out",
        )
    assert failed_exc.value.exit_code == 1

    monkeypatch.setattr(
        manifest,
        "run_generate_manifest",
        Mock(side_effect=RuntimeError("boom")),
    )
    with pytest.raises(typer.Exit) as error_exc:
        cli.generate_manifest(
            usd_file=tmp_path / "materials.usd",
            output_dir=tmp_path / "out",
        )
    assert error_exc.value.exit_code == 1
