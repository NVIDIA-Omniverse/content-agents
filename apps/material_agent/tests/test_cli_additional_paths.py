# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Additional CLI coverage for configure, run, manifest, and helpers."""

from __future__ import annotations

import json
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


def test_load_cli_dotenv_searches_up_from_cwd(monkeypatch: pytest.MonkeyPatch) -> None:
    dotenv_path = "/repo/.env"
    find_dotenv = Mock(return_value=dotenv_path)
    load_dotenv = Mock()
    monkeypatch.setattr(cli, "find_dotenv", find_dotenv)
    monkeypatch.setattr(cli, "load_dotenv", load_dotenv)

    cli._load_cli_dotenv()

    find_dotenv.assert_called_once_with(usecwd=True)
    load_dotenv.assert_called_once_with(dotenv_path=dotenv_path)


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


def test_validate_run_config_model_credentials_reports_missing_nim_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    config = _write_config(
        tmp_path,
        """
project:
  name: demo
steps:
  predict:
    vlm:
      backend: nim
      model: qwen/qwen3.5-397b-a17b
    llm:
      backend: nim
      model: qwen/qwen3.5-397b-a17b
""".strip(),
    )

    with pytest.raises(ValueError) as exc:
        cli._validate_run_config_model_credentials(config, [], [])

    message = str(exc.value)
    assert "steps.predict.vlm.backend='nim' requires NVIDIA_API_KEY" in message
    assert "steps.predict.llm.backend='nim' requires NVIDIA_API_KEY" in message
    assert "unedited run requires NVIDIA_API_KEY" in message
    assert "MA_VLM_BACKEND=openai" in message


def test_validate_run_config_model_credentials_checks_image_gen_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    config = _write_config(
        tmp_path,
        """
project:
  name: demo
steps:
  generate_reference_image:
    enabled: true
    prompt: make a clean material reference
""".strip(),
    )

    with pytest.raises(ValueError) as exc:
        cli._validate_run_config_model_credentials(config, [], [])

    assert (
        "steps.generate_reference_image.image_gen.backend='gemini' "
        "requires GOOGLE_API_KEY or GEMINI_API_KEY"
    ) in str(exc.value)


def test_validate_run_config_model_credentials_accepts_gemini_alias_for_image_gen(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    config = _write_config(
        tmp_path,
        """
project:
  name: demo
steps:
  generate_reference_image:
    enabled: true
    image_gen:
      backend: gemini
      model: gemini-3-pro-image-preview
""".strip(),
    )

    cli._validate_run_config_model_credentials(config, [], [])


def test_validate_run_config_model_credentials_checks_image_gen_backend(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("INFERENCE_NVIDIA_API_KEY", raising=False)
    config = _write_config(
        tmp_path,
        """
project:
  name: demo
steps:
  generate_reference_image:
    enabled: true
    image_gen:
      backend: nvidia_inference
      model: blackwell/flux
""".strip(),
    )

    with pytest.raises(ValueError) as exc:
        cli._validate_run_config_model_credentials(config, [], [])

    assert (
        "steps.generate_reference_image.image_gen.backend='nvidia_inference' "
        "requires INFERENCE_NVIDIA_API_KEY"
    ) in str(exc.value)


def test_validate_run_config_model_credentials_allows_openai_image_gen_no_auth_opt_in(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = _write_config(
        tmp_path,
        """
project:
  name: demo
steps:
  generate_reference_image:
    enabled: true
    image_gen:
      backend: openai
      model: local-image-gen
      base_url: http://image-gen:8000/v1
      api_key: not-used
""".strip(),
    )

    cli._validate_run_config_model_credentials(config, [], [])


def test_validate_run_config_model_credentials_rejects_remote_openai_image_gen_base_url(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = _write_config(
        tmp_path,
        """
project:
  name: demo
steps:
  generate_reference_image:
    enabled: true
    image_gen:
      backend: openai
      model: remote-image-gen
      base_url: https://api.openai-compatible.example/v1
""".strip(),
    )

    with pytest.raises(ValueError) as exc:
        cli._validate_run_config_model_credentials(config, [], [])

    assert (
        "steps.generate_reference_image.image_gen.backend='openai' "
        "requires explicit api_key in config"
    ) in str(exc.value)


def test_validate_run_config_model_credentials_requires_key_for_local_nim_image_gen(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("MA_NIM_API_KEY", raising=False)
    config = _write_config(
        tmp_path,
        """
project:
  name: demo
steps:
  generate_reference_image:
    enabled: true
    image_gen:
      backend: nim
      model: local-image-gen
      base_url: http://image-nim:8000/v1
""".strip(),
    )

    with pytest.raises(ValueError) as exc:
        cli._validate_run_config_model_credentials(config, [], [])

    assert (
        "steps.generate_reference_image.image_gen.backend='nim' "
        "requires MA_NIM_API_KEY or api_key: not-used" in str(exc.value)
    )


def test_validate_run_config_model_credentials_accepts_local_nim_image_gen_env_placeholder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.setenv("MA_NIM_API_KEY", "not-used")
    config = _write_config(
        tmp_path,
        """
project:
  name: demo
steps:
  generate_reference_image:
    enabled: true
    image_gen:
      backend: nim
      model: local-image-gen
      base_url: http://image-nim:8000/v1
""".strip(),
    )

    cli._validate_run_config_model_credentials(config, [], [])


def test_validate_run_config_model_credentials_accepts_local_nim_image_gen_config_placeholder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("MA_NIM_API_KEY", raising=False)
    config = _write_config(
        tmp_path,
        """
project:
  name: demo
steps:
  generate_reference_image:
    enabled: true
    image_gen:
      backend: nim
      model: local-image-gen
      base_url: http://image-nim:8000/v1
      api_key: not-used
""".strip(),
    )

    cli._validate_run_config_model_credentials(config, [], [])


def test_validate_run_config_model_credentials_skips_completed_resume_step(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    working_dir = tmp_path / ".resume-session"
    working_dir.mkdir()
    (working_dir / ".pipeline_state.json").write_text(
        json.dumps({"completed_steps": ["predict"]}),
        encoding="utf-8",
    )
    config = _write_config(
        tmp_path,
        """
project:
  name: demo
  working_dir: .resume-session
steps:
  predict:
    vlm:
      backend: nim
      model: qwen/qwen3.5-397b-a17b
""".strip(),
    )

    cli._validate_run_config_model_credentials(config, [], [], resume=True)


def test_validate_run_config_model_credentials_allows_local_nim_base_url(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.setenv("MA_NIM_API_KEY", "not-used")
    config = _write_config(
        tmp_path,
        """
project:
  name: demo
steps:
  predict:
    vlm:
      backend: nim
      model: local-vlm
      base_url: http://localhost:8000/v1
    llm:
      backend: nim
      model: local-llm
      base_url: http://localhost:8001/v1
""".strip(),
    )

    cli._validate_run_config_model_credentials(config, [], [])


def test_validate_run_config_model_credentials_allows_local_nim_dns_base_url(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.setenv("MA_NIM_API_KEY", "not-used")
    config = _write_config(
        tmp_path,
        """
project:
  name: demo
steps:
  predict:
    vlm:
      backend: nim
      model: local-vlm
      base_url: http://vlm-nim:8000/v1
""".strip(),
    )

    cli._validate_run_config_model_credentials(config, [], [])


def test_validate_run_config_model_credentials_rejects_generic_local_nim_env_placeholder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.setenv("MA_NIM_API_KEY", "YOUR_NIM_API_KEY")
    config = _write_config(
        tmp_path,
        """
project:
  name: demo
steps:
  predict:
    vlm:
      backend: nim
      model: local-vlm
      base_url: http://vlm-nim:8000/v1
""".strip(),
    )

    with pytest.raises(ValueError) as exc:
        cli._validate_run_config_model_credentials(config, [], [])

    assert (
        "steps.predict.vlm.backend='nim' requires MA_NIM_API_KEY or api_key: not-used"
    ) in str(exc.value)


def test_validate_run_config_model_credentials_rejects_hosted_key_for_local_nim(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "hosted-nvidia-key")
    monkeypatch.delenv("MA_NIM_API_KEY", raising=False)
    config = _write_config(
        tmp_path,
        """
project:
  name: demo
steps:
  predict:
    vlm:
      backend: nim
      model: local-vlm
      base_url: http://vlm-nim:8000/v1
""".strip(),
    )

    with pytest.raises(ValueError) as exc:
        cli._validate_run_config_model_credentials(config, [], [])

    message = str(exc.value)
    assert (
        "steps.predict.vlm.backend='nim' requires MA_NIM_API_KEY or api_key: not-used"
    ) in message
    assert "unedited run requires NVIDIA_API_KEY" not in message


def test_validate_run_config_model_credentials_rejects_generic_local_nim_config_placeholder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("MA_NIM_API_KEY", raising=False)
    config = _write_config(
        tmp_path,
        """
project:
  name: demo
steps:
  predict:
    vlm:
      backend: nim
      model: local-vlm
      base_url: http://vlm-nim:8000/v1
      api_key: YOUR_NVIDIA_API_KEY
""".strip(),
    )

    with pytest.raises(ValueError) as exc:
        cli._validate_run_config_model_credentials(config, [], [])

    assert (
        "steps.predict.vlm.backend='nim' requires MA_NIM_API_KEY or api_key: not-used"
    ) in str(exc.value)


def test_validate_run_config_model_credentials_honors_local_nim_env_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("MA_NIM_API_KEY", "not-used")
    monkeypatch.setenv("MA_VLM_NIM_BASE_URL", "http://vlm-nim:8000/v1")
    config = _write_config(
        tmp_path,
        """
project:
  name: demo
steps:
  predict:
    vlm:
      backend: openai
      model: gpt-4o
""".strip(),
    )

    cli._validate_run_config_model_credentials(config, [], [])


def test_validate_run_config_model_credentials_drops_stale_key_for_nim_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("MA_NIM_API_KEY", raising=False)
    monkeypatch.setenv("MA_VLM_NIM_BASE_URL", "http://vlm-nim:8000/v1")
    config = _write_config(
        tmp_path,
        """
project:
  name: demo
steps:
  predict:
    vlm:
      backend: openai
      model: gpt-4o
      api_key: hosted-openai-key
""".strip(),
    )

    with pytest.raises(ValueError) as exc:
        cli._validate_run_config_model_credentials(config, [], [])

    assert (
        "steps.predict.vlm.backend='nim' requires MA_NIM_API_KEY or api_key: not-used"
    ) in str(exc.value)


def test_validate_run_config_model_credentials_drops_existing_nim_key_for_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("MA_NIM_API_KEY", raising=False)
    monkeypatch.setenv("MA_VLM_NIM_BASE_URL", "http://vlm-nim:8000/v1")
    config = _write_config(
        tmp_path,
        """
project:
  name: demo
steps:
  predict:
    vlm:
      backend: nim
      model: hosted-nim-model
      base_url: https://integrate.api.nvidia.com/v1
      api_key: hosted-nim-key
""".strip(),
    )

    with pytest.raises(ValueError) as exc:
        cli._validate_run_config_model_credentials(config, [], [])

    assert (
        "steps.predict.vlm.backend='nim' requires MA_NIM_API_KEY or api_key: not-used"
    ) in str(exc.value)


def test_validate_run_config_model_credentials_accepts_local_openai_placeholder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = _write_config(
        tmp_path,
        """
project:
  name: demo
steps:
  predict:
    vlm:
      backend: openai
      model: qwen/qwen3.5-35b-a3b
      base_url: http://192.168.4.58:8001/v1
      api_key: not-used
    llm:
      backend: openai
      model: qwen/qwen3.5-35b-a3b
      base_url: http://192.168.4.58:8001/v1
      api_key: not-used
""".strip(),
    )

    cli._validate_run_config_model_credentials(config, [], [])


def test_validate_run_config_model_credentials_rejects_implicit_local_openai_dummy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = _write_config(
        tmp_path,
        """
project:
  name: demo
steps:
  predict:
    vlm:
      backend: openai
      model: qwen/qwen3.5-35b-a3b
      base_url: http://192.168.4.58:8001/v1
""".strip(),
    )

    with pytest.raises(ValueError) as exc:
        cli._validate_run_config_model_credentials(config, [], [])

    assert (
        "steps.predict.vlm.backend='openai' requires OPENAI_API_KEY "
        "or api_key: not-used"
    ) in str(exc.value)


def test_validate_run_config_model_credentials_rejects_env_key_for_local_openai(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Hosted ``OPENAI_API_KEY`` must not silently flow to a local OpenAI-
    compatible endpoint via env. Preflight requires an explicit endpoint-
    scoped ``api_key`` (or the ``not-used`` placeholder) for local URLs."""
    monkeypatch.setenv("OPENAI_API_KEY", "local-openai-key")
    config = _write_config(
        tmp_path,
        """
project:
  name: demo
steps:
  predict:
    vlm:
      backend: openai
      model: qwen/qwen3.5-35b-a3b
      base_url: http://192.168.4.58:8001/v1
""".strip(),
    )

    with pytest.raises(ValueError) as exc:
        cli._validate_run_config_model_credentials(config, [], [])

    assert "steps.predict.vlm.backend='openai'" in str(exc.value)


def test_validate_run_config_model_credentials_rejects_remote_openai_placeholder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = _write_config(
        tmp_path,
        """
project:
  name: demo
steps:
  predict:
    vlm:
      backend: openai
      model: gpt-4o
      base_url: https://api.openai-compatible.example/v1
      api_key: not-used
""".strip(),
    )

    with pytest.raises(ValueError) as exc:
        cli._validate_run_config_model_credentials(config, [], [])

    assert (
        "steps.predict.vlm.backend='openai' requires explicit api_key in config"
    ) in str(exc.value)


def test_validate_run_config_model_credentials_rejects_generic_local_openai_placeholder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = _write_config(
        tmp_path,
        """
project:
  name: demo
steps:
  predict:
    vlm:
      backend: openai
      model: qwen/qwen3.5-35b-a3b
      base_url: http://192.168.4.58:8001/v1
      api_key: YOUR_OPENAI_API_KEY
""".strip(),
    )

    with pytest.raises(ValueError) as exc:
        cli._validate_run_config_model_credentials(config, [], [])

    assert "steps.predict.vlm.backend='openai' requires OPENAI_API_KEY" in str(
        exc.value
    )


def test_validate_run_config_model_credentials_honors_llm_nim_env_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("MA_NIM_API_KEY", "not-used")
    monkeypatch.setenv("MA_LLM_NIM_BASE_URL", "http://llm-nim:8000/v1")
    config = _write_config(
        tmp_path,
        """
project:
  name: demo
steps:
  predict:
    vlm:
      backend: openai
      model: gpt-4o
    llm:
      backend: openai
      model: gpt-4o
""".strip(),
    )

    with pytest.raises(ValueError) as exc:
        cli._validate_run_config_model_credentials(config, [], [])

    assert "steps.predict.vlm.backend='openai' requires OPENAI_API_KEY" in str(
        exc.value
    )
    assert "steps.predict.llm.backend='openai'" not in str(exc.value)


def test_validate_run_config_model_credentials_rejects_placeholder_for_hosted_nim(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.setenv("MA_NIM_API_KEY", "not-used")
    config = _write_config(
        tmp_path,
        """
project:
  name: demo
steps:
  predict:
    vlm:
      backend: nim
      model: hosted-vlm
""".strip(),
    )

    with pytest.raises(ValueError) as exc:
        cli._validate_run_config_model_credentials(config, [], [])

    assert "steps.predict.vlm.backend='nim' requires NVIDIA_API_KEY" in str(exc.value)


def test_validate_run_config_model_credentials_rejects_placeholder_env_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "YOUR_NVIDIA_API_KEY")
    monkeypatch.delenv("MA_NIM_API_KEY", raising=False)
    config = _write_config(
        tmp_path,
        """
project:
  name: demo
steps:
  predict:
    vlm:
      backend: nim
      model: hosted-vlm
""".strip(),
    )

    with pytest.raises(ValueError) as exc:
        cli._validate_run_config_model_credentials(config, [], [])

    assert "steps.predict.vlm.backend='nim' requires NVIDIA_API_KEY" in str(exc.value)


def test_validate_run_config_model_credentials_rejects_placeholder_config_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("MA_NIM_API_KEY", raising=False)
    config = _write_config(
        tmp_path,
        """
project:
  name: demo
steps:
  predict:
    vlm:
      backend: nim
      model: hosted-vlm
      api_key: not-used
""".strip(),
    )

    with pytest.raises(ValueError) as exc:
        cli._validate_run_config_model_credentials(config, [], [])

    assert "steps.predict.vlm.backend='nim' requires NVIDIA_API_KEY" in str(exc.value)


def test_validate_run_config_model_credentials_applies_llm_nim_override_outside_predict(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``MA_LLM_NIM_BASE_URL`` / ``MA_VLM_NIM_BASE_URL`` are applied at runtime
    by ``create_chat_model_from_config`` for every LLM call, not just predict.
    Preflight must mirror that scope: an evaluate-step ``llm_judge: openai``
    config should be re-routed to NIM under the env override and pass when
    ``MA_NIM_API_KEY`` is provided, regardless of ``OPENAI_API_KEY``.
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.setenv("MA_VLM_NIM_BASE_URL", "http://vlm-nim:8000/v1")
    monkeypatch.setenv("MA_NIM_API_KEY", "not-used")

    config = _write_config(
        tmp_path,
        """
project:
  name: demo
steps:
  evaluate:
    enabled: true
    llm_judge:
      backend: openai
      model: gpt-4o
""".strip(),
    )

    # Should not raise: preflight follows runtime, which routes this through
    # the local NIM sidecar.
    cli._validate_run_config_model_credentials(config, [], [])


def test_validate_run_config_model_credentials_rejects_openai_env_redirect_with_hosted_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``OPENAI_BASE_URL`` redirects the OpenAI SDK to a custom endpoint;
    runtime now rejects a hosted ``OPENAI_API_KEY`` against that URL.
    Preflight must reject the same combination so configs cannot pass
    validation only to fail later during model construction."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real-openai-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai-compatible.example/v1")
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    config = _write_config(
        tmp_path,
        """
project:
  name: demo
steps:
  predict:
    vlm:
      backend: openai
      model: gpt-4o
""".strip(),
    )

    with pytest.raises(ValueError) as exc:
        cli._validate_run_config_model_credentials(config, [], [])

    assert "steps.predict.vlm.backend='openai'" in str(exc.value)


def test_validate_run_config_model_credentials_rejects_openai_explicit_key_without_paired_base_url(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An explicit YAML ``api_key`` without an explicit YAML ``base_url`` must
    not bypass the env-redirect check — runtime would reject the same shape."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai-compatible.example/v1")
    config = _write_config(
        tmp_path,
        """
project:
  name: demo
steps:
  predict:
    vlm:
      backend: openai
      model: gpt-4o
      api_key: sk-explicit-yaml-key
""".strip(),
    )

    with pytest.raises(ValueError) as exc:
        cli._validate_run_config_model_credentials(config, [], [])

    assert "steps.predict.vlm.backend='openai'" in str(exc.value)


def test_validate_run_config_model_credentials_does_not_pierce_mock_llm_under_nim_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``MA_LLM_NIM_BASE_URL`` rerouting is for real backends; a mock-backed
    LLM section must remain mock at preflight, matching runtime behavior.
    Otherwise a simulate run with the env var set would be rejected for
    missing NIM credentials."""
    monkeypatch.setenv("MA_LLM_NIM_BASE_URL", "http://llm-nim:8000/v1")
    monkeypatch.delenv("MA_NIM_API_KEY", raising=False)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    config = _write_config(
        tmp_path,
        """
project:
  name: demo
steps:
  predict:
    vlm:
      backend: mock
      api_key: not-used
    llm:
      backend: mock
      api_key: not-used
""".strip(),
    )

    # Should not raise: mock backends are opt-out from any external call.
    cli._validate_run_config_model_credentials(config, [], [])


def test_validate_run_config_windows_prerequisites_rejects_local_optimizer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(cli.sys, "platform", "win32")
    monkeypatch.delenv("NVCF_OPTIMIZER_FUNCTION_ID", raising=False)
    monkeypatch.delenv("OPTIMIZER_ENDPOINT", raising=False)
    monkeypatch.setenv("WU_SO_PACKAGE_DIR", str(tmp_path / "missing_so"))
    monkeypatch.setattr(cli.shutil, "which", lambda name: None)
    config = _write_config(
        tmp_path,
        """
project:
  name: demo
steps:
  optimize_usd:
    enabled: true
""".strip(),
    )

    with pytest.raises(ValueError) as exc:
        cli._validate_run_config_windows_prerequisites(config, [], [])

    message = str(exc.value)
    assert "native Windows" in message
    assert "WSL launcher" in message
    assert "`bash` was not found" in message
    assert "Scene Optimizer Core package" in message
    assert "--skip optimize_usd" in message


def test_validate_run_config_windows_prerequisites_respects_skip_and_remote(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(cli.sys, "platform", "win32")
    monkeypatch.delenv("NVCF_OPTIMIZER_FUNCTION_ID", raising=False)
    monkeypatch.delenv("OPTIMIZER_ENDPOINT", raising=False)
    monkeypatch.setenv("WU_SO_PACKAGE_DIR", str(tmp_path / "missing_so"))
    monkeypatch.setattr(cli.shutil, "which", lambda name: None)
    local_config = _write_config(
        tmp_path,
        """
project:
  name: demo
steps:
  optimize_usd:
    enabled: true
""".strip(),
    )

    cli._validate_run_config_windows_prerequisites(local_config, ["optimize_usd"], [])

    remote_config = tmp_path / "remote.yaml"
    remote_config.write_text(
        """
project:
  name: demo
steps:
  optimize_usd:
    enabled: true
    optimization_config:
      backend: remote
""".strip(),
        encoding="utf-8",
    )

    cli._validate_run_config_windows_prerequisites(remote_config, [], [])


def test_validate_run_config_model_credentials_keeps_vlm_nim_override_predict_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Unlike the LLM override, the VLM override only fires inside ``predict``
    at runtime (PredictConfigTask). Preflight must keep that scope so a
    benchmark step's ``vlm_judge`` is still validated against its declared
    backend's credentials."""
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-real")
    monkeypatch.setenv("MA_VLM_NIM_BASE_URL", "http://vlm-nim:8000/v1")
    monkeypatch.setenv("MA_NIM_API_KEY", "not-used")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    config = _write_config(
        tmp_path,
        """
project:
  name: demo
steps:
  benchmark:
    enabled: true
    vlm_judge:
      backend: gemini
      model: gemini-2.0-flash
""".strip(),
    )

    with pytest.raises(ValueError) as exc:
        cli._validate_run_config_model_credentials(config, [], [])

    # VLM override does not fire outside predict, so the benchmark vlm_judge
    # remains gemini and must require GOOGLE_API_KEY/GEMINI_API_KEY.
    assert "steps.benchmark.vlm_judge.backend='gemini'" in str(exc.value)


def test_validate_run_config_model_credentials_ignores_ma_nim_for_embeddings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.setenv("MA_NIM_API_KEY", "nim-key")
    config = _write_config(
        tmp_path,
        """
project:
  name: demo
steps:
  cluster_prims:
    embedding_service: nim
""".strip(),
    )

    with pytest.raises(ValueError) as exc:
        cli._validate_run_config_model_credentials(config, [], [])

    assert "steps.cluster_prims.embedding_service='nim' requires NVIDIA_API_KEY" in str(
        exc.value
    )


def test_validate_run_config_model_credentials_requires_key_for_remote_nim_base_url(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("MA_NIM_API_KEY", raising=False)
    config = _write_config(
        tmp_path,
        """
project:
  name: demo
steps:
  predict:
    vlm:
      backend: nim
      model: hosted-vlm
      base_url: https://inference-api.nvidia.com/v1
""".strip(),
    )

    with pytest.raises(ValueError) as exc:
        cli._validate_run_config_model_credentials(config, [], [])

    assert "steps.predict.vlm.backend='nim' requires NVIDIA_API_KEY" in str(exc.value)


def test_validate_run_config_model_credentials_rejects_env_key_for_custom_nim_base_url(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "hosted-nvidia-key")
    monkeypatch.delenv("MA_NIM_API_KEY", raising=False)
    config = _write_config(
        tmp_path,
        """
project:
  name: demo
steps:
  predict:
    vlm:
      backend: nim
      model: hosted-vlm
      base_url: https://nim.example.com/v1
""".strip(),
    )

    with pytest.raises(ValueError) as exc:
        cli._validate_run_config_model_credentials(config, [], [])

    assert "steps.predict.vlm.backend='nim' requires explicit api_key in config" in str(
        exc.value
    )


def test_validate_run_config_model_credentials_accepts_explicit_key_for_custom_nim_url(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "hosted-nvidia-key")
    config = _write_config(
        tmp_path,
        """
project:
  name: demo
steps:
  predict:
    vlm:
      backend: nim
      model: hosted-vlm
      base_url: https://nim.example.com/v1
      api_key: endpoint-nim-key
""".strip(),
    )

    cli._validate_run_config_model_credentials(config, [], [])


def test_validate_run_config_model_credentials_rejects_ma_nim_key_for_hosted_base_url(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.setenv("MA_NIM_API_KEY", "nim-key")
    config = _write_config(
        tmp_path,
        """
project:
  name: demo
steps:
  predict:
    vlm:
      backend: nim
      model: hosted-vlm
      base_url: https://inference-api.nvidia.com/v1
""".strip(),
    )

    with pytest.raises(ValueError) as exc:
        cli._validate_run_config_model_credentials(config, [], [])

    assert "steps.predict.vlm.backend='nim' requires NVIDIA_API_KEY" in str(exc.value)


def test_validate_run_config_model_credentials_checks_effective_step_defaults(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    config = _write_config(
        tmp_path,
        """
project:
  name: demo
steps:
  predict:
    enabled: true
""".strip(),
    )

    with pytest.raises(ValueError) as exc:
        cli._validate_run_config_model_credentials(config, [], [])

    assert "steps.predict.vlm.backend='nim' requires NVIDIA_API_KEY" in str(exc.value)


def test_validate_run_config_model_credentials_checks_cluster_embedding_service(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    config = _write_config(
        tmp_path,
        """
project:
  name: demo
steps:
  cluster_prims:
    embedding_service: nim
""".strip(),
    )

    with pytest.raises(ValueError) as exc:
        cli._validate_run_config_model_credentials(config, [], [])

    assert "steps.cluster_prims.embedding_service='nim' requires NVIDIA_API_KEY" in str(
        exc.value
    )


def test_validate_run_config_model_credentials_checks_pdf_embedding_defaults(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    config = _write_config(
        tmp_path,
        """
project:
  name: demo
steps:
  build_dataset_pdf_vectorstore:
    enabled: true
    source: docs
""".strip(),
    )

    with pytest.raises(ValueError) as exc:
        cli._validate_run_config_model_credentials(config, [], [])

    assert (
        "steps.build_dataset_pdf_vectorstore.embedding.service='nim' "
        "requires NVIDIA_API_KEY"
    ) in str(exc.value)


def test_validate_run_config_model_credentials_honors_step_selection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    config = _write_config(
        tmp_path,
        """
project:
  name: demo
steps:
  build_dataset_usd: {}
  predict:
    vlm:
      backend: nim
      model: qwen/qwen3.5-397b-a17b
""".strip(),
    )

    cli._validate_run_config_model_credentials(config, ["predict"], [])
    cli._validate_run_config_model_credentials(config, [], ["build_dataset_usd"])


def test_run_fails_fast_on_missing_model_credentials(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    logger, printed = _patch_cli_common(monkeypatch)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(
        cli,
        "get_listener",
        lambda *args, **kwargs: SimpleNamespace(event=lambda *a, **k: None),
    )
    config = _write_config(
        tmp_path,
        """
project:
  name: demo
steps:
  predict:
    vlm:
      backend: nim
      model: qwen/qwen3.5-397b-a17b
""".strip(),
    )
    monkeypatch.setattr(api, "run_pipeline", Mock())

    with pytest.raises(typer.Exit) as exc:
        cli.run(config=config)

    assert exc.value.exit_code == 1
    api.run_pipeline.assert_not_called()
    logger.error.assert_called()
    assert any("NVIDIA_API_KEY" in line for line in printed)


def test_run_rejects_windows_local_scene_optimizer_before_execution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    logger, printed = _patch_cli_common(monkeypatch)
    monkeypatch.setattr(cli.sys, "platform", "win32")
    monkeypatch.delenv("NVCF_OPTIMIZER_FUNCTION_ID", raising=False)
    monkeypatch.delenv("OPTIMIZER_ENDPOINT", raising=False)
    monkeypatch.setenv("WU_SO_PACKAGE_DIR", str(tmp_path / "missing_so"))
    monkeypatch.setattr(cli.shutil, "which", lambda name: None)
    monkeypatch.setattr(
        cli,
        "get_listener",
        lambda *args, **kwargs: SimpleNamespace(event=lambda *a, **k: None),
    )
    config = _write_config(
        tmp_path,
        """
project:
  name: demo
steps:
  optimize_usd:
    enabled: true
""".strip(),
    )
    monkeypatch.setattr(api, "run_pipeline", Mock())

    with pytest.raises(typer.Exit) as exc:
        cli.run(config=config)

    assert exc.value.exit_code == 1
    api.run_pipeline.assert_not_called()
    logger.error.assert_called()
    assert any("WSL/Linux" in line for line in printed)


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


def test_maybe_apply_backend_env_overrides_drops_stale_endpoint_fields(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``MA_VLM_BACKEND=openai`` against a config with ``backend: nim`` and an
    NVIDIA endpoint key must clear the stale ``api_key``/``base_url`` so the
    NIM credential cannot be forwarded to the new OpenAI endpoint."""
    import yaml as _yaml

    config = _write_config(
        tmp_path,
        _yaml.safe_dump(
            {
                "steps": {
                    "predict": {
                        "vlm": {
                            "backend": "nim",
                            "model": "nvidia/cosmos-reason2-8b",
                            "api_key": "nvidia-real-key",
                            "base_url": "https://integrate.api.nvidia.com/v1",
                        },
                        "llm": {
                            "backend": "nim",
                            "model": "nvidia/cosmos-reason2-8b",
                            "api_key": "nvidia-real-key",
                            "base_url": "https://integrate.api.nvidia.com/v1",
                        },
                    }
                }
            },
            sort_keys=False,
        ),
    )

    monkeypatch.setenv("MA_VLM_BACKEND", "openai")
    monkeypatch.setenv("MA_VLM_MODEL", "gpt-4o")
    monkeypatch.delenv("MA_LLM_BACKEND", raising=False)
    monkeypatch.delenv("MA_LLM_MODEL", raising=False)

    overridden = cli._maybe_apply_backend_env_overrides(config)
    assert overridden != config

    rewritten = _yaml.safe_load(overridden.read_text(encoding="utf-8"))
    vlm = rewritten["steps"]["predict"]["vlm"]
    assert vlm["backend"] == "openai"
    assert vlm["model"] == "gpt-4o"
    assert "api_key" not in vlm, "stale NVIDIA api_key leaked to new OpenAI backend"
    assert "base_url" not in vlm, "stale NIM base_url leaked to new backend"

    # LLM section was not overridden; stale fields stay (no backend change).
    llm = rewritten["steps"]["predict"]["llm"]
    assert llm["backend"] == "nim"
    assert llm["api_key"] == "nvidia-real-key"


def test_maybe_apply_backend_env_overrides_no_change_keeps_endpoint_fields(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``MA_VLM_MODEL`` alone must not clear endpoint fields; only a backend
    change drops them."""
    import yaml as _yaml

    config = _write_config(
        tmp_path,
        _yaml.safe_dump(
            {
                "steps": {
                    "predict": {
                        "vlm": {
                            "backend": "nim",
                            "model": "nvidia/cosmos-reason2-8b",
                            "api_key": "nvidia-key",
                            "base_url": "http://vlm-nim:8000/v1",
                        }
                    }
                }
            },
            sort_keys=False,
        ),
    )

    monkeypatch.delenv("MA_VLM_BACKEND", raising=False)
    monkeypatch.setenv("MA_VLM_MODEL", "nvidia/other-model")
    monkeypatch.delenv("MA_LLM_BACKEND", raising=False)
    monkeypatch.delenv("MA_LLM_MODEL", raising=False)

    overridden = cli._maybe_apply_backend_env_overrides(config)
    rewritten = _yaml.safe_load(overridden.read_text(encoding="utf-8"))
    vlm = rewritten["steps"]["predict"]["vlm"]
    assert vlm["backend"] == "nim"
    assert vlm["model"] == "nvidia/other-model"
    assert vlm["api_key"] == "nvidia-key"
    assert vlm["base_url"] == "http://vlm-nim:8000/v1"
