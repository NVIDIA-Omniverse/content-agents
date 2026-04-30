"""Tests for shared agentic configuration utilities."""

from __future__ import annotations

import pytest

from world_understanding.agentic.config import (
    LOCAL_NIM_API_KEY_PLACEHOLDER,
    get_api_key_for_backend,
    get_api_key_for_model_config,
    is_local_base_url,
    is_placeholder_api_key,
)
from world_understanding.agentic.config.utils import API_KEY_ENV_VAR_MAP


def test_get_api_key_for_backend_resolves_public_provider_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "google-key")

    assert get_api_key_for_backend("anthropic", "VLM") == "anthropic-key"
    assert get_api_key_for_backend("gemini", "VLM") == "google-key"


def test_get_api_key_for_backend_accepts_gemini_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")

    assert get_api_key_for_backend("gemini", "VLM") == "gemini-key"


def test_api_key_env_var_map_is_shared() -> None:
    assert API_KEY_ENV_VAR_MAP["nim"] == ("NVIDIA_API_KEY",)
    assert API_KEY_ENV_VAR_MAP["gemini"] == ("GOOGLE_API_KEY", "GEMINI_API_KEY")
    assert API_KEY_ENV_VAR_MAP["perflab"] == ("NSTORAGE_API_KEY",)


def test_get_api_key_for_model_config_requires_local_nim_placeholder_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("MA_NIM_API_KEY", raising=False)

    assert is_local_base_url("http://localhost:8000/v1")
    assert is_local_base_url("http://vlm-nim:8000/v1")
    assert is_local_base_url("http://192.168.4.58:8001/v1")
    assert is_local_base_url("http://nim.default.svc.cluster.local/v1")
    assert not is_local_base_url("https://inference-api.nvidia.com/v1")

    with pytest.raises(ValueError, match="NVIDIA_API_KEY"):
        get_api_key_for_model_config(
            "nim", {"base_url": "http://localhost:8000/v1"}, "VLM"
        )

    monkeypatch.setenv("MA_NIM_API_KEY", LOCAL_NIM_API_KEY_PLACEHOLDER)
    assert (
        get_api_key_for_model_config(
            "nim", {"base_url": "http://localhost:8000/v1"}, "VLM"
        )
        == LOCAL_NIM_API_KEY_PLACEHOLDER
    )


def test_get_api_key_for_model_config_ignores_global_nvidia_key_for_local_nim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "hosted-nvidia-key")
    monkeypatch.delenv("MA_NIM_API_KEY", raising=False)

    with pytest.raises(ValueError, match="NVIDIA_API_KEY"):
        get_api_key_for_model_config(
            "nim", {"base_url": "http://vlm-nim:8000/v1"}, "VLM"
        )


def test_get_api_key_for_model_config_accepts_ma_nim_key_for_local_nim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "hosted-nvidia-key")
    monkeypatch.setenv("MA_NIM_API_KEY", "local-nim-key")

    assert (
        get_api_key_for_model_config(
            "nim", {"base_url": "http://vlm-nim:8000/v1"}, "VLM"
        )
        == "local-nim-key"
    )


def test_is_local_base_url_edge_cases(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.setenv("MA_NIM_API_KEY", LOCAL_NIM_API_KEY_PLACEHOLDER)

    assert is_local_base_url("http://[::1]:8000/v1")
    assert not is_local_base_url("localhost:8000")
    assert not is_local_base_url("api.example.com:443/v1")
    assert not is_local_base_url("")
    assert not is_local_base_url(None)
    assert (
        get_api_key_for_model_config(
            "nim",
            {"base_url": "http://[::1]:8000/v1"},
            "VLM",
        )
        == LOCAL_NIM_API_KEY_PLACEHOLDER
    )


def test_get_api_key_for_model_config_accepts_local_placeholder_config_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("MA_NIM_API_KEY", raising=False)

    assert (
        get_api_key_for_model_config(
            "nim",
            {
                "base_url": "http://vlm-nim:8000/v1",
                "api_key": LOCAL_NIM_API_KEY_PLACEHOLDER,
            },
            "VLM",
        )
        == LOCAL_NIM_API_KEY_PLACEHOLDER
    )


def test_get_api_key_for_model_config_rejects_generic_local_placeholder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("MA_NIM_API_KEY", raising=False)

    with pytest.raises(ValueError, match="NVIDIA_API_KEY"):
        get_api_key_for_model_config(
            "nim",
            {
                "base_url": "http://vlm-nim:8000/v1",
                "api_key": "YOUR_NVIDIA_API_KEY",
            },
            "VLM",
        )


def test_get_api_key_for_model_config_rejects_private_url_without_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("MA_NIM_API_KEY", raising=False)

    with pytest.raises(ValueError, match="NVIDIA_API_KEY"):
        get_api_key_for_model_config(
            "nim", {"base_url": "http://192.168.4.58:8001/v1"}, "VLM"
        )


def test_get_api_key_for_model_config_uses_nvidia_key_for_hosted_nim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MA_NIM_API_KEY", "ma-nim-key")
    monkeypatch.setenv("NVIDIA_API_KEY", "nvidia-key")

    assert (
        get_api_key_for_model_config(
            "nim", {"base_url": "https://inference-api.nvidia.com/v1"}, "VLM"
        )
        == "nvidia-key"
    )


def test_get_api_key_for_model_config_uses_nvidia_key_without_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "nvidia-key")
    monkeypatch.delenv("MA_NIM_API_KEY", raising=False)

    assert get_api_key_for_model_config("nim", {}, "VLM") == "nvidia-key"


def test_get_api_key_for_model_config_rejects_nvidia_key_for_custom_nim_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "hosted-nvidia-key")
    monkeypatch.delenv("MA_NIM_API_KEY", raising=False)

    with pytest.raises(ValueError, match="NVIDIA_API_KEY"):
        get_api_key_for_model_config(
            "nim", {"base_url": "https://nim.example.com/v1"}, "VLM"
        )


def test_get_api_key_for_model_config_accepts_explicit_key_for_custom_nim_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "hosted-nvidia-key")
    monkeypatch.delenv("MA_NIM_API_KEY", raising=False)

    assert (
        get_api_key_for_model_config(
            "nim",
            {
                "base_url": "https://nim.example.com/v1",
                "api_key": "endpoint-nim-key",
            },
            "VLM",
        )
        == "endpoint-nim-key"
    )


def test_get_api_key_for_model_config_rejects_ma_nim_key_for_hosted_nim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.setenv("MA_NIM_API_KEY", "local-nim-key")

    with pytest.raises(ValueError, match="NVIDIA_API_KEY"):
        get_api_key_for_model_config(
            "nim", {"base_url": "https://inference-api.nvidia.com/v1"}, "VLM"
        )


def test_get_api_key_for_model_config_uses_ma_nim_key_for_custom_remote_nim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The helm chart wires ``vlmNim.endpointOverride`` with ``MA_NIM_API_KEY``;
    a non-NVIDIA, non-local NIM URL must accept that explicit NIM-scoped key."""
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.setenv("MA_NIM_API_KEY", "external-nim-key")

    assert (
        get_api_key_for_model_config(
            "nim", {"base_url": "https://external-vlm.example.com/v1"}, "VLM"
        )
        == "external-nim-key"
    )


def test_get_api_key_for_model_config_accepts_no_auth_placeholder_for_custom_remote_nim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``MA_NIM_API_KEY=not-used`` must opt a custom remote NIM into no-auth,
    matching the helm chart's documented endpointOverride wiring."""
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.setenv("MA_NIM_API_KEY", LOCAL_NIM_API_KEY_PLACEHOLDER)

    assert (
        get_api_key_for_model_config(
            "nim", {"base_url": "https://external-vlm.example.com/v1"}, "VLM"
        )
        == LOCAL_NIM_API_KEY_PLACEHOLDER
    )


def test_get_api_key_for_model_config_rejects_placeholder_for_hosted_nim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.setenv("MA_NIM_API_KEY", LOCAL_NIM_API_KEY_PLACEHOLDER)

    with pytest.raises(ValueError, match="NVIDIA_API_KEY"):
        get_api_key_for_model_config("nim", {}, "VLM")


def test_get_api_key_for_model_config_accepts_local_openai_placeholder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    assert (
        get_api_key_for_model_config(
            "openai",
            {
                "base_url": "http://192.168.4.58:8001/v1",
                "api_key": LOCAL_NIM_API_KEY_PLACEHOLDER,
            },
            "VLM",
        )
        == LOCAL_NIM_API_KEY_PLACEHOLDER
    )


def test_get_api_key_for_model_config_rejects_env_key_for_localhost_openai(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same protection for ``localhost`` — local OpenAI-compatible servers are
    a separate trust boundary from the OpenAI provider."""
    monkeypatch.setenv("OPENAI_API_KEY", "real-hosted-openai-key")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        get_api_key_for_model_config(
            "openai",
            {"base_url": "http://localhost:8000/v1"},
            "VLM",
        )


def test_get_api_key_for_model_config_rejects_env_key_for_local_openai(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hosted ``OPENAI_API_KEY`` must not silently flow to a local
    OpenAI-compatible endpoint, even when the user paired ``base_url`` in
    config. The trust boundary for the hosted key is the OpenAI provider
    URL set; local servers must opt in via an endpoint-scoped ``api_key``
    or the explicit ``not-used`` no-auth placeholder."""
    monkeypatch.setenv("OPENAI_API_KEY", "real-hosted-openai-key")

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        get_api_key_for_model_config(
            "openai",
            {"base_url": "http://192.168.4.58:8001/v1"},
            "VLM",
        )


def test_get_api_key_for_model_config_rejects_local_openai_without_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        get_api_key_for_model_config(
            "openai",
            {"base_url": "http://192.168.4.58:8001/v1"},
            "VLM",
        )


def test_get_api_key_for_model_config_rejects_hosted_openai_placeholder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        get_api_key_for_model_config(
            "openai",
            {
                "base_url": "https://api.openai-compatible.example/v1",
                "api_key": LOCAL_NIM_API_KEY_PLACEHOLDER,
            },
            "VLM",
        )


def test_get_api_key_for_model_config_uses_openai_key_without_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")

    assert get_api_key_for_model_config("openai", {}, "VLM") == "openai-key"


def test_get_api_key_for_model_config_uses_openai_key_for_hosted_openai_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")

    assert (
        get_api_key_for_model_config(
            "openai", {"base_url": "https://api.openai.com/v1"}, "VLM"
        )
        == "openai-key"
    )


def test_get_api_key_for_model_config_rejects_openai_key_for_custom_openai_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "hosted-openai-key")

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        get_api_key_for_model_config(
            "openai",
            {"base_url": "https://api.openai-compatible.example/v1"},
            "VLM",
        )


def test_get_api_key_for_model_config_rejects_openai_key_for_env_redirected_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``OPENAI_BASE_URL`` / ``OPENAI_API_BASE`` redirect the OpenAI SDK to a
    custom endpoint even when ``base_url`` is not in config. The hosted
    ``OPENAI_API_KEY`` must not silently follow the redirect."""
    monkeypatch.setenv("OPENAI_API_KEY", "hosted-openai-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai-compatible.example/v1")

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        get_api_key_for_model_config("openai", {}, "VLM")


def test_get_api_key_for_model_config_rejects_openai_key_for_env_legacy_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Legacy ``OPENAI_API_BASE`` env var must trigger the same protection."""
    monkeypatch.setenv("OPENAI_API_KEY", "hosted-openai-key")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_API_BASE", "https://api.openai-compatible.example/v1")

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        get_api_key_for_model_config("openai", {}, "VLM")


def test_get_api_key_for_model_config_rejects_explicit_key_when_env_redirects_to_custom(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit ``api_key`` in config must not be honored when the config
    has no ``base_url`` and ``OPENAI_BASE_URL`` redirects the SDK to a
    non-provider endpoint. Otherwise the explicit key follows the env
    redirect to an arbitrary OpenAI-compatible URL the user did not opt into."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai-compatible.example/v1")

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        get_api_key_for_model_config(
            "openai",
            {"api_key": "sk-explicit-config-key"},
            "VLM",
        )


def test_get_api_key_for_model_config_rejects_openai_key_for_env_local_redirect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A malicious or unintended ``OPENAI_BASE_URL`` set to a "local" host
    (single-label, ``.local``, private IP, ...) must not pull the hosted
    ``OPENAI_API_KEY`` through to that host. Local endpoints require an
    explicit config-supplied ``base_url`` pairing — otherwise env-only
    redirects become a key-exfiltration channel."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real-openai-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://attacker.local/v1")

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        get_api_key_for_model_config("openai", {}, "VLM")


def test_get_api_key_for_model_config_accepts_paired_explicit_key_for_local_openai(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A local OpenAI-compatible server is reached by explicitly pairing an
    endpoint key with the local URL. The hosted env ``OPENAI_API_KEY`` is
    *not* silently reused — local URLs are non-provider trust boundaries."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real-openai-key")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    assert (
        get_api_key_for_model_config(
            "openai",
            {
                "base_url": "http://localhost:8000/v1",
                "api_key": "not-used",
            },
            "VLM",
        )
        == "not-used"
    )


def test_get_api_key_for_model_config_keeps_explicit_key_when_config_supplies_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the config explicitly pairs an ``api_key`` with a custom
    ``base_url``, trust the user's pairing even if ``OPENAI_BASE_URL`` would
    point elsewhere — the config-supplied URL takes precedence."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://other.example/v1")

    assert (
        get_api_key_for_model_config(
            "openai",
            {
                "base_url": "https://api.openai-compatible.example/v1",
                "api_key": "sk-explicit-paired-key",
            },
            "VLM",
        )
        == "sk-explicit-paired-key"
    )


def test_get_api_key_for_model_config_accepts_explicit_key_for_custom_openai_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "hosted-openai-key")

    assert (
        get_api_key_for_model_config(
            "openai",
            {
                "base_url": "https://api.openai-compatible.example/v1",
                "api_key": "endpoint-openai-key",
            },
            "VLM",
        )
        == "endpoint-openai-key"
    )


def test_get_api_key_for_model_config_rejects_generic_local_openai_placeholder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        get_api_key_for_model_config(
            "openai",
            {
                "base_url": "http://192.168.4.58:8001/v1",
                "api_key": "YOUR_OPENAI_API_KEY",
            },
            "VLM",
        )


def test_get_api_key_for_backend_ignores_placeholder_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "YOUR_NVIDIA_API_KEY")

    assert is_placeholder_api_key("YOUR_NVIDIA_API_KEY")
    with pytest.raises(ValueError, match="NVIDIA_API_KEY"):
        get_api_key_for_backend("nim", "VLM")
