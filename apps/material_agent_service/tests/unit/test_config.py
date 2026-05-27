# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Material Agent Service configuration semantics."""

from pathlib import Path

from ...service.config import ServiceConfig


def test_has_required_api_keys_allows_local_render_with_public_nim(
    monkeypatch, tmp_path: Path
):
    """Local OVRTX rendering should not require NGC_API_KEY."""
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
    monkeypatch.setenv("RENDER_ENDPOINT", "http://ovrtx-rendering-api:8000")
    monkeypatch.delenv("NGC_API_KEY", raising=False)
    monkeypatch.delenv("NSTORAGE_API_KEY", raising=False)

    config = ServiceConfig(
        vlm_backend="nim",
        llm_backend="nim",
        session_storage_path=str(tmp_path / "sessions"),
    )

    assert config.has_required_api_keys is True


def test_has_required_api_keys_rejects_placeholder_nvidia_key(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setenv("NVIDIA_API_KEY", "YOUR_NVIDIA_API_KEY")
    monkeypatch.delenv("MA_NIM_API_KEY", raising=False)
    monkeypatch.delenv("MA_VLM_NIM_BASE_URL", raising=False)
    monkeypatch.delenv("MA_LLM_NIM_BASE_URL", raising=False)
    monkeypatch.delenv("NSTORAGE_API_KEY", raising=False)
    monkeypatch.delenv("NGC_API_KEY", raising=False)

    config = ServiceConfig(
        vlm_backend="nim",
        llm_backend="nim",
        session_storage_path=str(tmp_path / "sessions"),
    )

    assert config.has_required_api_keys is False


def test_has_required_api_keys_accepts_explicit_local_nim_placeholder(
    monkeypatch, tmp_path: Path
):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("RENDER_ENDPOINT", raising=False)
    monkeypatch.delenv("NVCF_RENDER_FUNCTION_ID", raising=False)
    monkeypatch.setenv("MA_NIM_API_KEY", "not-used")
    monkeypatch.setenv("MA_VLM_NIM_BASE_URL", "http://vlm-nim:8000/v1")
    monkeypatch.setenv("MA_LLM_NIM_BASE_URL", "http://llm-nim:8000/v1")
    monkeypatch.delenv("NSTORAGE_API_KEY", raising=False)
    monkeypatch.delenv("NGC_API_KEY", raising=False)

    config = ServiceConfig(
        vlm_backend="nim",
        llm_backend="nim",
        session_storage_path=str(tmp_path / "sessions"),
    )

    assert config.has_required_api_keys is True


def test_model_token_limits_load_from_prefixed_env(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("MA_VLM_MAX_TOKENS", "512")
    monkeypatch.setenv("MA_LLM_MAX_TOKENS", "256")
    monkeypatch.setenv("MA_LLM_TEMPERATURE", "0.2")

    config = ServiceConfig(session_storage_path=str(tmp_path / "sessions"))

    assert config.vlm_max_tokens == 512
    assert config.llm_max_tokens == 256
    assert config.llm_temperature == 0.2


def test_has_required_api_keys_rejects_hosted_nvidia_key_for_local_nim(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-hosted-key")
    monkeypatch.delenv("MA_NIM_API_KEY", raising=False)
    monkeypatch.setenv("MA_VLM_NIM_BASE_URL", "http://vlm-nim:8000/v1")
    monkeypatch.setenv("MA_LLM_NIM_BASE_URL", "http://llm-nim:8000/v1")
    monkeypatch.delenv("RENDER_ENDPOINT", raising=False)
    monkeypatch.delenv("NVCF_RENDER_FUNCTION_ID", raising=False)
    monkeypatch.delenv("NSTORAGE_API_KEY", raising=False)
    monkeypatch.delenv("NGC_API_KEY", raising=False)

    config = ServiceConfig(
        vlm_backend="nim",
        llm_backend="nim",
        session_storage_path=str(tmp_path / "sessions"),
    )

    assert config.has_required_api_keys is False


def test_has_required_api_keys_infers_llm_local_nim_override(
    monkeypatch, tmp_path: Path
):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("MA_VLM_NIM_BASE_URL", raising=False)
    monkeypatch.setenv("MA_LLM_NIM_BASE_URL", "http://llm-nim:8000/v1")
    monkeypatch.setenv("MA_NIM_API_KEY", "not-used")
    monkeypatch.delenv("RENDER_ENDPOINT", raising=False)
    monkeypatch.delenv("NVCF_RENDER_FUNCTION_ID", raising=False)
    monkeypatch.delenv("NSTORAGE_API_KEY", raising=False)
    monkeypatch.delenv("NGC_API_KEY", raising=False)

    config = ServiceConfig(
        vlm_backend="echo",
        llm_backend="openai",
        session_storage_path=str(tmp_path / "sessions"),
    )

    assert config.has_required_api_keys is True


def test_has_required_api_keys_rejects_hosted_key_for_llm_local_nim_override(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-hosted-key")
    monkeypatch.delenv("MA_NIM_API_KEY", raising=False)
    monkeypatch.delenv("MA_VLM_NIM_BASE_URL", raising=False)
    monkeypatch.setenv("MA_LLM_NIM_BASE_URL", "http://llm-nim:8000/v1")
    monkeypatch.delenv("RENDER_ENDPOINT", raising=False)
    monkeypatch.delenv("NVCF_RENDER_FUNCTION_ID", raising=False)
    monkeypatch.delenv("NSTORAGE_API_KEY", raising=False)
    monkeypatch.delenv("NGC_API_KEY", raising=False)

    config = ServiceConfig(
        vlm_backend="echo",
        llm_backend="nvidia_inference",
        session_storage_path=str(tmp_path / "sessions"),
    )

    assert config.has_required_api_keys is False


def test_has_required_api_keys_rejects_hosted_key_for_custom_nim_override(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-hosted-key")
    monkeypatch.delenv("MA_NIM_API_KEY", raising=False)
    monkeypatch.setenv("MA_VLM_NIM_BASE_URL", "https://nim.example.com/v1")
    monkeypatch.delenv("MA_LLM_NIM_BASE_URL", raising=False)
    monkeypatch.delenv("RENDER_ENDPOINT", raising=False)
    monkeypatch.delenv("NVCF_RENDER_FUNCTION_ID", raising=False)
    monkeypatch.delenv("NSTORAGE_API_KEY", raising=False)
    monkeypatch.delenv("NGC_API_KEY", raising=False)

    config = ServiceConfig(
        vlm_backend="nim",
        llm_backend="nim",
        session_storage_path=str(tmp_path / "sessions"),
    )

    assert config.has_required_api_keys is False


def test_has_required_api_keys_requires_ngc_for_remote_render(
    monkeypatch, tmp_path: Path
):
    """Remote render endpoints should still require NGC_API_KEY."""
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
    monkeypatch.setenv("RENDER_ENDPOINT", "https://renderer.example.com")
    monkeypatch.delenv("NGC_API_KEY", raising=False)
    monkeypatch.delenv("NSTORAGE_API_KEY", raising=False)

    config = ServiceConfig(
        vlm_backend="nim",
        llm_backend="nim",
        session_storage_path=str(tmp_path / "sessions"),
    )

    assert config.has_required_api_keys is False


def test_has_required_api_keys_rejects_placeholder_ngc_for_remote_render(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
    monkeypatch.setenv("RENDER_ENDPOINT", "https://renderer.example.com")
    monkeypatch.setenv("NGC_API_KEY", "YOUR_NGC_API_KEY")
    monkeypatch.delenv("NSTORAGE_API_KEY", raising=False)

    config = ServiceConfig(
        vlm_backend="nim",
        llm_backend="nim",
        session_storage_path=str(tmp_path / "sessions"),
    )

    assert config.has_required_api_keys is False


def test_has_required_api_keys_rejects_placeholder_nstorage_key(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setenv("NSTORAGE_API_KEY", "YOUR_NSTORAGE_API_KEY")
    monkeypatch.delenv("MA_NSTORAGE_API_KEY", raising=False)
    monkeypatch.delenv("NGC_API_KEY", raising=False)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)

    config = ServiceConfig(
        vlm_backend="azure_openai",
        llm_backend="azure_openai",
        session_storage_path=str(tmp_path / "sessions"),
    )

    assert config.has_required_api_keys is False


def test_has_required_api_keys_rejects_openai_with_env_redirected_base_url(
    monkeypatch, tmp_path: Path
):
    """When ``OPENAI_BASE_URL`` redirects the SDK to a custom endpoint, a
    hosted ``OPENAI_API_KEY`` alone must not make readiness pass — runtime
    model construction will reject the same combination, so reporting
    ``has_required_api_keys: True`` would mislead /health and suppress
    startup warnings."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real-openai-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai-compatible.example/v1")
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    monkeypatch.setenv("RENDER_ENDPOINT", "http://ovrtx-rendering-api:8000")
    monkeypatch.delenv("MA_VLM_NIM_BASE_URL", raising=False)
    monkeypatch.delenv("MA_LLM_NIM_BASE_URL", raising=False)
    monkeypatch.delenv("NGC_API_KEY", raising=False)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)

    config = ServiceConfig(
        vlm_backend="openai",
        llm_backend="openai",
        session_storage_path=str(tmp_path / "sessions"),
    )

    assert config.has_required_api_keys is False


def test_image_gen_ready_accepts_explicit_key_for_custom_backend(
    monkeypatch, tmp_path: Path
):
    """``MA_IMAGE_GEN_API_KEY`` must satisfy readiness for backends not in
    the standard env-var map (e.g. internal-only or registry-provided
    backends). Otherwise the route 503s before the registered factory can
    use the explicit key it now supports."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    config = ServiceConfig(
        image_gen_backend="custom_registry_provider",
        image_gen_api_key="secret-endpoint-key",
        session_storage_path=str(tmp_path / "sessions-custom"),
    )

    assert config.image_gen_ready is True


def test_image_gen_ready_rejects_custom_backend_without_explicit_key(
    monkeypatch, tmp_path: Path
):
    """Custom backends still need an explicit key — readiness should not
    silently report True when no credential is configured."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)

    config = ServiceConfig(
        image_gen_backend="custom_registry_provider",
        session_storage_path=str(tmp_path / "sessions-custom-noauth"),
    )

    assert config.image_gen_ready is False


def test_has_required_api_keys_accepts_openai_against_provider_endpoint(
    monkeypatch, tmp_path: Path
):
    """Without an env redirect, a hosted ``OPENAI_API_KEY`` is sufficient
    for the default (provider-owned) OpenAI endpoint."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real-openai-key")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    monkeypatch.setenv("RENDER_ENDPOINT", "http://ovrtx-rendering-api:8000")
    monkeypatch.delenv("MA_VLM_NIM_BASE_URL", raising=False)
    monkeypatch.delenv("MA_LLM_NIM_BASE_URL", raising=False)
    monkeypatch.delenv("NGC_API_KEY", raising=False)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)

    config = ServiceConfig(
        vlm_backend="openai",
        llm_backend="openai",
        session_storage_path=str(tmp_path / "sessions"),
    )

    assert config.has_required_api_keys is True


def test_image_gen_configuration_defaults_to_public_backend(tmp_path: Path):
    config = ServiceConfig(session_storage_path=str(tmp_path / "sessions"))

    assert config.image_gen_backend == "gemini"
    assert config.image_gen_model is None
    assert config.image_gen_base_url is None
    assert config.image_gen_api_key is None


def test_render_worker_cap_reads_deployment_env(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("MA_MAX_RENDER_NUM_WORKERS", "4")

    config = ServiceConfig(session_storage_path=str(tmp_path / "sessions"))

    assert config.max_render_num_workers == 4


def test_cluster_configuration_defaults_to_nim_nemotron_vl_embed(tmp_path: Path):
    config = ServiceConfig(session_storage_path=str(tmp_path / "sessions"))

    assert config.cluster_embedding_backend == "nim"
    assert config.cluster_embedding_model == "nvidia/llama-nemotron-embed-vl-1b-v2"
    assert config.cluster_embedding_base_url is None
    assert config.cluster_embedding_api_key is None
    assert config.cluster_embedding_max_workers == 4
    assert config.cluster_embedding_batch_size == 50
    assert config.cluster_min_prims == 50
    assert config.cluster_max_size == 25


def test_cluster_configuration_reads_deployment_env(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("MA_CLUSTER_EMBEDDING_BACKEND", "nim")
    monkeypatch.setenv(
        "MA_CLUSTER_EMBEDDING_MODEL", "nvidia/llama-nemotron-embed-vl-1b-v2"
    )
    monkeypatch.setenv("MA_CLUSTER_EMBEDDING_BASE_URL", "http://embedding-nim:8000/v1")
    monkeypatch.setenv("MA_CLUSTER_EMBEDDING_API_KEY", "cluster-secret")
    monkeypatch.setenv("MA_CLUSTER_EMBEDDING_MAX_WORKERS", "2")
    monkeypatch.setenv("MA_CLUSTER_EMBEDDING_BATCH_SIZE", "8")
    monkeypatch.setenv("MA_CLUSTER_MIN_PRIMS", "25")
    monkeypatch.setenv("MA_CLUSTER_MAX_SIZE", "12")

    config = ServiceConfig(session_storage_path=str(tmp_path / "sessions"))

    assert config.cluster_embedding_backend == "nim"
    assert config.cluster_embedding_model == "nvidia/llama-nemotron-embed-vl-1b-v2"
    assert config.cluster_embedding_base_url == "http://embedding-nim:8000/v1"
    assert config.cluster_embedding_api_key == "cluster-secret"
    assert config.cluster_embedding_max_workers == 2
    assert config.cluster_embedding_batch_size == 8
    assert config.cluster_min_prims == 25
    assert config.cluster_max_size == 12


def test_image_gen_configuration_reads_deployment_env(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("MA_IMAGE_GEN_BACKEND", "openai")
    monkeypatch.setenv("MA_IMAGE_GEN_MODEL", "gpt-image-1")
    monkeypatch.setenv("MA_IMAGE_GEN_BASE_URL", "http://image-gen:8000/v1")
    monkeypatch.setenv("MA_IMAGE_GEN_API_KEY", "not-used")

    config = ServiceConfig(session_storage_path=str(tmp_path / "sessions"))

    assert config.image_gen_backend == "openai"
    assert config.image_gen_model == "gpt-image-1"
    assert config.image_gen_base_url == "http://image-gen:8000/v1"
    assert config.image_gen_api_key == "not-used"


def test_image_gen_ready_validates_selected_backend(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    config = ServiceConfig(
        image_gen_backend="gemini",
        session_storage_path=str(tmp_path / "sessions"),
    )
    assert config.image_gen_ready is False

    monkeypatch.setenv("GEMINI_API_KEY", "gemini-test")
    assert config.image_gen_ready is True

    monkeypatch.setenv("GOOGLE_API_KEY", "google-test")
    assert config.image_gen_ready is True

    openai_config = ServiceConfig(
        image_gen_backend="openai",
        image_gen_base_url="http://image-gen.local/v1",
        session_storage_path=str(tmp_path / "sessions-openai"),
    )
    assert openai_config.image_gen_ready is False

    openai_local_no_auth_config = ServiceConfig(
        image_gen_backend="openai",
        image_gen_base_url="http://image-gen.local/v1",
        image_gen_api_key="not-used",
        session_storage_path=str(tmp_path / "sessions-openai-local-no-auth"),
    )
    assert openai_local_no_auth_config.image_gen_ready is True

    openai_host_docker_no_auth_config = ServiceConfig(
        image_gen_backend="openai",
        image_gen_base_url="http://host.docker.internal:8016/v1",
        image_gen_api_key="not-used",
        session_storage_path=str(tmp_path / "sessions-openai-host-docker"),
    )
    assert openai_host_docker_no_auth_config.image_gen_ready is True

    # Hosted ``OPENAI_API_KEY`` must not silently flow to a local
    # OpenAI-compatible endpoint via env: the local URL is a non-provider
    # trust boundary and requires an explicit endpoint-scoped api_key.
    monkeypatch.setenv("OPENAI_API_KEY", "local-openai-key")
    openai_local_no_explicit_key_config = ServiceConfig(
        image_gen_backend="openai",
        image_gen_base_url="http://image-gen.local/v1",
        session_storage_path=str(tmp_path / "sessions-openai-local-auth"),
    )
    assert openai_local_no_explicit_key_config.image_gen_ready is False
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    remote_openai_config = ServiceConfig(
        image_gen_backend="openai",
        image_gen_base_url="https://api.openai-compatible.example/v1",
        session_storage_path=str(tmp_path / "sessions-openai-remote"),
    )
    assert remote_openai_config.image_gen_ready is False

    openai_without_config = ServiceConfig(
        image_gen_backend="openai",
        session_storage_path=str(tmp_path / "sessions-openai-missing"),
    )
    assert openai_without_config.image_gen_ready is False

    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.setenv("MA_NIM_API_KEY", "not-used")
    nim_local_config = ServiceConfig(
        image_gen_backend="nim",
        image_gen_base_url="http://image-gen-nim:8000/v1",
        session_storage_path=str(tmp_path / "sessions-nim-local"),
    )
    assert nim_local_config.image_gen_ready is True


def test_image_gen_ready_accepts_explicit_key_for_gemini(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    config = ServiceConfig(
        image_gen_backend="gemini",
        image_gen_api_key="image-gen-key",
        session_storage_path=str(tmp_path / "sessions-gemini-explicit"),
    )

    assert config.image_gen_ready is True


def test_image_gen_ready_accepts_explicit_key_for_nvidia_inference(
    monkeypatch, tmp_path: Path
):
    monkeypatch.delenv("INFERENCE_NVIDIA_API_KEY", raising=False)

    config = ServiceConfig(
        image_gen_backend="nvidia_inference",
        image_gen_api_key="image-gen-key",
        session_storage_path=str(tmp_path / "sessions-nvidia-inference-explicit"),
    )

    assert config.image_gen_ready is True


def test_image_gen_ready_rejects_hosted_nvidia_key_for_local_nim(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-hosted-key")
    monkeypatch.delenv("MA_NIM_API_KEY", raising=False)

    config = ServiceConfig(
        image_gen_backend="nim",
        image_gen_base_url="http://image-gen-nim:8000/v1",
        session_storage_path=str(tmp_path / "sessions-nim-local"),
    )

    assert config.image_gen_ready is False


def test_image_gen_ready_rejects_hosted_nvidia_key_for_custom_nim(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-hosted-key")
    monkeypatch.delenv("MA_NIM_API_KEY", raising=False)

    config = ServiceConfig(
        image_gen_backend="nim",
        image_gen_base_url="https://nim.example.com/v1/genai",
        session_storage_path=str(tmp_path / "sessions-nim-custom"),
    )

    assert config.image_gen_ready is False


def test_image_gen_ready_accepts_explicit_key_for_custom_nim(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-hosted-key")
    monkeypatch.delenv("MA_NIM_API_KEY", raising=False)

    config = ServiceConfig(
        image_gen_backend="nim",
        image_gen_base_url="https://nim.example.com/v1/genai",
        image_gen_api_key="endpoint-nim-key",
        session_storage_path=str(tmp_path / "sessions-nim-custom-explicit"),
    )

    assert config.image_gen_ready is True
