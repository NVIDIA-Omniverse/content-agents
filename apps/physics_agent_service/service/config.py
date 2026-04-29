# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Backend configuration for Physics Agent Service.

Delegates VLM/LLM defaults to physics_agent.api.defaults.
Only FastAPI-specific service settings are defined here.
"""

import os
from pathlib import Path
from urllib.parse import urlparse

from physics_agent import __version__
from physics_agent.api.defaults import (
    DEFAULT_VLM_BACKEND,
    DEFAULT_VLM_LLMGATEWAY_CONFIG,
    DEFAULT_VLM_MODEL,
    DEFAULT_VLM_TEMPERATURE,
)
from pydantic import Field
from pydantic_settings import BaseSettings

_LOCAL_RENDER_HOSTS = {
    "localhost",
    "127.0.0.1",
    "0.0.0.0",
    "::1",
    "ovrtx-rendering-api",
    "physics-ovrtx-rendering-api",
}


def _is_local_render_endpoint(endpoint: str | None) -> bool:
    """Return True when the render endpoint targets a local renderer."""
    if not endpoint:
        return False

    host = urlparse(endpoint).hostname or endpoint
    return host.lower() in _LOCAL_RENDER_HOSTS


def _backend_has_credentials(
    backend: str | None,
    *,
    nvidia_api_key: str | None,
    nstorage_api_key: str | None,
) -> bool:
    """Check whether the active backend has the credential it needs."""
    backend_name = (backend or "").lower()

    if backend_name in ("", "echo", "mock"):
        return True
    if "llmgateway" in backend_name:
        return True
    if backend_name == "nim":
        return bool(nvidia_api_key)
    if backend_name == "nvidia_inference":
        return bool(os.getenv("INFERENCE_NVIDIA_API_KEY"))
    if backend_name == "openai":
        return bool(os.getenv("OPENAI_API_KEY"))
    if backend_name == "anthropic":
        return bool(os.getenv("ANTHROPIC_API_KEY"))
    if backend_name == "gemini":
        return bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))
    if backend_name in ("azure_openai", "perflab_azure_openai"):
        return bool(os.getenv("AZURE_OPENAI_API_KEY") or nstorage_api_key)

    return True


class ServiceConfig(BaseSettings):
    """Service configuration - FastAPI-specific settings only.

    All VLM/LLM/rendering defaults come from physics_agent.api.defaults.
    """

    # Service info
    service_name: str = "Physics Agent Service"
    service_version: str = __version__
    api_version: str = "v1"
    description: str | None = None

    # Session settings
    session_storage_path: str = "/var/physics-agent/sessions"
    session_ttl_hours: int = 24

    # File upload settings
    max_upload_size_mb: int = 500
    allowed_extensions: set[str] = {".usd", ".usda", ".usdc", ".usdz", ".yaml", ".yml"}

    # API Keys
    nvidia_api_key: str | None = None
    nstorage_api_key: str | None = None
    nvcf_api_key: str | None = None

    # Storage backend (local or s3 for multi-instance)
    storage_kind: str = "local"
    storage_s3_bucket: str = ""
    storage_s3_prefix: str = ""
    storage_s3_region: str = "us-east-2"
    storage_s3_endpoint_url: str | None = None
    storage_s3_access_key_id: str | None = None
    storage_s3_secret_access_key: str | None = None
    storage_s3_session_token: str | None = None
    storage_s3_use_path_style: bool = True
    storage_s3_create_bucket: bool = False
    storage_s3_presign: bool = True
    storage_s3_sessions_cache_ttl: int = 5

    # VLM/LLM settings
    vlm_backend: str = Field(
        default=DEFAULT_VLM_BACKEND, description="VLM backend to use"
    )
    vlm_model: str = Field(default=DEFAULT_VLM_MODEL, description="VLM model to use")
    vlm_temperature: float = Field(
        default=DEFAULT_VLM_TEMPERATURE, description="VLM temperature to use"
    )
    llmgateway_config: dict[str, str | list[str] | None] = Field(
        default=DEFAULT_VLM_LLMGATEWAY_CONFIG, description="LLM gateway config to use"
    )

    class Config:
        env_prefix = "PA_"
        case_sensitive = False

    def __init__(self, **kwargs):
        """Initialize config and load API keys."""
        super().__init__(**kwargs)

        # Load API keys from environment - try both prefixed and unprefixed
        if not self.nvidia_api_key:
            self.nvidia_api_key = os.getenv(
                "PA_NVIDIA_API_KEY", os.getenv("NVIDIA_API_KEY")
            )
        if not self.nstorage_api_key:
            self.nstorage_api_key = os.getenv(
                "PA_NSTORAGE_API_KEY", os.getenv("NSTORAGE_API_KEY")
            )
        if not self.nvcf_api_key:
            self.nvcf_api_key = os.getenv("NGC_API_KEY")

        # Use local sessions directory for development if /var/ doesn't exist
        if not Path(self.session_storage_path).exists():
            local_sessions = Path(__file__).parent.parent / "sessions"
            self.session_storage_path = str(local_sessions)

        # Load description from README.md
        self.description = self._load_description()

    @staticmethod
    def _load_description() -> str:
        """Load description from README.md file."""
        readme_path = Path(__file__).parent / "README.md"
        if readme_path.exists():
            with open(readme_path, encoding="utf-8") as f:
                return f.read()
        return "Physics Agent REST API Service"

    @property
    def has_required_api_keys(self) -> bool:
        """Check if the active backend and render settings are configured."""
        vlm_ready = _backend_has_credentials(
            self.vlm_backend,
            nvidia_api_key=self.nvidia_api_key,
            nstorage_api_key=self.nstorage_api_key,
        )

        render_ready = True
        if os.getenv("PA_RENDER_BACKEND", "remote").lower() == "remote":
            render_endpoint = os.getenv("RENDER_ENDPOINT")
            if render_endpoint:
                render_ready = _is_local_render_endpoint(render_endpoint) or bool(
                    self.nvcf_api_key
                )
            elif os.getenv("NVCF_RENDER_FUNCTION_ID"):
                render_ready = bool(self.nvcf_api_key)

        return vlm_ready and render_ready

    def build_session_store(self):
        """Build a SessionStore from config."""
        from .storage import LocalSessionStore, S3SessionStore, StorageConfig

        if self.storage_kind == "s3":
            storage_cfg = StorageConfig(
                kind="s3",
                s3_bucket=self.storage_s3_bucket,
                s3_prefix=self.storage_s3_prefix,
                s3_region=self.storage_s3_region,
                s3_endpoint_url=self.storage_s3_endpoint_url,
                s3_access_key_id=self.storage_s3_access_key_id,
                s3_secret_access_key=self.storage_s3_secret_access_key,
                s3_session_token=self.storage_s3_session_token,
                s3_use_path_style=self.storage_s3_use_path_style,
                s3_create_bucket=self.storage_s3_create_bucket,
                s3_presign=self.storage_s3_presign,
                s3_sessions_cache_ttl=self.storage_s3_sessions_cache_ttl,
            )
            return S3SessionStore.from_config(storage_cfg)

        return LocalSessionStore(root_dir=self.session_storage_path)


# Global config instance
config = ServiceConfig()
