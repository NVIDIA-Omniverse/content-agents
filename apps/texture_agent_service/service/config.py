# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Backend configuration for Texture Agent Service."""

import os
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings
from texture_agent.api.defaults import DEFAULT_LLM_BACKEND, DEFAULT_LLM_MODEL

from .utils import get_version


class ServiceConfig(BaseSettings):
    """Service configuration - FastAPI-specific settings only."""

    # Service info
    service_name: str = "Texture Agent Service"
    service_version: str = get_version()
    api_version: str = "v1"
    description: str | None = None

    # Session settings
    session_storage_path: str = "/var/texture-agent/sessions"
    session_ttl_hours: int = 24

    # File upload settings
    max_upload_size_mb: int = 500
    allowed_extensions: set[str] = {
        ".usd",
        ".usda",
        ".usdc",
        ".usdz",
        ".yaml",
        ".yml",
    }

    # Concurrency
    max_active_sessions: int = 4

    # API Keys
    nvidia_api_key: str | None = None

    # Texture generation defaults
    texture_backend: str = Field(
        default="simple_image_gen", description="Texture generation backend"
    )
    image_gen_backend: str = Field(
        default="nim",
        description=(
            "Image generation backend. Default `nim` points at NVIDIA's "
            "hosted FLUX.2 Klein 4B at build.nvidia.com and uses "
            "NVIDIA_API_KEY. The docker-compose image-gen overlay flips "
            "this to `openai` with a base_url override to route through "
            "a locally-hosted FLUX.2 NIM container."
        ),
    )
    image_gen_model: str | None = Field(
        default=None, description="Image generation model"
    )
    image_gen_base_url: str | None = Field(
        default=None,
        description=(
            "Override base URL for the image-gen backend. Used to point at "
            "a locally-hosted NIM container (OpenAI-compatible endpoint, "
            "e.g. http://image-gen-nim:8000/v1). None = use the backend's "
            "default."
        ),
    )
    llm_backend: str = Field(
        default=DEFAULT_LLM_BACKEND,
        description=(
            "Chat LLM backend used by auto-prompt generation for materials "
            "without an explicit prompt. Falls back to a templated "
            "user_prompt + material name when the backend is unavailable."
        ),
    )
    llm_model: str | None = Field(
        default=DEFAULT_LLM_MODEL,
        description="Chat LLM model name (backend-specific).",
    )
    llm_base_url: str | None = Field(
        default=None,
        description=(
            "Override base URL for the chat LLM backend. Used to route "
            "auto-prompt generation through a locally-hosted NIM container "
            "(e.g. http://llm-nim:8000/v1). None = use the backend's default."
        ),
    )
    texture_size: int = Field(default=1024, description="Texture resolution")
    texture_workers: int = Field(
        default=4, description="Parallel texture generation workers"
    )
    blend_opacity: float = Field(
        default=0.85, description="Default blend opacity (0-1)"
    )

    class Config:
        env_prefix = "TA_"
        case_sensitive = False

    def __init__(self, **kwargs):
        """Initialize config and load API keys."""
        super().__init__(**kwargs)

        # Load API keys from environment - try both prefixed and unprefixed
        if not self.nvidia_api_key:
            self.nvidia_api_key = os.getenv(
                "TA_NVIDIA_API_KEY", os.getenv("NVIDIA_API_KEY")
            )

        # Use local sessions directory for development if /var/ doesn't exist
        if not Path(self.session_storage_path).exists():
            local_sessions = Path(__file__).parent.parent / "sessions"
            self.session_storage_path = str(local_sessions)

        # Load description from README.md
        self.description = self._load_description()

    @staticmethod
    def _load_description() -> str:
        """Load description from README.md file."""
        readme_path = Path(__file__).parent.parent / "README.md"
        if readme_path.exists():
            with open(readme_path, encoding="utf-8") as f:
                return f.read()
        return "Texture Agent REST API Service"


# Global config instance
config = ServiceConfig()
