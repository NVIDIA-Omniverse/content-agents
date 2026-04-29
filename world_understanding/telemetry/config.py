# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Telemetry configuration models for the MAA (World Understanding) project.

This module provides Pydantic-based configuration classes for managing
OpenTelemetry exporters and tracing settings. Configuration can be loaded
from environment variables using the OTEL_ prefix.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class ExporterType(str, Enum):
    """Supported telemetry exporter types.

    Attributes:
        TEMPO: Grafana Tempo exporter using OTLP/gRPC.
        LANGFUSE: Langfuse observability platform exporter.
        HTTP: Generic HTTP/OTLP exporter.
        CONSOLE: Console/stdout exporter for development.
        NONE: Disabled exporter (no-op).
    """

    TEMPO = "tempo"
    LANGFUSE = "langfuse"
    HTTP = "http"
    FILE = "file"
    CONSOLE = "console"
    NONE = "none"


class TempoConfig(BaseModel):
    """Configuration for Grafana Tempo exporter.

    Attributes:
        endpoint: The Tempo OTLP gRPC endpoint (host:port).
        insecure: Whether to use insecure (non-TLS) connection.
        compression: Compression algorithm to use for exports.
    """

    endpoint: str = "localhost:4317"
    insecure: bool = True
    compression: Literal["none", "gzip"] = "gzip"


class LangfuseConfig(BaseModel):
    """Configuration for Langfuse observability platform exporter.

    Attributes:
        endpoint: The Langfuse OTEL API endpoint URL.
        public_key: Langfuse public API key for authentication.
        secret_key: Langfuse secret API key for authentication.
        environment: Environment name for trace categorization.
    """

    endpoint: str = "https://cloud.langfuse.com/api/public/otel"
    public_key: str
    secret_key: str
    environment: str = "default"


class HTTPConfig(BaseModel):
    """Configuration for generic HTTP/OTLP exporter.

    Attributes:
        endpoint: The OTLP HTTP endpoint URL.
        headers: Additional HTTP headers to include in requests.
        timeout_seconds: Request timeout in seconds.
    """

    endpoint: str
    headers: dict[str, str] = Field(default_factory=dict)
    timeout_seconds: int = 30


class FileConfig(BaseModel):
    """Configuration for local file exporter.

    Attributes:
        path: Output path for JSONL spans.
        append: Whether to append to existing file (True) or truncate (False).
    """

    path: str
    append: bool = True


class TelemetryConfig(BaseSettings):
    """Main telemetry configuration loaded from environment variables.

    This configuration class uses pydantic-settings to automatically load
    values from environment variables with the OTEL_ prefix. Nested
    configuration uses double underscore as delimiter.

    Environment variable examples:
        - OTEL_ENABLED=true
        - OTEL_SERVICE_NAME=my-service
        - OTEL_TEMPO__ENDPOINT=tempo:4317
        - OTEL_LANGFUSE__PUBLIC_KEY=pk-xxx

    Attributes:
        enabled: Whether telemetry collection is enabled.
        service_name: Name of the service for trace identification.
        service_version: Version string of the service.
        environment: Deployment environment (development, staging, production).
        exporters: List of exporters to use for sending traces.
        sample_rate: Probability of sampling a trace (0.0 to 1.0).
        tempo: Optional Tempo exporter configuration.
        langfuse: Optional Langfuse exporter configuration.
        http: Optional HTTP exporter configuration.
        file: Optional local file exporter configuration.
        batch_max_queue_size: Maximum spans queued before dropping.
        batch_schedule_delay_ms: Delay between batch exports in milliseconds.
        batch_max_export_size: Maximum spans per export batch.
        batch_export_timeout_ms: Timeout for export operations in milliseconds.
    """

    enabled: bool = True
    service_name: str = "world-understanding"
    service_version: str = "0.1.0"
    environment: str = "development"
    exporters: str = "console"
    sample_rate: float = Field(1.0, ge=0.0, le=1.0)

    def get_exporters(self) -> list[ExporterType]:
        """Parse exporters from comma-separated string."""
        parts = [p.strip() for p in self.exporters.split(",") if p.strip()]
        return [ExporterType(p) for p in parts] if parts else [ExporterType.CONSOLE]

    # Exporter configs (optional, loaded from env)
    tempo: TempoConfig | None = None
    langfuse: LangfuseConfig | None = None
    http: HTTPConfig | None = None
    file: FileConfig | None = None

    # Batch processor settings
    batch_max_queue_size: int = 2048
    batch_schedule_delay_ms: int = 5000
    batch_max_export_size: int = 512
    batch_export_timeout_ms: int = 30000

    model_config = {"env_prefix": "OTEL_", "env_nested_delimiter": "__"}
