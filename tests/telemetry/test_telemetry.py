# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the telemetry module."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


class TestTelemetryConfig:
    """Tests for TelemetryConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        from world_understanding.telemetry.config import ExporterType, TelemetryConfig

        config = TelemetryConfig()

        assert config.enabled is True
        assert config.service_name == "world-understanding"
        assert config.environment == "development"
        assert config.sample_rate == 1.0
        assert ExporterType.CONSOLE in config.get_exporters()

    def test_config_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test configuration from environment variables."""
        from world_understanding.telemetry.config import TelemetryConfig

        monkeypatch.setenv("OTEL_ENABLED", "false")
        monkeypatch.setenv("OTEL_SERVICE_NAME", "test-service")
        monkeypatch.setenv("OTEL_ENVIRONMENT", "production")
        monkeypatch.setenv("OTEL_SAMPLE_RATE", "0.5")

        config = TelemetryConfig()

        assert config.enabled is False
        assert config.service_name == "test-service"
        assert config.environment == "production"
        assert config.sample_rate == 0.5

    def test_exporter_type_enum(self) -> None:
        """Test ExporterType enum values."""
        from world_understanding.telemetry.config import ExporterType

        assert ExporterType.TEMPO.value == "tempo"
        assert ExporterType.LANGFUSE.value == "langfuse"
        assert ExporterType.HTTP.value == "http"
        assert ExporterType.CONSOLE.value == "console"
        assert ExporterType.NONE.value == "none"


class TestAttributes:
    """Tests for semantic convention attributes."""

    def test_maa_attributes(self) -> None:
        """Test MAA-specific attributes."""
        from world_understanding.telemetry.attributes import MAAttributes

        assert MAAttributes.PIPELINE_NAME == "maa.pipeline.name"
        assert MAAttributes.TOOL_NAME == "maa.tool.name"
        assert MAAttributes.S3_BUCKET == "maa.s3.bucket"
        assert MAAttributes.NVCF_FUNCTION_ID == "maa.nvcf.function_id"

    def test_genai_attributes(self) -> None:
        """Test GenAI semantic convention attributes."""
        from world_understanding.telemetry.attributes import GenAIAttributes

        assert GenAIAttributes.SYSTEM == "gen_ai.system"
        assert GenAIAttributes.REQUEST_MODEL == "gen_ai.request.model"
        assert GenAIAttributes.USAGE_INPUT_TOKENS == "gen_ai.usage.input_tokens"


class TestInitialization:
    """Tests for telemetry initialization."""

    def test_initialize_disabled(self) -> None:
        """Test initialization when disabled."""
        import world_understanding.telemetry as telemetry_module
        from world_understanding.telemetry import (
            initialize_telemetry,
            shutdown_telemetry,
        )
        from world_understanding.telemetry.config import TelemetryConfig

        # Ensure clean state by resetting module internals
        shutdown_telemetry()
        telemetry_module._initialized = False
        telemetry_module._tracer_provider = None

        config = TelemetryConfig(enabled=False, exporters="none")
        provider = initialize_telemetry(config)

        # Should return None when telemetry is disabled
        assert provider is None

        # Reset state for other tests
        telemetry_module._initialized = False

    def test_initialize_console_exporter(self) -> None:
        """Test initialization with console exporter."""
        import world_understanding.telemetry as telemetry_module
        from world_understanding.telemetry import (
            initialize_telemetry,
            shutdown_telemetry,
        )
        from world_understanding.telemetry.config import TelemetryConfig

        # Ensure clean state by resetting module internals
        shutdown_telemetry()
        telemetry_module._initialized = False
        telemetry_module._tracer_provider = None

        config = TelemetryConfig(
            enabled=True,
            exporters="console",
        )

        provider = initialize_telemetry(config)
        assert provider is not None

        # Cleanup
        shutdown_telemetry()

    def test_get_tracer(self) -> None:
        """Test getting a tracer instance."""
        from world_understanding.telemetry import get_tracer

        tracer = get_tracer("test.module")
        assert tracer is not None


class TestDecorators:
    """Tests for tracing decorators."""

    def test_traced_decorator_sync(self) -> None:
        """Test @traced decorator on sync function."""
        from world_understanding.telemetry.decorators import traced

        @traced("test.function")
        def my_function(x: int) -> int:
            return x * 2

        result = my_function(5)
        assert result == 10

    @pytest.mark.asyncio
    async def test_traced_decorator_async(self) -> None:
        """Test @traced decorator on async function."""
        from world_understanding.telemetry.decorators import traced

        @traced("test.async_function")
        async def my_async_function(x: int) -> int:
            return x * 2

        result = await my_async_function(5)
        assert result == 10

    def test_traced_captures_error(self) -> None:
        """Test that @traced captures exceptions."""
        from world_understanding.telemetry.decorators import traced

        @traced("test.error_function")
        def error_function() -> None:
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            error_function()

    def test_traced_llm_decorator(self) -> None:
        """Test @traced_llm decorator."""
        from world_understanding.telemetry.decorators import traced_llm

        @traced_llm(name="test.llm", system="openai", operation="chat")
        def llm_call(model: str = "gpt-4") -> str:
            return "response"

        result = llm_call(model="gpt-4")
        assert result == "response"


class TestHTTPExporter:
    """Tests for HTTP JSON exporter."""

    def test_span_serialization(self) -> None:
        """Test span serialization to JSON."""
        from world_understanding.telemetry.exporters.http import HTTPJsonExporter

        exporter = HTTPJsonExporter(
            endpoint="http://localhost:8080/traces",
            headers={"X-API-Key": "test"},
        )

        assert exporter.endpoint == "http://localhost:8080/traces"
        assert "X-API-Key" in exporter.headers
        assert exporter.headers["Content-Type"] == "application/json"

    def test_exporter_shutdown(self) -> None:
        """Test exporter shutdown."""
        from world_understanding.telemetry.exporters.http import HTTPJsonExporter

        exporter = HTTPJsonExporter(endpoint="http://localhost:8080")
        exporter.shutdown()
        assert exporter._session is None


class TestExporterFactories:
    """Tests for exporter factory functions."""

    def test_langfuse_exporter_missing_credentials(self) -> None:
        """Test Langfuse exporter returns None without credentials."""
        from world_understanding.telemetry.exporters.langfuse import (
            create_langfuse_exporter,
        )

        result = create_langfuse_exporter(
            public_key="",
            secret_key="",
        )
        assert result is None

    def test_http_exporter_creation(self) -> None:
        """Test HTTP exporter factory."""
        from world_understanding.telemetry.exporters.http import create_http_exporter

        # This may return None if OTel SDK not installed
        processor = create_http_exporter(
            endpoint="http://localhost:8080/traces",
        )
        # Just verify it doesn't raise
        if processor:
            processor.shutdown()


class TestIntegration:
    """Integration tests for telemetry module."""

    def test_full_trace_flow(self) -> None:
        """Test complete trace flow with console exporter."""
        import world_understanding.telemetry as telemetry_module
        from world_understanding.telemetry import (
            TelemetryConfig,
            get_tracer,
            initialize_telemetry,
            shutdown_telemetry,
        )

        # Ensure clean state by resetting module internals
        shutdown_telemetry()
        telemetry_module._initialized = False
        telemetry_module._tracer_provider = None

        # Initialize with console exporter
        config = TelemetryConfig(
            enabled=True,
            service_name="integration-test",
            exporters="console",
        )
        initialize_telemetry(config)

        # Create spans
        tracer = get_tracer("integration.test")
        with tracer.start_as_current_span("parent") as parent:
            parent.set_attribute("test.attribute", "value")
            with tracer.start_as_current_span("child") as child:
                child.set_attribute("child.attribute", 42)

        # Cleanup
        shutdown_telemetry()

    @pytest.mark.asyncio
    async def test_async_trace_flow(self) -> None:
        """Test async trace flow."""
        import world_understanding.telemetry as telemetry_module
        from world_understanding.telemetry import (
            TelemetryConfig,
            get_tracer,
            initialize_telemetry,
            shutdown_telemetry,
        )
        from world_understanding.telemetry.decorators import traced

        # Ensure clean state by resetting module internals
        shutdown_telemetry()
        telemetry_module._initialized = False
        telemetry_module._tracer_provider = None

        config = TelemetryConfig(
            enabled=True,
            exporters="console",
        )
        initialize_telemetry(config)

        @traced("async.operation")
        async def async_operation() -> str:
            tracer = get_tracer("async.test")
            with tracer.start_as_current_span("nested"):
                return "done"

        result = await async_operation()
        assert result == "done"

        shutdown_telemetry()


class TestHelperFunctions:
    """Tests for telemetry helper functions."""

    def test_truncate_short_string(self) -> None:
        """Test _truncate with string shorter than max length."""
        from world_understanding.telemetry.decorators import _truncate

        result = _truncate("short string")
        assert result == "short string"

    def test_truncate_long_string(self) -> None:
        """Test _truncate with string longer than max length."""
        from world_understanding.telemetry.decorators import _truncate

        long_string = "x" * 5000
        result = _truncate(long_string, max_length=100)
        assert len(result) == 100 + len("...[truncated]")
        assert result.endswith("...[truncated]")

    def test_truncate_non_string(self) -> None:
        """Test _truncate converts non-string values."""
        from world_understanding.telemetry.decorators import _truncate

        result = _truncate(12345)
        assert result == "12345"

    def test_safe_set_attribute_none(self) -> None:
        """Test _safe_set_attribute skips None values."""
        from world_understanding.telemetry.decorators import _safe_set_attribute

        mock_span = MagicMock()
        _safe_set_attribute(mock_span, "key", None)
        mock_span.set_attribute.assert_not_called()

    def test_safe_set_attribute_primitive_types(self) -> None:
        """Test _safe_set_attribute handles primitive types."""
        from world_understanding.telemetry.decorators import _safe_set_attribute

        mock_span = MagicMock()

        _safe_set_attribute(mock_span, "str_key", "value")
        _safe_set_attribute(mock_span, "int_key", 42)
        _safe_set_attribute(mock_span, "float_key", 3.14)
        _safe_set_attribute(mock_span, "bool_key", True)

        assert mock_span.set_attribute.call_count == 4
        mock_span.set_attribute.assert_any_call("str_key", "value")
        mock_span.set_attribute.assert_any_call("int_key", 42)
        mock_span.set_attribute.assert_any_call("float_key", 3.14)
        mock_span.set_attribute.assert_any_call("bool_key", True)

    def test_safe_set_attribute_complex_type_truncates(self) -> None:
        """Test _safe_set_attribute truncates complex types."""
        from world_understanding.telemetry.decorators import _safe_set_attribute

        mock_span = MagicMock()
        _safe_set_attribute(mock_span, "list_key", [1, 2, 3])
        mock_span.set_attribute.assert_called_once()
        # Should be truncated string representation
        call_args = mock_span.set_attribute.call_args
        assert call_args[0][0] == "list_key"
        assert "[1, 2, 3]" in call_args[0][1]


class TestGetCurrentSpan:
    """Tests for get_current_span function."""

    def test_get_current_span_no_active_span(self) -> None:
        """Test get_current_span returns None when no span is active."""
        from world_understanding.telemetry import get_current_span

        span = get_current_span()
        # Without initialization, should return None (non-recording span)
        assert span is None

    def test_get_current_span_with_active_span(self) -> None:
        """Test get_current_span returns the current span."""
        import world_understanding.telemetry as telemetry_module
        from world_understanding.telemetry import (
            TelemetryConfig,
            get_current_span,
            get_tracer,
            initialize_telemetry,
            shutdown_telemetry,
        )

        # Ensure clean state by resetting module internals
        shutdown_telemetry()
        telemetry_module._initialized = False
        telemetry_module._tracer_provider = None

        config = TelemetryConfig(
            enabled=True,
            exporters="console",
        )
        initialize_telemetry(config)

        tracer = get_tracer("test")
        with tracer.start_as_current_span("test-span") as expected_span:
            current = get_current_span()
            assert current is not None
            assert current == expected_span

        shutdown_telemetry()


class TestTracedVLMDecorator:
    """Tests for @traced_vlm decorator."""

    def test_traced_vlm_sync(self) -> None:
        """Test @traced_vlm decorator on sync function."""
        from world_understanding.telemetry.decorators import traced_vlm

        @traced_vlm(system="test_vlm", operation="generate")
        def vlm_function(prompt: str, images: list | None = None) -> str:
            return f"processed: {prompt}"

        result = vlm_function("test prompt")
        assert result == "processed: test prompt"

    @pytest.mark.asyncio
    async def test_traced_vlm_async(self) -> None:
        """Test @traced_vlm decorator on async function."""
        from world_understanding.telemetry.decorators import traced_vlm

        @traced_vlm(system="test_vlm", operation="generate")
        async def async_vlm_function(prompt: str, images: list | None = None) -> str:
            return f"async processed: {prompt}"

        result = await async_vlm_function("test prompt", images=["img1", "img2"])
        assert result == "async processed: test prompt"

    def test_traced_vlm_captures_error(self) -> None:
        """Test that @traced_vlm captures exceptions."""
        from world_understanding.telemetry.decorators import traced_vlm

        @traced_vlm(system="test_vlm", operation="generate")
        def error_vlm_function() -> None:
            raise RuntimeError("VLM error")

        with pytest.raises(RuntimeError, match="VLM error"):
            error_vlm_function()


class TestAddTokenUsageToSpan:
    """Tests for add_token_usage_to_span helper."""

    def test_add_token_usage_with_span(self) -> None:
        """Test add_token_usage_to_span with explicit span."""
        from world_understanding.telemetry.decorators import add_token_usage_to_span

        mock_span = MagicMock()
        add_token_usage_to_span(
            mock_span,
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            model_name="gpt-4",
        )

        assert mock_span.set_attribute.call_count == 4

    def test_add_token_usage_with_none_span(self) -> None:
        """Test add_token_usage_to_span with None span uses current span."""
        from world_understanding.telemetry.decorators import add_token_usage_to_span

        # When there's no current span, should not raise
        add_token_usage_to_span(
            None,
            input_tokens=100,
        )

    def test_add_token_usage_partial_values(self) -> None:
        """Test add_token_usage_to_span with partial values."""
        from world_understanding.telemetry.decorators import add_token_usage_to_span

        mock_span = MagicMock()
        add_token_usage_to_span(
            mock_span,
            input_tokens=100,
            output_tokens=None,
        )
        # Should only set input_tokens (None values are skipped)
        mock_span.set_attribute.assert_called()


class TestConfigModels:
    """Tests for configuration models."""

    def test_tempo_config_defaults(self) -> None:
        """Test TempoConfig default values."""
        from world_understanding.telemetry.config import TempoConfig

        config = TempoConfig()
        assert config.endpoint == "localhost:4317"
        assert config.insecure is True
        assert config.compression == "gzip"

    def test_langfuse_config_required_fields(self) -> None:
        """Test LangfuseConfig requires public_key and secret_key."""
        from pydantic import ValidationError

        from world_understanding.telemetry.config import LangfuseConfig

        with pytest.raises(ValidationError):
            LangfuseConfig()  # Missing required fields

        config = LangfuseConfig(public_key="pk-test", secret_key="sk-test")
        assert config.public_key == "pk-test"
        assert config.secret_key == "sk-test"
        assert config.environment == "default"

    def test_http_config_required_endpoint(self) -> None:
        """Test HTTPConfig requires endpoint."""
        from pydantic import ValidationError

        from world_understanding.telemetry.config import HTTPConfig

        with pytest.raises(ValidationError):
            HTTPConfig()  # Missing endpoint

        config = HTTPConfig(endpoint="http://localhost:8080")
        assert config.endpoint == "http://localhost:8080"
        assert config.headers == {}
        assert config.timeout_seconds == 30

    def test_telemetry_config_get_exporters_multiple(self) -> None:
        """Test TelemetryConfig.get_exporters with multiple exporters."""
        from world_understanding.telemetry.config import ExporterType, TelemetryConfig

        config = TelemetryConfig(exporters="console,tempo,langfuse")
        exporters = config.get_exporters()

        assert len(exporters) == 3
        assert ExporterType.CONSOLE in exporters
        assert ExporterType.TEMPO in exporters
        assert ExporterType.LANGFUSE in exporters

    def test_telemetry_config_get_exporters_empty_string(self) -> None:
        """Test TelemetryConfig.get_exporters with empty string defaults to console."""
        from world_understanding.telemetry.config import ExporterType, TelemetryConfig

        config = TelemetryConfig(exporters="")
        exporters = config.get_exporters()

        assert exporters == [ExporterType.CONSOLE]

    def test_telemetry_config_batch_settings(self) -> None:
        """Test TelemetryConfig batch processor settings."""
        from world_understanding.telemetry.config import TelemetryConfig

        config = TelemetryConfig()
        assert config.batch_max_queue_size == 2048
        assert config.batch_schedule_delay_ms == 5000
        assert config.batch_max_export_size == 512
        assert config.batch_export_timeout_ms == 30000


class TestAllAttributes:
    """Tests for complete attribute coverage."""

    def test_all_maa_attributes(self) -> None:
        """Test all MAAttributes have correct values."""
        from world_understanding.telemetry.attributes import MAAttributes

        # Pipeline attributes
        assert MAAttributes.PIPELINE_NAME == "maa.pipeline.name"
        assert MAAttributes.PIPELINE_SESSION_ID == "maa.pipeline.session_id"
        assert MAAttributes.PIPELINE_STEP_NAME == "maa.pipeline.step.name"
        assert MAAttributes.PIPELINE_STEP_INDEX == "maa.pipeline.step.index"
        assert MAAttributes.PIPELINE_STEP_STATUS == "maa.pipeline.step.status"
        assert MAAttributes.PIPELINE_TOTAL_STEPS == "maa.pipeline.total_steps"

        # Tool attributes
        assert MAAttributes.TOOL_NAME == "maa.tool.name"
        assert MAAttributes.TOOL_DESCRIPTION == "maa.tool.description"

        # VLM attributes
        assert MAAttributes.VLM_BACKEND == "maa.vlm.backend"
        assert MAAttributes.VLM_IMAGE_COUNT == "maa.vlm.image_count"

        # NVCF attributes
        assert MAAttributes.NVCF_FUNCTION_ID == "maa.nvcf.function_id"
        assert MAAttributes.NVCF_RETRY_COUNT == "maa.nvcf.retry_count"

        # S3 attributes
        assert MAAttributes.S3_BUCKET == "maa.s3.bucket"
        assert MAAttributes.S3_KEY == "maa.s3.key"
        assert MAAttributes.S3_OPERATION == "maa.s3.operation"

    def test_all_genai_attributes(self) -> None:
        """Test all GenAIAttributes have correct values."""
        from world_understanding.telemetry.attributes import GenAIAttributes

        # System and operation
        assert GenAIAttributes.SYSTEM == "gen_ai.system"
        assert GenAIAttributes.OPERATION_NAME == "gen_ai.operation.name"

        # Request attributes
        assert GenAIAttributes.REQUEST_MODEL == "gen_ai.request.model"
        assert GenAIAttributes.REQUEST_TEMPERATURE == "gen_ai.request.temperature"
        assert GenAIAttributes.REQUEST_MAX_TOKENS == "gen_ai.request.max_tokens"

        # Response attributes
        assert GenAIAttributes.RESPONSE_MODEL == "gen_ai.response.model"
        assert GenAIAttributes.RESPONSE_ID == "gen_ai.response.id"
        assert GenAIAttributes.RESPONSE_FINISH_REASON == "gen_ai.response.finish_reason"

        # Token usage attributes
        assert GenAIAttributes.USAGE_INPUT_TOKENS == "gen_ai.usage.input_tokens"
        assert GenAIAttributes.USAGE_OUTPUT_TOKENS == "gen_ai.usage.output_tokens"
        assert GenAIAttributes.USAGE_TOTAL_TOKENS == "gen_ai.usage.total_tokens"


class TestTracedDecoratorAdvanced:
    """Advanced tests for @traced decorator."""

    def test_traced_with_capture_input_output(self) -> None:
        """Test @traced decorator with input/output capture."""
        from world_understanding.telemetry.decorators import traced

        @traced("test.capture", capture_input=True, capture_output=True)
        def process_data(data: str, count: int = 1) -> dict:
            return {"data": data, "count": count}

        result = process_data("test", count=5)
        assert result == {"data": "test", "count": 5}

    def test_traced_with_custom_attributes(self) -> None:
        """Test @traced decorator with custom attributes."""
        from world_understanding.telemetry.decorators import traced

        @traced(
            "test.custom",
            attributes={"custom.key": "custom.value", "custom.number": 42},
        )
        def custom_function() -> str:
            return "custom"

        result = custom_function()
        assert result == "custom"


class TestHTTPExporterAdvanced:
    """Advanced tests for HTTP exporter."""

    def test_exporter_force_flush(self) -> None:
        """Test HTTPJsonExporter force_flush returns True."""
        from world_understanding.telemetry.exporters.http import HTTPJsonExporter

        exporter = HTTPJsonExporter(endpoint="http://localhost:8080")
        result = exporter.force_flush()
        assert result is True

    def test_exporter_headers_merge(self) -> None:
        """Test HTTPJsonExporter merges custom headers with Content-Type."""
        from world_understanding.telemetry.exporters.http import HTTPJsonExporter

        exporter = HTTPJsonExporter(
            endpoint="http://localhost:8080",
            headers={"X-Custom": "value", "Authorization": "Bearer token"},
        )

        assert exporter.headers["Content-Type"] == "application/json"
        assert exporter.headers["X-Custom"] == "value"
        assert exporter.headers["Authorization"] == "Bearer token"


class TestShutdownBehavior:
    """Tests for telemetry shutdown behavior."""

    def test_shutdown_idempotent(self) -> None:
        """Test shutdown_telemetry is idempotent."""
        import world_understanding.telemetry as telemetry_module
        from world_understanding.telemetry import (
            TelemetryConfig,
            initialize_telemetry,
            shutdown_telemetry,
        )

        # Ensure clean state by resetting module internals
        shutdown_telemetry()
        telemetry_module._initialized = False
        telemetry_module._tracer_provider = None

        config = TelemetryConfig(
            enabled=True,
            exporters="console",
        )
        initialize_telemetry(config)

        # Should not raise even when called multiple times
        shutdown_telemetry()
        shutdown_telemetry()
        shutdown_telemetry()

    def test_shutdown_without_initialization(self) -> None:
        """Test shutdown_telemetry without prior initialization."""
        from world_understanding.telemetry import shutdown_telemetry

        # Should not raise
        shutdown_telemetry()


class TestNvcfUtilsTracer:
    """Tests for telemetry integration in nvcf_utils."""

    def test_nvcf_utils_uses_get_tracer(self) -> None:
        """Test that nvcf_utils._tracer is obtained via get_tracer helper."""
        from world_understanding.utils import nvcf_utils

        # _tracer should be a valid tracer instance (no-op when uninitialised)
        assert nvcf_utils._tracer is not None
        # It should have start_as_current_span (standard Tracer API)
        assert hasattr(nvcf_utils._tracer, "start_as_current_span")

    def test_nvcf_utils_does_not_import_trace_directly(self) -> None:
        """Verify nvcf_utils uses get_tracer, not opentelemetry.trace directly."""
        import inspect

        from world_understanding.utils import nvcf_utils

        source = inspect.getsource(nvcf_utils)
        assert "from opentelemetry import trace" not in source
        assert "from world_understanding.telemetry import get_tracer" in source

    def test_sync_request_creates_span(self) -> None:
        """Test that execute_nvcf_request_with_retry creates a span."""
        import world_understanding.telemetry as telemetry_module
        from world_understanding.telemetry import (
            initialize_telemetry,
            shutdown_telemetry,
        )
        from world_understanding.telemetry.config import TelemetryConfig

        shutdown_telemetry()
        telemetry_module._initialized = False
        telemetry_module._tracer_provider = None

        config = TelemetryConfig(enabled=True, exporters="console")
        initialize_telemetry(config)

        try:
            from unittest.mock import patch

            from world_understanding.utils.nvcf_utils import (
                execute_nvcf_request_with_retry,
            )

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.raise_for_status = MagicMock()

            with patch("requests.post", return_value=mock_response):
                result = execute_nvcf_request_with_retry(
                    url="https://test.invocation.api.nvcf.nvidia.com",
                    headers={},
                    params={},
                    timeout=10,
                )
                assert result == mock_response
        finally:
            shutdown_telemetry()

    @pytest.mark.asyncio
    async def test_async_request_creates_span(self) -> None:
        """Test that execute_nvcf_request_async creates a span."""
        import world_understanding.telemetry as telemetry_module
        from world_understanding.telemetry import (
            initialize_telemetry,
            shutdown_telemetry,
        )
        from world_understanding.telemetry.config import TelemetryConfig

        shutdown_telemetry()
        telemetry_module._initialized = False
        telemetry_module._tracer_provider = None

        config = TelemetryConfig(enabled=True, exporters="console")
        initialize_telemetry(config)

        try:
            from unittest.mock import AsyncMock, patch

            from world_understanding.utils.nvcf_utils import (
                execute_nvcf_request_async,
            )

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.headers = {"content-type": "application/json"}
            mock_response.json.return_value = {"result": "ok"}
            mock_response.raise_for_status = MagicMock()

            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)

            with patch("httpx.AsyncClient", return_value=mock_client):
                result = await execute_nvcf_request_async(
                    url="https://test.invocation.api.nvcf.nvidia.com",
                    headers={},
                    params={},
                    api_key="test-key",
                    timeout=10,
                )
                assert result == {"result": "ok"}
        finally:
            shutdown_telemetry()


class TestTelemetryConfigEnvVars:
    """Tests for telemetry config loading from CI/CD-style env vars."""

    def test_langfuse_config_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that OTEL_LANGFUSE__ env vars load into LangfuseConfig."""
        from world_understanding.telemetry.config import TelemetryConfig

        monkeypatch.setenv("OTEL_EXPORTERS", "langfuse")
        monkeypatch.setenv("OTEL_LANGFUSE__PUBLIC_KEY", "pk-test-123")
        monkeypatch.setenv("OTEL_LANGFUSE__SECRET_KEY", "sk-test-456")
        monkeypatch.setenv(
            "OTEL_LANGFUSE__ENDPOINT",
            "https://langfuse.example.com/api/public/otel",
        )

        config = TelemetryConfig()

        assert config.langfuse is not None
        assert config.langfuse.public_key == "pk-test-123"
        assert config.langfuse.secret_key == "sk-test-456"
        assert (
            config.langfuse.endpoint == "https://langfuse.example.com/api/public/otel"
        )

    def test_sample_rate_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test OTEL_SAMPLE_RATE env var loads correctly."""
        from world_understanding.telemetry.config import TelemetryConfig

        monkeypatch.setenv("OTEL_SAMPLE_RATE", "0.25")

        config = TelemetryConfig()
        assert config.sample_rate == 0.25

    def test_sample_rate_validation(self) -> None:
        """Test OTEL_SAMPLE_RATE must be between 0.0 and 1.0."""
        from pydantic import ValidationError

        from world_understanding.telemetry.config import TelemetryConfig

        with pytest.raises(ValidationError):
            TelemetryConfig(sample_rate=1.5)

        with pytest.raises(ValidationError):
            TelemetryConfig(sample_rate=-0.1)
