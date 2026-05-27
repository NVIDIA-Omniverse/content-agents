# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Semantic conventions and span attributes for MAA telemetry.

This module defines standardized attribute names for spans to ensure
consistent telemetry data across the MAA (World Understanding) project.
It includes both MAA-specific attributes and OpenTelemetry GenAI
semantic conventions.
"""

from __future__ import annotations


class MAAttributes:
    """MAA-specific span attributes for the World Understanding project.

    These attributes follow the 'maa.' namespace convention and provide
    standardized keys for common operations in the MAA pipeline.

    Attributes:
        PIPELINE_NAME: Name of the pipeline being executed.
        PIPELINE_SESSION_ID: Unique session identifier for the pipeline run.
        PIPELINE_STEP_NAME: Name of the current pipeline step.
        PIPELINE_STEP_INDEX: Zero-based index of the current step.
        PIPELINE_STEP_STATUS: Status of the pipeline step (e.g., 'success', 'error').
        PIPELINE_TOTAL_STEPS: Total number of steps in the pipeline.
        TOOL_NAME: Name of the tool being invoked.
        TOOL_DESCRIPTION: Description of the tool's purpose.
        VLM_BACKEND: Vision-Language Model backend identifier.
        VLM_IMAGE_COUNT: Number of images processed by VLM.
        NVCF_FUNCTION_ID: NVIDIA Cloud Functions function identifier.
        NVCF_RETRY_COUNT: Number of retry attempts for NVCF calls.
        S3_BUCKET: S3 bucket name for storage operations.
        S3_KEY: S3 object key for storage operations.
        S3_OPERATION: Type of S3 operation (e.g., 'get', 'put', 'delete').
    """

    # Pipeline attributes
    PIPELINE_NAME = "maa.pipeline.name"
    PIPELINE_SESSION_ID = "maa.pipeline.session_id"
    PIPELINE_STEP_NAME = "maa.pipeline.step.name"
    PIPELINE_STEP_INDEX = "maa.pipeline.step.index"
    PIPELINE_STEP_STATUS = "maa.pipeline.step.status"
    PIPELINE_TOTAL_STEPS = "maa.pipeline.total_steps"

    # Per-step telemetry attributes
    PIPELINE_STEP_DURATION_SECONDS = "maa.pipeline.step.duration_seconds"
    PIPELINE_STEP_ERROR = "maa.pipeline.step.error"

    # Asset-level attributes
    ASSET_FILENAME = "maa.asset.filename"
    ASSET_FILE_SIZE_BYTES = "maa.asset.file_size_bytes"
    ASSET_FILE_EXTENSION = "maa.asset.file_extension"

    # Pipeline completion/telemetry attributes
    PIPELINE_USER_EMAIL = "maa.pipeline.user_email"
    PIPELINE_STATUS = "maa.pipeline.status"
    PIPELINE_DURATION_SECONDS = "maa.pipeline.duration_seconds"
    PIPELINE_PRIM_COUNT = "maa.pipeline.prim_count"
    PIPELINE_PRIMS_PROCESSED = "maa.pipeline.prims_processed"
    PIPELINE_IMAGES_GENERATED = "maa.pipeline.images_generated"
    PIPELINE_PREDICTIONS_MADE = "maa.pipeline.predictions_made"
    PIPELINE_MATERIALS_APPLIED = "maa.pipeline.materials_applied"
    PIPELINE_VLM_MODEL = "maa.pipeline.vlm_model"

    # Prim clustering attributes
    CLUSTERING_ENABLED = "maa.clustering.enabled"
    CLUSTER_EMBEDDING_BACKEND = "maa.clustering.embedding_backend"
    CLUSTER_EMBEDDING_MODEL = "maa.clustering.embedding_model"
    CLUSTER_TOTAL_PRIMS = "maa.clustering.total_prims"
    CLUSTER_COUNT = "maa.clustering.cluster_count"
    CLUSTER_REPRESENTATIVE_COUNT = "maa.clustering.representative_count"
    CLUSTER_REDUCTION_PERCENT = "maa.clustering.reduction_percent"
    CLUSTER_MULTI_MEMBER_COUNT = "maa.clustering.multi_member_count"
    CLUSTER_SINGLETON_COUNT = "maa.clustering.singleton_count"
    CLUSTER_MAX_SIZE = "maa.clustering.max_size"
    CLUSTER_CAPPED_COUNT = "maa.clustering.capped_count"

    # Tool attributes
    TOOL_NAME = "maa.tool.name"
    TOOL_DESCRIPTION = "maa.tool.description"

    # VLM (Vision-Language Model) attributes
    VLM_BACKEND = "maa.vlm.backend"
    VLM_IMAGE_COUNT = "maa.vlm.image_count"

    # NVCF (NVIDIA Cloud Functions) attributes
    NVCF_FUNCTION_ID = "maa.nvcf.function_id"
    NVCF_RETRY_COUNT = "maa.nvcf.retry_count"

    # S3 storage attributes
    S3_BUCKET = "maa.s3.bucket"
    S3_KEY = "maa.s3.key"
    S3_OPERATION = "maa.s3.operation"

    # Langfuse trace-level attributes
    # These map to Langfuse's native user/session fields for dashboard filtering.
    # See: https://langfuse.com/integrations/native/opentelemetry
    LANGFUSE_USER_ID = "langfuse.user.id"
    LANGFUSE_SESSION_ID = "langfuse.session.id"

    # Langfuse trace metadata attributes
    # These appear as filterable top-level keys in the Langfuse dashboard.
    LANGFUSE_META_ASSET_FILENAME = "langfuse.trace.metadata.asset_filename"
    LANGFUSE_META_ASSET_FILE_SIZE = "langfuse.trace.metadata.asset_file_size_bytes"
    LANGFUSE_META_ASSET_FILE_EXT = "langfuse.trace.metadata.asset_file_extension"
    LANGFUSE_META_VLM_MODEL = "langfuse.trace.metadata.vlm_model"


class GenAIAttributes:
    """OpenTelemetry GenAI semantic conventions.

    These attributes follow the OpenTelemetry semantic conventions for
    Generative AI operations. They provide standardized keys for tracking
    LLM/GenAI requests, responses, and token usage.

    Reference: https://opentelemetry.io/docs/specs/semconv/gen-ai/

    Attributes:
        SYSTEM: The GenAI system/provider (e.g., 'openai', 'anthropic').
        OPERATION_NAME: Type of GenAI operation (e.g., 'chat', 'completion').
        REQUEST_MODEL: Model identifier used for the request.
        REQUEST_TEMPERATURE: Sampling temperature for the request.
        REQUEST_MAX_TOKENS: Maximum tokens allowed in the response.
        RESPONSE_MODEL: Model identifier that generated the response.
        RESPONSE_ID: Unique identifier for the response.
        USAGE_INPUT_TOKENS: Number of tokens in the input/prompt.
        USAGE_OUTPUT_TOKENS: Number of tokens in the output/completion.
        USAGE_TOTAL_TOKENS: Total tokens used (input + output).
        RESPONSE_FINISH_REASON: Reason for response completion (e.g., 'stop', 'length').
    """

    # System and operation
    SYSTEM = "gen_ai.system"
    OPERATION_NAME = "gen_ai.operation.name"

    # Request attributes
    REQUEST_MODEL = "gen_ai.request.model"
    REQUEST_TEMPERATURE = "gen_ai.request.temperature"
    REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"

    # Response attributes
    RESPONSE_MODEL = "gen_ai.response.model"
    RESPONSE_ID = "gen_ai.response.id"
    RESPONSE_FINISH_REASON = "gen_ai.response.finish_reason"

    # Token usage attributes
    USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
    USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
    USAGE_TOTAL_TOKENS = "gen_ai.usage.total_tokens"
