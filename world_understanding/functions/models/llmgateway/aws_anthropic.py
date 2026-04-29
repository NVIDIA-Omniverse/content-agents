# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import abc
import base64
import logging
from typing import Any

import botocore.session
from botocore.config import Config
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
)
from pydantic import BaseModel, ConfigDict

from . import auth

logger = logging.getLogger(__name__)


class AuthBase(BaseModel, abc.ABC):
    """Base class for authentication implementations."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abc.abstractmethod
    def validate_auth(self):
        pass


class ConverseLLMGatewayAuthBase(AuthBase):
    """
    Performs API proxy token fetch for the LLM Gateway service, used with the
    AWS Converse-compatible endpoints.
    """

    cred_dict: dict | None = None
    token_data: dict | None = None

    def validate_auth(self):
        """Validates and refreshes the auth token as needed."""
        self.token_data = auth.validate_token_data(self.token_data)
        if not self.token_data:
            self.token_data = auth.get_oauth_token_data(cred_dict=self.cred_dict)
        if not self.token_data:
            logger.error("Failed to acquire token data!")
            return


def _is_data_url(url: str) -> bool:
    return isinstance(url, str) and url.startswith("data:")


def _parse_data_url(data_url: str) -> tuple[str, bytes]:
    """Parse a data URL and return (format, raw_bytes).

    Example: data:image/png;base64,AAAA...
    format -> "png" | "jpeg" | etc.
    """
    try:
        header, b64_payload = data_url.split(",", 1)
        # header like: data:image/png;base64
        mime_part = header[len("data:") :]
        mime_type = mime_part.split(";", 1)[0] if ";" in mime_part else mime_part
        subtype = mime_type.split("/")[-1] if "/" in mime_type else "png"
        fmt = "jpeg" if subtype == "jpg" else subtype
        try:
            raw_bytes = base64.b64decode(b64_payload, validate=True)
        except Exception:
            raw_bytes = base64.b64decode(b64_payload)
        return fmt, raw_bytes
    except Exception:
        # Fallback
        return "png", b""


def _split_model_kwargs(
    model_kwargs: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Split model kwargs into (inferenceConfig, additionalModelRequestFields).
    """
    if not model_kwargs:
        return {}, {}

    inference_keys = {"temperature", "top_p", "max_tokens", "stop_sequences"}
    inference_config: dict[str, Any] = {}
    additional_fields: dict[str, Any] = {}

    for k, v in model_kwargs.items():
        # The backend may already manage anthropic_version; sending it can conflict.
        if k == "anthropic_version":
            continue
        if k in inference_keys:
            # map to Converse API names where they differ
            if k == "max_tokens":
                inference_config["maxTokens"] = v
            elif k == "top_p":
                inference_config["topP"] = v
            elif k == "stop_sequences":
                inference_config["stopSequences"] = v
            else:
                inference_config[k] = v
        else:
            additional_fields[k] = v

    return inference_config, additional_fields


def _create_bedrock_client_with_gateway(
    proxy_url: str, auth_base: ConverseLLMGatewayAuthBase, aws_region: str
):
    session = botocore.session.Session()
    custom_config = Config(
        signature_version=None,
        region_name=aws_region,
    )
    client = session.create_client(
        "bedrock-runtime",
        endpoint_url=proxy_url,
        config=custom_config,
        aws_access_key_id="dummy",
        aws_secret_access_key="dummy",
        region_name=aws_region,
    )

    # Inject Bearer token into every request
    original_send = client._endpoint.http_session.send

    def _custom_send(request, *args, **kwargs):
        auth_base.validate_auth()
        access_token = (
            auth_base.token_data.get("access_token") if auth_base.token_data else None
        )
        if access_token:
            request.headers["Authorization"] = f"Bearer {access_token}"
        request.headers["dataClassification"] = "sensitive"
        return original_send(request, *args, **kwargs)

    client._endpoint.http_session.send = _custom_send
    return client


class ChatConverseAnthropic_LLMGateway(ChatBedrockConverse):
    """
    LangChain-compatible chat model that calls AWS Converse API through NVIDIA LLM Gateway
    using a botocore client (no AWS credentials required; bearer token auth is injected).

    Notes:
    - OAuth from `auth.py` is used to fetch/refresh a bearer token (SSA).
    - Requests are made via a botocore Bedrock client to the Gateway endpoint.
    - Supports Anthropic thinking content: pass `thinking`/`anthropic_version` via `model_kwargs`.
    - Image input: pass base64 data URLs only. Supported forms inside HumanMessage.content:
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
        {"type": "image", "data_url": "data:image/png;base64,..."}
    """

    def __init__(
        self,
        *args,
        proxy_base_url: str = "https://prod.api.nvidia.com/llm/v1/aws",
        aws_region: str = "us-east-2",
        cred_dict: dict[str, Any] | None = None,
        cred_fields: list[str] | None = None,
        env_prefix: str | None = None,
        cred_file_url: str | None = None,
        model_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        # Remove unsupported flags that could leak into requests
        kwargs.pop("streaming", None)
        # Initialize credentials
        if not cred_dict:
            cred_dict = auth.get_credentials(
                cred_fields=cred_fields,
                env_prefix=env_prefix,
                file_url=cred_file_url,
            )
            if cred_dict is None:
                raise ValueError(
                    "Credentials not found. Provide cred_dict or cred_fields+env_prefix/cred_file_url."
                )
            # Override scope as this is the only applicable scope really
            cred_dict["scope"] = "awsanthropic-readwrite"

        auth_base = ConverseLLMGatewayAuthBase(cred_dict=cred_dict)

        # Create a botocore client that points to the LLM Gateway and injects Bearer token
        bedrock_client = _create_bedrock_client_with_gateway(
            proxy_base_url, auth_base, aws_region
        )

        # Default model id if not provided
        if "model_id" not in kwargs:
            raise ValueError(
                "'model_id' not found. Please specify model, e.g. 'model_id=\"us.anthropic.claude-sonnet-4-20250514-v1:0\"'"
            )

        # Promote common inference params out of model_kwargs to avoid warnings
        if model_kwargs:
            for src_key, dst_key in (
                ("temperature", "temperature"),
                ("top_p", "top_p"),
                ("max_tokens", "max_tokens"),
                ("stop_sequences", "stop_sequences"),
            ):
                if src_key in model_kwargs and dst_key not in kwargs:
                    kwargs[dst_key] = model_kwargs.pop(src_key)

            # Remove local-only or conflicting keys
            for k in [
                "include_thinking",
                "thinking_begin_text",
                "thinking_end_text",
            ]:
                if k in model_kwargs:
                    model_kwargs.pop(k, None)

            # Split remaining into Converse-specific fields
            if model_kwargs:
                inference_config, additional_fields = _split_model_kwargs(model_kwargs)
                if inference_config:
                    kwargs.setdefault("inference_config", inference_config)
                if additional_fields:
                    kwargs.setdefault(
                        "additional_model_request_fields", additional_fields
                    )

        kwargs["client"] = bedrock_client
        kwargs["provider"] = "anthropic"

        # Strip local-only display knobs so LC doesn't forward them into model_kwargs
        local_include_thinking = kwargs.pop("include_thinking", None)
        local_thinking_begin = kwargs.pop("thinking_begin_text", None)
        local_thinking_end = kwargs.pop("thinking_end_text", None)

        super().__init__(*args, **kwargs)

        # Keep local-only display options for potential client-side formatting (not sent to API)
        object.__setattr__(
            self,
            "_include_thinking",
            bool(local_include_thinking)
            if local_include_thinking is not None
            else False,
        )
        object.__setattr__(
            self, "_thinking_begin_text", local_thinking_begin or "<thinking>\n"
        )
        object.__setattr__(
            self, "_thinking_end_text", local_thinking_end or "</thinking>\n"
        )

    # Message preprocessing to support data URLs in a model-agnostic way
    def _preprocess_messages(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        processed: list[BaseMessage] = []
        for msg in messages:
            content = getattr(msg, "content", None)
            if isinstance(content, list):
                new_parts: list[dict[str, Any]] = []
                for item in content:
                    if (
                        isinstance(item, dict)
                        and item.get("type") == "image_url"
                        and "image_url" in item
                    ):
                        raw = item["image_url"]
                        if isinstance(raw, dict):
                            raw = raw.get("url")
                        if _is_data_url(raw):
                            fmt, b64_payload = _parse_data_url(raw)
                            new_parts.append(
                                {
                                    "image": {
                                        "format": fmt,
                                        "source": {"bytes": b64_payload},
                                    }
                                }
                            )
                        else:
                            new_parts.append(item)
                    elif (
                        isinstance(item, dict)
                        and item.get("type") == "image"
                        and "data_url" in item
                    ):
                        raw = item.get("data_url")
                        if _is_data_url(raw):
                            fmt, b64_payload = _parse_data_url(raw)
                            new_parts.append(
                                {
                                    "image": {
                                        "format": fmt,
                                        "source": {"bytes": b64_payload},
                                    }
                                }
                            )
                        else:
                            new_parts.append(item)
                    else:
                        new_parts.append(item)
                try:
                    msg = msg.__class__(
                        content=new_parts,
                        additional_kwargs=getattr(msg, "additional_kwargs", {}),
                    )
                except Exception:
                    pass
            processed.append(msg)
        return processed

    # Response post-processing to convert content lists to printable strings
    def _process_content_list(
        self, content_list: list[dict[str, Any]], in_thinking_region: bool = False
    ) -> tuple[str, bool]:
        content_str = ""
        for item in content_list:
            if isinstance(item, dict):
                item_type = item.get("type")
                if item_type == "reasoning_content" and self._include_thinking:
                    data = item.get("reasoning_content", {})
                    text = data.get("text", "")
                    if not in_thinking_region:
                        content_str += self._thinking_begin_text
                        in_thinking_region = True
                    content_str += text
                elif item_type == "text":
                    if in_thinking_region and self._include_thinking:
                        content_str += self._thinking_end_text
                        in_thinking_region = False
                    content_str += item.get("text", "")
        return content_str, in_thinking_region

    def _process_streaming_chunk(
        self, chunk: AIMessageChunk, in_thinking_region: bool
    ) -> tuple[list[AIMessageChunk], bool]:
        chunks_to_yield: list[AIMessageChunk] = []

        if isinstance(chunk, AIMessageChunk) and isinstance(chunk.content, list):
            # Convert list content into a single string with optional thinking tags
            content_str, in_thinking_region = self._process_content_list(
                chunk.content, in_thinking_region
            )
            chunks_to_yield.append(
                AIMessageChunk(
                    content=content_str,
                    additional_kwargs=getattr(chunk, "additional_kwargs", {}),
                    response_metadata=getattr(chunk, "response_metadata", {}),
                    id=getattr(chunk, "id", None),
                    usage_metadata=getattr(chunk, "usage_metadata", None)
                    if hasattr(chunk, "usage_metadata")
                    else None,
                )
            )
        else:
            # For non-list chunks, pass through. If at final empty chunk, close thinking region.
            try:
                stop_reason = None
                if hasattr(chunk, "response_metadata") and isinstance(
                    chunk.response_metadata, dict
                ):
                    stop_reason = chunk.response_metadata.get(
                        "stopReason"
                    ) or chunk.response_metadata.get("stop_reason")
                if (
                    in_thinking_region
                    and self._include_thinking
                    and getattr(chunk, "content", "") == ""
                    and stop_reason
                ):
                    chunks_to_yield.append(
                        AIMessageChunk(
                            content=self._thinking_end_text,
                            additional_kwargs={},
                            response_metadata={},
                            id=getattr(chunk, "id", None),
                        )
                    )
                    in_thinking_region = False
            except Exception:
                pass
            chunks_to_yield.append(chunk)

        return chunks_to_yield, in_thinking_region

    def _process_response_content(self, response: AIMessage) -> AIMessage:
        if hasattr(response, "content") and isinstance(response.content, list):
            content_str, in_thinking_region = self._process_content_list(
                response.content
            )
            if in_thinking_region and self._include_thinking:
                content_str += self._thinking_end_text
            response.content = content_str
        return response

    # Override generation/streaming to preprocess images and post-process responses
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ):
        return super()._generate(
            self._preprocess_messages(messages),
            stop=stop,
            run_manager=run_manager,
            **kwargs,
        )

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ):
        return await super()._agenerate(
            self._preprocess_messages(messages),
            stop=stop,
            run_manager=run_manager,
            **kwargs,
        )

    def _stream(
        self, messages: list[BaseMessage], stop: list[str] | None = None, **kwargs: Any
    ):
        stream = super()._stream(
            self._preprocess_messages(messages), stop=stop, **kwargs
        )
        in_thinking_region = False
        for chunk in stream:
            processed_chunks, in_thinking_region = self._process_streaming_chunk(
                chunk, in_thinking_region
            )
            yield from processed_chunks

    async def _astream(
        self, messages: list[BaseMessage], stop: list[str] | None = None, **kwargs: Any
    ):
        stream = super()._astream(
            self._preprocess_messages(messages), stop=stop, **kwargs
        )
        in_thinking_region = False
        async for chunk in stream:
            processed_chunks, in_thinking_region = self._process_streaming_chunk(
                chunk, in_thinking_region
            )
            for processed_chunk in processed_chunks:
                yield processed_chunk

    # Public streaming wrappers to ensure clean text chunks even if base stream bypasses _stream
    def stream(
        self, messages: list[BaseMessage], stop: list[str] | None = None, **kwargs: Any
    ):
        upstream = super().stream(messages, stop=stop, **kwargs)
        in_thinking_region = False
        for chunk in upstream:
            processed_chunks, in_thinking_region = self._process_streaming_chunk(
                chunk, in_thinking_region
            )
            yield from processed_chunks

    async def astream(
        self, messages: list[BaseMessage], stop: list[str] | None = None, **kwargs: Any
    ):
        upstream = super().astream(messages, stop=stop, **kwargs)
        in_thinking_region = False
        async for chunk in upstream:
            processed_chunks, in_thinking_region = self._process_streaming_chunk(
                chunk, in_thinking_region
            )
            for processed_chunk in processed_chunks:
                yield processed_chunk

    # Public invoke wrappers to post-process content into printable strings
    def invoke(self, *args: Any, **kwargs: Any) -> AIMessage:
        response = super().invoke(*args, **kwargs)
        return self._process_response_content(response)

    async def ainvoke(self, *args: Any, **kwargs: Any) -> AIMessage:
        response = await super().ainvoke(*args, **kwargs)
        return self._process_response_content(response)

    def _invoke(self, *args: Any, **kwargs: Any) -> AIMessage:
        response = super()._invoke(*args, **kwargs)
        return self._process_response_content(response)

    async def _ainvoke(self, *args: Any, **kwargs: Any) -> AIMessage:
        response = await super()._ainvoke(*args, **kwargs)
        return self._process_response_content(response)
