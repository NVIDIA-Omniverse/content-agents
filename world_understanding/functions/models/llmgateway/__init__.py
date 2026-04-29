# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LLM Gateway wrappers and auth utilities.

Project-agnostic wrappers for Azure OpenAI and AWS Anthropic (via Bedrock
or Converse), plus shared OAuth-based auth helpers.
"""

from .aws_anthropic import ChatConverseAnthropic_LLMGateway
from .azure_openai import (
    AzureChatOpenAI_LLMGateway,
    AzureOpenAIEmbeddings_LLMGateway,
)

__all__ = [
    "AzureChatOpenAI_LLMGateway",
    "AzureOpenAIEmbeddings_LLMGateway",
    "ChatConverseAnthropic_LLMGateway",
]
