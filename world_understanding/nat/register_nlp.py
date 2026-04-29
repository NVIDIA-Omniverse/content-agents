# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import logging
from typing import Any

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class ChatToolConfig(FunctionBaseConfig, name="chat"):  # type: ignore[call-arg]
    pass


@register_function(config_type=ChatToolConfig)  # type: ignore[misc]
async def chat(config: ChatToolConfig, builder: Builder) -> Any:
    import json

    from world_understanding.functions.models.chat_models import create_chat_model
    from world_understanding.functions.nlp.chat import (
        generate_chat_response as generate_chat_response_fn,
    )

    async def _chat(
        prompt: str,
        backend: str = "echo",
        model: str = "",
        system_prompt: str = "You are a helpful AI assistant.",
    ) -> str:
        """
        Generate a chat response using various LLM backends.

        Args:
            prompt: The user's question or prompt
            backend: Chat backend to use (echo, nvidia, azureopenai)
            model: Model name (optional, uses backend default if empty)
            system_prompt: System instructions for the model

        Returns:
            JSON string with the chat response
        """
        try:
            # Create chat model instance
            chat_model = await asyncio.to_thread(
                create_chat_model, backend=backend, model=model
            )

            # Call the portable function
            result = await asyncio.to_thread(
                generate_chat_response_fn,
                chat_model=chat_model,
                prompt=prompt,
                system_prompt=system_prompt,
            )

            # Check for errors
            if "error" in result:
                return f"Error: {result['error']}"

            # Format the result
            formatted_result = {
                "response": result["response"],
                "backend": backend,
                "model": model if model else "default",
            }

            return json.dumps(formatted_result, indent=2)

        except ValueError as e:
            return f"Invalid backend or configuration: {str(e)}"
        except Exception as e:
            return f"Failed to generate response: {str(e)}"

    # Create a Generic NAT tool that can be used with any supported LLM framework
    yield FunctionInfo.from_fn(
        _chat,
        description=(
            "Generate text responses using various LLM backends including "
            "Echo (for testing), NVIDIA AI Foundation models, and Azure "
            "OpenAI. Useful for answering questions, generating content, "
            "or having conversations."
        ),
    )
