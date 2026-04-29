# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Router agent implementation."""

import asyncio
import json
import logging
from typing import Any

from world_understanding.agentic.base import BaseAgent
from world_understanding.functions.models.chat_models import create_chat_model
from world_understanding.functions.nlp.chat import generate_chat_response
from world_understanding.tools.base import get_tool_registry
from world_understanding.utils.object_store import ObjectStore

logger = logging.getLogger(__name__)


class RouterAgent(BaseAgent):
    """
    Router agent that analyzes tasks and routes them to appropriate
    tools or agents using LLM-based decision making.

    This agent uses a language model to understand the task and select
    the most appropriate tools based on their descriptions and capabilities.
    """

    def __init__(
        self,
        tools: dict[str, Any] | None = None,
        chat_model_config: dict[str, Any] | None = None,
        name: str = "RouterAgent",
        description: str = "Routes tasks to appropriate tools using LLM analysis",
    ):
        """Initialize the router agent with tools and chat model configuration.

        Args:
            tools: Dictionary of available tools (defaults to global registry)
            chat_model_config: Configuration for the chat model
                Expected keys: service, model_name, api_key, temperature, max_tokens
            name: Agent name
            description: Agent description
        """
        super().__init__(name, description)
        self.tools = tools or get_tool_registry()
        self.chat_model_config = chat_model_config or {}

    def can_handle(self, task: str) -> bool:
        """
        Check if this router can handle the task.

        The router can handle any task by default, as it can always
        fall back to analyzing available tools.
        """
        return True

    def analyze_task_with_llm(
        self, task: str, available_tools: list[dict[str, Any]]
    ) -> list[str]:
        """
        Use LLM to analyze the task and select appropriate tools.

        Args:
            task: Task description
            available_tools: List of available tools with their metadata

        Returns:
            List of selected tool names
        """
        # Prepare tool descriptions for the LLM
        tool_descriptions = []
        for tool in available_tools:
            desc = f"- {tool['name']}: {tool['description']}"
            if tool.get("tags"):
                desc += f" (tags: {', '.join(tool['tags'])})"
            tool_descriptions.append(desc)

        # Create the prompt
        system_prompt = """You are a routing agent that analyzes tasks and selects the most appropriate tools to complete them.

Given a task and a list of available tools, select the tools that would be most helpful for completing the task.
Return ONLY a JSON object with a 'tools' key containing a list of tool names.
Do not include any explanatory text, markdown formatting, or code blocks.

Example response:
{
    "tools": ["tool1", "tool2"],
    "reasoning": "Brief explanation of why these tools were selected"
}

Guidelines:
- Select only the tools that are directly relevant to the task
- Consider the tool descriptions and tags when making your selection
- If no tools seem relevant, return an empty list
- Provide brief reasoning for your selection"""

        user_prompt = f"""Task: {task}

Available tools:
{chr(10).join(tool_descriptions)}

Select the appropriate tools for this task."""

        try:
            # Create chat model
            chat_model = create_chat_model(
                backend=self.chat_model_config.get("service", "echo"),
                api_key=self.chat_model_config.get("api_key"),
                model=self.chat_model_config.get("model_name"),
            )

            # Generate response using chat model
            response = generate_chat_response(
                chat_model=chat_model,
                prompt=user_prompt,
                system_prompt=system_prompt,
            )

            # Parse JSON from response
            response_text = response.get("response", "")

            # Find JSON in the response
            start_idx = response_text.find("{")
            end_idx = response_text.rfind("}") + 1

            if start_idx >= 0 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                result = json.loads(json_str)

                selected_tools = result.get("tools", [])
                reasoning = result.get("reasoning", "")

                if reasoning:
                    logger.info(f"LLM reasoning: {reasoning}")

                # Validate that selected tools exist
                valid_tools = []
                tool_names = {tool["name"] for tool in available_tools}
                for tool in selected_tools:
                    if tool in tool_names:
                        valid_tools.append(tool)
                    else:
                        logger.warning(f"LLM selected non-existent tool: {tool}")

                return valid_tools
            else:
                logger.error("No JSON found in LLM response")
                return []

        except Exception as e:
            logger.error(f"Error during LLM analysis: {e}")
            return []

    def analyze_and_generate_inputs(
        self, task: str, tool_name: str, tool_spec: Any
    ) -> dict[str, Any]:
        """
        Use LLM to generate appropriate inputs for a tool based on the task.

        Args:
            task: Task description
            tool_name: Name of the tool
            tool_spec: Tool specification with input schema

        Returns:
            Dictionary of generated inputs for the tool
        """
        # Get input model schema if available
        input_schema = {}
        if hasattr(tool_spec, "input_model") and tool_spec.input_model:
            # Convert Pydantic model to JSON schema
            try:
                input_schema = tool_spec.input_model.model_json_schema()
            except Exception:
                input_schema = {"description": "Tool input parameters"}

        # Create the prompt
        system_prompt = """You are an intelligent task analyzer that generates tool inputs.

Given a task and a tool's input schema, generate appropriate input parameters that would help complete the task.
Return ONLY a valid JSON object containing the tool inputs.
Do not include any explanatory text, markdown formatting, or code blocks.

Guidelines:
- Generate values for all required fields in the schema
- For optional fields, only include them if they would be helpful for the task
- Use sensible defaults when the task doesn't specify exact values
- Ensure all values match the expected types and constraints in the schema
- Return ONLY the JSON object, nothing else"""

        user_prompt = f"""Task: {task}

Tool: {tool_name}

Input Schema:
{json.dumps(input_schema, indent=2)}

Generate appropriate inputs for this tool to help complete the task."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            # Combine messages into a single prompt
            user_prompt = messages[-1]["content"] if messages else ""
            system_prompt = (
                messages[0]["content"]
                if len(messages) > 1
                else "You are an intelligent task analyzer."
            )

            # Create chat model
            chat_model = create_chat_model(
                backend=self.chat_model_config.get("service", "echo"),
                api_key=self.chat_model_config.get("api_key"),
                model=self.chat_model_config.get("model_name"),
            )

            response = generate_chat_response(
                chat_model=chat_model,
                prompt=user_prompt,
                system_prompt=system_prompt,
            )

            response_text = response.get("response", "")

            # Extract JSON
            start_idx = response_text.find("{")
            end_idx = response_text.rfind("}") + 1

            if start_idx >= 0 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                inputs = json.loads(json_str)
                logger.info(f"LLM generated inputs for {tool_name}: {inputs}")
                return inputs
            else:
                logger.error(
                    f"Failed to extract JSON from LLM response for {tool_name}"
                )
                return {}

        except Exception as e:
            logger.error(f"Error generating inputs with LLM: {e}")
            return {}

    def select_tools(self, task: str) -> list[str]:
        """
        Select tools based on task analysis.

        Args:
            task: Task description

        Returns:
            List of tool names to use
        """
        # If we have chat model config, use LLM
        if self.chat_model_config:
            # Prepare available tools info
            available_tools = []
            for tool_name, tool in self.tools.items():
                tool_spec = tool.spec if hasattr(tool, "spec") else None
                if tool_spec:
                    available_tools.append(
                        {
                            "name": tool_name,
                            "description": tool_spec.description,
                            "tags": tool_spec.tags,
                        }
                    )

            # Use LLM to select tools
            selected = self.analyze_task_with_llm(task, available_tools)

            if selected:
                logger.info(f"LLM selected tools: {selected}")
                return selected
            else:
                logger.warning("LLM returned no tools for the task")
                return []
        else:
            # Fallback to simple keyword matching
            selected_tools = []
            task_lower = task.lower()

            for tool_name, tool in self.tools.items():
                tool_spec = tool.spec if hasattr(tool, "spec") else None
                if tool_spec:
                    # Check if any tag matches task keywords
                    for tag in tool_spec.tags:
                        if tag.lower() in task_lower:
                            selected_tools.append(tool_name)
                            break

            return selected_tools

    def execute_tool(
        self,
        tool_name: str,
        inputs: dict[str, Any],
        object_store: ObjectStore | None = None,
    ) -> dict[str, Any]:
        """
        Execute a tool with the given inputs synchronously.

        Args:
            tool_name: Name of the tool to execute
            inputs: Input parameters for the tool
            object_store: Optional object store for artifacts

        Returns:
            Tool execution results
        """
        return asyncio.run(self.aexecute_tool(tool_name, inputs, object_store))

    async def aexecute_tool(
        self,
        tool_name: str,
        inputs: dict[str, Any],
        object_store: ObjectStore | None = None,
    ) -> dict[str, Any]:
        """
        Execute a tool with the given inputs asynchronously.

        Args:
            tool_name: Name of the tool to execute
            inputs: Input parameters for the tool
            object_store: Optional object store for artifacts

        Returns:
            Tool execution results
        """
        if tool_name not in self.tools:
            raise ValueError(f"Tool '{tool_name}' not found")

        tool = self.tools[tool_name]

        # Execute the tool asynchronously
        try:
            result = await tool.arun(inputs)

            # Store result in object store if provided
            if object_store:
                result_key = f"{tool_name}_result"
                object_store.set(result_key, result)
                logger.info(
                    f"Stored {tool_name} result in object store as '{result_key}'"
                )

            return {"success": True, "result": result, "tool": tool_name}

        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            return {"success": False, "error": str(e), "tool": tool_name}

    def generate_answer_from_results(
        self, task: str, tool_results: list[dict[str, Any]]
    ) -> str:
        """
        Use LLM to generate a comprehensive answer based on tool results.

        Args:
            task: The original task/question
            tool_results: List of tool execution results

        Returns:
            A natural language answer to the task
        """
        if not self.chat_model_config:
            # Fallback to simple summary
            successful_tools = [r["tool"] for r in tool_results if r["success"]]
            if successful_tools:
                return f"Task completed using {', '.join(successful_tools)}."
            else:
                return "Failed to complete the task due to errors."

        # Prepare tool results for the LLM
        results_description = []
        for result in tool_results:
            if result["success"]:
                # Convert Pydantic models to dict for JSON serialization
                result_data = result.get("result", {})
                if hasattr(result_data, "model_dump"):
                    result_data = result_data.model_dump()
                elif hasattr(result_data, "dict"):
                    result_data = result_data.dict()
                results_description.append(
                    f"- {result['tool']} returned: {json.dumps(result_data, indent=2)}"
                )
            else:
                results_description.append(
                    f"- {result['tool']} failed with error: {result.get('error', 'Unknown error')}"
                )

        # Create the prompt
        system_prompt = """You are a helpful assistant that interprets tool results and provides clear, comprehensive answers to user questions.

Given a task/question and the results from various tools, synthesize the information into a natural, informative response.
Focus on answering the user's original question using the data from the tools.
Be specific and include relevant details from the tool outputs."""

        user_prompt = f"""Task/Question: {task}

Tool Results:
{chr(10).join(results_description)}

Please provide a clear and comprehensive answer to the original task/question based on these tool results."""

        try:
            # Create chat model
            chat_model = create_chat_model(
                backend=self.chat_model_config.get("service", "echo"),
                api_key=self.chat_model_config.get("api_key"),
                model=self.chat_model_config.get("model_name"),
            )

            response = generate_chat_response(
                chat_model=chat_model,
                prompt=user_prompt,
                system_prompt=system_prompt,
            )
            return response.get("response", "Task completed.")

        except Exception as e:
            logger.error(f"Error generating answer with LLM: {e}")
            # Fallback to simple summary
            successful_tools = [r["tool"] for r in tool_results if r["success"]]
            if successful_tools:
                return f"Task completed using {', '.join(successful_tools)}. (LLM summary unavailable)"
            else:
                return "Failed to complete the task due to errors"

    def run(
        self,
        task: str,
        context: dict[str, Any] | None = None,
        object_store: ObjectStore | None = None,
    ) -> dict[str, Any]:
        """
        Execute the router's logic for the given task synchronously.

        This is a wrapper around the async implementation.

        Args:
            task: The task to execute
            context: Optional context information
            object_store: Optional object store for artifacts

        Returns:
            Dictionary containing execution results
        """
        return asyncio.run(self.arun(task, context, object_store))

    async def arun(
        self,
        task: str,
        context: dict[str, Any] | None = None,
        object_store: ObjectStore | None = None,
    ) -> dict[str, Any]:
        """
        Execute the router's logic for the given task asynchronously.

        Args:
            task: The task to execute
            context: Optional context information
            object_store: Optional object store for artifacts

        Returns:
            Dictionary containing execution results
        """
        context = context or {}

        # Select tools based on task
        selected_tools = await asyncio.to_thread(self.select_tools, task)

        if not selected_tools:
            return {
                "task": task,
                "selected_tools": [],
                "tool_results": [],
                "success": False,
                "final_answer": f"No tools found for task: {task}",
            }

        logger.info(f"Selected tools for task: {selected_tools}")

        # Execute each selected tool
        tool_results = []
        for tool_name in selected_tools:
            if tool_name not in self.tools:
                logger.warning(f"Tool '{tool_name}' not found, skipping")
                continue

            tool = self.tools[tool_name]
            tool_spec = tool.spec if hasattr(tool, "spec") else None

            # Generate inputs using LLM if we have chat model config
            if self.chat_model_config and tool_spec:
                generated_inputs = await asyncio.to_thread(
                    self.analyze_and_generate_inputs, task, tool_name, tool_spec
                )
                # Convert to dict if it's a Pydantic model
                if hasattr(generated_inputs, "model_dump"):
                    tool_inputs = generated_inputs.model_dump()
                elif isinstance(generated_inputs, dict):
                    tool_inputs = generated_inputs
                else:
                    tool_inputs = {}
                # Merge with context
                tool_inputs.update(context)
            else:
                # Use context as inputs
                tool_inputs = context.copy()

            # Create proper input object for the tool
            try:
                if tool_spec and tool_spec.input_model:
                    input_obj = tool_spec.input_model(**tool_inputs)
                else:
                    input_obj = tool_inputs
            except Exception as e:
                logger.error(f"Failed to create input for {tool_name}: {e}")
                input_obj = tool_inputs  # Fallback to dict

            # Execute tool asynchronously
            tool_result = await self.aexecute_tool(tool_name, input_obj, object_store)
            tool_results.append(tool_result)

            # Update context with tool results for next tool
            if tool_result["success"]:
                context[f"{tool_name}_output"] = tool_result["result"]

        # Generate final answer
        final_answer = await asyncio.to_thread(
            self.generate_answer_from_results, task, tool_results
        )

        # Mark overall success
        success = any(r["success"] for r in tool_results)

        return {
            "task": task,
            "selected_tools": selected_tools,
            "tool_results": tool_results,
            "success": success,
            "final_answer": final_answer,
        }
