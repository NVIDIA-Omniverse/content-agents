# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multi-step agent for executing complex pipelines."""

from typing import Any

from world_understanding.agentic.base import BaseAgent
from world_understanding.tools.base import get_tool_registry
from world_understanding.utils.object_store import ObjectStore


class MultiStepAgent(BaseAgent):
    """
    Agent that executes multi-step workflows with confidence tracking.

    This agent orchestrates multiple tools in sequence, tracks confidence
    scores, and reports overall workflow status.
    """

    def __init__(
        self,
        tools: dict[str, Any] | None = None,
        pipeline: list[dict[str, Any]] | None = None,
        name: str = "MultiStepAgent",
        description: str = "Multi-step pipeline execution agent",
    ):
        """
        Initialize the multi-step agent.

        Args:
            tools: Tool registry (uses global registry if None)
            pipeline: Optional pipeline definition
            name: Agent name
            description: Agent description
        """
        super().__init__(name, description)
        self.tools = tools or get_tool_registry()
        self.pipeline = pipeline or []

    def run(
        self,
        task: str,
        context: dict[str, Any] | None = None,
        object_store: ObjectStore | None = None,
    ) -> dict[str, Any]:
        """
        Execute a multi-step workflow.

        Args:
            task: Task identifier (e.g., "execute_workflow")
            context: Workflow context
            object_store: Storage for artifacts

        Returns:
            Updated context with results and confidence scores
        """
        if context is None:
            context = {}

        if task == "execute_workflow":
            return self.execute_workflow(context, object_store)
        else:
            # Fallback to simple execution
            return self.execute_single_step(task, context, object_store)

    async def arun(
        self,
        task: str,
        context: dict[str, Any] | None = None,
        object_store: ObjectStore | None = None,
    ) -> dict[str, Any]:
        """Execute the multi-step agent's logic asynchronously."""
        if context is None:
            context = {}
        if task == "execute_workflow":
            return await self.aexecute_workflow(context, object_store)
        else:
            return await self.aexecute_single_step(task, context, object_store)

    def execute_workflow(
        self, context: dict[str, Any], object_store: ObjectStore | None = None
    ) -> dict[str, Any]:
        """
        Execute the full multi-step workflow.

        Args:
            context: Workflow context
            object_store: Storage for artifacts

        Returns:
            Updated context with results and confidence tracking
        """
        # Get pipeline from context or use default
        pipeline = context.get("pipeline", self.pipeline)
        if not pipeline:
            context["error"] = "No pipeline defined"
            context["completed"] = False
            return context

        confidence_scores = {}
        all_outputs = {}

        for i, step in enumerate(pipeline):
            step_name = step.get("name", f"step_{i}")
            tool_name = step.get("tool")
            params = step.get("params", {})

            # Get tool
            tool = self.tools.get(tool_name)
            if not tool:
                context["error"] = f"Tool '{tool_name}' not found"
                context["completed"] = False
                return context

            try:
                # Prepare inputs - may reference previous outputs
                resolved_params = self._resolve_params(
                    params, context, all_outputs, object_store
                )

                # Create input object
                input_obj = tool.spec.input_model(**resolved_params)

                # Execute tool
                output = tool.run(input_obj)
                output_dict = output.model_dump()

                # Store output
                all_outputs[step_name] = output_dict
                if object_store:
                    object_store.set(f"step_{step_name}_output", output_dict)

                # Extract confidence if available
                if "confidence" in output_dict:
                    if isinstance(output_dict["confidence"], dict):
                        confidence_scores.update(output_dict["confidence"])
                    else:
                        confidence_scores[step_name] = output_dict["confidence"]

            except Exception as e:
                context["error"] = f"Step '{step_name}' failed: {str(e)}"
                context["completed"] = False
                return context

        # Calculate average confidence
        if confidence_scores:
            avg_confidence = sum(confidence_scores.values()) / len(confidence_scores)
            context["avg_confidence"] = round(avg_confidence, 3)
            context["confidence_scores"] = confidence_scores

        # Store all outputs
        context["pipeline_outputs"] = all_outputs
        context["completed"] = True

        # Let the task decide on refinement based on confidence
        return context

    async def aexecute_workflow(
        self, context: dict[str, Any], object_store: ObjectStore | None = None
    ) -> dict[str, Any]:
        """
        Execute the full multi-step workflow asynchronously.

        Args:
            context: Workflow context
            object_store: Storage for artifacts

        Returns:
            Updated context with results and confidence tracking
        """
        # Get pipeline from context or use default
        pipeline = context.get("pipeline", self.pipeline)
        if not pipeline:
            context["error"] = "No pipeline defined"
            context["completed"] = False
            return context

        confidence_scores = {}
        all_outputs = {}

        for i, step in enumerate(pipeline):
            step_name = step.get("name", f"step_{i}")
            tool_name = step.get("tool")
            params = step.get("params", {})

            # Get tool
            tool = self.tools.get(tool_name)
            if not tool:
                context["error"] = f"Tool '{tool_name}' not found"
                context["completed"] = False
                return context

            try:
                # Prepare inputs - may reference previous outputs
                resolved_params = self._resolve_params(
                    params, context, all_outputs, object_store
                )

                # Create input object
                input_obj = tool.spec.input_model(**resolved_params)

                # Execute tool asynchronously
                output = await tool.arun(input_obj)
                output_dict = output.model_dump()

                # Store output
                all_outputs[step_name] = output_dict
                if object_store:
                    object_store.set(f"step_{step_name}_output", output_dict)

                # Extract confidence if available
                if "confidence" in output_dict:
                    if isinstance(output_dict["confidence"], dict):
                        confidence_scores.update(output_dict["confidence"])
                    else:
                        confidence_scores[step_name] = output_dict["confidence"]

            except Exception as e:
                context["error"] = f"Step '{step_name}' failed: {str(e)}"
                context["completed"] = False
                return context

        # Calculate average confidence
        if confidence_scores:
            avg_confidence = sum(confidence_scores.values()) / len(confidence_scores)
            context["avg_confidence"] = round(avg_confidence, 3)
            context["confidence_scores"] = confidence_scores

        # Store all outputs
        context["pipeline_outputs"] = all_outputs
        context["completed"] = True

        # Let the task decide on refinement based on confidence
        return context

    def execute_single_step(
        self,
        task: str,
        context: dict[str, Any],
        object_store: ObjectStore | None = None,
    ) -> dict[str, Any]:
        """
        Execute a single tool step.

        Args:
            task: Tool name
            context: Workflow context
            object_store: Storage for artifacts

        Returns:
            Updated context
        """
        tool = self.tools.get(task)
        if not tool:
            context["error"] = f"Tool '{task}' not found"
            return context

        # Get parameters from context
        params = context.get(f"{task}_params", {})

        try:
            input_obj = tool.spec.input_model(**params)
            output = tool.run(input_obj)

            context[f"{task}_output"] = output.model_dump()
            context[f"{task}_success"] = True

            if object_store:
                object_store.set(f"{task}_output", output.model_dump())

        except Exception as e:
            context[f"{task}_error"] = str(e)
            context[f"{task}_success"] = False

        return context

    async def aexecute_single_step(
        self,
        task: str,
        context: dict[str, Any],
        object_store: ObjectStore | None = None,
    ) -> dict[str, Any]:
        """
        Execute a single tool step asynchronously.

        Args:
            task: Tool name
            context: Workflow context
            object_store: Storage for artifacts

        Returns:
            Updated context
        """
        tool = self.tools.get(task)
        if not tool:
            context["error"] = f"Tool '{task}' not found"
            return context

        # Get parameters from context
        params = context.get(f"{task}_params", {})

        try:
            input_obj = tool.spec.input_model(**params)
            output = await tool.arun(input_obj)

            context[f"{task}_output"] = output.model_dump()
            context[f"{task}_success"] = True

            if object_store:
                object_store.set(f"{task}_output", output.model_dump())

        except Exception as e:
            context[f"{task}_error"] = str(e)
            context[f"{task}_success"] = False

        return context

    def _resolve_params(
        self,
        params: dict[str, Any],
        context: dict[str, Any],
        outputs: dict[str, Any],
        object_store: ObjectStore | None = None,
    ) -> dict[str, Any]:
        """
        Resolve parameter references from context and previous outputs.

        Supports references like:
        - "${context.key}" - reference from context
        - "${output.step_name.field}" - reference from previous step output
        - "${store.key}" - reference from object store

        Args:
            params: Raw parameters with potential references
            context: Workflow context
            outputs: Previous step outputs
            object_store: Storage for artifacts

        Returns:
            Resolved parameters
        """
        resolved = {}

        for key, value in params.items():
            if (
                isinstance(value, str)
                and value.startswith("${")
                and value.endswith("}")
            ):
                # Parse reference
                ref = value[2:-1]  # Remove ${ and }
                parts = ref.split(".")

                if parts[0] == "context" and len(parts) > 1:
                    # Reference from context
                    resolved[key] = self._get_nested(context, parts[1:])
                elif parts[0] == "output" and len(parts) > 2:
                    # Reference from previous output
                    step_name = parts[1]
                    if step_name in outputs:
                        resolved[key] = self._get_nested(outputs[step_name], parts[2:])
                    else:
                        resolved[key] = None
                elif parts[0] == "store" and len(parts) > 1 and object_store:
                    # Reference from object store
                    resolved[key] = object_store.get(parts[1])
                else:
                    # Unknown reference, keep as is
                    resolved[key] = value
            else:
                # Not a reference, keep as is
                resolved[key] = value

        return resolved

    def _get_nested(self, data: dict[str, Any], keys: list[str]) -> Any:
        """Get nested value from dictionary."""
        current = data
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        return current
