# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the router agent implementation."""

from unittest.mock import MagicMock, patch

import pytest

from world_understanding.agentic.agents import RouterAgent
from world_understanding.tools.base import ToolInput, ToolOutput, register_tool
from world_understanding.utils.object_store import InMemoryObjectStore


# Create test tools for the router to work with
class ColorAnalyzerInput(ToolInput):
    image_path: str


class ColorAnalyzerOutput(ToolOutput):
    dominant_colors: list[str]


class DetectionInput(ToolInput):
    image_path: str


class DetectionOutput(ToolOutput):
    objects: list[str]


class ChatInput(ToolInput):
    prompt: str


class ChatOutput(ToolOutput):
    response: str


@pytest.fixture
def test_tools():
    """Register test tools for router tests."""
    # Clear any existing test tools
    from world_understanding.tools.base import _TOOL_REGISTRY

    test_tool_names = ["test_color_analyzer", "test_detection_yolo", "test_chat_tool"]
    for name in test_tool_names:
        if name in _TOOL_REGISTRY:
            del _TOOL_REGISTRY[name]

    # Register test tools
    @register_tool(
        name="test_color_analyzer",
        version="0.1.0",
        description="Analyze colors in images",
        input_model=ColorAnalyzerInput,
        output_model=ColorAnalyzerOutput,
        tags=["color", "analysis", "vision"],
    )
    def color_analyzer_tool(inputs: ColorAnalyzerInput) -> ColorAnalyzerOutput:
        return ColorAnalyzerOutput(dominant_colors=["red", "blue", "green"])

    @register_tool(
        name="test_detection_yolo",
        version="0.1.0",
        description="Detect objects in images using YOLO",
        input_model=DetectionInput,
        output_model=DetectionOutput,
        tags=["detection", "vision", "yolo"],
    )
    def detection_tool(inputs: DetectionInput) -> DetectionOutput:
        return DetectionOutput(objects=["car", "person", "dog"])

    @register_tool(
        name="test_chat_tool",
        version="0.1.0",
        description="Chat with LLM",
        input_model=ChatInput,
        output_model=ChatOutput,
        tags=["nlp", "chat", "llm"],
    )
    def chat_tool(inputs: ChatInput) -> ChatOutput:
        return ChatOutput(response=f"Response to: {inputs.prompt}")

    return _TOOL_REGISTRY


@pytest.fixture
def router_agent(test_tools):
    """Create a router agent with test tools."""
    return RouterAgent(
        tools=test_tools,
        chat_model_config={
            "backend": "echo",
            "model_name": "echo",
            "api_key": "dummy-key",
        },
        name="test_router",
        description="Test router agent",
    )


@pytest.fixture
def object_store():
    """Create an in-memory object store for testing."""
    return InMemoryObjectStore()


def test_router_agent_initialization(router_agent):
    """Test router agent is properly initialized."""
    assert router_agent.name == "test_router"
    assert router_agent.description == "Test router agent"
    assert router_agent.tools is not None
    assert len(router_agent.tools) >= 3  # At least our test tools


def test_router_agent_can_handle(router_agent):
    """Test router agent can_handle method."""
    # RouterAgent should be able to handle any task
    assert router_agent.can_handle("analyze this image") is True
    assert router_agent.can_handle("detect objects") is True
    assert router_agent.can_handle("chat about something") is True


def test_select_tools_keyword_matching(router_agent):
    """Test tool selection with keyword matching (no LLM)."""
    # Without LLM, it should fall back to keyword matching
    router_agent.chat_model_config = None

    # Should match based on tags
    selected = router_agent.select_tools("I need color analysis")
    assert "test_color_analyzer" in selected

    selected = router_agent.select_tools("detect objects with yolo")
    assert "test_detection_yolo" in selected

    selected = router_agent.select_tools("chat with the model")
    assert "test_chat_tool" in selected


def test_select_tools_with_llm(router_agent):
    """Test tool selection with LLM."""
    with patch.object(router_agent, "analyze_task_with_llm") as mock_analyze:
        # Mock LLM response
        mock_analyze.return_value = {
            "tools": ["test_color_analyzer", "test_detection_yolo"],
            "reasoning": "Image analysis requires color and object detection",
        }

        selected = router_agent.select_tools(
            "Analyze this image for colors and objects"
        )
        # select_tools returns the dict from analyze_task_with_llm when using LLM
        assert isinstance(selected, dict) or isinstance(selected, list)
        if isinstance(selected, dict):
            assert "test_color_analyzer" in selected.get("tools", [])
            assert "test_detection_yolo" in selected.get("tools", [])
        else:
            assert "test_color_analyzer" in selected
            assert "test_detection_yolo" in selected


def test_execute_tool_success(router_agent, object_store):
    """Test successful tool execution."""
    # Create proper input object
    router_agent.tools["test_color_analyzer"]
    inputs = ColorAnalyzerInput(image_path="test.jpg")

    result = router_agent.execute_tool("test_color_analyzer", inputs, object_store)

    assert result["success"] is True
    assert "result" in result
    assert result["tool"] == "test_color_analyzer"

    # Check object store
    assert object_store.exists("test_color_analyzer_result")


def test_execute_tool_with_error(router_agent, object_store):
    """Test tool execution with error."""
    with pytest.raises(ValueError, match="Tool 'non_existent_tool' not found"):
        router_agent.execute_tool("non_existent_tool", {}, object_store)


def test_generate_answer_from_results(router_agent):
    """Test generating answer from tool results."""
    tool_results = [
        {
            "success": True,
            "tool": "test_color_analyzer",
            "result": ColorAnalyzerOutput(dominant_colors=["red", "blue"]).model_dump(),
        },
        {
            "success": True,
            "tool": "test_detection_yolo",
            "result": DetectionOutput(objects=["car", "person"]).model_dump(),
        },
    ]

    # Without LLM, should return a basic summary
    router_agent.chat_model_config = None
    answer = router_agent.generate_answer_from_results(
        "Analyze the image", tool_results
    )
    assert "test_color_analyzer" in answer
    assert "test_detection_yolo" in answer


def test_run_method(router_agent, object_store):
    """Test the main run method of router agent."""
    task = "Analyze colors in test.jpg"
    context = {"user_request": task}

    with patch.object(router_agent, "select_tools") as mock_select:
        with patch.object(router_agent, "analyze_and_generate_inputs") as mock_inputs:
            # Setup mocks
            mock_select.return_value = ["test_color_analyzer"]
            mock_inputs.return_value = ColorAnalyzerInput(image_path="test.jpg")

            # Run the agent
            result = router_agent.run(task, context, object_store)

            # Check results
            assert "selected_tools" in result
            assert "test_color_analyzer" in result["selected_tools"]
            assert "tool_results" in result
            assert "final_answer" in result


def test_run_with_no_tools_selected(router_agent, object_store):
    """Test run method when no tools are selected."""
    task = "This is an unclear request"
    context = {}

    with patch.object(router_agent, "select_tools") as mock_select:
        # Setup mocks
        mock_select.return_value = []

        # Run the agent
        result = router_agent.run(task, context, object_store)

        # Check results
        assert result["selected_tools"] == []
        assert "no tools" in result["final_answer"].lower()


def test_analyze_and_generate_inputs_with_llm(router_agent):
    """Test input generation with LLM."""
    tool = router_agent.tools["test_color_analyzer"]
    tool_spec = tool.spec if hasattr(tool, "spec") else None

    # Mock the create_chat_model to return a model that generates the expected response
    with patch(
        "world_understanding.agentic.agents.router.create_chat_model"
    ) as mock_create:
        mock_model = MagicMock()
        mock_model.invoke.return_value = MagicMock(content='{"image_path": "test.jpg"}')
        mock_create.return_value = mock_model

        result = router_agent.analyze_and_generate_inputs(
            "Analyze colors in test.jpg", "test_color_analyzer", tool_spec
        )

        # Should return the inputs from LLM
        if hasattr(result, "model_dump"):
            result = result.model_dump()
        assert result == {"image_path": "test.jpg"}


def test_analyze_and_generate_inputs_without_llm(router_agent):
    """Test input generation without LLM."""
    router_agent.chat_model_config = None
    tool = router_agent.tools["test_color_analyzer"]
    tool_spec = tool.spec if hasattr(tool, "spec") else None

    # Without LLM, should return empty dict
    result = router_agent.analyze_and_generate_inputs(
        "Analyze colors", "test_color_analyzer", tool_spec
    )

    assert result == {}


def test_error_handling_in_run(router_agent, object_store):
    """Test error handling in the run method."""
    task = "Analyze this"
    context = {}

    with patch.object(router_agent, "select_tools") as mock_select:
        # Simulate an error during tool selection
        mock_select.side_effect = Exception("Tool selection failed")

        # Run should raise the error (no error handling in run method)
        with pytest.raises(Exception, match="Tool selection failed"):
            router_agent.run(task, context, object_store)


def test_multiple_tool_execution(router_agent, object_store):
    """Test executing multiple tools in sequence."""
    task = "Analyze image comprehensively"
    context = {}

    with patch.object(router_agent, "select_tools") as mock_select:
        with patch.object(router_agent, "analyze_and_generate_inputs") as mock_inputs:
            # Setup mocks for multiple tools
            mock_select.return_value = ["test_color_analyzer", "test_detection_yolo"]
            mock_inputs.side_effect = [
                ColorAnalyzerInput(image_path="test.jpg"),
                DetectionInput(image_path="test.jpg"),
            ]

            # Run the agent
            result = router_agent.run(task, context, object_store)

            # Check that both tools were used
            assert len(result["selected_tools"]) == 2
            assert "test_color_analyzer" in result["selected_tools"]
            assert "test_detection_yolo" in result["selected_tools"]
