# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for base tool interface and core abstractions."""

import pytest
from pydantic import BaseModel, Field, ValidationError

from world_understanding.tools.base import (
    ExecutionPolicy,
    Tool,
    ToolInput,
    ToolOutput,
    ToolSpec,
    get_tool,
    get_tool_registry,
    register_tool,
)


class TestToolInput:
    """Tests for ToolInput base class."""

    def test_tool_input_is_base_model(self):
        """Test that ToolInput inherits from BaseModel."""
        assert issubclass(ToolInput, BaseModel)

    def test_tool_input_instantiation(self):
        """Test that ToolInput can be instantiated."""
        input_obj = ToolInput()
        assert isinstance(input_obj, ToolInput)

    def test_tool_input_subclass(self):
        """Test creating a subclass of ToolInput."""

        class CustomInput(ToolInput):
            param1: str
            param2: int = 10

        # Test instantiation with required param
        custom = CustomInput(param1="test")
        assert custom.param1 == "test"
        assert custom.param2 == 10

        # Test with both params
        custom2 = CustomInput(param1="test2", param2=20)
        assert custom2.param1 == "test2"
        assert custom2.param2 == 20

        # Test validation error
        with pytest.raises(ValidationError):
            CustomInput(param2=30)  # Missing required param1


class TestToolOutput:
    """Tests for ToolOutput base class."""

    def test_tool_output_is_base_model(self):
        """Test that ToolOutput inherits from BaseModel."""
        assert issubclass(ToolOutput, BaseModel)

    def test_tool_output_instantiation(self):
        """Test that ToolOutput can be instantiated."""
        output_obj = ToolOutput()
        assert isinstance(output_obj, ToolOutput)

    def test_tool_output_subclass(self):
        """Test creating a subclass of ToolOutput."""

        class CustomOutput(ToolOutput):
            result: str
            score: float = 0.0

        # Test instantiation
        output = CustomOutput(result="success")
        assert output.result == "success"
        assert output.score == 0.0

        # Test with custom score
        output2 = CustomOutput(result="success", score=0.95)
        assert output2.result == "success"
        assert output2.score == 0.95


class TestExecutionPolicy:
    """Tests for ExecutionPolicy model."""

    def test_default_values(self):
        """Test default values of ExecutionPolicy."""
        policy = ExecutionPolicy()
        assert policy.timeout_s == 60.0
        assert policy.max_retries == 0
        assert policy.device is None

    def test_custom_values(self):
        """Test ExecutionPolicy with custom values."""
        policy = ExecutionPolicy(timeout_s=120.0, max_retries=3, device="cuda:0")
        assert policy.timeout_s == 120.0
        assert policy.max_retries == 3
        assert policy.device == "cuda:0"

    def test_device_options(self):
        """Test various device options."""
        devices = ["cpu", "cuda:0", "cuda:1", "mps", None]
        for device in devices:
            policy = ExecutionPolicy(device=device)
            assert policy.device == device


class TestToolSpec:
    """Tests for ToolSpec model."""

    def test_required_fields(self):
        """Test that ToolSpec requires all mandatory fields."""
        # Missing required fields should raise ValidationError
        with pytest.raises(ValidationError):
            ToolSpec()

    def test_valid_toolspec(self):
        """Test creating a valid ToolSpec."""
        spec = ToolSpec(
            name="test_tool",
            version="1.0.0",
            description="A test tool",
            input_model=ToolInput,
            output_model=ToolOutput,
        )
        assert spec.name == "test_tool"
        assert spec.version == "1.0.0"
        assert spec.description == "A test tool"
        assert spec.input_model == ToolInput
        assert spec.output_model == ToolOutput
        assert spec.tags == []  # Default empty list
        assert isinstance(spec.policy, ExecutionPolicy)

    def test_toolspec_with_tags(self):
        """Test ToolSpec with tags."""
        spec = ToolSpec(
            name="test_tool",
            version="1.0.0",
            description="A test tool",
            tags=["test", "example", "demo"],
            input_model=ToolInput,
            output_model=ToolOutput,
        )
        assert spec.tags == ["test", "example", "demo"]

    def test_toolspec_with_custom_policy(self):
        """Test ToolSpec with custom execution policy."""
        custom_policy = ExecutionPolicy(timeout_s=30.0, max_retries=2)
        spec = ToolSpec(
            name="test_tool",
            version="1.0.0",
            description="A test tool",
            input_model=ToolInput,
            output_model=ToolOutput,
            policy=custom_policy,
        )
        assert spec.policy.timeout_s == 30.0
        assert spec.policy.max_retries == 2


class TestFunctionBasedTools:
    """Tests for the new function-based tool system."""

    def test_register_tool_decorator(self):
        """Test the @register_tool decorator."""
        # Clear registry first
        registry = get_tool_registry()
        if "test_double" in registry:
            del registry["test_double"]

        class DoubleInput(ToolInput):
            value: int

        class DoubleOutput(ToolOutput):
            doubled: int

        @register_tool(
            name="test_double",
            version="1.0.0",
            description="Doubles the input",
            input_model=DoubleInput,
            output_model=DoubleOutput,
        )
        def double_tool(inputs: DoubleInput) -> DoubleOutput:
            return DoubleOutput(doubled=inputs.value * 2)

        # Test that tool is registered
        tool = get_tool("test_double")
        assert tool is not None
        assert isinstance(tool, Tool)
        assert tool.spec.name == "test_double"

        # Test execution
        result = tool.run(DoubleInput(value=5))
        assert result.doubled == 10

    def test_tool_with_tags_and_policy(self):
        """Test registering a tool with tags and custom policy."""
        # Clear registry first
        registry = get_tool_registry()
        if "test_calculator" in registry:
            del registry["test_calculator"]

        class CalcInput(ToolInput):
            a: float
            b: float
            operation: str

        class CalcOutput(ToolOutput):
            result: float

        @register_tool(
            name="test_calculator",
            version="2.0.0",
            description="Simple calculator",
            input_model=CalcInput,
            output_model=CalcOutput,
            tags=["math", "calculator"],
            policy=ExecutionPolicy(timeout_s=10.0, device="cpu"),
        )
        def calculator_tool(inputs: CalcInput) -> CalcOutput:
            if inputs.operation == "add":
                result = inputs.a + inputs.b
            elif inputs.operation == "multiply":
                result = inputs.a * inputs.b
            else:
                raise ValueError(f"Unknown operation: {inputs.operation}")
            return CalcOutput(result=result)

        # Test registration
        tool = get_tool("test_calculator")
        assert tool.spec.tags == ["math", "calculator"]
        assert tool.spec.policy.timeout_s == 10.0
        assert tool.spec.policy.device == "cpu"

        # Test execution
        result = tool.run(CalcInput(a=3, b=4, operation="add"))
        assert result.result == 7

    def test_tool_validation(self):
        """Test input/output validation for function-based tools."""
        # Clear registry first
        registry = get_tool_registry()
        if "test_validator" in registry:
            del registry["test_validator"]

        class ValidatorInput(ToolInput):
            required_field: str
            optional_field: int = 10

        class ValidatorOutput(ToolOutput):
            message: str

        @register_tool(
            name="test_validator",
            version="1.0.0",
            description="Test validation",
            input_model=ValidatorInput,
            output_model=ValidatorOutput,
        )
        def validator_tool(inputs: ValidatorInput) -> ValidatorOutput:
            return ValidatorOutput(
                message=f"{inputs.required_field}: {inputs.optional_field}"
            )

        tool = get_tool("test_validator")

        # Test valid input
        result = tool.run(ValidatorInput(required_field="test"))
        assert result.message == "test: 10"

        # Test input validation with dict
        validated = tool.validate_input(
            {"required_field": "hello", "optional_field": 20}
        )
        assert validated.required_field == "hello"
        assert validated.optional_field == 20

        # Test invalid input
        with pytest.raises(ValidationError):
            tool.validate_input({"optional_field": 20})  # Missing required field

    def test_async_tool_execution(self):
        """Test async execution of function-based tools."""
        import asyncio

        # Clear registry first
        registry = get_tool_registry()
        if "test_async" in registry:
            del registry["test_async"]

        class AsyncInput(ToolInput):
            value: int

        class AsyncOutput(ToolOutput):
            result: int

        @register_tool(
            name="test_async",
            version="1.0.0",
            description="Async test",
            input_model=AsyncInput,
            output_model=AsyncOutput,
        )
        def async_tool(inputs: AsyncInput) -> AsyncOutput:
            return AsyncOutput(result=inputs.value + 1)

        tool = get_tool("test_async")

        # Test async execution
        async def test_async():
            result = await tool.arun(AsyncInput(value=5))
            assert result.result == 6

        asyncio.run(test_async())

    def test_tool_registry(self):
        """Test the global tool registry."""
        # Clear and register some test tools
        registry = get_tool_registry()

        # Clear test tools if they exist
        test_tools = ["registry_test_1", "registry_test_2"]
        for tool_name in test_tools:
            if tool_name in registry:
                del registry[tool_name]

        @register_tool(
            name="registry_test_1",
            version="1.0.0",
            description="Test tool 1",
        )
        def tool1(inputs: ToolInput) -> ToolOutput:
            return ToolOutput()

        @register_tool(
            name="registry_test_2",
            version="1.0.0",
            description="Test tool 2",
            tags=["test"],
        )
        def tool2(inputs: ToolInput) -> ToolOutput:
            return ToolOutput()

        # Test that tools are in registry
        assert "registry_test_1" in registry
        assert "registry_test_2" in registry

        # Test get_tool
        tool = get_tool("registry_test_1")
        assert tool is not None
        assert tool.spec.name == "registry_test_1"

        # Test get_tool with non-existent tool
        assert get_tool("non_existent_tool") is None


class TestToolWrapper:
    """Test the Tool wrapper class."""

    def test_tool_wrapper_functionality(self):
        """Test that the Tool wrapper properly wraps functions."""

        def my_function(inputs: ToolInput) -> ToolOutput:
            return ToolOutput()

        spec = ToolSpec(
            name="wrapper_test",
            version="1.0.0",
            description="Test wrapper",
            input_model=ToolInput,
            output_model=ToolOutput,
        )

        tool = Tool(my_function, spec)

        # Test that tool has the right attributes
        assert tool.spec == spec
        assert tool.func == my_function

        # Test execution
        result = tool.run(ToolInput())
        assert isinstance(result, ToolOutput)

    def test_tool_json_schema(self):
        """Test JSON schema generation for tools."""

        class SchemaInput(ToolInput):
            text: str = Field(..., description="Input text")
            max_length: int = Field(default=100, description="Maximum length")

        class SchemaOutput(ToolOutput):
            processed: str = Field(..., description="Processed text")

        @register_tool(
            name="schema_test",
            version="1.0.0",
            description="Test schema generation",
            input_model=SchemaInput,
            output_model=SchemaOutput,
        )
        def schema_tool(inputs: SchemaInput) -> SchemaOutput:
            text = inputs.text[: inputs.max_length]
            return SchemaOutput(processed=text)

        tool = get_tool("schema_test")

        # Test that to_json_schema method exists
        assert hasattr(tool, "to_json_schema")
        schema = tool.to_json_schema()

        # Basic schema structure test
        assert "input_schema" in schema
        assert "output_schema" in schema
        assert schema["name"] == "schema_test"
        assert schema["version"] == "1.0.0"
        assert schema["description"] == "Test schema generation"
