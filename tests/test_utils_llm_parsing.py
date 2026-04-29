# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for LLM parsing utilities."""

import pytest

from world_understanding.utils.llm_parsing import (
    create_json_prompt_instructions,
    extract_json_from_llm_response,
)


class TestExtractJsonFromLLMResponse:
    """Test cases for extract_json_from_llm_response function."""

    def test_extract_json_from_markdown_code_block(self):
        """Test extracting JSON from markdown code blocks."""
        response = """Here is the JSON response:
```json
{
    "name": "test",
    "value": 123,
    "active": true
}
```
Some additional text here."""

        result = extract_json_from_llm_response(response)
        assert result is not None
        assert result["name"] == "test"
        assert result["value"] == 123
        assert result["active"] is True

    def test_extract_json_from_plain_code_block(self):
        """Test extracting JSON from code blocks without json specifier."""
        response = """Here is the response:
```
{
    "tools": ["tool1", "tool2"],
    "reasoning": "This is why"
}
```
"""

        result = extract_json_from_llm_response(response)
        assert result is not None
        assert result["tools"] == ["tool1", "tool2"]
        assert result["reasoning"] == "This is why"

    def test_extract_json_from_text_with_surrounding_content(self):
        """Test extracting JSON from text with surrounding content."""
        response = """Let me generate the JSON for you.
{"key": "value", "number": 42, "array": [1, 2, 3]}
That's the JSON you requested."""

        result = extract_json_from_llm_response(response)
        assert result is not None
        assert result["key"] == "value"
        assert result["number"] == 42
        assert result["array"] == [1, 2, 3]

    def test_extract_plain_json(self):
        """Test extracting plain JSON without any surrounding text."""
        response = '{"status": "success", "count": 5}'

        result = extract_json_from_llm_response(response)
        assert result is not None
        assert result["status"] == "success"
        assert result["count"] == 5

    def test_extract_nested_json(self):
        """Test extracting nested JSON structures."""
        response = """```json
{
    "user": {
        "name": "John",
        "age": 30,
        "addresses": [
            {"type": "home", "city": "NYC"},
            {"type": "work", "city": "SF"}
        ]
    },
    "active": true
}
```"""

        result = extract_json_from_llm_response(response)
        assert result is not None
        assert result["user"]["name"] == "John"
        assert result["user"]["age"] == 30
        assert len(result["user"]["addresses"]) == 2
        assert result["user"]["addresses"][0]["city"] == "NYC"

    def test_extract_json_with_comments_fails(self):
        """Test that JSON with comments (invalid JSON) returns None."""
        response = """```json
{
    "name": "test", // This is a comment
    "value": 123
}
```"""

        result = extract_json_from_llm_response(response)
        assert result is None  # Should fail due to comments

    def test_extract_json_with_inline_comments_from_llm(self):
        """Test the exact format seen in the terminal output."""
        response = """Here is the JSON object containing the tool inputs:
```
{
  "image": {
    "path": null, // in-memory image
    "width": 640, // default width for a typical image
    "height": 480, // default height for a typical image
    "mime": "image/jpeg" // common MIME type for JPEG images
  },
  "target_color": [255, 105, 180], // pink color in RGB (0-255)
  "color_tolerance": 20, // relatively low tolerance to match exact pink color
  "min_percentage": 5.0 // require at least 5% of pixels to match the target color
}
```
Note that I've chosen sensible defaults for the image dimensions and MIME type."""

        result = extract_json_from_llm_response(response)
        assert result is None  # Should fail due to inline comments

    def test_empty_response(self):
        """Test handling of empty response."""
        result = extract_json_from_llm_response("")
        assert result is None

        result = extract_json_from_llm_response(None)
        assert result is None

    def test_no_json_in_response(self):
        """Test handling when no JSON is found."""
        response = "This is just plain text without any JSON content."

        result = extract_json_from_llm_response(response)
        assert result is None

    def test_invalid_json(self):
        """Test handling of invalid JSON."""
        response = """```json
{
    "name": "test"
    "missing": "comma"
}
```"""

        result = extract_json_from_llm_response(response)
        assert result is None

    def test_expected_keys_validation(self):
        """Test validation of expected keys."""
        response = '{"name": "test", "value": 123}'

        # All expected keys present
        result = extract_json_from_llm_response(
            response, expected_keys=["name", "value"]
        )
        assert result is not None
        assert result["name"] == "test"
        assert result["value"] == 123

        # Missing expected key - should still return result but log warning
        result = extract_json_from_llm_response(
            response, expected_keys=["name", "value", "missing"]
        )
        assert result is not None
        assert result["name"] == "test"
        assert result["value"] == 123

    def test_multiple_json_objects(self):
        """Test extracting when multiple JSON objects are present."""
        response = """First object: {"a": 1}
Second object: {"b": 2}"""

        # Should extract the first complete JSON object
        result = extract_json_from_llm_response(response)
        assert result is not None
        assert result == {"a": 1}

    def test_json_with_special_characters(self):
        """Test JSON with special characters and escape sequences."""
        response = """```json
{
    "message": "Hello\\nWorld",
    "path": "C:\\\\Users\\\\test",
    "unicode": "\\u00A9 2024"
}
```"""

        result = extract_json_from_llm_response(response)
        assert result is not None
        assert result["message"] == "Hello\nWorld"
        assert result["path"] == "C:\\Users\\test"
        assert result["unicode"] == "© 2024"

    def test_json_with_different_types(self):
        """Test JSON with various data types."""
        response = """```json
{
    "string": "text",
    "integer": 42,
    "float": 3.14,
    "boolean": true,
    "null_value": null,
    "array": [1, "two", 3.0, false, null],
    "empty_object": {},
    "empty_array": []
}
```"""

        result = extract_json_from_llm_response(response)
        assert result is not None
        assert result["string"] == "text"
        assert result["integer"] == 42
        assert result["float"] == 3.14
        assert result["boolean"] is True
        assert result["null_value"] is None
        assert result["array"] == [1, "two", 3.0, False, None]
        assert result["empty_object"] == {}
        assert result["empty_array"] == []


class TestCreateJsonPromptInstructions:
    """Test cases for create_json_prompt_instructions function."""

    def test_returns_string(self):
        """Test that the function returns a string."""
        result = create_json_prompt_instructions()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_key_instructions(self):
        """Test that the instructions contain key phrases."""
        result = create_json_prompt_instructions()
        assert "JSON" in result
        assert "ONLY" in result
        assert "valid" in result

    def test_consistent_output(self):
        """Test that the function returns consistent output."""
        result1 = create_json_prompt_instructions()
        result2 = create_json_prompt_instructions()
        assert result1 == result2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
