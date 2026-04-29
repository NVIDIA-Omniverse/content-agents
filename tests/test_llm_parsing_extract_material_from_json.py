# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for extract_material_from_json utility function."""

from typing import Any

from world_understanding.utils.llm_parsing import extract_material_from_json

# ruff: noqa: ARG005  # Allow unused arguments in tests


class TestExtractMaterialFromJson:
    """Test the flexible material extraction from various JSON schemas."""

    def test_simple_material_key(self) -> None:
        """Test extraction with standard {"material": "value"} schema."""
        json_obj = {"material": "Plastic Dark Blue"}
        result = extract_material_from_json(json_obj)
        assert result == "Plastic Dark Blue"

    def test_double_nested_material(self) -> None:
        """Test extraction from double-nested JSON like {{"material": "value"}}."""
        # This is the user's example case
        json_obj = {"material": "Plastic Dark Blue"}  # Single nesting (valid JSON)
        result = extract_material_from_json(json_obj)
        assert result == "Plastic Dark Blue"

        # Simulate what happens when JSON parser extracts the outer layer
        # The real issue is when we have: '{"material":"Plastic Dark Blue"}'
        # parsed which gives us the dict directly
        json_obj = {"material": "Plastic Dark Blue"}
        result = extract_material_from_json(json_obj)
        assert result == "Plastic Dark Blue"

    def test_nested_under_arbitrary_key(self) -> None:
        """Test extraction when material is nested under another key."""
        json_obj: dict[str, Any] = {"result": {"material": "Steel"}}
        result = extract_material_from_json(json_obj)
        assert result == "Steel"

        json_obj = {"data": {"predicted_material": "Aluminum"}}
        result = extract_material_from_json(json_obj)
        assert result == "Aluminum"

    def test_alternative_key_names(self) -> None:
        """Test extraction with various alternative key names."""
        test_cases = [
            ({"predicted_material": "Glass"}, "Glass"),
            ({"material_name": "Rubber"}, "Rubber"),
            ({"name": "Plastic"}, "Plastic"),
            ({"value": "Wood"}, "Wood"),
        ]

        for json_obj, expected in test_cases:
            result = extract_material_from_json(json_obj)
            assert result == expected

    def test_deeply_nested_material(self) -> None:
        """Test extraction from deeply nested structures."""
        json_obj = {"response": {"analysis": {"result": {"material": "Carbon Fiber"}}}}
        result = extract_material_from_json(json_obj)
        assert result == "Carbon Fiber"

    def test_material_with_extra_fields(self) -> None:
        """Test extraction when there are additional fields in the JSON."""
        json_obj: dict[str, Any] = {
            "material": "Bronze",
            "confidence": 0.95,
            "reasoning": "The color and texture suggest bronze",
        }
        result = extract_material_from_json(json_obj)
        assert result == "Bronze"

    def test_single_key_dict_recursion(self) -> None:
        """Test that single-key dicts are recursively explored."""
        # Case: {"outer": {"material": "value"}}
        json_obj = {"outer": {"material": "Titanium"}}
        result = extract_material_from_json(json_obj)
        assert result == "Titanium"

        # Case: Deeply nested single-key dicts
        json_obj = {"a": {"b": {"c": {"material": "Copper"}}}}
        result = extract_material_from_json(json_obj)
        assert result == "Copper"

    def test_string_input_returns_string(self) -> None:
        """Test that passing a string directly returns the string."""
        result = extract_material_from_json("Plastic Dark Blue")  # type: ignore
        assert result == "Plastic Dark Blue"

    def test_invalid_input_returns_none(self) -> None:
        """Test that invalid input types return None."""
        # List input
        result = extract_material_from_json([1, 2, 3])  # type: ignore
        assert result is None

        # Number input
        result = extract_material_from_json(123)  # type: ignore
        assert result is None

        # None input
        result = extract_material_from_json(None)  # type: ignore
        assert result is None

    def test_empty_dict_returns_none(self) -> None:
        """Test that empty dict returns None."""
        result = extract_material_from_json({})
        assert result is None

    def test_no_material_key_returns_none(self) -> None:
        """Test that dict without any material-related keys returns None."""
        json_obj = {"foo": "bar", "baz": 123}
        result = extract_material_from_json(json_obj)
        assert result is None

    def test_material_key_with_dict_value(self) -> None:
        """Test extraction when material key has nested dict value."""
        json_obj = {"material": {"name": "Steel Alloy"}}
        result = extract_material_from_json(json_obj)
        assert result == "Steel Alloy"

    def test_custom_possible_keys(self) -> None:
        """Test extraction with custom list of possible keys."""
        json_obj = {"my_custom_material_field": "Custom Material"}
        result = extract_material_from_json(
            json_obj, possible_keys=["my_custom_material_field"]
        )
        assert result == "Custom Material"

    def test_priority_of_keys(self) -> None:
        """Test that material key has priority over other keys."""
        # When both 'material' and 'name' exist, 'material' should be preferred
        json_obj = {"material": "Primary", "name": "Secondary"}
        result = extract_material_from_json(json_obj)
        assert result == "Primary"

    def test_real_world_vlm_responses(self) -> None:
        """Test with realistic VLM response formats."""
        # Case 1: Simple response
        json_obj: dict[str, Any] = {"material": "Brushed Aluminum"}
        assert extract_material_from_json(json_obj) == "Brushed Aluminum"

        # Case 2: Response with confidence
        json_obj = {"material": "Matte Black Plastic", "confidence": 0.89}
        assert extract_material_from_json(json_obj) == "Matte Black Plastic"

        # Case 3: Response wrapped in result
        json_obj = {"result": {"material": "Polished Chrome"}}
        assert extract_material_from_json(json_obj) == "Polished Chrome"

        # Case 4: Predicted material key
        json_obj = {"predicted_material": "Stainless Steel", "source": "VLM"}
        assert extract_material_from_json(json_obj) == "Stainless Steel"

        # Case 5: Nested under multiple levels
        json_obj = {
            "analysis": {
                "prediction": {"material_name": "Textured Rubber"},
                "metadata": {"timestamp": "2024-01-01"},
            }
        }
        assert extract_material_from_json(json_obj) == "Textured Rubber"

    def test_edge_case_numeric_material_value(self) -> None:
        """Test that non-string material values are handled."""
        # If material value is not a string, should return None
        json_obj: dict[str, Any] = {"material": 12345}
        result = extract_material_from_json(json_obj)
        assert result is None

    def test_edge_case_list_material_value(self) -> None:
        """Test that list material values are handled."""
        json_obj: dict[str, Any] = {"material": ["option1", "option2"]}
        result = extract_material_from_json(json_obj)
        assert result is None
