# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Utilities for parsing LLM responses."""

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


def extract_json_from_llm_response(
    response_text: str, expected_keys: list | None = None
) -> dict[str, Any] | None:
    """
    Extract JSON object from LLM response text.

    Handles various formats including:
    - JSON wrapped in markdown code blocks (```json or ```)
    - JSON with surrounding explanatory text
    - Plain JSON responses

    Args:
        response_text: The raw response text from the LLM
        expected_keys: Optional list of keys that should be present in the JSON

    Returns:
        Parsed JSON as a dictionary, or None if parsing fails
    """
    if not response_text:
        logger.error("Empty response text provided")
        return None

    try:
        # First try to find JSON in markdown code blocks
        code_block_match = re.search(
            r"```(?:json)?\s*\n(\{.*?\})\s*\n```", response_text, re.DOTALL
        )

        if code_block_match:
            json_str = code_block_match.group(1)
            logger.debug("Found JSON in markdown code block")
        else:
            # Fallback to finding any JSON object in the response
            # Try to find the first complete JSON object
            start_idx = response_text.find("{")
            if start_idx == -1:
                logger.error(f"No JSON found in LLM response: {response_text[:200]}...")
                return None

            # Find the matching closing brace
            brace_count = 0
            end_idx = start_idx
            for i in range(start_idx, len(response_text)):
                if response_text[i] == "{":
                    brace_count += 1
                elif response_text[i] == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i + 1
                        break

            if brace_count != 0:
                # Fallback to simple regex for incomplete JSON
                json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                else:
                    logger.error("No complete JSON object found")
                    return None
            else:
                json_str = response_text[start_idx:end_idx]

            logger.debug("Found JSON object in response text")

        # Clean up common VLM formatting issues before parsing
        # Some VLMs escape braces as {{ }} which breaks JSON parsing
        json_str = json_str.replace("{{", "{").replace("}}", "}")

        # Parse the JSON
        result = json.loads(json_str)

        # Ensure result is a dict
        if not isinstance(result, dict):
            logger.error(f"Expected dict but got {type(result)}")
            return None

        # Validate expected keys if provided
        if expected_keys:
            missing_keys = [key for key in expected_keys if key not in result]
            if missing_keys:
                logger.warning(f"JSON missing expected keys: {missing_keys}")

        return result

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON: {e}")
        logger.error(f"Response was: {response_text[:500]}...")
        return None
    except Exception as e:
        logger.error(f"Unexpected error parsing LLM response: {e}")
        return None


def extract_material_from_json(
    json_obj: dict[str, Any], possible_keys: list[str] | None = None
) -> str | None:
    """
    Recursively extract material value from JSON with flexible schema.

    Handles various JSON structures including:
    - Direct: {"material": "value"}
    - Nested: {{"material": "value"}}
    - Alternative keys: {"predicted_material": "value"}
    - Deep nesting: {"result": {"material": "value"}}

    Args:
        json_obj: The JSON object to extract from
        possible_keys: List of possible key names for material (default: common variations)

    Returns:
        The material string value, or None if not found

    Examples:
        >>> extract_material_from_json({"material": "Plastic Dark Blue"})
        "Plastic Dark Blue"
        >>> extract_material_from_json({{"material": "Plastic Dark Blue"}})
        "Plastic Dark Blue"
        >>> extract_material_from_json({"result": {"predicted_material": "Steel"}})
        "Steel"
    """
    if possible_keys is None:
        possible_keys = [
            "material",
            "predicted_material",
            "material_name",
            "name",
            "value",
            "result",
        ]

    # If json_obj is a string, return it directly
    if isinstance(json_obj, str):
        return json_obj

    # If not a dict, can't extract
    if not isinstance(json_obj, dict):
        logger.debug(f"Cannot extract material from non-dict: {type(json_obj)}")
        return None

    # Try direct key lookup first
    for key in possible_keys:
        if key in json_obj:
            value = json_obj[key]
            # If value is a string, we found it
            if isinstance(value, str):
                return value
            # If value is a dict, recurse into it
            elif isinstance(value, dict):
                result = extract_material_from_json(value, possible_keys)
                if result:
                    return result

    # If no direct match, check if there's a single nested dict
    # This handles cases like {{"material": "value"}}
    if len(json_obj) == 1:
        single_key = next(iter(json_obj))
        single_value = json_obj[single_key]

        # If the single value is a dict, recurse
        if isinstance(single_value, dict):
            result = extract_material_from_json(single_value, possible_keys)
            if result:
                return result

        # If it's a string and key matches material patterns
        if isinstance(single_value, str) and single_key in possible_keys:
            return single_value

    # Last resort: search all values recursively
    for value in json_obj.values():
        if isinstance(value, dict):
            result = extract_material_from_json(value, possible_keys)
            if result:
                return result

    logger.debug(f"No material found in JSON: {json_obj}")
    return None


def create_json_prompt_instructions() -> str:
    """
    Get standard instructions for prompting LLMs to return JSON.

    Returns:
        String with instructions to include in prompts
    """
    return (
        "Return ONLY a valid JSON object. Do not include any explanatory "
        "text, markdown formatting, or code blocks. Just the raw JSON."
    )
