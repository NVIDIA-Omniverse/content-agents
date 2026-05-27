# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from physics_agent.functions.prediction_schema import unwrap_output_key_payload


def test_unwrap_output_key_payload_returns_non_dict_payload_unchanged():
    assert unwrap_output_key_payload("raw response", "classification") == "raw response"
    assert unwrap_output_key_payload(None, "classification") is None


def test_unwrap_output_key_payload_keeps_canonical_payload():
    payload = {
        "component_type": "link",
        "physical_properties": {"density": 2700},
    }

    assert unwrap_output_key_payload(payload, "classification") is payload


def test_unwrap_output_key_payload_handles_one_level_output_wrapper():
    payload = {
        "classification": {
            "component_type": "optical",
            "physical_properties": {"density": 2500},
        }
    }

    assert unwrap_output_key_payload(payload, "classification") == {
        "component_type": "optical",
        "physical_properties": {"density": 2500},
    }


def test_unwrap_output_key_payload_preserves_wrapper_original_response():
    payload = {
        "classification": {
            "component_type": "optical",
            "physical_properties": {"density": 2500},
        },
        "original_response": "raw VLM output",
    }

    assert unwrap_output_key_payload(payload, "classification") == {
        "component_type": "optical",
        "physical_properties": {"density": 2500},
        "original_response": "raw VLM output",
    }


def test_unwrap_output_key_payload_does_not_overwrite_nested_original_response():
    payload = {
        "classification": {
            "component_type": "optical",
            "physical_properties": {"density": 2500},
            "original_response": "nested raw output",
        },
        "original_response": "wrapper raw output",
    }

    assert unwrap_output_key_payload(payload, "classification") == {
        "component_type": "optical",
        "physical_properties": {"density": 2500},
        "original_response": "nested raw output",
    }


def test_unwrap_output_key_payload_keeps_non_dict_wrapper_value():
    payload = {"classification": "unknown", "original_response": "raw VLM output"}

    assert unwrap_output_key_payload(payload, "classification") is payload
