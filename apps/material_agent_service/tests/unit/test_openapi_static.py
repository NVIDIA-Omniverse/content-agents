# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import yaml


def _load_openapi() -> dict:
    openapi_path = Path(__file__).parents[2] / "openapi.yaml"
    with open(openapi_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_static_openapi_documents_regenerate_and_event_log() -> None:
    spec = _load_openapi()

    regenerate = spec["paths"]["/pipeline/{session_id}/regenerate"]["post"]
    assert regenerate["requestBody"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/RegenerateRequest"
    }
    assert regenerate["responses"]["202"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/SessionCreated"
    }

    event_log = spec["paths"]["/pipeline/{session_id}/event-log"]["get"]
    assert (
        event_log["responses"]["200"]["content"]["application/json"]["schema"][
            "properties"
        ]["events"]["type"]
        == "array"
    )

    regenerate_schema = spec["components"]["schemas"]["RegenerateRequest"]
    assert regenerate_schema["required"] == ["steps"]
    assert regenerate_schema["properties"]["steps"]["items"] == {
        "$ref": "#/components/schemas/PipelineStep"
    }

    step_values = spec["components"]["schemas"]["PipelineStep"]["enum"]
    assert {"predict", "apply", "render"}.issubset(step_values)
