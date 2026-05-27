# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Example follow-up calls for a completed material-agent session."""

from apps.material_agent_service.client.client import MaterialAgentClient

client = MaterialAgentClient(base_url="http://localhost:8000")
session_id = "abc123"

regenerated = client.regenerate(
    session_id,
    steps=["predict"],
    user_prompt="Prefer brushed aluminum for exposed frame components",
)
print(regenerated["session_id"])

event_log = client.get_event_log(session_id)
print(event_log.get("total", len(event_log.get("events", []))))
