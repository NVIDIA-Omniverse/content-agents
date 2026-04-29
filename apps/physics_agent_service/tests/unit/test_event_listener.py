# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Physics Agent Service event listener progress mapping."""

from ...service.events.listener import FastAPIEventListener
from ...service.runtime.events import StepState


def test_predict_step_started_emits_running_progress():
    """The service should expose predict as the current step immediately."""
    listener = FastAPIEventListener("session-1234")

    event = listener._map_event_to_progress("step.started", {"step_name": "predict"})

    assert event is not None
    assert event.step == "predict"
    assert event.state == StepState.RUNNING
    assert event.percent == 0
