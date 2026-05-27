# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the NL → Scenario interpreter at
``physics_agent.tasks.interpret_user_prompt_tuning``.

Every test here monkeypatches the ``generate_chat_response`` binding inside the
interpreter module so no real LLM call is made. ``chat_model`` is always
supplied explicitly (as a stub object) to short-circuit the lazy default
chat-model resolution path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from physics_agent.tasks.interpret_user_prompt_tuning import (
    InterpreterError,
    infer_scenario_from_prompt,
)
from physics_agent.tuning.types import Scenario

# ---------------------------------------------------------------------------
# Test fakes
# ---------------------------------------------------------------------------


class _RecordingChat:
    """Callable stand-in for ``generate_chat_response``.

    Wraps a list of canned responses (each a dict with ``response`` or
    ``error``). Each call pops the next response and records the kwargs the
    interpreter passed in, so tests can assert on retry behaviour.
    """

    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        *,
        chat_model: Any,
        prompt: str,
        system_prompt: str,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "chat_model": chat_model,
                "prompt": prompt,
                "system_prompt": system_prompt,
            }
        )
        if not self.responses:
            raise AssertionError("RecordingChat ran out of canned responses")
        return self.responses.pop(0)


class _ChatStub:
    """Minimal chat-model stub. ``model_name`` lands in the audit record's _meta."""

    model_name = "test-model"


def _patch_chat(monkeypatch: pytest.MonkeyPatch, fake: _RecordingChat) -> None:
    monkeypatch.setattr(
        "physics_agent.tasks.interpret_user_prompt_tuning.generate_chat_response",
        fake,
    )


def _drop_settle_payload() -> dict[str, Any]:
    return {
        "name": "drop_settle",
        "metric": "settle_distance",
        "target": {"drop_height_m": 0.5, "duration_s": 2.0, "gravity": -9.81},
        "parameters": [
            {"name": "restitution", "min": 0.4, "max": 0.95},
            {"name": "mass_scale", "min": 0.7, "max": 1.3},
        ],
    }


def _freeform_payload() -> dict[str, Any]:
    return {
        "name": "freeform",
        "metric": "judge_score",
        "target": {
            "description": "spin a top on a smooth surface",
            "duration_s": 3.0,
            "gravity": -9.81,
            "initial_pose": {
                "position": [0.0, 0.5, 0.0],
                "rotation": [0.0, 0.0, 0.0],
            },
            "initial_velocity": [0.0, 0.0, 0.0],
            "initial_angular_velocity": [0.0, 30.0, 0.0],
            "surface": {"friction": 0.2},
            "observations": ["did the top fall over"],
        },
        "parameters": [
            {"name": "dynamic_friction", "min": 0.05, "max": 0.4},
            {"name": "mass_scale", "min": 0.5, "max": 1.5},
        ],
    }


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


def test_happy_path_drop_settle(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _RecordingChat([{"response": json.dumps(_drop_settle_payload())}])
    _patch_chat(monkeypatch, fake)

    sc = infer_scenario_from_prompt(
        "make this object bouncy",
        chat_model=_ChatStub(),
    )

    assert isinstance(sc, Scenario)
    assert sc.name == "drop_settle"


def test_happy_path_freeform(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _RecordingChat([{"response": json.dumps(_freeform_payload())}])
    _patch_chat(monkeypatch, fake)

    sc = infer_scenario_from_prompt(
        "spin a top on a smooth surface",
        chat_model=_ChatStub(),
    )

    assert sc.name == "freeform"


def test_backend_allowlist_shapes_newton_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "name": "drop_settle",
        "metric": "settle_distance",
        "target": {"drop_height_m": 0.5, "duration_s": 2.0, "gravity": -9.81},
        "parameters": [
            {"name": "contact_ke", "min": 10000.0, "max": 100000.0},
            {"name": "contact_kd", "min": 0.0, "max": 1000.0},
        ],
    }
    fake = _RecordingChat([{"response": json.dumps(payload)}])
    _patch_chat(monkeypatch, fake)

    sc = infer_scenario_from_prompt(
        "make this object bouncy",
        chat_model=_ChatStub(),
        backend_name="newton",
        supported_param_keys=(
            "mass_scale",
            "dynamic_friction",
            "contact_ke",
            "contact_kd",
        ),
    )

    assert tuple(p.name for p in sc.params) == ("contact_ke", "contact_kd")
    system_prompt = fake.calls[0]["system_prompt"]
    user_prompt = fake.calls[0]["prompt"]
    assert "Active backend: newton" in system_prompt
    assert "contact_ke" in system_prompt
    assert "contact_kd" in system_prompt
    assert "restitution biased high" not in system_prompt
    assert "Allowed tunable parameters for this backend" in user_prompt
    assert "restitution" not in user_prompt


def test_backend_allowlist_retries_llm_disallowed_parameter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bad = _drop_settle_payload()
    good = {
        "name": "drop_settle",
        "metric": "settle_distance",
        "target": {"drop_height_m": 0.5, "duration_s": 2.0, "gravity": -9.81},
        "parameters": [
            {"name": "contact_ke", "min": 10000.0, "max": 100000.0},
            {"name": "contact_kd", "min": 0.0, "max": 1000.0},
        ],
    }
    fake = _RecordingChat(
        [
            {"response": json.dumps(bad)},
            {"response": json.dumps(good)},
        ]
    )
    _patch_chat(monkeypatch, fake)

    sc = infer_scenario_from_prompt(
        "make this object bouncy",
        chat_model=_ChatStub(),
        backend_name="newton",
        supported_param_keys=(
            "mass_scale",
            "dynamic_friction",
            "contact_ke",
            "contact_kd",
        ),
    )

    assert tuple(p.name for p in sc.params) == ("contact_ke", "contact_kd")
    assert len(fake.calls) == 2
    assert "not allowed for the active backend" in fake.calls[1]["prompt"]


def test_backend_allowlist_rejects_disallowed_override_before_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _RecordingChat([{"response": json.dumps(_drop_settle_payload())}])
    _patch_chat(monkeypatch, fake)

    with pytest.raises(InterpreterError, match="scenario_override contains"):
        infer_scenario_from_prompt(
            "make this object bouncy",
            scenario_override={
                "parameters": [{"name": "restitution", "min": 0.4, "max": 0.95}]
            },
            chat_model=_ChatStub(),
            backend_name="newton",
            supported_param_keys=(
                "mass_scale",
                "dynamic_friction",
                "contact_ke",
                "contact_kd",
            ),
        )

    assert fake.calls == []


# ---------------------------------------------------------------------------
# Override merging
# ---------------------------------------------------------------------------


def test_explicit_scenario_override_wins_on_scalar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _drop_settle_payload()
    payload["metric"] = "settle_distance"
    fake = _RecordingChat([{"response": json.dumps(payload)}])
    _patch_chat(monkeypatch, fake)

    sc = infer_scenario_from_prompt(
        "make this object bouncy",
        scenario_override={"metric": "max_bounce_height"},
        chat_model=_ChatStub(),
    )

    # The override flips the metric onto a different valid registry
    # entry — proves user-supplied values win over the LLM payload.
    # Before round 5 the test used arbitrary strings here, but
    # ``parse_scenario`` now gates drop_settle metrics on the registry
    # so we use two real names instead.
    assert sc.metric == "max_bounce_height"


def test_explicit_override_wins_on_nested_dict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _drop_settle_payload()
    payload["target"] = {"drop_height_m": 0.5, "duration_s": 2.0, "gravity": -9.81}
    fake = _RecordingChat([{"response": json.dumps(payload)}])
    _patch_chat(monkeypatch, fake)

    sc = infer_scenario_from_prompt(
        "make this object bouncy",
        scenario_override={"target": {"duration_s": 7.5}},
        chat_model=_ChatStub(),
    )

    # Override wins on duration_s; LLM-provided drop_height_m and gravity stay.
    assert sc.target == {"drop_height_m": 0.5, "duration_s": 7.5, "gravity": -9.81}


def test_cross_kind_override_without_parameters_raises_clear_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forcing a different scenario kind without supplying parameters
    must raise a clear ``InterpreterError`` instead of falling out as
    an opaque schema-validation failure.

    The interpreter intentionally drops the LLM-authored fields when a
    cross-kind override is in play (the LLM's freeform parameters can't
    be reused for drop_settle and vice versa). When the override also
    has no ``parameters`` key, the merged dict becomes ``{"name":
    "drop_settle"}`` which fails ``parse_scenario`` for the missing
    parameters list. Surfacing the actionable advice up-front makes the
    constraint visible without forcing a user to read parser internals.
    """
    fake = _RecordingChat([{"response": json.dumps(_freeform_payload())}])
    _patch_chat(monkeypatch, fake)

    with pytest.raises(InterpreterError, match="scenario_override forced a different"):
        infer_scenario_from_prompt(
            "spin a top",
            scenario_override={"name": "drop_settle"},
            chat_model=_ChatStub(),
        )


def test_list_override_replaces_wholesale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _drop_settle_payload()
    payload["parameters"] = [{"name": "mass_scale", "min": 0.5, "max": 2.0}]
    fake = _RecordingChat([{"response": json.dumps(payload)}])
    _patch_chat(monkeypatch, fake)

    sc = infer_scenario_from_prompt(
        "make this object bouncy",
        scenario_override={
            "parameters": [{"name": "restitution", "min": 0.0, "max": 1.0}],
        },
        chat_model=_ChatStub(),
    )

    assert tuple(p.name for p in sc.params) == ("restitution",)


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------


def test_json_extraction_strips_preamble(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = "Sure, here is the JSON:\n" + json.dumps(_drop_settle_payload())
    fake = _RecordingChat([{"response": raw}])
    _patch_chat(monkeypatch, fake)

    sc = infer_scenario_from_prompt(
        "make this object bouncy",
        chat_model=_ChatStub(),
    )

    assert sc.name == "drop_settle"


def test_json_extraction_strips_code_fences(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = "```json\n" + json.dumps(_drop_settle_payload()) + "\n```"
    fake = _RecordingChat([{"response": raw}])
    _patch_chat(monkeypatch, fake)

    sc = infer_scenario_from_prompt(
        "make this object bouncy",
        chat_model=_ChatStub(),
    )

    assert sc.name == "drop_settle"


# ---------------------------------------------------------------------------
# Retry behaviour
# ---------------------------------------------------------------------------


def test_retry_on_parser_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    bad = {"name": "rolling_ball", "parameters": [{"name": "mass_scale"}]}
    good = _drop_settle_payload()
    fake = _RecordingChat(
        [{"response": json.dumps(bad)}, {"response": json.dumps(good)}]
    )
    _patch_chat(monkeypatch, fake)

    sc = infer_scenario_from_prompt(
        "make this object bouncy",
        chat_model=_ChatStub(),
    )

    assert sc.name == "drop_settle"
    assert len(fake.calls) == 2
    # The retry's user prompt must include the parser error as context — the
    # second call's prompt is strictly longer than the first.
    assert fake.calls[1]["prompt"] != fake.calls[0]["prompt"]
    assert "previous response failed" in fake.calls[1]["prompt"]


def test_two_consecutive_failures_raise_interpreter_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bad = {"name": "rolling_ball", "parameters": [{"name": "mass_scale"}]}
    fake = _RecordingChat(
        [{"response": json.dumps(bad)}, {"response": json.dumps(bad)}]
    )
    _patch_chat(monkeypatch, fake)

    with pytest.raises(InterpreterError) as exc_info:
        infer_scenario_from_prompt(
            "make this object bouncy",
            chat_model=_ChatStub(),
        )

    # Chained from the underlying ScenarioParseError on the final attempt.
    from physics_agent.tuning.scenario import ScenarioParseError

    assert isinstance(exc_info.value.__cause__, ScenarioParseError)


def test_llm_error_response_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _RecordingChat([{"error": "rate limit"}, {"error": "rate limit"}])
    _patch_chat(monkeypatch, fake)

    with pytest.raises(InterpreterError):
        infer_scenario_from_prompt(
            "make this object bouncy",
            chat_model=_ChatStub(),
        )

    # Both attempts were used (retry path triggered on first error).
    assert len(fake.calls) == 2


# ---------------------------------------------------------------------------
# Audit record (write-only — no caching, no read-back)
# ---------------------------------------------------------------------------


def test_audit_record_written(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake = _RecordingChat([{"response": json.dumps(_drop_settle_payload())}])
    _patch_chat(monkeypatch, fake)

    infer_scenario_from_prompt(
        "make this object bouncy",
        chat_model=_ChatStub(),
        audit_dir=tmp_path,
    )

    audit_path = tmp_path / "inferred_scenario.json"
    assert audit_path.exists()
    record = json.loads(audit_path.read_text(encoding="utf-8"))
    # Audit metadata: just enough to identify which prompt/model
    # produced the scenario. No cache schema version, no canonical
    # override hash, no physics_usd identity — those existed solely
    # to validate cache hits, and there is no cache.
    meta = record["_meta"]
    assert meta["user_prompt"] == "make this object bouncy"
    assert meta["model"] == "test-model"
    assert meta["merged_from_explicit"] is False
    # Validated scenario fields live at the top level alongside _meta.
    assert record["name"] == "drop_settle"


def test_audit_record_marks_explicit_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = _RecordingChat([{"response": json.dumps(_drop_settle_payload())}])
    _patch_chat(monkeypatch, fake)

    infer_scenario_from_prompt(
        "make this object bouncy",
        # Round 5 tightened parse_scenario to gate drop_settle metrics
        # on the registry, so the override has to land on a registered
        # name. The audit-marker test only cares that
        # merged_from_explicit reads True; the specific value is not
        # the load-bearing assertion.
        scenario_override={"metric": "max_bounce_height"},
        chat_model=_ChatStub(),
        audit_dir=tmp_path,
    )

    record = json.loads((tmp_path / "inferred_scenario.json").read_text())
    assert record["_meta"]["merged_from_explicit"] is True


def test_audit_write_failure_does_not_fail_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = _RecordingChat([{"response": json.dumps(_drop_settle_payload())}])
    _patch_chat(monkeypatch, fake)

    # Point audit_dir at a regular file so mkdir() raises OSError. The
    # interpreter must still return a valid Scenario (best-effort audit).
    blocker = tmp_path / "blocker"
    blocker.write_text("not-a-dir", encoding="utf-8")

    sc = infer_scenario_from_prompt(
        "make this object bouncy",
        chat_model=_ChatStub(),
        audit_dir=blocker,
    )

    assert sc.name == "drop_settle"


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_empty_user_prompt_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _RecordingChat([])
    _patch_chat(monkeypatch, fake)

    for prompt in ("", "   "):
        with pytest.raises(
            InterpreterError, match="user_prompt must be a non-empty string"
        ):
            infer_scenario_from_prompt(prompt, chat_model=_ChatStub())

    # LLM was never called.
    assert fake.calls == []


def test_non_dict_scenario_override_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _RecordingChat([])
    _patch_chat(monkeypatch, fake)

    with pytest.raises(InterpreterError):
        infer_scenario_from_prompt(
            "make this object bouncy",
            scenario_override="not a dict",  # type: ignore[arg-type]
            chat_model=_ChatStub(),
        )


# ---------------------------------------------------------------------------
# physics_usd handling
# ---------------------------------------------------------------------------


def test_physics_usd_basename_in_prompt_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _RecordingChat([{"response": json.dumps(_drop_settle_payload())}])
    _patch_chat(monkeypatch, fake)

    infer_scenario_from_prompt(
        "make this object bouncy",
        chat_model=_ChatStub(),
        physics_usd=Path("/abs/foo.usda"),
    )

    sent_prompt = fake.calls[0]["prompt"]
    assert "foo.usda" in sent_prompt
    assert "/abs/" not in sent_prompt
