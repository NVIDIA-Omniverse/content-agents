# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the freeform scenario's programmatic-score helper.

These lock the documented 0.4 / 0.3 / 0.3 weights for the
``upright`` / ``settled`` / ``finite_position`` components — the
previous implementation summed them with equal weight, which silently
diverged from the docstring contract and biased the freeform objective
the optimizer minimises.
"""

from __future__ import annotations

import pytest

from physics_agent.tuning.scenarios.freeform import (
    _normalize_observations,
    _normalize_weights,
    _score_programmatic_from_summary,
)


def _summary(
    *,
    fell_over: bool = False,
    settle_time_s: float | None = 0.5,
    duration_s: float = 1.0,
    final_position: tuple[float, float, float] = (0.0, 1.0, 0.0),
    n_samples: int = 30,
) -> dict[str, object]:
    return {
        "fell_over": fell_over,
        "settle_time_s": settle_time_s,
        "duration_s": duration_s,
        "final_position": list(final_position),
        "n_samples": n_samples,
    }


# ---------------------------------------------------------------------------
# Documented 0.4 / 0.3 / 0.3 weights — the contract the docstring advertises.
# ---------------------------------------------------------------------------


def test_all_three_components_pass_returns_one() -> None:
    """All checks pass → score 1.0 when ``upright`` is enabled."""
    score, critique = _score_programmatic_from_summary(
        _summary(),
        observations=["should stay upright"],
    )
    assert score == pytest.approx(1.0)
    assert "upright=pass" in critique
    assert "settled=pass" in critique
    assert "finite_position=pass" in critique


def test_only_upright_fails_costs_exactly_zero_point_four() -> None:
    """Failing only the upright check costs exactly its 0.4 weight."""
    score, _ = _score_programmatic_from_summary(
        _summary(fell_over=True),
        observations=["did the body stay upright"],
    )
    # earned = 0.0 (upright fail) + 0.3 (settled) + 0.3 (finite) = 0.6
    # total  = 1.0
    assert score == pytest.approx(0.6)


def test_only_settled_fails_costs_exactly_zero_point_three() -> None:
    """Failing only the settled check costs exactly its 0.3 weight."""
    score, _ = _score_programmatic_from_summary(
        _summary(settle_time_s=None),
        observations=["should stay upright"],
    )
    # earned = 0.4 (upright) + 0.0 (settled) + 0.3 (finite) = 0.7
    assert score == pytest.approx(0.7)


def test_only_finite_fails_costs_exactly_zero_point_three() -> None:
    """Failing only the finite-position check costs exactly its 0.3 weight."""
    score, _ = _score_programmatic_from_summary(
        _summary(final_position=(float("inf"), 0.0, 0.0)),
        observations=["should stay upright"],
    )
    # earned = 0.4 (upright) + 0.3 (settled) + 0.0 (finite) = 0.7
    assert score == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# Conditional ``upright`` component re-normalises when omitted
# ---------------------------------------------------------------------------


def test_upright_disabled_still_returns_one_when_remaining_pass() -> None:
    """When the prompt doesn't mention upright/stable/falling, the
    upright check is omitted. The remaining weights re-normalise so
    full pass on settled+finite still maps to 1.0 — toggling a check
    must never penalise a passing run.
    """
    score, critique = _score_programmatic_from_summary(
        _summary(),
        observations=["bounce around the room"],
    )
    assert score == pytest.approx(1.0)
    assert "upright" not in critique
    assert "settled=pass" in critique
    assert "finite_position=pass" in critique


def test_upright_disabled_settled_only_passes_renormalises_to_zero_point_five() -> None:
    """With upright omitted (remaining weights 0.3 + 0.3 = 0.6) and
    only settled passing, the score is 0.3 / 0.6 = 0.5.
    """
    score, _ = _score_programmatic_from_summary(
        _summary(final_position=(float("nan"), 0.0, 0.0)),
        observations=["just see what happens"],
    )
    assert score == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_zero_samples_returns_zero() -> None:
    """No trajectory data → score is 0 with a clear critique."""
    score, critique = _score_programmatic_from_summary(
        _summary(n_samples=0),
        observations=["upright"],
    )
    assert score == 0.0
    assert "no programmatic signal" in critique


# ---------------------------------------------------------------------------
# Weight validation (CodeRabbit R13 thread #6).
# ---------------------------------------------------------------------------


def test_normalize_weights_rejects_unknown_key() -> None:
    """Unknown keys must surface as a clear ValueError, not silently
    extend ``base`` and corrupt the optimizer signal."""
    with pytest.raises(ValueError, match="Unsupported freeform weight key"):
        _normalize_weights({"vision": 0.5}, vlm_available=True)


def test_normalize_weights_rejects_negative_weight() -> None:
    with pytest.raises(ValueError, match="finite non-negative"):
        _normalize_weights({"programmatic": -0.1}, vlm_available=True)


def test_normalize_weights_rejects_nan_weight() -> None:
    with pytest.raises(ValueError, match="finite non-negative"):
        _normalize_weights(
            {"programmatic": float("nan"), "vlm": 0.5}, vlm_available=True
        )


def test_normalize_weights_rejects_inf_weight() -> None:
    with pytest.raises(ValueError, match="finite non-negative"):
        _normalize_weights(
            {"programmatic": float("inf"), "vlm": 0.5}, vlm_available=True
        )


def test_normalize_weights_rejects_all_zero_total_when_vlm_available() -> None:
    """Both weights zero with VLM enabled → must raise. Otherwise the
    optimizer would consume an arbitrary tiebreak from epsilon clamps."""
    with pytest.raises(ValueError, match="At least one of freeform weights"):
        _normalize_weights({"programmatic": 0.0, "vlm": 0.0}, vlm_available=True)


def test_normalize_weights_rejects_bool() -> None:
    """``True`` is a Python int subclass but accepting it would let
    ``programmatic: yes`` (YAML coerces to True) silently authorize 1.0."""
    with pytest.raises(ValueError, match="must be a number"):
        _normalize_weights({"programmatic": True}, vlm_available=True)


def test_normalize_weights_unchanged_path_remains_05_05() -> None:
    """No weights supplied → 0.5 / 0.5 default; sanity-check the happy
    path isn't broken by the new validation."""
    weights = _normalize_weights(None, vlm_available=True)
    assert weights["programmatic"] == pytest.approx(0.5)
    assert weights["vlm"] == pytest.approx(0.5)


def test_normalize_weights_no_vlm_returns_programmatic_one() -> None:
    """VLM unavailable → programmatic = 1.0 regardless of inputs."""
    weights = _normalize_weights({"programmatic": 0.3, "vlm": 0.7}, vlm_available=False)
    assert weights == {"programmatic": 1.0, "vlm": 0.0}


# ---------------------------------------------------------------------------
# Observations normalization (CodeRabbit R13 thread #7).
# ---------------------------------------------------------------------------


def test_normalize_observations_scalar_string_stays_single_observation() -> None:
    """YAML scalar ``observations: "steady"`` must become ``["steady"]``,
    NOT ``['s', 't', 'e', 'a', 'd', 'y']``."""
    assert _normalize_observations("steady") == ["steady"]


def test_normalize_observations_list_pass_through() -> None:
    assert _normalize_observations(["a", "b"]) == ["a", "b"]


def test_normalize_observations_tuple_pass_through() -> None:
    assert _normalize_observations(("a", "b")) == ["a", "b"]


def test_normalize_observations_none_becomes_empty_list() -> None:
    assert _normalize_observations(None) == []


def test_normalize_observations_coerces_non_string_items_to_strings() -> None:
    assert _normalize_observations([1, 2.5, "stay"]) == ["1", "2.5", "stay"]


def test_normalize_observations_unexpected_shape_falls_through_to_str() -> None:
    """An unexpected dict shape becomes a one-item list of its repr —
    keeps audit artifacts honest instead of silently dropping the value."""
    out = _normalize_observations({"text": "should stay upright"})
    assert len(out) == 1
    assert "upright" in out[0]
