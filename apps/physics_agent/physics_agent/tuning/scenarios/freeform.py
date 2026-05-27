# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""freeform scenario evaluator: hybrid programmatic + VLM judge.

Pipeline (per ``OvPhysXBackend.evaluate`` trial):

  1. ``patch_physics_usd`` (existing utility).
  2. ``build_freeform_scene`` reads ``target.{gravity, surface,
     initial_pose, cameras}`` and authors the simulation USD.
  3. Daemon ``evaluate`` with ``initial_linear_velocity`` and
     ``initial_angular_velocity`` from ``target``.
  4. ``recorder.author_trajectory_usda`` → ``recording.usda``.
  5. Programmatic score from ``trajectory_summary`` against
     ``target.observations`` (e.g. "stayed upright", "settled within 1s").
  6. Optional VLM judge over rendered frames
     (``judge_callback(frames, user_prompt, observations)``).
  7. Combined score = ``weights["programmatic"] * programmatic_score +
     weights["vlm"] * vlm_score``. Default weights 0.5/0.5. Optimizer
     minimizes, so we return ``score = 1.0 - combined``.

Freeform is **NOT VLM-only**: programmatic trajectory metrics are
always computed, and they're the entire score when no VLM callback is
supplied (weights re-normalize). The VLM is one signal among several,
not the only driver.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from physics_agent.tuning.simulator import Simulator
    from physics_agent.tuning.types import Scenario


logger = logging.getLogger(__name__)


# (frames, user_prompt | None, observations) -> {score: float in [0,1],
#                                                 reasoning: str}
JudgeCallback = Callable[[list[Path], str | None, list[str]], dict[str, Any]]


def _normalize_observations(raw: Any) -> list[str]:
    """Normalize a YAML ``observations`` value into a list of strings.

    A YAML scalar like ``observations: "steady"`` is parsed as a Python
    ``str``; ``list(str)`` would explode it into its characters and
    silently break the upright/stable keyword scan + feed garbage tokens
    to the VLM judge_callback. Treat a scalar str as a single
    observation, a list / tuple as multiple, ``None`` / missing as
    empty, and any other shape as a one-item list of its repr (so the
    user still sees *something* in artifacts rather than a silent drop).
    (CodeRabbit R13 thread #7.)
    """
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list | tuple):
        return [str(item) for item in raw]
    return [str(raw)]


def _normalize_weights(
    weights: dict[str, float] | None, vlm_available: bool
) -> dict[str, float]:
    """Resolve weight defaults and re-normalize when VLM is unavailable.

    Default: ``{"programmatic": 0.5, "vlm": 0.5}``. When VLM is
    unavailable (no callback supplied or callback failed), we
    re-normalize to ``{"programmatic": 1.0, "vlm": 0.0}`` so the
    score is purely programmatic — no silent down-weighting.
    """
    base = {"programmatic": 0.5, "vlm": 0.5}
    if weights:
        # Validate up-front: unknown keys, negative or non-finite values
        # would silently distort the optimizer signal if we let the
        # later normalize/clamp absorb them. (CodeRabbit R13 thread #6.)
        unknown = set(weights) - set(base)
        if unknown:
            raise ValueError(
                "Unsupported freeform weight key(s) "
                f"{sorted(unknown)}; expected subset of {sorted(base)}."
            )
        for name, value in weights.items():
            if not isinstance(value, int | float) or isinstance(value, bool):
                raise ValueError(
                    f"Freeform weight {name!r} must be a number, "
                    f"got {type(value).__name__}: {value!r}"
                )
            if not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError(
                    f"Freeform weight {name!r} must be a finite non-negative "
                    f"number, got {value!r}"
                )
        base.update(weights)
    if not vlm_available:
        return {"programmatic": 1.0, "vlm": 0.0}
    total = float(base["programmatic"]) + float(base["vlm"])
    if total <= 0:
        raise ValueError(
            "At least one of freeform weights {'programmatic', 'vlm'} must be "
            f"positive; got programmatic={base['programmatic']!r}, "
            f"vlm={base['vlm']!r}."
        )
    return {
        "programmatic": float(base["programmatic"]) / total,
        "vlm": float(base["vlm"]) / total,
    }


def _score_programmatic_from_summary(
    summary: dict[str, Any], observations: list[str]
) -> tuple[float, str]:
    """Map ``trajectory_summary`` + ``observations`` to a score in [0,1]
    where 1.0 == fully-satisfies-prompt.

    v1 strategy is intentionally simple — it surfaces structural
    signals that almost any freeform observation cares about and
    leaves the nuance to the VLM. Each enabled component contributes
    its **weight** when it passes; the final score normalises by the
    sum of enabled weights so toggling the conditional ``upright``
    component never drops a passing run below the analogous score it
    would have earned with the component absent:

      • body did NOT fall over (when observations mention "upright" /
        "stable" / "didn't fall") → weight 0.4
      • body settled before the trajectory ended → weight 0.3
      • body did NOT escape the scene (no infinite / NaN positions) →
        weight 0.3

    Each check is a hard yes/no. Returns ``(score, critique)`` so the
    surrounding evaluator can include the critique in its result dict
    for audit.
    """
    obs_text = " ".join(o.lower() for o in observations)
    final_pos = summary.get("final_position") or [0.0, 0.0, 0.0]
    fell_over = bool(summary.get("fell_over", False))
    settle_time_s = summary.get("settle_time_s")
    duration_s = float(summary.get("duration_s") or 0.0)
    n_samples = int(summary.get("n_samples") or 0)

    # (name, passed, weight) — weights match the documented contract.
    components: list[tuple[str, bool, float]] = []

    # Component 1: stayed upright (when the prompt cares about that)
    cares_about_upright = any(
        kw in obs_text for kw in ("upright", "stable", "fall", "topple", "tip")
    )
    if cares_about_upright:
        components.append(("upright", not fell_over, 0.4))

    # Component 2: settled before trajectory ended
    settled = settle_time_s is not None and float(settle_time_s) <= duration_s
    components.append(("settled", bool(settled), 0.3))

    # Component 3: position is finite (no NaN/Inf escape).
    # ``math`` is imported at module level (see line 30).
    finite = all(math.isfinite(float(v)) for v in final_pos)
    components.append(("finite_position", bool(finite), 0.3))

    if not components or n_samples == 0:
        return 0.0, "no programmatic signal extracted"

    total_weight = sum(weight for _, _, weight in components)
    earned = sum(weight for _, ok, weight in components if ok)
    # ``total_weight`` is always > 0 here (we always append settled +
    # finite_position above), but guard divide-by-zero anyway.
    score = earned / total_weight if total_weight > 0 else 0.0
    critique = "; ".join(
        f"{name}={'pass' if ok else 'fail'}" for name, ok, _ in components
    )
    return float(score), critique


def evaluate(
    params: dict[str, float],
    scenario: Scenario,
    physics_usd: Path,
    *,
    seed: int,
    simulator: Simulator,
    work_dir: Path | None = None,
    judge_callback: JudgeCallback | None = None,
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Run one freeform tune trial against a physics simulator.

    See module docstring for the pipeline. Returns the dict shape
    consumed by ``backend.evaluate``: ``score`` (lower is better),
    ``programmatic_score``, ``vlm_score``, ``reasoning``, ``frames``,
    ``trajectory``, ``scene_usd``, ``recording_usda``, ``weights_used``.
    """
    from world_understanding.functions.physics.trajectory import (
        trajectory_summary,
    )

    from physics_agent.recording import author_trajectory_usda
    from physics_agent.tuning.scenario_resolution import get_resolved_bindings
    from physics_agent.tuning.scenarios._scene_builder import (
        build_freeform_scene,
    )
    from physics_agent.tuning.usd_patch import patch_physics_usd

    work = (
        Path(work_dir)
        if work_dir is not None
        else Path(physics_usd).parent / ".tune_scenes"
    )
    trial_dir = work / f"trial_seed_{int(seed)}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    # 1. Patch params.
    patched_path = trial_dir / "patched_physics.usda"
    resolved_bindings = get_resolved_bindings(scenario)
    if resolved_bindings is None:
        patch_physics_usd(Path(physics_usd), patched_path, dict(params))
    else:
        patch_physics_usd(
            Path(physics_usd),
            patched_path,
            dict(params),
            bindings=resolved_bindings,
        )

    # 2. Build scene USD from the target dict.
    target = dict(scenario.target or {})
    duration_s = float(target.get("duration_s", 2.0))
    sample_fps = int(target.get("sample_fps", 30))

    scene_path = trial_dir / "scene.usda"
    scene_info = build_freeform_scene(
        patched_path,
        scene_path,
        target=target,
    )

    # 3. Simulator evaluate with the LLM-authored initial conditions.
    init_lin = target.get("initial_velocity")
    init_ang = target.get("initial_angular_velocity")
    response = simulator.evaluate(
        scene_usd=scene_path,
        body_pattern=scene_info["body_pattern"],
        duration_s=duration_s,
        dt=float(target.get("dt", 1.0 / 240.0)),
        sample_fps=sample_fps,
        initial_linear_velocity=tuple(init_lin) if init_lin else None,
        initial_angular_velocity=tuple(init_ang) if init_ang else None,
    )
    trajectory = response["trajectory"]

    # 4. Recording.usda for VLM and audit.
    recording_path: Path | None = trial_dir / "recording.usda"
    try:
        author_trajectory_usda(
            scene_path,
            trajectory,
            scene_info["body_prim_path"],
            recording_path,
            fps=sample_fps,
            max_duration_s=duration_s,
        )
    except Exception as exc:
        logger.warning(
            "freeform: failed to author recording.usda (seed=%d): %s",
            int(seed),
            exc,
        )
        recording_path = None

    # 5. Programmatic score from trajectory summary + observations.
    # The simulator trajectory is already (t, pose7, vel6) tuples — pass
    # through unchanged; trajectory_summary reads velocity directly
    # from each sample.
    #
    # Round 12 (CX P2#4): pass the stage's actual up-axis through to
    # ``trajectory_summary``. ``trajectory_summary`` defaults to Y-up
    # when ``world_up`` is omitted, which mis-classifies a yaw spin on
    # a Z-up asset as ``fell_over=True`` and corrupts both the
    # programmatic score and any ``observations`` that mention
    # upright/stable. The scene builder records the actual axis under
    # ``scene_info["world_up"]``; if it's missing for any reason
    # (programmatic-only callers building scene_info dicts by hand) we
    # fall back to the trajectory_summary default.
    observations = _normalize_observations(target.get("observations"))
    world_up = scene_info.get("world_up")
    summary = trajectory_summary(trajectory, world_up=world_up)
    programmatic_score, prog_critique = _score_programmatic_from_summary(
        summary, observations
    )

    # 6. Optional video rendering and VLM judge.
    #
    # Rendering produces inspection-ready PNG/mp4 evidence and is the
    # input to the optional VLM judge. The two are independent triggers:
    # ``record_video`` ("off" / "always", default "off") writes the
    # render artifacts unconditionally on every trial it fires; the VLM
    # judge runs only when ``judge_callback`` was passed and rendering
    # produced frames. Without a record_video opt-in, the VLM judge
    # itself implicitly forces a render so existing behavior is
    # preserved when callers wire up the callback.
    #
    # Rendering belongs to ``world_understanding.functions.graphics``;
    # imported lazily so freeform stays usable when the helper isn't
    # installed (PR #66). When rendering is unavailable the VLM step
    # is skipped cleanly and the score collapses to programmatic-only
    # via ``_normalize_weights(..., vlm_available=False)``.
    record_video_mode = str(target.get("record_video", "off")).lower()
    record_video_on = record_video_mode in {"end_of_tune", "always"}
    vlm_score: float | None = None
    vlm_reasoning: str = ""
    frames: list[Path] = []
    vlm_available = False
    video_block: dict[str, Any] | None = None
    needs_render = (judge_callback is not None) or record_video_on
    if needs_render and recording_path is not None:
        try:
            from world_understanding.functions.graphics import (
                render_time_sampled_usd,
            )
        except ImportError:
            render_time_sampled_usd = None  # type: ignore[assignment]
            skip_reason = (
                "render_time_sampled_usd is not installed (see issue #50 / PR #66)"
            )
            if judge_callback is not None:
                vlm_reasoning = (
                    f"VLM unavailable: {skip_reason}; freeform falls back to "
                    "programmatic-only scoring."
                )
            if record_video_on:
                video_block = {
                    "mode": record_video_mode,
                    "status": "skipped",
                    "reason": skip_reason,
                }

        if render_time_sampled_usd is not None:
            try:
                frames = render_time_sampled_usd(
                    recording_path,
                    trial_dir / "render",
                    renderer=str(
                        target.get("video_renderer")
                        or target.get("vlm_renderer", "ovrtx")
                    ),
                    cameras=scene_info.get("camera_paths"),
                    fps=sample_fps,
                    max_duration_seconds=duration_s or 2.0,
                    image_width=int(target.get("video_image_width", 512)),
                    image_height=int(target.get("video_image_height", 512)),
                    num_sensor_updates=int(target.get("video_sensor_updates", 32)),
                    render_mode=str(target.get("video_render_mode", "rt2")),
                )
                if record_video_on:
                    video_block = {
                        "mode": record_video_mode,
                        "status": "ok" if frames else "no_frames",
                        "render_dir": str(trial_dir / "render"),
                        "frame_count": len(frames),
                    }
                if judge_callback is not None and frames:
                    verdict = judge_callback(
                        frames,
                        target.get("description"),
                        observations,
                    )
                    raw_score = verdict.get("score")
                    if isinstance(raw_score, int | float):
                        vlm_score = max(0.0, min(1.0, float(raw_score)))
                        vlm_reasoning = str(verdict.get("reasoning") or "")
                        vlm_available = True
            except Exception as exc:
                # Same intent as drop_settle: capture the exception
                # message in addition to the type, and log the
                # traceback, so silent render failures (frame-cap
                # mismatch, missing camera, ovrtx returning 0 images)
                # surface in history.jsonl + server logs.
                logger.exception(
                    "freeform: render/judge failed for trial seed=%d",
                    seed,
                )
                if judge_callback is not None:
                    vlm_reasoning = f"VLM unavailable: {type(exc).__name__}: {exc}"
                    vlm_available = False
                if record_video_on:
                    video_block = {
                        "mode": record_video_mode,
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }

    # 7. Combine weighted, then map to "lower is better" optimizer
    # objective: score = 1.0 - combined.
    used_weights = _normalize_weights(weights, vlm_available)
    if vlm_available and vlm_score is not None:
        combined = (
            used_weights["programmatic"] * programmatic_score
            + used_weights["vlm"] * vlm_score
        )
    else:
        combined = programmatic_score
    combined_clamped = max(0.0, min(1.0, float(combined)))
    score = 1.0 - combined_clamped

    out: dict[str, Any] = {
        "score": float(score),
        "combined_score": float(combined_clamped),
        "programmatic_score": float(programmatic_score),
        "programmatic_critique": prog_critique,
        "vlm_score": float(vlm_score) if vlm_score is not None else None,
        "reasoning": vlm_reasoning,
        "frames": [str(p) for p in frames],
        "trajectory": trajectory,
        "trajectory_summary": summary,
        "scene_usd": str(scene_path),
        "patched_usd": str(patched_path),
        "recording_usda": str(recording_path) if recording_path else None,
        "weights_used": used_weights,
        "metric": str(scenario.metric),
    }
    if video_block is not None:
        out["record_video"] = video_block
    return out


__all__ = ["evaluate", "JudgeCallback"]
