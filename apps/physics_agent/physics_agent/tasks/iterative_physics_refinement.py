# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Iterative physics-tune refinement task.

Ports the shape of
``apps/material_agent/material_agent/tasks/iteration.py``'s
``IterativeRefinementTask`` to the physics-agent: a
``while iteration < max_iterations`` loop that runs

    tune  →  judge_tune  →  scenario_refine

per iteration, exiting on ``judge.decision == "approve"`` or hitting
the cap. Same context-key contract as material's iteration so future
cross-domain code can reuse the convention.

Inputs (constructor):
    user_prompt
    initial_scenario_yaml (Path)
    physics_usd (Path)
    output_dir (Path)
    engine, optimizer, max_trials, seed
    max_iterations
    score_threshold
    chat_model (optional; refine degrades without a chat model, but the
        iterative judge fails closed when no VLM verdict is available)
    force_record_video (optional, "off"|"end_of_tune"|"always"|None) ⇒
        when set, every iteration's scenario.yaml gets ``record_video``
        rewritten to this value, overriding both the initial YAML and
        any LLM-refined value. Default ``None`` honors the YAML.
    render_winning_trial (default True) ⇒ post-tune render of the best
        trial's recording.usda into ``iter_N/render/`` so each iteration
        produces one mp4 even when per-trial rendering is suppressed.

Per-iteration on-disk layout::

    output_dir/
        iter_1/
            scenario.yaml
            best_params.json
            history.jsonl
            judge_result.json
            tune_results.json (from run_tune)
            report.md (from run_tune)
            tuned_physics.usda (from run_tune, optional)
            render/<trial>/  (when scenario.target.record_video is on)
        iter_2/...
        final/
            (copy / link of the winning iteration's artifacts)
        refine_summary.json (loop-level summary)

Context-key contract emitted to the listener (same as material's loop):

    continue_iteration    : True → next iter; False → terminate
    judge_reasoning       : str
    judge_score           : float
    iteration_count       : int

This task is **stateless across runs** — call ``run()`` once per
``physics-agent refine`` invocation. The constructor pins all loop
parameters, ``run()`` returns a result dict consumed by the CLI.
"""

from __future__ import annotations

import json
import logging
import math
import re
import shutil
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from world_understanding.agentic.events import get_listener

from physics_agent.tasks.judge_tune import run_tune_judge
from physics_agent.tasks.scenario_refine import RefineResult, run_scenario_refine
from physics_agent.tuning.artifacts import ARTIFACT_VISUAL_COMPARISON
from physics_agent.tuning.backend import get_backend
from physics_agent.tuning.capabilities import capabilities_for_backend
from physics_agent.tuning.errors import NewtonUnavailableError, OvPhysXUnavailableError
from physics_agent.tuning.scenario import load_scenario
from physics_agent.tuning.types import (
    Scenario,
    TrialRecord,
    TuneInput,
    TuneOutput,
)
from physics_agent.tuning.visual_evidence import (
    JudgeVisualEvidence,
    has_reference_media,
    prepare_reference_media,
    resolve_default_judge_vlm,
    write_comparison_contact_sheet,
)

logger = logging.getLogger(__name__)

__all__ = [
    "IterationRecord",
    "IterativePhysicsRefinementResult",
    "IterativePhysicsRefinementTask",
]


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class IterationRecord:
    """One iteration's snapshot — what happened, what got written."""

    iteration: int
    iteration_dir: Path
    scenario_yaml_path: Path
    tune_output_dir: Path
    best_params: dict[str, float]
    best_score: float
    n_trials: int
    judge_decision: str  # "approve" | "continue" | "skipped"
    judge_score: float
    judge_reasoning: str
    judge_llm_unavailable: bool
    refine_llm_unavailable: bool
    refine_reasoning: str
    metric_name: str
    metric_value: float | None  # the raw physical metric (e.g. peak height)
    cancelled: bool = False
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        # Coerce non-finite scores (the error path stores
        # ``best_score=float("inf")``) into ``None`` so the artifact is
        # parseable by strict JSON consumers — the Python default of
        # ``allow_nan=True`` would emit the bareword ``Infinity``.
        import math as _math

        def _finite_or_none(value: float) -> float | None:
            f = float(value)
            return f if _math.isfinite(f) else None

        return {
            "iteration": int(self.iteration),
            "iteration_dir": str(self.iteration_dir),
            "scenario_yaml_path": str(self.scenario_yaml_path),
            "tune_output_dir": str(self.tune_output_dir),
            "best_params": dict(self.best_params),
            "best_score": _finite_or_none(self.best_score),
            "n_trials": int(self.n_trials),
            "judge_decision": self.judge_decision,
            "judge_score": _finite_or_none(self.judge_score),
            "judge_reasoning": self.judge_reasoning,
            "judge_llm_unavailable": bool(self.judge_llm_unavailable),
            "refine_llm_unavailable": bool(self.refine_llm_unavailable),
            "refine_reasoning": self.refine_reasoning,
            "metric_name": self.metric_name,
            "metric_value": (
                None
                if self.metric_value is None
                else _finite_or_none(self.metric_value)
            ),
            "cancelled": bool(self.cancelled),
            "error": self.error,
        }


@dataclass
class IterativePhysicsRefinementResult:
    """Loop-level result returned by ``IterativePhysicsRefinementTask.run``."""

    output_dir: Path
    iterations: list[IterationRecord] = field(default_factory=list)
    termination_reason: str = "unknown"
    final_iteration: int = 0
    final_dir: Path | None = None
    user_prompt: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_dir": str(self.output_dir),
            "iterations": [r.to_dict() for r in self.iterations],
            "termination_reason": self.termination_reason,
            "final_iteration": int(self.final_iteration),
            "final_dir": str(self.final_dir) if self.final_dir is not None else None,
            "user_prompt": self.user_prompt,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _scenario_to_yaml_text(scenario: Scenario) -> str:
    """Round-trip a :class:`Scenario` into the YAML form ``load_scenario`` accepts."""
    payload: dict[str, Any] = {
        "name": scenario.name,
        "metric": scenario.metric,
        "target": dict(scenario.target),
        "parameters": [
            {"name": p.name, "min": p.min_value, "max": p.max_value}
            for p in scenario.params
        ],
    }
    if scenario.extra:
        for k, v in scenario.extra.items():
            payload[k] = v
    return yaml.safe_dump(payload, sort_keys=False)


def _history_to_summary(history: list[TrialRecord]) -> list[dict[str, Any]]:
    """Compact history for the refine-task prompt body."""
    return [
        {
            "trial_index": int(t.trial_index),
            "score": float(t.score) if math.isfinite(float(t.score)) else None,
            "params": dict(t.params),
            "failed": bool(t.failed),
        }
        for t in history
    ]


def _extract_metric_value(history: list[TrialRecord], metric_name: str) -> float | None:
    """Pull the metric's raw physical value off the best trial's
    backend_metrics block.

    The drop_settle evaluator surfaces ``max_bounce_height`` (and other
    metrics by name) on the trial result alongside the optimizer
    ``score``. When the requested metric is not present we return None.
    """
    if not history:
        return None
    successful = [t for t in history if not t.failed]
    if not successful:
        return None
    best = min(successful, key=lambda t: t.score)
    bm = best.backend_metrics or {}
    if metric_name in bm:
        try:
            return float(bm[metric_name])
        except (TypeError, ValueError):
            return None
    if metric_name == "settle_distance":
        # settle_distance is always present on a drop_settle trial.
        try:
            return float(bm.get("settle_distance", best.score))
        except (TypeError, ValueError):
            return None
    return None


def _copy_iteration_to_final(iter_dir: Path, final_dir: Path) -> None:
    """Snapshot the winning iteration directory under ``final/``."""
    if final_dir.exists():
        shutil.rmtree(final_dir)
    shutil.copytree(iter_dir, final_dir)


def _discover_camera_paths(stage_path: Path) -> list[str] | None:
    """Open ``stage_path`` and return all ``UsdGeom.Camera`` prim paths.

    The drop_settle scene authors cameras under ``/Cameras/<dir>``. The
    recording.usda subLayers scene.usda so cameras are visible from
    either path. We open the stage in memory and walk it for camera
    prims so we don't depend on the per-trial scene_info dict. Returns
    ``None`` when nothing is found so the renderer falls back to its
    default discovery.
    """
    try:
        from pxr import Usd, UsdGeom
    except ImportError:  # pragma: no cover — defensive
        return None
    try:
        stage = Usd.Stage.Open(str(stage_path))
    except Exception:
        return None
    if stage is None:
        return None
    paths: list[str] = []
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Camera):
            paths.append(str(prim.GetPath()))
    return paths or None


class _LLMTimeoutError(RuntimeError):
    """An LLM call inside the refine loop exceeded its wall-clock deadline.

    Mirrors ``physics_agent.tuning.runner._LLMTimeoutError`` so the refine
    loop benefits from the same hard deadline that the tune runner uses.
    Kept as a private sentinel — the orchestrator catches it locally and
    converts it into a degraded ``llm_unavailable`` result; callers see
    only the public ``RuntimeError`` if it ever escapes.
    """


def _run_with_llm_timeout(
    fn: Callable[..., Any],
    *args: Any,
    timeout_seconds: float,
    op_label: str,
    **kwargs: Any,
) -> Any:
    """Execute a synchronous LLM call under a wall-clock deadline.

    Light reimplementation of ``physics_agent.tuning.runner._run_with_llm_timeout``
    (we duplicate rather than import to keep this module's import surface
    free of optimizer/runner dependencies). Same daemon-thread shape, same
    contract: ``timeout_seconds <= 0`` disables the deadline.

    A hung NIM/ChatNVIDIA call (no SDK-level deadline) cannot be
    interrupted in pure Python, so the orphan daemon thread continues
    until the provider's own timeout (or process exit) ends it. The hard
    deadline here only ensures the refine loop itself does not block
    indefinitely — the right complementary fix is provider-level
    deadlines, same as the tune runner's docstring notes.
    """
    if timeout_seconds <= 0:
        return fn(*args, **kwargs)

    result_box: dict[str, Any] = {}
    error_box: dict[str, BaseException] = {}
    done = threading.Event()

    def _worker() -> None:
        try:
            result_box["value"] = fn(*args, **kwargs)
        except BaseException as exc:  # noqa: BLE001 - capture every error type
            error_box["error"] = exc
        finally:
            done.set()

    thread = threading.Thread(
        target=_worker,
        daemon=True,
        name=f"refine-{op_label}-llm",
    )
    thread.start()

    deadline = time.monotonic() + timeout_seconds
    poll = 0.5
    while True:
        if done.wait(poll):
            break
        if time.monotonic() >= deadline:
            raise _LLMTimeoutError(
                f"{op_label} LLM call exceeded {timeout_seconds}s deadline"
            )

    if "error" in error_box:
        raise error_box["error"]
    return result_box.get("value")


# ---------------------------------------------------------------------------
# IterativePhysicsRefinementTask
# ---------------------------------------------------------------------------


class IterativePhysicsRefinementTask:
    """Drive the (tune → judge → refine) loop.

    Constructor pins loop parameters; ``run()`` executes the loop and
    returns an :class:`IterativePhysicsRefinementResult`. Compatible with
    callers that don't have a chat model (judge + refine degrade
    gracefully when ``chat_model`` is ``None``).
    """

    def __init__(
        self,
        *,
        user_prompt: str,
        initial_scenario: Path | dict[str, Any] | Scenario,
        physics_usd: Path,
        output_dir: Path,
        engine: str = "ovphysx",
        optimizer: str = "auto",
        max_trials: int = 30,
        seed: int = 42,
        max_iterations: int = 5,
        score_threshold: float = 0.7,
        judge_max_tokens: int | None = None,
        judge_temperature: float | None = None,
        chat_model: Any | None = None,
        vlm_model: Any | None = None,
        reference_images: list[Path] | None = None,
        reference_videos: list[Path] | None = None,
        reference_descriptions: list[str] | None = None,
        reference_video_descriptions: list[str] | None = None,
        run_tune_callable: Any | None = None,
        force_record_video: str | None = None,
        render_winning_trial: bool = True,
        visual_evidence_enabled: bool = True,
        llm_timeout_seconds: float = 180.0,
    ) -> None:
        if not user_prompt or not user_prompt.strip():
            raise ValueError("user_prompt must be a non-empty string")
        if max_iterations < 1:
            raise ValueError(f"max_iterations must be >= 1, got {max_iterations}")
        if judge_max_tokens is not None and judge_max_tokens < 1:
            raise ValueError(f"judge_max_tokens must be >= 1, got {judge_max_tokens}")
        if judge_temperature is not None:
            temperature = float(judge_temperature)
            if not math.isfinite(temperature) or temperature < 0.0:
                raise ValueError(
                    "judge_temperature must be finite and >= 0, "
                    f"got {judge_temperature}"
                )
        if force_record_video is not None and force_record_video not in {
            "off",
            "end_of_tune",
            "always",
        }:
            raise ValueError(
                "force_record_video must be one of {'off','end_of_tune',"
                f"'always'}}, got {force_record_video!r}"
            )
        self.name = "IterativePhysicsRefinement"
        self.description = (
            "Iteratively tune+judge+refine a physics scenario from a user prompt"
        )
        self.user_prompt = user_prompt.strip()
        self._initial_scenario_input = initial_scenario
        self.physics_usd = Path(physics_usd)
        self.output_dir = Path(output_dir)
        self.engine = engine
        try:
            refine_capabilities = get_backend(engine).tuning_capabilities()
        except (NewtonUnavailableError, OvPhysXUnavailableError):
            refine_capabilities = capabilities_for_backend(engine)
        except Exception:
            logger.exception("Failed to resolve tuning capabilities for %s", engine)
            raise
        self._refine_supported_param_keys = tuple(
            capability.param_name for capability in refine_capabilities
        )
        self.optimizer = optimizer
        self.max_trials = max_trials
        self.seed = seed
        self.max_iterations = max_iterations
        self.score_threshold = score_threshold
        self.judge_max_tokens = judge_max_tokens
        self.judge_temperature = judge_temperature
        self.chat_model = chat_model
        self.vlm_model = vlm_model
        self.reference_images = [Path(p) for p in reference_images or []]
        self.reference_videos = [Path(p) for p in reference_videos or []]
        self.reference_descriptions = (
            list(reference_descriptions) if reference_descriptions is not None else None
        )
        self.reference_video_descriptions = (
            list(reference_video_descriptions)
            if reference_video_descriptions is not None
            else None
        )
        self.force_record_video = force_record_video
        self.render_winning_trial = render_winning_trial
        self.visual_evidence_enabled = bool(visual_evidence_enabled)
        self.llm_timeout_seconds = float(llm_timeout_seconds)
        # Indirection so tests can plug in a fake ``run_tune`` without
        # spinning the OvPhysX daemon. Defaults to the real one.
        if run_tune_callable is None:
            from physics_agent.tuning.runner import run_tune as _run_tune

            self._run_tune = _run_tune
        else:
            self._run_tune = run_tune_callable

    # ------------------------------------------------------------------
    # Public entry
    # ------------------------------------------------------------------

    def run(
        self,
        context: dict[str, Any] | None = None,
        object_store: Any | None = None,
    ) -> IterativePhysicsRefinementResult:
        """Run the loop.

        ``context`` and ``object_store`` are accepted for parity with the
        material agent's Task signature; this loop does not need them but
        emits the same context-key signals via the listener.

        ``context`` is mutated in place (matching material's loop) so
        callers can read back ``judge_score`` / ``judge_reasoning`` /
        ``continue_iteration`` / ``iteration_count`` after the call.
        """
        ctx = context if context is not None else {}
        listener = get_listener(ctx, logger_name=__name__)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Resolve the initial scenario once BEFORE wiping stale iter
        # directories — a caller may legitimately point at an earlier
        # run's artifact (e.g.
        # ``physics-agent refine output/refine/final/scenario.yaml --output-dir output/refine``)
        # and the cleanup below must not delete the file the loader is
        # about to read.
        scenario = self._load_initial_scenario()

        if self.visual_evidence_enabled:
            try:
                reference_evidence = _run_with_llm_timeout(
                    self._prepare_reference_evidence,
                    timeout_seconds=self.llm_timeout_seconds,
                    op_label="visual evidence preparation",
                )
            except _LLMTimeoutError as exc:
                listener.warning(
                    "  Visual evidence preparation timed out after "
                    f"{self.llm_timeout_seconds}s; judge will fail closed: {exc}"
                )
                reference_evidence = JudgeVisualEvidence(
                    reference_error="VisualEvidencePreparationTimeout"
                )
        else:
            reference_evidence = None
            listener.info(
                "  Judge visual evidence disabled; VLM judge will run text-only"
            )
        self._ensure_judge_vlm(listener)

        # Wipe iter_<number>/ and final/ subdirectories from any
        # previous run into this output_dir BEFORE the loop starts. The
        # per-iter mkdir below only initialises the iter_N this run will
        # produce, so a previous 5-iter run that approves at iter 2 this
        # time around would otherwise leak iter_3..iter_5/. We
        # deliberately leave any user files at the output_dir top level
        # untouched, AND restrict the iter_ pattern to iter_<digits>
        # exactly so user-authored siblings like ``iter_notes/`` or
        # ``iter_backup/`` survive.
        _iter_dir_re = re.compile(r"^iter_\d+$")
        for stale in self.output_dir.iterdir():
            if not stale.is_dir():
                continue
            if stale.name == "final" or _iter_dir_re.match(stale.name):
                shutil.rmtree(stale)

        listener.info("=" * 80)
        listener.info(
            f"Iterative physics refinement: {self.max_iterations} iter cap, "
            f"score_threshold={self.score_threshold}"
        )
        listener.info(f"  User prompt: {self.user_prompt!r}")
        listener.info(f"  Physics USD: {self.physics_usd}")
        listener.info(f"  Output dir: {self.output_dir}")
        listener.info(f"  Initial metric: {scenario.metric}")
        listener.info("=" * 80)

        records: list[IterationRecord] = []
        termination_reason = "max_iterations"
        final_iter_dir: Path | None = None

        for iteration in range(1, self.max_iterations + 1):
            listener.info("")
            listener.info("-" * 80)
            listener.info(f"ITERATION {iteration}/{self.max_iterations}")
            listener.info("-" * 80)

            iter_dir = self.output_dir / f"iter_{iteration}"
            iter_dir.mkdir(parents=True, exist_ok=True)

            # 1) Persist the scenario YAML used by THIS iteration. When
            #    ``force_record_video`` is set (the CLI passes "off"), it
            #    overrides whatever ``record_video`` was authored in the
            #    initial YAML or refined by the LLM — wins over both
            #    sources. The orchestrator's post-tune winning-trial
            #    render is the canonical "one mp4 per iter" output, so
            #    suppressing per-trial rendering keeps trials fast.
            #    Pass force_record_video=None at construction time to
            #    honor whatever the YAML / refine flow asks for.
            scenario_yaml_path = iter_dir / "scenario.yaml"
            yaml_text = _scenario_to_yaml_text(scenario)
            if self.force_record_video is not None:
                merged_dict = yaml.safe_load(yaml_text)
                merged_dict.setdefault("target", {})
                merged_dict["target"]["record_video"] = self.force_record_video
                yaml_text = yaml.safe_dump(merged_dict, sort_keys=False)
                # Re-parse so the in-memory scenario aligns with what
                # we wrote on disk for downstream judge / refine calls.
                scenario = load_scenario(merged_dict)
            scenario_yaml_path.write_text(yaml_text, encoding="utf-8")
            listener.info(f"  Wrote scenario.yaml → {scenario_yaml_path}")

            # 2) Run a tune sweep on the current scenario. We call run_tune
            #    with enable_judge=False because we run the judge ourselves
            #    so the iteration counter is correct and we can persist a
            #    judge_result.json per iteration.
            try:
                tune_output = self._run_tune_for_iteration(
                    scenario_yaml_path=scenario_yaml_path,
                    iter_output_dir=iter_dir,
                    iteration=iteration,
                )
            except Exception as exc:
                listener.error(f"  Iteration {iteration} tune raised: {exc}")
                records.append(
                    IterationRecord(
                        iteration=iteration,
                        iteration_dir=iter_dir,
                        scenario_yaml_path=scenario_yaml_path,
                        tune_output_dir=iter_dir,
                        best_params={},
                        best_score=float("inf"),
                        n_trials=0,
                        judge_decision="skipped",
                        judge_score=0.0,
                        judge_reasoning="tune raised before judge",
                        judge_llm_unavailable=True,
                        refine_llm_unavailable=True,
                        refine_reasoning="",
                        metric_name=scenario.metric,
                        metric_value=None,
                        error=str(exc),
                    )
                )
                termination_reason = "error"
                break

            # Tune can also signal failure via ``TuneOutput(success=False)``
            # without raising — e.g. every backend trial failed, or the run
            # was cancelled. Without this guard the loop would still run the
            # judge over the (often empty / non-finite) history, optionally
            # approve, and the CLI would print "Refinement completed". Treat
            # success=False as a hard error / cancelled iteration so the
            # error termination plumbing kicks in and the CLI exits non-zero.
            if not tune_output.success:
                error_msg = (
                    str(getattr(tune_output, "error", None))
                    or "tune reported success=False"
                )
                cancelled_flag = bool(getattr(tune_output, "cancelled", False))
                listener.error(
                    f"  Iteration {iteration} tune did not succeed "
                    f"(cancelled={cancelled_flag}): {error_msg}"
                )
                records.append(
                    IterationRecord(
                        iteration=iteration,
                        iteration_dir=iter_dir,
                        scenario_yaml_path=scenario_yaml_path,
                        tune_output_dir=iter_dir,
                        best_params=dict(tune_output.best_params or {}),
                        best_score=float(
                            tune_output.best_score
                            if math.isfinite(float(tune_output.best_score))
                            else float("inf")
                        ),
                        n_trials=len(tune_output.history or []),
                        judge_decision="skipped",
                        judge_score=0.0,
                        judge_reasoning=(
                            "tune cancelled" if cancelled_flag else "tune failed"
                        ),
                        judge_llm_unavailable=True,
                        refine_llm_unavailable=True,
                        refine_reasoning="",
                        metric_name=scenario.metric,
                        metric_value=None,
                        cancelled=cancelled_flag,
                        error=error_msg,
                    )
                )
                termination_reason = "cancelled" if cancelled_flag else "error"
                break

            best_params = dict(tune_output.best_params)
            best_score = float(tune_output.best_score)
            history: list[TrialRecord] = list(tune_output.history)

            listener.info(
                f"  Tune complete: {len(history)} trials, best_score={best_score:.6g}"
            )

            # 2.5) Post-iter render of the winning trial's recording.usda.
            #      The per-trial drop_settle evaluator only renders when
            #      ``record_video in {end_of_tune, always}``. With the
            #      default ``record_video=off`` we get fast trials and a
            #      single render of the best trial here, which matches
            #      the e2e contract (one mp4 per iter).
            generated_frames: list[Path] = []
            generated_error: str | None = None
            needs_visual_judge_render = (
                self.visual_evidence_enabled
                and reference_evidence is not None
                and reference_evidence.reference_error is None
            )
            if self.render_winning_trial or needs_visual_judge_render:
                try:
                    generated_frames, generated_error = _run_with_llm_timeout(
                        self._render_best_trial_into_iter_dir,
                        iter_dir=iter_dir,
                        history=history,
                        scenario=scenario,
                        listener=listener,
                        timeout_seconds=self.llm_timeout_seconds,
                        op_label="winning-trial-render",
                    )
                except _LLMTimeoutError as exc:
                    generated_frames = []
                    generated_error = "VisualEvidenceRenderTimeout"
                    listener.warning(
                        "Winning trial render timed out after "
                        f"{self.llm_timeout_seconds}s; judge will fail closed: {exc}"
                    )

            ctx["judge_score"] = None  # clear stale value before judge
            ctx["iteration_count"] = iteration

            # 3) Judge. ``run_tune_judge`` is single-shot; pass the
            #    iteration so the persisted judge_result.json reports the
            #    right counter. The judge call is wrapped in a wall-clock
            #    timeout — NIM/ChatNVIDIA has no SDK-level deadline, so a
            #    hung provider would otherwise wedge ``physics-agent
            #    refine`` forever. On timeout we synthesize an
            #    ``llm_unavailable`` JudgeResult that the next block can
            #    handle the same way as any other LLM failure.
            if not self.visual_evidence_enabled:
                visual_evidence = None
            elif reference_evidence is not None:
                visual_evidence = reference_evidence.with_generated_images(
                    generated_frames,
                    generated_error=generated_error,
                )
            elif generated_frames:
                visual_evidence = JudgeVisualEvidence(
                    generated_image_paths=tuple(generated_frames),
                )
            else:
                visual_evidence = None
            if (
                visual_evidence is not None
                and visual_evidence.has_reference_media
                and visual_evidence.generated_image_paths
            ):
                comparison_path, comparison_error = write_comparison_contact_sheet(
                    visual_evidence,
                    iter_dir / ARTIFACT_VISUAL_COMPARISON,
                )
                visual_evidence = visual_evidence.with_comparison_image(
                    comparison_path,
                    comparison_error=comparison_error,
                )
            try:
                judge_result = _run_with_llm_timeout(
                    run_tune_judge,
                    scenario,
                    history,
                    best_params,
                    user_prompt=self.user_prompt,
                    chat_model=self.chat_model,
                    vlm_model=self.vlm_model,
                    visual_evidence=visual_evidence,
                    visual_evidence_enabled=self.visual_evidence_enabled,
                    judge_max_tokens=self.judge_max_tokens,
                    judge_temperature=self.judge_temperature,
                    score_threshold=self.score_threshold,
                    iteration=iteration,
                    timeout_seconds=self.llm_timeout_seconds,
                    op_label="judge",
                )
            except _LLMTimeoutError as exc:
                listener.warning(
                    f"  Judge VLM timed out after "
                    f"{self.llm_timeout_seconds}s; treating as "
                    f"unavailable for iteration {iteration}: {exc}"
                )
                from physics_agent.tasks.judge_tune import JudgeResult

                judge_result = JudgeResult(
                    decision="continue",
                    score=0.0,
                    programmatic_score=0.0,
                    llm_score=0.0,
                    reasoning=f"LLM timeout after {self.llm_timeout_seconds}s",
                    iterations=iteration,
                    llm_unavailable=True,
                    programmatic_critique="(skipped: judge call timed out)",
                    llm_critique=f"timeout after {self.llm_timeout_seconds}s",
                    extra={
                        "judge_modality": "vlm",
                        "visual_evidence_enabled": self.visual_evidence_enabled,
                        "reference_image_count": (
                            len(visual_evidence.reference_image_caption_pairs)
                            if visual_evidence is not None
                            else 0
                        ),
                        "generated_image_count": (
                            len(visual_evidence.generated_image_paths)
                            if visual_evidence is not None
                            else 0
                        ),
                        "visual_evidence": (
                            visual_evidence.to_metadata()
                            if visual_evidence is not None
                            else None
                        ),
                    },
                )
            self._write_json(iter_dir / "judge_result.json", judge_result.to_dict())
            listener.info(
                f"  Judge: decision={judge_result.decision} "
                f"score={judge_result.score:.3f} "
                f"(prog={judge_result.programmatic_score:.3f}, "
                f"llm={judge_result.llm_score:.3f}, "
                f"unavail={judge_result.llm_unavailable})"
            )
            listener.info(f"  Judge reasoning: {judge_result.reasoning}")
            metric_value = _extract_metric_value(history, scenario.metric)
            # Refine is a tune -> VLM judge -> scenario-refine loop. Do not let
            # any run, including text-only runs with an empty media list, approve
            # from programmatic-only scores when the judge VLM is unavailable.
            if judge_result.llm_unavailable:
                error_msg = (
                    f"Judge VLM unavailable at iteration {iteration} "
                    f"(critique: {judge_result.llm_critique}). "
                    f"Refusing to fall back to programmatic-only verdict."
                )
                listener.error(f"  {error_msg}")
                ctx["judge_score"] = judge_result.score
                ctx["judge_reasoning"] = judge_result.reasoning
                ctx["continue_iteration"] = False
                records.append(
                    IterationRecord(
                        iteration=iteration,
                        iteration_dir=iter_dir,
                        scenario_yaml_path=scenario_yaml_path,
                        tune_output_dir=iter_dir,
                        best_params=best_params,
                        best_score=best_score,
                        n_trials=len(history),
                        judge_decision=judge_result.decision,
                        judge_score=judge_result.score,
                        judge_reasoning=judge_result.reasoning,
                        judge_llm_unavailable=True,
                        refine_llm_unavailable=True,
                        refine_reasoning="",
                        metric_name=scenario.metric,
                        metric_value=metric_value,
                        cancelled=bool(getattr(tune_output, "cancelled", False)),
                        error=error_msg,
                    )
                )
                termination_reason = "error"
                break

            # Listener context-key contract — same shape material's loop emits.
            ctx["judge_score"] = judge_result.score
            ctx["judge_reasoning"] = judge_result.reasoning
            should_continue = judge_result.decision == "continue"
            ctx["continue_iteration"] = should_continue

            # Capture this iteration's record before we (maybe) refine.
            record = IterationRecord(
                iteration=iteration,
                iteration_dir=iter_dir,
                scenario_yaml_path=scenario_yaml_path,
                tune_output_dir=iter_dir,
                best_params=best_params,
                best_score=best_score,
                n_trials=len(history),
                judge_decision=judge_result.decision,
                judge_score=judge_result.score,
                judge_reasoning=judge_result.reasoning,
                judge_llm_unavailable=judge_result.llm_unavailable,
                refine_llm_unavailable=False,  # filled below
                refine_reasoning="",  # filled below
                metric_name=scenario.metric,
                metric_value=metric_value,
                cancelled=bool(getattr(tune_output, "cancelled", False)),
            )

            # 4) Approve → terminate; otherwise refine and loop.
            if judge_result.decision == "approve":
                listener.info(
                    f"  Judge APPROVED at iteration {iteration} — terminating."
                )
                records.append(record)
                final_iter_dir = iter_dir
                termination_reason = "approved"
                break

            # 5) Refine for the next iteration.
            if iteration >= self.max_iterations:
                listener.info(
                    "  Reached max_iterations; persisting iteration record "
                    "and terminating."
                )
                # Override the listener-level continue flag: the judge
                # said "continue" but the iteration cap is final, so
                # callers reading the documented context contract should
                # see ``continue_iteration=False`` to match the actual
                # next-iteration behaviour ("there isn't one").
                ctx["continue_iteration"] = False
                records.append(record)
                final_iter_dir = iter_dir
                termination_reason = "max_iterations"
                break

            # Same wall-clock guard as the judge call (NIM has no
            # SDK-level deadline). On timeout we synthesize an
            # ``llm_unavailable`` RefineResult so the loop reuses the
            # current scenario for the next iteration instead of
            # blocking forever.
            try:
                refine: RefineResult = _run_with_llm_timeout(
                    run_scenario_refine,
                    current_scenario=scenario,
                    judge_result=judge_result,
                    user_goal_text=self.user_prompt,
                    history_summary=_history_to_summary(history),
                    iteration=iteration,
                    chat_model=self.chat_model,
                    backend_name=self.engine,
                    supported_param_keys=self._refine_supported_param_keys,
                    timeout_seconds=self.llm_timeout_seconds,
                    op_label="scenario_refine",
                )
            except _LLMTimeoutError as exc:
                listener.warning(
                    f"  Refine LLM timed out after "
                    f"{self.llm_timeout_seconds}s at iteration "
                    f"{iteration}; reusing current scenario: {exc}"
                )
                _current_yaml = _scenario_to_yaml_text(scenario)
                refine = RefineResult(
                    refined_yaml=_current_yaml,
                    scenario=scenario,
                    llm_unavailable=True,
                    reasoning=f"LLM timeout after {self.llm_timeout_seconds}s",
                    notes={"history_size": len(history)},
                )
            record.refine_llm_unavailable = refine.llm_unavailable
            record.refine_reasoning = refine.reasoning
            records.append(record)
            self._write_json(
                iter_dir / "refine_result.json",
                {
                    "llm_unavailable": refine.llm_unavailable,
                    "reasoning": refine.reasoning,
                    "next_metric": refine.scenario.metric,
                },
            )
            listener.info(
                f"  Refine: llm_unavailable={refine.llm_unavailable} "
                f"reasoning={refine.reasoning!r}"
            )

            # Promote the refined scenario for the next iteration.
            scenario = refine.scenario

        # Pick a final iteration dir if we exited without setting one
        # (e.g. some error path). Skip the final/ snapshot entirely on
        # error / cancelled termination — final/ is meant to point at a
        # usable iteration (approve or last continue), and an error iter
        # has only scenario.yaml + the exception text. Surface the
        # failure via termination_reason / records[].error instead.
        if (
            final_iter_dir is None
            and records
            and termination_reason not in ("error", "cancelled")
        ):
            final_iter_dir = records[-1].iteration_dir

        # Promote the winning iteration to ``final/`` for easy inspection.
        final_dir: Path | None = None
        if final_iter_dir is not None and final_iter_dir.exists():
            final_dir = self.output_dir / "final"
            try:
                _copy_iteration_to_final(final_iter_dir, final_dir)
            except Exception as exc:
                listener.warning(f"Failed to copy final iteration: {exc}")
                final_dir = final_iter_dir

        result = IterativePhysicsRefinementResult(
            output_dir=self.output_dir,
            iterations=records,
            termination_reason=termination_reason,
            final_iteration=records[-1].iteration if records else 0,
            final_dir=final_dir,
            user_prompt=self.user_prompt,
        )
        self._write_json(
            self.output_dir / "refine_summary.json",
            result.to_dict(),
        )

        listener.info("")
        listener.info("=" * 80)
        listener.info(
            f"Refinement summary: termination={termination_reason} "
            f"iters={len(records)}/{self.max_iterations} "
            f"final={final_dir}"
        )
        listener.info("=" * 80)
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_initial_scenario(self) -> Scenario:
        if isinstance(self._initial_scenario_input, Scenario):
            return self._initial_scenario_input
        return load_scenario(self._initial_scenario_input)

    def _prepare_reference_evidence(self) -> JudgeVisualEvidence | None:
        """Prepare user reference media once for all judge iterations."""
        if not has_reference_media(
            reference_images=self.reference_images,
            reference_videos=self.reference_videos,
        ):
            return None
        try:
            return prepare_reference_media(
                reference_images=self.reference_images,
                reference_videos=self.reference_videos,
                reference_descriptions=self.reference_descriptions,
                reference_video_descriptions=self.reference_video_descriptions,
                output_dir=self.output_dir,
            )
        except Exception as exc:  # noqa: BLE001 - surfaced by judge result
            return JudgeVisualEvidence(reference_error=type(exc).__name__)

    def _ensure_judge_vlm(self, listener: Any) -> None:
        """Resolve the default judge VLM unless one was injected."""
        if self.vlm_model is not None:
            return
        try:
            self.vlm_model = _run_with_llm_timeout(
                resolve_default_judge_vlm,
                timeout_seconds=self.llm_timeout_seconds,
                op_label="judge VLM setup",
            )
        except Exception as exc:  # noqa: BLE001 - judge will fail closed
            listener.warning(
                "  Failed to instantiate judge VLM; judge will "
                f"be unavailable: {type(exc).__name__}: {exc}"
            )

    def _run_tune_for_iteration(
        self,
        *,
        scenario_yaml_path: Path,
        iter_output_dir: Path,
        iteration: int,
    ) -> TuneOutput:
        """Invoke the underlying tune runner once. Returns its TuneOutput.

        Offsets the base seed by ``(iteration - 1) * max_trials`` so the
        runner's ``seed + trial_index`` arithmetic produces disjoint
        per-trial seeds across iterations. Without this offset
        ``drop_settle.evaluate`` keeps writing to the same
        ``.tune_scenes/trial_seed_<seed>/`` directory tree across
        iterations and overwrites the per-trial ``recording.usda`` /
        ``trajectory.jsonl`` files that the previous iteration's
        ``history.jsonl`` references — turning the saved iteration
        artifacts into pointers to the latest iteration's data.
        """
        seed_offset = (iteration - 1) * self.max_trials
        params = TuneInput(
            scenario=scenario_yaml_path,
            user_prompt=None,  # the loop owns the prompt; runner mustn't re-author
            physics_usd=self.physics_usd,
            output_dir=iter_output_dir,
            engine=self.engine,
            optimizer=self.optimizer,
            max_trials=self.max_trials,
            seed=self.seed + seed_offset,
            enable_judge=False,  # we run the judge ourselves per iteration
            judge_max_iterations=1,
        )
        return self._run_tune(params)

    def _render_best_trial_into_iter_dir(
        self,
        *,
        iter_dir: Path,
        history: list[TrialRecord],
        scenario: Scenario,
        listener: Any,
    ) -> tuple[list[Path], str | None]:
        """Render the winning trial's ``recording.usda`` into ``iter_dir/render/``.

        Picks the best (lowest-score) successful trial from ``history``,
        reads its ``recording_usda`` and ``scene_usd`` paths from
        ``backend_metrics``, and runs the world_understanding render
        helper to produce ``iter_dir/render/render.mp4`` plus a
        per-frame PNG sequence. Render failures are logged and dropped
        — they do not fail the loop.
        """
        successful = [t for t in history if not t.failed]
        if not successful:
            listener.warning(
                "  Skipping iter render: every trial failed in this iteration."
            )
            return [], "every trial failed; no winning trial to render"
        best = min(successful, key=lambda t: t.score)
        bm = best.backend_metrics or {}
        recording = bm.get("recording_usda")
        if not recording:
            listener.warning(
                "  Skipping iter render: best trial did not persist a recording.usda."
            )
            return [], "winning trial did not persist recording_usda"
        try:
            from world_understanding.functions.graphics import (
                render_time_sampled_usd,
            )
        except ImportError:
            # The helper ships in PR #66 (issue #50). Until that lands
            # the iter render is a no-op; not fatal — the tune output,
            # judge verdict, and refine_summary.json are still complete
            # without the mp4. Same fallback drop_settle.evaluate() takes.
            listener.warning(
                "  Skipping iter render: render_time_sampled_usd unavailable "
                "(world_understanding.functions.graphics; ships in PR #66 / issue #50)."
            )
            return [], "render_time_sampled_usd unavailable"

        render_dir = iter_dir / "render"
        render_dir.mkdir(parents=True, exist_ok=True)
        target = scenario.target or {}
        cameras = _discover_camera_paths(Path(recording))
        try:
            frames = render_time_sampled_usd(
                Path(recording),
                render_dir,
                renderer=str(
                    target.get("video_renderer")
                    or target.get("vlm_renderer")
                    or "ovrtx"
                ),
                cameras=cameras,
                fps=int(target.get("sample_fps", 30)),
                max_duration_seconds=float(target.get("duration_s", 2.0)),
                image_width=int(target.get("video_image_width", 512)),
                image_height=int(target.get("video_image_height", 512)),
                num_sensor_updates=int(target.get("video_sensor_updates", 32)),
                render_mode=str(target.get("video_render_mode", "rt2")),
            )
            listener.info(
                f"  Rendered winning trial → {render_dir} ({len(frames)} frames)"
            )
            if not frames:
                return [], "renderer produced no frames"
            return list(frames), None
        except Exception as exc:
            listener.warning(
                f"  Iter render failed (non-fatal): {type(exc).__name__}: {exc}"
            )
            return [], type(exc).__name__

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        # ``allow_nan=False`` makes us assert strict JSON: we surfaced
        # the only known non-finite producer (best_score=inf on the
        # tune-error path) via IterationRecord.to_dict's _finite_or_none
        # coercion; if some new consumer slips a NaN/Inf through, fail
        # loud here rather than silently emitting non-strict JSON.
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str, allow_nan=False),
            encoding="utf-8",
        )
