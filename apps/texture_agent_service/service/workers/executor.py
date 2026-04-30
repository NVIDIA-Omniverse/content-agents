# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pipeline execution for Texture Agent Service.

Wraps the synchronous texture-agent pipeline by running each task
individually via asyncio.to_thread(), emitting progress events between steps.
"""

import asyncio
import logging
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..config import config as service_config
from ..runtime.bus import get_event_bus
from ..runtime.events import ProgressEvent, StepState
from ..sanitization import sanitize_message, sanitize_step_stats

logger = logging.getLogger(__name__)

# Map task class names to step names
_TASK_CLASS_TO_STEP = {
    "PrepareUVsTask": "prepare_uvs",
    "DiscoverMaterialsTask": "discover_materials",
    "GeneratePromptsTask": "generate_prompts",
    "RenderMaterialPreviewsTask": "render_previews",
    "GenerateTexturesTask": "generate_textures",
    "BlendTexturesTask": "blend_textures",
    "ApplyTexturesTask": "apply_textures",
    "RenderOutputTask": "render",
}


def _clear_task_cancellation_requests() -> None:
    """Clear pending cancellation count while draining a shielded thread."""
    task = asyncio.current_task()
    if task is None:
        return

    uncancel = getattr(task, "uncancel", None)
    if uncancel is None:
        return

    while task.cancelling():
        uncancel()


def _mark_stalled_until_future_done(
    session_manager: Any,
    session_id: str,
    step_name: str,
    step_future: asyncio.Future,
    reason: str,
) -> None:
    """Block deletion while a cancelled worker thread continues in background."""
    mark_worker_stalled = getattr(session_manager, "mark_worker_stalled", None)
    if mark_worker_stalled is not None:
        mark_worker_stalled(session_id, reason)

    def _clear_marker(fut: asyncio.Future) -> None:
        try:
            fut.result()
        except BaseException:
            logger.debug(
                "Cancelled worker thread finished after stall marker for %s/%s",
                session_id[:8],
                step_name,
                exc_info=True,
            )

        clear_worker_stalled = getattr(session_manager, "clear_worker_stalled", None)
        if clear_worker_stalled is not None:
            clear_worker_stalled(session_id)

    step_future.add_done_callback(_clear_marker)


async def _drain_cancelled_step(
    *,
    session_id: str,
    step_name: str,
    step_future: asyncio.Future,
    session_manager: Any,
) -> None:
    """Wait for a cancelled threaded step with a hard deadline.

    The synchronous task cannot be interrupted by cancelling the asyncio
    wrapper. We keep the worker lock while the thread drains, but only up to a
    configured deadline so registry capacity cannot be pinned forever. If the
    deadline is exceeded, a stalled-worker marker keeps DELETE/TTL from
    removing artifacts until the thread future eventually finishes.
    """
    timeout_seconds = max(0.0, service_config.cancel_drain_timeout_seconds)
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_seconds

    while True:
        _clear_task_cancellation_requests()
        if step_future.done():
            try:
                step_future.result()
            except Exception:
                logger.exception(
                    "Step %s raised while draining cancelled session %s",
                    step_name,
                    session_id[:8],
                )
                raise
            return

        remaining = deadline - loop.time()
        if remaining <= 0:
            reason = (
                f"Cancellation timed out while waiting for step {step_name} "
                f"to stop after {timeout_seconds:.1f}s. The worker thread may "
                "still be writing artifacts."
            )
            _mark_stalled_until_future_done(
                session_manager,
                session_id,
                step_name,
                step_future,
                reason,
            )
            raise RuntimeError(reason)

        try:
            await asyncio.wait_for(asyncio.shield(step_future), timeout=remaining)
            return
        except TimeoutError:
            reason = (
                f"Cancellation timed out while waiting for step {step_name} "
                f"to stop after {timeout_seconds:.1f}s. The worker thread may "
                "still be writing artifacts."
            )
            _mark_stalled_until_future_done(
                session_manager,
                session_id,
                step_name,
                step_future,
                reason,
            )
            raise RuntimeError(reason)
        except asyncio.CancelledError:
            if step_future.done():
                continue
            logger.debug(
                "Additional cancellation while draining %s for %s",
                step_name,
                session_id[:8],
            )
            continue


def _task_to_step_name(task: Any) -> str:
    """Get the step name for a task instance."""
    class_name = type(task).__name__
    return _TASK_CLASS_TO_STEP.get(class_name, class_name)


def _prepare_config_and_context(
    config_dict: dict[str, Any],
    session_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a config dict with working_dir set, apply defaults, and convert to context.

    Returns:
        Tuple of (resolved_config, pipeline_context).
    """
    from texture_agent.config.schema import DEFAULTS, STEP_ORDER, STEP_OUTPUT_DIRS
    from texture_agent.config.unified_config import config_to_context

    working_dir = session_dir / "cache"

    # Set project working_dir
    config_dict.setdefault("project", {})
    config_dict["project"]["working_dir"] = str(working_dir)

    # Ensure input section exists
    config_dict.setdefault("input", {})

    # Apply defaults for texture config
    texture = config_dict.setdefault("texture", {})
    for key, val in DEFAULTS["texture"].items():
        texture.setdefault(key, val)

    # Apply defaults for variations
    variations = config_dict.setdefault("variations", {})
    for key, val in DEFAULTS["variations"].items():
        variations.setdefault(key, val)

    # Apply defaults for steps
    steps = config_dict.setdefault("steps", {})
    for step_name in STEP_ORDER:
        step_cfg = steps.setdefault(step_name, {})
        defaults = DEFAULTS["steps"].get(step_name, {})
        for key, val in defaults.items():
            step_cfg.setdefault(key, val)

    # Create working directory structure
    working_dir.mkdir(parents=True, exist_ok=True)
    for _step_name, dir_name in STEP_OUTPUT_DIRS.items():
        (working_dir / dir_name).mkdir(parents=True, exist_ok=True)

    context = config_to_context(config_dict)
    return config_dict, context


def _package_usdz(context: dict[str, Any], session_dir: Path) -> str | None:
    """Package the output USD + textures into a self-contained USDZ.

    Rewrites absolute texture paths to relative, then bundles everything
    into a single .usdz archive for easy download.

    Returns:
        Path to the USDZ file, or None if packaging failed.
    """
    import zipfile

    from pxr import Sdf, Usd, UsdShade, UsdUtils

    output_paths = context.get("output_usd_paths", [])
    if not output_paths:
        return None

    output_usd = Path(output_paths[0])
    if not output_usd.exists():
        logger.warning("Output USD not found: %s", output_usd)
        return None

    # Rewrite absolute texture paths to be relative to the USD file.
    # Textures live in cache/textures/ while USD is in cache/output/,
    # so the relative path from the USD is ../textures/<filename>.
    textures_dir = output_usd.parent.parent / "textures"

    stage = Usd.Stage.Open(str(output_usd))
    if not stage:
        return None

    rewritten = 0
    for prim in stage.Traverse():
        # String/token rewrites are scoped to UsdShade.Shader inputs whose
        # name ends in ``_texture`` (Codex round-11 finding) so we never
        # mutate unrelated authored metadata that happens to be a string
        # ending in ``.png``. Asset-typed rewrites stay broad — the existing
        # OpenPBR / tiledimage write path produces them across the stage.
        is_shader = prim.IsA(UsdShade.Shader)
        for attr in prim.GetAttributes():
            val = attr.Get()
            # Asset-typed PNG path → rewrite to bundle-relative.
            if isinstance(val, Sdf.AssetPath) and val.path:
                old_path = val.path
                filename = Path(old_path).name
                if filename.endswith(".png"):
                    new_path = f"../textures/{filename}"
                    attr.Set(Sdf.AssetPath(new_path))
                    rewritten += 1
                continue
            # String/token-typed PNG path (MDL shaders can author texture
            # inputs as `string` / `token`) → rewrite the same way so a
            # downloaded USDZ resolves the file via the OpenPBR side's
            # asset-typed dependency on the same generated PNG. Only
            # `inputs:*_texture` attributes on Shader prims qualify, and
            # the *original* path must resolve to a file inside this
            # session's ``cache/textures`` (apply_textures is the only
            # writer there, and it only writes verified bundle-safe
            # files). A bare basename match is not enough — a shader
            # input pointing somewhere else on disk could collide with a
            # generated PNG by basename and the rewrite would silently
            # substitute the wrong texture (Codex round-15 finding).
            if not (isinstance(val, str) and val and is_shader):
                continue
            attr_name = attr.GetName()
            if not (attr_name.startswith("inputs:") and attr_name.endswith("_texture")):
                continue
            filename = Path(val).name
            if not filename.endswith(".png"):
                continue
            try:
                src_resolved = Path(val).resolve()
                tex_resolved = textures_dir.resolve()
            except (OSError, ValueError):
                continue
            if not src_resolved.is_file():
                continue
            try:
                src_resolved.relative_to(tex_resolved)
            except ValueError:
                continue
            try:
                attr.Set(f"../textures/{filename}")
                rewritten += 1
            except Exception as err:
                logger.warning(
                    "Failed to rewrite string texture path on %s: %s",
                    attr.GetPath(),
                    err,
                )

    if rewritten > 0:
        stage.GetRootLayer().Export(str(output_usd))
        logger.info("Rewrote %d texture paths to relative", rewritten)

    # Package into USDZ
    usdz_path = output_usd.parent / "textured_output.usdz"

    # Remove stale file so a failed CreateNewUsdzPackage doesn't leave
    # a partial (raw USDC) file that the download endpoint would serve.
    usdz_path.unlink(missing_ok=True)

    success = UsdUtils.CreateNewUsdzPackage(str(output_usd), str(usdz_path))

    if success and usdz_path.exists():
        # Validate the output is actually a ZIP archive (USDZ spec).
        # CreateNewUsdzPackage can leave raw USDC bytes on failure.
        if not zipfile.is_zipfile(usdz_path):
            logger.warning(
                "CreateNewUsdzPackage wrote non-ZIP data to %s, removing",
                usdz_path,
            )
            usdz_path.unlink(missing_ok=True)
            return None

        size_mb = usdz_path.stat().st_size / (1024 * 1024)
        logger.info("Packaged USDZ: %s (%.1f MB)", usdz_path, size_mb)
        return str(usdz_path)

    # Clean up any partial file left behind on failure
    usdz_path.unlink(missing_ok=True)
    logger.warning("Failed to create USDZ package")
    return None


def _apply_textures_stats_summary(context: dict[str, Any]) -> dict[str, Any]:
    """Distil ``context['apply_textures_stats']`` into a flat stats dict.

    Both the per-step (``_extract_step_stats``) and final
    (``_extract_final_stats``) summaries need the same shape, so callers can see
    MDL override / clear / localize counts and a human-readable ``warnings``
    list whether they look at /status mid-run or at /results after completion.
    """
    out: dict[str, Any] = {}
    apply_stats = context.get("apply_textures_stats") or {}
    if "mdl_inputs_overridden" in apply_stats:
        out["mdl_inputs_overridden"] = apply_stats["mdl_inputs_overridden"]

    cleared = apply_stats.get("mdl_inputs_cleared") or []
    localized = apply_stats.get("mdl_inputs_localized") or []
    if cleared:
        out["mdl_inputs_cleared"] = list(cleared)
    if localized:
        out["mdl_inputs_localized"] = list(localized)

    if cleared:
        out["warnings"] = [
            "Cleared MDL texture inputs that could not be bundled (unbundleable "
            "URI refs or unresolvable local paths). Affected materials/inputs: "
            + ", ".join(cleared)
        ]
    return out


# Cap persisted per-unit error payloads. In per-prim mode with a backend-
# wide outage, the unbounded list could be one record per prim (thousands)
# in session.json, event_log.jsonl, SSE payloads, and /results. Counts +
# bounded sample preserve the diagnostic value while keeping persisted
# artifacts small during the very incidents we want diagnostics for.
_MAX_ERRORS_IN_PAYLOAD = 25
_MAX_ERROR_MESSAGE_CHARS = 500


def _truncate_errors(errors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Cap the error list size and truncate per-record messages.

    The full count is exposed via the sibling ``*_failed_count`` /
    ``textures_failed`` keys, so dropping the tail here doesn't lose the
    "how bad is it" signal -- only the per-material detail. The tail is
    still in container logs (``logger.exception``) for the small fraction
    of incidents where deeper diagnostics are needed.
    """
    capped = errors[:_MAX_ERRORS_IN_PAYLOAD]
    out: list[dict[str, Any]] = []
    for record in capped:
        message = record.get("message", "")
        if isinstance(message, str) and len(message) > _MAX_ERROR_MESSAGE_CHARS:
            message = message[:_MAX_ERROR_MESSAGE_CHARS] + "...(truncated)"
        out.append({**record, "message": message})
    return out


def _extract_step_stats(step_name: str, context: dict[str, Any]) -> dict:
    """Extract statistics from context after a step completes.

    For ``generate_textures`` and ``blend_textures``, propagate the
    structured per-unit error list and failure count surfaced by those
    tasks. Without this, partial-failure runs report ``state=completed``
    in SSE / ``/status`` with no diagnostic for which materials failed
    or why -- the silent-completed pattern from NVBugs 6126254.
    """
    stats: dict[str, Any] = {}

    if step_name == "discover_materials":
        materials = context.get("discovered_materials", [])
        stats["materials_found"] = len(materials)

    elif step_name == "generate_textures":
        generated = context.get("generated_textures", {})
        errors = context.get("generate_textures_errors", [])
        stats["textures_generated"] = len(generated)
        stats["textures_failed"] = context.get(
            "generate_textures_failed_count", len(errors)
        )
        if errors:
            stats["errors"] = _truncate_errors(errors)

    elif step_name == "blend_textures":
        blended = context.get("blended_textures", {})
        errors = context.get("blend_textures_errors", [])
        stats["textures_blended"] = len(blended)
        stats["textures_failed"] = context.get(
            "blend_textures_failed_count", len(errors)
        )
        if errors:
            stats["errors"] = _truncate_errors(errors)

    elif step_name == "apply_textures":
        output_paths = context.get("output_usd_paths", [])
        stats["output_usd_count"] = len(output_paths)
        stats.update(_apply_textures_stats_summary(context))

    elif step_name == "render":
        rendered = context.get("rendered_image_paths", [])
        stats["renders_count"] = len(rendered)

    return stats


def _extract_final_stats(context: dict[str, Any], session_dir: Path) -> dict[str, Any]:
    """Extract final pipeline statistics from context and files.

    Includes structured per-unit failure records when generate/blend
    completed below the threshold gate -- without this, a partial-failure
    run that completes (default threshold=1.0 + 1 success + N failures)
    looks identical to a clean run on ``GET /result/{session_id}`` after
    the SSE snapshot has been GC'd, leaving non-SSE consumers without the
    diagnostics this MR adds.
    """
    stats: dict[str, Any] = {
        "materials_found": len(context.get("discovered_materials", [])),
        "textures_generated": len(context.get("generated_textures", {})),
        "output_usd_count": len(context.get("output_usd_paths", [])),
        "renders_count": len(context.get("rendered_image_paths", [])),
    }

    # Fallback: count files if context stats are empty
    cache_dir = session_dir / "cache"

    if stats["textures_generated"] == 0:
        textures_dir = cache_dir / "textures"
        if textures_dir.exists():
            stats["textures_generated"] = len(list(textures_dir.glob("*.png")))

    if stats["output_usd_count"] == 0:
        output_dir = cache_dir / "output"
        if output_dir.exists():
            usd_files = list(output_dir.glob("*.usd")) + list(output_dir.glob("*.usda"))
            stats["output_usd_count"] = len(usd_files)

    if stats["renders_count"] == 0:
        renders_dir = cache_dir / "renders"
        if renders_dir.exists():
            stats["renders_count"] = len(list(renders_dir.glob("*.png")))

    # Persist apply_textures MDL override/clear/localize counts and the
    # `warnings` list so /results consumers see the same signal as /status.
    stats.update(_apply_textures_stats_summary(context))

    # Generate/blend partial-failure surfacing: per-step counts plus a
    # disjoint sum so an auth-issue gen failure isn't hidden when blend
    # also drops a unit.
    gen_failed = context.get("generate_textures_failed_count", 0)
    blend_failed = context.get("blend_textures_failed_count", 0)
    if gen_failed:
        stats["textures_generated_failed"] = gen_failed
        gen_errors = context.get("generate_textures_errors")
        if gen_errors:
            stats.setdefault("errors", {})["generate_textures"] = _truncate_errors(
                gen_errors
            )
    if blend_failed:
        stats["textures_blended_failed"] = blend_failed
        blend_errors = context.get("blend_textures_errors")
        if blend_errors:
            stats.setdefault("errors", {})["blend_textures"] = _truncate_errors(
                blend_errors
            )
    # Total is the sum: gen failures and blend failures cover disjoint
    # units (a unit either failed gen, OR was generated and failed
    # blend, OR succeeded both). Without this an auth-issue gen failure
    # followed by a downstream blend failure would hide the gen count
    # behind the blend count -- losing the "the backend is broken"
    # signal that ``textures_failed`` exists to surface.
    if gen_failed or blend_failed:
        stats["textures_failed"] = gen_failed + blend_failed

    return stats


def _get_step_validation_error(
    step_name: str,
    step_stats: dict[str, Any],
    planned_steps: list[str],
    context: dict[str, Any] | None = None,
) -> str | None:
    """Return a terminal validation error for a completed step, if any.

    Some task implementations can legitimately "complete" while doing no useful
    work. The service should convert those cases into failed sessions instead of
    reporting a false-positive success.

    ``context`` is the live pipeline context; when provided, the
    ``apply_textures`` empty-output message is enriched with the upstream
    cause (no textures generated / no textures blended) so the customer-
    visible failure points at the real root cause rather than the
    last-step symptom.
    """
    try:
        step_index = planned_steps.index(step_name)
    except ValueError:
        downstream_steps: list[str] = []
    else:
        downstream_steps = planned_steps[step_index + 1 :]

    if step_name == "discover_materials":
        materials_found = step_stats.get("materials_found", 0)
        if materials_found == 0 and downstream_steps:
            return (
                "No discoverable materials were found in the uploaded USD. "
                "Texture generation requires a USD with bound materials."
            )

    if step_name == "apply_textures":
        output_usd_count = step_stats.get("output_usd_count", 0)
        if output_usd_count == 0:
            base = (
                "Texture application produced no output USD files. "
                "The pipeline cannot be reported as completed."
            )
            if context is None:
                return base
            generated = context.get("generated_textures", {})
            blended = context.get("blended_textures", {})
            gen_errors = context.get("generate_textures_errors", [])
            blend_errors = context.get("blend_textures_errors", [])
            if not generated:
                cause = (
                    f"upstream generate_textures produced 0 textures "
                    f"({len(gen_errors)} per-material failure(s))"
                    if gen_errors
                    else "upstream generate_textures produced 0 textures"
                )
                return f"{base} Cause: {cause}."
            if not blended:
                cause = (
                    f"upstream blend_textures produced 0 textures "
                    f"({len(blend_errors)} per-material failure(s))"
                    if blend_errors
                    else "upstream blend_textures produced 0 textures"
                )
                return f"{base} Cause: {cause}."
            return base

    return None


async def execute_pipeline_async(
    session_id: str,
    config_dict: dict[str, Any],
    session_manager: Any,
    only_steps: list[str] | None = None,
    skip_steps: list[str] | None = None,
    acquire_worker_lock: bool = True,
) -> None:
    """Execute texture pipeline by running each task in a thread.

    Emits ProgressEvent for each step start/completion, enabling
    real-time SSE streaming to clients.

    Args:
        session_id: Session identifier
        config_dict: Pipeline configuration dict
        session_manager: SessionManager instance
        only_steps: If set, run only these steps
        skip_steps: Steps to skip
        acquire_worker_lock: If False, caller already reserved the cross-process
            worker lock and will release it after registry cleanup.
    """
    from texture_agent.workflows.factory import create_texture_pipeline_workflow

    logger.info(f"Pipeline execution started for {session_id[:8]}...")

    event_bus = get_event_bus()
    session_dir = session_manager.get_session_dir(session_id)
    lock_context = getattr(session_manager, "worker_lock", None)
    worker_lock = (
        lock_context(session_id)
        if acquire_worker_lock and lock_context is not None
        else nullcontext()
    )

    with worker_lock:
        try:
            await _execute_pipeline_inner(
                session_id,
                config_dict,
                session_manager,
                event_bus,
                session_dir,
                only_steps,
                skip_steps,
                create_texture_pipeline_workflow,
            )
        except asyncio.CancelledError:
            # task.cancel() (e.g. from POST /cancel) raises CancelledError at the
            # next await point. If the worker has not yet reached the between-step
            # is_cancelled() checkpoint in _execute_pipeline_inner, the cooperative
            # cleanup that normally persists "cancelled" is skipped -- handle that
            # final transition here so /status flips from "cancelling" to
            # "cancelled" instead of stalling.
            #
            # Persist the disk state synchronously BEFORE awaiting the event emit:
            # JobRegistry.cancel wraps task.cancel() in wait_for(timeout=5s) and
            # may fire a second task.cancel() if cleanup is slow. A re-raised
            # CancelledError on the await would skip the disk update otherwise.
            #
            # Keep the session worker lock held through this terminal update so
            # DELETE cannot remove artifacts while cancellation cleanup is still
            # writing metadata or queued events.
            logger.info("Pipeline cancelled via task.cancel for %s", session_id[:8])
            try:
                session_manager.update_session(session_id, {"status": "cancelled"})
            except Exception:
                logger.exception(
                    "Failed to persist cancelled status for %s", session_id[:8]
                )
            try:
                await event_bus.emit(
                    ProgressEvent(
                        session_id=session_id,
                        step="pipeline",
                        state=StepState.CANCELLED,
                        message="Pipeline cancelled by user",
                    )
                )
            except Exception:
                logger.exception(
                    "Failed to emit cancelled event for %s", session_id[:8]
                )
            raise
        except Exception as e:
            if getattr(e, "_wu_failure_handled", False):
                raise
            # Any uncaught error past the per-step guard (e.g. post-loop
            # packaging, final stats) must still flip the session to "failed"
            # so /status doesn't stay at "running" forever. The worker lock
            # remains held through this terminal update.
            logger.exception("Unhandled pipeline error for %s: %s", session_id[:8], e)
            try:
                session_manager.update_session(
                    session_id,
                    {
                        "status": "failed",
                        "error": sanitize_message(
                            str(e), service_config.session_storage_path
                        ),
                    },
                )
            except Exception:
                logger.exception(
                    "Failed to persist failed status for %s", session_id[:8]
                )
            sanitized_error = sanitize_message(
                str(e), service_config.session_storage_path
            )
            try:
                await event_bus.emit(
                    ProgressEvent(
                        session_id=session_id,
                        step="pipeline",
                        state=StepState.FAILED,
                        message=sanitized_error,
                    )
                )
            except Exception:
                logger.exception("Failed to emit failed event for %s", session_id[:8])
            raise


async def _execute_pipeline_inner(
    session_id: str,
    config_dict: dict[str, Any],
    session_manager: Any,
    event_bus: Any,
    session_dir: Path,
    only_steps: list[str] | None,
    skip_steps: list[str] | None,
    create_texture_pipeline_workflow: Any,
) -> None:
    """Body of execute_pipeline_async, kept separate so the outer function
    can wrap it in a try/except that persists failure state on unhandled
    errors.
    """
    # Build context from config
    config_dict, context = _prepare_config_and_context(config_dict, session_dir)

    # Create task list
    tasks = create_texture_pipeline_workflow(context, skip=skip_steps, only=only_steps)
    total_tasks = len(tasks)

    logger.info(
        f"Running texture pipeline ({total_tasks} steps) for {session_id[:8]}..."
    )
    session_manager.update_session(session_id, {"status": "running"})

    completed_step_names: list[str] = []
    planned_step_names = [_task_to_step_name(task) for task in tasks]

    for i, task in enumerate(tasks):
        step_name = _task_to_step_name(task)

        # Check cancellation between tasks
        if session_manager.is_cancelled(session_id):
            await event_bus.emit(
                ProgressEvent(
                    session_id=session_id,
                    step=step_name,
                    state=StepState.CANCELLED,
                    message="Pipeline cancelled by user",
                )
            )
            session_manager.update_session(session_id, {"status": "cancelled"})
            logger.info(f"Pipeline cancelled for {session_id[:8]}...")
            return

        # Emit step start
        await event_bus.emit(
            ProgressEvent(
                session_id=session_id,
                step=step_name,
                state=StepState.RUNNING,
                current=i + 1,
                total=total_tasks,
                percent=0,
                message=f"Starting {task.name}",
            )
        )

        try:
            # Run synchronous task in the thread pool. The outer wrapper
            # holds the cross-process worker lock for the full pipeline
            # lifetime, including final metadata/event writes.
            loop = asyncio.get_running_loop()
            step_future = loop.run_in_executor(None, task.run, context)
            try:
                context = await asyncio.shield(step_future)
            except asyncio.CancelledError:
                logger.info(
                    "Cancellation requested during %s for %s; waiting for worker "
                    "thread to finish before releasing the session worker lock",
                    step_name,
                    session_id[:8],
                )
                await _drain_cancelled_step(
                    session_id=session_id,
                    step_name=step_name,
                    step_future=step_future,
                    session_manager=session_manager,
                )
                raise
        except Exception as e:
            logger.error(f"Step {step_name} failed for {session_id[:8]}: {e}")
            # Tasks mutate `context` with structured per-unit error records
            # (e.g. ``generate_textures_errors``) BEFORE raising the
            # threshold-gate RuntimeError. Surface those on the FAILED event
            # and persisted session metadata; without this the highest-value
            # failure mode (the threshold gate firing) loses the very
            # diagnostics this code path was added to provide.
            failed_stats = _extract_step_stats(step_name, context)
            sanitized_message = sanitize_message(
                str(e), service_config.session_storage_path
            )
            sanitized_stats = sanitize_step_stats(
                failed_stats, service_config.session_storage_path
            )
            await event_bus.emit(
                ProgressEvent(
                    session_id=session_id,
                    step=step_name,
                    state=StepState.FAILED,
                    message=sanitized_message,
                    extra=sanitized_stats or None,
                )
            )
            session_manager.update_session(
                session_id,
                {
                    "status": "failed",
                    "error": sanitized_message,
                    "failed_step": step_name,
                    "failed_step_stats": sanitized_stats,
                },
            )
            setattr(e, "_wu_failure_handled", True)
            raise

        step_stats = _extract_step_stats(step_name, context)
        sanitized_step_stats = sanitize_step_stats(
            step_stats, service_config.session_storage_path
        )
        validation_error = _get_step_validation_error(
            step_name, step_stats, planned_step_names, context
        )
        if validation_error:
            partial_results = _extract_final_stats(context, session_dir)
            logger.error(
                "Step %s produced invalid terminal state for %s: %s",
                step_name,
                session_id[:8],
                validation_error,
            )
            # Validation failures (e.g. apply_textures emitting no USD)
            # are caused by upstream gen/blend errors that already
            # populated structured records on context. Bundle the
            # failing step's own stats together with any upstream
            # ``*_errors`` lists so REST consumers see WHY -- without
            # this the FAILED event and ``/result`` only carry the
            # generic prose message.
            failed_stats = dict(step_stats)
            for upstream_key, count_key in (
                ("generate_textures_errors", "generate_textures_failed_count"),
                ("blend_textures_errors", "blend_textures_failed_count"),
            ):
                upstream_errors = context.get(upstream_key)
                if upstream_errors:
                    failed_stats.setdefault("upstream_errors", {})[
                        upstream_key.removesuffix("_errors")
                    ] = {
                        "count": context.get(count_key, len(upstream_errors)),
                        "errors": _truncate_errors(upstream_errors),
                    }
            sanitized_validation_error = sanitize_message(
                validation_error, service_config.session_storage_path
            )
            sanitized_failed_stats = sanitize_step_stats(
                failed_stats, service_config.session_storage_path
            )
            sanitized_partial_results = sanitize_step_stats(
                partial_results, service_config.session_storage_path
            )
            await event_bus.emit(
                ProgressEvent(
                    session_id=session_id,
                    step=step_name,
                    state=StepState.FAILED,
                    message=sanitized_validation_error,
                    extra=sanitized_failed_stats or None,
                )
            )
            session_manager.update_session(
                session_id,
                {
                    "status": "failed",
                    "error": sanitized_validation_error,
                    "failed_step": step_name,
                    "partial_results": sanitized_partial_results,
                    "failed_step_stats": sanitized_failed_stats,
                },
            )
            handled_error = RuntimeError(validation_error)
            setattr(handled_error, "_wu_failure_handled", True)
            raise handled_error

        # Emit step completed
        await event_bus.emit(
            ProgressEvent(
                session_id=session_id,
                step=step_name,
                state=StepState.COMPLETED,
                percent=100,
                message=f"Completed {task.name}",
                extra=sanitized_step_stats,
            )
        )

        completed_step_names.append(step_name)
        logger.info(
            f"[{i + 1}/{total_tasks}] {step_name} complete for {session_id[:8]}"
        )

    # Package output into USDZ (self-contained with textures).
    # Treat packaging failures as non-fatal — the textured .usd is already
    # written and useful on its own; the USDZ bundle is a convenience. A
    # pxr/UsdUtils exception here (e.g. unresolved asset references from
    # the original input USD) must not leave the session stuck at
    # status=running / 95%.
    if "apply_textures" in completed_step_names:
        try:
            usdz_path = await asyncio.to_thread(_package_usdz, context, session_dir)
            if usdz_path:
                context["output_usdz_path"] = usdz_path
        except Exception:
            logger.exception(
                "USDZ packaging failed for %s; continuing with .usd output only",
                session_id[:8],
            )

    # Pipeline complete
    stats = _extract_final_stats(context, session_dir)
    logger.info(f"Pipeline stats for {session_id[:8]}: {stats}")
    sanitized_stats = sanitize_step_stats(stats, service_config.session_storage_path)

    # Write stats to session metadata BEFORE emitting completion event,
    # so clients reacting to SSE "done" can immediately GET /results.
    metadata = session_manager.get_session_metadata(session_id)
    duration_seconds = 0
    if metadata and metadata.get("created_at"):
        created_at = datetime.fromisoformat(metadata["created_at"])
        duration_seconds = int((datetime.now(UTC) - created_at).total_seconds())

    session_manager.update_session(
        session_id,
        {
            "results": sanitized_stats,
            "duration_seconds": duration_seconds,
            "completed_at": datetime.now(UTC).isoformat(),
        },
    )

    # Emit final completion event
    last_step = _task_to_step_name(tasks[-1]) if tasks else "pipeline"
    await event_bus.emit(
        ProgressEvent(
            session_id=session_id,
            step=last_step,
            state=StepState.COMPLETED,
            percent=100,
            message="Pipeline completed successfully",
            extra={"pipeline_completed": True, **(sanitized_stats or {})},
        )
    )

    logger.info(f"Pipeline execution completed for {session_id[:8]}")
