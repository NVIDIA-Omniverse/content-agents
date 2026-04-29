# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pipeline execution for Texture Agent Service.

Wraps the synchronous texture-agent pipeline by running each task
individually via asyncio.to_thread(), emitting progress events between steps.
"""

import asyncio
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..runtime.bus import get_event_bus
from ..runtime.events import ProgressEvent, StepState

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

    from pxr import Sdf, Usd, UsdUtils

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
    stage = Usd.Stage.Open(str(output_usd))
    if not stage:
        return None

    rewritten = 0
    for prim in stage.Traverse():
        for attr in prim.GetAttributes():
            val = attr.Get()
            if not isinstance(val, Sdf.AssetPath) or not val.path:
                continue
            old_path = val.path
            filename = Path(old_path).name
            if filename.endswith(".png"):
                new_path = f"../textures/{filename}"
                attr.Set(Sdf.AssetPath(new_path))
                rewritten += 1

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


def _extract_step_stats(step_name: str, context: dict[str, Any]) -> dict:
    """Extract statistics from context after a step completes."""
    stats: dict[str, Any] = {}

    if step_name == "discover_materials":
        materials = context.get("discovered_materials", [])
        stats["materials_found"] = len(materials)

    elif step_name == "generate_textures":
        generated = context.get("generated_textures", {})
        stats["textures_generated"] = len(generated)

    elif step_name == "blend_textures":
        blended = context.get("blended_textures", {})
        stats["textures_blended"] = len(blended)

    elif step_name == "apply_textures":
        output_paths = context.get("output_usd_paths", [])
        stats["output_usd_count"] = len(output_paths)

    elif step_name == "render":
        rendered = context.get("rendered_image_paths", [])
        stats["renders_count"] = len(rendered)

    return stats


def _extract_final_stats(context: dict[str, Any], session_dir: Path) -> dict[str, Any]:
    """Extract final pipeline statistics from context and files."""
    stats = {
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

    return stats


def _get_step_validation_error(
    step_name: str,
    step_stats: dict[str, Any],
    planned_steps: list[str],
) -> str | None:
    """Return a terminal validation error for a completed step, if any.

    Some task implementations can legitimately "complete" while doing no useful
    work. The service should convert those cases into failed sessions instead of
    reporting a false-positive success.
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
            return (
                "Texture application produced no output USD files. "
                "The pipeline cannot be reported as completed."
            )

    return None


async def execute_pipeline_async(
    session_id: str,
    config_dict: dict[str, Any],
    session_manager: Any,
    only_steps: list[str] | None = None,
    skip_steps: list[str] | None = None,
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
    """
    from texture_agent.workflows.factory import create_texture_pipeline_workflow

    logger.info(f"Pipeline execution started for {session_id[:8]}...")

    event_bus = get_event_bus()
    session_dir = session_manager.get_session_dir(session_id)

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
        # cleanup that normally persists "cancelled" is skipped — handle that
        # final transition here so /status flips from "cancelling" to
        # "cancelled" instead of stalling.
        #
        # Persist the disk state synchronously BEFORE awaiting the event emit:
        # JobRegistry.cancel wraps task.cancel() in wait_for(timeout=5s) and
        # may fire a second task.cancel() if cleanup is slow. A re-raised
        # CancelledError on the await would skip the disk update otherwise.
        #
        # Caveat: pipeline steps run via asyncio.to_thread, which cancels the
        # asyncio side but not the underlying thread. The thread can keep
        # writing artifacts after we mark "cancelled" — see follow-up ticket
        # for cooperative-cancellation inside step bodies.
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
            logger.exception("Failed to emit cancelled event for %s", session_id[:8])
        raise
    except Exception as e:
        # Any uncaught error past the per-step guard (e.g. post-loop
        # packaging, final stats) must still flip the session to "failed"
        # so /status doesn't stay at "running" forever.
        logger.exception("Unhandled pipeline error for %s: %s", session_id[:8], e)
        try:
            session_manager.update_session(
                session_id,
                {"status": "failed", "error": str(e)},
            )
        except Exception:
            logger.exception("Failed to persist failed status for %s", session_id[:8])
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
            # Run synchronous task in thread pool
            context = await asyncio.to_thread(task.run, context)
        except Exception as e:
            logger.error(f"Step {step_name} failed for {session_id[:8]}: {e}")
            await event_bus.emit(
                ProgressEvent(
                    session_id=session_id,
                    step=step_name,
                    state=StepState.FAILED,
                    message=str(e),
                )
            )
            session_manager.update_session(
                session_id,
                {
                    "status": "failed",
                    "error": str(e),
                    "failed_step": step_name,
                },
            )
            raise

        step_stats = _extract_step_stats(step_name, context)
        validation_error = _get_step_validation_error(
            step_name, step_stats, planned_step_names
        )
        if validation_error:
            partial_results = _extract_final_stats(context, session_dir)
            logger.error(
                "Step %s produced invalid terminal state for %s: %s",
                step_name,
                session_id[:8],
                validation_error,
            )
            await event_bus.emit(
                ProgressEvent(
                    session_id=session_id,
                    step=step_name,
                    state=StepState.FAILED,
                    message=validation_error,
                )
            )
            session_manager.update_session(
                session_id,
                {
                    "status": "failed",
                    "error": validation_error,
                    "failed_step": step_name,
                    "partial_results": partial_results,
                },
            )
            raise RuntimeError(validation_error)

        # Emit step completed
        await event_bus.emit(
            ProgressEvent(
                session_id=session_id,
                step=step_name,
                state=StepState.COMPLETED,
                percent=100,
                message=f"Completed {task.name}",
                extra=step_stats,
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
            "results": stats,
            "duration_seconds": duration_seconds,
            "completed_at": datetime.now(UTC).isoformat(),
        },
    )

    # Emit final completion event
    last_step = _task_to_step_name(tasks[-1]) if tasks else "render"
    await event_bus.emit(
        ProgressEvent(
            session_id=session_id,
            step=last_step,
            state=StepState.COMPLETED,
            percent=100,
            message="Pipeline completed successfully",
            extra={"pipeline_completed": True, **stats},
        )
    )

    logger.info(f"Pipeline execution completed for {session_id[:8]}")
