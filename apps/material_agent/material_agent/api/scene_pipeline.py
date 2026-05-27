# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public Python API for the large-scene material assignment pipeline."""

from __future__ import annotations

import asyncio
import copy
import logging
import os
import shutil
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

import yaml
from world_understanding.agentic.events import EventListener, create_default_listener

from material_agent.api.defaults import (
    DEFAULT_LLM_BACKEND,
    DEFAULT_LLM_MODEL,
    PIPELINE_STEP_NAMES,
)
from material_agent.api.types import APIResult
from material_agent.scene.manifest import SceneManifest

logger = logging.getLogger(__name__)


@dataclass
class ScenePipelineInput:
    """Input parameters for the large-scene material assignment API.

    Args:
        config: Scene config as a YAML path or in-memory unified config dict.
        config_base_dir: Base directory for resolving relative paths when
            ``config`` is a dict. File configs resolve paths from their parent.
        assets: Optional asset names or prim path prefixes to process.
        skip_steps: Per-asset pipeline steps to skip.
        only_steps: Per-asset pipeline steps to run exclusively.
        from_step: Resume per-asset pipelines from this step.
        skip_existing: Skip assets/payloads already marked completed.
        max_workers: Number of parallel scene asset workers.
        resume: Reuse existing analyze/extract outputs and per-asset checkpoints.
        clean: Delete the scene working directory before starting.
        no_render: Skip composed scene rendering.
        clear_materials: Clear original material bindings before final render.
        output_usd_path: Optional composed scene output path.
        validate_output: Run scene validation when supported.
        fail_on_validation_error: Mark API result failed when validation fails.
        simulate: Patch model/render backends to mock and generate fake predictions.
        simulate_mock_analyze: Also mock the scene analyze LLM in simulate mode.
        predict_max_workers: Override per-asset ``steps.predict.max_workers``.
        cancel_checker: Optional callback returning True when the run should
            stop before the next scene stage.
        event_listener: Optional progress/event listener.
        verbose: Enable verbose listener output.
    """

    config: Path | dict[str, Any]
    config_base_dir: Path | None = None
    assets: list[str] = field(default_factory=list)
    skip_steps: list[str] = field(default_factory=list)
    only_steps: list[str] = field(default_factory=list)
    from_step: str | None = None
    skip_existing: bool = False
    max_workers: int = 1
    resume: bool = False
    clean: bool = False
    no_render: bool = False
    clear_materials: bool = False
    output_usd_path: Path | None = None
    validate_output: bool = True
    fail_on_validation_error: bool = False
    simulate: bool = False
    simulate_mock_analyze: bool = False
    predict_max_workers: int | None = None
    cancel_checker: Callable[[], bool] | None = None
    event_listener: EventListener | None = None
    verbose: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.config, dict):
            if not self.config:
                raise ValueError("Config dictionary cannot be empty")
        else:
            self.config = Path(self.config)
            if not self.config.exists():
                raise FileNotFoundError(f"Config file not found: {self.config}")

        if self.max_workers < 1:
            raise ValueError("max_workers must be at least 1")

        if self.predict_max_workers is not None and self.predict_max_workers < 1:
            raise ValueError("predict_max_workers must be at least 1")


@dataclass
class ScenePipelineOutput(APIResult):
    """Output from a large-scene pipeline run."""

    working_dir: str = ""
    manifest_path: str = ""
    output_usd_path: str = ""
    rendered_images: list[str] = field(default_factory=list)
    stats_report_path: str = ""
    completed_assets: int = 0
    failed_assets: int = 0
    completed_payloads: int = 0
    failed_payloads: int = 0
    validation_passed: bool | None = None
    validation_report: dict[str, Any] | None = None
    warnings: list[str] = field(default_factory=list)
    raw_result: dict[str, Any] = field(default_factory=dict)


def _load_scene_config(
    params: ScenePipelineInput,
) -> tuple[dict[str, Any], Path | None, Path]:
    """Load config and return ``(config_dict, config_path, base_dir)``."""
    if isinstance(params.config, dict):
        base_dir = (params.config_base_dir or Path.cwd()).resolve()
        return copy.deepcopy(params.config), None, base_dir

    config_path = Path(params.config).resolve()
    with open(config_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Scene config must contain a mapping: {config_path}")
    return data, config_path, config_path.parent


def _resolve_path(path_value: str | os.PathLike[str], base_dir: Path) -> Path:
    """Resolve a config path relative to *base_dir* unless already absolute."""
    path = Path(path_value)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _get_working_dir(scene_config: dict[str, Any], base_dir: Path) -> Path:
    """Derive scene working directory from config."""
    project = scene_config.get("project", {})
    if not isinstance(project, dict):
        project = {}

    configured = project.get("working_dir")
    if configured:
        return _resolve_path(str(configured), base_dir)

    session_id = project.get("session_id", project.get("name", "scene"))
    return base_dir / f".{session_id}_scene"


def _get_manifest_path(working_dir: Path) -> Path:
    return working_dir / "manifest.json"


def _resolve_output_path(
    output_usd_path: Path | None,
    working_dir: Path,
    base_dir: Path,
) -> Path:
    """Resolve the composed scene output path for the public API.

    Callers may pass an absolute output path. Relative overrides are resolved
    from the same base used by scene config paths so dict and file configs have
    predictable behavior.
    """
    if output_usd_path is None:
        return working_dir / "output" / "composed_scene.usd"

    path = Path(output_usd_path)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _resolve_usd_path(scene_config: dict[str, Any], base_dir: Path) -> Path:
    input_section = scene_config.get("input", {})
    if not isinstance(input_section, dict):
        raise ValueError("Scene config input section must be a mapping")

    usd_path_str = input_section.get("usd_path")
    if not usd_path_str:
        raise ValueError("Scene config requires input.usd_path")

    usd_path = _resolve_path(str(usd_path_str), base_dir)
    if not usd_path.exists():
        raise FileNotFoundError(f"USD file not found: {usd_path}")
    return usd_path


def _validate_large_scene_stage_file(usd_path: Path) -> str:
    """Validate that the input is one composed USD stage with a default root."""
    from pxr import Usd

    try:
        stage = Usd.Stage.Open(str(usd_path))
    except Exception as exc:
        raise ValueError(
            "Large-scene input must be a valid composed USD stage; "
            f"failed to open {usd_path}: {exc}"
        ) from exc

    if not stage:
        raise ValueError(
            "Large-scene input must be a valid composed USD stage; "
            f"failed to open {usd_path}"
        )

    default_prim = stage.GetDefaultPrim()
    if not default_prim or not default_prim.IsValid():
        raise ValueError(
            "Large-scene input must be one composed USD stage with a valid "
            "default root prim (defaultPrim metadata). It is not accepted as "
            "a collection of USD files."
        )

    return str(default_prim.GetPath())


def _normalize_material_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """Normalize material entries for scene collect's flat YAML contract."""
    normalized = dict(entry)
    if "binding" not in normalized and "prim_path" in normalized:
        normalized["binding"] = normalized["prim_path"]
    return normalized


def _resolve_or_materialize_material_library_yaml(
    scene_config: dict[str, Any],
    base_dir: Path,
    working_dir: Path,
) -> Path:
    """Resolve or create the flat ``materials.yaml`` used by scene collect.

    Scene configs historically point to ``materials.path``. Service-built
    configs usually carry inline ``materials.library_path`` + ``entries``.
    This bridge writes a session-local flat YAML for the inline form.
    """
    materials_section = scene_config.setdefault("materials", {})
    if not isinstance(materials_section, dict):
        raise ValueError("Scene config materials section must be a mapping")

    material_path = materials_section.get("path")
    if material_path:
        resolved = _resolve_path(str(material_path), base_dir)
        if not resolved.exists():
            raise FileNotFoundError(f"Material library YAML not found: {resolved}")
        materials_section["path"] = str(resolved)
        return resolved

    library_path = materials_section.get("library_path")
    entries = materials_section.get("entries")
    if not library_path or not isinstance(entries, list):
        raise ValueError(
            "Scene config requires materials.path or "
            "materials.library_path + materials.entries"
        )

    library_usd_path = _resolve_path(str(library_path), base_dir)
    if not library_usd_path.exists():
        raise FileNotFoundError(f"Material library USD not found: {library_usd_path}")

    materials_dir = working_dir / "materials"
    materials_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = materials_dir / "materials.yaml"

    try:
        library_path_for_yaml = os.path.relpath(
            library_usd_path.resolve(), yaml_path.parent.resolve()
        )
    except ValueError:
        library_path_for_yaml = str(library_usd_path.resolve())

    material_data = {
        "library_path": library_path_for_yaml,
        "entries": [
            _normalize_material_entry(entry)
            for entry in entries
            if isinstance(entry, dict)
        ],
    }
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(material_data, f, sort_keys=False)

    materials_section.clear()
    materials_section["path"] = str(yaml_path)
    logger.info("Materialized scene materials YAML: %s", yaml_path)
    return yaml_path


def _steps_before(step_name: str) -> list[str]:
    """Return pipeline step names before *step_name* in canonical order."""
    step_names: list[str] = list(PIPELINE_STEP_NAMES)
    if step_name not in step_names:
        raise ValueError(
            f"Unknown step '{step_name}'. Valid steps: {', '.join(step_names)}"
        )
    return step_names[: step_names.index(step_name)]


def _merge_skip_steps(
    explicit_skip_steps: list[str],
    implied_skip_steps: list[str],
) -> list[str]:
    """Merge skip lists preserving canonical order for implied steps."""
    seen: set[str] = set()
    merged: list[str] = []
    for step in [*implied_skip_steps, *explicit_skip_steps]:
        if step not in seen:
            merged.append(step)
            seen.add(step)
    return merged


def _asset_counts(manifest: SceneManifest) -> tuple[int, int, int, int]:
    completed_assets = sum(1 for sa in manifest.sub_assets if sa.status == "completed")
    failed_assets = sum(1 for sa in manifest.sub_assets if sa.status == "failed")
    completed_payloads = sum(
        1 for pg in manifest.payload_groups if pg.status == "completed"
    )
    failed_payloads = sum(1 for pg in manifest.payload_groups if pg.status == "failed")
    return completed_assets, failed_assets, completed_payloads, failed_payloads


def _emit_stage_event(
    listener: EventListener,
    event_type: str,
    step_name: str,
    message: str,
    **data: Any,
) -> None:
    """Emit a service-friendly scene stage event."""
    payload = {
        "step_name": step_name,
        "workflow_type": "scene_pipeline",
        "message": message,
        **data,
    }
    listener.event(event_type, payload)


def _emit_scene_progress(
    listener: EventListener,
    step_name: str,
    message: str,
    progress: dict[str, Any],
) -> None:
    """Emit normalized progress for long-running scene stages."""
    current = int(progress.get("current") or 0)
    total = int(progress.get("total") or 0)
    percent = int((current / total) * 100) if total > 0 else 0
    _emit_stage_event(
        listener,
        "step.progress",
        step_name,
        message,
        current=current,
        total=total,
        percent=min(100, max(0, percent)),
        completed=progress.get("completed", 0),
        failed=progress.get("failed", 0),
        asset_id=progress.get("asset_id"),
        asset_name=progress.get("asset_name"),
        asset_status=progress.get("asset_status"),
    )


def _report_validation_passed(report: Any) -> bool:
    assets_failed = any(not asset.ok for asset in getattr(report, "assets", []))
    payloads_failed = any(not payload.ok for payload in getattr(report, "payloads", []))
    scene_errors = bool(getattr(report, "errors", []))
    return not (assets_failed or payloads_failed or scene_errors)


def _report_to_dict(report: Any) -> dict[str, Any]:
    if is_dataclass(report) and not isinstance(report, type):
        data = asdict(report)
        if isinstance(data, dict):
            return data
    if isinstance(report, dict):
        return report
    return {"repr": repr(report)}


def run_scene_pipeline(params: ScenePipelineInput) -> ScenePipelineOutput:
    """Run the large-scene material assignment pipeline synchronously."""
    listener = params.event_listener or create_default_listener(verbose=params.verbose)
    warnings: list[str] = []
    current_stage: str | None = None

    def check_cancelled(step_name: str | None = None) -> None:
        if not params.cancel_checker or not params.cancel_checker():
            return
        cancelled_step = step_name or current_stage or "scene_pipeline"
        if current_stage:
            _emit_stage_event(
                listener,
                "step.cancelled",
                current_stage,
                "Scene pipeline cancelled",
            )
        listener.event(
            "workflow.cancelled",
            {
                "workflow_type": "scene_pipeline",
                "step_name": cancelled_step,
                "message": "Scene pipeline cancellation requested",
            },
        )
        raise asyncio.CancelledError("Scene pipeline cancellation requested")

    def check_cancelled_for_worker() -> bool:
        check_cancelled()
        return False

    def start_stage(step_name: str, message: str) -> None:
        nonlocal current_stage
        check_cancelled(step_name)
        current_stage = step_name
        _emit_stage_event(
            listener,
            "step.started",
            step_name,
            message,
        )

    def complete_stage(step_name: str, message: str, **data: Any) -> None:
        nonlocal current_stage
        _emit_stage_event(
            listener,
            "step.completed",
            step_name,
            message,
            **data,
        )
        if current_stage == step_name:
            current_stage = None

    try:
        scene_config, config_path, base_dir = _load_scene_config(params)
        check_cancelled("scene_pipeline")

        if params.simulate:
            from material_agent.api.simulate_config import patch_config_for_simulate

            scene_config = patch_config_for_simulate(
                scene_config, mock_analyze=params.simulate_mock_analyze
            )

        usd_path = _resolve_usd_path(scene_config, base_dir)
        default_prim_path = _validate_large_scene_stage_file(usd_path)
        check_cancelled("scene_pipeline")
        working_dir = _get_working_dir(scene_config, base_dir)
        from world_understanding.utils.token_tracking import TokenTracker

        scene_token_tracker = TokenTracker()

        if params.clean and working_dir.exists():
            listener.info(f"Cleaning scene working directory: {working_dir}")
            shutil.rmtree(working_dir)

        working_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = _get_manifest_path(working_dir)
        names_filter = params.assets or None

        material_library_yaml = _resolve_or_materialize_material_library_yaml(
            scene_config, base_dir, working_dir
        )

        skip_steps = list(params.skip_steps)
        resume = params.resume
        if params.from_step:
            skip_steps = _merge_skip_steps(skip_steps, _steps_before(params.from_step))
            resume = True

        listener.event(
            "workflow.started",
            {
                "workflow_type": "scene_pipeline",
                "config_type": "dict" if config_path is None else "file",
                "usd_path": str(usd_path),
                "default_prim_path": default_prim_path,
                "working_dir": str(working_dir),
                "max_workers": params.max_workers,
            },
        )

        scene_section = scene_config.get("scene", {})
        if not isinstance(scene_section, dict):
            scene_section = {}

        # Step 1: analyze
        if resume and manifest_path.exists():
            start_stage("scene_analyze", "Reusing existing scene manifest")
            listener.info("Scene analyze skipped: reusing existing manifest")
            manifest = SceneManifest.load(manifest_path)
            complete_stage(
                "scene_analyze",
                "Reused existing scene manifest",
                skipped=True,
                manifest_path=str(manifest_path),
            )
        else:
            start_stage("scene_analyze", "Analyzing large scene")
            listener.info("Analyzing large scene")
            analyze_opts = scene_section.get("analyze", {})
            if not isinstance(analyze_opts, dict):
                analyze_opts = {}
            filters = scene_section.get("filters", {})
            if not isinstance(filters, dict):
                filters = {}

            llm_config = analyze_opts.get("llm") or {
                "backend": DEFAULT_LLM_BACKEND,
                "model": DEFAULT_LLM_MODEL,
                "temperature": 0.1,
                "max_tokens": 256,
            }

            from material_agent.scene.analyze import analyze_scene

            manifest = analyze_scene(
                scene_usd_path=usd_path,
                skip_geometry=analyze_opts.get("skip_geometry", False),
                building_block_min_reuse=analyze_opts.get(
                    "building_block_min_reuse", 20
                ),
                filters=filters,
                llm_config=llm_config,
                token_tracker=scene_token_tracker,
            )
            manifest.save(manifest_path)
            complete_stage(
                "scene_analyze",
                "Analyzed large scene",
                manifest_path=str(manifest_path),
                sub_assets=len(manifest.sub_assets),
                instance_groups=len(manifest.instance_groups),
                payload_groups=len(manifest.payload_groups),
            )

        # Step 2: extract + config generation
        extracted_dir = working_dir / "extracted"
        configs_dir = working_dir / "configs"
        if resume and extracted_dir.exists() and configs_dir.exists():
            start_stage("scene_extract", "Reusing extracted scene assets")
            listener.info("Scene extraction skipped: reusing extracted assets")
            complete_stage(
                "scene_extract",
                "Reused extracted scene assets",
                skipped=True,
                extracted_dir=str(extracted_dir),
                configs_dir=str(configs_dir),
            )
        else:
            start_stage("scene_extract", "Extracting scene sub-assets")
            listener.info("Extracting scene sub-assets")
            extract_opts = scene_section.get("extract", {})
            if not isinstance(extract_opts, dict):
                extract_opts = {}

            from material_agent.scene.config_gen import (
                generate_all_configs,
                generate_all_payload_configs,
            )
            from material_agent.scene.extract import extract_all

            manifest = extract_all(
                scene_usd_path=usd_path,
                manifest=manifest,
                output_dir=extracted_dir,
                names_filter=names_filter,
                flatten=extract_opts.get("flatten", True),
                max_workers=extract_opts.get("max_workers", 1),
            )
            manifest = generate_all_configs(
                manifest=manifest,
                scene_config=scene_config,
                configs_dir=configs_dir,
                scene_config_dir=base_dir,
                names_filter=names_filter,
            )
            if manifest.payload_groups:
                manifest = generate_all_payload_configs(
                    manifest=manifest,
                    scene_config=scene_config,
                    configs_dir=configs_dir,
                    scene_config_dir=base_dir,
                )
            manifest.save(manifest_path)
            complete_stage(
                "scene_extract",
                "Extracted scene sub-assets",
                extracted_dir=str(extracted_dir),
                configs_dir=str(configs_dir),
                sub_assets=len(manifest.sub_assets),
                payload_groups=len(manifest.payload_groups),
            )

        if params.from_step:
            for sub_asset in manifest.sub_assets:
                if sub_asset.status == "completed":
                    sub_asset.status = "extracted"
            for payload_group in manifest.payload_groups:
                if payload_group.status == "completed":
                    payload_group.status = "pending"
            manifest.save(manifest_path)

        # Step 3: per-asset and payload pipelines
        material_names: list[str] | None = None
        if params.simulate:
            from material_agent.scene.simulate import load_material_names_from_config

            config_ref = config_path or (base_dir / "scene_config.yaml")
            material_names = load_material_names_from_config(scene_config, config_ref)

        listener.info("Running per-asset material pipelines")
        from material_agent.scene.run import run_all, run_all_payloads_bottomup

        start_stage("scene_run_assets", "Running per-asset material pipelines")

        def report_asset_progress(progress: dict[str, Any]) -> None:
            current = int(progress.get("current") or 0)
            total = int(progress.get("total") or 0)
            asset_name = progress.get("asset_name") or "asset"
            asset_status = progress.get("asset_status") or "processed"
            message = progress.get("message")
            if not isinstance(message, str) or not message:
                message = (
                    f"Processed {current}/{total} scene assets "
                    f"({asset_name}: {asset_status})"
                )
            _emit_scene_progress(
                listener,
                "scene_run_assets",
                message,
                progress,
            )

        manifest = run_all(
            manifest=manifest,
            manifest_path=manifest_path,
            names_filter=names_filter,
            skip_steps=skip_steps,
            only_steps=params.only_steps or None,
            skip_existing=params.skip_existing,
            max_workers=params.max_workers,
            verbose=params.verbose,
            simulate=params.simulate,
            material_names=material_names,
            resume=resume,
            from_step=params.from_step,
            predict_max_workers=params.predict_max_workers,
            cancel_checker=check_cancelled_for_worker,
            progress_callback=report_asset_progress,
        )
        completed_assets, failed_assets, completed_payloads, failed_payloads = (
            _asset_counts(manifest)
        )
        complete_stage(
            "scene_run_assets",
            "Completed per-asset material pipelines",
            completed_assets=completed_assets,
            failed_assets=failed_assets,
            completed_payloads=completed_payloads,
            failed_payloads=failed_payloads,
        )

        if manifest.payload_groups:
            start_stage("scene_run_payloads", "Running payload material pipelines")
            listener.info("Running payload material pipelines")
            manifest = run_all_payloads_bottomup(
                manifest=manifest,
                manifest_path=manifest_path,
                scene_config=scene_config,
                configs_dir=configs_dir,
                scene_config_dir=base_dir,
                skip_steps=skip_steps,
                only_steps=params.only_steps or None,
                skip_existing=params.skip_existing,
                max_workers=params.max_workers,
                verbose=params.verbose,
                simulate=params.simulate,
                material_names=material_names,
                resume=resume,
                from_step=params.from_step,
                predict_max_workers=params.predict_max_workers,
                cancel_checker=check_cancelled_for_worker,
            )
            completed_assets, failed_assets, completed_payloads, failed_payloads = (
                _asset_counts(manifest)
            )
            complete_stage(
                "scene_run_payloads",
                "Completed payload material pipelines",
                completed_assets=completed_assets,
                failed_assets=failed_assets,
                completed_payloads=completed_payloads,
                failed_payloads=failed_payloads,
            )
        else:
            start_stage("scene_run_payloads", "No payload material pipelines to run")
            complete_stage(
                "scene_run_payloads",
                "No payload material pipelines to run",
                skipped=True,
            )

        manifest.save(manifest_path)

        # Step 3.5: reconcile
        reconcile_config = scene_section.get("reconcile")
        if isinstance(reconcile_config, dict) and reconcile_config.get("enabled", True):
            start_stage("scene_reconcile", "Reconciling scene predictions")
            listener.info("Reconciling scene predictions")
            from material_agent.scene.reconcile import (
                apply_remapping,
                reconcile_predictions,
            )

            reconcile_llm = reconcile_config.get("llm") or scene_section.get(
                "analyze", {}
            ).get("llm", {})
            material_names_for_context = _load_material_names(material_library_yaml)
            remap = reconcile_predictions(
                manifest=manifest,
                llm_config=reconcile_llm,
                materials_list=material_names_for_context,
                token_tracker=scene_token_tracker,
            )
            if remap:
                apply_remapping(manifest, remap)
            complete_stage(
                "scene_reconcile",
                "Reconciled scene predictions",
                remapped_materials=len(remap),
            )
        else:
            start_stage("scene_reconcile", "Scene reconciliation skipped")
            complete_stage(
                "scene_reconcile",
                "Scene reconciliation skipped",
                skipped=True,
            )

        # Step 3.6: harmonize
        harmonize_config = scene_section.get("harmonize", {})
        if not isinstance(harmonize_config, dict):
            harmonize_config = {}
        if harmonize_config.get("enabled", True):
            start_stage("scene_harmonize", "Harmonizing scene predictions")
            listener.info("Harmonizing scene predictions")
            from material_agent.scene.harmonize import harmonize_scene_predictions

            harmonize_llm = (
                harmonize_config.get("llm")
                or (
                    scene_section.get("reconcile", {}).get("llm", {})
                    if isinstance(scene_section.get("reconcile", {}), dict)
                    else {}
                )
                or (
                    scene_section.get("analyze", {}).get("llm", {})
                    if isinstance(scene_section.get("analyze", {}), dict)
                    else {}
                )
            )
            harmonize_mode = harmonize_config.get("mode") or (
                "full" if harmonize_llm else "simple"
            )
            harmonize_scene_predictions(
                manifest=manifest,
                llm_config=harmonize_llm if harmonize_mode == "full" else None,
                mode=harmonize_mode,
                token_tracker=scene_token_tracker,
            )
            complete_stage(
                "scene_harmonize",
                "Harmonized scene predictions",
                mode=harmonize_mode,
            )
        else:
            start_stage("scene_harmonize", "Scene harmonization skipped")
            complete_stage(
                "scene_harmonize",
                "Scene harmonization skipped",
                skipped=True,
            )

        # Step 4: collect + render
        output_path = _resolve_output_path(
            params.output_usd_path,
            working_dir,
            base_dir,
        )
        start_stage("scene_collect", "Applying materials and composing scene")
        listener.info(f"Applying materials and composing scene: {output_path}")
        from material_agent.scene.collect import apply_and_compose

        apply_and_compose(
            scene_usd_path=usd_path,
            manifest=manifest,
            output_usd_path=output_path,
            material_library_yaml=material_library_yaml,
            names_filter=names_filter,
        )
        complete_stage(
            "scene_collect",
            "Applied materials and composed scene",
            output_usd_path=str(output_path),
        )

        rendered_images: list[str] = []
        render_config = scene_config.get("steps", {}).get("render", {})
        if not isinstance(render_config, dict):
            render_config = {}
        if not params.no_render and render_config.get("enabled", True):
            start_stage("scene_render", "Rendering composed scene")
            listener.info("Rendering composed scene")
            from material_agent.scene.collect import render_composed_scene

            rendered = render_composed_scene(
                composed_usd_path=output_path,
                output_dir=output_path.parent,
                camera_corners=render_config.get(
                    "camera_corners", ["+x+y+z", "-x-y-z"]
                ),
                image_width=render_config.get("image_width", 1024),
                image_height=render_config.get("image_height", 1024),
                camera_margin=render_config.get("camera_margin", 1.0),
                background_color=tuple(
                    render_config.get("background_color", [1.0, 1.0, 1.0])
                ),
                clear_materials=params.clear_materials,
            )
            rendered_images = [str(path) for path in rendered]
            complete_stage(
                "scene_render",
                "Rendered composed scene",
                rendered_images=rendered_images,
            )
        else:
            start_stage("scene_render", "Composed scene rendering skipped")
            complete_stage(
                "scene_render",
                "Composed scene rendering skipped",
                skipped=True,
            )

        # Step 5: validate composed-scene material coverage.
        validation_passed: bool | None = None
        validation_report: dict[str, Any] | None = None
        if params.validate_output:
            start_stage("scene_validate", "Validating composed scene output")
            listener.info("Validating composed scene output")
            from material_agent.scene.validate import validate_scene_outputs

            report = validate_scene_outputs(
                manifest_path=manifest_path,
                working_dir=working_dir,
                composed_scene_path=output_path,
                verbose=params.verbose,
            )
            validation_passed = _report_validation_passed(report)
            validation_report = _report_to_dict(report)
            complete_stage(
                "scene_validate",
                "Validated composed scene output",
                validation_passed=validation_passed,
                validation_report=validation_report,
            )
            if params.fail_on_validation_error and not validation_passed:
                from material_agent.scene.stats import write_scene_stats_report

                stats_report_path = write_scene_stats_report(
                    manifest=manifest,
                    working_dir=working_dir,
                    output_dir=output_path.parent,
                    output_usd_path=output_path,
                    scene_operation_token_stats=scene_token_tracker.get_stats(),
                )
                listener.event(
                    "workflow.failed",
                    {
                        "workflow_type": "scene_pipeline",
                        "error": "Scene validation failed",
                        "validation_passed": validation_passed,
                        "validation_report": validation_report,
                        "working_dir": str(working_dir),
                        "manifest_path": str(manifest_path),
                        "output_usd_path": str(output_path),
                        "stats_report_path": str(stats_report_path),
                    },
                )
                return _build_output(
                    success=False,
                    error="Scene validation failed",
                    working_dir=working_dir,
                    manifest_path=manifest_path,
                    output_path=output_path,
                    rendered_images=rendered_images,
                    manifest=manifest,
                    validation_passed=validation_passed,
                    validation_report=validation_report,
                    warnings=warnings,
                    stats_report_path=stats_report_path,
                )
        else:
            start_stage("scene_validate", "Scene validation skipped")
            complete_stage(
                "scene_validate",
                "Scene validation skipped",
                skipped=True,
            )

        from material_agent.scene.stats import write_scene_stats_report

        stats_report_path = write_scene_stats_report(
            manifest=manifest,
            working_dir=working_dir,
            output_dir=output_path.parent,
            output_usd_path=output_path,
            scene_operation_token_stats=scene_token_tracker.get_stats(),
        )
        listener.info(f"Scene stats report written to {stats_report_path}")

        listener.event(
            "workflow.completed",
            {
                "workflow_type": "scene_pipeline",
                "manifest_path": str(manifest_path),
                "output_usd_path": str(output_path),
                "stats_report_path": str(stats_report_path),
            },
        )
        return _build_output(
            success=True,
            error=None,
            working_dir=working_dir,
            manifest_path=manifest_path,
            output_path=output_path,
            rendered_images=rendered_images,
            manifest=manifest,
            validation_passed=validation_passed,
            validation_report=validation_report,
            warnings=warnings,
            stats_report_path=stats_report_path,
        )

    except Exception as exc:
        logger.exception("Scene pipeline failed")
        listener.error(f"Scene pipeline failed: {exc}")
        if current_stage:
            listener.event(
                "step.failed",
                {
                    "step_name": current_stage,
                    "workflow_type": "scene_pipeline",
                    "error": str(exc),
                },
            )
        listener.event(
            "workflow.failed",
            {"workflow_type": "scene_pipeline", "error": str(exc)},
        )
        return ScenePipelineOutput(success=False, error=str(exc), warnings=warnings)


def _load_material_names(material_library_yaml: Path) -> list[str]:
    """Load material names from the flat or nested material YAML formats."""
    with open(material_library_yaml, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return []
    if not isinstance(data, dict):
        raise ValueError(
            f"Material library YAML must be a mapping: {material_library_yaml}"
        )
    materials_section = data.get("materials", data)
    if not isinstance(materials_section, dict):
        raise ValueError(
            "Material library YAML must contain a mapping at "
            f"'materials': {material_library_yaml}"
        )
    entries = materials_section.get("entries", [])
    if entries is None:
        return []
    if not isinstance(entries, list):
        raise ValueError(
            f"Material library YAML 'entries' must be a list: {material_library_yaml}"
        )
    return [
        entry.get("name", "")
        for entry in entries
        if isinstance(entry, dict) and entry.get("name")
    ]


def _build_output(
    success: bool,
    error: str | None,
    working_dir: Path,
    manifest_path: Path,
    output_path: Path,
    rendered_images: list[str],
    manifest: SceneManifest,
    validation_passed: bool | None,
    validation_report: dict[str, Any] | None,
    warnings: list[str],
    stats_report_path: Path | None = None,
) -> ScenePipelineOutput:
    completed_assets, failed_assets, completed_payloads, failed_payloads = (
        _asset_counts(manifest)
    )
    return ScenePipelineOutput(
        success=success,
        error=error,
        working_dir=str(working_dir),
        manifest_path=str(manifest_path),
        output_usd_path=str(output_path),
        rendered_images=rendered_images,
        stats_report_path=str(stats_report_path) if stats_report_path else "",
        completed_assets=completed_assets,
        failed_assets=failed_assets,
        completed_payloads=completed_payloads,
        failed_payloads=failed_payloads,
        validation_passed=validation_passed,
        validation_report=validation_report,
        warnings=warnings,
        raw_result={
            "analysis": manifest.analysis,
            "sub_assets": len(manifest.sub_assets),
            "instance_groups": len(manifest.instance_groups),
            "payload_groups": len(manifest.payload_groups),
        },
    )


async def arun_scene_pipeline(params: ScenePipelineInput) -> ScenePipelineOutput:
    """Run the large-scene pipeline from async code.

    The scene implementation currently uses synchronous USD operations and
    per-asset sync pipeline calls, so this wrapper runs it in a worker thread.
    """
    return await asyncio.to_thread(run_scene_pipeline, params)


def scene_pipeline(
    config: Path | dict[str, Any],
    **kwargs: Any,
) -> ScenePipelineOutput:
    """Convenience sync function for the large-scene pipeline API."""
    return run_scene_pipeline(ScenePipelineInput(config=config, **kwargs))


async def ascene_pipeline(
    config: Path | dict[str, Any],
    **kwargs: Any,
) -> ScenePipelineOutput:
    """Convenience async function for the large-scene pipeline API."""
    return await arun_scene_pipeline(ScenePipelineInput(config=config, **kwargs))
