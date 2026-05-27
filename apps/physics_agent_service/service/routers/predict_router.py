# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Predict API endpoints — first-class predict workflow.

Distinct route group from ``/pipeline``:

* ``POST /predict`` — start an async prediction job (Mode A or Mode B).
* ``GET  /predict/{session_id}/status`` — current status (mirrors /pipeline).
* ``GET  /predict/{session_id}/results`` — completed predict results.
* ``GET  /predict/{session_id}/events`` — SSE progress stream.
* ``POST /predict/{session_id}/cancel`` — cancel an in-flight predict job.

The predict route is intentionally NOT a thin alias for ``/pipeline``: it
runs a prediction-only workflow that auto-detects whether the session
already has a prepared dataset (Mode A → just predict) or needs the
minimum upstream prep first (Mode B → optimize_usd → identify_asset →
build_dataset_usd → build_dataset_prepare_dataset → predict). The
``/pipeline`` workflow remains unchanged and continues to be the right
entry point for the full classify/apply flow.

Reuses shared infra (session manager, job registry, event bus, SSE,
cancellation, artifact storage) — only the route definitions and request /
response schemas live here.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path

import yaml
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from physics_agent.api.defaults import build_default_pipeline_config
from sse_starlette import EventSourceResponse
from world_understanding.utils.s3_utils import download_file_from_s3

from ..config import config
from ..models.responses import (
    PipelineError,
    PipelineStatus,
    PredictResults,
    SessionCreated,
)
from ..runtime import get_event_bus, get_job_registry
from ..session.manager import SessionManager
from ..workers.predict_executor import execute_predict_async

logger = logging.getLogger(__name__)

# Distinct prefix and tag — clients should treat /predict as its own route group,
# parallel to /pipeline and #36's planned /tune.
router = APIRouter(prefix="/predict", tags=["predict"])

# Global session manager (set by main app — same instance as /pipeline).
session_manager: SessionManager | None = None

_VALID_USD_EXTENSIONS = {".usd", ".usda", ".usdc", ".usdz"}


def get_session_manager() -> SessionManager:
    """Get the global session manager instance."""
    if session_manager is None:
        raise RuntimeError("SessionManager not initialized")
    return session_manager


def set_session_manager(manager: SessionManager) -> None:
    """Set the global session manager instance."""
    global session_manager
    session_manager = manager


async def _stream_copy(
    upload: UploadFile,
    dest: Path,
    *,
    max_bytes: int,
    chunk_size: int = 2 * 1024 * 1024,
) -> int:
    """Stream upload file to disk in chunks; abort early on size overrun.

    Raises HTTPException(413) as soon as the running byte total passes
    ``max_bytes`` so an oversized upload cannot fill the session volume.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    total_bytes = 0
    try:
        with dest.open("wb") as f:
            while True:
                data = await upload.read(chunk_size)
                if not data:
                    break
                total_bytes += len(data)
                if total_bytes > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"File too large: exceeds {max_bytes // (1024 * 1024)}MB"
                        ),
                    )
                f.write(data)
        return total_bytes
    except HTTPException:
        dest.unlink(missing_ok=True)
        raise


def _resolve_dataset_path_safely(raw_path: str, manager: SessionManager) -> Path:
    """Resolve and validate an absolute ``dataset_path`` arg.

    The path must canonicalize inside one of the allowed roots — the
    SessionManager's actual storage path plus any operator-provided extras
    from ``PA_DATASET_ALLOWED_ROOTS`` (colon-separated, read live so test
    fixtures that rebind env after import still apply). Anything outside is
    rejected with 403 to prevent the route from acting as a local-file-read
    primitive.
    """
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        raise HTTPException(
            status_code=400, detail="dataset_path must be an absolute path"
        )
    try:
        real = candidate.resolve(strict=True)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=400,
            detail=f"dataset_path does not exist: {raw_path}",
        ) from e
    if not real.is_file():
        raise HTTPException(
            status_code=400,
            detail=f"dataset_path is not a regular file: {raw_path}",
        )
    if real.name != "dataset.jsonl":
        # Restrict to the canonical dataset filename so this route can't be
        # used to copy out arbitrary files (session.json, predictions.jsonl,
        # etc.) that happen to live under the session storage root.
        raise HTTPException(
            status_code=400,
            detail="dataset_path must point at a file named 'dataset.jsonl'",
        )

    env_roots = os.environ.get(
        "PA_DATASET_ALLOWED_ROOTS", config.dataset_allowed_roots or ""
    )
    extra_roots = [p for p in env_roots.split(":") if p.strip()]
    allowed_roots = [Path(manager.storage_path), *map(Path, extra_roots)]
    resolved_roots = []
    for root in allowed_roots:
        try:
            resolved_roots.append(root.resolve(strict=False))
        except OSError:
            continue

    for root in resolved_roots:
        try:
            common = Path(os.path.commonpath([str(real), str(root)]))
        except ValueError:
            # commonpath raises on different drives (Windows) or empty input.
            continue
        if common == root:
            return real

    raise HTTPException(
        status_code=403,
        detail=(
            "dataset_path resolves outside allowed roots. Set "
            "PA_DATASET_ALLOWED_ROOTS to opt-in additional locations."
        ),
    )


def _preflight_s3_object_size(s3_uri: str, max_bytes: int) -> None:
    """HEAD the S3 object and reject oversized payloads before any download.

    Raises HTTPException(413) when the advertised ``ContentLength`` already
    exceeds the configured cap, so a multi-GB object cannot fill the session
    volume during a transfer that we'd reject anyway.

    Network/permission/missing-object errors here are swallowed: we want the
    real download path to produce the canonical 404/403/502 response. This
    preflight is a best-effort fast-fail for the common case where
    ``ContentLength`` is available.
    """
    try:
        # Imported lazily so unit tests can monkey-patch s3_utils internals
        # without paying the import cost on every request.
        from world_understanding.utils.s3_utils import (
            _create_s3_client,
            _parse_s3_path,
        )

        bucket, key = _parse_s3_path(s3_uri)
        s3_client = _create_s3_client()
        head = s3_client.head_object(Bucket=bucket, Key=key)
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        # Any failure (NoSuchBucket, AccessDenied, ProfileNotFound, network)
        # falls through to the real download_file_from_s3 path which has
        # full error-code translation. A failed preflight is not itself a
        # client-visible error.
        logger.debug(f"S3 head_object preflight skipped for {s3_uri}: {e}")
        return

    content_length = head.get("ContentLength")
    if content_length is None:
        return
    size_bytes = int(content_length)
    if size_bytes > max_bytes:
        size_mb = size_bytes / (1024 * 1024)
        raise HTTPException(
            status_code=413,
            detail=(
                f"S3 file too large: {size_mb:.1f}MB. "
                f"Max: {max_bytes // (1024 * 1024)}MB"
            ),
        )


def _download_s3_to_session(s3_uri: str, session_dir: Path) -> Path:
    """Download a USD file from S3 into session_dir/input/."""
    if not s3_uri.startswith("s3://") or s3_uri.count("/") < 3:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid S3 URI format: {s3_uri}. "
            "Expected s3://bucket/path/to/file.ext",
        )

    s3_filename = s3_uri.rstrip("/").rsplit("/", 1)[-1]
    if not s3_filename:
        raise HTTPException(
            status_code=400,
            detail=f"S3 URI must include an object key: {s3_uri}",
        )
    ext = Path(s3_filename).suffix.lower()
    if ext not in _VALID_USD_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid USD file type in S3 URI: {ext}. "
                f"Allowed: {', '.join(sorted(_VALID_USD_EXTENSIONS))}"
            ),
        )

    # Reject oversized objects via head_object BEFORE writing anything to
    # disk. The post-download guard at the end of this function is kept as
    # a safety net in case ContentLength is missing or the object grows
    # between HEAD and GET.
    max_bytes = config.max_upload_size_mb * 1024 * 1024
    _preflight_s3_object_size(s3_uri, max_bytes)

    local_path = session_dir / "input" / f"scene{ext}"
    local_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        download_file_from_s3(s3_uri, local_path)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"S3 object not found: {s3_uri}")
    except PermissionError:
        raise HTTPException(
            status_code=403, detail=f"Access denied to S3 object: {s3_uri}"
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to download from S3: {e}")

    size_mb = local_path.stat().st_size / (1024 * 1024)
    if size_mb > config.max_upload_size_mb:
        local_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=413,
            detail=(
                f"S3 file too large: {size_mb:.1f}MB. "
                f"Max: {config.max_upload_size_mb}MB"
            ),
        )

    return local_path


def _find_input_usd(session_dir: Path) -> Path | None:
    """Find the input USD file in a session directory."""
    input_dir = session_dir / "input"
    for ext in _VALID_USD_EXTENSIONS:
        candidate = input_dir / f"scene{ext}"
        if candidate.exists():
            return candidate
    return None


@router.post("", response_model=SessionCreated, status_code=202)
async def create_predict(
    usd_file: UploadFile | None = File(
        None,
        description=(
            "USD file to predict on (optional if dataset_path, session_id or "
            "s3_uri is provided)."
        ),
    ),
    session_id: str | None = Form(
        None,
        description=(
            "Existing session ID (e.g. from POST /pipeline/upload-usd). When "
            "the session already has a prepared dataset, /predict will run "
            "Mode A (predict only)."
        ),
    ),
    s3_uri: str | None = Form(
        None,
        description="S3 URI to a USD file (e.g. s3://bucket/path/scene.usdz)",
    ),
    dataset_path: str | None = Form(
        None,
        description=(
            "Absolute path to a prepared dataset.jsonl on the server. When "
            "set and readable, forces Mode A (predict-only)."
        ),
    ),
    user_prompt: str = Form(
        default="",
        description="Custom user prompt for VLM (optional, used in Mode B)",
    ),
    render_backend: str = Form(
        default="",
        description=(
            "Rendering backend for Mode B: 'remote' (default), 'warp', or "
            "'ovrtx'. Ignored in Mode A."
        ),
    ),
    optimize_usd: bool = Form(
        default=False,
        description=(
            "Enable USD optimization step in Mode B (default: false). Ignored "
            "in Mode A."
        ),
    ),
    enable_deinstance: bool = Form(
        default=True,
        description="Enable deinstance op when optimize_usd=true (Mode B only).",
    ),
    enable_split: bool = Form(
        default=False,
        description="Enable split-meshes op when optimize_usd=true (Mode B only).",
    ),
    enable_deduplicate: bool = Form(
        default=False,
        description="Enable deduplicate op when optimize_usd=true (Mode B only).",
    ),
) -> SessionCreated:
    """Create and execute a prediction job.

    Two input modes (auto-detected at job start):

    * **Mode A — dataset already prepared.** Triggered when ``dataset_path``
      points at a readable dataset.jsonl, or when an existing ``session_id``
      already has ``cache/dataset/dataset.jsonl``. Only the predict step
      runs; upstream prep (rendering, dataset prep) is skipped.
    * **Mode B — USD upload / s3_uri / fresh session_id.** When no prepared
      dataset is present, /predict runs the minimum upstream steps
      (``optimize_usd`` if enabled → ``identify_asset`` → ``build_dataset_usd``
      → ``build_dataset_prepare_dataset``) before predicting. ``apply_physics``
      is intentionally not part of /predict — use POST /pipeline if you need
      the full classify/apply flow.

    The detected mode is persisted to session metadata under ``predict_mode``
    and surfaced in the GET /predict/{id}/results response.
    """
    manager = get_session_manager()
    user_prompt_text = user_prompt.strip() if user_prompt else None

    # Reject ambiguous input combinations up front. The route advertises four
    # input sources, but only specific combinations are well-defined:
    #
    #   * exactly one of {usd_file, session_id, s3_uri} as the primary source
    #   * dataset_path may be supplied alone (pure Mode A) or together with
    #     session_id (override the session's prepared dataset)
    #   * dataset_path with usd_file or s3_uri is contradictory — Mode A would
    #     ignore the upload while the docs say dataset_path forces Mode A.
    #
    # Rejecting these combinations early prevents silent precedence games
    # where, for example, session_id won over a non-empty usd_file or a
    # dataset_path + s3_uri request still downloaded the S3 object.
    has_usd_file = usd_file is not None and (usd_file.filename or "").strip() != ""
    primary_sources = [
        ("usd_file", has_usd_file),
        ("session_id", bool(session_id)),
        ("s3_uri", bool(s3_uri)),
    ]
    provided_primary = [name for name, present in primary_sources if present]
    if len(provided_primary) > 1:
        raise HTTPException(
            status_code=400,
            detail=(
                "Provide exactly one of usd_file, session_id, or s3_uri "
                f"(got: {', '.join(provided_primary)})."
            ),
        )
    if dataset_path and (has_usd_file or s3_uri):
        # session_id + dataset_path is the one supported override.
        raise HTTPException(
            status_code=400,
            detail=(
                "dataset_path is incompatible with usd_file or s3_uri "
                "(those are Mode B inputs; dataset_path forces Mode A). "
                "Combine dataset_path with session_id instead, or send it alone."
            ),
        )

    # Resolve dataset_path early — must canonicalize inside an allowed root
    # so /predict cannot be used as an arbitrary local-file-read primitive.
    resolved_dataset_path: Path | None = None
    if dataset_path:
        resolved_dataset_path = _resolve_dataset_path_safely(dataset_path, manager)

    # Track whether THIS request created the session — only sessions created
    # here are safe to delete on later validation failures. Reused sessions
    # belong to the caller and may already hold uploaded USDs / artifacts.
    session_created_here = False

    # Resolve session_dir / input USD
    if session_id:
        if not await manager.session_exists(session_id):
            raise HTTPException(status_code=404, detail="Session not found")
        # Reject re-queuing while an earlier predict job on this session is
        # still in-flight or being torn down. Without this guard, two POSTs
        # for the same session could race on the same cache/ paths and the
        # second job's metadata would clobber the first's. Mirrors the
        # /pipeline/{id}/regenerate guard and uses the persisted store
        # status so it works cross-pod, plus the in-process JobRegistry
        # to close the same-pod TOCTOU window where two concurrent POSTs
        # both observe a terminal status before either writes "pending".
        # Cross-pod concurrent reruns are still a known limitation shared
        # with /pipeline (no distributed lock).
        if get_job_registry().is_running(session_id):
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Predict already running on this instance for session "
                    f"{session_id}. Wait for it to finish or cancel first."
                ),
            )
        existing_metadata = await manager.get_session_metadata(session_id)
        existing_status = (existing_metadata or {}).get("status")
        if existing_status in ("pending", "running", "cancelling"):
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Predict already {existing_status} for session "
                    f"{session_id}. Wait for it to reach a terminal state "
                    f"or cancel it first."
                ),
            )
        session_dir = manager.get_session_dir(session_id)
    elif s3_uri:
        session_id = str(uuid.uuid4())
        session_dir = await manager.create_session(session_id)
        session_created_here = True
        try:
            # _download_s3_to_session is sync (boto3 + post-download size
            # check); run it on a thread so a slow transfer doesn't block
            # the request event loop and stall other handlers on this worker.
            local_path = await asyncio.to_thread(
                _download_s3_to_session, s3_uri, session_dir
            )
            size_mb = local_path.stat().st_size / (1024 * 1024)
            logger.info(
                f"USD downloaded from S3 for /predict session "
                f"{session_id[:8]}: {size_mb:.2f}MB ({local_path.suffix})"
            )
        except HTTPException:
            await manager.delete_session(session_id)
            raise
        except Exception as e:
            logger.error(f"Failed to download USD from S3: {e}")
            await manager.delete_session(session_id)
            raise HTTPException(
                status_code=500, detail=f"Failed to download USD from S3: {e}"
            )
    elif has_usd_file:
        # has_usd_file (computed above) treats UploadFile with an empty
        # filename as "no file", matching what FastAPI hands us when the
        # multipart field is absent.
        assert usd_file is not None  # narrowed by has_usd_file
        session_id = str(uuid.uuid4())
        session_dir = await manager.create_session(session_id)
        session_created_here = True
        try:
            if usd_file.filename:
                ext = Path(usd_file.filename).suffix.lower()
                if ext not in _VALID_USD_EXTENSIONS:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"Invalid USD file type: {ext}. "
                            f"Allowed: {', '.join(sorted(_VALID_USD_EXTENSIONS))}"
                        ),
                    )
            original_ext = (
                Path(usd_file.filename).suffix.lower() if usd_file.filename else ".usd"
            )
            usd_path = session_dir / "input" / f"scene{original_ext}"
            total_bytes = await _stream_copy(
                usd_file,
                usd_path,
                max_bytes=config.max_upload_size_mb * 1024 * 1024,
            )
            size_mb = total_bytes / (1024 * 1024)
            logger.info(
                f"USD uploaded for /predict session {session_id[:8]}: "
                f"{size_mb:.2f}MB ({original_ext})"
            )
        except HTTPException:
            await manager.delete_session(session_id)
            raise
        except Exception as e:
            logger.error(f"Failed to save USD file: {e}")
            await manager.delete_session(session_id)
            raise HTTPException(status_code=500, detail=f"Failed to save USD file: {e}")
    elif resolved_dataset_path is not None:
        # Pure Mode A from explicit dataset path with no session/USD context.
        # We still need a session_dir for outputs.
        session_id = str(uuid.uuid4())
        session_dir = await manager.create_session(session_id)
        session_created_here = True
    else:
        raise HTTPException(
            status_code=400,
            detail=(
                "One of usd_file, session_id, s3_uri, or dataset_path must be provided"
            ),
        )

    # Mode B (USD-driven) needs an input USD on disk so the renderer can run.
    # Mode A doesn't — it can predict from dataset.jsonl alone. We don't
    # require an input USD when dataset_path was supplied OR when the session
    # already has a prepared dataset.
    #
    # The actual copy of the external dataset_path into the session cache is
    # deferred until *after* job_registry.reserve() succeeds, so a losing
    # concurrent rerun cannot clobber the winner's cached dataset.jsonl. We
    # only need to know *whether* mode A applies here, not have the file
    # staged yet.
    session_dataset = session_dir / "cache" / "dataset" / "dataset.jsonl"
    session_has_dataset = session_dataset.exists()
    will_be_mode_a = resolved_dataset_path is not None or session_has_dataset

    input_usd_path = _find_input_usd(session_dir)
    if not input_usd_path and not will_be_mode_a:
        # Maybe the input — or a prepared dataset — lives on a different
        # instance. Pull both before declaring failure; this mirrors what
        # execute_predict_async does at job start so the preflight can't
        # reject a session that the worker would have happily resumed.
        pulled = await manager.sync_from_store(session_id, prefix="input/")
        if pulled > 0:
            logger.info(
                f"Pulled {pulled} input file(s) from store for /predict session "
                f"{session_id[:8]}"
            )
        await manager.sync_from_store(session_id, prefix="cache/dataset/")
        input_usd_path = _find_input_usd(session_dir)
        session_has_dataset = session_dataset.exists()
        will_be_mode_a = resolved_dataset_path is not None or session_has_dataset

    if not input_usd_path and not will_be_mode_a:
        raise HTTPException(
            status_code=400,
            detail=(
                "No input USD or prepared dataset found for this session. "
                "Provide usd_file, s3_uri, or dataset_path."
            ),
        )

    # If the only Mode-A trigger is the session's own staged JSONL (no
    # caller-supplied dataset_path) AND there's no USD to fall back on,
    # verify the JSONL's images are actually resolvable now. The executor
    # would otherwise accept the JSONL, then fail asynchronously when it
    # tries to predict against missing images and falls back to Mode B
    # with no USD available — surfacing as a 202 + later "failed" status,
    # which is worse than rejecting the request up front. This applies
    # to a session whose previous run staged an external dataset_path
    # (only the JSONL was copied, not the images), or to a cross-pod
    # rerun where the image PNGs have not been synced down.
    if resolved_dataset_path is None and session_has_dataset and not input_usd_path:
        from ..workers.predict_executor import _dataset_jsonl_has_resolvable_images

        if not _dataset_jsonl_has_resolvable_images(session_dataset):
            raise HTTPException(
                status_code=400,
                detail=(
                    "Session has a staged dataset.jsonl but its referenced "
                    "images are not present (likely a previous run staged the "
                    "JSONL alone, or the per-prim PNGs have not been synced "
                    "down on this instance). Re-supply dataset_path with the "
                    "original directory, upload the USD again, or provide "
                    "s3_uri so /predict can rebuild from source."
                ),
            )

    # The optimizer flags and render_backend are ignored in Mode A per the
    # docstring; only validate them when the request actually triggers
    # Mode B so a dataset-only call can't 400 (or 500 from
    # build_default_pipeline_config) on options that won't run.
    if not will_be_mode_a:
        if optimize_usd and not any(
            [enable_deinstance, enable_split, enable_deduplicate]
        ):
            raise HTTPException(
                status_code=400,
                detail=(
                    "At least one optimization operation must be enabled when "
                    "optimize_usd is true (enable_deinstance, enable_split, or "
                    "enable_deduplicate)."
                ),
            )

    render_backend_text = render_backend.strip() if render_backend else None

    # Build a pipeline config dict so Mode B can drive the full upstream
    # workflow. In Mode A only the `predict` step + project/working_dir
    # actually matter — the rest is harmless.
    # When no input USD exists yet (Mode A from dataset_path with a fresh
    # session), use a sentinel string. Mode B never reaches that branch
    # because we already raised 400 above.
    usd_path_for_config = (
        str(input_usd_path)
        if input_usd_path
        else str(session_dir / "input" / "scene.usda")
    )

    try:
        predict_config = build_default_pipeline_config(
            session_id=session_id,
            usd_path=usd_path_for_config,
            working_dir=str(session_dir / "cache"),
            user_prompt=user_prompt_text,
            # Mode A ignores the render backend; suppress it so a typo'd
            # value can't make build_default_pipeline_config raise on a
            # request that wasn't going to render anything anyway.
            render_backend=None if will_be_mode_a else render_backend_text,
            optimize_usd=False if will_be_mode_a else optimize_usd,
            enable_deinstance=enable_deinstance,
            enable_split=enable_split,
            enable_deduplicate=enable_deduplicate,
        )
    except ValueError as e:
        # Translate config-builder rejections (bad render backend, etc.)
        # into a clean 400 instead of a 500. Only tear down the session
        # if we created it in this request — never delete a caller-supplied
        # session_id, even on a malformed Mode B option.
        if session_created_here:
            await manager.delete_session(session_id)
        raise HTTPException(status_code=400, detail=str(e)) from e
    # Mirror /pipeline: clamp render-step concurrency to the process-wide
    # WU_NVCF_GLOBAL_MAX_CONCURRENT_REQUESTS cap so a Mode B /predict job
    # cannot bypass the global render throttle that protects shared NVCF
    # endpoints.
    from .pipeline_router import _apply_render_request_limit

    _apply_render_request_limit(predict_config)
    # /predict never runs apply_physics; flip it off so Mode B's
    # only_steps filter doesn't accidentally include it.
    predict_config.setdefault("steps", {}).setdefault("apply_physics", {})[
        "enabled"
    ] = False

    config_path = session_dir / "input" / "predict_config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(predict_config, f, default_flow_style=False)

    job_registry = get_job_registry()

    # Atomically claim the slot in the in-process registry BEFORE writing
    # any session state. A losing concurrent rerun for the same terminal
    # session must NOT mutate session metadata/config: under the previous
    # ordering it could pass the up-front is_running()/persisted-status
    # check, write `status=pending` + a new config block, and only then
    # get rejected by JobRegistry — leaving the winning job running with
    # the loser's metadata. Reserving first inverts the order: the loser
    # raises ValueError here, propagates as 409, and never touches
    # session state. See registry.JobRegistry.reserve().
    try:
        reservation = await job_registry.reserve(session_id)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e

    async with reservation:
        # Stage the explicit dataset.jsonl into the session cache whenever
        # dataset_path is present. The copy happens AFTER the reservation
        # is claimed so a losing concurrent rerun (which 409s out at
        # reserve()) cannot overwrite the winner's
        # session_dir/cache/dataset/dataset.jsonl — that would otherwise
        # let `/artifacts/{id}/dataset` and on-demand report generation
        # describe a different dataset than the predictions. Inference
        # still reads the *original* dataset_path so the JSONL's
        # relative image entries continue to resolve next to the rendered
        # PNGs; only the JSONL itself is copied for the artifact contract.
        #
        # We deliberately stage *before* writing status="pending" so a
        # copyfile failure on a caller-supplied session (no
        # session_created_here teardown to fall back on) leaves the
        # session in its previous terminal status — retryable — instead
        # of permanently wedged in "pending" with no way out short of
        # TTL expiry. The reservation context manager still releases the
        # registry slot via __aexit__ when we raise.
        if resolved_dataset_path is not None:
            session_dataset_target = session_dir / "cache" / "dataset" / "dataset.jsonl"
            session_dataset_target.parent.mkdir(parents=True, exist_ok=True)
            # Skip the copy when the caller pointed dataset_path at this
            # session's own staged dataset.jsonl (e.g. a Mode A rerun on the
            # same session). shutil.copyfile would otherwise raise
            # SameFileError and 500 a perfectly valid retry.
            already_staged = (
                session_dataset_target.exists()
                and resolved_dataset_path.exists()
                and resolved_dataset_path.samefile(session_dataset_target)
            )
            if not already_staged:
                try:
                    shutil.copyfile(resolved_dataset_path, session_dataset_target)
                except Exception as e:  # noqa: BLE001
                    if session_created_here:
                        await manager.delete_session(session_id)
                    raise HTTPException(
                        status_code=500,
                        detail=f"Failed to stage dataset into session cache: {e}",
                    ) from e

        existing = await manager.get_session_metadata(session_id) or {}
        existing_config = existing.get("config") or {}
        # Reset any prior EventBus snapshot for this session before queuing a
        # rerun. /predict/{id}/status prefers the in-memory snapshot, and the
        # bus's COMPLETED-state handling never demotes back to "running" on a
        # new RUNNING event — without this clear, a rerun on a previously-
        # completed session would report "completed" the entire time the new
        # job is actually executing.
        get_event_bus().cleanup_session(session_id)
        await manager.update_session(
            session_id,
            {
                "status": "pending",
                "can_cancel": True,
                "config": {
                    **existing_config,
                    "project_name": predict_config.get("project", {}).get("name", ""),
                    "usd_path": str(input_usd_path) if input_usd_path else None,
                    "has_usd_upload": existing_config.get("has_usd_upload", False)
                    or has_usd_file,
                    "s3_uri": s3_uri or existing_config.get("s3_uri"),
                    "user_prompt": user_prompt_text,
                    "optimize_usd": optimize_usd,
                    "enable_deinstance": enable_deinstance,
                    "enable_split": enable_split,
                    "enable_deduplicate": enable_deduplicate,
                    "predict_route": True,
                },
            },
        )

        await reservation.start(
            execute_predict_async(
                session_id=session_id,
                config_dict=predict_config,
                session_manager=manager,
                dataset_path=resolved_dataset_path,
            ),
        )

    logger.info(f"/predict registered for session {session_id}")

    return SessionCreated(
        session_id=session_id,
        status="pending",
        message="Predict job queued for execution",
        estimated_duration_minutes=10,
    )


@router.get("/{session_id}/status", response_model=PipelineStatus)
async def get_predict_status(session_id: str) -> PipelineStatus:
    """Get predict execution status with detailed progress.

    Reuses the same response schema as /pipeline/{id}/status so existing
    progress UIs work unchanged. Reads from the in-memory event bus first,
    falls back to the SessionManager store for cross-instance visibility.
    """
    event_bus = get_event_bus()
    manager = get_session_manager()

    snapshot = event_bus.get_snapshot(session_id)
    if snapshot:
        metadata = snapshot
        preview_images = snapshot.get("preview_images", [])
    else:
        metadata = await manager.get_session_metadata(session_id)
        if not metadata:
            raise HTTPException(status_code=404, detail="Session not found")
        preview_images = metadata.get("preview_images", [])

    preview_urls = [f"/artifacts/{session_id}/preview/{img}" for img in preview_images]

    created_at = datetime.fromisoformat(metadata["created_at"])
    now = datetime.now(UTC)
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=UTC)
    elapsed_seconds = int((now - created_at).total_seconds())
    can_cancel = metadata.get("status") in ["pending", "running"]

    return PipelineStatus(
        session_id=session_id,
        status=metadata["status"],
        current_step=metadata.get("current_step"),
        completed_steps=metadata.get("completed_steps", []),
        overall_progress=metadata.get("overall_progress", {}),
        preview_images=preview_urls,
        can_cancel=can_cancel,
        elapsed_seconds=elapsed_seconds,
        created_at=metadata["created_at"],
        updated_at=metadata["updated_at"],
    )


@router.get("/{session_id}/results", response_model=PredictResults | PipelineError)
async def get_predict_results(session_id: str):
    """Get predict execution results (only available when completed)."""
    manager = get_session_manager()

    metadata = await manager.get_session_metadata(session_id)
    if not metadata:
        raise HTTPException(status_code=404, detail="Session not found")

    status = metadata["status"]

    if status == "completed":
        results = metadata.get("results") or {}
        # Build download URLs. We only advertise dataset when one actually
        # exists for this session — Mode A from an external dataset_path
        # leaves the session's dataset/ dir empty, so claiming the URL works
        # would be a lie.
        download_urls: dict[str, str] = {
            "predictions": f"/artifacts/{session_id}/predictions",
            "report": f"/artifacts/{session_id}/report",
        }
        session_dir = manager.get_session_dir(session_id)
        dataset_local = session_dir / "cache" / "dataset" / "dataset.jsonl"
        if not dataset_local.exists():
            # Cross-instance case: the worker pod synced the dataset to the
            # shared store but this pod doesn't have the local copy yet.
            # Pull just the dataset prefix before deciding whether to advertise
            # the URL, so /artifacts/{id}/dataset can serve it on this pod too.
            try:
                await manager.sync_from_store(session_id, prefix="cache/dataset/")
            except Exception as e:  # noqa: BLE001
                logger.debug(
                    f"sync_from_store(cache/dataset/) failed for {session_id[:8]}: {e}"
                )
        if dataset_local.exists():
            download_urls["dataset"] = f"/artifacts/{session_id}/dataset"

        # The worker normalizes PredictOutput.predictions_count into
        # results["predictions_made"], but if a future code path ever stores
        # a PredictOutput-shaped dict directly we still want the REST layer
        # to surface the right number — accept either key.
        predictions_count = int(
            results.get("predictions_made", results.get("predictions_count", 0))
        )

        return PredictResults(
            session_id=session_id,
            status=status,
            mode=metadata.get("predict_mode", "unknown"),
            steps_run=metadata.get("predict_steps_run", []),
            stats=results,
            predictions_count=predictions_count,
            failed_count=int(results.get("failed_count", 0)),
            predictions_path=results.get("predictions_path"),
            token_stats=results.get("token_stats", {}) or {},
            download_urls=download_urls,
            duration_seconds=metadata.get("duration_seconds", 0),
            completed_at=metadata.get("completed_at", ""),
        )

    if status == "failed":
        return PipelineError(
            session_id=session_id,
            status=status,
            error_message=metadata.get("error", "Unknown error"),
            failed_step=metadata.get("failed_step", "predict"),
            completed_steps=[s["name"] for s in metadata.get("completed_steps", [])],
            partial_results=metadata.get("partial_results"),
        )

    raise HTTPException(
        status_code=202,
        detail=f"Predict still {status}. Check status endpoint for progress.",
    )


@router.post("/{session_id}/cancel")
async def cancel_predict(session_id: str) -> dict[str, str]:
    """Cancel a running predict job."""
    job_registry = get_job_registry()
    manager = get_session_manager()

    metadata = await manager.get_session_metadata(session_id)
    if not metadata:
        raise HTTPException(status_code=404, detail="Session not found")

    # Refuse to cancel a session that wasn't started via /predict. Without
    # this guard, /predict/{id}/cancel would happily cancel a /pipeline
    # session and respond "Predict cancellation requested" — confusing and
    # incorrect. The predict route stamps `predict_route: True` into
    # session metadata.config when it queues; we use that as the
    # discriminator. Sessions without a config block fall through to the
    # standard cancel semantics (typically just-created predict sessions
    # caught between create and the metadata stamp).
    session_config = metadata.get("config") or {}
    if session_config and not session_config.get("predict_route"):
        raise HTTPException(
            status_code=409,
            detail=(
                f"Session {session_id} is not a predict session "
                "(it was created via /pipeline or /pipeline/upload-usd). Use "
                "POST /pipeline/{session_id}/cancel instead."
            ),
        )

    if metadata["status"] not in ["pending", "running"]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel predict job with status: {metadata['status']}",
        )

    await manager.request_cancellation(session_id)

    if job_registry.is_running(session_id):
        cancelled = await job_registry.cancel(session_id)
        if cancelled:
            # If the task was still queued (waiting on the JobRegistry
            # semaphore), CancelledError fires before execute_predict_async
            # ever enters its CancelledError handler, so the session would
            # otherwise be stuck on "cancelling" forever. Drive the metadata
            # to a terminal "cancelled" ourselves when the persisted state
            # is still mid-cancel.
            post_cancel = await manager.get_session_metadata(session_id)
            if post_cancel and post_cancel.get("status") in (
                "cancelling",
                "pending",
                "running",
            ):
                await manager.update_session(
                    session_id,
                    {
                        "status": "cancelled",
                        "cancelled_at": datetime.now(UTC).isoformat(),
                        "can_cancel": False,
                    },
                )

    return {
        "session_id": session_id,
        "status": "cancelling",
        "message": "Predict cancellation requested",
    }


@router.get("/{session_id}/events")
async def stream_predict_events(session_id: str) -> EventSourceResponse:
    """Stream real-time predict progress events via Server-Sent Events (SSE).

    Mirrors /pipeline/{id}/events semantics: only works when connected to the
    instance executing the job; for cross-instance progress, poll
    GET /predict/{id}/status instead.
    """
    event_bus = get_event_bus()
    manager = get_session_manager()

    snapshot = event_bus.get_snapshot(session_id)
    if snapshot is None:
        if not await manager.session_exists(session_id):
            raise HTTPException(status_code=404, detail="Session not found")

    terminal_states = ("completed", "failed", "cancelled")

    if snapshot is None:
        metadata = await manager.get_session_metadata(session_id)
        final_state = (metadata or {}).get("status", "unknown")
        if final_state not in terminal_states:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Predict is running on a different instance; use polling instead"
                ),
            )

    async def event_generator():
        queue = event_bus.get_queue(session_id)

        if snapshot is not None and snapshot.get("status") in terminal_states:
            final_state = snapshot["status"]
            yield {
                "event": "done",
                "data": (
                    f'{{"session_id": "{session_id}", "final_state": "{final_state}"}}'
                ),
            }
            return

        if snapshot is None:
            metadata = await manager.get_session_metadata(session_id)
            if metadata and metadata.get("status") in terminal_states:
                final_state = metadata["status"]
                yield {
                    "event": "done",
                    "data": (
                        f'{{"session_id": "{session_id}", '
                        f'"final_state": "{final_state}"}}'
                    ),
                }
                return

        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30.0)
                    event_data = event.model_dump_json()
                    yield {"event": "progress", "data": event_data}

                    should_close = False
                    if event.state in ["failed", "cancelled"]:
                        should_close = True
                    elif event.extra and event.extra.get("pipeline_ready"):
                        should_close = True

                    if should_close:
                        yield {
                            "event": "done",
                            "data": (
                                f'{{"session_id": "{session_id}", '
                                f'"final_state": "{event.state}"}}'
                            ),
                        }
                        break

                except TimeoutError:
                    metadata = await manager.get_session_metadata(session_id)
                    if metadata and metadata.get("status") in terminal_states:
                        final_state = metadata["status"]
                        yield {
                            "event": "done",
                            "data": (
                                f'{{"session_id": "{session_id}", '
                                f'"final_state": "{final_state}"}}'
                            ),
                        }
                        break
                    yield {"event": "ping", "data": "keepalive"}

        except asyncio.CancelledError:
            logger.debug(f"/predict SSE stream cancelled for {session_id[:8]}...")
            raise

    return EventSourceResponse(event_generator(), ping=15)
