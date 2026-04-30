# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task: Generate PBR texture sets from prompts.

Supports two backend types (configured via texture_config.backend_type):
- "simple_image_gen" (default): Uses ImageGenEngine (Gemini image gen) in-process
- "service": Calls a remote Texture Variation API service via REST endpoint

Iterates over PrimTextureUnit list (from DiscoverMaterialsTask), which
handles both per-material and per-prim modes transparently.
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.error import HTTPError

from world_understanding.agentic.tasks import Task

from texture_agent.functions.material_discovery import PrimTextureUnit
from texture_agent.functions.texture_generation import (
    Conditioning,
    GeneratedTextures,
    ImageGenEngine,
    TextureVariationClient,
    TextureVariationConfig,
)
from texture_agent.tasks.thresholds import validate_failure_threshold

logger = logging.getLogger(__name__)

_HTTP_STATUS_RE = re.compile(r"HTTP\s*(?:Error\s*)?(\d{3})", re.IGNORECASE)


def _classify_unit_failure(unit_key: str, exc: BaseException) -> dict[str, Any]:
    """Build a structured per-unit error record for SSE/status surfacing.

    Best-effort HTTP status extraction, in order of preference:
      1. ``httpx.HTTPStatusError.response.status_code`` -- raised by the
         service backend's ``RestTextureVariationClient`` polling path.
         The default ``httpx`` message format ("Client error '403 Forbidden'
         for url ...") does NOT contain a literal ``HTTP <NNN>`` substring,
         so the regex fallback below would miss it.
      2. ``urllib.error.HTTPError.code`` -- raised by the simple-image-gen
         backend through stdlib urllib.
      3. ``HTTP <NNN>`` regex scrape of the message -- catches strings
         raised by ``image_generation_models.py`` and the per-unit
         ``RuntimeError`` wrappers in this module.
    """
    try:
        import httpx as _httpx
    except ImportError:  # pragma: no cover -- httpx is a hard dep here.
        _httpx = None  # type: ignore[assignment]

    cause: BaseException | None = exc
    while cause is not None:
        if _httpx is not None and isinstance(cause, _httpx.HTTPStatusError):
            return {
                "material": unit_key,
                "type": "HTTPStatusError",
                "status": cause.response.status_code,
                "message": str(exc),
            }
        if isinstance(cause, HTTPError):
            return {
                "material": unit_key,
                "type": "HTTPError",
                "status": cause.code,
                "message": str(exc),
            }
        cause = cause.__cause__ or cause.__context__

    message = str(exc)
    status: int | None = None
    match = _HTTP_STATUS_RE.search(message)
    if match:
        status = int(match.group(1))

    return {
        "material": unit_key,
        "type": type(exc).__name__,
        "status": status,
        "message": message,
    }


def _validate_textures_or_raise(unit_key: str, textures: GeneratedTextures) -> None:
    """Reject completed-but-unusable texture results.

    A `JobStatus(status="completed")` only tells us the upstream call returned
    without an error code. The texture set itself can still be unusable -- a
    schema-skewed service may parse to ``GeneratedTextures(albedo="", ...)``,
    or ``_localize_textures`` may have failed to download a remote file and
    left a non-local URI in place. Such results must not silently flow into
    blend/apply, which would skip them and exit 0.

    Downstream ``BlendTexturesTask`` calls ``Path(...)`` on raw strings;
    ``Path`` does not parse URI schemes. So ANY URI -- including ``file://`` --
    would silently skip downstream even though the underlying bytes might be
    reachable. The texture-agent's own ``_localize_textures`` already strips
    ``file://`` from accessible service URIs and writes bare local paths, so by
    the time this validator runs the only forms a correctly-behaving caller
    passes in are bare local paths. Anything else (any ``://``) is treated as a
    per-unit failure here -- failing loud beats silently re-creating the very
    bug this task is meant to fix. Relax this when ``BlendTexturesTask`` learns
    to resolve URIs.
    """
    for texture_name, texture_path in (
        ("albedo", textures.albedo),
        ("normal", textures.normal),
        ("orm", textures.orm),
    ):
        if not texture_path:
            raise RuntimeError(
                f"Generation reported success for {unit_key} but produced "
                f"no {texture_name} path (got empty string)"
            )
        if "://" in texture_path:
            raise RuntimeError(
                f"Generation reported success for {unit_key} but produced an "
                f"unsupported {texture_name} URI: {texture_path!r}. The "
                f"texture-agent pipeline currently only consumes local file "
                f"paths (BlendTexturesTask uses Path(...) which does not "
                f"parse URIs); the service backend's _localize_textures "
                f"strips file:// from accessible files before reaching here."
            )
        if not Path(texture_path).exists():
            raise RuntimeError(
                f"Generation reported success for {unit_key} but the "
                f"{texture_name} path does not exist on disk: {texture_path!r}"
            )


def _raise_if_above_threshold(
    attempted: list[PrimTextureUnit],
    fresh_generated: dict[str, GeneratedTextures],
    errors: list[dict[str, Any]],
    *,
    backend_label: str,
    failure_threshold: float,
) -> None:
    """Raise when the per-unit failure rate hits ``failure_threshold``.

    ``attempted`` is the slice of units actually submitted this run (after
    skip-existing filtering); ``fresh_generated`` is the result map for
    THIS run only -- cached entries from a previous run are deliberately
    excluded. A resumed run where every fresh request failed (e.g. expired
    NIM key returning HTTP 403 on every unit) raises even if cache from an
    earlier successful run partially populates the merged output. The
    customer's environment is broken; don't paper over it with stale cache.

    ``failure_threshold`` is a fraction in [0.0, 1.0]:
      - 1.0 (default): raise only when 100% of fresh attempts failed
        (preserves the original "all must fail" gate).
      - 0.5: raise when at least half of fresh attempts failed.
      - 0.0: raise on any failure.

    Sub-threshold failures are logged as a warning and allowed to continue;
    downstream steps can still apply whatever textures did succeed. Per-unit
    error records are surfaced separately via ``context`` regardless.
    """
    if not attempted:
        return
    if not errors:
        return

    failure_rate = len(errors) / len(attempted)
    if failure_rate >= failure_threshold:
        sample_lines = [
            f"{e['material']}: [{e['type']}"
            + (f" {e['status']}" if e.get("status") is not None else "")
            + f"] {e['message']}"
            for e in errors[:3]
        ]
        sample = "\n  - ".join(sample_lines)
        more = f"\n  ... ({len(errors) - 3} more)" if len(errors) > 3 else ""
        threshold_pct = int(failure_threshold * 100)
        raise RuntimeError(
            f"{len(errors)}/{len(attempted)} texture generation requests "
            f"failed via {backend_label} "
            f"(failure rate {failure_rate:.0%} >= threshold {threshold_pct}%). "
            f"First errors:\n  - {sample}{more}"
        )
    logger.warning(
        "Texture generation completed with %d/%d failures via %s "
        "(below threshold %.0f%%)",
        len(errors),
        len(attempted),
        backend_label,
        failure_threshold * 100,
    )


def _cached_texture_set(out_dir: Path, key: str) -> GeneratedTextures | None:
    """Return a valid cached texture set from flat or per-variant layout.

    Candidates are tested in order, using ``albedo.exists()`` as a quick
    pre-filter before validating the full PBR set with
    ``_validate_textures_or_raise``. Partial or stale candidates, such as
    albedo-only outputs from failed generations, log a warning and fall through
    to the next layout. Returns ``None`` when no candidate passes validation.
    """
    candidates = [
        (
            out_dir / f"{key}_albedo.png",
            out_dir / f"{key}_normal.png",
            out_dir / f"{key}_orm.png",
        ),
        (
            out_dir / key / f"{key}_albedo.png",
            out_dir / key / f"{key}_normal.png",
            out_dir / key / f"{key}_orm.png",
        ),
    ]
    for albedo, normal, orm in candidates:
        if albedo.exists():
            textures = GeneratedTextures(
                albedo=str(albedo),
                normal=str(normal),
                orm=str(orm),
            )
            try:
                _validate_textures_or_raise(key, textures)
            except RuntimeError as exc:
                logger.warning("Skipping invalid cached textures for %s: %s", key, exc)
                continue
            return textures
    return None


class GenerateTexturesTask(Task):
    """Generate PBR texture sets (albedo, normal, ORM) from text prompts.

    Iterates over prim_texture_units (from DiscoverMaterialsTask).

    Backend types:
        simple_image_gen: Local image generation (Gemini via NVIDIA Inference).
            No external service needed. Generates albedo + normal + roughness
            using AI image gen model with tailored prompts.
        service: Remote Texture Variation API service (e.g., Step1X-3D).
            Calls POST /v1/texture-variations on the configured endpoint URL.
            The service handles texture extraction, generation, and write-back.

    Context keys read:
        prim_texture_units (list[PrimTextureUnit]): From DiscoverMaterialsTask.
        texture_config (dict): Configuration including:
            backend_type: "simple_image_gen" or "service"
            endpoint: REST endpoint URL (required for "service")
            backend: Image gen backend name (for "simple_image_gen")
            model: Model override
            size: Texture resolution
            workers: Number of parallel workers (default 4)
            skip_existing: Skip if texture already exists (default True)
        working_dir (str): Working directory.
        usd_path (str): Input USD path.

    Context keys written:
        generated_textures (dict[str, GeneratedTextures]):
            Unit key -> GeneratedTextures (albedo, normal, orm paths).
    """

    def __init__(self) -> None:
        self.name = "GenerateTextures"
        self.description = "Generate PBR texture sets from prompts"

    def _run_simple_image_gen(
        self,
        units: list[PrimTextureUnit],
        context: dict[str, Any],
        out_dir: Path,
        texture_config: dict,
    ) -> tuple[dict[str, GeneratedTextures], list[dict[str, Any]], str]:
        """Generate textures using local ImageGenEngine.

        Returns ``(generated, errors, backend_label)``. Each ``errors`` entry
        is a structured dict (``{material, type, status, message}``) so the
        service layer can surface per-unit failures in SSE/status payloads
        instead of forcing customers to grep container logs.
        """
        image_gen_config = texture_config.get("image_gen", {})
        backend = image_gen_config.get("backend", "nim")
        model = image_gen_config.get("model")
        base_url = image_gen_config.get("base_url")
        workers = texture_config.get("workers", 4)

        engine = ImageGenEngine(backend=backend, model=model, base_url=base_url)
        engine._ensure_model()
        client = TextureVariationClient(engine=engine, output_dir=out_dir)

        logger.info(
            "Generating %d PBR texture sets with %s (simple_image_gen, workers=%d)",
            len(units),
            engine.name,
            workers,
        )

        generated: dict[str, GeneratedTextures] = {}
        errors: list[dict[str, Any]] = []

        def _gen(unit: PrimTextureUnit) -> tuple[str, GeneratedTextures]:
            status = client.generate(
                source_asset_uri=context.get("usd_path", ""),
                conditioning=Conditioning(text_prompt=unit.prompt),
                config=TextureVariationConfig(
                    strength=unit.opacity,
                    variant_name=unit.key,
                    seed=unit.seed,
                ),
            )
            if status.status != "completed" or not status.result:
                raise RuntimeError(
                    f"Generation failed for {unit.key}: {status.error_message}"
                )
            textures = status.result.generated_textures
            _validate_textures_or_raise(unit.key, textures)
            return unit.key, textures

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_gen, unit): unit.key for unit in units}
            for future in as_completed(futures):
                key = futures[future]
                try:
                    k, textures = future.result()
                    generated[k] = textures
                except Exception as exc:
                    logger.exception("Failed to generate textures for %s", key)
                    errors.append(_classify_unit_failure(key, exc))

        return generated, errors, engine.name

    def _run_service(
        self,
        units: list[PrimTextureUnit],
        context: dict[str, Any],
        out_dir: Path,
        texture_config: dict,
    ) -> tuple[dict[str, GeneratedTextures], list[dict[str, Any]], str]:
        """Generate textures using a remote REST service.

        Returns ``(generated, errors, backend_label)``. Each ``errors`` entry
        is a structured dict (``{material, type, status, message}``) so the
        service layer can surface per-unit failures in SSE/status payloads
        instead of forcing customers to grep container logs.
        """
        from texture_agent.functions.rest_client import RestTextureVariationClient

        endpoint = texture_config.get("endpoint")
        if not endpoint:
            raise ValueError(
                "texture_config.endpoint is required for backend_type='service'"
            )
        workers = texture_config.get("workers", 4)

        client = RestTextureVariationClient(endpoint, timeout=1200)

        logger.info(
            "Generating %d PBR texture sets via service (%s, workers=%d)",
            len(units),
            endpoint,
            workers,
        )

        generated: dict[str, GeneratedTextures] = {}
        errors: list[dict[str, Any]] = []

        def _gen_one(unit: PrimTextureUnit) -> tuple[str, GeneratedTextures]:
            status = client.generate(
                source_asset_uri=context.get("usd_path", ""),
                conditioning=Conditioning(text_prompt=unit.prompt),
                config=TextureVariationConfig(
                    strength=unit.opacity,
                    variant_name=unit.key,
                    seed=unit.seed,
                    custom_parameters=texture_config.get("custom_parameters", {}),
                ),
                wait=True,
                timeout_sec=600,
            )

            if status.status != "completed" or not status.result:
                raise RuntimeError(
                    f"Service failed for {unit.key}: "
                    f"{status.error_message[:200] if status.error_message else 'unknown'}"
                )

            textures = status.result.generated_textures

            # Download results if they are file:// URIs on a remote host
            local_textures = self._localize_textures(
                textures, unit.key, out_dir, endpoint
            )
            _validate_textures_or_raise(unit.key, local_textures)
            return unit.key, local_textures

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_gen_one, unit): unit.key for unit in units}
            for future in as_completed(futures):
                key = futures[future]
                try:
                    k, textures = future.result()
                    generated[k] = textures
                    logger.info("[%s] Complete", k)
                except Exception as exc:
                    logger.exception("Failed to generate textures for %s", key)
                    errors.append(_classify_unit_failure(key, exc))

        return generated, errors, f"service ({endpoint})"

    @staticmethod
    def _localize_textures(
        textures: GeneratedTextures,
        key: str,
        out_dir: Path,
        endpoint: str,
    ) -> GeneratedTextures:
        """Ensure texture paths are local files.

        If the service returns file:// URIs (local to the service host),
        download them via HTTP. If the service returns http:// URLs or
        local paths, use them directly.
        """

        def _download_if_needed(uri: str, suffix: str) -> str:
            if not uri:
                return ""
            # Already a local path
            if not uri.startswith("file://") and Path(uri).exists():
                return uri
            # For file:// URIs, try to download via a /files endpoint
            # or just use the path if it's accessible
            remote_path = uri.replace("file://", "").replace("file:", "")
            local_path = out_dir / f"{key}_{suffix}.png"
            if Path(remote_path).exists():
                import shutil

                shutil.copy2(remote_path, str(local_path))
                return str(local_path)
            # Path not accessible locally — leave as-is
            return uri

        return GeneratedTextures(
            albedo=_download_if_needed(textures.albedo, "albedo"),
            normal=_download_if_needed(textures.normal, "normal"),
            orm=_download_if_needed(textures.orm, "orm"),
        )

    def run(self, context: dict[str, Any], object_store: Any = None) -> dict[str, Any]:
        units: list[PrimTextureUnit] = context.get("prim_texture_units", [])
        if not units:
            logger.warning(
                "No prim_texture_units in context — was generate_prompts step skipped?"
            )
            context.setdefault("generated_textures", {})
            return context

        texture_config: dict = context.get("texture_config", {})
        working_dir = Path(context["working_dir"])
        skip_existing = bool(context.get("resume")) or texture_config.get(
            "skip_existing", True
        )

        # Validate the threshold BEFORE any backend dispatch so a typo
        # (``failure_threshold: "nan"`` / ``1.1``) fails fast instead of
        # racking up 8x network round-trips and only THEN raising a config
        # error.
        failure_threshold = validate_failure_threshold(
            texture_config.get("failure_threshold", 1.0),
            config_key="texture_config.failure_threshold",
        )

        out_dir = working_dir / "generated"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Filter to units that need generation
        to_generate: list[PrimTextureUnit] = []
        generated: dict[str, GeneratedTextures] = {}

        for unit in units:
            cached_textures = (
                _cached_texture_set(out_dir, unit.key) if skip_existing else None
            )
            if cached_textures:
                logger.info("Skipping %s (already generated)", unit.key)
                generated[unit.key] = cached_textures
                continue
            to_generate.append(unit)

        if not to_generate:
            logger.info("No textures to generate")
            context["generated_textures"] = generated
            return context

        # Choose backend
        backend = texture_config.get("backend", "simple_image_gen")

        if backend == "service":
            new_generated, errors, backend_label = self._run_service(
                to_generate, context, out_dir, texture_config
            )
        elif backend == "simple_image_gen":
            new_generated, errors, backend_label = self._run_simple_image_gen(
                to_generate, context, out_dir, texture_config
            )
        else:
            raise ValueError(
                f"Unknown texture backend: {backend}. "
                "Use 'simple_image_gen' or 'service'."
            )

        # Merge fresh successes onto cached hits and publish to context
        # BEFORE the threshold raise. The executor's per-step except block
        # extracts step stats from context; without this write a partial-
        # failure-above-threshold raise would report ``textures_generated:
        # 0`` even when some materials succeeded and were written to disk.
        generated.update(new_generated)
        context["generated_textures"] = generated
        context["generate_textures_errors"] = errors
        context["generate_textures_failed_count"] = len(errors)
        context["generate_textures_attempted_count"] = len(to_generate)

        # Threshold decision uses FRESH attempts only -- cached entries
        # from prior runs must not mask a totally-broken backend (e.g.
        # expired NIM key returning HTTP 403 on every fresh request).
        _raise_if_above_threshold(
            to_generate,
            new_generated,
            errors,
            backend_label=backend_label,
            failure_threshold=failure_threshold,
        )
        logger.info("Generated %d PBR texture sets", len(generated))
        return context
