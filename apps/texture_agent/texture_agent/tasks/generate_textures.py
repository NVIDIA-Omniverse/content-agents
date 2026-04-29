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
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from world_understanding.agentic.tasks import Task

from texture_agent.functions.material_discovery import PrimTextureUnit
from texture_agent.functions.texture_generation import (
    Conditioning,
    GeneratedTextures,
    ImageGenEngine,
    TextureVariationClient,
    TextureVariationConfig,
)

logger = logging.getLogger(__name__)


def _validate_textures_or_raise(unit_key: str, textures: GeneratedTextures) -> None:
    """Reject completed-but-unusable texture results.

    A `JobStatus(status="completed")` only tells us the upstream call returned
    without an error code. The texture set itself can still be unusable -- a
    schema-skewed service may parse to ``GeneratedTextures(albedo="", ...)``,
    or ``_localize_textures`` may have failed to download a remote file and
    left a non-local URI in place. Such results must not silently flow into
    blend/apply, which would skip them and exit 0.

    Downstream ``BlendTexturesTask`` calls ``Path(albedo).exists()`` on the
    raw string; ``Path`` does not parse URI schemes. So ANY URI -- including
    ``file://`` -- would silently skip downstream even though the underlying
    bytes might be reachable. The texture-agent's own ``_localize_textures``
    already strips ``file://`` from accessible service URIs and writes a
    bare local path, so by the time this validator runs the only forms a
    correctly-behaving caller passes in are bare local paths. Anything else
    (any ``://``) is treated as a per-unit failure here -- failing loud
    beats silently re-creating the very bug this task is meant to fix.
    Relax this when ``BlendTexturesTask`` learns to resolve URIs.
    """
    albedo = textures.albedo
    if not albedo:
        raise RuntimeError(
            f"Generation reported success for {unit_key} but produced "
            f"no albedo path (got empty string)"
        )
    if "://" in albedo:
        raise RuntimeError(
            f"Generation reported success for {unit_key} but produced an "
            f"unsupported albedo URI: {albedo!r}. The texture-agent "
            f"pipeline currently only consumes local file paths "
            f"(BlendTexturesTask uses Path(albedo).exists() which does "
            f"not parse URIs); the service backend's _localize_textures "
            f"strips file:// from accessible files before reaching here."
        )
    if not Path(albedo).exists():
        raise RuntimeError(
            f"Generation reported success for {unit_key} but the albedo "
            f"path does not exist on disk: {albedo!r}"
        )


def _raise_if_all_failed(
    attempted: list[PrimTextureUnit],
    fresh_generated: dict[str, GeneratedTextures],
    errors: list[str],
    *,
    backend_label: str,
) -> None:
    """Raise when every fresh attempt failed so backend health surfaces.

    ``attempted`` is the slice of units actually submitted this run (after
    skip-existing filtering); ``fresh_generated`` is the result map for
    THIS run only -- cached entries from a previous run are deliberately
    excluded. So a resumed run where every fresh request failed (e.g.
    expired NIM key returning HTTP 403 on every unit) raises even if
    cache from an earlier successful run partially populates the merged
    output. The customer's environment is broken; don't paper over it
    with stale cache.

    Partial failures (any fresh success) are logged as a warning but
    allowed to continue, since downstream steps can still apply whatever
    textures did succeed.
    """
    if not attempted:
        return
    if not fresh_generated and errors:
        sample = "\n  - ".join(errors[:3])
        more = f"\n  ... ({len(errors) - 3} more)" if len(errors) > 3 else ""
        raise RuntimeError(
            f"All {len(attempted)} texture generation requests failed via "
            f"{backend_label}. First errors:\n  - {sample}{more}"
        )
    if errors:
        logger.warning(
            "Texture generation completed with %d/%d failures via %s",
            len(errors),
            len(attempted),
            backend_label,
        )


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
    ) -> tuple[dict[str, GeneratedTextures], list[str], str]:
        """Generate textures using local ImageGenEngine.

        Returns ``(generated, errors, backend_label)``. ``run`` aggregates
        ``errors`` against the merged-with-cache result before deciding
        whether to fail the step.
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
        errors: list[str] = []

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
                    errors.append(f"{key}: {exc}")

        return generated, errors, engine.name

    def _run_service(
        self,
        units: list[PrimTextureUnit],
        context: dict[str, Any],
        out_dir: Path,
        texture_config: dict,
    ) -> tuple[dict[str, GeneratedTextures], list[str], str]:
        """Generate textures using a remote REST service.

        Returns ``(generated, errors, backend_label)``. ``run`` aggregates
        ``errors`` against the merged-with-cache result before deciding
        whether to fail the step.
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
        errors: list[str] = []

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
                    errors.append(f"{key}: {exc}")

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
        skip_existing = texture_config.get("skip_existing", True)

        out_dir = working_dir / "generated"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Filter to units that need generation
        to_generate: list[PrimTextureUnit] = []
        generated: dict[str, GeneratedTextures] = {}

        for unit in units:
            albedo_path = out_dir / f"{unit.key}_albedo.png"
            if skip_existing and albedo_path.exists():
                logger.info("Skipping %s (already generated)", unit.key)
                normal = out_dir / f"{unit.key}_normal.png"
                orm = out_dir / f"{unit.key}_orm.png"
                generated[unit.key] = GeneratedTextures(
                    albedo=str(albedo_path),
                    normal=str(normal) if normal.exists() else "",
                    orm=str(orm) if orm.exists() else "",
                )
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

        # Decision is made against FRESH attempts only -- cached entries
        # from prior runs must not mask a totally-broken backend (e.g.
        # expired NIM key returning HTTP 403 on every fresh request).
        _raise_if_all_failed(
            to_generate, new_generated, errors, backend_label=backend_label
        )
        generated.update(new_generated)
        context["generated_textures"] = generated
        logger.info("Generated %d PBR texture sets", len(generated))
        return context
