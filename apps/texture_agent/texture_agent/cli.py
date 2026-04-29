# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Texture Agent CLI -- generate and apply textures to USD materials."""

from __future__ import annotations

import logging
from pathlib import Path

import typer
from dotenv import load_dotenv

from texture_agent.utils import get_version

load_dotenv()

app = typer.Typer(
    name="texture-agent",
    help="Generate and apply textures to USD materials.",
    no_args_is_help=True,
)


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"texture-agent {get_version()}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    """Texture Agent: generate and apply textures to USD materials."""
    pass


@app.command()
def run(
    config: Path = typer.Argument(..., help="Path to the pipeline config YAML"),
    skip: str | None = typer.Option(
        None, "--skip", help="Comma-separated step names to skip"
    ),
    only: str | None = typer.Option(
        None, "--only", help="Comma-separated step names to run exclusively"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show execution plan without running"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
) -> None:
    """Run the full texture pipeline."""
    _setup_logging(verbose)
    logger = logging.getLogger(__name__)

    from texture_agent.config.unified_config import config_to_context, load_config
    from texture_agent.workflows.factory import run_pipeline

    try:
        cfg = load_config(config)
        context = config_to_context(cfg)

        skip_list = skip.split(",") if skip else None
        only_list = only.split(",") if only else None

        context = run_pipeline(context, skip=skip_list, only=only_list, dry_run=dry_run)

        if not dry_run:
            # Print summary
            output_paths = context.get("output_usd_paths", [])
            render_paths = context.get("rendered_image_paths", [])
            typer.echo("\nPipeline complete!")
            if output_paths:
                typer.echo("Output USD files:")
                for p in output_paths:
                    typer.echo(f"  {p}")
            if render_paths:
                typer.echo("Rendered images:")
                for p in render_paths:
                    typer.echo(f"  {p}")

    except Exception as e:
        logger.error("Pipeline failed: %s", e)
        if verbose:
            logger.exception("Full traceback:")
        raise typer.Exit(1) from e


@app.command()
def discover(
    config: Path = typer.Argument(..., help="Path to the pipeline config YAML"),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
) -> None:
    """Discover and list materials in the input USD."""
    _setup_logging(verbose)
    logger = logging.getLogger(__name__)

    from texture_agent.config.unified_config import config_to_context, load_config
    from texture_agent.tasks import DiscoverMaterialsTask

    try:
        cfg = load_config(config)
        context = config_to_context(cfg)

        task = DiscoverMaterialsTask()
        context = task.run(context)

        materials = context.get("discovered_materials", [])
        typer.echo(f"\nDiscovered {len(materials)} materials:\n")
        typer.echo(f"{'Name':<30} {'Base Color':<25} {'Texture':<8} {'Prims':<6}")
        typer.echo("-" * 69)
        for m in materials:
            color_str = (
                f"({m.base_color[0]:.2f}, {m.base_color[1]:.2f}, {m.base_color[2]:.2f})"
            )
            typer.echo(
                f"{m.name:<30} {color_str:<25} "
                f"{'yes' if m.has_existing_texture else 'no':<8} "
                f"{len(m.bound_prim_paths):<6}"
            )

    except Exception as e:
        logger.error("Discover failed: %s", e)
        if verbose:
            logger.exception("Full traceback:")
        raise typer.Exit(1) from e


@app.command()
def generate(
    config: Path = typer.Argument(..., help="Path to the pipeline config YAML"),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
) -> None:
    """Generate and blend textures (without applying to USD)."""
    _setup_logging(verbose)
    logger = logging.getLogger(__name__)

    from texture_agent.config.unified_config import config_to_context, load_config
    from texture_agent.workflows.factory import run_pipeline

    try:
        cfg = load_config(config)
        context = config_to_context(cfg)

        context = run_pipeline(
            context,
            only=[
                "discover_materials",
                "generate_prompts",
                "generate_textures",
                "blend_textures",
            ],
        )

        blended = context.get("blended_textures", {})
        typer.echo(f"\nGenerated and blended {len(blended)} textures:")
        for name, path in blended.items():
            typer.echo(f"  {name}: {path}")

    except Exception as e:
        logger.error("Generate failed: %s", e)
        if verbose:
            logger.exception("Full traceback:")
        raise typer.Exit(1) from e


@app.command("apply")
def apply_cmd(
    config: Path = typer.Argument(..., help="Path to the pipeline config YAML"),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
) -> None:
    """Apply textures to USD (assumes textures already generated)."""
    _setup_logging(verbose)
    logger = logging.getLogger(__name__)

    from texture_agent.config.unified_config import config_to_context, load_config
    from texture_agent.workflows.factory import run_pipeline

    try:
        cfg = load_config(config)
        context = config_to_context(cfg)

        context = run_pipeline(
            context,
            only=["discover_materials", "generate_prompts", "apply_textures"],
        )

        output_paths = context.get("output_usd_paths", [])
        typer.echo(f"\nApplied textures to {len(output_paths)} USD file(s):")
        for p in output_paths:
            typer.echo(f"  {p}")

    except Exception as e:
        logger.error("Apply failed: %s", e)
        if verbose:
            logger.exception("Full traceback:")
        raise typer.Exit(1) from e
