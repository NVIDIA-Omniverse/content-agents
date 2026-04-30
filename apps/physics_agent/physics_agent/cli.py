# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Physics Agent CLI interface using Typer and Rich."""

import logging
from pathlib import Path
from typing import Annotated

import typer
import yaml
from dotenv import load_dotenv
from rich import print
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from world_understanding.agentic.events import get_listener

from .utils import get_version

__version__ = get_version()

# Load environment variables from .env file. The package __init__ also loads this
# before submodule imports for console-script entry points.
load_dotenv()

# Initialize Typer app and Rich console
app = typer.Typer(
    name="physics-agent",
    help="Physics Agent - VLM-based physics property classification for 3D assets",
    add_completion=False,
    rich_markup_mode="rich",
)
console = Console()


def setup_logging(
    verbose: bool = False,
    log_file: Path | None = None,
    log_level: str = "INFO",
) -> logging.Logger:
    """Setup logging configuration with Rich handler.

    This function now delegates to the shared logging utility.

    Args:
        verbose: Enable verbose output (sets DEBUG level)
        log_file: Optional path to log file
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        Configured logger instance
    """
    from world_understanding.agentic.cli import setup_logging as shared_setup_logging

    return shared_setup_logging(
        agent_name="physics_agent",
        verbose=verbose,
        log_file=log_file,
        log_level=log_level,
    )


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        print(
            f"[bold blue]Physics Agent[/bold blue] version [green]{__version__}[/green]"
        )
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            "-V",
            help="Show version and exit",
            callback=version_callback,
            is_eager=True,
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose output (DEBUG logging)"),
    ] = False,
    log_file: Annotated[
        Path | None,
        typer.Option("--log-file", help="Path to log file"),
    ] = None,
    log_level: Annotated[
        str,
        typer.Option(
            "--log-level",
            help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
        ),
    ] = "INFO",
) -> None:
    """
    Physics Agent - VLM-based physics property classification for 3D assets.

    Use [bold]physics-agent --help[/bold] to see available commands.
    """
    # Setup logging
    logger = setup_logging(verbose=verbose, log_file=log_file, log_level=log_level)

    # Store logger in app context for use in commands
    if not hasattr(app, "state"):
        app.state = {}
    app.state["logger"] = logger
    app.state["verbose"] = verbose

    if verbose:
        logger.debug("Verbose mode enabled")
        logger.debug(f"Log level: {log_level}")
        if log_file:
            logger.debug(f"Logging to file: {log_file}")


@app.command()
def predict(
    config: Annotated[
        Path,
        typer.Argument(
            help="Path to unified YAML configuration file",
        ),
    ],
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose output (DEBUG logging)"),
    ] = False,
    log_file: Annotated[
        Path | None,
        typer.Option("--log-file", help="Path to log file"),
    ] = None,
    log_level: Annotated[
        str,
        typer.Option(
            "--log-level",
            help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
        ),
    ] = "INFO",
) -> None:
    """
    Run asset classification predictions on a dataset.

    This is equivalent to: physics-agent run CONFIG --only predict

    Uses the unified configuration format where all paths are auto-derived from
    project.working_dir. The predict step will run VLM inference to classify assets.

    Example usage:
    ```bash
    physics-agent predict configs/unified_asset.yaml
    ```

    Output:
    - {working_dir}/predictions/predictions.jsonl: Classification predictions with reasoning
    - {working_dir}/predictions/report.html: HTML report with visualizations
    """
    # This is just an alias for: run --only predict
    return run(
        config=config,
        skip=None,
        only="predict",
        session_id=None,
        resume=False,
        dry_run=False,
        clean=False,
        verbose=verbose,
        log_file=log_file,
        log_level=log_level,
    )


# Create a sub-app for build-dataset commands
build_dataset_app = typer.Typer(
    name="build-dataset",
    help="Commands for building datasets from various sources",
    rich_markup_mode="rich",
)

# Add build-dataset as a command group to the main app
app.add_typer(build_dataset_app, name="build-dataset")


@build_dataset_app.command(name="prepare-dataset")
def prepare_dataset(
    config: Annotated[
        Path,
        typer.Argument(
            help="Path to YAML configuration file",
        ),
    ],
    dataset: Annotated[
        Path | None,
        typer.Option(
            "--dataset",
            "-d",
            help="Override dataset path from config",
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose output (DEBUG logging)"),
    ] = False,
    log_file: Annotated[
        Path | None,
        typer.Option("--log-file", help="Path to log file"),
    ] = None,
    log_level: Annotated[
        str,
        typer.Option(
            "--log-level",
            help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
        ),
    ] = "INFO",
) -> None:
    """
    Prepare dataset for classification or prediction.

    This command prepares datasets by organizing data entries
    with images and metadata for VLM classification.

    Example usage:
    ```bash
    # Using config file
    physics-agent build-dataset prepare-dataset configs/prepare_dataset.yaml

    # Override dataset path
    physics-agent build-dataset prepare-dataset configs/prepare_dataset.yaml \\
      --dataset ./data/custom
    ```
    """
    # Setup logging for this command
    logger = setup_logging(verbose=verbose, log_file=log_file, log_level=log_level)

    logger.info("Starting prepare dataset workflow")

    # Check if config exists
    if not config.exists():
        logger.error(f"Configuration file not found: {config}")
        console.print(f"[red]Error:[/red] Configuration file not found: {config}")
        raise typer.Exit(1)

    # Display configuration info
    console.print(
        Panel.fit(
            "[bold]Prepare Dataset[/bold]\n\n"
            f"Configuration: {config}\n"
            f"Dataset Override: {dataset or 'None'}\n"
            f"Output: dataset.jsonl saved to dataset directory\n"
            f"Verbose mode: {'ON' if verbose else 'OFF'}",
            border_style="blue",
        )
    )

    logger.info(f"Configuration file: {config}")
    if dataset:
        logger.info(f"Dataset override: {dataset}")

    if verbose:
        logger.debug("Verbose mode enabled - detailed logging active")

    # Use API for dataset preparation
    from physics_agent.api import (
        BuildDatasetPrepareDatasetInput,
        build_dataset_prepare_dataset,
    )

    try:
        # Create API parameters
        api_params = BuildDatasetPrepareDatasetInput(
            config=config,
            dataset_override=dataset,
            verbose=verbose,
        )

        # Run the workflow via API
        logger.info("Running prepare dataset workflow...")
        console.print("\n[cyan]Loading config and preparing dataset...[/cyan]")

        result = build_dataset_prepare_dataset(api_params)

        # Check if workflow completed successfully
        if result.success:
            dataset_entries = result.dataset_entries
            failed_models = result.failed_models
            dataset_jsonl_path = result.dataset_jsonl_path

            console.print(
                "\n[bold green]✨ Dataset preparation completed![/bold green]"
            )
            console.print(f"  • Dataset entries: {len(dataset_entries)}")
            console.print(f"  • Failed entries: {len(failed_models)}")
            console.print(f"  • Dataset saved to: {dataset_jsonl_path}")

            if failed_models:
                console.print(
                    f"[yellow]Failed entries: {', '.join(failed_models)}[/yellow]"
                )
                logger.info(f"Failed entries: {failed_models}")
        else:
            logger.error(f"Prepare dataset failed: {result.error}")
            console.print(f"[red]Error:[/red] {result.error}")
            raise typer.Exit(1)

    except Exception as e:
        logger.error(f"Error running workflow: {str(e)}", exc_info=True)
        console.print(f"\n[red]Error running workflow: {str(e)}[/red]")
        raise typer.Exit(1) from e


@build_dataset_app.command(name="usd")
def usd(
    config: Annotated[
        Path,
        typer.Argument(
            help="Path to the data preparation configuration file.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ],
    source: Annotated[
        Path | None,
        typer.Option(
            "--source",
            "-s",
            help="Path to the USD file or directory (overrides config).",
            exists=False,
            resolve_path=True,
        ),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Output directory for dataset (overrides config).",
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ] = None,
    extract_metadata: Annotated[
        bool,
        typer.Option(
            "--extract-metadata/--no-extract-metadata",
            help="Extract prim metadata (materials, transforms, etc.).",
        ),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose output (DEBUG logging)"),
    ] = False,
    log_file: Annotated[
        Path | None,
        typer.Option("--log-file", help="Path to log file"),
    ] = None,
    log_level: Annotated[
        str,
        typer.Option(
            "--log-level",
            help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
        ),
    ] = "INFO",
) -> None:
    """
    Build a dataset from USD file(s) by rendering views of each prim.

    This command will intelligently handle both single file and batch processing:
    - If config has 'usd_path': processes a single USD file
    - If config has 'usd_dir': processes all USD files in that directory

    For batch processing, subdirectories will be created for each USD file.

    Example usage:
    ```bash
    # Single file config (with usd_path)
    physics-agent build-dataset usd configs/single_usd.yaml

    # Batch processing config (with usd_dir)
    physics-agent build-dataset usd configs/usd_batch.yaml

    # Override source (file or directory)
    physics-agent build-dataset usd configs/data_prep.yaml \\
        --source path/to/file_or_dir

    # With metadata extraction
    physics-agent build-dataset usd configs/data_prep.yaml \\
        --extract-metadata
    ```
    """
    # Setup logging
    logger = setup_logging(verbose=verbose, log_file=log_file, log_level=log_level)

    logger.info("Starting Physics Agent Dataset Build Workflow")

    # Check if config exists
    if not config.exists():
        logger.error(f"Configuration file not found: {config}")
        raise typer.Exit(code=1)

    # Load config to determine if it's single file or batch processing
    try:
        with open(config, encoding="utf-8") as f:
            config_data = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        raise typer.Exit(code=1) from e

    # Determine if source override points to a directory or file
    is_batch_mode = False
    if source:
        source = Path(source)
        if source.is_dir():
            is_batch_mode = True
    elif "usd_dir" in config_data:
        is_batch_mode = True
    elif "usd_path" not in config_data:
        # Neither usd_path nor usd_dir specified
        logger.error(
            "Configuration must contain either 'usd_path' (for single file) "
            "or 'usd_dir' (for batch processing)"
        )
        raise typer.Exit(code=1)

    # Handle batch processing
    if is_batch_mode:
        logger.info("Detected batch processing mode")

        # Get USD directory
        if source and source.is_dir():
            usd_dir = source
            logger.info(f"Using USD directory override: {usd_dir}")
        elif "usd_dir" in config_data:
            # Resolve path relative to config file location
            config_dir = config.parent
            usd_dir = config_dir / Path(config_data["usd_dir"])
            usd_dir = usd_dir.resolve()
            logger.info(f"Using usd_dir from config: {usd_dir}")
        else:
            logger.error("Batch mode requires usd_dir in config or --source directory")
            raise typer.Exit(code=1)

        # Get output directory
        if output_dir:
            batch_output_dir = output_dir
        elif "output_dir" in config_data:
            config_dir = config.parent
            batch_output_dir = config_dir / Path(config_data["output_dir"])
            batch_output_dir = batch_output_dir.resolve()
        else:
            batch_output_dir = Path("output")

        # Check if USD directory exists
        if not usd_dir.exists():
            logger.error(f"USD directory not found: {usd_dir}")
            raise typer.Exit(code=1)

        # Use API for batch processing
        from physics_agent.api import BuildDatasetUsdInput, build_dataset_usd

        try:
            # Create API parameters
            api_params = BuildDatasetUsdInput(
                config=config,
                source_override=usd_dir,
                output_dir_override=batch_output_dir,
                extract_metadata=extract_metadata,
                verbose=verbose,
            )

            # Run via API
            api_result = build_dataset_usd(api_params)

            if not api_result.success:
                logger.error(f"Batch processing failed: {api_result.error}")
                raise RuntimeError(api_result.error)

            results = api_result.batch_results
            successful_builds = sum(
                1 for r in results.values() if r.get("status") == "success"
            )
            failed_builds = sum(
                1 for r in results.values() if r.get("status") != "success"
            )

        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            if verbose:
                console.print_exception(show_locals=True)
            raise typer.Exit(code=1) from e

        # Display batch results
        table = Table(title="Batch Dataset Build Results", show_header=True)
        table.add_column("USD File", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Prims", justify="right")
        table.add_column("Images", justify="right")
        table.add_column("Output Directory", style="dim")

        for usd_name, result in results.items():
            status = "✓ Success" if result["status"] == "success" else "✗ Failed"
            status_style = "green" if result["status"] == "success" else "red"

            prims = str(result.get("num_prims", "N/A"))
            images = str(result.get("num_images", "N/A"))
            output_path = Path(result["output_dir"]).name

            table.add_row(
                usd_name,
                f"[{status_style}]{status}[/{status_style}]",
                prims,
                images,
                output_path,
            )

        console.print("\n")
        console.print(table)
        console.print("\n")

        if failed_builds == 0:
            console.print(
                Panel.fit(
                    "[bold green]✓[/bold green] All datasets built successfully!",
                    border_style="green",
                )
            )
        elif successful_builds > 0:
            console.print(
                Panel.fit(
                    f"[bold yellow]⚠[/bold yellow] Completed with {failed_builds} failures",
                    border_style="yellow",
                )
            )
        else:
            console.print(
                Panel.fit(
                    "[bold red]✗[/bold red] All builds failed",
                    border_style="red",
                )
            )
            raise typer.Exit(code=1)

    else:
        # Single file processing
        logger.info("Processing single USD file")

        try:
            from physics_agent.api import BuildDatasetUsdInput, build_dataset_usd

            # Create API parameters
            api_params = BuildDatasetUsdInput(
                config=config,
                source_override=source,
                output_dir_override=output_dir,
                extract_metadata=extract_metadata,
                verbose=verbose,
            )

            # Run workflow via API
            logger.info("Executing dataset build workflow")
            result = build_dataset_usd(api_params)

            if not result.success:
                logger.error(f"Dataset build failed: {result.error}")
                raise RuntimeError(result.error)

            # Create results table
            table = Table(title="Dataset Build Results", show_header=True)
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")

            table.add_row("Dataset Manifest", str(result.dataset_path or "N/A"))
            table.add_row("Total Prims", str(result.num_prims))
            table.add_row("Total Images", str(result.num_images))

            console.print("\n")
            console.print(table)
            console.print("\n")

            console.print(
                Panel.fit(
                    "[bold green]✓[/bold green] Dataset build completed successfully!",
                    border_style="green",
                )
            )

        except Exception as e:
            logger.error(f"Workflow failed: {e}")
            if verbose:
                console.print_exception(show_locals=True)
            raise typer.Exit(code=1) from e


@app.command()
def run(
    config: Annotated[
        Path,
        typer.Argument(
            help="Path to unified YAML configuration file",
        ),
    ],
    skip: Annotated[
        str | None,
        typer.Option(
            "--skip",
            help="Comma-separated list of steps to skip",
        ),
    ] = None,
    only: Annotated[
        str | None,
        typer.Option(
            "--only",
            help="Comma-separated list of steps to run exclusively",
        ),
    ] = None,
    session_id: Annotated[
        str | None,
        typer.Option(
            "--session-id",
            help="Reuse existing session ID instead of generating a new one",
        ),
    ] = None,
    resume: Annotated[
        bool,
        typer.Option(
            "--resume",
            help="Resume from last successful checkpoint",
        ),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Show pipeline plan without executing",
        ),
    ] = False,
    clean: Annotated[
        bool,
        typer.Option(
            "--clean",
            help="Clean (delete) working directory before starting",
        ),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose output (DEBUG logging)"),
    ] = False,
    log_file: Annotated[
        Path | None,
        typer.Option("--log-file", help="Path to log file"),
    ] = None,
    log_level: Annotated[
        str,
        typer.Option(
            "--log-level",
            help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
        ),
    ] = "INFO",
) -> None:
    """
    Execute a multi-step asset agent pipeline.

    Uses the unified configuration format where all paths are auto-derived from
    project.working_dir and input.usd_path.

    A typical pipeline includes:
    1. build_dataset_usd: Build dataset from USD files
    2. build_dataset_prepare_dataset: Prepare dataset for classification
    3. predict: Run VLM inference for asset classification

    The pipeline automatically connects outputs from one step to inputs of the next.

    Example usage:
    ```bash
    # Run complete pipeline
    physics-agent run configs/unified_asset.yaml

    # Skip USD dataset building (already exists)
    physics-agent run configs/unified_asset.yaml --skip build_dataset_usd

    # Run only prediction step
    physics-agent run configs/unified_asset.yaml --only predict

    # Dry run to see execution plan
    physics-agent run configs/unified_asset.yaml --dry-run
    ```
    """
    # Setup logging for this command
    logger = setup_logging(verbose=verbose, log_file=log_file, log_level=log_level)

    # Get event listener for CLI output
    listener = get_listener({}, logger_name="physics_agent.cli")

    logger.info("Starting Physics Agent Pipeline")

    # Check if config exists
    if not config.exists():
        logger.error(f"Pipeline configuration file not found: {config}")
        console.print(
            f"[red]Error:[/red] Pipeline configuration file not found: {config}"
        )
        raise typer.Exit(1)

    # Parse skip/only options
    skip_steps = [s.strip() for s in skip.split(",")] if skip else []
    only_steps = [s.strip() for s in only.split(",")] if only else []

    # Display configuration info via event system
    listener.event(
        "pipeline.config.display",
        {
            "config": str(config),
            "skip_steps": skip_steps,
            "only_steps": only_steps,
            "resume": resume,
            "dry_run": dry_run,
            "clean": clean,
        },
    )

    if dry_run:
        # Load config and display plan without executing
        try:
            with open(config, encoding="utf-8") as f:
                pipeline_config = yaml.safe_load(f)

            console.print("\n[bold cyan]Pipeline Execution Plan:[/bold cyan]\n")

            # Detect config format (unified vs old)
            is_unified = "project" in pipeline_config

            if is_unified:
                # Unified config format
                project_name = pipeline_config.get("project", {}).get("name", "unknown")
                working_dir = pipeline_config.get("project", {}).get(
                    "working_dir", f".{project_name}"
                )

                console.print(f"[cyan]Project:[/cyan] {project_name}")
                console.print(f"[cyan]Working Directory:[/cyan] {working_dir}")
                console.print(
                    f"[cyan]Input USD:[/cyan] {pipeline_config.get('input', {}).get('usd_path', 'N/A')}"
                )

                steps_section = pipeline_config.get("steps", {})
            else:
                # Old config format
                steps_section = pipeline_config

            # Use centralized step names
            from physics_agent.api.defaults import PIPELINE_STEP_NAMES

            step_names = PIPELINE_STEP_NAMES

            table = Table(title="Steps", show_header=True)
            table.add_column("Step", style="cyan")
            table.add_column("Status", style="yellow")
            table.add_column("Enabled", style="green")

            for step in step_names:
                if step not in steps_section:
                    continue

                step_config = steps_section[step]

                # Check if enabled (for unified format)
                if is_unified:
                    enabled = step_config.get("enabled")
                    if enabled is None:
                        # Implicitly enable if step has any configuration besides 'enabled'
                        has_config = any(k != "enabled" for k in step_config.keys())
                        enabled = has_config
                    if not enabled:
                        continue

                if skip_steps and step in skip_steps:
                    status = "⊘ Skipped"
                    style_name = "dim"
                elif only_steps and step not in only_steps:
                    status = "⊘ Excluded"
                    style_name = "dim"
                else:
                    status = "→ Will Run"
                    style_name = "green"

                enabled = "Yes" if step_config.get("enabled", True) else "No"

                table.add_row(
                    f"[{style_name}]{step}[/{style_name}]",
                    f"[{style_name}]{status}[/{style_name}]",
                    f"[{style_name}]{enabled}[/{style_name}]",
                )

            console.print(table)
            console.print("\n[bold green]✓ Dry run complete[/bold green]")
            logger.info("Dry run completed successfully")
            return

        except Exception as e:
            logger.error(f"Error during dry run: {str(e)}", exc_info=True)
            console.print(f"\n[red]Error during dry run: {str(e)}[/red]")
            raise typer.Exit(1) from e

    # Execute unified pipeline using API
    try:
        from physics_agent.api import PipelineInput, run_pipeline

        logger.info("Creating unified pipeline workflow")

        # Create CLI event listener with Rich formatting
        from physics_agent.api import CLIEventListener

        cli_listener = CLIEventListener(
            logger=logger, console=console, show_events=False
        )

        # Create API parameters
        api_params = PipelineInput(
            config=config,
            skip_steps=skip_steps,
            only_steps=only_steps,
            session_id=session_id,
            resume=resume,
            dry_run=False,
            clean=clean,
            verbose=False,  # Logging already set up
            event_listener=cli_listener,
        )

        logger.info("Running unified pipeline workflow")
        console.print()

        result = run_pipeline(api_params)

        # Display results
        if result.success:
            console.print()

            # Display summary of each step
            if result.step_results:
                console.print("[bold cyan]Pipeline Results Summary:[/bold cyan]\n")

                for step_name, step_output in result.step_results.items():
                    console.print(f"[green]✓[/green] {step_name}")
                    if step_output:
                        for key, value in step_output.items():
                            if value is not None:
                                console.print(f"  • {key}: {value}")

            logger.info("Pipeline completed successfully")
        else:
            logger.error(f"Pipeline failed: {result.error}")
            console.print(f"[red]Error:[/red] {result.error}")
            raise typer.Exit(1)

    except Exception as e:
        logger.error(f"Error running pipeline: {str(e)}", exc_info=True)
        console.print(f"\n[red]Error running pipeline: {str(e)}[/red]")
        if resume:
            console.print(
                "\n[yellow]Tip:[/yellow] Pipeline checkpoint saved. Use --resume to continue."
            )
        raise typer.Exit(1) from e


@app.command()
def pipeline(
    config: Annotated[
        Path,
        typer.Argument(
            help="Path to unified YAML configuration file",
        ),
    ],
    skip: Annotated[
        str | None,
        typer.Option(
            "--skip",
            help="Comma-separated list of steps to skip",
        ),
    ] = None,
    only: Annotated[
        str | None,
        typer.Option(
            "--only",
            help="Comma-separated list of steps to run exclusively",
        ),
    ] = None,
    resume: Annotated[
        bool,
        typer.Option(
            "--resume",
            help="Resume from last successful checkpoint",
        ),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Show pipeline plan without executing",
        ),
    ] = False,
    clean: Annotated[
        bool,
        typer.Option(
            "--clean",
            help="Clean (delete) working directory before starting",
        ),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose output (DEBUG logging)"),
    ] = False,
    log_file: Annotated[
        Path | None,
        typer.Option("--log-file", help="Path to log file"),
    ] = None,
    log_level: Annotated[
        str,
        typer.Option(
            "--log-level",
            help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
        ),
    ] = "INFO",
) -> None:
    """
    [DEPRECATED] Execute a multi-step asset agent pipeline.

    **This command is deprecated. Please use 'physics-agent run' instead.**

    This is an alias for the 'run' command and will be removed in a future version.
    """
    # Print deprecation warning
    console.print(
        "[yellow]⚠ Warning:[/yellow] The 'pipeline' command is deprecated and will be removed in a future version."
    )
    console.print(
        "[yellow]           Please use 'physics-agent run' instead.[/yellow]\n"
    )

    # Call the run command with the same arguments
    run(
        config=config,
        skip=skip,
        only=only,
        resume=resume,
        dry_run=dry_run,
        clean=clean,
        verbose=verbose,
        log_file=log_file,
        log_level=log_level,
    )


if __name__ == "__main__":
    app()
