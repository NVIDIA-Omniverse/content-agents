# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Material Agent CLI interface using Typer and Rich."""

import atexit
import io
import logging
import os
import sys
import uuid
from pathlib import Path
from typing import Annotated

# Ensure stdout/stderr use UTF-8 on Windows (avoids charmap errors with Unicode
# characters such as arrows printed by Rich tables).
if hasattr(sys.stdout, "buffer") and (sys.stdout.encoding or "").lower() not in (
    "utf-8",
    "utf8",
):
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
    )
if hasattr(sys.stderr, "buffer") and (sys.stderr.encoding or "").lower() not in (
    "utf-8",
    "utf8",
):
    sys.stderr = io.TextIOWrapper(
        sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True
    )

import typer
import yaml
from dotenv import load_dotenv
from rich import print
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from world_understanding.agentic.events import get_listener

# Import telemetry initialization functions
from world_understanding.telemetry import (
    TelemetryConfig,
    get_tracer,
    initialize_telemetry,
    shutdown_telemetry,
)
from world_understanding.telemetry.attributes import MAAttributes

from .scene.cli import scene_app
from .utils import get_version

__version__ = get_version()

# Load environment variables from .env file
load_dotenv()

# Initialize Typer app and Rich console
app = typer.Typer(
    name="material-agent",
    help="Material Agent - VLM-based material assignment for 3D objects",
    add_completion=False,
    rich_markup_mode="rich",
)
console = Console()


def _get_cli_user_email() -> str | None:
    """Get optional user email for telemetry from environment."""
    user_email = os.getenv("MA_USER_EMAIL", "").strip()
    return user_email or None


_ENV_OVERRIDE_VLM_BACKEND = "MA_VLM_BACKEND"
_ENV_OVERRIDE_VLM_MODEL = "MA_VLM_MODEL"
_ENV_OVERRIDE_LLM_BACKEND = "MA_LLM_BACKEND"
_ENV_OVERRIDE_LLM_MODEL = "MA_LLM_MODEL"


def _maybe_apply_backend_env_overrides(config_path: Path) -> Path:
    """Apply MA_VLM_* / MA_LLM_* env overrides to the pipeline config.

    CI jobs and local users commonly prepend ``MA_VLM_BACKEND=…`` to
    ``material-agent run`` expecting the config's VLM/LLM backend+model
    to be overridden at runtime (the service already honors these env
    vars). If any override is set, load the YAML, patch every step's
    ``vlm`` and ``llm`` subsections, write the patched config to a
    temp file in the same directory as the original (so relative
    ``input.usd_path`` and ``materials.path`` entries still resolve),
    and return that temp path. Otherwise return ``config_path``
    unchanged.

    The temp file is registered with atexit for cleanup.
    """
    overrides: dict[str, str] = {}
    for env_var, field in (
        (_ENV_OVERRIDE_VLM_BACKEND, ("vlm", "backend")),
        (_ENV_OVERRIDE_VLM_MODEL, ("vlm", "model")),
        (_ENV_OVERRIDE_LLM_BACKEND, ("llm", "backend")),
        (_ENV_OVERRIDE_LLM_MODEL, ("llm", "model")),
    ):
        value = os.environ.get(env_var, "").strip()
        if value:
            overrides[f"{field[0]}.{field[1]}"] = value

    if not overrides:
        return config_path

    import tempfile

    with open(config_path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    steps = raw.get("steps", {}) or {}
    for step_config in steps.values():
        if not isinstance(step_config, dict):
            continue
        for section in ("vlm", "llm"):
            section_config = step_config.get(section)
            if not isinstance(section_config, dict):
                continue
            for key in ("backend", "model"):
                env_key = f"{section}.{key}"
                if env_key in overrides:
                    section_config[key] = overrides[env_key]

    temp_dir = config_path.parent
    fd, tmp_name = tempfile.mkstemp(
        prefix=f"{config_path.stem}.env-override.",
        suffix=config_path.suffix,
        dir=str(temp_dir),
    )
    temp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            yaml.safe_dump(raw, fh, sort_keys=False)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
    atexit.register(lambda p=temp_path: p.unlink(missing_ok=True))

    summary = ", ".join(f"{k}={v}" for k, v in overrides.items())
    print(
        f"[yellow][run] Applied backend env overrides:[/yellow] {summary} "
        f"[dim](temp config: {temp_path.name})[/dim]"
    )
    return temp_path


def _get_cli_telemetry_session_id(session_id: str | None) -> str:
    """Get session identifier used for CLI telemetry tagging."""
    if session_id:
        return session_id
    return str(uuid.uuid4())


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
        agent_name="material_agent",
        verbose=verbose,
        log_file=log_file,
        log_level=log_level,
    )


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        print(
            f"[bold blue]Material Agent[/bold blue] version [green]{__version__}[/green]"
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
    Material Agent - VLM-based material assignment for 3D objects.

    Use [bold]material-agent --help[/bold] to see available commands.
    """
    # Setup logging
    logger = setup_logging(verbose=verbose, log_file=log_file, log_level=log_level)

    # Initialize telemetry (reads from env vars via TelemetryConfig)
    # Telemetry is optional - failures are logged but don't crash the app
    telemetry_config = TelemetryConfig()
    tracer_provider = initialize_telemetry(telemetry_config)
    if tracer_provider is not None:
        logger.info(
            f"Telemetry initialized: enabled={telemetry_config.enabled}, "
            f"service={telemetry_config.service_name}, "
            f"exporters={telemetry_config.exporters}"
        )
        # Register shutdown handler for clean telemetry shutdown
        atexit.register(shutdown_telemetry)
    elif telemetry_config.enabled:
        logger.warning("Telemetry enabled but failed to initialize (check logs above)")
    else:
        logger.debug("Telemetry disabled via OTEL_ENABLED=false")

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
def benchmark(
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
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Override output directory from config",
        ),
    ] = None,
    resume: Annotated[
        bool,
        typer.Option(
            "--resume/--no-resume", help="Resume from existing predictions.jsonl"
        ),
    ] = False,
    stream_predictions: Annotated[
        bool,
        typer.Option(
            "--stream-predictions/--no-stream-predictions",
            help="Append predictions to predictions.jsonl as they are produced",
        ),
    ] = True,
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
    Run benchmarks to evaluate Material Agent performance.

    This command runs the Material Agent on a dataset of test cases
    and generates performance metrics including Functional Correctness Score (FCS).

    Example usage:
    ```bash
    # Using config file
    material-agent benchmark configs/benchmark_azure.yaml

    # Override dataset from command line
    material-agent benchmark configs/benchmark_azure.yaml --dataset data/custom.jsonl

    # Override output directory
    material-agent benchmark configs/benchmark_azure.yaml --output results/
    ```
    """
    # Setup logging for this command
    logger = setup_logging(verbose=verbose, log_file=log_file, log_level=log_level)

    logger.info("Starting Material Agent Benchmark")

    # Check if config file exists
    if not config.exists():
        logger.error(f"Configuration file not found: {config}")
        console.print(f"[red]Error:[/red] Configuration file not found: {config}")
        raise typer.Exit(1)

    config = _maybe_apply_backend_env_overrides(config)

    console.print(
        Panel.fit(
            "[bold]Material Agent Benchmark[/bold]\n\n"
            f"Configuration: {config}\n"
            f"Dataset override: {dataset or 'None'}\n"
            f"Output override: {output or 'None'}\n"
            f"Verbose mode: {'ON' if verbose else 'OFF'}",
            border_style="blue",
        )
    )

    logger.info(f"Configuration file: {config}")
    if dataset:
        logger.info(f"Dataset override: {dataset}")
    if output:
        logger.info(f"Output directory override: {output}")

    if verbose:
        logger.debug("Verbose mode enabled - detailed logging active")

    # Use API instead of directly creating workflow
    from material_agent.api import BenchmarkInput, run_benchmark

    try:
        # Create API parameters
        api_params = BenchmarkInput(
            config=config,
            dataset_override=dataset,
            output_dir_override=output,
            resume=resume,
            stream_predictions=stream_predictions,
            verbose=verbose,
        )

        # Run benchmark via API
        logger.info("Running benchmark workflow...")
        console.print(
            "\n[cyan]Loading config, provisioning models, and running benchmark...[/cyan]"
        )

        result = run_benchmark(api_params)

        # Check if successful
        if result.success and result.metrics:
            logger.info("Benchmark completed successfully")

            # Display results using same format as evaluate command
            console.print("\n[bold green]Benchmark Results[/bold green]")
            console.print("=" * 50)

            # Create metrics table
            table = Table(title="Performance Metrics", show_header=True)
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")

            # Use the metrics from API result
            metrics = result.metrics
            table.add_row(
                "Functional Correctness Score (FCS)",
                f"{metrics.functional_correctness_score}/5.0",
            )
            table.add_row(
                "Success Rate (Judge)",
                f"{metrics.success_rate}%",
            )
            table.add_row(
                "Exact Match Rate",
                f"{metrics.exact_match_rate}%",
            )
            table.add_row("Total Cases", str(metrics.total_cases))
            table.add_row("Valid Cases", str(metrics.valid_cases))
            table.add_row("Successful Cases (Judge)", str(metrics.successful_cases))
            table.add_row("Exact Matches", str(metrics.exact_matches))
            table.add_row("Failed Cases", str(metrics.failure_count))

            console.print(table)

            # Show score distribution if available
            if metrics.score_distribution:
                console.print("\n[cyan]Score Distribution:[/cyan]")
                for score, count in sorted(metrics.score_distribution.items()):
                    bar = "█" * count
                    console.print(f"  Score {score}: {bar} ({count})")

            console.print(
                "\n[bold green]✨ Benchmark completed successfully![/bold green]"
            )

            # Get output paths from API result
            if result.evaluation_path:
                logger.info(f"Evaluation results saved to: {result.evaluation_path}")
                console.print(
                    f"[dim]Evaluation results saved to: {result.evaluation_path}[/dim]"
                )
            if result.predictions_path:
                logger.info(f"Predictions saved to: {result.predictions_path}")
                console.print(
                    f"[dim]Predictions saved to: {result.predictions_path}[/dim]"
                )
        else:
            # Handle API error
            logger.error(f"Benchmark failed: {result.error}")
            console.print(f"[red]Error:[/red] {result.error}")
            raise typer.Exit(1)
    except Exception as e:
        logger.error(f"Error running benchmark: {str(e)}", exc_info=True)
        console.print(f"\n[red]Error running benchmark: {str(e)}[/red]")
        raise typer.Exit(1) from e


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
    Run material predictions on a dataset without evaluation.

    This is equivalent to: material-agent pipeline CONFIG --only predict

    Uses the unified configuration format where all paths are auto-derived from
    project.working_dir. The predict step will run VLM inference to predict materials.

    Example usage:
    ```bash
    material-agent predict configs/unified_ladder.yaml
    ```

    Output:
    - {working_dir}/predictions/predictions.jsonl: Material predictions with reasoning
    - {working_dir}/predictions/report.html: HTML report with visualizations
    """
    # This is just an alias for: pipeline --only predict
    return pipeline(
        config=config,
        skip=None,
        only="predict",
        resume=False,
        dry_run=False,
        verbose=verbose,
        log_file=log_file,
        log_level=log_level,
    )


@app.command()
def evaluate(
    config: Annotated[
        Path,
        typer.Argument(
            help="Path to evaluation configuration YAML file",
        ),
    ],
    predictions: Annotated[
        Path | None,
        typer.Argument(
            help="Path to predictions JSONL file to evaluate (overrides config)",
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
    Evaluate existing predictions using an LLM judge.

    This command loads an evaluation configuration file and evaluates predictions
    against ground truth using the configured LLM judge. It calculates
    metrics including Functional Correctness Score (FCS) and success rate.

    The configuration file must specify:
    - predictions_path: Path to predictions JSONL file
    - llm_judge: LLM configuration for evaluation
    - dataset_path: Optional path to dataset for ground truth

    The predictions file must contain:
    - id: Entry identifier
    - materials: Predicted material assignments
    - ground_truth: Expected material assignments (or loaded from dataset)

    Example usage:
    ```bash
    # Evaluate using config file
    material-agent evaluate configs/evaluate_azure.yaml

    # Evaluate with predictions override
    material-agent evaluate configs/evaluate.yaml output/predictions.jsonl
    ```
    """
    # Setup logging for this command
    logger = setup_logging(verbose=verbose, log_file=log_file, log_level=log_level)

    logger.info("Starting Material Agent Evaluation")

    # Check if config exists
    if not config.exists():
        logger.error(f"Configuration file not found: {config}")
        console.print(f"[red]Error:[/red] Configuration file not found: {config}")
        raise typer.Exit(1)

    config = _maybe_apply_backend_env_overrides(config)

    # Display evaluation info
    panel_content = "[bold]Material Agent Evaluation[/bold]\n\n"
    panel_content += f"Configuration: {config}\n"
    if predictions:
        panel_content += f"Predictions override: {predictions}\n"
    panel_content += f"Verbose mode: {'ON' if verbose else 'OFF'}"

    console.print(
        Panel.fit(
            panel_content,
            border_style="blue",
        )
    )

    logger.info(f"Configuration file: {config}")
    if predictions:
        # Validate predictions file exists if provided as override
        if not predictions.exists():
            logger.error(f"Predictions file not found: {predictions}")
            console.print(
                f"[red]Error:[/red] Predictions file not found: {predictions}"
            )
            raise typer.Exit(1)
        logger.info(f"Predictions override: {predictions}")

    if verbose:
        logger.debug("Verbose mode enabled - detailed logging active")

    # Use API instead of directly creating workflow
    from material_agent.api import EvaluateInput, run_evaluate

    try:
        # Create API parameters
        api_params = EvaluateInput(
            config=config,
            predictions_override=predictions,
            verbose=verbose,
        )

        # Run evaluation via API
        logger.info("Running evaluation...")
        console.print(
            "\n[cyan]Loading config, provisioning LLM judge, and evaluating predictions...[/cyan]"
        )

        result = run_evaluate(api_params)

        # Check if evaluation was successful
        if result.success and result.metrics:
            metrics = result.metrics

            # Display results
            console.print("\n[bold green]Evaluation Results[/bold green]")
            console.print("=" * 50)

            # Create metrics table
            table = Table(title="Performance Metrics", show_header=True)
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")

            # Use the metrics from API result
            table.add_row(
                "Functional Correctness Score (FCS)",
                f"{metrics.functional_correctness_score}/5.0",
            )
            table.add_row(
                "Success Rate (Judge)",
                f"{metrics.success_rate}%",
            )
            table.add_row(
                "Exact Match Rate",
                f"{metrics.exact_match_rate}%",
            )
            table.add_row("Total Cases", str(metrics.total_cases))
            table.add_row("Valid Cases", str(metrics.valid_cases))
            table.add_row("Successful Cases (Judge)", str(metrics.successful_cases))
            table.add_row("Exact Matches", str(metrics.exact_matches))
            table.add_row("Failed Cases", str(metrics.failure_count))

            console.print(table)

            # Show score distribution if available
            if metrics.score_distribution:
                console.print("\n[cyan]Score Distribution:[/cyan]")
                for score, count in sorted(metrics.score_distribution.items()):
                    bar = "█" * count
                    console.print(f"  Score {score}: {bar} ({count})")

            console.print(
                "\n[bold green]✨ Evaluation completed successfully![/bold green]"
            )

            if result.evaluation_path:
                logger.info(f"Evaluation results saved to: {result.evaluation_path}")
                console.print(f"Evaluation results saved to: {result.evaluation_path}")

            # Display HTML report path if generated
            if result.html_report_path:
                console.print(
                    f"[cyan]HTML report generated: {result.html_report_path}[/cyan]"
                )
        else:
            # Handle API error
            logger.error(f"Evaluation failed: {result.error}")
            console.print(f"[red]Error:[/red] {result.error}")
            raise typer.Exit(1)

    except Exception as e:
        logger.error(f"Error running evaluation: {str(e)}", exc_info=True)
        console.print(f"\n[red]Error running evaluation: {str(e)}[/red]")
        raise typer.Exit(1) from e


# Create a sub-app for build-dataset commands
build_dataset_app = typer.Typer(
    name="build-dataset",
    help="Commands for building datasets from various sources",
    rich_markup_mode="rich",
)

# Add build-dataset as a command group to the main app
app.add_typer(build_dataset_app, name="build-dataset")


@build_dataset_app.command(name="pdf_vectorstore")
def build_pdf_vectorstore(
    config: Annotated[
        Path,
        typer.Argument(
            help="Path to YAML configuration file",
        ),
    ],
    source: Annotated[
        Path | None,
        typer.Option(
            "--source",
            "-s",
            help="Override source path from config (PDF file or directory)",
        ),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Override output directory from config",
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
    Build a multimodal vector store from PDF documents.

    This command processes PDF files to extract content (text, images, tables),
    splits them by type, and creates a searchable vector store.

    Example usage:
    ```bash
    # Using config file
    material-agent build-dataset pdf_vectorstore configs/pdf_vectorstore.yaml

    # Override source path
    material-agent build-dataset pdf_vectorstore configs/pdf_vectorstore.yaml --source docs/

    # Override output directory
    material-agent build-dataset pdf_vectorstore configs/pdf_vectorstore.yaml --output ./vectorstore/
    ```
    """
    # Setup logging for this command
    logger = setup_logging(verbose=verbose, log_file=log_file, log_level=log_level)

    logger.info("Starting PDF to VectorStore workflow")

    # Check if config exists
    if not config.exists():
        logger.error(f"Configuration file not found: {config}")
        console.print(f"[red]Error:[/red] Configuration file not found: {config}")
        raise typer.Exit(1)

    # Display configuration info
    console.print(
        Panel.fit(
            "[bold]PDF to VectorStore Builder[/bold]\n\n"
            f"Configuration: {config}\n"
            f"Source override: {source or 'None'}\n"
            f"Output override: {output or 'None'}\n"
            f"Verbose mode: {'ON' if verbose else 'OFF'}",
            border_style="blue",
        )
    )

    logger.info(f"Configuration file: {config}")
    if source:
        logger.info(f"Source override: {source}")
    if output:
        logger.info(f"Output directory override: {output}")

    if verbose:
        logger.debug("Verbose mode enabled - detailed logging active")

    # Use API for PDF vectorstore building
    from material_agent.api import (
        BuildDatasetPdfVectorstoreInput,
        build_dataset_pdf_vectorstore,
    )

    try:
        # Create API parameters
        api_params = BuildDatasetPdfVectorstoreInput(
            config=config,
            source_override=source,
            output_dir_override=output,
            verbose=verbose,
        )

        # Run the workflow via API
        console.print("\n[cyan]Processing PDFs and building vector store...[/cyan]")

        result = build_dataset_pdf_vectorstore(api_params)

        # Check if workflow completed successfully
        if result.success:
            logger.info("PDF vectorstore workflow completed successfully")

            # Display results
            console.print(
                "\n[bold green]✨ Vector store created successfully![/bold green]"
            )

            # Show extraction results if available
            if result.extraction_result:
                extraction = result.extraction_result
                console.print("\n[bold]Extraction Results:[/bold]")
                console.print(
                    f"  • Documents processed: {extraction.get('document_count', 0)}"
                )
                if "content_types" in extraction:
                    console.print(f"  • Content types: {extraction['content_types']}")

            # Show split results if available
            if result.split_result:
                split = result.split_result
                console.print("\n[bold]Content Split Results:[/bold]")
                console.print(
                    f"  • Files created: {split.get('total_files_created', 0)}"
                )
                if "content_type_distribution" in split:
                    console.print(
                        f"  • Distribution: {split['content_type_distribution']}"
                    )

            # Show vectorstore results
            console.print("\n[bold]Vector Store Results:[/bold]")
            console.print(f"  • Documents indexed: {result.num_documents_indexed}")
            console.print(f"  • Text documents: {result.num_texts}")
            console.print(f"  • Image documents: {result.num_images}")
            console.print(f"  • Embedding dimension: {result.embedding_dimension}")
            if result.vectorstore_path:
                console.print(f"  • Saved to: {result.vectorstore_path}")
        else:
            logger.error(f"PDF vectorstore workflow failed: {result.error}")
            console.print("\n[red]❌ Workflow failed![/red]")
            console.print(f"[red]Error:[/red] {result.error}")
            raise typer.Exit(1)

    except Exception as e:
        logger.error(f"Error running workflow: {str(e)}", exc_info=True)
        console.print(f"\n[red]Error running workflow: {str(e)}[/red]")
        raise typer.Exit(1) from e


@build_dataset_app.command(name="prepare-dataset")
def prepare_dataset(
    config: Annotated[
        Path,
        typer.Argument(
            help="Path to YAML configuration file",
        ),
    ],
    vector_store: Annotated[
        Path | None,
        typer.Option(
            "--vector-store",
            help="Override vector store path from config",
        ),
    ] = None,
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
    Prepare dataset with CMF specifications for benchmark or prediction.

    This command prepares datasets by extracting CMF specifications
    for model numbers using the spec_rag functionality. Can prepare either
    benchmark datasets (with ground truth) or prediction datasets (without ground truth).

    Example usage:
    ```bash
    # Using config file
    material-agent build-dataset prepare-dataset configs/prepare_dataset_pcba.yaml

    # Override vector store and dataset paths
    material-agent build-dataset prepare-dataset configs/prepare_dataset_pcba.yaml \
      --vector-store ./vectorstore --dataset ./data/pcba
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
            f"Vector Store Override: {vector_store or 'None'}\n"
            f"Dataset Override: {dataset or 'None'}\n"
            f"Models: Auto-discovered from dataset\n"
            f"Output: dataset.jsonl saved to dataset directory\n"
            f"Verbose mode: {'ON' if verbose else 'OFF'}",
            border_style="blue",
        )
    )

    logger.info(f"Configuration file: {config}")
    if vector_store:
        logger.info(f"Vector store override: {vector_store}")
    if dataset:
        logger.info(f"Dataset override: {dataset}")

    if verbose:
        logger.debug("Verbose mode enabled - detailed logging active")

    # Use API for dataset preparation
    from material_agent.api import (
        BuildDatasetPrepareDatasetInput,
        build_dataset_prepare_dataset,
    )

    try:
        # Create API parameters
        api_params = BuildDatasetPrepareDatasetInput(
            config=config,
            vector_store_override=vector_store,
            dataset_override=dataset,
            verbose=verbose,
        )

        # Run the workflow via API
        logger.info("Running prepare dataset workflow...")
        console.print(
            "\n[cyan]Loading config, provisioning LLM, and preparing benchmark data...[/cyan]"
        )

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
            console.print(f"  • Failed models: {len(failed_models)}")
            console.print(f"  • Dataset saved to: {dataset_jsonl_path}")

            if failed_models:
                console.print(
                    f"[yellow]Failed models: {', '.join(failed_models)}[/yellow]"
                )
                logger.info(f"Failed models: {failed_models}")
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
    material-agent build-dataset usd configs/single_usd.yaml

    # Batch processing config (with usd_dir)
    material-agent build-dataset usd configs/usd_pcba.yaml

    # Override source (file or directory)
    material-agent build-dataset usd configs/data_prep.yaml \\
        --source path/to/file_or_dir

    # With metadata extraction
    material-agent build-dataset usd configs/data_prep.yaml \\
        --extract-metadata
    ```
    """
    # Setup logging
    logger = setup_logging(verbose=verbose, log_file=log_file, log_level=log_level)

    logger.info("Starting Material Agent Dataset Build Workflow")

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
        from material_agent.api import BuildDatasetUsdInput, build_dataset_usd

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
            from material_agent.api import BuildDatasetUsdInput, build_dataset_usd

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
def apply(
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
    Apply predicted materials to a USD file.

    This is equivalent to: material-agent pipeline CONFIG --only apply

    Uses the unified configuration format where all paths are auto-derived from
    project.working_dir. The apply step will apply predicted materials to the USD file.

    Example usage:
    ```bash
    material-agent apply configs/unified_ladder.yaml
    ```

    Output:
    - output.usd_path: USD file with materials applied (as specified in config)
    """
    # This is just an alias for: pipeline --only apply
    return pipeline(
        config=config,
        skip=None,
        only="apply",
        resume=False,
        dry_run=False,
        verbose=verbose,
        log_file=log_file,
        log_level=log_level,
    )


# Keep old functions below for reference during migration
# These will be deleted after migration is complete
def _legacy_apply(
    config: Path,
    input_usd: Path | None,
    predictions: Path | None,
    output: Path | None,
    layer_only: bool,
    render: bool,
    verbose: bool,
    log_file: Path | None,
    log_level: str,
) -> None:
    """Legacy apply implementation - for migration reference only."""
    # Setup logging for this command
    logger = setup_logging(verbose=verbose, log_file=log_file, log_level=log_level)

    logger.info("Starting Material Agent Apply")

    # Check if config exists
    if not config.exists():
        logger.error(f"Configuration file not found: {config}")
        console.print(f"[red]Error:[/red] Configuration file not found: {config}")
        raise typer.Exit(1)

    # Display configuration info
    console.print(
        Panel.fit(
            "[bold]Material Agent Apply[/bold]\n\n"
            f"Configuration: {config}\n"
            f"Input USD override: {input_usd or 'None'}\n"
            f"Predictions override: {predictions or 'None'}\n"
            f"Output override: {output or 'None'}\n"
            f"Output mode: {'Layer only' if layer_only else 'Full stage'}\n"
            f"Verbose mode: {'ON' if verbose else 'OFF'}",
            border_style="blue",
        )
    )

    logger.info(f"Configuration file: {config}")
    if input_usd:
        logger.info(f"Input USD override: {input_usd}")
    if predictions:
        logger.info(f"Predictions override: {predictions}")
    if output:
        logger.info(f"Output override: {output}")

    if verbose:
        logger.debug("Verbose mode enabled - detailed logging active")

    # Import workflow factory
    from material_agent.workflows.factory import create_apply_workflow_from_config

    # Create config-driven apply workflow
    try:
        logger.info("Creating config-driven apply workflow...")
        workflow = create_apply_workflow_from_config()
        console.print("[green]✓ Config-driven apply workflow created[/green]")
    except Exception as e:
        logger.error(f"Failed to create workflow: {str(e)}")
        console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(1) from e

    # Run the apply workflow
    try:
        logger.info("Running material application...")
        console.print(
            "\n[cyan]Loading config, identifying materials, and applying to USD...[/cyan]"
        )

        # Prepare initial context with config path and overrides
        initial_context = {
            "config_path": str(config),
            "input_usd_override": str(input_usd) if input_usd else None,
            "predictions_override": str(predictions) if predictions else None,
            "output_usd_override": str(output) if output else None,
            "layer_only": layer_only,  # Pass layer_only flag
            "render_enabled": render,  # Pass render flag
            "verbose": verbose,
        }

        result = workflow.run(initial_context=initial_context)

        # Check if application was successful
        if result.get("application_complete"):
            unique_materials = result.get("unique_materials", [])
            matched_materials = result.get("matched_materials", {})
            materials_applied = result.get("materials_applied", {})
            assignment_stats = result.get("assignment_stats", {})
            output_path = result.get("output_usd_path")
            layer_only = result.get("layer_only", False)
            result.get("rendered_image_path")
            rendered_images = result.get("rendered_image_paths", [])
            rendering_skipped = result.get("rendering_skipped", True)

            output_message = (
                f"\n[bold green]✨ Material application complete![/bold green]\n"
                f"  • Unique materials found: {len(unique_materials)}\n"
                f"  • Materials matched via USD Search: {len(matched_materials)}\n"
                f"  • Materials applied to USD: {len(materials_applied)}\n"
                f"  • Prims with materials: {assignment_stats.get('total_prims', 0)}\n"
                f"  • Output mode: {'Layer only' if layer_only else 'Full stage'}\n"
                f"  • Output USD file: {output_path}"
            )

            # Add rendering information if enabled
            if not rendering_skipped and rendered_images:
                if len(rendered_images) == 1:
                    output_message += f"\n  • Rendered image: {rendered_images[0]}"
                else:
                    output_message += (
                        f"\n  • Rendered images ({len(rendered_images)} views):"
                    )
                    for img_path in rendered_images:
                        output_message += f"\n    - {img_path}"

            console.print(output_message)

            # Display material search results
            if matched_materials:
                console.print("\n[cyan]Material Search Results:[/cyan]")
                for material, path_infos in matched_materials.items():
                    console.print(f"  • {material}: {len(path_infos)} matches found")
                    # Always show first match details if available
                    if path_infos and len(path_infos) > 0:
                        path_info = path_infos[0]
                        if isinstance(path_info, dict):
                            if path_info.get("source_path"):
                                console.print(f"    Source: {path_info['source_path']}")
                            if path_info.get("s3_path"):
                                console.print(f"    S3:     {path_info['s3_path']}")
                        else:
                            # Fallback for old format
                            console.print(f"    - {path_info}")
                    # Show more matches in verbose mode
                    if verbose and len(path_infos) > 1:
                        for i, path_info in enumerate(
                            path_infos[1:3], start=2
                        ):  # Show next 2 paths
                            console.print(f"    [{i}]")
                            if isinstance(path_info, dict):
                                if path_info.get("source_path"):
                                    console.print(
                                        f"        Source: {path_info['source_path']}"
                                    )
                                if path_info.get("s3_path"):
                                    console.print(
                                        f"        S3:     {path_info['s3_path']}"
                                    )
                            else:
                                # Fallback for old format
                                console.print(f"        - {path_info}")
                        if len(path_infos) > 3:
                            console.print(f"    ... and {len(path_infos) - 3} more")

            # Display resolved material files
            resolved_materials = result.get("resolved_materials", {})
            download_stats = result.get("download_stats", {})

            if resolved_materials:
                console.print("\n[cyan]Resolved Material Files:[/cyan]")
                for material, local_path in resolved_materials.items():
                    # Check if it's a local file or S3 path
                    if local_path.startswith("s3://"):
                        console.print(
                            f"  • {material}: [yellow]S3[/yellow] {local_path}"
                        )
                    else:
                        console.print(
                            f"  • {material}: [green]Local[/green] {local_path}"
                        )

                # Show download statistics
                if download_stats:
                    console.print("\n[cyan]Resolution Statistics:[/cyan]")
                    console.print(
                        f"  • Found locally: {download_stats.get('found_local', 0)}"
                    )
                    console.print(
                        f"  • Downloaded from S3: {download_stats.get('downloaded', 0)}"
                    )
                    console.print(f"  • Failed: {download_stats.get('failed', 0)}")
                    console.print(f"  • Skipped: {download_stats.get('skipped', 0)}")

            # Display USD assignment results
            if materials_applied:
                console.print("\n[cyan]USD Material Assignment:[/cyan]")
                console.print(
                    f"  • Materials created: {assignment_stats.get('materials_created', 0)}"
                )
                console.print(
                    f"  • Materials applied: {assignment_stats.get('materials_applied', 0)}"
                )
                console.print(
                    f"  • Prims updated: {assignment_stats.get('total_prims', 0)}"
                )
                console.print(
                    f"  • Failed assignments: {assignment_stats.get('failed', 0)}"
                )

            logger.info(f"Material application saved to: {output_path}")
        else:
            logger.error("Apply workflow did not complete successfully")
            console.print("[red]Error:[/red] Apply workflow did not complete")
            raise typer.Exit(1)

    except Exception as e:
        logger.error(f"Error running apply workflow: {str(e)}", exc_info=True)
        console.print(f"\n[red]Error running apply workflow: {str(e)}[/red]")
        raise typer.Exit(1) from e


@app.command()
def refine(
    config: Annotated[
        Path,
        typer.Argument(
            help="Path to YAML configuration file",
        ),
    ],
    max_iterations: Annotated[
        int | None,
        typer.Option(
            "--max-iterations",
            "-n",
            help="Override maximum number of iterations from config",
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
    Refine materials on USD with VLM-based iterative refinement.

    This command executes a predict-apply-judge loop repeatedly until the judge
    approves the results or maximum iterations is reached. It uses VLM to predict
    materials, applies them to USD, renders the result, and has a VLM judge evaluate
    quality by comparing against reference images.

    The configuration file must specify:
    - dataset: Path to the dataset JSONL file
    - input_usd_path: Path to the input USD file
    - output_usd_path: Path for the final output (optional)
    - iteration: Iteration settings (max_iterations, save_intermediate, etc.)
    - judge: Judge configuration (reference_images, vlm settings, etc.)

    Example usage:
    ```bash
    # Run material refinement with iterative predict-apply-judge loop
    material-agent refine configs/iterative_apply.yaml

    # Override max iterations
    material-agent refine configs/iterative_apply.yaml --max-iterations 3
    ```
    """
    # Setup logging for this command
    logger = setup_logging(verbose=verbose, log_file=log_file, log_level=log_level)

    logger.info("Starting Material Agent Material Refinement")

    # Validate config file exists
    if not config.exists():
        logger.error(f"Configuration file not found: {config}")
        console.print(f"[red]Error:[/red] Configuration file not found: {config}")
        raise typer.Exit(1)

    config = _maybe_apply_backend_env_overrides(config)

    # Run the material refinement workflow using API
    try:
        from material_agent.api import RefineInput, run_refine

        logger.info("Running material refinement with iterative loop...")
        console.print(
            "\n[cyan]Starting iterative predict-apply-judge workflow...[/cyan]"
        )

        # Create API parameters
        api_params = RefineInput(
            config=config,
            max_iterations_override=max_iterations,
            verbose=verbose,
        )

        result = run_refine(api_params)

        # Check if successful
        if result.success and result.iteration_count > 0:
            iteration_count = result.iteration_count
            termination_reason = result.termination_reason
            final_score = result.final_judge_score
            final_output_path = result.final_output_path

            final_score_str = f"{final_score:.2f}" if final_score is not None else "N/A"

            # Get materials info from last iteration
            final_materials_applied = 0
            final_prims_with_materials = 0
            if result.iteration_results:
                last_iter = result.iteration_results[-1]
                final_materials_applied = last_iter.materials_applied_count
                final_prims_with_materials = last_iter.prims_with_materials

            console.print(
                f"\n[bold green]Iterative material refinement complete![/bold green]\n"
                f"  • Total iterations: {iteration_count}\n"
                f"  • Termination reason: {termination_reason}\n"
                f"  • Final judge score: {final_score_str}\n"
                f"  • Final materials applied: {final_materials_applied}\n"
                f"  • Final prims with materials: {final_prims_with_materials}"
            )

            if final_output_path:
                console.print(
                    f"\n[bold cyan]Final Output:[/bold cyan]\n  {final_output_path}"
                )

            if result.all_iteration_outputs:
                console.print("\n[cyan]Iteration Outputs:[/cyan]")
                for i, output_path in enumerate(result.all_iteration_outputs, 1):
                    console.print(f"  [{i}] {output_path}")

            if result.iteration_results:
                console.print("\n[cyan]Iteration Summary:[/cyan]")
                for iter_result in result.iteration_results:
                    iter_num = iter_result.iteration
                    score = iter_result.judge_score
                    decision = (
                        "CONTINUE" if iter_result.continue_iteration else "APPROVE"
                    )
                    score_str = f"{score:.2f}" if score is not None else "N/A"
                    console.print(
                        f"  • Iteration {iter_num}: Score={score_str}, Decision={decision}"
                    )

            logger.info(
                f"Material refinement completed after {iteration_count} iterations"
            )
        else:
            logger.error(f"Material refinement failed: {result.error}")
            console.print(f"[red]Error:[/red] {result.error}")
            raise typer.Exit(1)

    except Exception as e:
        logger.error(f"Error during material refinement: {e}", exc_info=True)
        console.print(f"\n[red]Error during material refinement: {str(e)}[/red]")
        raise typer.Exit(1) from e


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
            help="Clean (delete) working directory and output files (USD + renders) before starting",
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
    Execute a multi-step material agent pipeline.

    Uses the unified configuration format where all paths are auto-derived from
    project.working_dir, input.usd_path, and output.usd_path.

    A typical pipeline includes:
    1. build_dataset_usd: Build dataset from USD files
    2. build_dataset_pdf_vectorstore: Build vector store from PDFs (optional)
    3. build_dataset_prepare_dataset: Prepare dataset with specifications
    4. predict/benchmark: Run VLM inference
    5. apply: Apply predicted materials to USD

    The pipeline automatically connects outputs from one step to inputs of the next.

    Example usage:
    ```bash
    # Run complete pipeline
    material-agent run configs/unified_ladder.yaml

    # Skip USD dataset building (already exists)
    material-agent run configs/unified_ladder.yaml --skip build_dataset_usd

    # Run only prediction and apply steps
    material-agent run configs/unified_ladder.yaml --only predict,apply

    # Dry run to see execution plan
    material-agent run configs/unified_ladder.yaml --dry-run
    ```
    """
    # Setup logging for this command
    logger = setup_logging(verbose=verbose, log_file=log_file, log_level=log_level)

    # Get event listener for CLI output
    listener = get_listener({}, logger_name="material_agent.cli")

    logger.info("Starting Material Agent Pipeline")

    # Check if config exists
    if not config.exists():
        logger.error(f"Pipeline configuration file not found: {config}")
        console.print(
            f"[red]Error:[/red] Pipeline configuration file not found: {config}"
        )
        raise typer.Exit(1)

    # Apply MA_VLM_* / MA_LLM_* env-var overrides if any are set. The service
    # honours these env vars via its own config path; doing the same here
    # keeps CLI and service behaviour in sync and lets CI jobs redirect a
    # public-defaults config to an internal backend without editing YAML.
    config = _maybe_apply_backend_env_overrides(config)

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
                    f"[cyan]Input USD:[/cyan] {(pipeline_config.get('input') or {}).get('usd_path', 'N/A')}"
                )
                console.print(
                    f"[cyan]Output USD:[/cyan] {(pipeline_config.get('output') or {}).get('usd_path', 'N/A')}\n"
                )

                steps_section = pipeline_config.get("steps", {})
            else:
                # Old config format
                steps_section = pipeline_config

            # Use centralized step names
            from material_agent.api.defaults import PIPELINE_STEP_NAMES

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
        from material_agent.api import PipelineInput, run_pipeline

        logger.info("Creating unified pipeline workflow")

        # Create CLI event listener with Rich formatting
        from material_agent.api import CLIEventListener

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

        user_email = _get_cli_user_email()
        if user_email:
            telemetry_session_id = _get_cli_telemetry_session_id(session_id)
            tracer = get_tracer(__name__)
            with tracer.start_as_current_span("maa.pipeline.execution") as span:
                span.set_attribute(MAAttributes.PIPELINE_USER_EMAIL, user_email)
                span.set_attribute(MAAttributes.LANGFUSE_USER_ID, user_email)
                span.set_attribute(
                    MAAttributes.PIPELINE_SESSION_ID, telemetry_session_id
                )
                span.set_attribute(
                    MAAttributes.LANGFUSE_SESSION_ID, telemetry_session_id
                )
                try:
                    result = run_pipeline(api_params)
                except Exception:
                    span.set_attribute(MAAttributes.PIPELINE_STATUS, "failed")
                    raise
                span.set_attribute(
                    MAAttributes.PIPELINE_STATUS,
                    "completed" if result.success else "failed",
                )
        else:
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
            help="Clean (delete) working directory and output files (USD + renders) before starting",
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
    [DEPRECATED] Execute a multi-step material agent pipeline.

    **This command is deprecated. Please use 'material-agent run' instead.**

    This is an alias for the 'run' command and will be removed in a future version.
    """
    # Print deprecation warning
    console.print(
        "[yellow]⚠ Warning:[/yellow] The 'pipeline' command is deprecated and will be removed in a future version."
    )
    console.print(
        "[yellow]           Please use 'material-agent run' instead.[/yellow]\n"
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


@app.command()
def configure(
    output_config: Annotated[
        Path,
        typer.Argument(
            help="Path to output YAML configuration file to create",
        ),
    ],
    materials_manifest: Annotated[
        Path | None,
        typer.Option(
            "--materials-manifest",
            "-m",
            help="Path to materials manifest YAML file (contains library_path and entries)",
        ),
    ] = None,
    reference_images: Annotated[
        list[Path] | None,
        typer.Option(
            "--reference-image",
            "-r",
            help="Reference image path (can be specified multiple times)",
        ),
    ] = None,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            "-f",
            help="Overwrite existing configuration file",
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
    Create a new pipeline configuration file interactively.

    This command guides you through creating a pipeline configuration
    by asking for essential parameters and auto-populating the rest
    with sensible defaults.

    Example usage:
    ```bash
    # Create a new configuration file
    material-agent configure my_pipeline.yaml

    # Create with a materials manifest
    material-agent configure my_pipeline.yaml -m data/materials/material_libs_new/materials.yaml

    # Create with reference images
    material-agent configure my_pipeline.yaml -m materials.yaml -r ref1.jpg -r ref2.jpg

    # Overwrite existing file
    material-agent configure my_pipeline.yaml --force
    ```
    """
    # Setup logging for this command
    logger = setup_logging(verbose=verbose, log_file=log_file, log_level=log_level)

    logger.info("Starting Material Agent Configuration")

    console.print(
        Panel.fit(
            "[bold]Material Agent Configuration Wizard[/bold]\n\n"
            "This wizard will help you create a pipeline configuration file.\n"
            "You'll be asked a few questions, and the rest will be auto-populated.",
            border_style="blue",
        )
    )

    # Use API for configuration creation
    from material_agent.api import ConfigureInput, run_configure

    # Run the workflow using API
    try:
        logger.info("Running configuration wizard...")

        # Create API parameters
        api_params = ConfigureInput(
            output_config_path=output_config,
            materials_manifest=materials_manifest,
            reference_images=[str(p) for p in reference_images]
            if reference_images
            else None,
            force=force,
            verbose=verbose,
        )

        result = run_configure(api_params)

        # Check if configuration was created successfully
        if result.success:
            console.print("\n[bold green]✓ Configuration file created!")
            console.print(
                f"\n[cyan]Configuration saved to:[/cyan] {result.config_path}"
            )

            # Display summary
            console.print("\n[bold]Configuration Summary:[/bold]")
            console.print(f"  • Pipeline name: {result.pipeline_name}")
            console.print(f"  • Input USD: {result.input_usd_path}")
            if result.materials_library_path:
                console.print(f"  • Materials library: {result.materials_library_path}")
                console.print(
                    "  • Materials approach: Unified (library_path + entries)"
                )
            else:
                console.print("  • Materials library: Not specified")
                console.print("  • Materials approach: Legacy (materials_mapping)")
            console.print(
                f"  • Session ID: {result.pipeline_name}"
                f" (working dir: .{result.pipeline_name}/)"
            )

            console.print("\n[yellow]Next steps:[/yellow]")
            console.print(f"  1. Review and customize: {result.config_path}")
            if result.materials_library_path:
                console.print(
                    "  2. Update the materials.entries section with your materials"
                )
            else:
                console.print(
                    "  2. Update the materials_list and materials_mapping sections"
                )
            console.print(
                f"  3. Run the pipeline: material-agent pipeline {result.config_path}"
            )

            logger.info("Configuration wizard completed successfully")
        else:
            logger.error(f"Configuration creation failed: {result.error}")
            console.print(f"[red]Error:[/red] {result.error}")
            raise typer.Exit(1)

    except FileExistsError as e:
        logger.error(f"Configuration file already exists: {str(e)}")
        console.print(
            f"[red]Error:[/red] Configuration file already exists: {output_config}"
        )
        console.print("[yellow]Use --force to overwrite[/yellow]")
        raise typer.Exit(1) from e
    except Exception as e:
        logger.error(f"Error running configuration: {str(e)}", exc_info=True)
        console.print(f"\n[red]Error creating configuration: {str(e)}[/red]")
        raise typer.Exit(1) from e


@app.command("generate-manifest")
def generate_manifest(
    usd_file: Annotated[
        Path,
        typer.Argument(help="Path to the USD material library file"),
    ],
    output_dir: Annotated[
        Path,
        typer.Argument(help="Output directory for materials.yaml and thumbs/"),
    ],
    image_size: Annotated[
        int,
        typer.Option("--image-size", help="Thumbnail size in pixels"),
    ] = 256,
    skip_existing: Annotated[
        bool,
        typer.Option(
            "--skip-existing",
            help="Skip rendering thumbnails that already exist in the output dir",
        ),
    ] = False,
    library_path: Annotated[
        str | None,
        typer.Option(
            "--library-path",
            help="Value for library_path in materials.yaml (default: uses usd-file path)",
        ),
    ] = None,
    template: Annotated[
        Path | None,
        typer.Option(
            "--template",
            help="Path to the thumbnail template USD file (default: built-in template)",
        ),
    ] = None,
    max_workers: Annotated[
        int,
        typer.Option("--max-workers", help="Number of parallel NVCF render workers"),
    ] = 4,
    skip_descriptions: Annotated[
        bool,
        typer.Option(
            "--skip-descriptions",
            help="Skip VLM description generation (leave descriptions empty)",
        ),
    ] = False,
    vlm_backend: Annotated[
        str,
        typer.Option("--vlm-backend", help="VLM backend"),
    ] = "nim",
    vlm_model: Annotated[
        str | None,
        typer.Option("--vlm-model", help="VLM model name"),
    ] = "qwen/qwen3.5-397b-a17b",
    vlm_workers: Annotated[
        int,
        typer.Option("--vlm-workers", help="Number of parallel VLM workers"),
    ] = 8,
    list_materials: Annotated[
        bool,
        typer.Option(
            "--list-materials",
            help="List all material prims in the USD file and exit",
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
    Generate materials.yaml manifest and thumbnails from a USD material library.

    Discovers all Material prims in a USD file, renders thumbnails via NVCF
    cloud rendering, optionally generates VLM descriptions, and outputs a
    complete materials.yaml with a thumbs/ directory.

    Example usage:
    ```bash
    # Generate manifest with thumbnails and descriptions
    material-agent generate-manifest materials.usd output/

    # Skip VLM descriptions
    material-agent generate-manifest materials.usd output/ --skip-descriptions

    # Use a custom template and larger thumbnails
    material-agent generate-manifest materials.usd output/ --template my_template.usd --image-size 512

    # List materials without generating anything
    material-agent generate-manifest materials.usd output/ --list-materials

    # Resume (skip already-rendered thumbnails)
    material-agent generate-manifest materials.usd output/ --skip-existing
    ```
    """
    logger = setup_logging(verbose=verbose, log_file=log_file, log_level=log_level)

    from material_agent.manifest import (
        GenerateManifestInput,
        run_generate_manifest,
    )

    # Build params, using default template if not specified
    params_kwargs: dict = {
        "usd_file": usd_file,
        "output_dir": output_dir,
        "image_size": image_size,
        "skip_existing": skip_existing,
        "library_path": library_path,
        "max_workers": max_workers,
        "skip_descriptions": skip_descriptions,
        "vlm_backend": vlm_backend,
        "vlm_model": vlm_model,
        "vlm_workers": vlm_workers,
        "list_materials": list_materials,
        "verbose": verbose,
    }
    if template is not None:
        params_kwargs["template"] = template

    try:
        params = GenerateManifestInput(**params_kwargs)
        result = run_generate_manifest(params)

        if not result.success:
            console.print(f"[red]Error:[/red] {result.error}")
            raise typer.Exit(1)

        if list_materials:
            console.print(
                f"\n[bold]Materials in {usd_file.name}[/bold] "
                f"({result.materials_count} found):\n"
            )
            from material_agent.manifest import prim_path_to_name

            for pp in result.material_paths:
                console.print(f"  {pp}  ->  {prim_path_to_name(pp)}")
            return

        # Summary
        console.print("\n[bold green]Manifest generated successfully[/bold green]")
        table = Table(show_header=False)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Materials discovered", str(result.materials_count))
        table.add_row("Thumbnails rendered", str(result.thumbnails_count))
        table.add_row("Descriptions generated", str(result.descriptions_count))
        table.add_row("Output", str(result.yaml_path))
        table.add_row(
            "Thumbnails",
            f"{output_dir}/thumbs/{image_size}x{image_size}/",
        )
        console.print(table)

    except typer.Exit:
        raise
    except Exception as e:
        logger.error(f"Error generating manifest: {str(e)}", exc_info=True)
        console.print(f"\n[red]Error generating manifest: {str(e)}[/red]")
        raise typer.Exit(1) from e


# Register scene subcommand for large-scene multi-asset pipeline
app.add_typer(scene_app, name="scene")

if __name__ == "__main__":
    app()
