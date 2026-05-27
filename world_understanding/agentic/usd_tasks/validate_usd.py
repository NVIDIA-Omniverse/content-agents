# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tasks for validating USD files using NVIDIA USD Validation.

Uses the ``usd-validation-nvidia`` pip package (no Kit, no NVCF,
no network calls). Included as a required dependency.
"""

import json
import logging
from pathlib import Path
from typing import Any

from world_understanding.agentic.events import EventListener, get_listener
from world_understanding.agentic.tasks import Task

logger = logging.getLogger(__name__)

# Valid on_failure modes
ON_FAILURE_MODES = ("warn", "block", "fix")
# validate_output only supports warn/block (fix doesn't apply to final output)
ON_FAILURE_MODES_OUTPUT = ("warn", "block")


def _validator_import_error_message(exc: ImportError) -> str:
    """Return the user-facing message for unavailable USD validation bindings."""
    return (
        "USD validation skipped: optional dependency `usd_validation_nvidia` "
        "is not installed or could not be imported. Install the "
        "`usd-validation-nvidia` package to enable USD validation. "
        f"Import error: {exc}"
    )


def _is_usd_validator_import_error(exc: ImportError) -> bool:
    """Return True when an ImportError is from the optional USD validator."""
    current: BaseException | None = exc
    while current:
        if isinstance(current, ImportError):
            name = getattr(current, "name", None)
            if name and str(name).split(".", maxsplit=1)[0] == "usd_validation_nvidia":
                return True
            if "usd_validation_nvidia" in str(current):
                return True
        current = current.__cause__ or current.__context__
    return False


def _mark_validation_skipped(
    context: dict[str, Any],
    listener: EventListener,
    on_failure: str,
    exc: ImportError,
) -> bool:
    """Record a skipped validation step when warn mode allows continuation."""
    message = _validator_import_error_message(exc)
    context["validation_success"] = False
    context["validation_error"] = message

    if on_failure == "warn":
        context["validation_skipped"] = True
        listener.warning(f"{message} Continuing (on_failure=warn).")
        logger.warning("Input USD validation skipped: %s", exc)
        return True

    listener.error(f"{message} Pipeline blocked by on_failure={on_failure}.")
    return False


def _fixed_usd_path(input_usd: Path, output_dir: Path | str) -> Path:
    """Return a writable path for a repaired USD root layer."""
    if input_usd.suffix.lower() == ".usdz":
        return Path(output_dir) / f"fixed_{input_usd.stem}.usda"
    return Path(output_dir) / f"fixed_{input_usd.name}"


async def _run_validation(
    usd_path: Path,
    validation_config: dict[str, Any],
    listener: EventListener,
    label: str,
    fix: bool = False,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Run validation on a single USD file using NVIDIA USD Validation.

    Requires ``usd-validation-nvidia`` (install with ``uv pip install -e .``).

    Args:
        usd_path: Path to the USD file to validate
        validation_config: Validation parameters
        listener: Event listener for logging
        label: Label for log messages (e.g. "input", "output")
        fix: Whether to run local auto-fix
        output_path: Path where local fix mode should write the repaired USD

    Returns:
        Validation result dict

    Raises:
        RuntimeError: If validation fails or validator not installed
    """
    listener.info(f"Validating {label} USD: {usd_path}")

    from world_understanding.functions.graphics.validate_usd import (
        validate_usd,
    )

    result = validate_usd(
        input_path=usd_path,
        categories=validation_config.get("categories"),
        fix=fix,
        output_path=output_path,
        stage_timeout=validation_config.get("stage_timeout", 180.0),
    )

    if result.get("status") != "success":
        error_msg = result.get("error", "Unknown validation error")
        raise RuntimeError(f"Validation of {label} USD failed: {error_msg}")

    summary = result.get("summary", {})
    issues = result.get("issues", [])

    validation_time = result.get("validation_time", 0)
    listener.info(f"[{label}] Validation completed in {validation_time:.2f}s")
    listener.info(
        f"[{label}] {summary.get('total_issues', 0)} issues "
        f"({summary.get('failures', 0)} failures, "
        f"{summary.get('warnings', 0)} warnings, "
        f"{summary.get('errors', 0)} errors)"
    )
    listener.info(f"[{label}] Asset is valid: {summary.get('is_valid', True)}")

    # Log top issues by rule
    if issues:
        from collections import Counter

        rules = Counter(i.get("rule", "unknown") for i in issues)
        listener.info(f"[{label}] Top issues by rule:")
        for rule, count in rules.most_common(10):
            listener.info(f"  {rule}: {count}")

    return result


def _compare_issues(
    baseline_issues: list[dict[str, Any]],
    current_issues: list[dict[str, Any]],
    listener: EventListener,
) -> list[dict[str, Any]]:
    """Compare current validation issues against baseline to find new ones.

    An issue is considered "new" if its (rule, severity, at) tuple appears
    more times in current than in the baseline.

    Args:
        baseline_issues: Issues from input validation
        current_issues: Issues from output validation
        listener: Event listener for logging

    Returns:
        List of newly introduced issues
    """
    from collections import Counter

    baseline_counts: Counter[tuple[str, str, str]] = Counter()
    for issue in baseline_issues:
        sig = (
            issue.get("rule", ""),
            issue.get("severity", ""),
            issue.get("at", ""),
        )
        baseline_counts[sig] += 1

    current_counts: Counter[tuple[str, str, str]] = Counter()
    new_issues = []
    for issue in current_issues:
        sig = (
            issue.get("rule", ""),
            issue.get("severity", ""),
            issue.get("at", ""),
        )
        current_counts[sig] += 1
        if current_counts[sig] > baseline_counts.get(sig, 0):
            new_issues.append(issue)

    listener.info(
        f"Baseline comparison: {len(baseline_issues)} baseline issues, "
        f"{len(current_issues)} current issues, {len(new_issues)} new issues"
    )

    return new_issues


def _log_issues(
    issues: list[dict[str, Any]], listener: EventListener, limit: int = 10
) -> None:
    """Log validation issues."""
    for issue in issues[:limit]:
        sev = issue.get("severity", "?")
        rule = issue.get("rule", "?")
        msg = issue.get("message", "?")
        at = issue.get("at", "stage-level")
        listener.warning(f"  [{sev}] {rule}: {msg} (at {at})")
    if len(issues) > limit:
        listener.warning(f"  ... and {len(issues) - limit} more")


class ValidateUSDTask(Task):
    """Task to validate a single USD file using NVIDIA USD Validation.

    Used for pre-validation (validate_input step) to check if the input asset
    has existing validation problems before the pipeline runs.

    The ``on_failure`` setting controls behavior when issues are found:

    - ``warn``  -- Log warnings and continue (default).
    - ``block`` -- Fail the pipeline if the asset is invalid.
    - ``fix``   -- Run local auto-fix with ``usd-validation-nvidia``. If the
      fix succeeds the fixed USD replaces the input for downstream steps.
      If the fix fails (no fixed stage written, or fixed stage still has
      issues) the pipeline is blocked.

    Input context keys:
        - input_usd_path: Path to the USD file to validate
        - validation_config: Dict with validation parameters
        - on_failure: "warn" | "block" | "fix" (default "warn")

    Output context keys:
        - validation_result: Full validation result dict
        - validation_summary: Summary with counts
        - validation_issues: List of individual validation issues
        - validation_is_valid: Boolean indicating if asset is valid
        - validation_success: Boolean indicating the API call succeeded
        - validation_fixed_usd_path: (fix mode only) Path to the fixed USD
    """

    def run(self, context: dict[str, Any], object_store: Any = None) -> dict[str, Any]:
        """Execute USD validation synchronously."""
        import asyncio

        return asyncio.run(self.arun(context, object_store))

    async def arun(
        self, context: dict[str, Any], object_store: Any = None
    ) -> dict[str, Any]:
        """Execute USD validation asynchronously."""
        listener = get_listener(context)

        input_usd = context.get("input_usd_path")
        validation_config = context.get("validation_config", {})
        on_failure = context.get("on_failure", "warn")

        if not input_usd:
            raise ValueError("input_usd_path is required in context")
        if on_failure not in ON_FAILURE_MODES:
            raise ValueError(
                f"Invalid on_failure mode: {on_failure!r}. "
                f"Must be one of {ON_FAILURE_MODES}"
            )

        input_usd = Path(input_usd)
        listener.info("Pre-validation: checking input asset for existing issues")
        listener.info(f"On failure mode: {on_failure}")

        categories = validation_config.get("categories", [])
        if categories:
            listener.info(f"Categories: {', '.join(categories)}")

        # In fix mode, we need a path for the fixed stage
        wants_fix = on_failure == "fix"
        output_dir = context.get("output_dir")
        fixed_path = None
        if wants_fix and output_dir:
            fixed_path = _fixed_usd_path(input_usd, output_dir)
            if fixed_path.exists() and fixed_path.is_file():
                fixed_path.unlink()

        try:
            result = await _run_validation(
                usd_path=input_usd,
                validation_config=validation_config,
                listener=listener,
                label="input",
                fix=wants_fix,
                output_path=fixed_path,
            )

            summary = result.get("summary", {})
            issues = result.get("issues", [])
            is_valid = summary.get("is_valid", True)

            context["validation_result"] = result
            context["validation_summary"] = summary
            context["validation_issues"] = issues
            context["validation_is_valid"] = is_valid
            context["validation_success"] = True

            # Handle failure modes
            if not is_valid:
                _log_issues(issues, listener)

                if on_failure == "block":
                    raise RuntimeError(
                        f"Input asset validation failed with {len(issues)} issue(s). "
                        f"Pipeline blocked by on_failure=block."
                    )

                elif on_failure == "fix":
                    # Check if fix was applied and saved
                    if fixed_path and fixed_path.exists():
                        # Re-validate the fixed file to confirm issues resolved
                        listener.info("Fix applied. Re-validating fixed USD...")
                        fixed_result = await _run_validation(
                            usd_path=fixed_path,
                            validation_config=validation_config,
                            listener=listener,
                            label="fixed input",
                        )
                        fixed_summary = fixed_result.get("summary", {})
                        if not fixed_summary.get("is_valid", True):
                            remaining = fixed_summary.get("total_issues", 0)
                            raise RuntimeError(
                                f"Auto-fix applied but fixed USD still has "
                                f"{remaining} issue(s). Pipeline blocked."
                            )
                        listener.info(f"Fixed USD is valid. Using: {fixed_path}")
                        context["validation_fixed_usd_path"] = str(fixed_path)
                        # Downstream USD tasks consume input_usd_path, so route
                        # the pipeline to the repaired asset after validation.
                        context["input_usd_path"] = str(fixed_path)
                    elif not fixed_path:
                        raise RuntimeError(
                            "on_failure=fix but no output_dir configured "
                            "to save fixed stage. Pipeline blocked."
                        )
                    else:
                        raise RuntimeError(
                            f"Auto-fix requested but validator did not write "
                            f"a fixed stage ({len(issues)} issues remain). "
                            f"Pipeline blocked."
                        )

                else:  # warn
                    listener.warning(
                        f"Input asset has {len(issues)} validation issue(s). "
                        f"Continuing (on_failure=warn)."
                    )

            # Save validation report
            if output_dir:
                report_path = Path(output_dir) / "validation_report.json"
                report_path.parent.mkdir(parents=True, exist_ok=True)
                report: dict[str, Any] = {
                    "input_usd": str(input_usd),
                    "on_failure": on_failure,
                    "summary": summary,
                    "categories_checked": result.get("categories_checked", []),
                    "issues": issues,
                    "fixes": result.get("fixes", []),
                }
                if context.get("validation_fixed_usd_path"):
                    report["fixed_usd_path"] = context["validation_fixed_usd_path"]
                with open(report_path, "w", encoding="utf-8") as f:
                    json.dump(report, f, indent=2)
                listener.info(f"Saved validation report to: {report_path}")

            listener.info("Pre-validation step completed")

        except ImportError as e:
            if _is_usd_validator_import_error(e) and _mark_validation_skipped(
                context, listener, on_failure, e
            ):
                return context
            listener.error(f"USD validation failed: {e}")
            context["validation_success"] = False
            context["validation_error"] = str(e)
            raise
        except Exception as e:
            listener.error(f"USD validation failed: {e}")
            context["validation_success"] = False
            context["validation_error"] = str(e)
            raise

        return context


class ValidateOutputUSDTask(Task):
    """Task to validate the output USD and compare against the input baseline.

    Validates BOTH the original input and the output USD, then compares to
    detect regressions. Self-contained — does not require validate_input.

    The ``on_failure`` setting controls behavior when **new** issues are found:

    - ``warn``  -- Log warnings and continue (default).
    - ``block`` -- Fail the pipeline if regressions are detected.

    Input context keys:
        - input_usd_path: Path to the output USD to validate
        - original_usd_path: Path to the original input USD for baseline
        - validation_config: Dict with validation parameters
        - on_failure: "warn" | "block" (default "warn")
        - baseline_validation: (optional) Cached baseline from validate_input

    Output context keys:
        - validation_result: Full validation result for the output USD
        - validation_summary: Output USD summary
        - validation_issues: Output USD issues
        - validation_is_valid: Whether the output USD is valid
        - validation_success: Whether the API call succeeded
        - validation_baseline_result: Baseline validation result (input USD)
        - validation_baseline_summary: Baseline summary
        - validation_new_issues: Issues not in input baseline
        - validation_regression: True if new issues were introduced
    """

    def run(self, context: dict[str, Any], object_store: Any = None) -> dict[str, Any]:
        """Execute USD output validation synchronously."""
        import asyncio

        return asyncio.run(self.arun(context, object_store))

    async def arun(
        self, context: dict[str, Any], object_store: Any = None
    ) -> dict[str, Any]:
        """Execute USD output validation asynchronously."""
        listener = get_listener(context)

        output_usd = context.get("input_usd_path")
        original_usd = context.get("original_usd_path")
        validation_config = context.get("validation_config", {})
        cached_baseline = context.get("baseline_validation")
        on_failure = context.get("on_failure", "warn")

        if not output_usd:
            raise ValueError("input_usd_path is required in context")
        if on_failure not in ON_FAILURE_MODES_OUTPUT:
            raise ValueError(
                f"Invalid on_failure mode for validate_output: {on_failure!r}. "
                f"Must be one of {ON_FAILURE_MODES_OUTPUT}"
            )

        output_usd = Path(output_usd)

        listener.info("Post-validation: validating output and comparing with input")
        listener.info(f"On failure mode: {on_failure}")

        categories = validation_config.get("categories", [])
        if categories:
            listener.info(f"Categories: {', '.join(categories)}")

        output_dir = context.get("output_dir")

        try:
            # Step 1: Get baseline (input USD validation)
            if cached_baseline:
                listener.info(
                    "Using cached baseline from validate_input step "
                    f"({len(cached_baseline.get('issues', []))} issues)"
                )
                baseline_result = cached_baseline
            elif original_usd:
                original_usd = Path(original_usd)
                listener.info(
                    f"Validating original input USD for baseline: {original_usd}"
                )
                baseline_result = await _run_validation(
                    usd_path=original_usd,
                    validation_config=validation_config,
                    listener=listener,
                    label="input (baseline)",
                )
            else:
                listener.warning(
                    "No original_usd_path or baseline_validation provided. "
                    "Cannot compare — will only validate output."
                )
                baseline_result = None

            # Step 2: Validate output USD
            output_result = await _run_validation(
                usd_path=output_usd,
                validation_config=validation_config,
                listener=listener,
                label="output",
            )

            output_summary = output_result.get("summary", {})
            output_issues = output_result.get("issues", [])
            is_valid = output_summary.get("is_valid", True)

            context["validation_result"] = output_result
            context["validation_summary"] = output_summary
            context["validation_issues"] = output_issues
            context["validation_is_valid"] = is_valid
            context["validation_success"] = True

            # Step 3: Compare if baseline is available
            new_issues: list[dict[str, Any]] = []
            has_regression = False

            if baseline_result:
                baseline_issues = baseline_result.get("issues", [])
                baseline_summary = baseline_result.get("summary", {})

                context["validation_baseline_result"] = baseline_result
                context["validation_baseline_summary"] = baseline_summary

                new_issues = _compare_issues(
                    baseline_issues=baseline_issues,
                    current_issues=output_issues,
                    listener=listener,
                )
                has_regression = len(new_issues) > 0
                context["validation_new_issues"] = new_issues
                context["validation_regression"] = has_regression

                if has_regression:
                    listener.warning(
                        f"REGRESSION: {len(new_issues)} new validation issue(s) "
                        f"introduced by material assignment!"
                    )
                    _log_issues(new_issues, listener)
                else:
                    listener.info(
                        f"No new validation issues introduced "
                        f"(input: {len(baseline_issues)}, "
                        f"output: {len(output_issues)})"
                    )

            # Step 4: Block if regression detected and on_failure=block
            if has_regression and on_failure == "block":
                raise RuntimeError(
                    f"Output validation regression: {len(new_issues)} new "
                    f"issue(s) introduced. Pipeline blocked by "
                    f"on_failure=block."
                )

            # Save validation report
            if output_dir:
                report_path = Path(output_dir) / "validation_report.json"
                report_path.parent.mkdir(parents=True, exist_ok=True)
                report: dict[str, Any] = {
                    "output_usd": str(output_usd),
                    "original_usd": str(original_usd) if original_usd else None,
                    "on_failure": on_failure,
                    "output_summary": output_summary,
                    "output_issues": output_issues,
                    "categories_checked": output_result.get("categories_checked", []),
                }
                if baseline_result:
                    report["input_summary"] = baseline_result.get("summary", {})
                    report["input_issues"] = baseline_result.get("issues", [])
                    report["new_issues"] = new_issues
                    report["regression"] = has_regression

                with open(report_path, "w", encoding="utf-8") as f:
                    json.dump(report, f, indent=2)
                listener.info(f"Saved validation report to: {report_path}")

            listener.info("Post-validation step completed")

        except ImportError as e:
            if _is_usd_validator_import_error(e) and _mark_validation_skipped(
                context, listener, on_failure, e
            ):
                return context
            listener.error(f"USD output validation failed: {e}")
            context["validation_success"] = False
            context["validation_error"] = str(e)
            raise
        except Exception as e:
            listener.error(f"USD output validation failed: {e}")
            context["validation_success"] = False
            context["validation_error"] = str(e)
            raise

        return context
