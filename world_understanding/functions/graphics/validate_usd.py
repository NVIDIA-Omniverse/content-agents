# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""USD asset validation using the standalone omni.asset_validator library.

This module provides local validation without requiring Kit or NVCF.
It uses the omniverse-asset-validator pip package which runs purely on
Python + OpenUSD.

Install: uv pip install omniverse-asset-validator
"""

import logging
import sys
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default validation categories matching omni.asset_validator rule categories
DEFAULT_VALIDATION_CATEGORIES = [
    "Basic",
    "Usd:Performance",
    "Usd:Schema",
    "Omni:Material",
    "Omni:Layout",
    "Omni:Basic",
    "Omni:Geometry",
]


def _ensure_usd_validation_compat() -> None:
    """Apply compatibility shim for usd-core versions with broken UsdValidation bindings.

    usd-core 25.11 has a C++ binding bug in pxr.UsdValidation that crashes
    on import. The omni.asset_validator library catches ImportError but not
    TypeError, so we pre-inject a stub module to avoid the crash.
    """
    if "pxr.UsdValidation" in sys.modules:
        return

    try:
        from pxr import UsdValidation  # noqa: F401
    except (ImportError, TypeError):
        # Create a stub that makes UsdValidatorAdapter.__contains__ return False
        # This disables the UsdValidation-based rules but keeps all other rules working
        logger.debug(
            "pxr.UsdValidation not available (broken bindings or missing). "
            "Using stub — UsdValidation-based rules will be skipped."
        )

        class _StubUsdValidation:
            class ValidationRegistry:
                def GetOrLoadValidatorByName(self, _name: str) -> None:
                    return None

        sys.modules["pxr.UsdValidation"] = _StubUsdValidation()  # type: ignore[assignment]


def is_available() -> bool:
    """Check if the standalone validator is available.

    Returns:
        True if omniverse-asset-validator is installed and importable
    """
    try:
        _ensure_usd_validation_compat()
        from omni.asset_validator import ValidationEngine  # noqa: F401

        return True
    except ImportError:
        return False


def validate_usd(
    input_path: Path | str,
    categories: list[str] | None = None,
    fix: bool = False,
    output_path: Path | str | None = None,
    stage_timeout: float = 180.0,
) -> dict[str, Any]:
    """Validate a USD file using the standalone omni.asset_validator library.

    Args:
        input_path: Path to the USD file to validate
        categories: Validation categories to check (default: all)
        fix: Attempt to auto-fix issues
        output_path: Path to save fixed stage (only used when fix=True)
        stage_timeout: Not used in standalone mode (kept for API compat)

    Returns:
        Dict matching the NVCF /validate response format:
            - status: "success" or "error"
            - validation_time: Time in seconds
            - issues: List of issue dicts
            - summary: Summary dict with counts
            - categories_checked: List of categories checked
            - fixes: List of fix result dicts (empty when fix=False)

    Raises:
        ImportError: If omniverse-asset-validator is not installed
        ValueError: If input file does not exist
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise ValueError(f"Input file does not exist: {input_path}")

    _ensure_usd_validation_compat()

    from omni.asset_validator import IssueFixer, ValidationEngine

    start_time = time.time()

    logger.info("Validating USD locally: %s", input_path)
    if categories:
        logger.info("Categories: %s", ", ".join(categories))

    try:
        engine = ValidationEngine()
        results = engine.validate(str(input_path))

        # Collect all issues
        all_issues = list(results.issues())

        # Filter by category if specified.
        # Issues with unmapped rules ("Unknown" category) are always included
        # to avoid silently dropping findings from new/unmapped checkers.
        if categories:
            cat_set = set(categories)
            filtered_issues = [
                issue
                for issue in all_issues
                if _infer_category(issue) in cat_set
                or _infer_category(issue) == "Unknown"
            ]
        else:
            filtered_issues = all_issues
            categories = list(DEFAULT_VALIDATION_CATEGORIES)

        # Build structured issue list
        issues_list = []
        severity_counts = {"failures": 0, "warnings": 0, "errors": 0}

        for issue in filtered_issues:
            severity = _map_severity(issue.severity)
            severity_counts[f"{severity}s"] = severity_counts.get(f"{severity}s", 0) + 1

            issue_dict: dict[str, Any] = {
                "severity": severity,
                "rule": _get_rule_name(issue),
                "category": _infer_category(issue),
                "message": str(issue.message),
                "at": str(issue.at) if issue.at else None,
                "suggestion": str(issue.suggestion) if issue.suggestion else None,
            }
            issues_list.append(issue_dict)

        is_valid = severity_counts["failures"] == 0 and severity_counts["errors"] == 0

        # Handle fix mode
        fixes_list: list[dict[str, Any]] = []
        if fix and filtered_issues:
            logger.info("Attempting to fix %d issue(s)...", len(filtered_issues))
            fixer = IssueFixer()
            fix_results = fixer.fix(filtered_issues)

            for fix_result in fix_results:
                fixes_list.append(
                    {
                        "rule": _get_rule_name(fix_result.issue),
                        "status": fix_result.status.name.lower(),
                        "message": str(fix_result.message)
                        if fix_result.message
                        else None,
                    }
                )

            # Save fixed stage if output_path provided
            if output_path and results.stage:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                results.stage.GetRootLayer().Export(str(output_path))
                logger.info("Saved fixed stage to: %s", output_path)

        validation_time = time.time() - start_time
        logger.info(
            "Local validation completed in %.2fs: %d issues",
            validation_time,
            len(issues_list),
        )

        return {
            "status": "success",
            "validation_time": validation_time,
            "issues": issues_list,
            "summary": {
                "total_issues": len(issues_list),
                "failures": severity_counts["failures"],
                "warnings": severity_counts["warnings"],
                "errors": severity_counts["errors"],
                "is_valid": is_valid,
            },
            "categories_checked": categories,
            "fixes": fixes_list,
        }

    except Exception as e:
        validation_time = time.time() - start_time
        logger.exception("Local validation failed")
        return {
            "status": "error",
            "validation_time": validation_time,
            "error": str(e),
        }


def _map_severity(severity: Any) -> str:
    """Map omni.asset_validator severity enum to string."""
    name = str(severity.name).lower() if hasattr(severity, "name") else str(severity)
    if "failure" in name:
        return "failure"
    elif "warning" in name:
        return "warning"
    elif "error" in name:
        return "error"
    return "warning"


def _get_rule_name(issue: Any) -> str:
    """Extract human-readable rule name from issue."""
    rule = getattr(issue, "rule", None)
    if rule is None:
        return "Unknown"
    # rule is typically a class like <class 'omni.asset_validator._geometry_checker.IndexedPrimvarChecker'>
    name = str(rule)
    if "." in name and "'" in name:
        # Extract class name from repr
        name = name.split("'")[1].split(".")[-1]
    return name


def _infer_category(issue: Any) -> str:
    """Infer the category for an issue based on its rule checker class."""
    rule_name = _get_rule_name(issue)

    # Map known rule names to categories.
    # NOTE: Update this when omni.asset_validator adds new rules.
    # Unmapped rules get "Unknown" category and are always included in results.
    category_map = {
        "StageMetadataChecker": "Basic",
        "MissingReferenceChecker": "Basic",
        "DefaultPrimChecker": "Basic",
        "TypeChecker": "Omni:Basic",
        "MaterialPathChecker": "Omni:Material",
        "MaterialBindingChecker": "Omni:Material",
        "OmniOrphanedPrimChecker": "Omni:Layout",
        "OmniDefaultPrimChecker": "Omni:Layout",
        "IndexedPrimvarChecker": "Omni:Geometry",
        "NormalsChecker": "Omni:Geometry",
        "ExtentChecker": "Usd:Schema",
        "SubdivisionSchemaChecker": "Usd:Schema",
        "UpAxisChecker": "Usd:Performance",
    }

    return category_map.get(rule_name, "Unknown")
