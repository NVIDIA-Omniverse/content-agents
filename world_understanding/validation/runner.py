# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runner extension contracts for Validation Agent V1."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from world_understanding.validation.models import (
    ValidationPlan,
    ValidationRequest,
    ValidationTemplateResult,
)
from world_understanding.validation.templates import (
    ValidationTemplateDefinition,
)


@dataclass(frozen=True)
class ValidationTemplateContext:
    """Context passed to a concrete validation template implementation."""

    request: ValidationRequest
    plan: ValidationPlan
    working_dir: Path
    previous_template_results: tuple[ValidationTemplateResult, ...] = ()


class ValidationTemplate(Protocol):
    """Protocol for concrete V1 validation template implementations."""

    @property
    def definition(self) -> ValidationTemplateDefinition:
        """Static template contract metadata."""

    def run(self, context: ValidationTemplateContext) -> ValidationTemplateResult:
        """Run the template and return a stable template result."""
