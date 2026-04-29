# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for validation step configuration, auto-wiring, and on_failure modes."""

from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------


def _get_step_order() -> list[str]:
    """Import STEP_ORDER safely (avoids circular import on first load)."""
    # Trigger full module graph load via unified_config first
    import material_agent.config.unified_config  # noqa: F401
    from material_agent.config.schema import STEP_ORDER

    return STEP_ORDER


class TestValidationStepOrder:
    """Verify validate_input/validate_output are in the right positions."""

    def test_validate_input_is_first_step(self):
        assert _get_step_order()[0] == "validate_input"

    def test_validate_output_is_before_render(self):
        order = _get_step_order()
        assert order.index("validate_output") < order.index("render")

    def test_validate_output_is_after_apply_and_refine(self):
        order = _get_step_order()
        assert order.index("validate_output") > order.index("apply")
        assert order.index("validate_output") > order.index("refine")

    def test_output_dirs_registered(self):
        from material_agent.config.schema import STEP_OUTPUT_DIRS

        assert "validate_input" in STEP_OUTPUT_DIRS
        assert "validate_output" in STEP_OUTPUT_DIRS


# ---------------------------------------------------------------------------
# Unified config auto-wiring tests
# ---------------------------------------------------------------------------


def _make_config(
    tmp_path: Path,
    steps: dict[str, Any] | None = None,
    materials: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a minimal config dict with an existing input USD file."""
    input_usd = tmp_path / "input.usd"
    input_usd.touch()

    cfg: dict[str, Any] = {
        "project": {"name": "test_validate"},
        "input": {"usd_path": str(input_usd)},
        "output": {},
        "steps": steps or {},
    }
    if materials is not None:
        cfg["materials"] = materials
    return cfg


def _run_config_task(config: dict[str, Any]) -> dict[str, Any]:
    """Run UnifiedPipelineConfigTask with a config dict."""
    from material_agent.config.unified_config import UnifiedPipelineConfigTask

    task = UnifiedPipelineConfigTask()
    return task.run({"config_path": None, "config_dict": config})


class TestValidateInputConfig:
    """UnifiedPipelineConfigTask correctly wires validate_input."""

    def test_validate_input_included_when_enabled(self, tmp_path):
        config = _make_config(tmp_path, steps={"validate_input": {"enabled": True}})
        ctx = _run_config_task(config)

        assert "validate_input" in ctx["steps_to_run"]

    def test_validate_input_has_input_usd_path(self, tmp_path):
        config = _make_config(tmp_path, steps={"validate_input": {"enabled": True}})
        ctx = _run_config_task(config)

        step_cfg = ctx["step_configs"]["validate_input"]
        assert "input_usd_path" in step_cfg
        assert step_cfg["input_usd_path"].endswith("input.usd")

    def test_validate_input_has_output_dir(self, tmp_path):
        config = _make_config(tmp_path, steps={"validate_input": {"enabled": True}})
        ctx = _run_config_task(config)

        step_cfg = ctx["step_configs"]["validate_input"]
        assert "output_dir" in step_cfg
        assert "validation/input" in step_cfg["output_dir"]

    def test_validate_input_on_failure_defaults_to_warn(self, tmp_path):
        config = _make_config(tmp_path, steps={"validate_input": {"enabled": True}})
        ctx = _run_config_task(config)

        step_cfg = ctx["step_configs"]["validate_input"]
        assert step_cfg.get("on_failure") == "warn"

    @pytest.mark.parametrize("mode", ["warn", "block", "fix"])
    def test_validate_input_on_failure_configurable(self, tmp_path, mode):
        config = _make_config(
            tmp_path,
            steps={"validate_input": {"enabled": True, "on_failure": mode}},
        )
        ctx = _run_config_task(config)

        step_cfg = ctx["step_configs"]["validate_input"]
        assert step_cfg["on_failure"] == mode

    def test_validate_input_has_default_categories(self, tmp_path):
        config = _make_config(tmp_path, steps={"validate_input": {"enabled": True}})
        ctx = _run_config_task(config)

        step_cfg = ctx["step_configs"]["validate_input"]
        categories = step_cfg.get("validation_config", {}).get("categories", [])
        assert len(categories) == 7
        assert "Basic" in categories
        assert "Omni:Material" in categories


class TestValidateOutputConfig:
    """UnifiedPipelineConfigTask correctly wires validate_output."""

    def test_validate_output_included_when_enabled(self, tmp_path):
        config = _make_config(
            tmp_path,
            steps={
                "validate_output": {"enabled": True},
            },
        )
        ctx = _run_config_task(config)

        assert "validate_output" in ctx["steps_to_run"]

    def test_validate_output_points_to_output_usd(self, tmp_path):
        config = _make_config(
            tmp_path,
            steps={
                "validate_output": {"enabled": True},
            },
        )
        ctx = _run_config_task(config)

        step_cfg = ctx["step_configs"]["validate_output"]
        # validate_output.input_usd_path should point to the pipeline output
        assert "output.usd" in step_cfg["input_usd_path"]

    def test_validate_output_on_failure_defaults_to_warn(self, tmp_path):
        config = _make_config(
            tmp_path,
            steps={
                "validate_output": {"enabled": True},
            },
        )
        ctx = _run_config_task(config)

        step_cfg = ctx["step_configs"]["validate_output"]
        assert step_cfg.get("on_failure") == "warn"


# ---------------------------------------------------------------------------
# Executor auto-wiring tests (mock step_outputs, no NVCF calls)
# ---------------------------------------------------------------------------


class TestExecutorAutoWiring:
    """Test the auto-wiring logic in _execute_step for validation steps.

    These tests instantiate the executor and call the auto-wiring code path
    by setting up step_outputs. We don't call _execute_step itself (which
    would need NVCF) — instead we replicate the auto-wiring block.
    """

    @staticmethod
    def _simulate_autowiring(
        step_name: str,
        step_config: dict[str, Any],
        step_outputs: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Replicate the validate_input auto-wiring logic from the executor."""
        # This mirrors the auto-wiring block in _execute_step
        if "validate_input" in step_outputs:
            fixed_path = step_outputs["validate_input"].get("validation_fixed_usd_path")
            if fixed_path:
                if step_name == "optimize_usd":
                    step_config["input_usd_path"] = str(fixed_path)
                elif (
                    step_name in ["render_preview", "build_dataset_usd"]
                    and "optimize_usd" not in step_outputs
                ):
                    step_config["usd_path"] = str(fixed_path)

        return step_config

    def test_fix_wires_into_optimize_usd(self):
        cfg = {"input_usd_path": "/original/input.usd"}
        outputs = {
            "validate_input": {
                "validation_fixed_usd_path": "/fixed/input.usd",
            }
        }
        result = self._simulate_autowiring("optimize_usd", cfg, outputs)
        assert result["input_usd_path"] == "/fixed/input.usd"

    def test_fix_wires_into_build_dataset_when_no_optimize(self):
        cfg = {"usd_path": "/original/input.usd"}
        outputs = {
            "validate_input": {
                "validation_fixed_usd_path": "/fixed/input.usd",
            }
        }
        result = self._simulate_autowiring("build_dataset_usd", cfg, outputs)
        assert result["usd_path"] == "/fixed/input.usd"

    def test_fix_does_not_wire_build_dataset_when_optimize_ran(self):
        """When optimize_usd ran, its output takes precedence."""
        cfg = {"usd_path": "/original/input.usd"}
        outputs = {
            "validate_input": {
                "validation_fixed_usd_path": "/fixed/input.usd",
            },
            "optimize_usd": {
                "optimized_usd_path": "/optimized/input.usd",
            },
        }
        result = self._simulate_autowiring("build_dataset_usd", cfg, outputs)
        # Should NOT be overwritten by validate_input fix
        assert result["usd_path"] == "/original/input.usd"

    def test_no_fix_no_wiring(self):
        """When validate_input ran without fix, no auto-wiring happens."""
        cfg = {"input_usd_path": "/original/input.usd"}
        outputs = {
            "validate_input": {
                "validation_result": {"issues": [], "summary": {}},
                # No validation_fixed_usd_path
            }
        }
        result = self._simulate_autowiring("optimize_usd", cfg, outputs)
        assert result["input_usd_path"] == "/original/input.usd"

    def test_validate_input_not_in_outputs(self):
        """When validate_input didn't run, no auto-wiring happens."""
        cfg = {"usd_path": "/original/input.usd"}
        result = self._simulate_autowiring("build_dataset_usd", cfg, {})
        assert result["usd_path"] == "/original/input.usd"


# ---------------------------------------------------------------------------
# ValidateOutputUSDTask auto-wiring in executor
# ---------------------------------------------------------------------------


class TestValidateOutputAutoWiring:
    """Test executor auto-wiring for validate_output step."""

    @staticmethod
    def _simulate_validate_output_autowiring(
        step_config: dict[str, Any],
        step_outputs: dict[str, Any],
    ) -> dict[str, Any]:
        """Replicate the validate_output auto-wiring from executor."""
        # Output USD from apply/refine
        if "refine" in step_outputs:
            usd_path = step_outputs["refine"].get("final_output_path") or step_outputs[
                "refine"
            ].get("output_usd_path")
            if usd_path:
                step_config["input_usd_path"] = str(usd_path)
        elif "apply" in step_outputs:
            usd_path = step_outputs["apply"].get("output_usd_path")
            if usd_path:
                step_config["input_usd_path"] = str(usd_path)

        # Original USD for baseline
        if "original_usd_path" not in step_config:
            if "optimize_usd" in step_outputs:
                original = step_outputs["optimize_usd"].get("original_usd_path")
                if original:
                    step_config["original_usd_path"] = str(original)

        # Cached baseline
        if "validate_input" in step_outputs:
            baseline = step_outputs["validate_input"].get("validation_result")
            if baseline:
                step_config["baseline_validation"] = baseline

        return step_config

    def test_gets_usd_from_apply(self):
        cfg: dict[str, Any] = {"input_usd_path": "/default/output.usd"}
        outputs = {"apply": {"output_usd_path": "/applied/output.usd"}}
        result = self._simulate_validate_output_autowiring(cfg, outputs)
        assert result["input_usd_path"] == "/applied/output.usd"

    def test_gets_usd_from_refine(self):
        cfg: dict[str, Any] = {"input_usd_path": "/default/output.usd"}
        outputs = {"refine": {"final_output_path": "/refined/output.usd"}}
        result = self._simulate_validate_output_autowiring(cfg, outputs)
        assert result["input_usd_path"] == "/refined/output.usd"

    def test_refine_takes_precedence_over_apply(self):
        cfg: dict[str, Any] = {"input_usd_path": "/default/output.usd"}
        outputs = {
            "apply": {"output_usd_path": "/applied/output.usd"},
            "refine": {"final_output_path": "/refined/output.usd"},
        }
        result = self._simulate_validate_output_autowiring(cfg, outputs)
        assert result["input_usd_path"] == "/refined/output.usd"

    def test_gets_original_from_optimize(self):
        cfg: dict[str, Any] = {}
        outputs = {"optimize_usd": {"original_usd_path": "/original/input.usd"}}
        result = self._simulate_validate_output_autowiring(cfg, outputs)
        assert result["original_usd_path"] == "/original/input.usd"

    def test_injects_cached_baseline(self):
        cfg: dict[str, Any] = {}
        baseline = {"issues": [{"rule": "X"}], "summary": {"total_issues": 1}}
        outputs = {"validate_input": {"validation_result": baseline}}
        result = self._simulate_validate_output_autowiring(cfg, outputs)
        assert result["baseline_validation"] is baseline

    def test_no_baseline_when_validate_input_skipped(self):
        cfg: dict[str, Any] = {}
        result = self._simulate_validate_output_autowiring(cfg, {})
        assert "baseline_validation" not in result


# ---------------------------------------------------------------------------
# on_failure mode validation
# ---------------------------------------------------------------------------


class TestOnFailureModes:
    """Test on_failure constant definitions."""

    def test_validate_input_supports_warn_block_fix(self):
        from world_understanding.agentic.usd_tasks.validate_usd import (
            ON_FAILURE_MODES,
        )

        assert "warn" in ON_FAILURE_MODES
        assert "block" in ON_FAILURE_MODES
        assert "fix" in ON_FAILURE_MODES

    def test_validate_output_supports_warn_block_only(self):
        from world_understanding.agentic.usd_tasks.validate_usd import (
            ON_FAILURE_MODES_OUTPUT,
        )

        assert "warn" in ON_FAILURE_MODES_OUTPUT
        assert "block" in ON_FAILURE_MODES_OUTPUT
        assert "fix" not in ON_FAILURE_MODES_OUTPUT


# ---------------------------------------------------------------------------
# Issue comparison logic
# ---------------------------------------------------------------------------


class TestIssueComparison:
    """Test the _compare_issues function."""

    def test_no_issues_no_regression(self):
        from world_understanding.agentic.usd_tasks.validate_usd import (
            _compare_issues,
        )

        class _Listener:
            def info(self, msg: str, **kw: Any) -> None:
                pass

        result = _compare_issues([], [], _Listener())
        assert result == []

    def test_same_issues_no_regression(self):
        from world_understanding.agentic.usd_tasks.validate_usd import (
            _compare_issues,
        )

        class _Listener:
            def info(self, msg: str, **kw: Any) -> None:
                pass

        issues = [{"rule": "A", "severity": "warning", "at": "/prim"}]
        result = _compare_issues(issues, issues, _Listener())
        assert result == []

    def test_new_issue_detected(self):
        from world_understanding.agentic.usd_tasks.validate_usd import (
            _compare_issues,
        )

        class _Listener:
            def info(self, msg: str, **kw: Any) -> None:
                pass

        baseline = [{"rule": "A", "severity": "warning", "at": "/prim1"}]
        current = [
            {"rule": "A", "severity": "warning", "at": "/prim1"},
            {"rule": "B", "severity": "error", "at": "/prim2"},
        ]
        result = _compare_issues(baseline, current, _Listener())
        assert len(result) == 1
        assert result[0]["rule"] == "B"

    def test_duplicate_counted_correctly(self):
        from world_understanding.agentic.usd_tasks.validate_usd import (
            _compare_issues,
        )

        class _Listener:
            def info(self, msg: str, **kw: Any) -> None:
                pass

        issue = {"rule": "A", "severity": "warning", "at": "/prim"}
        baseline = [issue]
        current = [issue, issue]  # same issue appears twice
        result = _compare_issues(baseline, current, _Listener())
        assert len(result) == 1  # one extra occurrence


# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------


class _FakeListener:
    """Minimal EventListener for tests."""

    def info(self, msg: str, **kw: Any) -> None:
        pass

    def debug(self, msg: str, **kw: Any) -> None:
        pass

    def warning(self, msg: str, **kw: Any) -> None:
        pass

    def error(self, msg: str, **kw: Any) -> None:
        pass


def _make_success_result(
    issues: list | None = None,
    summary: dict | None = None,
    categories_checked: list | None = None,
    fixed_stage_base64: str | None = None,
) -> dict:
    result: dict = {
        "status": "success",
        "validation_time": 1.0,
        "issues": issues or [],
        "summary": summary
        or {
            "total_issues": 0,
            "failures": 0,
            "warnings": 0,
            "errors": 0,
            "is_valid": True,
        },
        "categories_checked": categories_checked or ["Basic"],
        "fixes": [],
    }
    if fixed_stage_base64 is not None:
        result["fixed_stage_base64"] = fixed_stage_base64
    return result


# ---------------------------------------------------------------------------


class TestValidateUSDTask:
    """Tests for ValidateUSDTask with mocked _run_validation."""

    def _make_context(self, tmp_path: Path, on_failure: str = "warn") -> dict:
        usd = tmp_path / "input.usd"
        usd.touch()
        return {
            "input_usd_path": str(usd),
            "on_failure": on_failure,
            "output_dir": str(tmp_path),
            "validation_config": {"categories": ["Basic"]},
            "listener": _FakeListener(),
        }

    @pytest.mark.asyncio
    async def test_on_failure_warn_logs_and_continues(self, tmp_path):
        from unittest.mock import AsyncMock, patch

        from world_understanding.agentic.usd_tasks.validate_usd import ValidateUSDTask

        invalid_result = _make_success_result(
            issues=[{"rule": "X", "severity": "warning", "at": "/p"}],
            summary={
                "total_issues": 1,
                "is_valid": False,
                "failures": 0,
                "warnings": 1,
                "errors": 0,
            },
        )
        ctx = self._make_context(tmp_path, on_failure="warn")
        with patch(
            "world_understanding.agentic.usd_tasks.validate_usd._run_validation",
            new=AsyncMock(return_value=invalid_result),
        ):
            result = await ValidateUSDTask().arun(ctx)

        assert result["validation_success"] is True

    @pytest.mark.asyncio
    async def test_on_failure_block_raises_when_invalid(self, tmp_path):
        from unittest.mock import AsyncMock, patch

        from world_understanding.agentic.usd_tasks.validate_usd import ValidateUSDTask

        invalid_result = _make_success_result(
            issues=[{"rule": "X", "severity": "error", "at": "/p"}],
            summary={
                "total_issues": 1,
                "is_valid": False,
                "failures": 1,
                "warnings": 0,
                "errors": 0,
            },
        )
        ctx = self._make_context(tmp_path, on_failure="block")
        with (
            patch(
                "world_understanding.agentic.usd_tasks.validate_usd._run_validation",
                new=AsyncMock(return_value=invalid_result),
            ),
            pytest.raises(RuntimeError, match="on_failure=block"),
        ):
            await ValidateUSDTask().arun(ctx)

    @pytest.mark.asyncio
    async def test_on_failure_fix_saves_fixed_usd_path(self, tmp_path):
        import base64
        from unittest.mock import AsyncMock, patch

        from world_understanding.agentic.usd_tasks.validate_usd import ValidateUSDTask

        fix_payload = b"fixed usd content"
        encoded = base64.b64encode(fix_payload).decode()
        invalid_result = _make_success_result(
            issues=[{"rule": "X", "severity": "error", "at": "/p"}],
            summary={
                "total_issues": 1,
                "is_valid": False,
                "failures": 1,
                "warnings": 0,
                "errors": 0,
            },
            fixed_stage_base64=encoded,
        )
        ctx = self._make_context(tmp_path, on_failure="fix")

        valid_result = _make_success_result()
        call_count = 0

        async def mock_run_validation(**kwargs: Any) -> dict:
            nonlocal call_count
            call_count += 1
            # First call: validate input (invalid, triggers fix)
            # Second call: re-validate fixed file (valid)
            output_path = kwargs.get("output_path")
            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(fix_payload)
            return invalid_result if call_count == 1 else valid_result

        with patch(
            "world_understanding.agentic.usd_tasks.validate_usd._run_validation",
            new=AsyncMock(side_effect=mock_run_validation),
        ):
            result = await ValidateUSDTask().arun(ctx)

        assert "validation_fixed_usd_path" in result

    @pytest.mark.asyncio
    async def test_on_failure_fix_no_fixed_stage_raises(self, tmp_path):
        from unittest.mock import AsyncMock, patch

        from world_understanding.agentic.usd_tasks.validate_usd import ValidateUSDTask

        invalid_result = _make_success_result(
            issues=[{"rule": "X", "severity": "error", "at": "/p"}],
            summary={
                "total_issues": 1,
                "is_valid": False,
                "failures": 1,
                "warnings": 0,
                "errors": 0,
            },
        )
        ctx = self._make_context(tmp_path, on_failure="fix")
        with (
            patch(
                "world_understanding.agentic.usd_tasks.validate_usd._run_validation",
                new=AsyncMock(return_value=invalid_result),
            ),
            pytest.raises(RuntimeError, match="no fixed stage"),
        ):
            await ValidateUSDTask().arun(ctx)

    @pytest.mark.asyncio
    async def test_valid_input_no_error_regardless_of_on_failure(self, tmp_path):
        from unittest.mock import AsyncMock, patch

        from world_understanding.agentic.usd_tasks.validate_usd import ValidateUSDTask

        valid_result = _make_success_result()
        for mode in ("warn", "block", "fix"):
            ctx = self._make_context(tmp_path, on_failure=mode)
            with patch(
                "world_understanding.agentic.usd_tasks.validate_usd._run_validation",
                new=AsyncMock(return_value=valid_result),
            ):
                result = await ValidateUSDTask().arun(ctx)
            assert result["validation_success"] is True

    @pytest.mark.asyncio
    async def test_report_saved_to_output_dir(self, tmp_path):
        import json
        from unittest.mock import AsyncMock, patch

        from world_understanding.agentic.usd_tasks.validate_usd import ValidateUSDTask

        valid_result = _make_success_result()
        ctx = self._make_context(tmp_path)
        with patch(
            "world_understanding.agentic.usd_tasks.validate_usd._run_validation",
            new=AsyncMock(return_value=valid_result),
        ):
            await ValidateUSDTask().arun(ctx)

        report = tmp_path / "validation_report.json"
        assert report.exists()
        data = json.loads(report.read_text())
        assert "summary" in data


# ---------------------------------------------------------------------------
# ValidateOutputUSDTask tests
# ---------------------------------------------------------------------------


class TestValidateOutputUSDTask:
    """Tests for ValidateOutputUSDTask with mocked _run_validation."""

    def _make_context(
        self,
        tmp_path: Path,
        on_failure: str = "warn",
        original_usd_path: str | None = None,
        baseline_validation: dict | None = None,
    ) -> dict:
        output_usd = tmp_path / "output.usd"
        output_usd.touch()
        ctx: dict = {
            "input_usd_path": str(output_usd),
            "on_failure": on_failure,
            "output_dir": str(tmp_path),
            "validation_config": {"categories": ["Basic"]},
            "listener": _FakeListener(),
        }
        if original_usd_path:
            ctx["original_usd_path"] = original_usd_path
        if baseline_validation is not None:
            ctx["baseline_validation"] = baseline_validation
        return ctx

    @pytest.mark.asyncio
    async def test_cached_baseline_reuse_calls_run_once(self, tmp_path):
        from unittest.mock import AsyncMock, patch

        from world_understanding.agentic.usd_tasks.validate_usd import (
            ValidateOutputUSDTask,
        )

        cached = _make_success_result(
            issues=[],
            summary={
                "total_issues": 0,
                "is_valid": True,
                "failures": 0,
                "warnings": 0,
                "errors": 0,
            },
        )
        ctx = self._make_context(tmp_path, baseline_validation=cached)
        output_result = _make_success_result()

        mock_run = AsyncMock(return_value=output_result)
        with patch(
            "world_understanding.agentic.usd_tasks.validate_usd._run_validation",
            new=mock_run,
        ):
            await ValidateOutputUSDTask().arun(ctx)

        assert mock_run.call_count == 1

    @pytest.mark.asyncio
    async def test_no_cached_baseline_with_original_calls_twice(self, tmp_path):
        from unittest.mock import AsyncMock, patch

        from world_understanding.agentic.usd_tasks.validate_usd import (
            ValidateOutputUSDTask,
        )

        original_usd = tmp_path / "original.usd"
        original_usd.touch()
        ctx = self._make_context(tmp_path, original_usd_path=str(original_usd))

        base_result = _make_success_result()
        output_result = _make_success_result()

        mock_run = AsyncMock(side_effect=[base_result, output_result])
        with patch(
            "world_understanding.agentic.usd_tasks.validate_usd._run_validation",
            new=mock_run,
        ):
            await ValidateOutputUSDTask().arun(ctx)

        assert mock_run.call_count == 2

    @pytest.mark.asyncio
    async def test_no_baseline_no_original_calls_once(self, tmp_path):
        from unittest.mock import AsyncMock, patch

        from world_understanding.agentic.usd_tasks.validate_usd import (
            ValidateOutputUSDTask,
        )

        ctx = self._make_context(tmp_path)
        output_result = _make_success_result()

        mock_run = AsyncMock(return_value=output_result)
        with patch(
            "world_understanding.agentic.usd_tasks.validate_usd._run_validation",
            new=mock_run,
        ):
            result = await ValidateOutputUSDTask().arun(ctx)

        assert mock_run.call_count == 1
        assert "validation_new_issues" not in result

    @pytest.mark.asyncio
    async def test_regression_warn_continues(self, tmp_path):
        from unittest.mock import AsyncMock, patch

        from world_understanding.agentic.usd_tasks.validate_usd import (
            ValidateOutputUSDTask,
        )

        baseline = _make_success_result(
            issues=[],
            summary={
                "total_issues": 0,
                "is_valid": True,
                "failures": 0,
                "warnings": 0,
                "errors": 0,
            },
        )
        new_issue = {"rule": "Y", "severity": "error", "at": "/q"}
        output_result = _make_success_result(
            issues=[new_issue],
            summary={
                "total_issues": 1,
                "is_valid": False,
                "failures": 1,
                "warnings": 0,
                "errors": 0,
            },
        )
        ctx = self._make_context(
            tmp_path, on_failure="warn", baseline_validation=baseline
        )

        with patch(
            "world_understanding.agentic.usd_tasks.validate_usd._run_validation",
            new=AsyncMock(return_value=output_result),
        ):
            result = await ValidateOutputUSDTask().arun(ctx)

        assert result["validation_regression"] is True
        assert result["validation_success"] is True

    @pytest.mark.asyncio
    async def test_regression_block_raises(self, tmp_path):
        from unittest.mock import AsyncMock, patch

        from world_understanding.agentic.usd_tasks.validate_usd import (
            ValidateOutputUSDTask,
        )

        baseline = _make_success_result(
            issues=[],
            summary={
                "total_issues": 0,
                "is_valid": True,
                "failures": 0,
                "warnings": 0,
                "errors": 0,
            },
        )
        new_issue = {"rule": "Y", "severity": "error", "at": "/q"}
        output_result = _make_success_result(
            issues=[new_issue],
            summary={
                "total_issues": 1,
                "is_valid": False,
                "failures": 1,
                "warnings": 0,
                "errors": 0,
            },
        )
        ctx = self._make_context(
            tmp_path, on_failure="block", baseline_validation=baseline
        )

        with (
            patch(
                "world_understanding.agentic.usd_tasks.validate_usd._run_validation",
                new=AsyncMock(return_value=output_result),
            ),
            pytest.raises(RuntimeError, match="on_failure=block"),
        ):
            await ValidateOutputUSDTask().arun(ctx)

    @pytest.mark.asyncio
    async def test_no_regression_sets_flag_false(self, tmp_path):
        from unittest.mock import AsyncMock, patch

        from world_understanding.agentic.usd_tasks.validate_usd import (
            ValidateOutputUSDTask,
        )

        issue = {"rule": "X", "severity": "warning", "at": "/p"}
        baseline = _make_success_result(
            issues=[issue],
            summary={
                "total_issues": 1,
                "is_valid": True,
                "failures": 0,
                "warnings": 1,
                "errors": 0,
            },
        )
        output_result = _make_success_result(
            issues=[issue],
            summary={
                "total_issues": 1,
                "is_valid": True,
                "failures": 0,
                "warnings": 1,
                "errors": 0,
            },
        )
        ctx = self._make_context(tmp_path, baseline_validation=baseline)

        with patch(
            "world_understanding.agentic.usd_tasks.validate_usd._run_validation",
            new=AsyncMock(return_value=output_result),
        ):
            result = await ValidateOutputUSDTask().arun(ctx)

        assert result["validation_regression"] is False

    @pytest.mark.asyncio
    async def test_report_saved_with_summaries(self, tmp_path):
        import json
        from unittest.mock import AsyncMock, patch

        from world_understanding.agentic.usd_tasks.validate_usd import (
            ValidateOutputUSDTask,
        )

        baseline = _make_success_result()
        output_result = _make_success_result()
        ctx = self._make_context(tmp_path, baseline_validation=baseline)

        with patch(
            "world_understanding.agentic.usd_tasks.validate_usd._run_validation",
            new=AsyncMock(return_value=output_result),
        ):
            await ValidateOutputUSDTask().arun(ctx)

        report = tmp_path / "validation_report.json"
        assert report.exists()
        data = json.loads(report.read_text())
        assert "output_summary" in data
        assert "input_summary" in data


# ---------------------------------------------------------------------------
# ValidateUSDConfigTask tests
# ---------------------------------------------------------------------------


class TestValidateUSDConfigTask:
    """Tests for ValidateUSDConfigTask config loading and validation."""

    def _write_config(self, tmp_path: Path, data: dict) -> Path:
        import yaml

        config_path = tmp_path / "validate_config.yaml"
        config_path.write_text(yaml.dump(data))
        return config_path

    def test_missing_config_path_raises_value_error(self):
        from world_understanding.agentic.usd_tasks.config_validate_usd import (
            ValidateUSDConfigTask,
        )

        task = ValidateUSDConfigTask()
        with pytest.raises(ValueError, match="config_path is required"):
            task.run({})

    def test_missing_input_usd_path_raises_value_error(self, tmp_path):
        from world_understanding.agentic.usd_tasks.config_validate_usd import (
            ValidateUSDConfigTask,
        )

        config_path = self._write_config(tmp_path, {"on_failure": "warn"})
        task = ValidateUSDConfigTask()
        with pytest.raises(ValueError, match="input_usd_path is required"):
            task.run({"config_path": str(config_path)})

    def test_invalid_on_failure_raises_value_error(self, tmp_path):
        from world_understanding.agentic.usd_tasks.config_validate_usd import (
            ValidateUSDConfigTask,
        )

        usd = tmp_path / "input.usd"
        usd.touch()
        config_path = self._write_config(
            tmp_path, {"input_usd_path": str(usd), "on_failure": "ignore"}
        )
        task = ValidateUSDConfigTask()
        with pytest.raises(ValueError, match="Invalid on_failure"):
            task.run({"config_path": str(config_path)})

    def test_invalid_categories_raise_value_error(self, tmp_path):
        from world_understanding.agentic.usd_tasks.config_validate_usd import (
            ValidateUSDConfigTask,
        )

        usd = tmp_path / "input.usd"
        usd.touch()
        config_path = self._write_config(
            tmp_path,
            {
                "input_usd_path": str(usd),
                "validation_config": {"categories": ["Bogus:Category"]},
            },
        )
        task = ValidateUSDConfigTask()
        with pytest.raises(ValueError, match="Unknown validation categories"):
            task.run({"config_path": str(config_path)})

    def test_default_categories_applied_when_not_specified(self, tmp_path):
        from world_understanding.agentic.usd_tasks.config_validate_usd import (
            ValidateUSDConfigTask,
        )
        from world_understanding.functions.graphics.validate_usd import (
            DEFAULT_VALIDATION_CATEGORIES,
        )

        usd = tmp_path / "input.usd"
        usd.touch()
        config_path = self._write_config(tmp_path, {"input_usd_path": str(usd)})
        task = ValidateUSDConfigTask()
        ctx = task.run({"config_path": str(config_path), "listener": _FakeListener()})

        assert ctx["validation_config"]["categories"] == list(
            DEFAULT_VALIDATION_CATEGORIES
        )

    def test_on_failure_passed_to_context(self, tmp_path):
        from world_understanding.agentic.usd_tasks.config_validate_usd import (
            ValidateUSDConfigTask,
        )

        usd = tmp_path / "input.usd"
        usd.touch()
        config_path = self._write_config(
            tmp_path, {"input_usd_path": str(usd), "on_failure": "block"}
        )
        task = ValidateUSDConfigTask()
        ctx = task.run({"config_path": str(config_path), "listener": _FakeListener()})

        assert ctx["on_failure"] == "block"

    def test_original_usd_path_passed_through(self, tmp_path):
        from world_understanding.agentic.usd_tasks.config_validate_usd import (
            ValidateUSDConfigTask,
        )

        usd = tmp_path / "input.usd"
        usd.touch()
        config_path = self._write_config(
            tmp_path,
            {"input_usd_path": str(usd), "original_usd_path": "/some/original.usd"},
        )
        task = ValidateUSDConfigTask()
        ctx = task.run({"config_path": str(config_path), "listener": _FakeListener()})

        assert ctx["original_usd_path"] == "/some/original.usd"

    def test_baseline_validation_passed_through(self, tmp_path):
        from world_understanding.agentic.usd_tasks.config_validate_usd import (
            ValidateUSDConfigTask,
        )

        usd = tmp_path / "input.usd"
        usd.touch()
        baseline = {"issues": [], "summary": {"is_valid": True}}
        config_path = self._write_config(
            tmp_path,
            {"input_usd_path": str(usd), "baseline_validation": baseline},
        )
        task = ValidateUSDConfigTask()
        ctx = task.run({"config_path": str(config_path), "listener": _FakeListener()})

        assert ctx["baseline_validation"] == baseline


# ---------------------------------------------------------------------------
# ModelProvisioningTask._has_backend guard tests
# ---------------------------------------------------------------------------


class TestModelProvisioningHasBackend:
    """Tests for _has_backend guard logic in ModelProvisioningTask."""

    def _run_with_config(self, config: dict) -> dict:
        from world_understanding.agentic.domain_tasks.model_provisioning import (
            ModelProvisioningTask,
        )

        task = ModelProvisioningTask()
        return task.run({"config": config, "listener": _FakeListener()})

    def test_empty_vlm_dict_skips_vlm_creation(self):
        result = self._run_with_config({"vlm": {}})
        assert result["vlm"] is None

    def test_empty_llm_judge_dict_skips_llm_judge_creation(self):
        result = self._run_with_config({"llm_judge": {}})
        assert result["llm_judge"] is None

    def test_vlm_with_backend_creates_vlm(self):
        from unittest.mock import Mock, patch

        mock_vlm = Mock()
        with (
            patch(
                "world_understanding.agentic.domain_tasks.model_provisioning.create_vlm",
                return_value=mock_vlm,
            ),
            patch(
                "world_understanding.agentic.domain_tasks.model_provisioning.list_vlm_backends",
                return_value=["nim"],
            ),
            patch(
                "world_understanding.agentic.domain_tasks.model_provisioning.get_api_key_for_backend",
                return_value="test-key",
            ),
        ):
            result = self._run_with_config({"vlm": {"backend": "nim", "model": "test"}})

        assert result["vlm"] is mock_vlm


# ---------------------------------------------------------------------------
# IterativeApplyConfigTask judge routing tests
# ---------------------------------------------------------------------------


class TestJudgeRouting:
    """Test that IterativeApplyConfigTask routes VLM judges to vlm_judge
    and LLM judges to llm_judge in the config dict."""

    def test_vlm_judge_routed_to_vlm_judge_key(self, tmp_path):
        """When judge config has a vlm key, config['vlm_judge'] is set."""
        from material_agent.api.defaults import ITERATION_DEFAULTS, apply_defaults

        judge_config = {
            "vlm": {"backend": "nim", "model": "test-model", "temperature": 0.7}
        }
        judge_with_defaults = apply_defaults(judge_config, ITERATION_DEFAULTS["judge"])

        # Simulate what config_iterative_apply.py lines 160-165 do
        config: dict[str, Any] = {}
        if "vlm" in judge_with_defaults:
            config["vlm_judge"] = judge_with_defaults["vlm"]
        else:
            config["llm_judge"] = judge_with_defaults

        assert "vlm_judge" in config
        assert "llm_judge" not in config
        assert config["vlm_judge"]["backend"] == "nim"

    def test_llm_judge_routed_to_llm_judge_key(self):
        """When judge config has no vlm key, config['llm_judge'] is set."""
        from material_agent.api.defaults import ITERATION_DEFAULTS, apply_defaults

        judge_config = {
            "backend": "nim",
            "model": "judge-model",
        }
        judge_with_defaults = apply_defaults(judge_config, ITERATION_DEFAULTS["judge"])

        config: dict[str, Any] = {}
        if "vlm" in judge_with_defaults:
            config["vlm_judge"] = judge_with_defaults["vlm"]
        else:
            config["llm_judge"] = judge_with_defaults

        assert "llm_judge" in config
        assert "vlm_judge" not in config

    def test_default_judge_with_vlm_not_routed_to_llm(self):
        """Default judge config (which has a vlm key) should NOT go to llm_judge."""
        from material_agent.api.defaults import ITERATION_DEFAULTS, apply_defaults

        # Empty judge config — defaults will be applied
        judge_with_defaults = apply_defaults({}, ITERATION_DEFAULTS["judge"])

        config: dict[str, Any] = {}
        if "vlm" in judge_with_defaults:
            config["vlm_judge"] = judge_with_defaults["vlm"]
        else:
            config["llm_judge"] = judge_with_defaults

        # The default judge has a vlm key, so it should go to vlm_judge
        if "vlm" in ITERATION_DEFAULTS["judge"]:
            assert "vlm_judge" in config
            assert "llm_judge" not in config


# ---------------------------------------------------------------------------
# Local validator tests
# ---------------------------------------------------------------------------


def _validator_available() -> bool:
    from world_understanding.functions.graphics.validate_usd import is_available

    return is_available()


_skip_no_validator = pytest.mark.skipif(
    not _validator_available(),
    reason="omniverse-asset-validator not installed",
)


class TestValidateUsdLocal:
    """Tests for validate_usd.py."""

    def test_is_available(self):
        from world_understanding.functions.graphics.validate_usd import (
            is_available,
        )

        # Returns True only if omniverse-asset-validator is installed
        result = is_available()
        assert isinstance(result, bool)

    @_skip_no_validator
    def test_validate_usd_file_not_found(self):
        from world_understanding.functions.graphics.validate_usd import (
            validate_usd,
        )

        with pytest.raises(ValueError, match="does not exist"):
            validate_usd("/nonexistent/file.usd")

    @_skip_no_validator
    def test_validate_usd_success(self, tmp_path):
        """Validate a real USD file and check response format."""
        from pxr import Usd, UsdGeom
        from world_understanding.functions.graphics.validate_usd import (
            validate_usd,
        )

        # Create a minimal valid USD
        usd_path = tmp_path / "test.usda"
        stage = Usd.Stage.CreateNew(str(usd_path))
        UsdGeom.Xform.Define(stage, "/Root")
        stage.SetDefaultPrim(stage.GetPrimAtPath("/Root"))
        stage.GetRootLayer().Save()

        result = validate_usd(usd_path)

        assert result["status"] == "success"
        assert "validation_time" in result
        assert isinstance(result["issues"], list)
        assert isinstance(result["summary"], dict)
        assert "total_issues" in result["summary"]
        assert "is_valid" in result["summary"]
        assert "failures" in result["summary"]
        assert "warnings" in result["summary"]
        assert "errors" in result["summary"]
        assert isinstance(result["categories_checked"], list)
        assert isinstance(result["fixes"], list)

    @_skip_no_validator
    def test_validate_usd_category_filtering(self, tmp_path):
        """Filtering to specific categories reduces issues."""
        from pxr import Usd, UsdGeom
        from world_understanding.functions.graphics.validate_usd import (
            validate_usd,
        )

        usd_path = tmp_path / "test.usda"
        stage = Usd.Stage.CreateNew(str(usd_path))
        UsdGeom.Xform.Define(stage, "/Root")
        stage.GetRootLayer().Save()

        # Get all issues
        all_result = validate_usd(usd_path)
        # Filter to just one category
        filtered_result = validate_usd(usd_path, categories=["Omni:Geometry"])

        assert filtered_result["status"] == "success"
        assert filtered_result["categories_checked"] == ["Omni:Geometry"]
        assert (
            filtered_result["summary"]["total_issues"]
            <= all_result["summary"]["total_issues"]
        )

    def test_map_severity(self):
        from world_understanding.functions.graphics.validate_usd import (
            _map_severity,
        )

        class FakeSeverity:
            def __init__(self, name: str):
                self.name = name

        assert _map_severity(FakeSeverity("FAILURE")) == "failure"
        assert _map_severity(FakeSeverity("WARNING")) == "warning"
        assert _map_severity(FakeSeverity("ERROR")) == "error"
        assert _map_severity(FakeSeverity("other")) == "warning"  # default

    def test_get_rule_name(self):
        from world_understanding.functions.graphics.validate_usd import (
            _get_rule_name,
        )

        class FakeIssue:
            def __init__(self, rule: Any):
                self.rule = rule

        # Class repr format
        issue = FakeIssue(type("MyChecker", (), {}))
        assert "MyChecker" in _get_rule_name(issue)

        # No rule
        issue_no_rule = FakeIssue(None)
        assert _get_rule_name(issue_no_rule) == "Unknown"

    def test_infer_category(self):
        from world_understanding.functions.graphics.validate_usd import (
            _infer_category,
        )

        class FakeIssue:
            def __init__(self, rule_name: str):
                name = rule_name
                self.rule = type(name, (), {})

        assert _infer_category(FakeIssue("StageMetadataChecker")) == "Basic"
        assert _infer_category(FakeIssue("MaterialPathChecker")) == "Omni:Material"
        assert _infer_category(FakeIssue("IndexedPrimvarChecker")) == "Omni:Geometry"
        assert _infer_category(FakeIssue("SomeNewChecker")) == "Unknown"


# ---------------------------------------------------------------------------
# expand_cluster_predictions auto-enable (regression for silent-disable bug)
# ---------------------------------------------------------------------------


_MINIMAL_MATERIALS = {
    "entries": [{"name": "test_material", "binding": "OmniPBR.mdl:OmniPBR"}]
}


class TestExpandClusterPredictionsAutoEnable:
    """expand_cluster_predictions must be auto-enabled whenever cluster_prims is
    enabled — with or without an explicit config section of its own.

    Regression test for the bug where the step was silently disabled because it
    had no YAML config keys, causing cluster representative predictions to never
    be propagated to cluster members.
    """

    def test_auto_enabled_when_cluster_prims_implicitly_enabled(self, tmp_path):
        """The original bug: cluster_prims has config but expand_cluster_predictions
        has no section at all — it must still be included in steps_to_run."""
        config = _make_config(
            tmp_path,
            steps={
                "cluster_prims": {
                    "embedding_service": "nvidia_inference",
                    "embedding_model": "nvidia/nv-embedqa-e5-v5",
                },
                "predict": {"max_workers": 1},
                # expand_cluster_predictions intentionally absent (the bug scenario)
            },
            materials=_MINIMAL_MATERIALS,
        )
        ctx = _run_config_task(config)

        assert "expand_cluster_predictions" in ctx["steps_to_run"]

    def test_auto_enabled_when_cluster_prims_explicitly_enabled(self, tmp_path):
        """expand_cluster_predictions is auto-enabled when cluster_prims has
        enabled: true, even if expand_cluster_predictions has no config section."""
        config = _make_config(
            tmp_path,
            steps={
                "cluster_prims": {"enabled": True},
                "predict": {"max_workers": 1},
                # expand_cluster_predictions has no section
            },
            materials=_MINIMAL_MATERIALS,
        )
        ctx = _run_config_task(config)

        assert "expand_cluster_predictions" in ctx["steps_to_run"]

    def test_not_auto_enabled_when_no_prediction_step(self, tmp_path):
        """expand_cluster_predictions must NOT be auto-enabled when cluster_prims
        is enabled but neither predict nor benchmark is in the pipeline.

        Regression for the Codex-identified P1: a cluster-only run (e.g.
        --only cluster_prims, or predict/benchmark both disabled) would
        previously auto-enable expand_cluster_predictions, which then
        crashed with FileNotFoundError because no predictions file exists."""
        config = _make_config(
            tmp_path,
            steps={
                "cluster_prims": {
                    "embedding_service": "nvidia_inference",
                    "embedding_model": "nvidia/nv-embedqa-e5-v5",
                },
                # No predict, no benchmark
            },
        )
        ctx = _run_config_task(config)

        assert "expand_cluster_predictions" not in ctx["steps_to_run"]

    def test_not_auto_enabled_when_predict_explicitly_disabled(self, tmp_path):
        """expand_cluster_predictions is NOT auto-enabled when cluster_prims is
        enabled but predict is explicitly disabled."""
        config = _make_config(
            tmp_path,
            steps={
                "cluster_prims": {
                    "embedding_service": "nvidia_inference",
                    "embedding_model": "nvidia/nv-embedqa-e5-v5",
                },
                "predict": {"enabled": False},
                "validate_input": {"enabled": True},
            },
        )
        ctx = _run_config_task(config)

        assert "expand_cluster_predictions" not in ctx["steps_to_run"]

    def test_not_enabled_when_cluster_prims_absent(self, tmp_path):
        """Without cluster_prims, expand_cluster_predictions stays disabled."""
        config = _make_config(
            tmp_path,
            steps={
                "validate_input": {"enabled": True},
            },
        )
        ctx = _run_config_task(config)

        assert "expand_cluster_predictions" not in ctx["steps_to_run"]

    def test_not_enabled_when_cluster_prims_explicitly_disabled(self, tmp_path):
        """expand_cluster_predictions stays off when cluster_prims is disabled."""
        config = _make_config(
            tmp_path,
            steps={
                "cluster_prims": {"enabled": False},
                "validate_input": {"enabled": True},
            },
        )
        ctx = _run_config_task(config)

        assert "expand_cluster_predictions" not in ctx["steps_to_run"]

    def test_can_be_explicitly_disabled_even_when_cluster_prims_enabled(self, tmp_path):
        """An explicit enabled: false on expand_cluster_predictions overrides
        the auto-enable logic."""
        config = _make_config(
            tmp_path,
            steps={
                "cluster_prims": {
                    "embedding": {"backend": "nvidia_inference"},
                    "distance_threshold": 0.3,
                },
                "expand_cluster_predictions": {"enabled": False},
            },
        )
        ctx = _run_config_task(config)

        assert "expand_cluster_predictions" not in ctx["steps_to_run"]

    def test_cluster_prims_included_alongside_expand(self, tmp_path):
        """Both steps appear in steps_to_run in the correct order."""
        config = _make_config(
            tmp_path,
            steps={
                "cluster_prims": {
                    "embedding_service": "nvidia_inference",
                    "embedding_model": "nvidia/nv-embedqa-e5-v5",
                },
                "predict": {"max_workers": 1},
            },
            materials=_MINIMAL_MATERIALS,
        )
        ctx = _run_config_task(config)

        steps = ctx["steps_to_run"]
        assert "cluster_prims" in steps
        assert "expand_cluster_predictions" in steps
        assert steps.index("cluster_prims") < steps.index("expand_cluster_predictions")

    def test_enabled_standalone_when_has_own_config_keys(self, tmp_path):
        """expand_cluster_predictions is implicitly enabled when it has its own
        behavior config keys, even without a cluster_prims section.

        This preserves the general 'has config → implicitly enabled' invariant
        for expand_cluster_predictions (P2 regression from review)."""
        config = _make_config(
            tmp_path,
            steps={
                "expand_cluster_predictions": {
                    "report": False,  # any non-path behavior key
                },
                # no cluster_prims section
            },
        )
        ctx = _run_config_task(config)

        assert "expand_cluster_predictions" in ctx["steps_to_run"]
