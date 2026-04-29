# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for OptimizeUSDTask local backend dispatch and asyncio.to_thread wrapping."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from world_understanding.agentic.usd_tasks.optimize_usd import OptimizeUSDTask


def _make_context(tmp_path: Path, backend: str = "local") -> dict:
    input_usd = tmp_path / "input.usd"
    input_usd.touch()
    output_usd = tmp_path / "output.usd"
    return {
        "input_usd_path": str(input_usd),
        "output_usd_path": str(output_usd),
        "optimization_config": {
            "backend": backend,
            "flatten_prototypes": False,
            "scene_optimizer_settings": {
                "enable_deinstance": False,
                "enable_split_meshes": True,
                "enable_deduplicate": True,
            },
        },
    }


def _mock_usd_open(prim_count: int = 3):
    """Return a patched pxr.Usd.Stage.Open that returns a mock stage."""
    mock_prim = MagicMock()
    mock_prim.IsA = MagicMock(return_value=True)
    mock_stage = MagicMock()
    mock_stage.Traverse.return_value = [mock_prim] * prim_count
    return mock_stage


class TestOptimizeUSDTaskLocalBackend:
    """Tests for the local backend branch in OptimizeUSDTask.arun() (lines 196-207)."""

    @pytest.mark.asyncio
    async def test_local_backend_uses_asyncio_to_thread(self, tmp_path):
        """backend='local' must wrap optimize_usd_local in asyncio.to_thread."""
        context = _make_context(tmp_path, backend="local")
        mock_stage = _mock_usd_open()

        local_result = {
            "status": "success",
            "optimization_time": 1.0,
            "operations_executed": ["split", "deduplicate"],
        }

        captured_calls: list = []

        async def fake_to_thread(fn, **kwargs):
            captured_calls.append({"fn": fn, "kwargs": kwargs})
            return local_result

        with (
            patch("pxr.Usd.Stage.Open", return_value=mock_stage),
            patch("pxr.UsdGeom.Mesh", MagicMock()),
            patch(
                "world_understanding.agentic.usd_tasks.optimize_usd.optimize_usd_from_path",
            ) as mock_nvcf,
            patch(
                "world_understanding.functions.graphics.scene_optimizer_local.optimize_usd_local",
                return_value=local_result,
            ),
            patch("asyncio.to_thread", side_effect=fake_to_thread),
        ):
            task = OptimizeUSDTask()
            result = await task.arun(context)

        # NVCF path must NOT have been invoked
        mock_nvcf.assert_not_called()

        # asyncio.to_thread must have been called exactly once
        assert len(captured_calls) == 1, (
            f"Expected asyncio.to_thread to be called once, got {len(captured_calls)}"
        )

        # The callable passed to to_thread must be optimize_usd_local.
        # When patched, it is a MagicMock whose repr contains the name.
        fn = captured_calls[0]["fn"]
        assert "optimize_usd_local" in str(fn), (
            f"Expected optimize_usd_local callable, got {fn!r}"
        )

        # Context should reflect success
        assert result["optimization_success"] is True

    @pytest.mark.asyncio
    async def test_nvcf_backend_does_not_use_to_thread(self, tmp_path):
        """backend='remote' must call optimize_usd_from_path and not asyncio.to_thread."""
        context = _make_context(tmp_path, backend="remote")
        mock_stage = _mock_usd_open()

        nvcf_result = {
            "status": "success",
            "optimization_time": 2.0,
            "operations_executed": ["split", "deduplicate"],
        }

        with (
            patch("pxr.Usd.Stage.Open", return_value=mock_stage),
            patch("pxr.UsdGeom.Mesh", MagicMock()),
            patch(
                "world_understanding.agentic.usd_tasks.optimize_usd.optimize_usd_from_path",
                new_callable=AsyncMock,
                return_value=nvcf_result,
            ) as mock_nvcf,
            patch("asyncio.to_thread") as mock_to_thread,
        ):
            task = OptimizeUSDTask()
            result = await task.arun(context)

        mock_nvcf.assert_called_once()
        mock_to_thread.assert_not_called()
        assert result["optimization_success"] is True

    @pytest.mark.asyncio
    async def test_local_failure_without_nvcf_raises_with_guidance(
        self, tmp_path, monkeypatch
    ):
        """Local backend missing + no NVCF endpoint should raise, not fall through to NVCF."""
        monkeypatch.delenv("NVCF_OPTIMIZER_FUNCTION_ID", raising=False)
        monkeypatch.delenv("OPTIMIZER_ENDPOINT", raising=False)

        context = _make_context(tmp_path, backend="local")
        mock_stage = _mock_usd_open()

        with (
            patch("pxr.Usd.Stage.Open", return_value=mock_stage),
            patch("pxr.UsdGeom.Mesh", MagicMock()),
            patch(
                "world_understanding.agentic.usd_tasks.optimize_usd.optimize_usd_from_path",
                new_callable=AsyncMock,
            ) as mock_nvcf,
            patch(
                "world_understanding.functions.graphics.scene_optimizer_local.optimize_usd_local",
                side_effect=RuntimeError(
                    "WU_SO_PACKAGE_DIR environment variable is not set"
                ),
            ),
        ):
            task = OptimizeUSDTask()
            with pytest.raises(RuntimeError, match="fetch_build_resources"):
                await task.arun(context)

        mock_nvcf.assert_not_called()
