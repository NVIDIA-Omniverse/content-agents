# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Additional tests for material_agent.batch_processor."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from material_agent.batch_processor import process_usd_batch


async def _never_called(context: dict[str, Any]) -> dict[str, Any]:
    raise AssertionError("workflow_runner should not be called")


@pytest.mark.asyncio
async def test_process_usd_batch_raises_for_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="USD directory not found"):
        await process_usd_batch(
            tmp_path / "missing",
            tmp_path / "out",
            workflow_runner=_never_called,
        )


@pytest.mark.asyncio
async def test_process_usd_batch_raises_when_no_usd_files_found(tmp_path: Path) -> None:
    usd_dir = tmp_path / "usd"
    usd_dir.mkdir()
    (usd_dir / "note.txt").write_text("not a usd")

    async def workflow_runner(context: dict[str, object]) -> dict[str, object]:
        raise AssertionError("workflow_runner should not be called")

    with pytest.raises(RuntimeError, match="No USD files found"):
        await process_usd_batch(usd_dir, tmp_path / "out", workflow_runner)


@pytest.mark.asyncio
async def test_process_usd_batch_aggregates_success_error_and_exception(
    tmp_path: Path,
) -> None:
    usd_dir = tmp_path / "usd"
    usd_dir.mkdir()
    for name in ["alpha.usd", "beta.usda", "gamma.usdc"]:
        (usd_dir / name).write_text("usd")

    calls: list[dict[str, object]] = []
    base_context = {"shared": "value"}

    async def workflow_runner(context: dict[str, object]) -> dict[str, object]:
        calls.append(dict(context))
        stem = Path(context["source_override"]).stem
        if stem == "alpha":
            return {"dataset_path": "dataset.jsonl", "num_prims": 4, "num_images": 2}
        if stem == "beta":
            return {"error": "bad input"}
        raise ValueError("boom")

    result = await process_usd_batch(
        usd_dir,
        tmp_path / "batch_out",
        workflow_runner,
        base_context=base_context,
    )

    assert result["num_files_processed"] == 1
    assert result["num_files_failed"] == 2
    assert result["total_files"] == 3
    assert set(result["results"]) == {"alpha", "beta", "gamma"}

    alpha = result["results"]["alpha"]
    assert alpha["status"] == "success"
    assert alpha["dataset_path"] == "dataset.jsonl"
    assert alpha["num_prims"] == 4
    assert alpha["num_images"] == 2

    beta = result["results"]["beta"]
    assert beta["status"] == "failed"
    assert beta["error"] == "bad input"

    gamma = result["results"]["gamma"]
    assert gamma["status"] == "failed"
    assert gamma["error"] == "boom"

    assert base_context == {"shared": "value"}
    assert {Path(call["source_override"]).stem for call in calls} == {
        "alpha",
        "beta",
        "gamma",
    }
    assert {Path(call["output_dir_override"]).name for call in calls} == {
        "alpha",
        "beta",
        "gamma",
    }
    assert all(call["shared"] == "value" for call in calls)


@pytest.mark.asyncio
async def test_process_usd_batch_raises_when_all_files_fail(tmp_path: Path) -> None:
    usd_dir = tmp_path / "usd"
    usd_dir.mkdir()
    (usd_dir / "broken.usd").write_text("usd")

    async def workflow_runner(context: dict[str, object]) -> dict[str, object]:
        return {"error": "still broken"}

    with pytest.raises(RuntimeError, match="All 1 USD files failed to process"):
        await process_usd_batch(usd_dir, tmp_path / "out", workflow_runner)
