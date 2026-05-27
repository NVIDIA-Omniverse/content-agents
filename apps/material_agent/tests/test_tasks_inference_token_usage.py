# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for VLM inference token usage artifacts."""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

from material_agent.tasks.inference import _write_token_usage_artifact


def _stats(
    *,
    total_input_tokens: int,
    total_output_tokens: int,
    total_tokens: int,
    invocation_count: int,
    model: str = "vlm",
) -> dict:
    return {
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_tokens,
        "invocation_count": invocation_count,
        "by_model": {
            model: {
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens,
                "total_tokens": total_tokens,
                "count": invocation_count,
            }
        },
        "by_type": {
            "vlm": {
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens,
                "total_tokens": total_tokens,
                "count": invocation_count,
            }
        },
        "all_usages": [
            {
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens,
                "total_tokens": total_tokens,
                "model_name": model,
                "invocation_type": "vlm",
            }
        ]
        if invocation_count
        else [],
    }


def _write_usage_artifact_in_process(args: tuple[str, int]) -> str | None:
    predictions_path, index = args
    token_path = _write_token_usage_artifact(
        Path(predictions_path),
        _stats(
            total_input_tokens=1,
            total_output_tokens=2,
            total_tokens=3,
            invocation_count=1,
            model=f"vlm-process-{index}",
        ),
        merge_existing=True,
    )
    return str(token_path) if token_path else None


def test_write_token_usage_artifact_merges_existing_resume_usage(
    tmp_path: Path,
) -> None:
    predictions_path = tmp_path / "predictions" / "predictions.jsonl"
    predictions_path.parent.mkdir()
    existing_path = _write_token_usage_artifact(
        predictions_path,
        _stats(
            total_input_tokens=10,
            total_output_tokens=5,
            total_tokens=15,
            invocation_count=1,
        ),
    )
    assert existing_path is not None

    _write_token_usage_artifact(
        predictions_path,
        _stats(
            total_input_tokens=3,
            total_output_tokens=4,
            total_tokens=7,
            invocation_count=1,
        ),
        merge_existing=True,
    )

    payload = json.loads(existing_path.read_text(encoding="utf-8"))
    usage = payload["token_usage"]
    assert usage["total_input_tokens"] == 13
    assert usage["total_output_tokens"] == 9
    assert usage["total_tokens"] == 22
    assert usage["invocation_count"] == 2
    assert usage["by_model"]["vlm"]["count"] == 2
    assert len(usage["all_usages"]) == 2


def test_write_token_usage_artifact_overwrites_without_resume_merge(
    tmp_path: Path,
) -> None:
    predictions_path = tmp_path / "predictions" / "predictions.jsonl"
    predictions_path.parent.mkdir()
    existing_path = _write_token_usage_artifact(
        predictions_path,
        _stats(
            total_input_tokens=10,
            total_output_tokens=5,
            total_tokens=15,
            invocation_count=1,
        ),
    )
    assert existing_path is not None

    _write_token_usage_artifact(
        predictions_path,
        _stats(
            total_input_tokens=3,
            total_output_tokens=4,
            total_tokens=7,
            invocation_count=1,
        ),
    )

    payload = json.loads(existing_path.read_text(encoding="utf-8"))
    usage = payload["token_usage"]
    assert usage["total_input_tokens"] == 3
    assert usage["total_output_tokens"] == 4
    assert usage["total_tokens"] == 7
    assert usage["invocation_count"] == 1
    assert len(usage["all_usages"]) == 1


def test_write_token_usage_artifact_serializes_concurrent_resume_merges(
    tmp_path: Path,
) -> None:
    predictions_path = tmp_path / "predictions" / "predictions.jsonl"
    predictions_path.parent.mkdir()

    def write_usage(index: int) -> None:
        _write_token_usage_artifact(
            predictions_path,
            _stats(
                total_input_tokens=1,
                total_output_tokens=2,
                total_tokens=3,
                invocation_count=1,
                model=f"vlm-{index}",
            ),
            merge_existing=True,
        )

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(write_usage, range(8)))

    payload = json.loads(
        (predictions_path.parent / "token_usage.json").read_text(encoding="utf-8")
    )
    usage = payload["token_usage"]
    assert usage["total_input_tokens"] == 8
    assert usage["total_output_tokens"] == 16
    assert usage["total_tokens"] == 24
    assert usage["invocation_count"] == 8
    assert len(usage["by_model"]) == 8
    assert len(usage["all_usages"]) == 8


def test_write_token_usage_artifact_serializes_process_resume_merges(
    tmp_path: Path,
) -> None:
    predictions_path = tmp_path / "predictions" / "predictions.jsonl"
    predictions_path.parent.mkdir()

    with ProcessPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(
                _write_usage_artifact_in_process,
                [(str(predictions_path), index) for index in range(4)],
            )
        )

    assert all(results)
    payload = json.loads(
        (predictions_path.parent / "token_usage.json").read_text(encoding="utf-8")
    )
    usage = payload["token_usage"]
    assert usage["total_input_tokens"] == 4
    assert usage["total_output_tokens"] == 8
    assert usage["total_tokens"] == 12
    assert usage["invocation_count"] == 4
    assert len(usage["by_model"]) == 4
    assert len(usage["all_usages"]) == 4
