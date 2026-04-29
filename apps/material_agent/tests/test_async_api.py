# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test async API functionality."""

import asyncio

import pytest

# Test that async imports work
from material_agent.api import (
    aapply,
    abenchmark,
    aconfigure,
    aevaluate,
    apipeline,
    apredict,
    arefine,
    arun_benchmark,
    arun_pipeline,
    arun_predict,
)


def test_async_imports():
    """Test that all async functions are importable."""
    # Verify async functions exist and are coroutines
    assert asyncio.iscoroutinefunction(abenchmark)
    assert asyncio.iscoroutinefunction(apredict)
    assert asyncio.iscoroutinefunction(apipeline)
    assert asyncio.iscoroutinefunction(aapply)
    assert asyncio.iscoroutinefunction(aevaluate)
    assert asyncio.iscoroutinefunction(arefine)
    assert asyncio.iscoroutinefunction(aconfigure)
    assert asyncio.iscoroutinefunction(arun_benchmark)
    assert asyncio.iscoroutinefunction(arun_predict)
    assert asyncio.iscoroutinefunction(arun_pipeline)


def test_sync_backward_compatibility():
    """Test that sync functions still exist."""
    from material_agent.api import (
        apply,
        benchmark,
        configure,
        evaluate,
        pipeline,
        predict,
        refine,
        run_benchmark,
        run_pipeline,
        run_predict,
    )

    # Verify sync functions exist
    assert callable(benchmark)
    assert callable(predict)
    assert callable(pipeline)
    assert callable(apply)
    assert callable(evaluate)
    assert callable(refine)
    assert callable(configure)
    assert callable(run_benchmark)
    assert callable(run_predict)
    assert callable(run_pipeline)


# Note: Full integration tests would require actual config files and USD assets
# These are basic smoke tests to ensure the async infrastructure is in place
