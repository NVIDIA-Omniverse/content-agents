# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test that temporary config files have unique paths.

These tests verify Critical Issue #1 fix: temp config files must have
unique paths to prevent collisions during concurrent execution.
"""

import tempfile
import threading
import time
from pathlib import Path

from material_agent.tasks.unified_pipeline_executor import UnifiedPipelineExecutorTask


def test_temp_config_files_are_unique():
    """Verify that repeated calls generate unique file paths."""
    executor = UnifiedPipelineExecutorTask()

    with tempfile.TemporaryDirectory() as tmpdir:
        working_dir = Path(tmpdir)

        # Create 100 temp config files with same step name
        paths = []
        for i in range(100):
            path = executor._create_temp_config_file(
                step_name="predict",  # Same step name for all
                step_config={"model": f"test_{i}"},
                working_dir=working_dir,
            )
            paths.append(path)

        # All paths must be unique
        assert len(paths) == len(set(paths)), (
            f"Found duplicate paths! {len(paths)} calls created "
            f"{len(set(paths))} unique paths"
        )

        # All files should exist
        assert all(p.exists() for p in paths), "Some temp files were not created"

        # All should be in the same temp directory
        assert all(p.parent == working_dir / ".pipeline_temp" for p in paths), (
            "Temp files created in wrong directory"
        )


def test_temp_config_files_contain_correct_data():
    """Verify that each temp file contains the correct configuration."""
    import yaml

    executor = UnifiedPipelineExecutorTask()

    with tempfile.TemporaryDirectory() as tmpdir:
        working_dir = Path(tmpdir)

        # Create multiple configs with different data
        configs_created = []
        for i in range(10):
            config = {"model": f"model_{i}", "batch_size": i * 10}
            path = executor._create_temp_config_file(
                step_name="predict", step_config=config, working_dir=working_dir
            )
            configs_created.append((path, config))

        # Read back each file and verify contents
        for path, original_config in configs_created:
            with open(path) as f:
                loaded_config = yaml.safe_load(f)

            # Should match what we wrote (subject to serialization)
            assert "model" in loaded_config
            assert loaded_config["model"] == original_config["model"]


def test_concurrent_temp_file_creation_no_collision():
    """Test that concurrent threads don't overwrite each other's temp files."""
    import yaml

    executor = UnifiedPipelineExecutorTask()

    with tempfile.TemporaryDirectory() as tmpdir:
        working_dir = Path(tmpdir)

        results = []
        errors = []
        lock = threading.Lock()

        def create_and_verify(thread_id):
            """Create temp config and verify it contains correct data after delay."""
            try:
                # Create config with thread-specific data
                config = {"thread_id": thread_id, "data": f"thread_{thread_id}"}
                path = executor._create_temp_config_file(
                    step_name="predict",  # All threads use same step name
                    step_config=config,
                    working_dir=working_dir,
                )

                # Small delay to increase chance of collision (if bug exists)
                time.sleep(0.01)

                # Read back the file
                with open(path) as f:
                    loaded = yaml.safe_load(f)

                # Verify we got OUR data, not another thread's
                with lock:
                    if loaded.get("thread_id") != thread_id:
                        errors.append(
                            {
                                "thread": thread_id,
                                "expected": thread_id,
                                "actual": loaded.get("thread_id"),
                                "path": str(path),
                            }
                        )
                    else:
                        results.append((thread_id, path))

            except Exception as e:
                with lock:
                    errors.append({"thread": thread_id, "exception": str(e)})

        # Launch 20 threads simultaneously
        threads = [
            threading.Thread(target=create_and_verify, args=(i,)) for i in range(20)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have no errors or collisions
        assert len(errors) == 0, f"Thread collisions detected: {errors}"

        # Should have 20 successful results
        assert len(results) == 20, f"Expected 20 successes, got {len(results)}"

        # All paths should be unique
        paths = [p for _, p in results]
        assert len(paths) == len(set(paths)), "Threads created duplicate file paths"
