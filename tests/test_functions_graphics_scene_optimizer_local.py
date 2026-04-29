# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for local Scene Optimizer backend.

Unit tests run without the Scene Optimizer package — all subprocess calls
are mocked.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from world_understanding.functions.graphics.scene_optimizer_local import (
    _build_operations_list,
    optimize_usd_local,
)
from world_understanding.functions.graphics.so_worker import (
    _natural_sort_key,
    build_correspondence_map,
    track_split_meshes,
)

# ---------------------------------------------------------------------------
# _build_operations_list tests
# ---------------------------------------------------------------------------


class TestBuildOperationsList:
    """Test _build_operations_list() operation ordering and params."""

    def test_defaults(self):
        """Default settings produce deinstance, splitMeshes, deduplicateGeometry."""
        settings = {
            "enable_deinstance": True,
            "enable_split_meshes": True,
            "enable_deduplicate": True,
            "deduplicate": {
                "tolerance": 0.001,
                "consider_deep_transforms": True,
            },
        }
        ops = _build_operations_list(settings)

        assert len(ops) == 3
        assert ops[0][0] == "utilityFunction"
        assert ops[0][1]["function"] == 0  # DEINSTANCE
        assert ops[1][0] == "splitMeshes"
        assert ops[1][1]["splitOn"] == 1
        assert ops[2][0] == "deduplicateGeometry"
        assert ops[2][1]["duplicateMethod"] == 2
        assert ops[2][1]["tolerance"] == 0.001

    def test_disabled(self):
        """All ops disabled — empty list."""
        settings = {
            "enable_deinstance": False,
            "enable_split_meshes": False,
            "enable_deduplicate": False,
        }
        ops = _build_operations_list(settings)

        assert len(ops) == 0

    def test_custom_dedup_params(self):
        """Deduplicate params are passed through correctly."""
        settings = {
            "enable_deinstance": False,
            "enable_split_meshes": False,
            "enable_deduplicate": True,
            "deduplicate": {
                "tolerance": 0.01,
                "consider_deep_transforms": False,
                "fuzzy": True,
                "use_gpu": True,
                "allow_scaling": True,
                "ignore_attributes": ["primvars:st"],
            },
        }
        ops = _build_operations_list(settings)

        assert len(ops) == 1
        dedup_params = ops[0][1]
        assert dedup_params["tolerance"] == 0.01
        assert dedup_params["considerDeepTransforms"] is False
        assert dedup_params["fuzzy"] is True
        assert dedup_params["useGpu"] is True
        assert dedup_params["allowScaling"] is True
        assert dedup_params["ignoreAttributes"] == ["primvars:st"]

    def test_deinstance_only(self):
        """enable_deinstance=True adds utilityFunction(DEINSTANCE) operation."""
        settings = {
            "enable_deinstance": True,
            "enable_split_meshes": False,
            "enable_deduplicate": False,
        }
        ops = _build_operations_list(settings)

        assert len(ops) == 1
        assert ops[0][0] == "utilityFunction"
        assert ops[0][1]["function"] == 0
        assert ops[0][1]["primPaths"] == []

    def test_empty_settings(self):
        """Empty settings dict uses defaults (deinstance + split + dedup enabled)."""
        ops = _build_operations_list({})

        assert len(ops) == 3
        assert ops[0][0] == "utilityFunction"
        assert ops[1][0] == "splitMeshes"
        assert ops[2][0] == "deduplicateGeometry"


# ---------------------------------------------------------------------------
# optimize_usd_local tests
# ---------------------------------------------------------------------------


class TestOptimizeUsdLocalMissingEnv:
    """Test that missing env var or bad layout raises RuntimeError."""

    def test_missing_so_package_dir(self, monkeypatch, tmp_path):
        monkeypatch.delenv("WU_SO_PACKAGE_DIR", raising=False)
        # Pin CWD so the auto-discovery fallback (.build-resources/...) points
        # somewhere that does not exist, forcing the "not found" error.
        monkeypatch.chdir(tmp_path)

        with pytest.raises(RuntimeError, match="WU_SO_PACKAGE_DIR"):
            optimize_usd_local("/tmp/in.usd", "/tmp/out.usd")

    def test_missing_subdirectory(self, monkeypatch, tmp_path):
        """Package dir without the expected subdirs raises with package path."""
        so_dir = tmp_path / "so_pkg"
        (so_dir / "python").mkdir(parents=True)
        (so_dir / "lib").mkdir()
        (so_dir / "extraLibs").mkdir()
        # Missing `usdpy/` — the new single-dir contract

        monkeypatch.setenv("WU_SO_PACKAGE_DIR", str(so_dir))

        with pytest.raises(
            RuntimeError, match="Scene Optimizer package directory missing"
        ):
            optimize_usd_local("/tmp/in.usd", "/tmp/out.usd")

    def test_auto_discovers_build_resources_when_env_unset(self, monkeypatch, tmp_path):
        """With WU_SO_PACKAGE_DIR unset, resolver picks up the fetch-script default."""
        from world_understanding.functions.graphics.scene_optimizer_local import (
            _resolve_so_package_dir,
        )

        monkeypatch.delenv("WU_SO_PACKAGE_DIR", raising=False)

        so_dir = tmp_path / ".build-resources" / "scene_optimizer_core"
        (so_dir / "python").mkdir(parents=True)
        (so_dir / "lib").mkdir()
        (so_dir / "extraLibs").mkdir()
        (so_dir / "usdpy").mkdir()
        monkeypatch.chdir(tmp_path)

        assert _resolve_so_package_dir() == so_dir


class TestOptimizeUsdLocalSubprocess:
    """Test subprocess invocation with mocked subprocess.run."""

    @pytest.fixture()
    def env_setup(self, monkeypatch, tmp_path):
        """Set up valid SO Core package layout and env vars."""
        so_dir = tmp_path / "so_pkg"
        (so_dir / "python").mkdir(parents=True)
        (so_dir / "lib").mkdir()
        (so_dir / "extraLibs").mkdir()
        (so_dir / "usdpy").mkdir()

        monkeypatch.setenv("WU_SO_PACKAGE_DIR", str(so_dir))
        monkeypatch.setenv("WU_SO_PYTHON", "python3.12")

        return {"so_dir": so_dir}

    def test_subprocess_env_vars(self, env_setup, monkeypatch, tmp_path):
        """Verify LD_LIBRARY_PATH, PYTHONPATH, PXR_PLUGINPATH_NAME are set correctly."""
        so_dir = env_setup["so_dir"]

        # Prepare a manifest that the "subprocess" would write
        manifest = {
            "status": "success",
            "optimization_time": 1.5,
            "operations_executed": [{"name": "merge", "success": True, "time": 0.5}],
            "stage_size_bytes": 1024,
        }

        def mock_run(cmd, **kwargs):
            # Parse params from the command to find manifest_path
            params = json.loads(cmd[2])
            manifest_path = params["manifest_path"]
            with open(manifest_path, "w") as f:
                json.dump(manifest, f)

            # Verify environment variables
            env = kwargs.get("env", {})
            ld_path = env.get("LD_LIBRARY_PATH", "")
            py_path = env.get("PYTHONPATH", "")
            plug_path = env.get("PXR_PLUGINPATH_NAME", "")

            assert str(so_dir / "lib") in ld_path
            assert str(so_dir / "extraLibs") in ld_path

            assert str(so_dir / "python") in py_path
            assert str(so_dir / "usdpy") in py_path

            assert str(so_dir / "extraLibs" / "usd") == plug_path

            result = MagicMock()
            result.returncode = 0
            result.stdout = ""
            result.stderr = ""
            return result

        with patch("subprocess.run", side_effect=mock_run):
            input_path = tmp_path / "input.usd"
            input_path.touch()
            output_path = tmp_path / "output.usd"

            result = optimize_usd_local(
                input_path=input_path,
                output_path=output_path,
                optimization_config={"scene_optimizer_settings": {}},
            )

        assert result["status"] == "success"
        assert result["optimization_time"] == 1.5
        assert result["stage_size_bytes"] == 1024
        assert len(result["operations_executed"]) == 1

    def test_subprocess_params_json(self, env_setup, tmp_path):
        """Verify the params JSON passed to the subprocess."""
        captured_params = {}

        def mock_run(cmd, **kwargs):
            params = json.loads(cmd[2])
            captured_params.update(params)

            manifest_path = params["manifest_path"]
            with open(manifest_path, "w") as f:
                json.dump({"status": "success", "optimization_time": 0.1}, f)

            result = MagicMock()
            result.returncode = 0
            result.stdout = ""
            result.stderr = ""
            return result

        with patch("subprocess.run", side_effect=mock_run):
            input_path = tmp_path / "input.usd"
            input_path.touch()
            output_path = tmp_path / "output.usd"

            optimize_usd_local(
                input_path=input_path,
                output_path=output_path,
                optimization_config={
                    "scene_optimizer_settings": {
                        "enable_split_meshes": True,
                        "enable_deduplicate": False,
                        "enable_deinstance": False,
                    }
                },
            )

        assert captured_params["input_usd_path"] == str(input_path)
        assert captured_params["output_usd_path"] == str(output_path)
        # splitMeshes enabled, dedup disabled, no merge
        assert len(captured_params["operations"]) == 1
        assert captured_params["operations"][0][0] == "splitMeshes"

    def test_subprocess_failure(self, env_setup, tmp_path):
        """Verify RuntimeError is raised on subprocess failure."""

        def mock_run(cmd, **kwargs):
            result = MagicMock()
            result.returncode = 1
            result.stdout = "some output"
            result.stderr = "segfault"
            return result

        with patch("subprocess.run", side_effect=mock_run):
            input_path = tmp_path / "input.usd"
            input_path.touch()
            output_path = tmp_path / "output.usd"

            with pytest.raises(RuntimeError, match="Scene Optimizer subprocess failed"):
                optimize_usd_local(
                    input_path=input_path,
                    output_path=output_path,
                )

    def test_pythonpath_stripped(self, env_setup, monkeypatch, tmp_path):
        """Verify parent PYTHONPATH is replaced, not appended to."""
        monkeypatch.setenv("PYTHONPATH", "/some/parent/path")

        def mock_run(cmd, **kwargs):
            env = kwargs.get("env", {})
            py_path = env.get("PYTHONPATH", "")
            # Parent PYTHONPATH must NOT leak into the subprocess
            assert "/some/parent/path" not in py_path

            params = json.loads(cmd[2])
            with open(params["manifest_path"], "w") as f:
                json.dump({"status": "success", "optimization_time": 0.1}, f)

            result = MagicMock()
            result.returncode = 0
            result.stdout = ""
            result.stderr = ""
            return result

        with patch("subprocess.run", side_effect=mock_run):
            input_path = tmp_path / "input.usd"
            input_path.touch()
            output_path = tmp_path / "output.usd"

            optimize_usd_local(input_path, output_path)

    def test_correspondence_map_passthrough(self, env_setup, tmp_path):
        """Verify correspondence_map from manifest is returned in result."""
        correspondence_map = {
            "summary": {
                "operations_run": {"split": True, "deduplicate": True},
                "total_original_prims": 3,
            },
            "split_mapping": {
                "/World/Mesh1": ["/World/Mesh1_part", "/World/Mesh1_part_1"],
            },
            "deduplication_mapping": {
                "instance_to_prototype": {
                    "/World/Mesh2/Geometry": "/World/Mesh1/Geometry",
                },
            },
            "full_mapping": {
                "original_to_prototype": {
                    "/World/Mesh1": [
                        "/World/Mesh1_part/Geometry",
                        "/World/Mesh1_part_1/Geometry",
                    ],
                    "/World/Mesh2": ["/World/Mesh1/Geometry"],
                    "/World/Mesh3": ["/World/Mesh3"],
                },
            },
        }
        manifest = {
            "status": "success",
            "optimization_time": 0.5,
            "operations_executed": [],
            "stage_size_bytes": 100,
            "correspondence_map": correspondence_map,
        }

        def mock_run(cmd, **kwargs):
            params = json.loads(cmd[2])
            with open(params["manifest_path"], "w") as f:
                json.dump(manifest, f)
            result = MagicMock()
            result.returncode = 0
            result.stdout = ""
            result.stderr = ""
            return result

        with patch("subprocess.run", side_effect=mock_run):
            input_path = tmp_path / "input.usd"
            input_path.touch()
            output_path = tmp_path / "output.usd"

            result = optimize_usd_local(input_path, output_path)

        assert result["correspondence_map"] == correspondence_map
        assert result["correspondence_map"]["split_mapping"]["/World/Mesh1"] == [
            "/World/Mesh1_part",
            "/World/Mesh1_part_1",
        ]
        full = result["correspondence_map"]["full_mapping"]["original_to_prototype"]
        assert "/World/Mesh1" in full
        assert "/World/Mesh2" in full
        assert "/World/Mesh3" in full

    def test_correspondence_map_defaults_to_empty(self, env_setup, tmp_path):
        """When manifest has no correspondence_map, result defaults to {}."""
        manifest = {
            "status": "success",
            "optimization_time": 0.1,
            "operations_executed": [],
        }

        def mock_run(cmd, **kwargs):
            params = json.loads(cmd[2])
            with open(params["manifest_path"], "w") as f:
                json.dump(manifest, f)
            result = MagicMock()
            result.returncode = 0
            result.stdout = ""
            result.stderr = ""
            return result

        with patch("subprocess.run", side_effect=mock_run):
            input_path = tmp_path / "input.usd"
            input_path.touch()

            result = optimize_usd_local(input_path, tmp_path / "output.usd")

        assert result["correspondence_map"] == {}


# ---------------------------------------------------------------------------
# Worker script helper function tests
# ---------------------------------------------------------------------------


class TestNaturalSortKey:
    """Test _natural_sort_key for correct numerical ordering."""

    def test_basic(self):
        paths = ["/World/Mesh_part_10", "/World/Mesh_part_2", "/World/Mesh_part_1"]
        result = sorted(paths, key=_natural_sort_key)
        assert result == [
            "/World/Mesh_part_1",
            "/World/Mesh_part_2",
            "/World/Mesh_part_10",
        ]

    def test_no_numbers(self):
        paths = ["/World/C", "/World/A", "/World/B"]
        result = sorted(paths, key=_natural_sort_key)
        assert result == ["/World/A", "/World/B", "/World/C"]

    def test_mixed(self):
        paths = ["/World/Mesh_part", "/World/Mesh_part_1", "/World/Mesh_part_0"]
        result = sorted(paths, key=_natural_sort_key)
        assert result[0] == "/World/Mesh_part"


class TestWorkerTrackSplitMeshes:
    """Test track_split_meshes from the worker module."""

    def test_no_changes(self):
        """No meshes removed or added — empty split mapping."""
        assert (
            track_split_meshes(["/World/A", "/World/B"], ["/World/A", "/World/B"]) == {}
        )

    def test_single_split(self):
        """One mesh split into two parts."""
        result = track_split_meshes(
            ["/World/A", "/World/B"],
            ["/World/A", "/World/B_part", "/World/B_part_1"],
        )
        assert result == {"/World/B": ["/World/B_part", "/World/B_part_1"]}

    def test_multiple_splits(self):
        """Multiple meshes split."""
        result = track_split_meshes(
            ["/World/A", "/World/B", "/World/C"],
            [
                "/World/A_part",
                "/World/A_part_1",
                "/World/B",
                "/World/C_part",
                "/World/C_part_1",
                "/World/C_part_2",
            ],
        )
        assert result["/World/A"] == ["/World/A_part", "/World/A_part_1"]
        assert len(result["/World/C"]) == 3
        assert "/World/B" not in result

    def test_nested_paths(self):
        """Split detection works with nested prim paths."""
        result = track_split_meshes(
            ["/World/Group/Mesh1"],
            ["/World/Group/Mesh1_part", "/World/Group/Mesh1_part_1"],
        )
        assert result == {
            "/World/Group/Mesh1": [
                "/World/Group/Mesh1_part",
                "/World/Group/Mesh1_part_1",
            ]
        }


class TestWorkerBuildCorrespondenceMap:
    """Test build_correspondence_map from the worker module."""

    def test_identity_no_ops(self):
        """No split or dedup — all meshes map to themselves."""
        result = build_correspondence_map(
            ["/World/A", "/World/B"], {}, {}, False, False
        )

        o2p = result["full_mapping"]["original_to_prototype"]
        assert o2p == {"/World/A": ["/World/A"], "/World/B": ["/World/B"]}
        assert result["split_mapping"] == {}
        assert result["deduplication_mapping"]["instance_to_prototype"] == {}

    def test_split_only(self):
        """Split without dedup — parts map directly."""
        split = {"/World/A": ["/World/A_part", "/World/A_part_1"]}
        result = build_correspondence_map(
            ["/World/A", "/World/B"], split, {}, True, False
        )

        o2p = result["full_mapping"]["original_to_prototype"]
        assert o2p["/World/A"] == ["/World/A_part", "/World/A_part_1"]
        assert o2p["/World/B"] == ["/World/B"]

    def test_dedup_only(self):
        """Dedup without split — instances map to prototypes."""
        i2p = {"/World/B": "/World/A", "/World/C": "/World/A"}
        result = build_correspondence_map(
            ["/World/A", "/World/B", "/World/C"], {}, i2p, False, True
        )

        o2p = result["full_mapping"]["original_to_prototype"]
        assert o2p["/World/B"] == ["/World/A"]
        assert o2p["/World/C"] == ["/World/A"]

    def test_split_then_dedup(self):
        """Split + dedup — chained mapping."""
        split = {
            "/World/A": ["/World/A_part", "/World/A_part_1"],
            "/World/B": ["/World/B_part", "/World/B_part_1"],
        }
        i2p = {"/World/B_part": "/World/A_part", "/World/B_part_1": "/World/A_part_1"}
        result = build_correspondence_map(
            ["/World/A", "/World/B"], split, i2p, True, True
        )

        o2p = result["full_mapping"]["original_to_prototype"]
        assert o2p["/World/A"] == ["/World/A_part", "/World/A_part_1"]
        assert o2p["/World/B"] == ["/World/A_part", "/World/A_part_1"]

    def test_summary_structure(self):
        """Summary has expected keys and values."""
        split = {"/World/A": ["/World/A_part", "/World/A_part_1"]}
        i2p = {"/World/B": "/World/C"}
        result = build_correspondence_map(
            ["/World/A", "/World/B", "/World/C"], split, i2p, True, True
        )

        summary = result["summary"]
        assert summary["total_original_prims"] == 3
        assert summary["meshes_split"] == 1
        assert summary["instances_tracked"] == 1
