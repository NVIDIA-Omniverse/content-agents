# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for USD data preparation tasks."""

import os
from unittest.mock import Mock, patch

import pytest
import yaml

from world_understanding.agentic.usd_tasks import (
    USDDataPrepConfigTask,
    USDDatasetManifestTask,
    USDLoadingTask,
    USDPrimTraversalAndRenderingTask,
    USDRendererProvisioningTask,
)
from world_understanding.agentic.usd_tasks.prim_traversal import (
    prim_path_to_directory_structure,
)
from world_understanding.functions.graphics.rendering import (
    CameraViewType,
    NVCFRenderingBackend,
    RenderingConfig,
)
from world_understanding.utils.object_store import ObjectStore
from world_understanding.utils.usd.stage import MAX_PATH_COMPONENT_LEN


class TestUSDDataPrepTasks:
    """Basic tests for USD data preparation tasks."""

    def test_usd_data_prep_config_task_init(self):
        """Test USDDataPrepConfigTask initialization."""
        task = USDDataPrepConfigTask()
        assert task.name == "USDDataPrepConfig"
        assert (
            task.description == "Load and validate USD data preparation configuration"
        )
        assert hasattr(task, "run")

    def test_usd_renderer_provisioning_task_init(self):
        """Test USDRendererProvisioningTask initialization."""
        task = USDRendererProvisioningTask()
        assert task.name == "USDRendererProvisioning"
        assert task.description == "Provision USD renderer backend and resources"
        assert hasattr(task, "run")

    def test_usd_loading_task_init(self):
        """Test USDLoadingTask initialization."""
        task = USDLoadingTask()
        assert task.name == "USDLoading"
        assert task.description == "Load USD stage and apply configurations"
        assert hasattr(task, "run")

    def test_usd_prim_traversal_task_init(self):
        """Test USDPrimTraversalAndRenderingTask initialization."""
        task = USDPrimTraversalAndRenderingTask()
        assert task.name == "USDPrimTraversalAndRendering"
        assert task.description == "Traverse USD prims and render configured views"
        assert hasattr(task, "run")

    def test_usd_dataset_manifest_task_init(self):
        """Test USDDatasetManifestTask initialization."""
        task = USDDatasetManifestTask()
        assert task.name == "USDDatasetManifest"
        assert (
            task.description == "Create dataset.json and prims.jsonl files for USD data"
        )
        assert hasattr(task, "run")

    def test_all_tasks_are_importable(self):
        """Test that all USD data prep tasks can be imported."""
        # All tasks are already imported at the top of the file
        # This test verifies they are accessible
        assert USDDataPrepConfigTask is not None
        assert USDDatasetManifestTask is not None
        assert USDLoadingTask is not None
        assert USDPrimTraversalAndRenderingTask is not None
        assert USDRendererProvisioningTask is not None

    def test_tasks_inherit_from_base_task(self):
        """Test that all USD tasks inherit from the base Task class."""
        from world_understanding.agentic.tasks import Task

        assert issubclass(USDDataPrepConfigTask, Task)
        assert issubclass(USDDatasetManifestTask, Task)
        assert issubclass(USDLoadingTask, Task)
        assert issubclass(USDPrimTraversalAndRenderingTask, Task)
        assert issubclass(USDRendererProvisioningTask, Task)

    def test_config_task_propagates_max_concurrent_requests(self, tmp_path):
        """Config should pass render concurrency through to traversal."""
        config_path = tmp_path / "build_dataset_usd.yaml"
        config_path.write_text(
            "\n".join(
                [
                    "usd_path: scene.usd",
                    "output_dir: output",
                    "batch_size: 64",
                    "num_workers: 1",
                    "max_concurrent_requests: 1",
                ]
            )
        )

        task = USDDataPrepConfigTask()
        result = task.run({"config_path": str(config_path)}, Mock(spec=ObjectStore))

        assert result["batch_size"] == 64
        assert result["num_workers"] == 1
        assert result["max_concurrent_requests"] == 1

    @pytest.mark.parametrize(
        ("field_name", "value"),
        [
            ("max_concurrent_requests", 0),
            ("max_concurrent_requests", -1),
            ("max_concurrent_requests", "1"),
            ("max_concurrent_requests", True),
            ("num_workers", 0),
            ("num_workers", -1),
            ("num_workers", "1"),
            ("num_workers", False),
        ],
    )
    def test_config_task_rejects_invalid_render_concurrency(
        self, tmp_path, field_name, value
    ):
        """Render concurrency settings must be positive integers."""
        config_path = tmp_path / "build_dataset_usd.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "usd_path": "scene.usd",
                    "output_dir": "output",
                    field_name: value,
                }
            )
        )

        task = USDDataPrepConfigTask()
        with pytest.raises(
            ValueError, match=f"{field_name} must be a positive integer"
        ):
            task.run({"config_path": str(config_path)}, Mock(spec=ObjectStore))


class TestUSDDataPrepConfigTask:
    """Functional tests for USDDataPrepConfigTask."""

    def test_run_plumbs_max_concurrent_requests(self, tmp_path):
        """Config should carry async render request concurrency into context."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "usd_path": "scene.usd",
                    "output_dir": "dataset",
                    "batch_size": 16,
                    "max_concurrent_requests": 4,
                }
            ),
            encoding="utf-8",
        )
        context = {"config_path": str(config_path)}
        object_store = Mock(spec=ObjectStore)

        result = USDDataPrepConfigTask().run(context, object_store)

        assert result["usd_path"] == tmp_path / "scene.usd"
        assert result["output_dir"] == tmp_path / "dataset"
        assert result["batch_size"] == 16
        assert result["max_concurrent_requests"] == 4


class TestUSDRendererProvisioningTask:
    """Functional tests for USDRendererProvisioningTask."""

    def test_run_with_nvcf_backend(self):
        """Test running with NVCF backend configuration."""
        task = USDRendererProvisioningTask()
        context = {
            "renderer_config": {
                "backend": "remote",
                "image_width": 1920,
                "image_height": 1080,
            }
        }
        object_store = Mock(spec=ObjectStore)

        with patch.dict(os.environ, {"NGC_API_KEY": "test-api-key"}):
            result = task.run(context, object_store)

        # Verify NVCF backend was created
        assert "rendering_backend" in result
        assert isinstance(result["rendering_backend"], NVCFRenderingBackend)
        assert result["rendering_backend"].api_key == "test-api-key"

        # Verify custom image dimensions
        assert result["image_height"] == 1080

    def test_run_with_unknown_backend_raises_error(self):
        """Test that unknown backend raises ValueError."""
        task = USDRendererProvisioningTask()
        context = {
            "renderer_config": {
                "backend": "unknown_backend",
            }
        }
        object_store = Mock(spec=ObjectStore)

        with pytest.raises(
            ValueError, match="Unknown USD renderer backend: unknown_backend"
        ):
            task.run(context, object_store)

    def test_run_with_custom_configuration(self):
        """Test running with custom configuration values."""
        task = USDRendererProvisioningTask()
        context = {
            "renderer_config": {
                "backend": "remote",
                "image_width": 2048,
                "image_height": 1536,
                "cull_style": "front",
                "should_highlight_prim": False,
                "should_assign_random_colors": True,
            }
        }
        object_store = Mock(spec=ObjectStore)

        result = task.run(context, object_store)

        # Verify custom configuration was applied
        config = result["rendering_config"]
        assert config.image_width == 2048
        assert config.cull_style == "front"
        assert config.should_highlight_prim is False
        assert config.should_assign_random_colors is True
        assert result["image_height"] == 1536

    def test_run_with_minimal_configuration(self):
        """Test running with minimal configuration (only backend specified)."""
        task = USDRendererProvisioningTask()
        context = {
            "renderer_config": {
                "backend": "remote",
            }
        }
        object_store = Mock(spec=ObjectStore)

        result = task.run(context, object_store)

        # Verify defaults were applied
        config = result["rendering_config"]
        assert config.image_width == 512  # default
        assert config.cull_style == "back"  # default
        assert config.should_highlight_prim is False  # default
        assert config.should_assign_random_colors is True  # default
        assert result["image_height"] == 512  # default

    def test_run_with_empty_renderer_config(self):
        """Test running with empty renderer_config dict."""
        task = USDRendererProvisioningTask()
        context = {"renderer_config": {}}
        object_store = Mock(spec=ObjectStore)

        result = task.run(context, object_store)

        # Verify all defaults were applied
        config = result["rendering_config"]
        assert config.image_width == 512
        assert config.cull_style == "back"
        assert config.should_highlight_prim is False
        assert config.should_assign_random_colors is True
        assert result["image_height"] == 512

    def test_run_without_renderer_config_key(self):
        """Test running when renderer_config key is missing from context."""
        task = USDRendererProvisioningTask()
        context = {}  # No renderer_config key
        object_store = Mock(spec=ObjectStore)

        result = task.run(context, object_store)

        # Verify defaults were applied
        config = result["rendering_config"]
        assert config.image_width == 512
        assert config.cull_style == "back"
        assert config.should_highlight_prim is False
        assert config.should_assign_random_colors is True
        assert result["image_height"] == 512

    def test_run_with_corner_camera_view_type(self):
        """Test running with corner camera view type (default)."""
        task = USDRendererProvisioningTask()
        context = {
            "renderer_config": {
                "backend": "remote",
                "camera_view_type": "corner",
            }
        }
        object_store = Mock(spec=ObjectStore)

        result = task.run(context, object_store)

        # Verify corner view configuration
        config = result["rendering_config"]
        assert config.camera_view_type == CameraViewType.CORNER
        assert config.camera_name_prefix == "CornerViewCamera"

        # Verify default corner camera directions (8 corners)
        expected_directions = [
            "+x+y+z",
            "-x+y+z",
            "-x-y+z",
            "+x-y+z",
            "+x+y-z",
            "-x+y-z",
            "-x-y-z",
            "+x-y-z",
        ]
        assert config.camera_ordering == expected_directions

    def test_run_with_side_camera_view_type(self):
        """Test running with side camera view type."""
        task = USDRendererProvisioningTask()
        context = {
            "renderer_config": {
                "backend": "remote",
                "camera_view_type": "side",
            }
        }
        object_store = Mock(spec=ObjectStore)

        result = task.run(context, object_store)

        # Verify side view configuration
        config = result["rendering_config"]
        assert config.camera_view_type == CameraViewType.SIDE
        assert config.camera_name_prefix == "SideViewCamera"

        # Verify default side camera directions (6 cardinal directions)
        expected_directions = ["+x", "-x", "+y", "-y", "+z", "-z"]
        assert config.camera_ordering == expected_directions

    def test_run_with_custom_camera_directions(self):
        """Test running with custom camera directions."""
        task = USDRendererProvisioningTask()
        custom_directions = ["+x", "+y", "+z"]
        context = {
            "renderer_config": {
                "backend": "remote",
                "camera_view_type": "corner",
                "camera_directions": custom_directions,
            }
        }
        object_store = Mock(spec=ObjectStore)

        result = task.run(context, object_store)

        # Verify custom directions were used
        config = result["rendering_config"]
        assert config.camera_view_type == CameraViewType.CORNER
        assert config.camera_ordering == custom_directions

    def test_run_with_invalid_camera_view_type_falls_back_to_corner(self):
        """Test that invalid camera view type falls back to corner."""
        task = USDRendererProvisioningTask()
        context = {
            "renderer_config": {
                "backend": "remote",
                "camera_view_type": "invalid_type",
            }
        }
        object_store = Mock(spec=ObjectStore)

        # Mock the event listener to capture warning calls
        mock_listener = Mock()
        with patch(
            "world_understanding.agentic.usd_tasks.renderer.get_listener",
            return_value=mock_listener,
        ):
            result = task.run(context, object_store)

        # Verify fallback to corner view
        config = result["rendering_config"]
        assert config.camera_view_type == CameraViewType.CORNER
        assert config.camera_name_prefix == "CornerViewCamera"

        # Verify warning was logged through the event listener
        mock_listener.warning.assert_called_once()
        warning_call = mock_listener.warning.call_args[0][0]
        assert "Invalid camera_view_type 'invalid_type', using 'corner'" in warning_call

    def test_run_with_case_insensitive_camera_view_type(self):
        """Test that camera view type is case insensitive."""
        task = USDRendererProvisioningTask()
        context = {
            "renderer_config": {
                "backend": "remote",
                "camera_view_type": "SIDE",  # uppercase
            }
        }
        object_store = Mock(spec=ObjectStore)

        result = task.run(context, object_store)

        # Verify side view was recognized despite case
        config = result["rendering_config"]
        assert config.camera_view_type == CameraViewType.SIDE
        assert config.camera_name_prefix == "SideViewCamera"

    def test_run_with_nvcf_api_key_from_environment(self):
        """Test running with NVCF API key from environment."""
        task = USDRendererProvisioningTask()
        context = {
            "renderer_config": {
                "backend": "remote",
            }
        }
        object_store = Mock(spec=ObjectStore)

        api_key = "test-nvcf-api-key-12345"
        with patch.dict(os.environ, {"NGC_API_KEY": api_key}):
            result = task.run(context, object_store)

        # Verify API key was used
        backend = result["rendering_backend"]
        assert isinstance(backend, NVCFRenderingBackend)
        assert backend.api_key == api_key

    def test_run_with_nvcf_no_api_key(self):
        """Test running with NVCF backend when API key is not set."""
        task = USDRendererProvisioningTask()
        context = {
            "renderer_config": {
                "backend": "remote",
            }
        }
        object_store = Mock(spec=ObjectStore)

        # Ensure NGC_API_KEY is not set
        with patch.dict(os.environ, {}, clear=True):
            result = task.run(context, object_store)

        # Verify backend was created with None API key
        backend = result["rendering_backend"]
        assert isinstance(backend, NVCFRenderingBackend)
        assert backend.api_key is None

    def test_run_with_multiple_environment_variables(self):
        """Test running with multiple environment variables set."""
        task = USDRendererProvisioningTask()
        context = {
            "renderer_config": {
                "backend": "remote",
            }
        }
        object_store = Mock(spec=ObjectStore)

        env_vars = {
            "NGC_API_KEY": "nvcf-key-123",
            "OTHER_VAR": "ignored",
        }

        with patch.dict(os.environ, env_vars):
            result = task.run(context, object_store)

        # Verify only relevant environment variable was used
        backend = result["rendering_backend"]
        assert isinstance(backend, NVCFRenderingBackend)
        assert backend.api_key == "nvcf-key-123"

    def test_run_with_none_context_raises_error(self):
        """Test that None context raises appropriate error."""
        task = USDRendererProvisioningTask()
        object_store = Mock(spec=ObjectStore)

        with pytest.raises(AttributeError):
            task.run(None, object_store)

    def test_run_with_none_object_store_raises_error(self):
        """Test that None object store raises appropriate error."""
        task = USDRendererProvisioningTask()
        context = {"renderer_config": {}}

        with pytest.raises(AttributeError):
            task.run(context, None)

    def test_run_with_invalid_renderer_config_type(self):
        """Test that invalid renderer_config type raises AttributeError."""
        task = USDRendererProvisioningTask()
        context = {"renderer_config": "invalid_string"}  # Should be dict
        object_store = Mock(spec=ObjectStore)

        # This should raise an AttributeError because strings don't have .get() method
        with pytest.raises(AttributeError, match="'str' object has no attribute 'get'"):
            task.run(context, object_store)

    def test_run_with_malformed_camera_directions(self):
        """Test handling of malformed camera directions."""
        task = USDRendererProvisioningTask()
        context = {
            "renderer_config": {
                "backend": "remote",
                "camera_view_type": "corner",
                "camera_directions": "not_a_list",  # Should be list
            }
        }
        object_store = Mock(spec=ObjectStore)

        # This should not raise an error, but should use the malformed value
        result = task.run(context, object_store)

        # Verify the malformed value was used as-is
        config = result["rendering_config"]
        assert config.camera_ordering == "not_a_list"

    def test_run_with_empty_camera_directions(self):
        """Test handling of empty camera directions list."""
        task = USDRendererProvisioningTask()
        context = {
            "renderer_config": {
                "backend": "remote",
                "camera_view_type": "corner",
                "camera_directions": [],  # Empty list
            }
        }
        object_store = Mock(spec=ObjectStore)

        result = task.run(context, object_store)

        # Verify empty list was used
        config = result["rendering_config"]
        assert config.camera_ordering == []

    def test_run_with_negative_image_dimensions(self):
        """Test handling of negative image dimensions."""
        task = USDRendererProvisioningTask()
        context = {
            "renderer_config": {
                "backend": "remote",
                "image_width": -100,  # Negative width
                "image_height": -50,  # Negative height
            }
        }
        object_store = Mock(spec=ObjectStore)

        # This should not raise an error, but should use the negative values
        result = task.run(context, object_store)

        # Verify negative values were used (validation happens elsewhere)
        config = result["rendering_config"]
        assert config.image_width == -100
        assert result["image_height"] == -50

    def test_run_with_zero_image_dimensions(self):
        """Test handling of zero image dimensions."""
        task = USDRendererProvisioningTask()
        context = {
            "renderer_config": {
                "backend": "remote",
                "image_width": 0,
                "image_height": 0,
            }
        }
        object_store = Mock(spec=ObjectStore)

        result = task.run(context, object_store)

        # Verify zero values were used
        config = result["rendering_config"]
        assert config.image_width == 0
        assert result["image_height"] == 0

    def test_run_stores_backend_in_object_store(self):
        """Test that backend is properly stored in object store."""
        task = USDRendererProvisioningTask()
        context = {
            "renderer_config": {
                "backend": "remote",
            }
        }
        object_store = Mock(spec=ObjectStore)

        task.run(context, object_store)

        # Verify backend was stored in object store
        backend_calls = [
            call
            for call in object_store.set.call_args_list
            if call[0][0] == "rendering_backend"
        ]
        assert len(backend_calls) == 1
        assert isinstance(backend_calls[0][0][1], NVCFRenderingBackend)

    def test_run_stores_config_in_object_store(self):
        """Test that config is properly stored in object store."""
        task = USDRendererProvisioningTask()
        context = {
            "renderer_config": {
                "backend": "remote",
                "image_width": 1024,
            }
        }
        object_store = Mock(spec=ObjectStore)

        task.run(context, object_store)

        # Verify config was stored in object store
        config_calls = [
            call
            for call in object_store.set.call_args_list
            if call[0][0] == "rendering_config"
        ]
        assert len(config_calls) == 1
        assert isinstance(config_calls[0][0][1], RenderingConfig)
        assert config_calls[0][0][1].image_width == 1024

    def test_run_updates_context_with_backend_and_config(self):
        """Test that context is updated with backend and config references."""
        task = USDRendererProvisioningTask()
        context = {
            "renderer_config": {
                "backend": "remote",
            }
        }
        object_store = Mock(spec=ObjectStore)

        result = task.run(context, object_store)

        # Verify context was updated
        assert "rendering_backend" in result
        assert "rendering_config" in result
        assert "image_height" in result

        # Verify the objects are the same as those stored in object store
        backend_call = next(
            call
            for call in object_store.set.call_args_list
            if call[0][0] == "rendering_backend"
        )
        config_call = next(
            call
            for call in object_store.set.call_args_list
            if call[0][0] == "rendering_config"
        )

        assert result["rendering_backend"] is backend_call[0][1]
        assert result["rendering_config"] is config_call[0][1]

    def test_run_preserves_existing_context_keys(self):
        """Test that existing context keys are preserved."""
        task = USDRendererProvisioningTask()
        context = {
            "existing_key": "existing_value",
            "another_key": 123,
            "renderer_config": {
                "backend": "remote",
            },
        }
        object_store = Mock(spec=ObjectStore)

        result = task.run(context, object_store)

        # Verify existing keys are preserved
        assert result["existing_key"] == "existing_value"
        assert result["another_key"] == 123

        # Verify new keys were added
        assert "rendering_backend" in result
        assert "rendering_config" in result
        assert "image_height" in result

    def test_run_object_store_called_twice(self):
        """Test that object store set method is called exactly twice."""
        task = USDRendererProvisioningTask()
        context = {
            "renderer_config": {
                "backend": "remote",
            }
        }
        object_store = Mock(spec=ObjectStore)

        task.run(context, object_store)

        # Verify set was called exactly twice
        assert object_store.set.call_count == 2

        # Verify the keys that were stored
        stored_keys = [call[0][0] for call in object_store.set.call_args_list]
        assert "rendering_backend" in stored_keys
        assert "rendering_config" in stored_keys

    def test_run_with_nvcf_stores_correct_backend_type(self):
        """Test that NVCF backend is stored with correct type."""
        task = USDRendererProvisioningTask()
        context = {
            "renderer_config": {
                "backend": "remote",
            }
        }
        object_store = Mock(spec=ObjectStore)

        with patch.dict(os.environ, {"NGC_API_KEY": "test-key"}):
            task.run(context, object_store)

        # Verify NVCF backend was stored
        backend_call = next(
            call
            for call in object_store.set.call_args_list
            if call[0][0] == "rendering_backend"
        )
        stored_backend = backend_call[0][1]
        assert isinstance(stored_backend, NVCFRenderingBackend)
        assert stored_backend.api_key == "test-key"


class TestUSDPrimTraversalZeroImages:
    """Tests that USDPrimTraversalAndRenderingTask fails when no images are rendered."""

    def test_raises_on_zero_images(self, tmp_path):
        """Task must raise RuntimeError when total_images_rendered is 0."""
        task = USDPrimTraversalAndRenderingTask()

        mock_stage = Mock()
        mock_backend = Mock()
        rendering_config = RenderingConfig()

        object_store = Mock(spec=ObjectStore)
        object_store.get.side_effect = lambda key, default=None: {
            "usd_stage": mock_stage,
            "rendering_backend": mock_backend,
            "rendering_config": rendering_config,
            "usd_model": None,
        }.get(key, default)

        context: dict = {
            "prim_filters": {"types": ["UsdGeom.Mesh"]},
            "render_output_dir": str(tmp_path),
            "output_dir": str(tmp_path),
            "skip_existing": False,
            "batch_size": 10,
            "rendering_modes": ["prim_only"],
        }

        # Patch USD-dependent calls and return no prims → 0 images
        with (
            patch("world_understanding.agentic.usd_tasks.prim_traversal.UsdGeom"),
            patch(
                "world_understanding.agentic.usd_tasks.prim_traversal.get_stage_world_bbox",
                return_value=None,
            ),
            patch.object(task, "_collect_prims", return_value=[]),
            patch.object(task, "_collect_and_filter_prims", return_value=([], [], 0)),
            patch.object(task, "_prepare_stages", return_value={}),
            patch.object(task, "_upload_stages_to_s3", return_value=[]),
            patch.object(task, "_cleanup_s3"),
        ):
            with pytest.raises(RuntimeError, match="Rendering produced 0 images"):
                task.run(context, object_store)


class TestUSDPrimCollectionFilters:
    """Tests for USD prim collection filters."""

    def test_collect_prims_can_skip_invisible_meshes(self):
        """Hidden source/library meshes can be excluded from render datasets."""
        from pxr import Usd, UsdGeom

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.Mesh.Define(stage, "/World/VisibleMesh")
        hidden_parent = UsdGeom.Xform.Define(stage, "/World/HiddenLibrary")
        UsdGeom.Imageable(hidden_parent.GetPrim()).MakeInvisible()
        UsdGeom.Mesh.Define(stage, "/World/HiddenLibrary/HiddenMesh")

        task = USDPrimTraversalAndRenderingTask()
        prims = task._collect_prims(
            stage,
            {
                "types": ["UsdGeom.Mesh"],
                "skip_instances": False,
                "skip_invisible": True,
            },
            Mock(),
        )

        assert prims == ["/World/VisibleMesh"]

    def test_collect_prims_can_skip_invisible_specific_paths(self):
        """The skip_invisible filter also applies to explicit path lists."""
        from pxr import Usd, UsdGeom

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.Mesh.Define(stage, "/World/VisibleMesh")
        hidden_parent = UsdGeom.Xform.Define(stage, "/World/HiddenLibrary")
        UsdGeom.Imageable(hidden_parent.GetPrim()).MakeInvisible()
        UsdGeom.Mesh.Define(stage, "/World/HiddenLibrary/HiddenMesh")

        task = USDPrimTraversalAndRenderingTask()
        prims = task._collect_prims(
            stage,
            {
                "paths": [
                    "/World/VisibleMesh",
                    "/World/HiddenLibrary/HiddenMesh",
                ],
                "skip_instances": False,
                "skip_invisible": True,
            },
            Mock(),
        )

        assert prims == ["/World/VisibleMesh"]


class TestUSDDatasetManifestZeroImages:
    """Tests that USDDatasetManifestTask fails when no images are in the dataset."""

    def test_raises_on_zero_images(self, tmp_path):
        """Task must raise RuntimeError when dataset has 0 images."""
        task = USDDatasetManifestTask()

        # prim_data with a prim that has no rendered images
        prim_data = [
            {
                "prim_path": "/World/Mesh",
                "images": {},
                "metadata": {"type": "UsdGeom.Mesh"},
            }
        ]

        context: dict = {
            "output_dir": str(tmp_path),
            "export_usd_model": False,
            "usd_path": "test.usd",
        }
        object_store = Mock(spec=ObjectStore)
        object_store.get.side_effect = lambda key, default=None: {
            "prim_data": prim_data,
            "usd_model": None,
            "usd_stage": None,
        }.get(key, default)

        with pytest.raises(RuntimeError, match="Dataset has 0 images"):
            task.run(context, object_store)


class TestPrimPathToDirectoryStructure:
    """Tests for prim path to filesystem path conversion."""

    def test_long_path_segments_are_bounded(self, tmp_path):
        """Long prim path segments should be truncated and still create a file path."""
        long_segment = "segment_" + ("x" * 500)
        prim_path = f"/World/{long_segment}/{long_segment}"
        output_path = prim_path_to_directory_structure(
            prim_path=prim_path,
            base_dir=tmp_path,
            filename="render.png",
        )

        assert output_path.parent.exists()
        assert output_path.name == "render.png"
        assert len(output_path.parent.name) <= MAX_PATH_COMPONENT_LEN
        assert len(output_path.parent.parent.name) <= MAX_PATH_COMPONENT_LEN
