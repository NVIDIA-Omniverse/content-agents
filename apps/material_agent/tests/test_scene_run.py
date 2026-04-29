# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for material_agent.scene.run module."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest
import yaml

from material_agent.scene.manifest import (
    InstanceGroup,
    PayloadGroup,
    SceneManifest,
    SubAsset,
)
from material_agent.scene.run import (
    _patch_config_predict_max_workers,
    _run_parallel,
    run_all,
    run_payload,
    run_sub_asset,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sub_asset(
    name: str = "asset_a",
    prim_path: str = "/World/AssetA",
    config_path: str | None = "/tmp/cfg.yaml",
    status: str = "pending",
    mesh_count: int = 10,
    instance_group: str | None = None,
    asset_id: str | None = None,
) -> SubAsset:
    return SubAsset(
        id=asset_id or str(uuid.uuid4()),
        name=name,
        prim_path=prim_path,
        config_path=config_path,
        status=status,
        mesh_count=mesh_count,
        instance_group=instance_group,
    )


def _make_payload_group(
    group_name: str = "payload_a",
    config_path: str | None = "/tmp/pg_cfg.yaml",
    status: str = "pending",
    instance_count: int = 5,
    pg_id: str | None = None,
) -> PayloadGroup:
    return PayloadGroup(
        id=pg_id or str(uuid.uuid4()),
        group_name=group_name,
        payload_file="/tmp/payload.usd",
        config_path=config_path,
        status=status,
        instance_count=instance_count,
    )


@dataclass
class FakePipelineOutput:
    """Lightweight stand-in for PipelineOutput in tests."""

    success: bool
    error: str | None = None
    step_results: dict[str, dict[str, Any]] = field(default_factory=dict)
    completed_steps: list[str] = field(default_factory=list)
    skipped_steps: list[str] = field(default_factory=list)
    raw_result: dict[str, Any] | None = None


def _write_config(path: Path, session_id: str = "test_session") -> None:
    """Write a minimal YAML config file."""
    cfg = {"project": {"session_id": session_id}, "steps": {}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg))


# ---------------------------------------------------------------------------
# _patch_config_predict_max_workers
# ---------------------------------------------------------------------------


class TestPatchConfigPredictMaxWorkers:
    def test_creates_predict_section_if_missing(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.safe_dump({"project": {"session_id": "s1"}}))

        _patch_config_predict_max_workers(cfg_path, 8)

        result = yaml.safe_load(cfg_path.read_text())
        assert result["steps"]["predict"]["max_workers"] == 8

    def test_overwrites_existing_value(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.safe_dump({"steps": {"predict": {"max_workers": 64}}}))

        _patch_config_predict_max_workers(cfg_path, 4)

        result = yaml.safe_load(cfg_path.read_text())
        assert result["steps"]["predict"]["max_workers"] == 4

    def test_preserves_other_keys(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(
            yaml.safe_dump(
                {
                    "project": {"name": "test"},
                    "steps": {"predict": {"model": "gpt", "max_workers": 64}},
                }
            )
        )

        _patch_config_predict_max_workers(cfg_path, 2)

        result = yaml.safe_load(cfg_path.read_text())
        assert result["steps"]["predict"]["max_workers"] == 2
        assert result["steps"]["predict"]["model"] == "gpt"
        assert result["project"]["name"] == "test"


# ---------------------------------------------------------------------------
# run_sub_asset
# ---------------------------------------------------------------------------


class TestRunSubAsset:
    """Tests for run_sub_asset including SO retry logic."""

    def test_no_config_path_raises(self) -> None:
        sa = _make_sub_asset(config_path=None)
        with pytest.raises(ValueError, match="no config_path"):
            run_sub_asset(sa)

    def test_missing_config_file_raises(self, tmp_path: Path) -> None:
        sa = _make_sub_asset(config_path=str(tmp_path / "nonexistent.yaml"))
        with pytest.raises(FileNotFoundError):
            run_sub_asset(sa)

    @patch("material_agent.scene.run._update_output_paths")
    @patch("material_agent.api.pipeline.run_pipeline")
    def test_success_no_retry(
        self,
        mock_run: MagicMock,
        mock_update: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Pipeline succeeds on first try without SO — no retry."""
        cfg_path = tmp_path / "config.yaml"
        _write_config(cfg_path)
        sa = _make_sub_asset(config_path=str(cfg_path))

        mock_run.return_value = FakePipelineOutput(
            success=True,
            completed_steps=["build_dataset_usd", "predict"],
            step_results={"predict": {"predictions_count": 5}},
        )

        result = run_sub_asset(sa)
        assert result.status == "completed"
        assert mock_run.call_count == 1

    @patch("material_agent.scene.run._update_output_paths")
    @patch("material_agent.scene.run._clean_working_dir_for_so_retry")
    @patch("material_agent.api.pipeline.run_pipeline")
    def test_so_retry_when_pipeline_failed_after_so(
        self,
        mock_run: MagicMock,
        mock_clean: MagicMock,
        mock_update: MagicMock,
        tmp_path: Path,
    ) -> None:
        """SO ran but pipeline failed -> retry without SO."""
        cfg_path = tmp_path / "config.yaml"
        _write_config(cfg_path)
        sa = _make_sub_asset(config_path=str(cfg_path))

        # First call: SO ran but pipeline failed
        fail_result = FakePipelineOutput(
            success=False,
            error="predict failed",
            completed_steps=["optimize_usd", "build_dataset_usd"],
        )
        # Second call (retry): succeeds
        success_result = FakePipelineOutput(
            success=True,
            completed_steps=["build_dataset_usd", "predict"],
        )
        mock_run.side_effect = [fail_result, success_result]

        result = run_sub_asset(sa)
        assert result.status == "completed"
        assert mock_run.call_count == 2
        mock_clean.assert_called_once()

        # Verify retry call has optimize_usd in skip_steps
        retry_call = mock_run.call_args_list[1]
        retry_input = retry_call[0][0]  # first positional arg = PipelineInput
        assert "optimize_usd" in retry_input.skip_steps

    @patch("material_agent.scene.run._update_output_paths")
    @patch("material_agent.scene.run._clean_working_dir_for_so_retry")
    @patch("material_agent.api.pipeline.run_pipeline")
    def test_so_retry_when_zero_predictions(
        self,
        mock_run: MagicMock,
        mock_clean: MagicMock,
        mock_update: MagicMock,
        tmp_path: Path,
    ) -> None:
        """SO ran, pipeline succeeded, but 0 predictions -> retry without SO."""
        cfg_path = tmp_path / "config.yaml"
        _write_config(cfg_path)
        sa = _make_sub_asset(config_path=str(cfg_path))

        zero_pred = FakePipelineOutput(
            success=True,
            completed_steps=["optimize_usd", "build_dataset_usd", "predict"],
            step_results={"predict": {"predictions_count": 0}},
        )
        success_result = FakePipelineOutput(
            success=True,
            completed_steps=["build_dataset_usd", "predict"],
            step_results={"predict": {"predictions_count": 3}},
        )
        mock_run.side_effect = [zero_pred, success_result]

        result = run_sub_asset(sa)
        assert result.status == "completed"
        assert mock_run.call_count == 2
        mock_clean.assert_called_once()

    @patch("material_agent.scene.run._clean_working_dir_for_so_retry")
    @patch("material_agent.api.pipeline.run_pipeline")
    def test_no_retry_when_non_so_step_fails(
        self,
        mock_run: MagicMock,
        mock_clean: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Pipeline failed but SO was NOT in completed_steps -> no retry."""
        cfg_path = tmp_path / "config.yaml"
        _write_config(cfg_path)
        sa = _make_sub_asset(config_path=str(cfg_path))

        mock_run.return_value = FakePipelineOutput(
            success=False,
            error="build_dataset_usd failed",
            completed_steps=["build_dataset_usd"],
        )

        result = run_sub_asset(sa)
        assert result.status == "failed"
        assert mock_run.call_count == 1
        mock_clean.assert_not_called()

    @patch("material_agent.scene.run._update_output_paths")
    @patch("material_agent.api.pipeline.run_pipeline")
    def test_no_retry_when_so_succeeded_with_predictions(
        self,
        mock_run: MagicMock,
        mock_update: MagicMock,
        tmp_path: Path,
    ) -> None:
        """SO ran, pipeline succeeded, predictions > 0 -> no retry needed."""
        cfg_path = tmp_path / "config.yaml"
        _write_config(cfg_path)
        sa = _make_sub_asset(config_path=str(cfg_path))

        mock_run.return_value = FakePipelineOutput(
            success=True,
            completed_steps=["optimize_usd", "build_dataset_usd", "predict"],
            step_results={"predict": {"predictions_count": 5}},
        )

        result = run_sub_asset(sa)
        assert result.status == "completed"
        assert mock_run.call_count == 1

    @patch("material_agent.scene.run._update_output_paths")
    @patch("material_agent.scene.run._patch_config_predict_max_workers")
    @patch("material_agent.api.pipeline.run_pipeline")
    def test_predict_max_workers_patches_config(
        self,
        mock_run: MagicMock,
        mock_patch: MagicMock,
        mock_update: MagicMock,
        tmp_path: Path,
    ) -> None:
        cfg_path = tmp_path / "config.yaml"
        _write_config(cfg_path)
        sa = _make_sub_asset(config_path=str(cfg_path))

        mock_run.return_value = FakePipelineOutput(
            success=True, completed_steps=["predict"]
        )

        run_sub_asset(sa, predict_max_workers=4)
        mock_patch.assert_called_once_with(cfg_path, 4)

    @patch("material_agent.api.pipeline.run_pipeline")
    def test_failed_pipeline_sets_status_failed(
        self,
        mock_run: MagicMock,
        tmp_path: Path,
    ) -> None:
        cfg_path = tmp_path / "config.yaml"
        _write_config(cfg_path)
        sa = _make_sub_asset(config_path=str(cfg_path))

        mock_run.return_value = FakePipelineOutput(
            success=False, error="boom", completed_steps=[]
        )

        result = run_sub_asset(sa)
        assert result.status == "failed"


# ---------------------------------------------------------------------------
# _run_parallel
# ---------------------------------------------------------------------------


class TestRunParallel:
    """Tests for _run_parallel: thread-safe saves, error handling, sorting."""

    @patch("material_agent.scene.run._run_sub_asset_worker")
    def test_manifest_saved_per_completion(
        self, mock_worker: MagicMock, tmp_path: Path
    ) -> None:
        """Manifest is saved once per completed future."""
        sa1 = _make_sub_asset(name="a1", asset_id="id1", mesh_count=20)
        sa2 = _make_sub_asset(name="a2", asset_id="id2", mesh_count=10)

        manifest = SceneManifest(sub_assets=[sa1, sa2])
        manifest_path = tmp_path / "manifest.json"

        # Workers return updated copies
        def worker_side_effect(sa, *args, **kwargs):
            sa.status = "completed"
            return sa

        mock_worker.side_effect = worker_side_effect

        manifest.save = MagicMock()  # type: ignore[method-assign]

        completed, failed = _run_parallel(
            [sa1, sa2],
            manifest,
            manifest_path,
            skip_steps=None,
            only_steps=None,
            verbose=False,
            max_workers=2,
        )

        assert completed == 2
        assert failed == 0
        assert manifest.save.call_count == 2

    @patch("material_agent.scene.run._run_sub_asset_worker")
    def test_worker_exception_counts_as_failed(
        self, mock_worker: MagicMock, tmp_path: Path
    ) -> None:
        """If the worker future raises, the asset is marked failed."""
        sa = _make_sub_asset(name="boom", asset_id="id_boom", mesh_count=5)
        manifest = SceneManifest(sub_assets=[sa])
        manifest_path = tmp_path / "manifest.json"
        manifest.save = MagicMock()  # type: ignore[method-assign]

        mock_worker.side_effect = RuntimeError("worker crashed")

        completed, failed = _run_parallel(
            [sa],
            manifest,
            manifest_path,
            skip_steps=None,
            only_steps=None,
            verbose=False,
            max_workers=1,
        )

        assert completed == 0
        assert failed == 1
        assert manifest.sub_assets[0].status == "failed"

    @patch("material_agent.scene.run._run_sub_asset_worker")
    def test_largest_first_sorting(
        self, mock_worker: MagicMock, tmp_path: Path
    ) -> None:
        """Assets should be submitted largest mesh_count first."""
        sa_small = _make_sub_asset(name="small", asset_id="id_s", mesh_count=5)
        sa_large = _make_sub_asset(name="large", asset_id="id_l", mesh_count=100)

        manifest = SceneManifest(sub_assets=[sa_small, sa_large])
        manifest_path = tmp_path / "manifest.json"
        manifest.save = MagicMock()  # type: ignore[method-assign]

        call_order: list[str] = []

        def worker_side_effect(sa, *args, **kwargs):
            call_order.append(sa.name)
            sa.status = "completed"
            return sa

        mock_worker.side_effect = worker_side_effect

        # Use max_workers=1 to force sequential submission order
        _run_parallel(
            [sa_small, sa_large],
            manifest,
            manifest_path,
            skip_steps=None,
            only_steps=None,
            verbose=False,
            max_workers=1,
        )

        assert call_order[0] == "large"
        assert call_order[1] == "small"


# ---------------------------------------------------------------------------
# run_all
# ---------------------------------------------------------------------------


class TestRunAll:
    """Tests for run_all: skip completed, sequential vs parallel routing."""

    @patch("material_agent.scene.run._run_sequential")
    def test_skip_completed_assets(self, mock_seq: MagicMock, tmp_path: Path) -> None:
        sa_done = _make_sub_asset(
            name="done", config_path="/tmp/c.yaml", status="completed"
        )
        sa_pending = _make_sub_asset(
            name="pending", config_path="/tmp/c2.yaml", status="pending"
        )

        manifest = SceneManifest(sub_assets=[sa_done, sa_pending])
        manifest_path = tmp_path / "manifest.json"

        mock_seq.return_value = (1, 0)

        run_all(manifest, manifest_path, skip_existing=True)

        # Only the pending asset should be passed to _run_sequential
        args = mock_seq.call_args
        to_process = args[0][0]
        assert len(to_process) == 1
        assert to_process[0].name == "pending"

    @patch("material_agent.scene.run._run_parallel")
    def test_parallel_routing_when_workers_gt_1(
        self, mock_par: MagicMock, tmp_path: Path
    ) -> None:
        sa = _make_sub_asset(config_path="/tmp/c.yaml")
        manifest = SceneManifest(sub_assets=[sa])
        manifest_path = tmp_path / "manifest.json"

        mock_par.return_value = (1, 0)

        run_all(manifest, manifest_path, max_workers=4)

        mock_par.assert_called_once()

    @patch("material_agent.scene.run._run_sequential")
    def test_sequential_routing_when_workers_eq_1(
        self, mock_seq: MagicMock, tmp_path: Path
    ) -> None:
        sa = _make_sub_asset(config_path="/tmp/c.yaml")
        manifest = SceneManifest(sub_assets=[sa])
        manifest_path = tmp_path / "manifest.json"

        mock_seq.return_value = (1, 0)

        run_all(manifest, manifest_path, max_workers=1)

        mock_seq.assert_called_once()

    def test_returns_immediately_when_nothing_to_process(self, tmp_path: Path) -> None:
        manifest = SceneManifest(sub_assets=[])
        manifest_path = tmp_path / "manifest.json"

        result = run_all(manifest, manifest_path)
        assert result is manifest

    @patch("material_agent.scene.run._run_sequential")
    def test_skips_assets_without_config_path(
        self, mock_seq: MagicMock, tmp_path: Path
    ) -> None:
        sa_no_cfg = _make_sub_asset(name="no_cfg", config_path=None)
        sa_with_cfg = _make_sub_asset(name="has_cfg", config_path="/tmp/c.yaml")
        manifest = SceneManifest(sub_assets=[sa_no_cfg, sa_with_cfg])
        manifest_path = tmp_path / "manifest.json"

        mock_seq.return_value = (1, 0)

        run_all(manifest, manifest_path)

        to_process = mock_seq.call_args[0][0]
        assert len(to_process) == 1
        assert to_process[0].name == "has_cfg"


# ---------------------------------------------------------------------------
# run_payload
# ---------------------------------------------------------------------------


class TestRunPayload:
    """Tests for run_payload: SO retry for payload groups."""

    def test_no_config_path_raises(self) -> None:
        pg = _make_payload_group(config_path=None)
        with pytest.raises(ValueError, match="no config_path"):
            run_payload(pg)

    def test_missing_config_file_raises(self, tmp_path: Path) -> None:
        pg = _make_payload_group(config_path=str(tmp_path / "nope.yaml"))
        with pytest.raises(FileNotFoundError):
            run_payload(pg)

    @patch("material_agent.scene.run._update_payload_output_paths")
    @patch("material_agent.api.pipeline.run_pipeline")
    def test_success_no_retry(
        self,
        mock_run: MagicMock,
        mock_update: MagicMock,
        tmp_path: Path,
    ) -> None:
        cfg_path = tmp_path / "pg.yaml"
        _write_config(cfg_path)
        pg = _make_payload_group(config_path=str(cfg_path))

        mock_run.return_value = FakePipelineOutput(
            success=True,
            completed_steps=["build_dataset_usd", "predict"],
            step_results={"predict": {"predictions_count": 5}},
        )

        result = run_payload(pg)
        assert result.status == "completed"
        assert mock_run.call_count == 1

    @patch("material_agent.scene.run._update_payload_output_paths")
    @patch("material_agent.scene.run._clean_working_dir_for_so_retry")
    @patch("material_agent.api.pipeline.run_pipeline")
    def test_so_retry_on_failure_after_so(
        self,
        mock_run: MagicMock,
        mock_clean: MagicMock,
        mock_update: MagicMock,
        tmp_path: Path,
    ) -> None:
        cfg_path = tmp_path / "pg.yaml"
        _write_config(cfg_path)
        pg = _make_payload_group(config_path=str(cfg_path))

        fail = FakePipelineOutput(
            success=False,
            error="predict boom",
            completed_steps=["optimize_usd"],
        )
        ok = FakePipelineOutput(
            success=True, completed_steps=["build_dataset_usd", "predict"]
        )
        mock_run.side_effect = [fail, ok]

        result = run_payload(pg)
        assert result.status == "completed"
        assert mock_run.call_count == 2
        mock_clean.assert_called_once()

    @patch("material_agent.scene.run._update_payload_output_paths")
    @patch("material_agent.scene.run._clean_working_dir_for_so_retry")
    @patch("material_agent.api.pipeline.run_pipeline")
    def test_so_retry_on_zero_predictions(
        self,
        mock_run: MagicMock,
        mock_clean: MagicMock,
        mock_update: MagicMock,
        tmp_path: Path,
    ) -> None:
        cfg_path = tmp_path / "pg.yaml"
        _write_config(cfg_path)
        pg = _make_payload_group(config_path=str(cfg_path))

        zero = FakePipelineOutput(
            success=True,
            completed_steps=["optimize_usd", "predict"],
            step_results={"predict": {"predictions_count": 0}},
        )
        ok = FakePipelineOutput(
            success=True,
            completed_steps=["predict"],
            step_results={"predict": {"predictions_count": 2}},
        )
        mock_run.side_effect = [zero, ok]

        result = run_payload(pg)
        assert result.status == "completed"
        assert mock_run.call_count == 2

    @patch("material_agent.api.pipeline.run_pipeline")
    def test_no_retry_when_no_so(
        self,
        mock_run: MagicMock,
        tmp_path: Path,
    ) -> None:
        """No SO in completed_steps + failure -> no retry."""
        cfg_path = tmp_path / "pg.yaml"
        _write_config(cfg_path)
        pg = _make_payload_group(config_path=str(cfg_path))

        mock_run.return_value = FakePipelineOutput(
            success=False,
            error="build failed",
            completed_steps=["build_dataset_usd"],
        )

        result = run_payload(pg)
        assert result.status == "failed"
        assert mock_run.call_count == 1
