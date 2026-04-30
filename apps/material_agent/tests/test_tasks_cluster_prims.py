# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for cluster_prims and expand_cluster_predictions tasks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import yaml
from PIL import Image
from world_understanding.functions.models.image_embedding_models import (
    LocalVisualImageEmbeddingModel,
)

from material_agent.tasks.cluster_prims import (
    DEFAULT_COMPLEXITY_THRESHOLDS,
    ClusterPrimsTask,
    ExpandClusterPredictionsTask,
    _cluster_by_tier,
    _complexity_tier,
    _edge_density,
    _select_representatives,
)
from material_agent.tasks.config_cluster_prims import (
    ClusterPrimsConfigTask,
    ExpandClusterPredictionsConfigTask,
)

# ---------------------------------------------------------------------------
# _edge_density
# ---------------------------------------------------------------------------


class TestEdgeDensity:
    def test_missing_file_returns_zero(self, tmp_path: Path) -> None:
        assert _edge_density(tmp_path / "nonexistent.png") == 0.0

    def test_valid_image_returns_float_in_range(self, tmp_path: Path) -> None:
        from PIL import Image

        # Create a simple gradient image that will have some edges
        img = Image.new("RGB", (64, 64), color="white")
        # Draw a black rectangle to create edges
        pixels = img.load()
        for x in range(20, 44):
            for y in range(20, 44):
                pixels[x, y] = (0, 0, 0)
        path = tmp_path / "test.png"
        img.save(path)

        result = _edge_density(path)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_blank_image_has_low_density(self, tmp_path: Path) -> None:
        from PIL import Image

        img = Image.new("RGB", (64, 64), color="white")
        path = tmp_path / "blank.png"
        img.save(path)

        result = _edge_density(path)
        assert result == 0.0


# ---------------------------------------------------------------------------
# _complexity_tier
# ---------------------------------------------------------------------------


class TestComplexityTier:
    def test_low_tier(self) -> None:
        assert _complexity_tier(0.01, DEFAULT_COMPLEXITY_THRESHOLDS) == "low"

    def test_medium_tier(self) -> None:
        assert _complexity_tier(0.05, DEFAULT_COMPLEXITY_THRESHOLDS) == "medium"

    def test_high_tier(self) -> None:
        assert _complexity_tier(0.10, DEFAULT_COMPLEXITY_THRESHOLDS) == "high"

    def test_boundary_low_medium(self) -> None:
        # 0.02 is >= low.hi, so should be medium
        assert _complexity_tier(0.02, DEFAULT_COMPLEXITY_THRESHOLDS) == "medium"

    def test_boundary_medium_high(self) -> None:
        assert _complexity_tier(0.08, DEFAULT_COMPLEXITY_THRESHOLDS) == "high"

    def test_out_of_range_negative_falls_back_to_high(self) -> None:
        # Negative is not in any [lo, hi) range
        assert _complexity_tier(-1.0, DEFAULT_COMPLEXITY_THRESHOLDS) == "high"

    def test_out_of_range_above_falls_back_to_high(self) -> None:
        # The high tier goes up to 1.0 inclusive (due to max_hi logic in
        # _cluster_by_tier), but _complexity_tier uses strict < for hi.
        # Score of 1.5 is above all tiers.
        assert _complexity_tier(1.5, DEFAULT_COMPLEXITY_THRESHOLDS) == "high"


# ---------------------------------------------------------------------------
# _cluster_by_tier
# ---------------------------------------------------------------------------


class TestClusterByTier:
    def test_identical_vectors_cluster_together(self) -> None:
        # 4 identical vectors should form one cluster
        vec = np.random.default_rng(42).random(128)
        vec /= np.linalg.norm(vec)
        embeddings = np.tile(vec, (4, 1))
        complexities = np.array([0.01, 0.01, 0.01, 0.01])  # all low tier

        labels = _cluster_by_tier(
            embeddings, complexities, DEFAULT_COMPLEXITY_THRESHOLDS
        )
        assert len(np.unique(labels)) == 1

    def test_different_vectors_separate_clusters(self) -> None:
        rng = np.random.default_rng(42)
        # Two very different unit vectors in low-complexity tier
        v1 = rng.random(128)
        v1 /= np.linalg.norm(v1)
        v2 = -v1  # opposite direction -> cosine distance = 2.0
        embeddings = np.stack([v1, v2])
        complexities = np.array([0.01, 0.01])

        labels = _cluster_by_tier(
            embeddings, complexities, DEFAULT_COMPLEXITY_THRESHOLDS
        )
        assert labels[0] != labels[1]

    def test_single_prim_gets_own_cluster(self) -> None:
        rng = np.random.default_rng(42)
        vec = rng.random(128)
        vec /= np.linalg.norm(vec)
        embeddings = vec.reshape(1, -1)
        complexities = np.array([0.05])  # medium tier

        labels = _cluster_by_tier(
            embeddings, complexities, DEFAULT_COMPLEXITY_THRESHOLDS
        )
        assert labels[0] >= 0

    def test_different_tiers_separate_clusters(self) -> None:
        # Two identical vectors in different complexity tiers should be in
        # different clusters (they are clustered independently per tier).
        rng = np.random.default_rng(42)
        vec = rng.random(128)
        vec /= np.linalg.norm(vec)
        embeddings = np.tile(vec, (2, 1))
        complexities = np.array([0.01, 0.10])  # low vs high tier

        labels = _cluster_by_tier(
            embeddings, complexities, DEFAULT_COMPLEXITY_THRESHOLDS
        )
        assert labels[0] != labels[1]

    def test_all_labels_assigned(self) -> None:
        rng = np.random.default_rng(42)
        embeddings = rng.random((5, 64))
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings /= norms
        complexities = np.array([0.01, 0.03, 0.05, 0.09, 0.5])

        labels = _cluster_by_tier(
            embeddings, complexities, DEFAULT_COMPLEXITY_THRESHOLDS
        )
        assert np.all(labels >= 0)

    def test_gb300_flat_material_palette_stays_separate_with_local_visual(
        self,
    ) -> None:
        model = LocalVisualImageEmbeddingModel()
        colors = [
            (25, 25, 25),  # black plastic / powder coat
            (20, 90, 140),  # blue/teal signal accent
            (160, 160, 150),  # anodized aluminum
            (192, 192, 192),  # nickel/silver metal
            (190, 140, 40),  # gold connector contacts
            (184, 115, 51),  # copper conductors
        ]
        images = [Image.new("RGB", (32, 32), color=color) for color in colors]
        embeddings = np.asarray(model.embed_images(images))
        complexities = np.zeros(len(colors), dtype=np.float32)

        labels = _cluster_by_tier(
            embeddings,
            complexities,
            DEFAULT_COMPLEXITY_THRESHOLDS,
        )

        assert len(np.unique(labels)) == len(colors)


# ---------------------------------------------------------------------------
# _select_representatives
# ---------------------------------------------------------------------------


class TestSelectRepresentatives:
    def test_picks_closest_to_centroid(self) -> None:
        # 3 vectors in one cluster; the centroid-closest should be chosen
        v_center = np.array([1.0, 0.0, 0.0])
        v_near = np.array([0.99, 0.1, 0.0])
        v_near /= np.linalg.norm(v_near)
        v_far = np.array([0.5, 0.5, 0.5])
        v_far /= np.linalg.norm(v_far)

        embeddings = np.stack([v_far, v_center, v_near])
        labels = np.array([0, 0, 0])

        reps = _select_representatives(embeddings, labels)
        assert 0 in reps
        # The centroid of these vectors should be closest to v_center or v_near
        rep_idx = reps[0]
        assert rep_idx in (1, 2)  # v_center or v_near

    def test_singleton_cluster(self) -> None:
        embeddings = np.array([[1.0, 0.0, 0.0]])
        labels = np.array([5])

        reps = _select_representatives(embeddings, labels)
        assert reps[5] == 0

    def test_multiple_clusters(self) -> None:
        embeddings = np.array(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
                [0.1, 0.9],
            ]
        )
        labels = np.array([0, 0, 1, 1])

        reps = _select_representatives(embeddings, labels)
        assert set(reps.keys()) == {0, 1}
        assert reps[0] in (0, 1)
        assert reps[1] in (2, 3)


# ---------------------------------------------------------------------------
# ClusterPrimsTask.run — skip path
# ---------------------------------------------------------------------------


def _make_dataset_jsonl(path: Path, n: int) -> Path:
    """Write a minimal dataset.jsonl with n entries."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for i in range(n):
            f.write(json.dumps({"id": f"prim_{i}", "images": {"prim_only": []}}) + "\n")
    return path


class TestClusterPrimsTaskRun:
    def test_skips_when_below_min_prims(self, tmp_path: Path) -> None:
        dataset_path = _make_dataset_jsonl(tmp_path / "dataset" / "dataset.jsonl", n=5)
        context: dict[str, Any] = {
            "dataset_path": str(dataset_path),
            "working_dir": str(tmp_path / "work"),
            "cluster_prims_config": {"min_prims_to_activate": 10},
        }

        task = ClusterPrimsTask()
        result = task.run(context)

        assert result["cluster_prims_ran"] is False
        assert "cluster_map_path" not in result

    def test_copies_dataset_json_to_clusters_dir(self, tmp_path: Path) -> None:
        """Verify that dataset.json is copied into the clusters/ directory."""
        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()

        # Write dataset.jsonl with enough prims and prim_only images
        n = 3
        entries = []
        for i in range(n):
            img_path = dataset_dir / f"prim_{i}.png"
            # Create a tiny image
            from PIL import Image

            Image.new("RGB", (8, 8), color="red").save(img_path)
            entries.append(
                {
                    "id": f"prim_{i}",
                    "images": {"prim_only": [str(img_path)]},
                }
            )

        dataset_jsonl = dataset_dir / "dataset.jsonl"
        with open(dataset_jsonl, "w") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")

        # Write a dataset.json config file
        dataset_json = dataset_dir / "dataset.json"
        dataset_json.write_text(json.dumps({"system_prompt": "test"}))

        working_dir = tmp_path / "work"
        context: dict[str, Any] = {
            "dataset_path": str(dataset_jsonl),
            "working_dir": str(working_dir),
            "cluster_prims_config": {
                "min_prims_to_activate": 1,
                "report": False,  # skip HTML report generation
            },
        }

        # Mock the embedding model
        mock_model = MagicMock()
        mock_model.embedding_dimension = 8
        mock_model.embed_images = MagicMock(
            side_effect=lambda imgs: [np.random.default_rng(42).random(8) for _ in imgs]
        )

        with patch(
            "world_understanding.functions.models.image_embedding_models.create_image_embedding_model",
            return_value=mock_model,
        ):
            task = ClusterPrimsTask()
            result = task.run(context)

        assert result["cluster_prims_ran"] is True
        # Check dataset.json was copied
        copied = working_dir / "clusters" / "dataset.json"
        assert copied.exists()
        assert json.loads(copied.read_text()) == {"system_prompt": "test"}

    def test_retries_transient_embedding_batch_failure(self, tmp_path: Path) -> None:
        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()

        from PIL import Image

        entries = []
        for i in range(2):
            img_path = dataset_dir / f"prim_{i}.png"
            Image.new("RGB", (8, 8), color="red").save(img_path)
            entries.append(
                {"id": f"prim_{i}", "images": {"prim_only": [str(img_path)]}}
            )

        dataset_jsonl = dataset_dir / "dataset.jsonl"
        with open(dataset_jsonl, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")

        mock_model = MagicMock()
        mock_model.embedding_dimension = 8
        mock_model.embed_images = MagicMock(
            side_effect=[
                RuntimeError("temporary embedding outage"),
                [np.ones(8), np.ones(8)],
            ]
        )

        context: dict[str, Any] = {
            "dataset_path": str(dataset_jsonl),
            "working_dir": str(tmp_path / "work"),
            "cluster_prims_config": {
                "min_prims_to_activate": 1,
                "batch_size": 2,
                "max_workers": 1,
                "embedding_retries": 2,
                "embedding_retry_initial_delay": 0,
                "report": False,
            },
        }

        with (
            patch(
                "world_understanding.functions.models.image_embedding_models.create_image_embedding_model",
                return_value=mock_model,
            ),
            patch("material_agent.tasks.cluster_prims.time.sleep"),
        ):
            result = ClusterPrimsTask().run(context)

        assert result["cluster_prims_ran"] is True
        assert mock_model.embed_images.call_count == 2


# ---------------------------------------------------------------------------
# ExpandClusterPredictionsTask.run
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


class TestExpandClusterPredictionsTaskRun:
    def test_skips_when_cluster_prims_not_ran(self) -> None:
        context: dict[str, Any] = {"cluster_prims_ran": False}
        task = ExpandClusterPredictionsTask()
        result = task.run(context)
        assert result is context

    def test_expands_predictions(self, tmp_path: Path) -> None:
        predictions_path = tmp_path / "predictions" / "predictions.jsonl"
        cluster_map_path = tmp_path / "clusters" / "cluster_map.jsonl"

        # Rep prim_0 predicted "metal"; prim_1 is a member of the same cluster
        _write_jsonl(
            predictions_path,
            [
                {"id": "prim_0", "material": "metal", "confidence": 0.9},
            ],
        )
        _write_jsonl(
            cluster_map_path,
            [
                {
                    "id": "prim_0",
                    "cluster_id": 0,
                    "is_representative": True,
                    "cluster_representative_id": "prim_0",
                    "cluster_size": 2,
                    "complexity_score": 0.01,
                    "complexity_tier": "low",
                },
                {
                    "id": "prim_1",
                    "cluster_id": 0,
                    "is_representative": False,
                    "cluster_representative_id": "prim_0",
                    "cluster_size": 2,
                    "complexity_score": 0.01,
                    "complexity_tier": "low",
                },
            ],
        )

        context: dict[str, Any] = {
            "cluster_prims_ran": True,
            "predictions_path": str(predictions_path),
            "cluster_map_path": str(cluster_map_path),
        }

        task = ExpandClusterPredictionsTask()
        result = task.run(context)

        preds = _read_jsonl(Path(result["predictions_path"]))
        assert len(preds) == 2

        # Representative keeps its own prediction
        rep = next(p for p in preds if p["id"] == "prim_0")
        assert rep["material"] == "metal"
        assert "prediction_source" not in rep

        # Member gets propagated prediction
        member = next(p for p in preds if p["id"] == "prim_1")
        assert member["material"] == "metal"
        assert member["prediction_source"] == "cluster_representative"
        assert member["cluster_representative_id"] == "prim_0"
        assert member["cluster_id"] == 0


# ---------------------------------------------------------------------------
# Config tasks
# ---------------------------------------------------------------------------


class TestClusterPrimsConfigTask:
    def test_loads_config(self, tmp_path: Path) -> None:
        config = {
            "dataset_path": "/some/dataset.jsonl",
            "working_dir": "/some/workdir",
            "batch_size": 100,
        }
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.dump(config))

        context: dict[str, Any] = {"config_path": str(config_path)}
        task = ClusterPrimsConfigTask()
        result = task.run(context)

        assert result["dataset_path"] == "/some/dataset.jsonl"
        assert result["working_dir"] == "/some/workdir"
        assert result["cluster_prims_config"]["batch_size"] == 100

    def test_raises_on_missing_dataset_path(self, tmp_path: Path) -> None:
        config = {"working_dir": "/some/workdir"}
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.dump(config))

        context: dict[str, Any] = {"config_path": str(config_path)}
        task = ClusterPrimsConfigTask()
        with pytest.raises(ValueError, match="dataset_path is required"):
            task.run(context)

    def test_raises_on_missing_working_dir(self, tmp_path: Path) -> None:
        config = {"dataset_path": "/some/dataset.jsonl"}
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.dump(config))

        context: dict[str, Any] = {"config_path": str(config_path)}
        task = ClusterPrimsConfigTask()
        with pytest.raises(ValueError, match="working_dir is required"):
            task.run(context)

    def test_raises_on_empty_config(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.yaml"
        config_path.write_text("")

        context: dict[str, Any] = {"config_path": str(config_path)}
        task = ClusterPrimsConfigTask()
        with pytest.raises(ValueError, match="Empty config"):
            task.run(context)

    def test_raises_on_missing_config_file(self, tmp_path: Path) -> None:
        context: dict[str, Any] = {"config_path": str(tmp_path / "nope.yaml")}
        task = ClusterPrimsConfigTask()
        with pytest.raises(FileNotFoundError):
            task.run(context)


class TestExpandClusterPredictionsConfigTask:
    def test_loads_config_when_cluster_ran(self, tmp_path: Path) -> None:
        config = {
            "cluster_prims_ran": True,
            "predictions_path": "/pred.jsonl",
            "cluster_map_path": "/map.jsonl",
        }
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.dump(config))

        context: dict[str, Any] = {"config_path": str(config_path)}
        task = ExpandClusterPredictionsConfigTask()
        result = task.run(context)

        assert result["predictions_path"] == "/pred.jsonl"
        assert result["cluster_map_path"] == "/map.jsonl"

    def test_skips_when_cluster_not_ran(self, tmp_path: Path) -> None:
        config = {"cluster_prims_ran": False}
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.dump(config))

        context: dict[str, Any] = {"config_path": str(config_path)}
        task = ExpandClusterPredictionsConfigTask()
        result = task.run(context)

        assert result["cluster_prims_ran"] is False
        assert "predictions_path" not in result

    def test_raises_on_missing_predictions_path(self, tmp_path: Path) -> None:
        config = {
            "cluster_prims_ran": True,
            "cluster_map_path": "/map.jsonl",
        }
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.dump(config))

        context: dict[str, Any] = {"config_path": str(config_path)}
        task = ExpandClusterPredictionsConfigTask()
        with pytest.raises(ValueError, match="predictions_path is required"):
            task.run(context)

    def test_raises_on_missing_cluster_map_path(self, tmp_path: Path) -> None:
        config = {
            "cluster_prims_ran": True,
            "predictions_path": "/pred.jsonl",
        }
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.dump(config))

        context: dict[str, Any] = {"config_path": str(config_path)}
        task = ExpandClusterPredictionsConfigTask()
        with pytest.raises(ValueError, match="cluster_map_path is required"):
            task.run(context)
