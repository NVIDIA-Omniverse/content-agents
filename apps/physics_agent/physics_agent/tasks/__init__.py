# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Physics Agent Tasks."""

from physics_agent.tasks.apply_physics import ApplyPhysicsTask
from physics_agent.tasks.config_apply_physics import ApplyPhysicsConfigTask
from physics_agent.tasks.config_identify_asset import IdentifyAssetConfigTask
from physics_agent.tasks.config_optimize_usd import OptimizeUSDConfigTask
from physics_agent.tasks.config_predict import PredictConfigTask
from physics_agent.tasks.config_prepare_dataset import PrepareDatasetConfigTask
from physics_agent.tasks.config_restore_usd import RestoreUSDConfigTask
from physics_agent.tasks.config_usd_dataset import USDDatasetConfigTask
from physics_agent.tasks.dataset_loading import DatasetLoadingTask
from physics_agent.tasks.identify_asset import IdentifyAssetTask
from physics_agent.tasks.inference import VLMInferenceTask
from physics_agent.tasks.optimize_usd import OptimizeUSDTask
from physics_agent.tasks.predictions import SavePredictionsTask
from physics_agent.tasks.prepare_dataset import PrepareDatasetTask
from physics_agent.tasks.reporting import GeneratePredictionReportTask
from physics_agent.tasks.restore_usd import RestoreUSDTask
from physics_agent.tasks.unified_pipeline_executor import UnifiedPipelineExecutorTask

__all__ = [
    "ApplyPhysicsConfigTask",
    "ApplyPhysicsTask",
    "IdentifyAssetConfigTask",
    "IdentifyAssetTask",
    "OptimizeUSDConfigTask",
    "OptimizeUSDTask",
    "PredictConfigTask",
    "PrepareDatasetConfigTask",
    "USDDatasetConfigTask",
    "DatasetLoadingTask",
    "VLMInferenceTask",
    "SavePredictionsTask",
    "PrepareDatasetTask",
    "GeneratePredictionReportTask",
    "RestoreUSDConfigTask",
    "RestoreUSDTask",
    "UnifiedPipelineExecutorTask",
]
