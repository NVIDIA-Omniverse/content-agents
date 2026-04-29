# Physics Agent Python API

This module provides programmatic access to Physics Agent functionality. All commands available in the CLI are also available as Python functions.

## Quick Start

The API offers **two usage patterns** - choose based on your needs:

### Pattern 1: Convenience Functions (Simplest)

```python
from physics_agent.api import pipeline
from pathlib import Path

# Minimal usage - just pass config
result = pipeline(Path("config.yaml"))

# With optional overrides
result = pipeline(Path("config.yaml"), verbose=True, resume=True)

if result.success:
    print(f"Completed steps: {result.completed_steps}")
```

### Pattern 2: Full Input Classes (Maximum Control)

```python
from physics_agent.api import run_pipeline, PipelineInput
from pathlib import Path

params = PipelineInput(
    config=Path("config.yaml"),
    only_steps=["build_dataset_usd"],
    verbose=True,
)

result = run_pipeline(params)
if result.success:
    print(f"Completed: {result.completed_steps}")
else:
    print(f"Error: {result.error}")
```

**When to use each pattern:**
- **Convenience functions**: Quick scripts, notebooks, simple use cases
- **Input classes**: Web services, complex logic, when you need type safety

## Available APIs

### Pipeline (end-to-end)

Runs the full classification pipeline: USD dataset build → dataset preparation → VLM prediction → (optional) prediction restore onto original USD → (optional) `UsdPhysics` application producing a simulation-ready USD.

```python
from physics_agent.api import pipeline, run_pipeline, PipelineInput
```

| Function | Description |
|----------|-------------|
| `pipeline(config, **overrides)` | Convenience function |
| `run_pipeline(PipelineInput)` | Input-class variant |
| `apipeline(config, **overrides)` | Async convenience |
| `arun_pipeline(PipelineInput)` | Async input-class variant |

**`PipelineInput` fields:**

| Field | Type | Description |
|-------|------|-------------|
| `config` | `Path \| dict` | Path to YAML config or a config dict |
| `skip_steps` | `list[str]` | Step names to skip |
| `only_steps` | `list[str]` | Run only these steps |
| `session_id` | `str \| None` | Reuse existing session directory |
| `resume` | `bool` | Resume from last checkpoint |
| `dry_run` | `bool` | Show execution plan without running |
| `clean` | `bool` | Clean working directory first |
| `verbose` | `bool` | Enable verbose output |
| `event_listener` | `EventListener \| None` | Progress reporting callback |

### Predict (VLM inference only)

Runs VLM inference on a prepared dataset.

```python
from physics_agent.api import run_predict, PredictInput
```

| Function | Description |
|----------|-------------|
| `run_predict(PredictInput)` | Run prediction synchronously |
| `arun_predict(PredictInput)` | Run prediction asynchronously |

### Build Dataset

Build a dataset from USD files and prepare it for VLM inference.

```python
from physics_agent.api import (
    build_dataset_usd, BuildDatasetUsdInput,
    build_dataset_prepare_dataset, BuildDatasetPrepareDatasetInput,
)
```

| Function | Description |
|----------|-------------|
| `build_dataset_usd(BuildDatasetUsdInput)` | Render prim views from USD |
| `build_dataset_prepare_dataset(BuildDatasetPrepareDatasetInput)` | Add prompts/context to dataset |
| `abuild_dataset_usd(...)` | Async variant |
| `abuild_dataset_prepare_dataset(...)` | Async variant |

## Config Requirements

**API parameters have defaults, but config contents don't!**

While you only need to pass `config` to the API functions, the config itself has required fields:

```python
from physics_agent.api import predict

config = {
    "predict": {
        "vlm": {
            "backend": "nim",                    # REQUIRED
            "model": "qwen/qwen3.5-397b-a17b",    # REQUIRED
        },
    },
    "input": {
        "usd_path": "path/to/asset.usd",         # REQUIRED
    },
}
```

See the CLI README and example configs under `apps/physics_agent/configs/` for full config shape.

## Defaults

The API exposes default values you can reuse:

```python
from physics_agent.api import (
    DEFAULT_VLM_BACKEND,
    DEFAULT_VLM_MODEL,
    DEFAULT_CAMERA_DIRECTIONS,
    PIPELINE_STEP_NAMES,
    PREDICT_DEFAULTS,
    apply_defaults,
    build_default_pipeline_config,
)
```

For prediction, no separate `llm` backend is injected by default. If
`predict.llm` is omitted, the runtime falls back to `llm = vlm` for any
response-parsing logic. Only set `predict.llm` when you explicitly want a
dedicated parser model.

## Event Listeners

All API functions accept an optional `event_listener` for progress reporting. The shared event system from `world_understanding.agentic.events` provides:

```python
from physics_agent.api import (
    EventListener,
    CLIEventListener,          # Rich console output
    CollectingEventListener,   # Collects events for later inspection
    NoOpEventListener,         # Silent
    LoggerAsListener,          # Forwards to Python logging
    create_default_listener,
    get_listener,
)

# Use Rich console for progress
result = pipeline(Path("config.yaml"), event_listener=CLIEventListener())

# Collect events for testing
collector = CollectingEventListener()
result = pipeline(Path("config.yaml"), event_listener=collector)
print(collector.events)
```

## Result Types

All API functions return an `APIResult` (or a pipeline/predict-specific subclass):

```python
from physics_agent.api import APIResult

result = pipeline(Path("config.yaml"))

result.success          # bool
result.error            # str | None
result.output_dir       # Path | None
result.completed_steps  # list[str]  (pipeline only)
```
