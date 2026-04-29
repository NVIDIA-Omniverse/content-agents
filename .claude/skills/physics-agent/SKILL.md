---
name: physics-agent
description: Run the Physics Agent CLI for VLM-based physics property classification of 3D assets. Use when user wants to classify asset components, run the physics agent pipeline, identify asset types, build a dataset from USD files, or predict component properties (material, type, physics). Trigger phrases include "classify asset", "run physics agent", "physics agent pipeline", "identify asset", "build dataset from USD", "predict component properties", "physics classification".
---

# Physics Agent CLI

The Physics Agent is a VLM-based system for classifying the components of 3D assets. It renders per-component views, identifies the overall asset, then predicts the material, component type, and physical properties of each part.

## Prerequisites

- The `physics-agent` CLI must be installed and available on PATH
- Environment variables from `.env` must be set:
  - **VLM**: `NVIDIA_API_KEY` (nim backend, default — uses https://integrate.api.nvidia.com/v1). Alternatives: `OPENAI_API_KEY` (OpenAI), `ANTHROPIC_API_KEY` (Anthropic), `GOOGLE_API_KEY` (Google Gemini)
  - **Remote rendering/optimization** (optional): `RENDER_ENDPOINT` / `OPTIMIZER_ENDPOINT`, plus any auth required by that service. Skip if using local rendering.
  - **AWS** (optional, only for S3 asset upload): `WU_S3_BUCKET`, `WU_S3_PROFILE`, `WU_S3_REGION`, plus standard AWS credentials.
- A unified YAML configuration file (see Config Template below)

## Primary Command: `run`

The `run` command executes the full multi-step pipeline. This is the recommended entry point.

```bash
physics-agent run <config.yaml> [OPTIONS]
```

### Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--skip` | | None | Comma-separated steps to skip |
| `--only` | | None | Comma-separated steps to run exclusively |
| `--session-id` | | None | Reuse existing session ID |
| `--resume` | | False | Resume from last successful checkpoint |
| `--dry-run` | | False | Show pipeline plan without executing |
| `--clean` | | False | Delete working directory before starting |
| `--verbose` | `-v` | False | Enable DEBUG logging |
| `--log-file` | | None | Path to log file |
| `--log-level` | | INFO | Logging level |

### Pipeline Steps (execution order)

1. **optimize_usd** — Flatten/deinstance USD via the configured scene optimizer
2. **identify_asset** — Render lightweight whole-scene preview images and identify the overall asset (e.g., "forklift") via VLM
3. **build_dataset_usd** — Render per-prim views for VLM input
4. **build_dataset_prepare_dataset** — Prepare dataset entries with classification specs
5. **predict** — VLM inference for per-component classification (type, material, physics)
6. **restore_usd** — Remap predictions from the optimized prim paths back onto the original input USD's prim paths. **Required whenever both `optimize_usd` and `apply_physics` are enabled** (see warning below).
7. **apply_physics** — Apply `UsdPhysics` schemas (`RigidBodyAPI`, `CollisionAPI`, `MeshCollisionAPI`, `MassAPI`, plus a cached `MaterialAPI` bound with the `physics` purpose) to each predicted prim and write a simulation-ready USD to `physics/<stem>_physics.usda`. Always targets the **original** input USD (not the optimized one).

> ⚠️ **`optimize_usd` + `apply_physics` requires `restore_usd`.** `apply_physics` writes onto the original USD, but raw `predict` output is keyed by the *optimized* prim paths. Without `restore_usd` those paths won't resolve — `apply_physics` would trigger `Prim not found` on every entry and produce a USD with no physics schemas. The pipeline enforces this: running `optimize_usd` → `predict` → `apply_physics` with `restore_usd` disabled raises `ValueError: apply_physics cannot run after optimize_usd without restore_usd`. Either enable `restore_usd`, or disable `optimize_usd`.

## Other Commands

| Command | Description |
|---------|-------------|
| `physics-agent predict <config.yaml>` | Run VLM prediction only (alias for `run --only predict`) |
| `physics-agent build-dataset usd <config.yaml>` | Build dataset from USD renders |
| `physics-agent build-dataset prepare-dataset <config.yaml>` | Prepare dataset with classification specs |

## Common Workflows

### Run full pipeline on a USD file

```bash
physics-agent run configs/lightbulb.yaml
```

### Preview what the pipeline will do

```bash
physics-agent run configs/lightbulb.yaml --dry-run
```

### Run only specific steps

```bash
physics-agent run configs/lightbulb.yaml --only predict
```

### Skip a step

```bash
physics-agent run configs/lightbulb.yaml --skip optimize_usd
```

### Resume after a failure

```bash
physics-agent run configs/lightbulb.yaml --resume
```

## Creating a Config

When asked to run the physics agent on a USD file, write a config YAML based on the template below. Adapt the marked fields and save it next to the input USD file.

**Fields to adapt:**
- `project.name` and `project.session_id` — descriptive name for the run
- `input.usd_path` — path to the user's input USD file
- `steps.predict.vlm.model` — VLM model to use

All paths are resolved relative to the config file's location. Working directory is auto-derived as `.{session_id}/` next to the config file.

### Config Template

```yaml
project:
  name: "CHANGE_ME"
  session_id: "CHANGE_ME"
  description: "Physics agent classification pipeline"

input:
  usd_path: "CHANGE_ME"

steps:
  optimize_usd:
    enabled: true

  identify_asset:
    renderer:
      backend: remote
      image_width: 512
      image_height: 512
      cameras: ["+x+y+z", "-x-y-z"]
    vlm:
      backend: nim
      model: qwen/qwen3.5-397b-a17b

  build_dataset_usd:
    renderer:
      backend: remote
      image_width: 512
      image_height: 512
      cull_style: back
      should_highlight_prim: true
      should_assign_random_colors: true
      highlight_color: [0.7, 0.0, 0.0]
      other_color_range: [0.1, 0.2]
      rendering_modes:
        prim_only:
          margin: 1.2
          cameras: ["+x+y+z", "-x-y-z"]
          camera_focus_mode: prim
        composition:
          margin: 6.0
          cameras: ["+x", "+y", "+z"]
          camera_focus_mode: stage
          skip_occluded_images: false
    prim_filters:
      types:
        - "UsdGeom.Mesh"
        - "UsdGeom.Cube"
        - "UsdGeom.Cylinder"
        - "UsdGeom.Capsule"
        - "UsdGeom.Sphere"
        - "UsdGeom.Cone"
      skip_instances: true
    extract_material_bindings: true
    extract_hierarchy: true
    extract_metadata: true
    build_usd_model: true
    export_usd_model: true
    skip_existing: true
    batch_size: 16

  build_dataset_prepare_dataset:
    enabled: true
    include_ground_truth: false
    include_prim_path_context: true

  predict:
    vlm:
      backend: nim
      model: qwen/qwen3.5-397b-a17b
      temperature: 1.0
      max_completion_tokens: 24576
    max_workers: 16

advanced:
  keep_temp_files: true
  log_level: INFO
```

## Output

The pipeline writes results to a working directory (`.{session_id}/` next to the config file by default):

- `predictions/predictions.jsonl` — one entry per prim: `id` (prim path), `classification` (material, component_type, physical_properties), and reasoning
- `restored_predictions.jsonl` — predictions remapped back onto the original USD's prim paths (only when `restore_usd` ran; lives directly in the working dir)
- `physics/<stem>_physics.usda` — input USD with `UsdPhysics` schemas baked in, ready for simulation (only when `apply_physics` ran)

## Key Differences from Material Agent

- **No material library**: Classifies materials from VLM knowledge, not a reference library
- **No final render step**: Stops after `apply_physics` — no PBR/visual material application or final beauty render
- **identify_asset step**: Extra step that identifies the overall asset before per-component classification
- **Component classification**: Predicts type + material + physics per component, not just material assignment

## Common Issues

### "API key required" error
Cause: Missing environment variable for the selected VLM backend.
Solution: Set the required keys in `.env`:
- `NVIDIA_API_KEY` (nim backend, default — uses https://integrate.api.nvidia.com/v1). Alternatives: `OPENAI_API_KEY` (OpenAI), `ANTHROPIC_API_KEY` (Anthropic), `GOOGLE_API_KEY` (Google Gemini)
- Remote rendering/optimization credentials, if using remote services

### Pipeline fails midway
Solution: Re-run with `--resume` to continue from the last checkpoint.

### Config file path resolution
All relative paths in configs are resolved relative to the config file's directory, not the current working directory.
