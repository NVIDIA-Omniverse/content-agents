# Physics Agent

VLM-based classification agent for 3D assets that identifies component types, surface materials, and physical properties for physics simulation.

## Overview

Physics Agent processes USD files to classify components using Vision-Language Models. Given rendered views of an asset, it predicts:

- **Component type** (e.g., wheel, chassis, sensor housing)
- **Surface material** (e.g., rubber, steel, plastic)
- **Physical properties** (e.g., mass estimate, friction class, rigidity)

Results are structured for downstream physics simulation pipelines.

## Prefer the REST service?

This README covers the `physics-agent` CLI (Option B in the root [README](../../README.md#two-ways-to-use-this)). If you'd rather drive the same pipeline over HTTP with session management and progress streaming, see [`../physics_agent_service/`](../physics_agent_service/) — it brings up with a single `docker compose up`.

## Installation

From the repository root:

```bash
uv pip install -e .
uv pip install -e apps/physics_agent
```

## Rendering

The pipeline renders multi-view images of each prim for VLM analysis. Two options:

- **Local OVRTX subprocess** (default in `lightbulb.yaml`) — `physics-agent` launches an OVRTX process locally to render. Requires an NVIDIA GPU + driver on the machine running the CLI; no separate rendering service needed.
- **Remote rendering endpoint** — change `render.backend` to `remote` in the config and point `RENDER_ENDPOINT=http://localhost:8001` at a running OVRTX rendering API (e.g. the sidecar bundled with `physics_agent_service`, or a standalone deployment). Alternatively, set `NVCF_RENDER_FUNCTION_ID` + `NGC_API_KEY` to call an NVCF-hosted function.

Without one of the two, rendering steps fail.

## Quick Start

```bash
source .venv/bin/activate
physics-agent run apps/physics_agent/configs/lightbulb.yaml
```

## Where outputs land

Every run writes into a single **working directory** placed next to the config file. By default the directory is `.{session_id}` (a hidden folder), where `session_id` comes from `project.session_id` in the config. To override the directory name, set `project.working_dir` to a simple child path of the config directory — e.g. `working_dir: ".my_run"` — without `..` segments.

> ⚠️ The working directory must be a dedicated, pipeline-owned directory. `physics-agent run --clean` recursively deletes the resolved `working_dir` before the run; a `..` segment or a path that escapes the config directory can wipe unrelated files. Never point `working_dir` at a directory that holds files you care about.

For the bundled `lightbulb.yaml` (`session_id: lightbulb`), the working directory is `apps/physics_agent/configs/.lightbulb/`, with this layout:

```text
apps/physics_agent/configs/.lightbulb/
├── identification/                              # whole-asset preview + identification
├── dataset/
│   └── usd/                                     # per-prim renders
├── predictions/
│   ├── predictions.jsonl                        # VLM classifications (component, material, physics)
│   └── report.html                              # HTML report
└── physics/
    └── light_bulb_01_physics.usda               # simulation-ready USD (apply_physics output)
```

The simulation-ready USD is at `<working_dir>/physics/<input-stem>_physics.usda`, where `<input-stem>` is the input USD filename without its extension (so `light_bulb_01.usdz` produces `light_bulb_01_physics.usda`). It is the input USD with `UsdPhysics.RigidBodyAPI` / `CollisionAPI` / `MassAPI` / `MaterialAPI` schemas applied to each predicted prim.

## CLI Reference

```bash
# Run complete pipeline
physics-agent run CONFIG

# Pipeline options
physics-agent run CONFIG --skip build_dataset_usd    # skip a step
physics-agent run CONFIG --only predict              # run specific steps
physics-agent run CONFIG --resume                    # resume from checkpoint
physics-agent run CONFIG --dry-run                   # show execution plan
physics-agent run CONFIG --clean                     # wipe working dir first
physics-agent run CONFIG -v                          # verbose logging

# Individual commands
physics-agent predict CONFIG                         # VLM prediction only
physics-agent build-dataset usd CONFIG               # Build dataset from USD
physics-agent build-dataset prepare-dataset CONFIG   # Prepare dataset for VLM
```

## Configuration

Pipeline configs are YAML files under `configs/`. Use `lightbulb.yaml` as a reference:

```yaml
project:
  name: "my_asset"

input:
  usd_path: "path/to/asset.usd"

predict:
  vlm:
    backend: nim                    # or: openai, anthropic, gemini
    model: qwen/qwen3.5-397b-a17b
```

Paths in config files are relative to the config file's directory.

## Documentation

- **[API Reference](docs/api.md)** -- Python API reference
