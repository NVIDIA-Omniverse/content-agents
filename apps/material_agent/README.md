# Material Agent

A Vision-Language Model (VLM) based system for intelligent material assignment to 3D-rendered object parts. The Material Agent analyzes visual characteristics of object components and assigns appropriate materials from a material library, enabling automated material selection for 3D modeling and rendering workflows.

## Overview

The Material Agent addresses a fundamental challenge in 3D content creation: accurately identifying object parts and assigning suitable materials based on visual analysis. By leveraging Vision-Language Models, it can:

- **Identify object parts** from multi-view 3D renders
- **Understand visual characteristics** through geometric and structural analysis
- **Select appropriate materials** from a provided material library
- **Provide reasoning** for material choices based on functional requirements
- **Orchestrate complete workflows** through a unified pipeline command

### Key Features

- VLM-powered material assignment with multi-view analysis
- Pipeline orchestration with automatic data flow between steps
- Material library matching with fuzzy validation
- Scene pipeline for large multi-asset USD files
- RAG enhancement with technical specification documents
- Batch processing with parallel execution
- Checkpointing and resume from failures
- USD instance handling for cost savings and consistency

## Prefer the REST service?

This README covers the `material-agent` CLI (Option B in the root [README](../../README.md#two-ways-to-use-this)). If you'd rather drive the same pipeline over HTTP with session management and progress streaming, see [`../material_agent_service/`](../material_agent_service/) — it brings up with a single `docker compose up`.

## Installation

From the repository root:

```bash
# Install the core library first
uv pip install -e .

# Install the material agent
uv pip install -e apps/material_agent
```

For development:

```bash
uv pip install -e "apps/material_agent[dev]"
```

### Scene Optimizer (required for the `optimize_usd` pipeline step)

The first pipeline step, `optimize_usd`, runs a local Scene Optimizer
subprocess. Before the first CLI run, fetch the public Scene Optimizer
Core package (~332 MB, one-time):

```bash
./scripts/fetch_build_resources.sh
```

The agent auto-discovers the unpacked package at
`.build-resources/scene_optimizer_core/` when invoked from the repo root.
To point at a different location, set `WU_SO_PACKAGE_DIR`.

(Users of the `material_agent_service` docker-compose stack don't need to
do this separately — the Dockerfile runs the fetch during `docker compose
build`.)

**Escape hatches** (only when you really don't want the local fetch):

- `optimize_usd.enabled: false` in your config — skips the step entirely.
  Use for fast evaluation or when working with already-flat USD assets.
- `optimize_usd.backend: remote` + `NVCF_OPTIMIZER_FUNCTION_ID` +
  `NGC_API_KEY` — runs optimization on a remote NVCF function instead.

## Environment Setup

Copy `.env_example` to `.env` and add your VLM provider API key:

```bash
cp .env_example .env
```

You only need a key for the backend you plan to use:

| Backend | Environment Variable | Provider |
|---------|---------------------|----------|
| `nim` | `NVIDIA_API_KEY` | [NVIDIA NIM](https://build.nvidia.com/) |
| `openai` | `OPENAI_API_KEY` | [OpenAI](https://platform.openai.com/) |
| `anthropic` | `ANTHROPIC_API_KEY` | [Anthropic](https://console.anthropic.com/) |
| `gemini` | `GOOGLE_API_KEY` | [Google Gemini](https://aistudio.google.com/) |

### Rendering Backend

The pipeline renders multi-view images of each prim for VLM analysis. Two
options:

- **Local via `material_agent_service`** (recommended for getting started) —
  bring up the bundled docker-compose stack, which starts an OVRTX rendering
  API container. Point `material-agent` at it by exporting
  `RENDER_ENDPOINT=http://localhost:8001` (the default port exposed by the
  service's `docker-compose.yml`).
- **Remote NVCF rendering function** — if you have access to a deployed
  [NVIDIA Cloud Function](https://docs.nvidia.com/cloud-functions/) that
  serves the OVRTX render API, set `RENDER_ENDPOINT` to its full URL (or
  set `NVCF_RENDER_FUNCTION_ID` to the function ID and
  `NGC_API_KEY` for auth).

The `unified_example.yaml` config uses `backend: remote`, which resolves
via `RENDER_ENDPOINT` → `NVCF_RENDER_FUNCTION_ID`. Without one of these
set, the rendering steps will fail.

## Quick Start

### Run the Example Pipeline

```bash
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
# RENDER_ENDPOINT or NVCF_RENDER_FUNCTION_ID must be set — see above.
material-agent run apps/material_agent/configs/unified_example.yaml
```

### Try with Public SimReady Assets

Once the shipped ladder example runs, you can try the same pipeline on
assets from NVIDIA's public SimReady catalogs. Download a curated prop
and run the pipeline on it with the dataset's thumbnail as the reference
image:

```bash
# Example: steel rolling scaffold from the HuggingFace SimReady Warehouse
pip install -U huggingface_hub
hf download --repo-type dataset nvidia/PhysicalAI-SimReady-Warehouse-01 \
  --include "Props/general/SM_SteelRollingScaffold_A01_01/*" \
  --local-dir ~/content-agents-data/simready/hf/

# Copy the example config, then edit input.usd_path / reference_images
# to point at the downloaded prop (absolute paths — ~ is not expanded
# by the config loader):
cp apps/material_agent/configs/unified_example.yaml \
   apps/material_agent/configs/simready_scaffold.yaml
# ...edit, then:
material-agent run apps/material_agent/configs/simready_scaffold.yaml
```

Four curated assets (HF: scaffold, cleaning trolley; GitHub
`NVIDIA/simready-foundation`: electricians toolbox, UR10 robot arm) are
documented in
[`.claude/skills/material-agent/references/simready-quickstart.md`](../../.claude/skills/material-agent/references/simready-quickstart.md),
including the VLM key pre-flight probe and the `skip_instances: false`
caveat required for the UR10.

### Create a Configuration

```bash
# Interactive configuration wizard
material-agent configure my_pipeline.yaml

# With a materials manifest
material-agent configure my_pipeline.yaml -m data/materials/materials.yaml

# With reference images
material-agent configure my_pipeline.yaml -m materials.yaml -r ref1.jpg -r ref2.jpg
```

## CLI Reference

### Pipeline Command (Recommended)

```bash
# Run complete end-to-end pipeline
material-agent run CONFIG

# Skip specific steps
material-agent run CONFIG --skip build_dataset_usd

# Run only specific steps
material-agent run CONFIG --only predict,apply

# Resume from checkpoint after failure
material-agent run CONFIG --resume

# Dry run to see execution plan
material-agent run CONFIG --dry-run

# Verbose logging
material-agent run CONFIG -v
```

### Individual Step Commands

```bash
material-agent predict CONFIG              # VLM prediction only
material-agent apply CONFIG                # Apply materials to USD
material-agent benchmark CONFIG            # Benchmark with LLM-judge scoring
material-agent evaluate CONFIG             # Evaluate existing predictions
material-agent build-dataset usd CONFIG    # Build dataset from USD
material-agent build-dataset prepare-dataset CONFIG  # Prepare dataset for VLM
material-agent configure CONFIG            # Interactive config creation
material-agent generate-manifest USD OUT   # Generate materials.yaml from USD
```

### Scene Pipeline (Large Multi-Asset Scenes)

```bash
# Full end-to-end
material-agent scene run CONFIG --workers 4 -v

# Individual steps
material-agent scene analyze CONFIG -v
material-agent scene extract CONFIG -v
material-agent scene run-agent CONFIG --workers 4 -v
material-agent scene collect CONFIG -v
```

## Pipeline Steps

1. `optimize_usd` -- Flatten/deinstance USD via scene optimizer
2. `render_preview` -- Lightweight whole-scene preview rendering
3. `generate_reference_image` -- Generate photorealistic reference images
4. `build_dataset_usd` -- Render prim views for VLM input
5. `build_dataset_prepare_dataset` -- Prepare dataset entries with material specs
6. `predict` -- VLM inference for material assignment
7. `validate_predictions` -- Validate/repair predicted material names
8. `harmonize_predictions` -- Resolve conflicts for instanced parts
9. `apply` -- Apply predicted materials to USD
10. `render` -- Render final output

## Configuration

Pipeline configs are YAML files under `configs/`. Use `unified_example.yaml` as a template:

```yaml
project:
  name: "my_asset"

input:
  usd_path: "path/to/asset.usd"

materials:
  path: "path/to/materials.yaml"

predict:
  vlm:
    backend: nim                    # or: openai, anthropic, gemini
    model: qwen/qwen3.5-397b-a17b    # model from your chosen provider

render:
  backend: ovrtx
```

Paths in config files are relative to the config file's directory.

## Documentation

- **[API Reference](docs/api.md)** -- Python API reference
