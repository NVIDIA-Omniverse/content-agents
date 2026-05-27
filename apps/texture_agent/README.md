# Texture Agent

AI-driven texture generation and application for USD assets with OpenPBR,
MaterialX, and MDL-style material metadata.

## Overview

The Texture Agent takes a USD file with materials already assigned (e.g., output of the Material Agent) and fills empty texture slots with AI-generated texture maps -- transforming flat, constant-color surfaces into visually rich textured ones.

### Key Features

- Material texture generation for OpenPBR, MaterialX, and MDL-style metadata
  (albedo, roughness, metalness, normal)
- Per-material or per-prim texture modes
- Texture blending and compositing
- Multiple generation backends
- UV readiness reporting with policy-controlled projection UV generation

## Prefer the REST service?

This README covers the `texture-agent` CLI (Option B in the root [README](../../README.md#two-ways-to-use-this)). If you'd rather drive the same pipeline over HTTP with session management and progress streaming, see [`../texture_agent_service/`](../texture_agent_service/) — it brings up with a single `docker compose up`.

## Installation

From the repository root:

```bash
uv pip install -e .
uv pip install -e apps/texture_agent
```

## Rendering

Two of the pipeline steps render USD views and need a rendering endpoint:

- `render_previews` — renders the current state of each material for VLM-based prompt generation.
- `render` — final render of the textured output.

Both default to `backend: remote`, so running `texture-agent run` without a rendering endpoint will fail at the first rendering step. Options:

- **Point at a running OVRTX rendering API** — export `RENDER_ENDPOINT=http://localhost:8001` (the port exposed by the bundled `material_agent_service` / `physics_agent_service` sidecars, or a standalone OVRTX deployment).
- **Use an NVCF-hosted function** — set `NVCF_RENDER_FUNCTION_ID` and `NGC_API_KEY` instead of `RENDER_ENDPOINT`.
- **Skip the rendering steps** — use `--skip render_previews,render`, or disable them in the config's `steps.render_previews.enabled` / `steps.render.enabled`. Texture generation and application still run; you just don't get previews or a final composite.

## Quick Start

```bash
source .venv/bin/activate
texture-agent run apps/texture_agent/configs/texture_example.yaml
```

## CLI Reference

```bash
# Run complete pipeline
texture-agent run CONFIG

# Pipeline options
texture-agent run CONFIG --skip render_previews                      # skip a step
texture-agent run CONFIG --only generate_textures,apply_textures     # run specific steps
texture-agent run CONFIG --resume                                    # reuse existing artifacts
texture-agent run CONFIG --session-id previous-run                   # reuse a session directory
texture-agent run CONFIG --dry-run                                   # show execution plan
texture-agent run CONFIG --verbose                                   # verbose logging

# Individual commands
texture-agent discover CONFIG        # Discover materials in the scene
texture-agent generate CONFIG        # Generate textures only
texture-agent apply CONFIG           # Apply textures to USD only
```

To resume after a partial local run, use the same config and either rerun
`texture-agent run CONFIG --resume` or split the last stages explicitly:
`texture-agent generate CONFIG` writes generated/blended texture artifacts,
and `texture-agent apply CONFIG` reloads those artifacts from the config's
working directory before writing the textured USD.

## Pipeline Steps

1. `prepare_uvs` -- Inspect UVs, preserve valid UVs, and optionally prepare missing UVs
2. `discover_materials` -- Discover and catalog materials in the scene
3. `render_previews` -- Render preview images of the current state
4. `generate_textures` -- Generate texture images via the configured backend
5. `blend_textures` -- Blend generated textures (e.g., albedo compositing)
6. `apply_textures` -- Apply generated textures to USD materials
7. `render` -- Render final output

## Configuration

Pipeline configs are YAML files under `configs/`. Key settings:

```yaml
project:
  name: "my_textured_asset"

input:
  usd_path: "path/to/materialized_asset.usd"

generate_textures:
  backend: simple_image_gen     # or: service
  mode: per_material            # or: per_prim
```

UV behavior is controlled under `texture`:

```yaml
texture:
  uv_policy: generate_missing       # validate, preserve_or_fix, generate_missing, force_projection
  uv_projection: box                # box or planar for Python projection
  uv_normalize_out_of_range: false  # preserve tiled UVs by default
```

`prepare_uvs` writes `prepared/uv_report.json` in the run directory so missing,
invalid, repaired, generated, and out-of-range UV conditions are inspectable.

By default, `material_textures` is also a strict processing scope: materials
not listed there are skipped. Set `auto_prompt.enabled: true` to generate
prompts for discovered materials that are missing explicit specs.

### Reviewable Beta Runs

For inspectable v0.4-style runs, use an explicit `material_textures` scope,
keep `auto_prompt.enabled: false` unless you want discovered materials to be
added, and inspect these outputs before treating the run as successful:

- `prepared/uv_report.json`
- Generated and blended PNG maps in the run directory
- Textured output USD or package
- Service `/pipeline/{session_id}/results` stats when using the REST service
- Final or close-up render when rendering is enabled
- `usd-validation-nvidia` schema (`Basic`, `Layer`, `Layout`, `Other`),
  and `Material` rule coverage for textured USD outputs
- UV readiness in `prepared/uv_report.json`

Real backend runs are useful evidence, but repeatable fake-backend smoke tests
should remain the CI gate for CLI/service parity.

### Texture Modes

- **`per_material`** (default) -- One texture set per material, shared across all geometry referencing it.
- **`per_prim`** -- Clones materials per geometry prim, allowing unique textures on each mesh.
