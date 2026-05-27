---
name: texture-agent-cli
description: Run the Texture Agent CLI for AI-driven texture generation on USD materials. Use when the user wants to run the texture-agent CLI directly, add textures to a materialized USD asset, generate weathering or aging effects, run the texture pipeline, apply textures per material or per prim, or create texture variations.
version: "0.1.0"
author: NVIDIA Content Agents
tags:
  - content-agents
  - texture-agent
  - cli
  - usd
  - image-generation
tools:
  - Shell
  - Filesystem
  - Python
  - wu
compatibility: Requires the texture-agent CLI, a repo Python environment, a USD asset with materials, provider credentials or a service endpoint for the selected image-generation backend, and a render endpoint only when preview/final rendering is enabled.
---

# Texture Agent CLI

The Texture Agent generates and applies AI-driven texture maps to USD assets
with existing materials. It can add weathering, aging, scratches, rust, patina,
and other visual detail through per-material or per-prim texture generation.

## When to Use

- Use when the user asks to run `texture-agent` directly from the command line.
- Use when the user has a materialized USD and wants generated texture maps.
- Use when the user wants per-material texture variation or unique per-prim
  texture variation.
- Use when the user wants to discover available materials before writing
  prompts.
- Use service or Docker deploy skills instead when the user wants to operate
  the REST service rather than the local CLI.

## Limitations

- Keep secrets out of chat and commits. Tell the user to set provider keys in
  their local environment or repo-root `.env`; never ask them to paste keys.
- The input USD must already have materials suitable for texture assignment.
  Run the Material Agent first when the asset has no material bindings.
- Config paths resolve relative to the config file, not the current shell
  directory.
- `simple_image_gen` uses a configured image-generation provider and does not
  require a local GPU by itself.
- `service` mode requires a reachable Texture Variation API-compatible
  endpoint.
- Rendering previews or final images requires a render endpoint. For a local
  OVRTX Docker sidecar, use `RENDER_ENDPOINT=http://localhost:8001`.

## Prerequisites

- Activate the repo Python environment before running commands.
- Confirm `texture-agent` is installed and on `PATH`.
- Set credentials or endpoints for the selected image-generation backend.
  Public defaults commonly use `GOOGLE_API_KEY` for Gemini or
  `NVIDIA_API_KEY` for NVIDIA-hosted NIM.
- Set `RENDER_ENDPOINT` only when `render_previews` or `render` uses a remote
  render service.
- Prepare a texture config YAML. The public reference config is
  `apps/texture_agent/configs/texture_example.yaml`.

## Instructions

1. Start from the repo root and activate `.venv`.
2. Choose or create a texture config. For a first run, copy
   `apps/texture_agent/configs/texture_example.yaml`.
3. Point `input.usd_path` at a materialized USD and choose `texture.backend`.
4. Run discovery to list materials before writing prompts.
5. Add `material_textures` entries for material names. For prim-specific
   variants, keep the prim overrides nested under that material's `per_prim`
   map.
6. Run a dry run before a new or heavily edited config.
7. Run the full pipeline, or use `--only` and `--skip` to control execution.
8. Report generated texture files and the textured USD from the output format.

```bash
source .venv/bin/activate
texture-agent discover apps/texture_agent/configs/texture_example.yaml
texture-agent run apps/texture_agent/configs/texture_example.yaml --dry-run
texture-agent run apps/texture_agent/configs/texture_example.yaml
```

### Primary Command

```bash
texture-agent run <config.yaml> [OPTIONS]
```

| Option | Description |
|---|---|
| `--skip <steps>` | Comma-separated steps to skip. |
| `--only <steps>` | Comma-separated steps to run exclusively. |
| `--session-id <id>` | Reuse or override the session ID. |
| `--resume` | Reuse existing artifacts when supported by the step. |
| `--dry-run` | Show the execution plan without running. |
| `--verbose`, `-v` | Enable debug logging. |

### Other Commands

| Command | Description |
|---|---|
| `texture-agent discover <config.yaml>` | Discover materials in the input USD. |
| `texture-agent generate <config.yaml>` | Generate and blend textures without applying them to USD. |
| `texture-agent apply <config.yaml>` | Apply already generated textures to USD. |

### Pipeline Steps

| Step | Description |
|---|---|
| `prepare_uvs` | Generate or normalize UVs for meshes that need them. |
| `discover_materials` | Find OpenPBR materials and expand them to per-prim units when configured. |
| `generate_prompts` | Generate missing material prompts when `auto_prompt.enabled` is true. |
| `render_previews` | Render optional material preview images through the configured render service. |
| `generate_textures` | Generate albedo, normal, and roughness textures through the configured backend. |
| `blend_textures` | Composite generated textures onto material base colors. |
| `apply_textures` | Set texture files on USD materials, cloning materials for per-prim mode. |
| `render` | Render final result images when enabled. |

### Config Authoring

Start from the public reference config and adapt these fields:

```yaml
project:
  name: "my_texture_run"
  session_id: "my_texture_run"

input:
  usd_path: "path/to/materialized_asset.usd"

texture:
  backend: "simple_image_gen"
  image_gen:
    backend: gemini
    model: gemini-3-pro-image-preview
  mode: "per_prim"
  size: 1024
  workers: 4
  uv_mode: "box"

material_textures:
  Material_Name:
    prompt: "weathered painted metal with light edge wear"
    opacity: 0.80
    per_prim:
      /World/Prim_Path:
        prompt: "heavier scratches on the front edge"
        opacity: 0.90
```

For a remote texture service, use:

```yaml
texture:
  backend: "service"
  endpoint: "http://host:port"
  workers: 1
```

### Common Workflows

```bash
# Run only generation and application.
texture-agent run apps/texture_agent/configs/texture_example.yaml --only generate_textures,apply_textures

# Skip optional preview rendering.
texture-agent run apps/texture_agent/configs/texture_example.yaml --skip render_previews

# Discover materials before writing prompts.
texture-agent discover apps/texture_agent/configs/texture_example.yaml
```

### Modes and Backends

- `per_material` mode creates one texture set per material and shares it across
  geometry that uses the material.
- `per_prim` mode clones materials so each prim can receive unique textures.
- `simple_image_gen` calls the configured image-generation provider directly.
- `service` calls a remote Texture Variation API-compatible service.
- Generated textures are blended onto the material base color at the configured
  opacity so untextured areas retain the original look.

## Output Format

Report these items after a run or handoff:

- Command executed and whether it was full pipeline, `--only`, `--skip`,
  `--resume`, or `--dry-run`.
- Config path, session ID, backend, and mode.
- Working directory, usually `.<session_id>/` next to the config.
- Key artifacts when present:
  - `generated/` for raw generated PBR texture maps.
  - `textures/` for blended and localized texture maps.
  - `output/textured_output.usd` for the textured USD.
  - `renders/` for final render images when `render` ran.
- Any missing credentials, unreachable service endpoints, missing material
  bindings, or render endpoint failures.

For a quick before/after render:

```bash
wu render-usd <input.usd> -o before.png --focus /RootNode --margin 0.6 --focal-length 100
wu render-usd <output_dir>/textured_output.usd -o after.png --focus /RootNode --margin 0.6 --focal-length 100
```

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| No materials discovered | The input USD has no suitable material bindings. | Run Material Agent first or point at a materialized USD. |
| Image generation credential error | The configured image-generation backend has no key. | Set the required key locally or in `.env`; do not paste it into chat. |
| Service backend is unreachable | `texture.endpoint` is missing or the service is down. | Check the endpoint URL and service health; reduce workers to 1 for fragile endpoints. |
| Preview or final render fails | A render endpoint is required for render steps. | Set `RENDER_ENDPOINT` or disable `render_previews` and `render`. |
| Textures look too strong or too subtle | Opacity is mismatched to the desired effect. | Use 0.6-0.8 for moderate wear and 0.9+ for heavy damage. |
| Per-prim changes affect multiple meshes | The config is still in `per_material` mode. | Set `texture.mode: "per_prim"` so materials are cloned per geometry prim. |
