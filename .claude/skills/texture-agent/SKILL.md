---
name: texture-agent
description: Run the Texture Agent pipeline for AI-driven texture generation on USD materials. Use when user wants to add textures to a materialized USD asset, generate weathering/aging textures, run the texture pipeline, apply textures per-prim, or create texture variations. Trigger phrases include "add textures", "run texture agent", "texture pipeline", "generate textures", "weather the asset", "age the materials", "per-prim textures", "texture variations".
---

# Texture Agent Pipeline

The Texture Agent generates and applies AI-driven texture maps to USD assets with OpenPBR materials. It takes material-agent output (or any USD with materials) and adds visual detail — weathering, aging, scratches, rust, patina, etc.

## Prerequisites

- The `texture-agent` CLI must be installed and available on PATH
- `NVIDIA_API_KEY` for image generation (nim backend, default — uses https://integrate.api.nvidia.com/v1)
- Remote rendering credentials, if using a remote render service
- A texture config YAML file

## Command Reference

```bash
texture-agent run <config.yaml> [OPTIONS]
```

### Subcommands

| Command | Description |
|---------|-------------|
| `texture-agent run <config.yaml>` | Run the full texture pipeline |
| `texture-agent discover <config.yaml>` | Discover and list materials in the input USD |
| `texture-agent generate <config.yaml>` | Generate and blend textures without applying to USD |
| `texture-agent apply <config.yaml>` | Apply textures to USD (assumes textures already generated) |

### Options for `run`

| Option | Description |
|--------|-------------|
| `--skip` | Comma-separated step names to skip |
| `--only` | Comma-separated step names to run exclusively |
| `--dry-run` | Show execution plan without running |
| `--verbose` / `-v` | Enable verbose logging |

## Example Configs

| Config | Asset | Mode | Description |
|--------|-------|------|-------------|
| `apps/texture_agent/configs/texture_ladder.yaml` | Ladder | per-material | Basic weathering on ladder materials |
| `apps/texture_agent/configs/texture_ladder_per_prim.yaml` | Ladder | per-prim | Unique texture per geometry prim |
| `apps/texture_agent/configs/texture_agv_per_prim.yaml` | Siemens AGV | per-prim | Industrial weathering on AGV (60 prims) |
| `apps/texture_agent/configs/texture_ladder_step1x3d.yaml` | Ladder | service | Step1X-3D backend via REST |
| `apps/texture_agent/configs/texture_example.yaml` | Generic | per-material | Reference config template |

## Config Structure

```yaml
project:
  name: "my_texture_run"
  session_id: "my_texture_run"

input:
  usd_path: "path/to/materialized_asset.usd"  # relative to config file

texture:
  backend: "simple_image_gen"     # "simple_image_gen" or "service"
  image_gen:                       # for simple_image_gen backend
    backend: gemini                # Public; needs GOOGLE_API_KEY
    model: gemini-3-pro-image-preview
  # endpoint: "http://host:port"  # for service backend
  mode: "per_prim"                 # "per_material" (default) or "per_prim"
  size: 1024
  workers: 4                       # parallel generation workers
  uv_mode: "box"                   # "box" or "planar" (auto UV projection)

material_textures:
  Material_Name:
    prompt: "description of desired texture"
    opacity: 0.80                  # blend strength (0=none, 1=full)
    # Optional per-prim overrides:
    per_prim:
      /World/Prim_Path:
        prompt: "specific prompt for this prim"

steps:
  prepare_uvs: { enabled: true }
  discover_materials: { enabled: true }
  render_previews: { enabled: false }
  generate_textures: { enabled: true, skip_existing: false }
  blend_textures: { enabled: true, default_opacity: 0.75, output_size: 1024 }
  apply_textures: { enabled: true }
  render: { enabled: false }
```

## Pipeline Steps

| Step | Description |
|------|-------------|
| `prepare_uvs` | Auto-generate UVs for meshes without them (box/planar projection), fix interpolation, normalize range |
| `discover_materials` | Find OpenPBR materials, extract properties, expand to per-prim units |
| `render_previews` | Render each material on a sphere via the configured render service (optional) |
| `generate_textures` | Generate albedo + normal + roughness textures via AI |
| `blend_textures` | Composite generated textures onto material base colors |
| `apply_textures` | Set texture files on USD materials (with material cloning for per-prim) |
| `render` | Render final result via the configured render service (optional) |

## Common Workflows

### Run full pipeline

```bash
texture-agent run apps/texture_agent/configs/texture_ladder.yaml
```

### Dry run to preview execution plan

```bash
texture-agent run apps/texture_agent/configs/texture_ladder.yaml --dry-run
```

### Run only specific steps

```bash
texture-agent run apps/texture_agent/configs/texture_ladder.yaml --only generate_textures,apply_textures
```

### Skip a step

```bash
texture-agent run apps/texture_agent/configs/texture_ladder.yaml --skip render_previews
```

### Discover materials first

```bash
texture-agent discover apps/texture_agent/configs/texture_ladder.yaml
```

## Creating a New Config

1. Copy `apps/texture_agent/configs/texture_example.yaml`
2. Set `input.usd_path` to your materialized asset
3. Run discover to see available materials:
   ```bash
   texture-agent discover your_config.yaml
   ```
4. Add `material_textures` entries for materials you want to texture
5. Write prompts describing the desired weathering/aging effect
6. Set opacity (0.6-0.8 for moderate wear, 0.9+ for heavy damage)

## Rendering Results

After the pipeline, render before/after:

```bash
# Before (clean)
wu render-usd <input.usd> -o before.png --focus /RootNode --margin 0.6 --focal-length 100

# After (textured)
wu render-usd <output_dir>/textured_output.usd -o after.png --focus /RootNode --margin 0.6 --focal-length 100
```

Output files are in `<config_dir>/.<session_id>/output/`.

## Backend Types

### simple_image_gen (default)
Runs locally using Gemini image generation. Generates albedo + normal + roughness per material/prim. No GPU required.

### service
Calls a remote REST API implementing the Texture Variation API spec. For Step1X-3D or other dedicated texture generation services:

```yaml
texture:
  backend: "service"
  endpoint: "http://192.168.4.58:8000"
  workers: 1
```

## Key Concepts

- **Per-material mode** (default): One texture per material. All prims sharing a material get the same texture.
- **Per-prim mode**: Unique texture per geometry prim. Materials are cloned via `Sdf.CopySpec` so each prim gets its own texture slot. Enable with `texture.mode: "per_prim"`.
- **Texture blending**: Generated textures are composited onto the material's constant base_color at the configured opacity. This ensures untextured areas retain the original material color.
- **Auto UV preparation**: Meshes without UVs get box-projected coordinates. Wrong interpolation (`constant`) is fixed to `faceVarying`. Out-of-range UVs are normalized.
