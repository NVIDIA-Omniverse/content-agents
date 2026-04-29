---
name: material-agent
description: Run the Material Agent CLI for VLM-based material assignment to 3D objects. Use when user wants to assign materials to a USD file, run the material agent pipeline, benchmark material predictions, build a dataset from USD files, apply materials, configure a material agent run, or try a quickstart demo on public SimReady assets. Trigger phrases include "assign materials", "run material agent", "material agent pipeline", "benchmark materials", "build dataset from USD", "apply materials to USD", "configure material agent", "material agent quickstart", "try material agent", "demo material agent with simready", "run material agent on simready asset".
---

# Material Agent CLI

The Material Agent is a VLM-based system for intelligent material assignment to 3D object parts. It analyzes visual characteristics of rendered object components and assigns appropriate materials from a constrained list.

## Prerequisites

- The `material-agent` CLI must be installed and available on PATH
- Environment variables from `.env` must be set:
  - **VLM/LLM**: `NVIDIA_API_KEY` (nim backend, default — uses https://integrate.api.nvidia.com/v1). Alternatives: `OPENAI_API_KEY` (OpenAI), `ANTHROPIC_API_KEY` (Anthropic), `GOOGLE_API_KEY` (Google Gemini)
  - **Remote rendering/optimization** (optional): `RENDER_ENDPOINT` / `OPTIMIZER_ENDPOINT`, plus any auth required by that service. Skip if using local rendering.
  - **AWS** (optional, only for S3 asset upload): `WU_S3_BUCKET`, `WU_S3_PROFILE`, `WU_S3_REGION`, plus standard AWS credentials.
- A unified YAML configuration file (see references/config-reference.md)
- A materials manifest YAML with material names, descriptions, and USD bindings

## Primary Command: `run`

The `run` command executes the full multi-step pipeline. This is the recommended way to use the material agent.

```bash
material-agent run <config.yaml> [OPTIONS]
```

### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `config` | Yes | Path to unified YAML configuration file |

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

1. **optimize_usd** -- Flatten/deinstance USD via the configured scene optimizer
2. **render_preview** -- Render lightweight whole-scene previews (optional, for ref image generation)
3. **generate_reference_image** -- Generate photorealistic reference images from previews + text prompt (optional)
4. **build_dataset_usd** -- Render multi-view prim images for VLM input
5. **build_dataset_pdf_vectorstore** -- Build RAG vector store from PDFs (optional)
6. **build_dataset_prepare_dataset** -- Prepare VLM dataset with prompts and context
7. **predict** -- VLM inference for material assignment
8. **apply** -- Apply predicted materials back to USD
9. **render** -- Render final output images

See references/pipeline-steps.md for details on each step.

## Other Commands

| Command | Description |
|---------|-------------|
| `material-agent configure <output.yaml>` | Interactive config creation wizard |
| `material-agent predict <config.yaml>` | Run VLM prediction only (alias for `run --only predict`) |
| `material-agent apply <config.yaml>` | Apply materials only (alias for `run --only apply`) |
| `material-agent benchmark <config.yaml>` | Predict + evaluate with LLM-judge scoring |
| `material-agent evaluate <config.yaml> [predictions.jsonl]` | Evaluate existing predictions |
| `material-agent build-dataset usd <config.yaml>` | Build dataset from USD renders |
| `material-agent build-dataset pdf_vectorstore <config.yaml>` | Build vector store from PDFs |
| `material-agent build-dataset prepare-dataset <config.yaml>` | Prepare dataset with specs |

See references/commands.md for full argument/option details on each command.

## Common Workflows

### Run full pipeline on a USD file

```bash
material-agent run configs/unified_example.yaml
```

### Preview what the pipeline will do

```bash
material-agent run configs/unified_example.yaml --dry-run
```

### Run only specific steps

```bash
material-agent run configs/unified_example.yaml --only predict,apply,render
```

### Skip a step

```bash
material-agent run configs/unified_example.yaml --skip optimize_usd
```

### Resume after a failure

```bash
material-agent run configs/unified_example.yaml --resume
```

### Run with AI-generated reference images (no reference photos needed)

```bash
material-agent run configs/unified_example_gen_ref.yaml
```

This enables `render_preview` and `generate_reference_image` steps. The pipeline renders a scene preview, then uses an image generation model to create a photorealistic reference image from a text prompt (e.g., "aluminium frame with blue plastic tray"). The generated reference is automatically injected into the dataset for VLM prediction.

### Create a new config interactively

```bash
material-agent configure my_pipeline.yaml -m materials/my_materials.yaml -r reference.jpg
```

### Benchmark VLM performance

```bash
material-agent benchmark configs/benchmark.yaml -d dataset.jsonl -o results/
```

## Quickstart: Demo with SimReady Assets

When the user wants a zero-data-of-their-own demo — "try material agent",
"run material agent on an example", "download a SimReady asset and material
it" — follow references/simready-quickstart.md.

It covers:

1. **Downloading a curated asset** from either the HuggingFace
   `nvidia/PhysicalAI-SimReady-Warehouse-01` dataset (scaffold, trolley) or
   `NVIDIA/simready-foundation` on GitHub (toolbox, UR10), using
   `hf download` / sparse `git clone`. The shipped ladder example
   (`apps/material_agent/data/examples/ladder/`) already covers the
   zero-setup demo path — this skill is for users who want to go beyond it.
2. **Writing a config** pointing at the downloaded USD with the dataset's
   built-in thumbnail as the reference image, using the shipped default
   material library (`apps/material_agent/data/materials/material_libs_default/materials.yaml`).
3. **Running the pipeline** end-to-end with `material-agent run`.

Assets land in `~/content-agents-data/simready/` by default (override with
`CONTENT_AGENTS_DATA`). For UR10 the config must set
`prim_filters.skip_instances: false` — otherwise the agent sees no meshes.

## Creating a Config

When asked to run the material agent on a USD file, write a config YAML based on the template below. Adapt the marked fields to the user's needs and save it next to the input USD file.

**Fields to adapt:**
- `project.name` and `project.session_id` -- descriptive name for the run
- `input.usd_path` -- path to the user's input USD file
- `input.reference_images` -- paths to reference images (if any; remove section if none)
- `materials.path` -- path to the materials YAML file (see "Finding Materials" below)
- `steps.predict.vlm.model` -- VLM model to use
- `steps.render.enabled` -- whether to render final output

All paths are resolved relative to the config file's location. Working directory is auto-derived as `.{session_id}/` next to the config file.

### Config Template

```yaml
project:
  name: "CHANGE_ME"
  session_id: "CHANGE_ME"
  description: "Material agent pipeline"

input:
  usd_path: "CHANGE_ME"
  reference_images:
    - CHANGE_ME.jpeg

output:
  layer_only: false
  flatten_output: false

materials:
  path: "CHANGE_ME"

steps:
  optimize_usd:
    enabled: true

  # Optional: render scene preview + generate reference image from text prompt
  # Enable these two steps INSTEAD of providing input.reference_images
  # render_preview:
  #   enabled: true
  #   cameras: ["+x+y+z"]
  #   should_reset_materials: true
  #   use_lights: false
  #   image_width: 512
  #   image_height: 512
  #   background_color: [1.0, 1.0, 1.0]
  #
  # generate_reference_image:
  #   enabled: true
  #   image_gen:
  #     backend: gemini                       # Public; needs GOOGLE_API_KEY
  #     model: gemini-3-pro-image-preview
  #   prompt: "Describe desired materials here, e.g. aluminium frame with blue plastic tray"
  #   num_images: 1

  build_dataset_usd:
    renderer:
      backend: remote
      image_width: 512
      image_height: 512
      cull_style: back
      should_highlight_prim: false
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

  build_dataset_pdf_vectorstore:
    enabled: false

  build_dataset_prepare_dataset:
    enabled: true
    include_ground_truth: false
    include_prim_path_context: true
    pdf_conversion:
      dpi: 150
      format: "png"
      first_page: 1
      last_page: 2
    prompts:
      vlm_system: |
        You are an expert at identifying object parts and their materials.
        Provided images are 3D renderings of a part of an object from different angles.

        The material property in the render are irrelevant to the task, you will only consider
        the shape and position of the part.

        The part of interest are highlighted in orange contour outline.

        When the part is occluded, it may not contain orange contour outline, but just the part itself.

        Other parts are rendered in muted colors again their color is irrelevant to the task
        just consider the shape and position of the parts.

        When asked to identify the material of the part, you should only focus on the part
        that is highlighted in orange contour outline.

        In summary:
        - DO NOT judge the material by the material of the rendered image. Only consider the shape and position of the part from the rendered images.
        - DO judge the color and material by the reference images.

        Additional context of the part and materials will be provided with the question.

        Available materials:
        {materials_list}

        Please answer the question with a structured JSON output using the following format:
        {{
        "material": "material name"
        }}

        Answer the task requirements in the following format:
        <reasoning>your reasoning</reasoning>
        <answer>your answer</answer>
      vlm_user: |
        Please identify the highlighted part and select the appropriate material from the predefined list of materials.

        you will match the look of the asset exactly to the reference images.

        you will think about the best material for it

        but if you can't find it in the list of materials, you will select the closest match.

        Below is the additional context of the part and materials:
        {context}
      vlm_image_prompts:
        reference_images:
          - "This is a reference image of the asset. You will match this look exactly"
        prim_only: "This is a rendered part of interest only without highlighting."
        composition: "This is an orthographic view of the object with the part of interest highlighted with an orange outline."

  predict:
    vlm:
      backend: nim
      model: qwen/qwen3.5-397b-a17b
      temperature: 1.0
      max_completion_tokens: 24576
    max_workers: 16
    llm:
      backend: nim
      model: qwen/qwen3.5-397b-a17b
      temperature: 0.1
      max_tokens: 512
    report:
      image_max_size: 256
      image_format: jpeg
      image_quality: 75

  apply:
    layer_only: false
    flatten_output: false

  restore_usd:
    enabled: false

  render:
    enabled: true
    backend: "remote"
    image_width: 1024
    image_height: 1024
    camera_corners: ["+x+y+z", "-x-y-z"]
    camera_margin: 1.0
    background_color: [1.0, 1.0, 1.0]

advanced:
  keep_temp_files: true
  log_level: INFO
```

See references/config-reference.md for full config schema documentation.

### Finding Materials

A materials file is a YAML with a `library_path` (USD containing material definitions) and `entries` (list of available materials). Format:

```yaml
library_path: "path/to/material_libs.usd"
entries:
  - name: "Aluminum Polished"
    description: "A polished aluminum for structural parts"
    binding: "/World/metal_library/Looks/Aluminum_Polished"
  - name: "Polycarbonate Blue"
    description: "A blue polycarbonate for decorative components"
    binding: "/World/plastic_library/Looks/Polycarbonate_Blue"
```

To find the right materials file:
1. **Default material library**: Check `apps/material_agent/data/materials/material_libs_default/` in the repo — it ships with `materials_libs_v2.usd` and a manifest YAML.
2. **User-provided**: Ask the user if they have a custom materials YAML for their project.
3. **Create one**: Write a new materials YAML following the format above, pointing to the user's material USD file and listing available materials.

## Common Issues

### "API key required" error
Cause: Missing environment variable for the selected VLM backend.
Solution: Set the required keys in `.env`:
- **VLM/LLM**: `NVIDIA_API_KEY` (nim backend, default — uses https://integrate.api.nvidia.com/v1). Alternatives: `OPENAI_API_KEY` (OpenAI), `ANTHROPIC_API_KEY` (Anthropic), `GOOGLE_API_KEY` (Google Gemini)
- **Remote rendering/optimization** (optional): `RENDER_ENDPOINT` / `OPTIMIZER_ENDPOINT`, plus any auth required by that service. Skip if using local rendering.
- **AWS** (optional, only for S3 asset upload): `WU_S3_BUCKET`, `WU_S3_PROFILE`, `WU_S3_REGION`, plus standard AWS credentials.

### Pipeline fails midway
Solution: Re-run with `--resume` to continue from the last checkpoint.

### "Forbidden path key in step config" validation error
Cause: Step configs must not contain path keys like `usd_path` or `output_dir`.
Solution: Remove those keys from step configs. Paths are auto-derived from `project` and `input` sections.

### Config file path resolution
All relative paths in configs are resolved relative to the config file's directory, not the current working directory.

### Config template
See the references/config-template.yaml bundled with this skill for a complete, ready-to-adapt config template with all steps and options.
