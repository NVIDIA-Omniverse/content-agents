---
name: material-agent-cli
description: Run the Material Agent CLI for VLM-based material assignment to 3D objects. Use when the user wants to run the material-agent CLI directly, assign materials to a USD file, run or resume the material pipeline, benchmark material predictions, build datasets from USD files, apply predicted materials, configure a material run, or try a SimReady demo.
version: "0.1.0"
author: NVIDIA Content Agents
tags:
  - content-agents
  - material-agent
  - cli
  - usd
  - vlm
tools:
  - Shell
  - Filesystem
  - Python
  - wu
compatibility: Requires the material-agent CLI, a repo Python environment, provider credentials for the selected VLM/LLM/image-generation backends, a render endpoint for remote rendering configs, and a materials manifest with USD material bindings.
---

# Material Agent CLI

The Material Agent assigns materials to 3D object parts by rendering USD prims,
asking a VLM to choose from a constrained material library, and optionally
applying the predictions back to a USD layer.

## When to Use

- Use when the user asks to run `material-agent` directly from the command
  line.
- Use when the user wants to assign materials to a USD asset, resume a failed
  material pipeline, or apply existing predictions to USD.
- Use when the user wants to iteratively refine material assignments with a
  predict/apply/render/judge loop.
- Use when the user wants to benchmark material predictions or build VLM-ready
  datasets from USD renders.
- Use when the user wants a public SimReady demo asset with minimal local data
  setup.
- Use service or Docker deploy skills instead when the user wants to operate
  the REST service rather than the local CLI.

## Limitations

- Keep secrets out of chat and commits. Tell the user to set provider keys in
  their local environment or repo-root `.env`; never ask them to paste keys.
- The CLI needs a valid config YAML, a readable USD input, and a materials
  manifest containing material names, descriptions, and USD bindings.
- Config paths such as `input.usd_path`, `input.reference_images`, and
  `materials.path` resolve relative to the config file, not the current shell
  directory.
- Step configs must not contain path keys such as `usd_path`, `output_dir`,
  `dataset`, or `predictions_path`; the executor wires paths from the project
  and input sections.
- Remote rendering or optimization configs need deployed services. For a local
  OVRTX Docker sidecar, use `RENDER_ENDPOINT=http://localhost:8001` and keep
  render concurrency conservative unless the endpoint fronts multiple service
  instances.
- Generated reference images are optional and need a configured image
  generation backend plus its required key or endpoint.

## Prerequisites

- Activate the repo Python environment before running commands.
- Confirm `material-agent` is installed and on `PATH`.
- Set the VLM/LLM provider key required by the selected backend. Public
  defaults usually use `NVIDIA_API_KEY`; other supported backends can use
  `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or `GOOGLE_API_KEY`.
- Set `RENDER_ENDPOINT` and `OPTIMIZER_ENDPOINT` only when the config uses
  remote rendering or optimization services.
- Set `WU_S3_BUCKET`, `WU_S3_PROFILE`, `WU_S3_REGION`, and standard AWS
  credentials only when the run uploads assets to S3.
- Prepare a materials manifest YAML. The default library lives under
  `apps/material_agent/data/materials/material_libs_default/`.

## Instructions

1. Start from the repo root and activate `.venv`.
2. Choose a config. For a first local run, use
   `apps/material_agent/configs/unified_example.yaml`. For a user asset, copy
   that config or use `material-agent configure`.
3. Verify the config points at the input USD, optional reference images, and
   materials manifest. Keep relative paths relative to the config file.
4. Run a dry run before a new or heavily edited config.
5. Run the full pipeline, or use `--only`, `--skip`, and `--resume` to control
   execution.
6. Inspect the working directory and report the key artifacts from the output
   format.

```bash
source .venv/bin/activate
material-agent run apps/material_agent/configs/unified_example.yaml --dry-run
material-agent run apps/material_agent/configs/unified_example.yaml
```

### Primary Command

```bash
material-agent run <config.yaml> [OPTIONS]
```

| Option | Description |
|---|---|
| `--skip <steps>` | Comma-separated steps to skip. |
| `--only <steps>` | Comma-separated steps to run exclusively. |
| `--session-id <id>` | Reuse or override the session ID. |
| `--resume` | Continue from the last successful checkpoint. |
| `--dry-run` | Show the pipeline plan without executing. |
| `--clean` | Delete the working directory before starting. |
| `--verbose`, `-v` | Enable debug logging. |
| `--log-file <path>` | Write logs to a file. |
| `--log-level <level>` | Override the default `INFO` log level. |

### Other Commands

| Command | Description |
|---|---|
| `material-agent configure <output.yaml>` | Interactive config creation wizard. |
| `material-agent predict <config.yaml>` | Run VLM prediction only. |
| `material-agent apply <config.yaml>` | Apply predictions to USD only. |
| `material-agent refine <config.yaml>` | Iteratively refine materials with predict/apply/render/judge. |
| `material-agent benchmark <config.yaml>` | Predict and evaluate with LLM-judge scoring. |
| `material-agent evaluate <config.yaml> [predictions.jsonl]` | Evaluate existing predictions. |
| `material-agent build-dataset usd <config.yaml>` | Build dataset images from USD renders. |
| `material-agent build-dataset pdf_vectorstore <config.yaml>` | Build a RAG vector store from PDFs. |
| `material-agent build-dataset prepare-dataset <config.yaml>` | Prepare VLM dataset records. |

See `references/commands.md` for full command options.

### Pipeline Steps

The unified config schema recognizes these steps in execution order:

1. `validate_input` - establish an optional USD validation baseline before any
   material processing.
2. `optimize_usd` - flatten, split, deduplicate, or deinstance USD through the
   configured optimizer.
3. `render_preview` - render lightweight scene previews for reference-image
   generation.
4. `identify_asset` - classify the overall asset and derive prompt context from
   previews.
5. `generate_reference_image` - generate optional photorealistic reference
   images from previews and text prompts.
6. `build_dataset_usd` - render prim-level VLM input images.
7. `build_dataset_pdf_vectorstore` - build optional RAG context from PDFs.
8. `build_dataset_prepare_dataset` - assemble material specs, prompts, and
   rendered images into dataset records.
9. `cluster_prims` - group visually similar prims before prediction.
10. `predict` - run VLM material assignment.
11. `expand_cluster_predictions` - expand cluster-level predictions back to
    member prims.
12. `benchmark` - run prediction plus LLM-judge evaluation; mutually exclusive
    with `predict` in one run.
13. `validate_predictions` - validate or repair predicted material names.
14. `harmonize_predictions` - resolve conflicts for instanced or repeated
    parts.
15. `restore_usd` - remap predictions from optimized paths back to original
    paths before application or refinement.
16. `apply` - apply predictions to USD.
17. `evaluate` - score existing predictions against ground truth with an LLM
    judge.
18. `refine` - run the iterative predict/apply/render/judge loop; mutually
    exclusive with `apply` in one run.
19. `validate_output` - compare the materialized output against the input
    baseline.
20. `render` - render final output images when enabled.

See `references/pipeline-steps.md` for configuration details, outputs, and
step-specific caveats.

### Common Workflows

```bash
# Run the configured pipeline.
material-agent run apps/material_agent/configs/unified_example.yaml

# Run only prediction, application, and final render.
material-agent run apps/material_agent/configs/unified_example.yaml --only predict,apply,render

# Skip optimization when the USD is already prepared.
material-agent run apps/material_agent/configs/unified_example.yaml --skip optimize_usd

# Preview what the pipeline will do.
material-agent run apps/material_agent/configs/unified_example.yaml --dry-run

# Resume after a failed step.
material-agent run apps/material_agent/configs/unified_example.yaml --resume

# Create a config with a materials manifest and reference image.
material-agent configure my_pipeline.yaml -m materials/my_materials.yaml -r reference.jpg

# Benchmark a configured dataset.
material-agent benchmark configs/benchmark.yaml -d dataset.jsonl -o results/
```

### Generated Reference Images

Enable `render_preview` and `generate_reference_image` when the user has no
reference photos and wants a text-described target appearance. The generated
image is injected into the dataset for the prediction step.

```yaml
steps:
  render_preview:
    enabled: true
    cameras: ["+x+y+z"]
  generate_reference_image:
    enabled: true
    prompt: "aluminum frame with a blue plastic tray"
    num_images: 1
```

### SimReady Demo

When the user asks to "try material agent" or run a public demo asset, follow
`references/simready-quickstart.md`. It covers downloading curated SimReady
assets, writing a config that uses the shipped default material library, and
running the pipeline end to end. For UR10 assets, keep
`prim_filters.skip_instances: false` so the agent sees meshes.

### Config Authoring

Prefer copying `apps/material_agent/configs/unified_example.yaml` for new
configs. Adapt only the user-specific fields:

- `project.name` and `project.session_id`
- `input.usd_path`
- `input.reference_images` when reference photos are available
- `materials.path`
- `steps.predict.vlm.model`
- `steps.render.enabled`

Use `references/config-template.yaml` for a complete ready-to-adapt template
and `references/config-reference.md` for the full schema. Keep the prompts,
renderer settings, and prediction settings unless the user explicitly asks to
change them.

### Finding Materials

A materials file is a YAML manifest with a `library_path` pointing at the USD
material library and `entries` listing available materials:

```yaml
library_path: "path/to/material_libs.usd"
entries:
  - name: "Aluminum Polished"
    description: "A polished aluminum for structural parts"
    binding: "/World/metal_library/Looks/Aluminum_Polished"
```

Use the default library first, ask for a user-provided manifest second, and
create a new manifest only when the user has a material USD and wants that
library enumerated.

## Output Format

Report these items after a run or handoff:

- Command executed and whether it was full pipeline, `--only`, `--skip`,
  `--resume`, or `--dry-run`.
- Config path and session ID.
- Working directory, usually `.<session_id>/` next to the config unless
  `project.working_dir` overrides it.
- Key artifacts when present:
  - `validation/input/` for pre-run USD validation reports.
  - `dataset/usd/` for rendered prim images and manifests.
  - `dataset/dataset.jsonl` for VLM-ready records.
  - `clusters/` for clustering reports and representative mappings.
  - `generated_refs/` for generated reference images.
  - `predictions/predictions.jsonl` and `report.html` for VLM output.
  - `evaluation/` for LLM-judge scoring outputs.
  - `iterations/` for iterative refinement artifacts.
  - `restored/restored_predictions.jsonl` when restore/remap ran.
  - `output/output.usd` or the configured `output.usd_path`.
  - `validation/output/` for post-run USD validation reports.
  - `renders/` for final render images.
- Any missing credentials, service endpoints, invalid material bindings, or
  config-path issues.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| API key required | The selected VLM, LLM, or image-generation backend has no credential. | Set the required key locally or in `.env`; do not paste it into chat. |
| Pipeline fails midway | A step failed after writing partial artifacts. | Re-run with `--resume`; use `--clean` only when the user wants to discard prior artifacts. |
| Forbidden path key in step config | Step configs contain path keys that the executor owns. | Remove `usd_path`, `output_dir`, `dataset`, or `predictions_path` from step configs. |
| Relative paths resolve unexpectedly | Config paths resolve from the config file directory. | Rewrite paths relative to the config file or make them absolute. |
| No meshes found for a SimReady UR10 asset | Instance filtering hid the geometry. | Set `prim_filters.skip_instances: false`. |
| Remote rendering fails or stalls | `RENDER_ENDPOINT` is missing, unhealthy, or over-concurrent. | Check endpoint health and keep local OVRTX worker/request concurrency at 1. |
| Material names do not match the library | Predictions chose names outside the manifest. | Keep `validate_predictions` enabled and verify the manifest entries are descriptive. |
