# Pipeline Steps Reference

The material agent pipeline runs steps in a fixed order. Each step is optional and can be controlled via `enabled: true/false` in the config, or via `--skip`/`--only` CLI flags.

## Step Execution Order

```
optimize_usd -> render_preview -> generate_reference_image -> build_dataset_usd -> build_dataset_pdf_vectorstore -> build_dataset_prepare_dataset -> predict -> apply -> render
```

### Mutually Exclusive Steps

- **predict** and **benchmark** cannot both run
- **apply** and **assign** cannot both run (assign includes apply internally)

## Step Details

### 1. optimize_usd

Flattens and deinstances USD files via the configured scene optimizer service. This ensures prims are individually addressable for rendering and material assignment.

**Default config:**
```yaml
steps:
  optimize_usd:
    enabled: true
```

**Output:** `{working_dir}/optimized/` -- optimized USD file

### 2. render_preview (optional)

Renders lightweight whole-scene preview images. These previews are used as conditioning input for the `generate_reference_image` step. This step is only needed when using AI-generated reference images instead of user-supplied photos.

**Key config options:**
```yaml
steps:
  render_preview:
    enabled: true
    cameras: ["+x+y+z"]             # Camera angles for preview
    should_reset_materials: true     # Strip materials for a clean preview
    use_lights: false                # Disable scene lights
    image_width: 512
    image_height: 512
    background_color: [1.0, 1.0, 1.0]
    camera_margin: 1.0
```

**Output:** `{working_dir}/preview/` -- preview images, auto-wired to `generate_reference_image`

### 3. generate_reference_image (optional)

Generates photorealistic reference images from the scene previews and a user text prompt using an image generation model. The generated images are automatically injected into `build_dataset_prepare_dataset` as reference images for VLM context.

**Requires:** `render_preview` step to run first (provides preview images as conditioning input).

**Key config options:**
```yaml
steps:
  generate_reference_image:
    enabled: true
    image_gen:
      backend: gemini                                  # Public image-gen backend (Google)
      model: gemini-3-pro-image-preview                # Text → pixels; needs GOOGLE_API_KEY
    prompt: "Describe desired materials, e.g. aluminium frame with blue plastic tray"
    num_images: 1                    # Number of reference images to generate
    reference_images: []             # Optional existing images to condition on
```

**Output:** `{working_dir}/generated_refs/` -- generated reference images, auto-injected into dataset preparation

### 5. build_dataset_usd

Renders multi-view images of each prim in the USD scene. These images are used as VLM input for material prediction.

**Key config options:**
```yaml
steps:
  build_dataset_usd:
    renderer:
      backend: remote              # remote, ovrtx, or warp
      image_width: 512
      image_height: 512
      should_assign_random_colors: true
      rendering_modes:
        prim_only:                # Isolated prim view
          margin: 1.2
          cameras: ["+x+y+z", "-x-y-z"]
          camera_focus_mode: prim
        composition:              # Prim in context of full scene
          margin: 6.0
          cameras: ["+x", "+y", "+z"]
          camera_focus_mode: stage
    prim_filters:
      types: ["UsdGeom.Mesh", "UsdGeom.Cube", "UsdGeom.Cylinder",
              "UsdGeom.Capsule", "UsdGeom.Sphere", "UsdGeom.Cone"]
      skip_instances: true
    extract_material_bindings: true
    extract_hierarchy: true
    batch_size: 16
```

**Output:** `{working_dir}/dataset/usd/` -- rendered images and dataset manifest

### 6. build_dataset_pdf_vectorstore

Builds a searchable vector store from PDF documents (e.g., technical specifications). Used for RAG-enhanced material prediction.

**Key config options:**
```yaml
steps:
  build_dataset_pdf_vectorstore:
    enabled: false  # Disabled by default
```

Requires `input.reference_pdfs` to be set in the config. Source PDFs are processed into text/image chunks and indexed.

**Output:** `{working_dir}/vectorstore/` -- FAISS vector store

### 7. build_dataset_prepare_dataset

Prepares the final dataset for VLM inference by combining rendered images, material lists, prompts, and optionally RAG context.

**Key config options:**
```yaml
steps:
  build_dataset_prepare_dataset:
    enabled: true
    include_ground_truth: false      # true for benchmarking
    include_prim_path_context: true
    prompts:
      vlm_system: "System prompt with {materials_list} placeholder"
      vlm_user: "User prompt with {context} placeholder"
```

**Output:** `{working_dir}/dataset/dataset.jsonl` -- VLM-ready dataset

### 8. predict

Runs VLM inference on the prepared dataset to predict material assignments for each prim.

**Key config options:**
```yaml
steps:
  predict:
    vlm:
      backend: nim                   # nim (default), openai, anthropic, gemini
      model: qwen/qwen3.5-397b-a17b
      temperature: 1.0
      max_completion_tokens: 24576
    max_workers: 16                 # Parallel inference workers
    llm:                            # Optional LLM for post-processing
      backend: nim
      model: qwen/qwen3.5-397b-a17b
    report:
      image_max_size: 256
```

**Output:** `{working_dir}/predictions/predictions.jsonl` and `report.html`

### 9. apply

Applies predicted materials to the USD file by creating material bindings for each prim.

**Key config options:**
```yaml
steps:
  apply:
    layer_only: false     # true to create a separate layer file
    flatten_output: false # true to flatten output USD
```

**Output:** `{output.usd_path}` or `{working_dir}/output/output.usd`

### 10. render

Renders the final USD file with applied materials to produce output images.

**Key config options:**
```yaml
steps:
  render:
    enabled: true
    backend: remote
    image_width: 1024
    image_height: 1024
    camera_corners: ["+x+y+z", "-x-y-z"]
    camera_margin: 1.0
    background_color: [1.0, 1.0, 1.0]
```

**Output:** `{working_dir}/renders/` -- rendered images

## Special Steps

### benchmark (alternative to predict)

Runs VLM predictions AND evaluates them with an LLM judge. Produces FCS (Functional Correctness Score) metrics.

### assign (alternative to predict + apply)

Runs an iterative loop: predict -> apply -> render -> judge. Repeats until the judge approves or max iterations is reached.

```yaml
steps:
  assign:
    iteration:
      max_iterations: 5
      save_intermediate: true
    predict:
      vlm: { ... }
    apply: { ... }
    render: { ... }
    judge:
      reference_images: [...]
      vlm: { ... }
```

### restore_usd

Remaps predictions from optimized USD prim paths back to original USD prim paths. Required when applying materials to the original (pre-optimization) USD. Uses the `correspondence_map` from `optimize_usd` metadata to handle all 8 combinations of the optimizer's 3 operations (deinstance, split, deduplicate). Outputs `restored_predictions.jsonl` with original prim paths (including GeomSubset children for split meshes). Reports detailed stats: identity/dedup/split counts, consumed predictions, and uncovered originals.

## Auto-Derived Directory Structure

All paths are automatically created under the working directory:

```
.{session_id}/
  optimized/          # optimize_usd output
  preview/            # render_preview output
  generated_refs/     # generate_reference_image output
  dataset/
    usd/              # build_dataset_usd output
    dataset.jsonl     # prepare_dataset output
  vectorstore/        # pdf_vectorstore output
  predictions/        # predict/benchmark output
  iterations/         # assign output
  restored/           # restore_usd output
  renders/            # render output
  output/
    output.usd        # final output USD
```
