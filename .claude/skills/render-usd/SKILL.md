---
name: render-usd
description: Render USD files using the wu CLI with a configured remote rendering service. Use when user wants to render a USD scene, generate images from USD files, or produce renders from .usd/.usda/.usdc/.usdz files. Supports single and multi-camera renders, sensor outputs (depth, segmentation), and frame ranges.
---

# Render USD Files

Renders USD files using the `wu render-usd` CLI command and a configured remote rendering service.

## Prerequisites

- The `wu` CLI must be installed and available on PATH
- Environment variables from `.env` must include `RENDER_ENDPOINT` with the
  full URL of the remote render service and any auth required by that service
- AWS credentials configured for S3 upload, if the remote service requires S3
  asset transfer

## Command Reference

```bash
wu render-usd <usd_path> [OPTIONS]
```

### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `usd_path` | Yes | Path to USD file (.usd, .usda, .usdc, .usdz) |

### Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--output` | `-o` | None | Output path for single frame/camera |
| `--output-dir` | | None | Output directory for multi-camera/multi-frame |
| `--width` | `-w` | 1920 | Image width in pixels |
| `--height` | | None | Image height in pixels (defaults to width) |
| `--camera` | `-c` | `Camera` | Camera name or path to render |
| `--frames` | `-f` | `0` | Frames to render (e.g., `0`, `0:10`) |
| `--backend` | `-b` | `remote` | Rendering backend (only `remote` supported by this CLI) |
| `--sensors` | | None | Comma-separated sensors (remote backend only): `linear_depth,depth,instance_id_segmentation` |
| `--all-cameras` | | False | Render all cameras (requires `--output-dir`) |
| `--save-camera-json` | | False | Save camera parameters to JSON |
| `--verbose` | `-v` | False | Enable verbose logging |

### Output Rules

- **Single frame + single camera**: Use `--output` OR `--output-dir` (not both)
- **Multiple frames or cameras**: Use `--output-dir` only

## How It Works

1. Opens the USD stage
2. **Flattens** the stage to inline all payloads, references, and sublayers for upload
3. If the specified camera doesn't exist, **auto-creates a corner view camera** looking at the scene
4. Uploads the flattened stage to S3 (bundled with textures if found)
5. Calls the configured render service
6. Saves output images and cleans up S3

## Common Workflows

### Render a single image

```bash
wu render-usd scene.usd --output render.png
```

### Render with custom resolution

```bash
wu render-usd scene.usd --output render.png --width 1024 --height 1024
```

### Render a USDZ file (textures are auto-handled)

```bash
wu render-usd model.usdz --output render.png --width 512
```

### Render with depth sensor output

```bash
wu render-usd scene.usd --output render.png --sensors linear_depth
```

### Render multiple frames

```bash
wu render-usd scene.usd --frames 0:10 --output-dir renders/
```

### Render all cameras in the scene

```bash
wu render-usd scene.usd --all-cameras --output-dir renders/
```

## Supported File Formats

- `.usd` - Universal Scene Description (auto-detected binary or ASCII)
- `.usda` - ASCII USD (may reference external payloads - auto-flattened)
- `.usdc` - Binary/Crate USD
- `.usdz` - Packaged USD archive with embedded textures (auto-flattened)

All formats are flattened before upload so that external references, payloads, and embedded assets are inlined into a self-contained file for the remote renderer.

## Common Issues

### HTTP 400 from the render service
Cause: Usually means the camera path doesn't exist in the uploaded scene.
Solution: The CLI auto-creates a camera if missing. If you specify `--camera /MyCamera`, make sure that prim exists. Use `wu print-usd scene.usd --show-types` to verify.

### Black or empty render
Cause: Scene has no lights, or materials use custom MDL shaders the renderer can't resolve.
Solution: This is expected for assets without lighting. The geometry is still rendered correctly.

### Timeout / 504 errors
Cause: Large scenes or remote service cold start.
Solution: The CLI retries automatically (up to 3 times with backoff). For very large scenes, try reducing `--width`.
