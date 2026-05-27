---
name: render-usd
description: Render USD files using the wu CLI with a configured remote rendering service. Use when the user wants to render a USD scene, generate images from .usd/.usda/.usdc/.usdz files, render all cameras or frame ranges, produce depth or segmentation sensors, focus or isolate prims, or test a render endpoint.
version: "0.1.0"
author: NVIDIA Content Agents
tags:
  - content-agents
  - usd
  - rendering
  - cli
  - ovrtx
tools:
  - Shell
  - Filesystem
  - wu
compatibility: Requires the wu CLI, readable USD input files, a configured RENDER_ENDPOINT for the remote backend, and any service authentication or asset-transfer credentials required by the render deployment.
---

# Render USD

Render USD assets through `wu render-usd` and a configured remote render
service.

## When to Use

- Use when the user wants rendered images from `.usd`, `.usda`, `.usdc`, or
  `.usdz` files.
- Use when the user asks for depth, instance segmentation, all-camera renders,
  frame ranges, focused prim renders, isolated prim renders, or camera JSON.
- Use `print-usd` first when the user needs to discover cameras or prim paths.
- Use `deploy-ovrtx-docker` first when no render endpoint is running.

## Limitations

- The CLI backend is `remote`; local renderer selection is handled by the
  service behind `RENDER_ENDPOINT`.
- Single-frame single-camera runs must use exactly one of `--output` or
  `--output-dir`.
- Multi-frame or all-camera runs require `--output-dir` and cannot use
  `--output`.
- Large scenes can take time to flatten, upload, render, and download.
- Missing lights or unsupported material shaders can produce dark output even
  when geometry renders correctly.

## Prerequisites

- Activate the repo Python environment and confirm `wu` is on `PATH`.
- Set `RENDER_ENDPOINT` to the render service URL.
- Configure any endpoint authentication or asset-transfer credentials required
  by the deployment.
- Ensure the USD input and referenced assets are readable.

## Instructions

1. Inspect the input path and choose single image, multi-frame, all-camera, or
   focused/isolation mode.
2. Use `wu print-usd <file> --show-types --max-depth 3` when camera or prim
   paths are unknown.
3. Choose output flags according to the output rules.
4. Add `--focus` to auto-frame a prim, `--isolate` to hide everything except
   listed prims, or both for an object-only render.
5. Add sensor outputs only when the render service supports them.
6. Report output files, camera JSON, and any render warnings.

## Command Reference

```bash
wu render-usd <usd_path> [OPTIONS]
```

| Option | Description |
|---|---|
| `--output`, `-o` | Output path for a single frame and camera. |
| `--output-dir` | Directory for multi-camera, multi-frame, or directory-based single renders. |
| `--width`, `-w` | Image width. Default is `1920`. |
| `--height` | Image height. Defaults to width. |
| `--camera`, `-c` | Camera name or prim path. Default is `Camera`. |
| `--frames`, `-f` | Frame selector such as `0`, `0:10`, or comma-separated values. |
| `--backend`, `-b` | Rendering backend. Use `remote`. |
| `--sensors` | Comma-separated sensors such as `linear_depth`, `depth`, or `instance_id_segmentation`. |
| `--all-cameras` | Render every camera. Requires `--output-dir`. |
| `--save-camera-json` | Save camera parameters next to rendered images. |
| `--focus` | Prim path to auto-frame with the camera. |
| `--isolate` | Comma-separated prim paths to render while hiding other geometry. |
| `--hide` | Comma-separated prim paths or subtrees to hide before rendering. |
| `--direction` | Camera direction such as `+x+y+z` or `+x-0.5y+z`. |
| `--margin` | Camera distance margin multiplier. |
| `--focal-length` | Camera focal length in millimeters. |
| `--aperture` | Horizontal aperture in millimeters. |
| `--cam-x`, `--cam-y`, `--cam-z` | Override camera position. |
| `--target-x`, `--target-y`, `--target-z` | Override look-at target. |
| `--near-clip` | Override camera near clipping plane distance. |
| `--far-clip` | Override camera far clipping plane distance. |
| `--dome-light` | Replace scene lights with a dome light intensity. |
| `--distant-light` | Replace scene lights with a distant light intensity. |
| `--verbose`, `-v` | Enable debug logging. |

## Common Workflows

```bash
# Single image.
wu render-usd scene.usd --output render.png

# Square thumbnail.
wu render-usd scene.usd --output render.png --width 512 --height 512

# All cameras.
wu render-usd scene.usd --all-cameras --output-dir renders/

# Frame range.
wu render-usd scene.usd --frames 0:10 --output-dir frames/

# Depth sensor.
wu render-usd scene.usd --output render.png --sensors linear_depth

# Focus and isolate one object.
wu render-usd scene.usd --focus /World/Chair --isolate /World/Chair \
  --output chair.png

# Add simple lighting when a scene is dark.
wu render-usd scene.usd --output lit.png --dome-light 1500
```

## Output Format

Report:

- Command executed and render endpoint used, without printing credentials.
- Input USD path, camera selection, frame selection, and output path or
  directory.
- Any focus, isolate, light, sensor, or camera override options.
- Rendered image files, sensor files, and camera JSON files when present.
- Any failure cause, such as invalid output flag combination, missing USD file,
  missing prim/camera, endpoint error, timeout, or dark render caveat.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Missing `RENDER_ENDPOINT` | Remote backend has no service URL. | Start or configure a render service, then export `RENDER_ENDPOINT`. |
| Multi-output flag error | Multi-frame or all-camera run used `--output`. | Use `--output-dir` for multi-output runs. |
| Single-output flag error | Single-frame, single-camera run used both `--output` and `--output-dir`, or neither. | Use exactly one output flag for single-output runs. |
| Camera path error | The named camera is absent or misspelled. | Inspect with `wu print-usd scene.usd --show-types` or omit `--camera` for auto-created view. |
| Dark or black image | Scene lacks lights or uses unsupported shaders. | Try `--dome-light 1500` or `--distant-light 800`; inspect the USD materials separately. |
| Timeout or 504 | Large scene, cold render service, or slow asset transfer. | Reduce resolution, render a focused prim, or retry after endpoint warm-up. |
