---
name: flatten-usd
description: Flatten a composed USD stage into a single self-contained USD layer using the wu CLI. Use when the user wants to flatten a USD file, resolve sublayers, references, payloads, and inherits, merge USD composition into one file, or prepare a scene for sharing or rendering.
version: "0.1.0"
author: NVIDIA Content Agents
tags:
  - content-agents
  - usd
  - flatten
  - cli
tools:
  - Shell
  - Filesystem
  - wu
compatibility: Requires the wu CLI, USD Python bindings, readable .usd/.usda/.usdc input, and write permission for a .usd/.usda/.usdc destination file.
---

# Flatten USD

Flatten a composed USD stage with `wu flatten-usd`.

## When to Use

- Use when the user wants a composed USD layer written as one self-contained
  `.usd`, `.usda`, or `.usdc` file.
- Use when sublayers, references, payloads, or inherits need to be resolved for
  sharing, debugging, or rendering.
- Use before sending a composed stage to tooling that cannot resolve the
  original dependency chain.
- Use `render-usd` when the user wants images, and `print-usd` when they only
  need inspection.

## Limitations

- The CLI accepts `.usd`, `.usda`, and `.usdc` inputs and outputs. It does not
  accept `.usdz` sources or write `.usdz`.
- Flattening changes composition structure. It is useful for handoff, but not
  a substitute for preserving an editable layered asset.
- Large composed stages can produce large outputs and may take a minute or
  more.
- Existing destination files are not overwritten unless `--force` is set.

## Prerequisites

- Activate the repo Python environment and confirm `wu` is on `PATH`.
- Ensure USD Python bindings are installed.
- Confirm the source file exists and the destination directory is writable.

## Instructions

1. Confirm the source extension is `.usd`, `.usda`, or `.usdc`.
2. Choose a destination path, or omit it to use `<source>_flat.<ext>`.
3. If the destination exists, ask before using `--force`.
4. Run with `--verbose` when the user needs stage details.
5. Verify the output exists and report its path.

## Command Reference

```bash
wu flatten-usd <source.usd> [destination.usd] [OPTIONS]
```

| Option | Description |
|---|---|
| `--force`, `-f` | Overwrite destination if it already exists. |
| `--verbose`, `-v` | Print additional stage and export information. |

## Common Workflows

```bash
# Flatten to <stem>_flat.<ext>.
wu flatten-usd scene.usd

# Flatten to a specific ASCII USD output.
wu flatten-usd scene.usd scene_flat.usda

# Overwrite an existing destination after the user confirms.
wu flatten-usd scene.usd scene_flat.usd --force

# Show additional details.
wu flatten-usd scene.usd --verbose
```

## Output Format

Report:

- Source path and destination path.
- Whether the destination was defaulted or explicit.
- Whether `--force` was used.
- Output file size when available.
- Any warning that did not block export, such as non-standard Kit metadata
  that cannot transfer during flatten.
- Any failure cause, including missing source, unsupported extension, existing
  destination, or export failure.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Unsupported extension | Source or destination is not `.usd`, `.usda`, or `.usdc`. | Choose a supported USD layer extension. |
| Destination already exists | The command refuses to overwrite by default. | Ask the user before rerunning with `--force`. |
| USD bindings missing | The active environment lacks `pxr`. | Activate the repo environment and install the USD extra if needed. |
| Large output | Flattening inlined the composed dependency chain. | Keep the layered source asset for editing and use the flat file for handoff. |
| Metadata warnings | Kit-authored custom metadata may not copy to the flattened layer. | Treat harmless `_CopyMetadata` warnings as informational unless export fails. |
