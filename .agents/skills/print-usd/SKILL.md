---
name: print-usd
description: Inspect and print USD scene hierarchy using the wu CLI. Use when the user wants to examine a USD file structure, list prims, show prim types, variants, API schemas, collections, custom token attributes, query a specific prim, limit traversal depth, or get scene statistics for .usd/.usda/.usdc/.usdz files.
version: "0.1.0"
author: NVIDIA Content Agents
tags:
  - content-agents
  - usd
  - inspection
  - cli
tools:
  - Shell
  - Filesystem
  - wu
compatibility: Requires the wu CLI, readable USD input files, and USD Python bindings installed through the repository environment.
---

# Print USD

Inspect USD scene hierarchy and prim metadata with `wu print-usd`.

## When to Use

- Use when the user wants a quick tree view of a USD asset.
- Use when the user needs prim paths for rendering, material binding,
  filtering, debugging, or follow-up CLI commands.
- Use when the user asks for prim types, variants, API schemas, collections,
  custom tokens, active-only traversal, depth-limited output, or statistics.
- Use `flatten-usd` when the user wants to resolve composition arcs from a
  `.usd`, `.usda`, or `.usdc` stage, not merely inspect the composed stage.

## Limitations

- Very large scenes can produce thousands of lines. Prefer `--max-depth`,
  `--start-prim`, `--query-prim`, or `--stats` for focused output.
- `--query-prim` requires an exact prim path.
- Output is textual inspection only; it does not modify the USD file.

## Prerequisites

- Activate the repo Python environment and confirm `wu` is on `PATH`.
- Ensure the input `.usd`, `.usda`, `.usdc`, or `.usdz` file exists.
- USD Python bindings must be available in the environment.

## Instructions

1. Start with a shallow overview for unknown scenes.
2. Add `--stats` for counts or `--max-depth` for large assets.
3. Add `--show-types` before choosing render, material, or physics targets.
4. Add `--show-all` only when the user needs all metadata.
5. Use `--query-prim` for a specific prim path and report whether the prim was
   found.
6. Return useful prim paths and suggested next commands.

## Command Reference

```bash
wu print-usd <usd_path> [OPTIONS]
```

| Option | Description |
|---|---|
| `--start-prim`, `-p` | Start traversal from a prim path. |
| `--show-types`, `-t` | Show prim type names. |
| `--show-variants` | Show variant set selections. |
| `--show-api-schemas` | Show applied API schemas. |
| `--show-collections` | Show collections and their includes. |
| `--show-custom-tokens` | Show custom token attributes. |
| `--show-all`, `-a` | Enable all metadata display flags. |
| `--active-only` | Show only active prims. |
| `--max-depth`, `-d` | Limit traversal depth. |
| `--no-info` | Hide the stage information header. |
| `--stats`, `-s` | Show scene statistics. |
| `--query-prim`, `-q` | Show detailed information for one prim. |
| `--verbose`, `-v` | Enable debug logging. |

## Common Workflows

```bash
# Basic tree and stage header.
wu print-usd scene.usd

# Types and variants for the top three levels.
wu print-usd scene.usd --show-types --show-variants --max-depth 3

# Metadata-rich inspection.
wu print-usd scene.usd --show-all --active-only

# Focus on a subtree.
wu print-usd scene.usd --start-prim /World/Geometry --show-types

# Query one prim.
wu print-usd scene.usd --query-prim /World/Geometry/mesh1

# Summary counts.
wu print-usd scene.usd --stats
```

## Output Format

Report:

- Command executed and input path.
- Whether output was full tree, subtree, statistics, or a prim query.
- Important prim paths, types, active/instance state, collection membership, or
  counts relevant to the user's goal.
- Suggested follow-up command when useful, such as `render-usd --focus` or
  `flatten-usd` for `.usd`, `.usda`, or `.usdc` sources.
- Any error cause, including missing file or missing prim path.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| USD file not found | Path is wrong or relative to a different directory. | Use an absolute path or rerun from the expected working directory. |
| Prim not found | `--query-prim` path is not exact. | Run a shallow `--show-types --max-depth` listing and copy the exact path. |
| Output is too large | Scene has many prims. | Use `--max-depth`, `--start-prim`, `--query-prim`, or `--stats`. |
| USD loading error | Missing USD bindings or unresolved asset issue. | Activate the repo environment and inspect the asset dependencies. |
