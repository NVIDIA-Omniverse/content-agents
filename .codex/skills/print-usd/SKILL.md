---
name: print-usd
description: Inspect and print USD scene hierarchy using the wu CLI. Use when user wants to examine a USD file's structure, list prims, show prim types, query a specific prim, or get scene statistics. Relevant for .usd, .usda, .usdc, .usdz files.
---

# Print USD Scene Hierarchy

Prints a USD file's scene hierarchy as a tree with optional metadata about prims, types, variants, collections, and transforms using the `wu print-usd` CLI command.

## Prerequisites

- The `wu` CLI must be installed and available on PATH
- A valid USD file (.usd, .usda, .usdc, .usdz)

## Command Reference

```bash
wu print-usd <usd_path> [OPTIONS]
```

### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `usd_path` | Yes | Path to the USD file to analyze |

### Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--start-prim` | `-p` | None | Start traversal from specific prim path (e.g., `/World/Geo`) |
| `--show-types` | `-t` | False | Show prim types in brackets |
| `--show-variants` | | False | Show variant set selections |
| `--show-api-schemas` | | False | Show applied API schemas |
| `--show-collections` | | False | Show collections defined on each prim |
| `--show-custom-tokens` | | False | Show custom token attributes |
| `--show-all` | `-a` | False | Enable all `--show-xxx` options at once |
| `--active-only` | | False | Show only active prims |
| `--max-depth` | `-d` | None | Maximum depth to traverse |
| `--no-info` | | False | Don't show stage information header |
| `--stats` | `-s` | False | Show statistics about the scene |
| `--query-prim` | `-q` | None | Query a specific prim for detailed info |
| `--verbose` | `-v` | False | Enable verbose logging |

## Common Workflows

### Step 1: Get a quick overview of the scene

```bash
wu print-usd scene.usd
```

This prints the basic tree structure with stage info header.

### Step 2: Show all metadata

```bash
wu print-usd scene.usd --show-all
```

Enables all `--show-xxx` flags: types, variants, API schemas, collections, and custom tokens.

### Step 3: Limit depth for large scenes

```bash
wu print-usd scene.usd --show-all --max-depth 3
```

### Step 4: Focus on a subtree

```bash
wu print-usd scene.usd --start-prim /World/Geometry
```

### Step 5: Query a specific prim

```bash
wu print-usd scene.usd --query-prim /World/Geometry/mesh1
```

This shows: prim name, type, active status, parent, children count, collections membership, and Xform ownership.

### Step 6: Get scene statistics

```bash
wu print-usd scene.usd --stats
```

Shows prim count, max depth, type distribution, etc.

## Examples

### Basic tree
```bash
wu print-usd model.usda
```

### Types + variants for top 3 levels
```bash
wu print-usd model.usda --show-types --show-variants --max-depth 3
```

### Only active prims with all metadata
```bash
wu print-usd scene.usd --show-all --active-only
```

### Inspect a specific prim
```bash
wu print-usd scene.usd -q /World/Looks/Material_0
```

## Common Issues

### Error: USD file not found
Cause: The path is invalid or the file doesn't exist.
Solution: Verify the file path. Use an absolute path or path relative to the current working directory.

### Very large output
Cause: The USD scene has thousands of prims.
Solution: Use `--max-depth` to limit traversal depth, or `--start-prim` to focus on a subtree. Use `--stats` for a summary instead of the full tree.
