---
name: flatten-usd
description: Flatten a composed USD stage into a single self-contained file. Use when user wants to flatten a USD file, resolve sublayers/references/payloads, create a self-contained USD, or merge USD layers. Trigger phrases include "flatten usd", "flatten the scene", "make a flat usd", "resolve usd layers", "self-contained usd".
---

# Flatten USD

Flatten a composed USD stage (with sublayers, references, payloads, inherits) into a single self-contained file using the `wu flatten-usd` CLI command.

## Prerequisites

Activate the virtual environment first:
```bash
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

## Usage

```bash
# Flatten to default name (<source>_flat.<ext>)
wu flatten-usd <source.usd>

# Flatten to specific output
wu flatten-usd <source.usd> <output.usd>

# Force overwrite existing output
wu flatten-usd <source.usd> <output.usd> --force

# Verbose mode (shows sublayer count, prim count)
wu flatten-usd <source.usd> -v
```

## Options

| Flag | Short | Description |
|---|---|---|
| `--force` | `-f` | Overwrite destination file if it exists |
| `--verbose` | `-v` | Show additional stage info (sublayer count, prim count) |

## Notes

- Output format is determined by file extension: `.usd` (binary), `.usda` (ASCII), `.usdc` (crate binary)
- If no destination is given, outputs to `<stem>_flat.<ext>` in the same directory
- Flattening resolves ALL composition arcs — the output has no external dependencies
- Large scenes (100K+ prims) may produce files of 100+ MB and take a minute or more
- Harmless `_CopyMetadata` warnings about `hide_in_stage_window` / `no_delete` may appear for Kit-authored scenes — these are non-standard schema fields that don't transfer during flatten
