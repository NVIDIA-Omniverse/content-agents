# Custom Materials Examples

This directory contains examples and instructions for creating custom materials
ZIP files to use with the Material Agent Service.

## Overview

By default, the service uses its built-in materials library. You can override
this by uploading a custom materials ZIP file when running the pipeline.

## ZIP File Requirements

Your materials ZIP must contain:

| File | Required | Description |
|------|----------|-------------|
| `materials.yaml` | ✅ Yes | Material definitions (names, descriptions, bindings) |
| `materials_libs.usda` | ✅ Yes | USD file containing material definitions (.usd/.usda/.usdc) |
| `thumbs/` | ❌ No | Optional thumbnail images for UI preview |

### Size Limits

- Maximum ZIP file size: **500MB** (configurable via `MA_MAX_UPLOAD_SIZE_MB` environment variable)
- This is the same limit applied to USD file uploads

### Directory Structure

The expected structure is a directory containing your materials files:

```
custom_materials.zip
└── custom_materials/           # Your materials directory
    ├── materials.yaml          # Required: Material catalog
    ├── materials_libs.usda     # Required: USD material definitions
    └── thumbs/                  # Optional: Preview thumbnails
        └── 256x256/
            └── My_Material.png
```

This is created by zipping a directory: `zip -r custom_materials.zip custom_materials/`

## materials.yaml Format

**Important**: The YAML file **must** be a dictionary with a top-level `materials` key that contains another dictionary. This structure is validated for security.

```yaml
materials:
  # Path to USD library (relative to materials.yaml location)
  # Security: This path cannot escape the materials directory
  library_path: "materials_libs.usda"

  # List of available materials (at least one required)
  entries:
    - name: "My Custom Metal"
      description: "Brushed aluminum with subtle reflections"
      binding: "/World/Looks/My_Custom_Metal"   # USD prim path
      icon: "thumbs/256x256/My_Custom_Metal.png"  # Optional

    - name: "My Custom Plastic"
      description: "Glossy red plastic surface"
      binding: "/World/Looks/My_Custom_Plastic"
      # icon field is optional - omit if no thumbnails
```

### Field Descriptions

| Field | Required | Description |
|-------|----------|-------------|
| `library_path` | ✅ Yes | Relative path to USD library file. Cannot use `..` or absolute paths for security |
| `entries` | ✅ Yes | List of materials (at least one required) |
| `name` | ✅ Yes | Human-readable material name (used by VLM for selection) |
| `description` | ✅ Yes | Detailed description to help VLM choose appropriate materials |
| `binding` | ✅ Yes | USD prim path to the material in the USD library specified by `materials.library_path` |
| `icon` | ❌ No | Optional path to thumbnail image (for UI preview only) |

## materials_libs.usda Format

The USD file must define materials at the paths specified in `binding` fields.
You can use `.usd`, `.usda`, or `.usdc` extensions. Example structure:

```usda
#usda 1.0

def "World"
{
    def "Looks"
    {
        def Material "My_Custom_Metal"
        {
            # Material definition (MDL, UsdPreviewSurface, etc.)
        }
        
        def Material "My_Custom_Plastic"
        {
            # Material definition
        }
    }
}
```

## Creating the ZIP

```bash
cd apps/material_agent_service/examples
zip -r my_materials.zip custom_materials/
```

## Usage

### curl

```bash
curl -X POST http://localhost:8000/pipeline \
  -F "usd_file=@/path/to/scene.usd" \
  -F "materials_zip=@/path/to/my_materials.zip"
```

### CLI Client

```bash
python apps/material_agent_service/client/client.py \
  --materials-zip /path/to/my_materials.zip \
  /path/to/scene.usd
```

### Python API

```python
from apps.material_agent_service.client.client import MaterialAgentClient

client = MaterialAgentClient(base_url="http://localhost:8000")
session_id, results = client.run_and_monitor(
    usd_path="/path/to/scene.usd",
    materials_zip_path="/path/to/my_materials.zip",
)
```

## Validation and Error Handling

The service performs several security and validation checks:

1. **YAML Structure**: Must be a valid YAML dictionary with a `materials` dictionary
2. **Required Fields**: `library_path` and `entries` must be present
3. **Path Security**: `library_path` cannot reference files outside the extraction directory
4. **File Existence**: The USD library file must exist in the ZIP
5. **Size Limits**: ZIP file must not exceed the upload size limit

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| "materials.yaml must be a YAML dictionary" | Invalid YAML format | Ensure the file is valid YAML starting with `materials:` |
| "materials.yaml must have a 'materials' dictionary at top level" | Missing `materials:` key | Add the required top-level `materials:` key |
| "materials.library_path is required" | Missing `library_path` | Add `library_path: "materials_libs.usda"` under `materials:` |
| "must contain a non-empty list in materials.entries" | Wrong type or empty | `entries:` must be a YAML list with at least one item |
| "materials.entries must be a list of objects" | List contains non-dict items | Each entry must be a YAML object/mapping with fields |
| "library_path escapes base directory" | Path traversal attempt | Use only relative paths without `..` |
| "USD library file not found" | Missing USD file | Ensure the file name matches `library_path` exactly |
| "Materials ZIP too large" | Exceeds size limit | Reduce ZIP size or ask admin to increase limit |

## Tips

1. **Good Descriptions**: Write detailed, distinctive descriptions for each
   material. The VLM uses these to match materials to object parts.

2. **Unique Names**: Use clear, unique names that describe the material's
   appearance (e.g., "Brushed Stainless Steel" vs just "Metal").

3. **Binding Paths**: Ensure binding paths exactly match the prim paths in
   your USD library file.

4. **Testing**: Test your custom materials with a simple scene first to
   verify bindings work correctly.

5. **Security**: Do not use absolute paths or `..` in `library_path` - they
   will be rejected for security reasons.

## Example Files

See `custom_materials/` directory for a working example:
- `materials.yaml` - Example configuration with 3 materials
- `materials_libs.usda` - Working USD file with UsdPreviewSurface materials


