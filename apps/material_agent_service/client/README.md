### Material Agent Service Python Client (CLI + API)

Minimal Python client to start the Material Agent pipeline and monitor progress via SSE (with polling fallback).

#### Requirements
- Python 3.12+
- requests

Install:
```bash
uv pip install requests   # preferred if uv is available
# or
pip install requests
```

#### CLI Usage
From repo root:
```bash
python apps/material_agent_service/client/client.py \
  --base-url http://localhost:8000 \
  --email user@example.com \
  --upload-first \
  --prompt "Metal frames should be aluminum" \
  --generate-ref-prompt "Brushed aluminum frame with matte black plastic steps" \
  --ref /path/to/ref1.png --ref /path/to/ref2.jpg \
  --ref-desc "Top view" --ref-desc "Side detail" \
  /path/to/scene.usd
```

Auth (Bearer token):
- Flag: `--token "$YOUR_TOKEN"`
- Or env: `export MATERIAL_AGENT_TOKEN="$YOUR_TOKEN"`

Examples:
```bash
# Simple
python apps/material_agent_service/client/client.py \
  --email user@example.com \
  /path/to/scene.usd

# With token and prompt
python apps/material_agent_service/client/client.py \
  --base-url http://localhost:8000 \
  --token "$TOKEN" \
  --email user@example.com \
  --prompt "Prefer matte plastics" \
  /path/to/scene.usd

# With custom materials (overrides server defaults)
python apps/material_agent_service/client/client.py \
  --email user@example.com \
  --materials-zip /path/to/custom_materials.zip \
  /path/to/scene.usd

# Generate an AI reference image before running the pipeline
python apps/material_agent_service/client/client.py \
  --email user@example.com \
  --upload-first \
  --generate-ref-prompt "Satin red painted metal body with black rubber wheels" \
  /path/to/scene.usd
```

Custom Materials ZIP:
- Use `--materials-zip` to provide a ZIP file with custom materials
- ZIP must contain: `materials.yaml` (service format) + USD library file
- Icons in `thumbs/` are optional (for UI previews only)
- Overrides server default materials for this pipeline run only

Exit behavior:
- Streams live progress (SSE) and prints updates like: `[render] running overall=87%`.
- Falls back to status polling if SSE is unavailable.
- Prints artifact URLs on completion.

#### Programmatic Use
```python
from apps.material_agent_service.client.client import MaterialAgentClient

client = MaterialAgentClient(base_url="http://localhost:8000", token="YOUR_TOKEN")
session_id, results = client.run_and_monitor(
    usd_path="/path/to/scene.usd",
    reference_images=["/path/to/ref.png"],
    reference_descriptions=["Front view"],
    user_prompt="Use stainless steel for rollers",
    generated_reference_prompt=(
        "Satin red painted metal body with black rubber wheels"
    ),
    camera_views="+x+y+z,-x-y-z",
    upload_first=True,
    materials_zip_path="/path/to/custom_materials.zip",  # Optional
)
print(session_id, results)
```

Key endpoints the client uses:
- POST `/pipeline` (start)
- POST `/pipeline/upload-usd` (optional pre-upload)
- POST `/pipeline/{session_id}/generate-reference-image` (returns an explicit `reference_id`)
- GET `/assets/{session_id}/generated-ref/{reference_id}` (generated reference image)
- GET `/pipeline/{session_id}/events` (SSE)
- GET `/pipeline/{session_id}/status` (polling)
- GET `/pipeline/{session_id}/results` (final)
- POST `/pipeline/{session_id}/cancel` (cancel)
