### Physics Agent Service Python Client (CLI + API)

Minimal Python client to start the Physics Agent pipeline and monitor progress via SSE (with polling fallback).

Supports two input modes:
- **File upload** — upload a local USD file over HTTP
- **S3 reference** — pass an S3 URI and the service downloads it server-side (better for large files)

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
# Local file upload
python apps/physics_agent_service/client/client.py /path/to/scene.usdz

# S3 URI (service downloads server-side — no local file needed)
python apps/physics_agent_service/client/client.py \
  --s3-uri s3://your-bucket/path/to/scene.usdz
```

Auth (Bearer token):
- Flag: `--token "$YOUR_TOKEN"`
- Or env: `export PHYSICS_AGENT_TOKEN="$YOUR_TOKEN"`

Examples:
```bash
# Simple — local USD file
python apps/physics_agent_service/client/client.py /path/to/scene.usdz

# S3 URI — large asset, no upload needed
python apps/physics_agent_service/client/client.py \
  --s3-uri s3://your-bucket/path/to/large_scene.usdz

# With user prompt
python apps/physics_agent_service/client/client.py \
  --prompt "Identify electronic components" \
  /path/to/scene.usdz

# Choose rendering backend (remote, warp, or ovrtx)
python apps/physics_agent_service/client/client.py \
  --render-backend remote \
  /path/to/scene.usdz

# Upload USD first, then start pipeline (two-step)
python apps/physics_agent_service/client/client.py \
  --upload-first \
  /path/to/scene.usdz

# With token and custom base URL
python apps/physics_agent_service/client/client.py \
  --base-url http://localhost:8000 \
  --token "$TOKEN" \
  /path/to/scene.usdz
```

Exit behavior:
- Streams live progress (SSE) and prints updates like: `[build_dataset_usd] running overall=45%`.
- Falls back to status polling if SSE is unavailable.
- Prints artifact URLs on completion.

#### Programmatic Use

**Basic usage:**
```python
from apps.physics_agent_service.client.client import PhysicsAgentClient

client = PhysicsAgentClient(base_url="http://localhost:8000")
session_id, status = client.run_and_monitor(
    usd_path="/path/to/scene.usdz",
    user_prompt="Focus on identifying furniture parts",
    render_backend="remote",  # or "warp", "ovrtx"
)
print(session_id, status)
```

**Instanced assets:**
```python
session_id, status = client.run_and_monitor(
    usd_path="/path/to/robot.usdz",
    optimize_usd=True,
    enable_deinstance=True,
)
print(session_id, status)
```

**S3 URI (large assets):**
```python
from apps.physics_agent_service.client.client import PhysicsAgentClient

client = PhysicsAgentClient(base_url="http://localhost:8000")
session_id, status = client.run_and_monitor(
    s3_uri="s3://your-bucket/path/to/scene.usdz",
    user_prompt="Classify robot components",
)
print(session_id, status)
```

**Two-step (upload/download first, then run):**
```python
client = PhysicsAgentClient(base_url="http://localhost:8000")

# Upload from local file
session_id = client.upload_usd("/path/to/scene.usdz")
# Or download from S3
session_id = client.upload_usd(s3_uri="s3://bucket/path/robot.usdz")

# Start pipeline with existing session
run_id = client.start_pipeline(session_id=session_id)
```

#### Endpoints

Key endpoints the client uses:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/pipeline` | POST | Start pipeline (USD file, S3 URI, or session_id + optional prompt) |
| `/pipeline/upload-usd` | POST | Upload USD file or provide S3 URI, returns session_id |
| `/pipeline/{session_id}/status` | GET | Poll pipeline status |
| `/pipeline/{session_id}/events` | GET | SSE stream (progress/done/ping) |
| `/pipeline/{session_id}/results` | GET | Final results |
| `/pipeline/{session_id}/cancel` | POST | Cancel running pipeline |
| `/pipeline/{session_id}/regenerate` | POST | Re-run specific steps |
| `/artifacts/{session_id}/predictions` | GET | Download predictions JSONL |
| `/artifacts/{session_id}/report` | GET | Download HTML report |
| `/artifacts/{session_id}/dataset` | GET | Download dataset JSONL |
