# Texture Agent Service on Brev

Use the repo-level Brev planner to choose a deployment topology before creating
instances. The planner prints `brev search` and `brev create --dry-run`
commands first, then separates cost-incurring provisioning commands for manual
review.

```bash
python scripts/brev_agent_services.py --service texture --preset service-only
python scripts/brev_agent_services.py --service texture --preset single-host-local-sidecars
python scripts/brev_agent_services.py --service texture --preset hybrid
```

## Presets

| Preset | Brev nodes | Use when |
|---|---|---|
| `hybrid` | 1 image-gen GPU node plus optional L4/24 GB LLM node | You want the local texture pipeline to call Brev-hosted dependency endpoints. |
| `service-only` | 1 CPU node | You are explicitly validating service-on-Brev behavior. |
| `single-host-local-sidecars` | 1 multi-GPU node | You want local FLUX image generation and local LLM sidecars on one host. |

Texture Agent does not require OVRTX rendering. Local image generation is the
main GPU cost driver, so the default target keeps the pipeline local and runs
image generation as a Brev-hosted endpoint. The validated image-generation
target is AWS `g6e.xlarge` / L40S with FLUX NIM; use that exact type before
falling back to generic L40S searches. The planner skips the optional LLM node
by default when the pipeline config already provides prompts for every
material. Add `--texture-include-llm` only when auto-prompt generation needs a
Brev-hosted Qwen endpoint.

## Hybrid Endpoint Wiring

In the default hybrid path, keep the texture pipeline on the local machine and
point it at Brev dependency endpoints through port-forwards:

```bash
export TA_IMAGE_GEN_BACKEND=openai
export TA_IMAGE_GEN_BASE_URL=http://localhost:8005/v1
export TA_IMAGE_GEN_MODEL=black-forest-labs/flux.2-klein-4b
export TA_IMAGE_GEN_API_KEY=not-used
```

Use `TA_IMAGE_GEN_API_KEY=not-used` only for local no-auth endpoints such as a
planner-created Brev port-forward. When `--image-gen-base-url` points at an
existing HTTPS, tunnel, or otherwise authenticated endpoint, put the real
`TA_IMAGE_GEN_API_KEY` in `.env` and do not use the dummy key.

When auto-prompt generation is required, rerun the planner with
`--texture-include-llm` and add the LLM endpoint wiring:

```bash
export TA_LLM_BACKEND=nim
export TA_LLM_BASE_URL=http://localhost:8003/v1
export TA_LLM_MODEL=Qwen/Qwen3.5-4B
export TA_NIM_API_KEY=not-used
```

Use `TA_NIM_API_KEY=not-used` only for local no-auth LLM endpoints such as a
Brev port-forward. For tunnel, external URL, private-IP, or otherwise
authenticated endpoints, put the real `TA_NIM_API_KEY` or `TA_LLM_API_KEY` in
`.env` instead of the dummy key.

Keep the planner-generated `brev port-forward` running while the local pipeline
is active. This path does not require Brev instance-to-instance networking.

For texture-agent-service Docker Compose modes, put real endpoint credentials
such as `TA_IMAGE_GEN_API_KEY`, `TA_LLM_API_KEY`, or `TA_NIM_API_KEY` in the
repo-root or service `.env` file. Host shell exports are enough for local CLI
pipelines, but Compose service containers read these values through `env_file`.

For local Qwen-family serving, disable Qwen thinking in vLLM so the Texture
Agent receives direct answer text in `message.content`.

The validated FLUX endpoint uses:

Copy only `NGC_API_KEY` and `HF_TOKEN` to the remote NIM env file; do not copy
the full local `.env`. FLUX NIM needs `NGC_API_KEY` for the container pull and
`HF_TOKEN` for model weight access.

```bash
brev create wu-ta-image-gen --dry-run --type g6e.xlarge --min-disk 500
brev create wu-ta-image-gen --type g6e.xlarge --min-disk 500 --timeout 1200
brev exec wu-ta-image-gen \
  "set -e; \
   sudo mkdir -p /opt/dlami/nvme/docker /opt/dlami/nvme/nim-cache && \
   sudo chown -R \$(id -un):\$(id -gn) /opt/dlami/nvme/nim-cache && \
   sudo env DOCKER_DATA_ROOT=/opt/dlami/nvme/docker python3 - <<'PY'
import json
import os
from pathlib import Path

path = Path('/etc/docker/daemon.json')
text = path.read_text() if path.exists() else ''
data = json.loads(text) if text.strip() else {}
if not isinstance(data, dict):
    raise SystemExit(f'{path} must contain a JSON object')
data['data-root'] = os.environ['DOCKER_DATA_ROOT']
path.write_text(json.dumps(data, indent=2) + '\n')
PY
   sudo nvidia-ctk runtime configure --runtime=docker --set-as-default && \
   sudo systemctl restart docker && \
   sudo chmod -R u+rwX,g+rwX,o-rwx /opt/dlami/nvme/nim-cache"
umask 077
if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi
test -n "$NGC_API_KEY"
test -n "$HF_TOKEN"
printf 'NGC_API_KEY=%s\nHF_TOKEN=%s\n' "$NGC_API_KEY" "$HF_TOKEN" > /tmp/wu-ngc-nim.env
brev copy /tmp/wu-ngc-nim.env wu-ta-image-gen:/home/ubuntu/.ngc-nim.env
rm -f /tmp/wu-ngc-nim.env
brev exec wu-ta-image-gen \
  "chmod 600 /home/ubuntu/.ngc-nim.env && \
   set -a; . /home/ubuntu/.ngc-nim.env; set +a; \
   printf '%s\n' \"\$NGC_API_KEY\" | \
     docker login nvcr.io -u '\$oauthtoken' --password-stdin"
brev exec wu-ta-image-gen \
  "docker rm -f flux-image-gen >/dev/null 2>&1 || true; \
   docker run -d --name flux-image-gen --gpus all --ipc=host -p 8000:8000 \
     --env-file /home/ubuntu/.ngc-nim.env \
     -e NIM_CACHE_PATH=/opt/nim/.cache \
     -v /opt/dlami/nvme/nim-cache:/opt/nim/.cache \
     nvcr.io/nim/black-forest-labs/flux.2-klein-4b:1.0.1-variant"
brev exec wu-ta-image-gen "curl -fsS http://localhost:8000/v1/health/ready"
brev port-forward wu-ta-image-gen -p 8005:8000
```

Use `/v1/health/ready` for FLUX readiness; `/v1/models` returned 404 for the
image-generation NIM. The endpoint supports the image generation and image
edit routes used by Texture Agent's OpenAI-compatible image generation client.

If you intentionally run `texture-agent-service` on the same Brev node as the
LLM, use `--hybrid-connectivity same-node`; the planner will include
`docker-compose.brev-host-llm.yml` so Linux Docker containers can resolve
`host.docker.internal`.

## Credit Safety

Run planner-generated dry-runs first:

```bash
brev create <name> --dry-run ...
```

Only run planner-generated `brev create`, `brev copy`, `brev exec`, and
`brev port-forward` commands after the dry-run output shows acceptable capacity
and price.
