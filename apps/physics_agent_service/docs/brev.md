# Physics Agent Service on Brev

Use the repo-level Brev planner to choose a deployment topology before creating
instances. The planner prints `brev search` and `brev create --dry-run`
commands first, then separates cost-incurring provisioning commands for manual
review.

```bash
python scripts/brev_agent_services.py --service physics --preset render-only
python scripts/brev_agent_services.py --service physics --preset single-host-local-vlm
python scripts/brev_agent_services.py --service physics --preset hybrid
```

## Presets

| Preset | Brev nodes | Use when |
|---|---|---|
| `render-only` | 1 RTX GPU node | You want local OVRTX rendering plus hosted VLM/LLM. |
| `single-host-local-vlm` | 1 multi-GPU RTX node | You want a self-contained host using the existing Cosmos Reason2 VLM sidecar. |
| `hybrid` | 1 RTX render endpoint node + 1 A100/H100-grade VLM endpoint node | You want the physics pipeline to run locally while Brev hosts the heavy render and VLM endpoints. |

OVRTX rendering requires RTX-capable GPUs. Prefer AWS `g7e.2xlarge` / RTX PRO
Server 6000 for fast validation when budget allows. Use AWS `g6e.xlarge` /
L40S as the cheaper validated fallback when lower cost matters more than
cold-start latency. Other L40, L40S, A6000, RTX6000 Ada, and RTX PRO classes
remain RTX-capable candidates, but validate the provider image before promoting
one to the default. Do not use A100 or H100 as render nodes; use them for
VLM/LLM serving.

For self-hosted VLM, the first validated path is Denvr
`denvr_A100_sxm4_80G` with `Qwen/Qwen3.5-4B` served by vLLM. Before launching
the model, check `df -h / /home /mnt/*`, `lsblk -f`, Docker's root directory,
and `nvidia-smi`; use the large writable data disk for Docker or model cache
if the root filesystem is small. The validated vLLM launch used bfloat16,
`--max-model-len 4096`, `--gpu-memory-utilization 0.80`,
`--max-num-seqs 1`, `--trust-remote-code`, `--enforce-eager`,
`--reasoning-parser qwen3`, and chat-template kwargs with
`enable_thinking=false`.

The larger validated NIM option is
`nvcr.io/nim/qwen/qwen3.5-35b-a3b:1.7.0-variant` on the same A100 VM. It
requires request-level `chat_template_kwargs: {"enable_thinking": false}` for
direct agent responses. For this endpoint, set the physics-agent VLM model env
var to `qwen/qwen3.5-35b-a3b`; the endpoint env vars do not inject
`chat_template_kwargs` by themselves.

The validated H100 NIM path is `hyperstack_H100` with the same NIM image and
model ID. It uses `/ephemeral` for Docker and NIM cache, requires
`VLLM_ENABLE_CUDA_COMPATIBILITY=1`, sets `NIM_MAX_MODEL_LEN=65536` for agent
smokes, sets `NIM_MAX_IMAGES_PER_PROMPT=20` for multi-view prompts, and can
take several minutes on first cold start for DeepGEMM warmup/autotuning. The
default 262,144-token context can be killed during H100 startup. On the
validated Hyperstack image, also move containerd root to
`/ephemeral/containerd`; Docker data-root alone did not keep NIM image layers
off the small root filesystem. Mount both `/ephemeral/nim` and a writable
`/ephemeral/nim-workspace` into the container.

Qwen3.6 35B A3B on the same H100 path is experimental. Its NIM serves
`Qwen/Qwen3.6-35B-A3B` and passed text plus public image URL smokes, but
rejected local/base64 image payloads with HTTP 400. Do not promote it for
physics VLM use until the physics image payload path is explicitly smoked.

## Hybrid Endpoint Wiring

In the default hybrid path, keep the physics pipeline on the local machine and
point it at the two Brev endpoints through port-forwards:

```bash
export RENDER_ENDPOINT=http://localhost:8001
export MA_RENDERING_USE_DATA_URI=true
export PA_VLM_NIM_BASE_URL=http://localhost:8003/v1
export PA_VLM_MODEL=Qwen/Qwen3.5-4B
export PA_LLM_NIM_BASE_URL=http://localhost:8003/v1
export PA_NIM_API_KEY=not-used
```

For the 35B NIM endpoint, use:

```bash
export PA_VLM_MODEL=qwen/qwen3.5-35b-a3b
```

Use `PA_NIM_API_KEY=not-used` only for local no-auth VLM/LLM endpoints such as
a Brev port-forward. For tunnel, external URL, private-IP, or otherwise
authenticated endpoints, put the real `PA_NIM_API_KEY` in `.env` instead of the
dummy key.

The planner emits separate `brev port-forward` commands for OVRTX and the VLM.
Keep both running while the local pipeline is active. This path does not require
Brev instance-to-instance networking.
`MA_RENDERING_USE_DATA_URI=true` is the shared renderer switch that keeps local
USD payloads inside the port-forwarded render request instead of using S3.

## Validated Hybrid CLI Smoke

The validated Physics Agent CLI smoke used AWS `g7e.2xlarge` / RTX PRO Server
6000 for standalone OVRTX rendering and Hyperstack H100 for
`qwen/qwen3.5-35b-a3b` NIM. The local pipeline rendered 2 preview images,
rendered 32 dataset images for 8 mesh prims, produced 8 predictions, and wrote
a physics USD with schemas applied to all 8 prims.

For the lightbulb smoke config:

- Base the file on `apps/physics_agent/configs/lightbulb.yaml`.
- Override both renderers to `backend: remote`.
- Use valid render mode names such as `composition` and `prim_only`.
- Keep one-worker rendering for a single OVRTX endpoint.
- Set `chat_template_kwargs.enable_thinking=false` under both VLM configs.
- Ensure the predict prompt returns top-level `physical_properties`; the
  current `apply_physics` step reads `classification.physical_properties`.

## Credit Safety

Run planner-generated dry-runs first:

```bash
brev create <name> --dry-run ...
```

Only run planner-generated `brev create`, `brev copy`, `brev exec`, and
`brev port-forward` commands after the dry-run output shows acceptable capacity
and price.
