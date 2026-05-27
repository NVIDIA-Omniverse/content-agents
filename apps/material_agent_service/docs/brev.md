# Material Agent Service on Brev

Use the repo-level Brev planner to choose a deployment topology before creating
instances. The planner prints `brev search` and `brev create --dry-run`
commands first, then separates cost-incurring provisioning commands for manual
review.

```bash
python scripts/brev_agent_services.py --service material --preset render-only
python scripts/brev_agent_services.py --service material --preset single-host-local-vlm
python scripts/brev_agent_services.py --service material --preset hybrid
```

## Presets

| Preset | Brev nodes | Use when |
|---|---|---|
| `render-only` | 1 RTX GPU node | You want the cheapest reliable path: local OVRTX rendering plus hosted VLM/LLM. |
| `single-host-local-vlm` | 1 multi-GPU RTX node | You want a self-contained host using the existing Cosmos Reason2 VLM sidecar. |
| `hybrid` | 1 RTX render endpoint node + 1 A100/H100-grade VLM endpoint node | You want the material pipeline to run locally while Brev hosts the heavy render and VLM endpoints. |

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
direct agent responses. For this endpoint, set both material-agent model env
vars to `qwen/qwen3.5-35b-a3b`; the endpoint env vars do not inject
`chat_template_kwargs` by themselves.

The validated H100 NIM path is `hyperstack_H100` with the same NIM image and
model ID. It uses `/ephemeral` for Docker and NIM cache, requires
`VLLM_ENABLE_CUDA_COMPATIBILITY=1`, sets `NIM_MAX_MODEL_LEN=65536` for
Material Agent, sets `NIM_MAX_IMAGES_PER_PROMPT=20` for multi-view prompts,
and can take several minutes on first cold start for DeepGEMM
warmup/autotuning. The default 262,144-token context can be killed during H100
startup; 65,536 tokens passed the ladder end-to-end smoke.

Qwen3.6 35B A3B on the same H100 path is experimental for Material Agent. Its
NIM serves `Qwen/Qwen3.6-35B-A3B` and passed text plus public image URL smokes,
but rejected local/base64 image payloads with HTTP 400. Keep Qwen3.5 35B as the
larger self-hosted Material Agent VLM until that image transport issue is
resolved.

## Hybrid Endpoint Wiring

In the default hybrid path, keep the material pipeline on the local machine and
point it at the two Brev endpoints through port-forwards:

```bash
export RENDER_ENDPOINT=http://localhost:8001
export MA_RENDERING_USE_DATA_URI=true
export MA_VLM_NIM_BASE_URL=http://localhost:8003/v1
export MA_VLM_MODEL=Qwen/Qwen3.5-4B
export MA_LLM_NIM_BASE_URL=http://localhost:8003/v1
export MA_LLM_MODEL=Qwen/Qwen3.5-4B
export MA_NIM_API_KEY=not-used
```

`MA_RENDERING_USE_DATA_URI=true` is required when the local pipeline points the
`remote` render backend at a port-forwarded OVRTX service. For smoke assets,
skip `optimize_usd` unless the local Scene Optimizer package or a deliberate
remote optimizer endpoint is available.
Use `MA_NIM_API_KEY=not-used` only for local no-auth VLM/LLM endpoints such as
a Brev port-forward. For tunnel, external URL, private-IP, or otherwise
authenticated endpoints, put the real `MA_NIM_API_KEY` in `.env` instead of the
dummy key.

For the 35B NIM endpoint, use:

```bash
export MA_VLM_MODEL=qwen/qwen3.5-35b-a3b
export MA_LLM_MODEL=qwen/qwen3.5-35b-a3b
```

The RTX PRO Server 6000 render endpoint plus H100 Qwen3.5 35B NIM passed the
local ladder CLI smoke with 20 dataset renders, 4 VLM predictions, material
application, output validation, and 2 final renders.

The planner emits separate `brev port-forward` commands for OVRTX and the VLM.
Keep both running while the local pipeline is active. This path does not require
Brev instance-to-instance networking.

## Credit Safety

Run planner-generated dry-runs first:

```bash
brev create <name> --dry-run ...
```

Only run planner-generated `brev create`, `brev copy`, `brev exec`, and
`brev port-forward` commands after the dry-run output shows acceptable capacity
and price.
