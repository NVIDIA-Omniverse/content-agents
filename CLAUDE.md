# Content Agents Claude Code Guide

This file provides public, repo-local guidance for Claude Code working in
`NVIDIA-Omniverse/content-agents`.

## Start Here

Use `README.md` for the canonical quick start and choose the smallest supported
workflow for the task:

1. Inspect `README.md`, `.env_example`, and the relevant app README under
   `apps/`.
2. Check prerequisites before installing: Python 3.12+, `uv`, Docker Compose
   v2.24+ for service mode, and an NVIDIA GPU/runtime when using the bundled
   rendering sidecars.
3. Keep API keys in `.env`. Do not print, commit, or paste secrets.
4. For local CLI mode from the repo root:

   ```bash
   uv venv --python=3.12
   source .venv/bin/activate
   uv pip install -e . -e apps/material_agent -e apps/physics_agent -e apps/texture_agent
   ./scripts/fetch_build_resources.sh
   ```

5. Use a dry run before expensive VLM or rendering calls when the CLI supports
   it:

   ```bash
   material-agent run apps/material_agent/configs/unified_example.yaml --dry-run
   material-agent run apps/material_agent/configs/unified_example.yaml
   ```

## Repository Map

- `world_understanding/` - shared library code, tool registry, model wrappers,
  utility functions, and minimal agent framework.
- `apps/material_agent/` - Material Agent CLI and configs.
- `apps/material_agent_service/` - Material Agent REST service and client.
- `apps/physics_agent/` - Physics Agent CLI and configs.
- `apps/physics_agent_service/` - Physics Agent REST service and client.
- `apps/texture_agent/` - Texture Agent CLI and configs.
- `apps/texture_agent_service/` - Texture Agent REST service and client.
- `apps/ovrtx_rendering_api/` - shared OVRTX rendering API sidecar.
- `.agents/skills/` - canonical checked-in agent skills.
- `.claude/skills/` and `.codex/skills/` - compatibility mirrors of the
  canonical skill tree.

## Agent Skills

Start Claude Code from the repo root so it can discover the checked-in skills.

| Workflow | Claude skill | First command |
|---|---|---|
| Material CLI | `.claude/skills/material-agent-cli` | `material-agent run apps/material_agent/configs/unified_example.yaml` |
| Physics CLI | `.claude/skills/physics-agent-cli` | `physics-agent run apps/physics_agent/configs/lightbulb.yaml` |
| Texture CLI | `.claude/skills/texture-agent-cli` | `texture-agent run apps/texture_agent/configs/texture_example.yaml` |
| Material service | `.claude/skills/deploy-material-agent-docker` | `docker compose --env-file .env -f apps/material_agent_service/docker-compose.yml up --build` |
| Physics service | `.claude/skills/deploy-physics-agent-docker` | `docker compose --env-file .env -f apps/physics_agent_service/docker-compose.yml up --build` |
| Texture service | `.claude/skills/deploy-texture-agent-docker` | `docker compose --env-file .env -f apps/texture_agent_service/docker-compose.yml up --build` |
| Full collection | `.claude/skills/deploy-collection` | `./deploy/collection/deploy.py plan && ./deploy/collection/deploy.py up` |
| USD utilities | `.claude/skills/flatten-usd`, `.claude/skills/print-usd`, `.claude/skills/render-usd` | Inspect, flatten, or render USD assets. |

## Public Backends

Public docs, configs, and examples should use only public model providers:

- `nim` with `NVIDIA_API_KEY`
- `openai` with `OPENAI_API_KEY`
- `anthropic` with `ANTHROPIC_API_KEY`
- `gemini` with `GOOGLE_API_KEY`

Do not hardcode credentials. Use placeholders in examples.

## Validation

Use lightweight checks before claiming a change is ready:

```bash
python3 -m pytest tests/test_public_quickstart_regression.py -q
python3 -m pytest tests/test_packaging_requests_declared.py tests/test_pyproject_fallback_version.py -q
```

If `pytest` is missing, install the development extras in the active
environment first:

```bash
uv pip install -e ".[dev]"
```

## Safety

- Do not commit `.env`, downloaded assets, generated textures, rendered images,
  session directories, service logs, or credentials.
- Use `--dry-run` before VLM or render calls when supported.
- Treat clean flags and commands that delete working directories as destructive.
- Keep changes scoped to the requested workflow and follow existing repo style.
