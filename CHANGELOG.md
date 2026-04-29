# Content Agents 0.3.9 (28 Apr 2026)

Bug-fix release addressing 10 issues filed against the 0.3.8 public release, plus a new SimReady teaser pipeline and texture-agent improvements.

## Added

- SimReady teaser GIF grid in `README.md` and `README_PUBLIC.md` —
  four animated teasers (cleaning trolley, electrician's toolbox,
  steel rolling scaffold, UR10) showing each asset progressing from
  the gray input through Material Agent material assignment, Texture
  Agent rusty texture pass, and Physics Agent drop simulation.
- `texture-agent` CLI now auto-loads a project-local `.env`, so the
  documented Quick Start runs zero-edit when keys live in a `.env`
  file alongside the config.
- `texture-agent` `apply_textures` step now also writes the concrete
  OpenPBR `tiledimage_*` shader inputs alongside the existing abstract
  inputs, so materials authored against the NVCF tiledimage variant
  pick up the new textures.
- `texture-agent` `prepare_uvs` step has a Python-only fallback when
  the Scene Optimizer UV path is unavailable.
- README adds a **Use a Coding Agent** section with both a one-line
  paste-ready prompt and an explicit 6-step setup prompt, launch
  examples for the Codex app, Codex CLI, Claude Code CLI, and OpenClaw
  CLI, plus an **Agent Follow-Up Prompts** subsection covering the
  hello-world ladder, BYO USD material run, and BYO physics-agent
  config. The companion **Bring Your Own Asset** walkthrough copies
  the shipped material/physics/texture configs and edits the asset
  path.

## Changed

- README marks Material Agent and Physics Agent as **(Beta)** to
  reflect their current public-readiness.
- `ovrtx-rendering-api` extracts ZIP payloads (including `.usdz`
  packages) into a working directory before re-export, so relative
  texture references resolve as real files instead of staying behind
  `ArPackageResolver` and disappearing on export.

## Fixed

- Public Quick Start (Option B) now succeeds when the source is a ZIP
  download (no `.git` directory). Every `pyproject.toml` declares a
  fallback version so `uv-dynamic-versioning` resolves cleanly, and
  chained `uv pip install -e apps/<svc>` continues to satisfy the
  `world-understanding>=0.2.0` floor.
- `pip install -e apps/<svc>_agent_service` now ships a working editable
  install for all four agent services (material, physics, texture,
  joint). The documented `from client.client import <Svc>AgentClient`
  import works from any cwd.
- The texture-agent-service Dockerfile no longer references an
  unreachable internal `--extra-index-url`; sibling material- and
  physics-agent service Dockerfiles already used the same pattern.
- The public material-agent and physics-agent service
  `docker-compose.yml` files no longer reference internal-only
  inference env vars; both services pick up the configured public
  VLM backend (NVIDIA NIM / OpenAI / Anthropic / Gemini) without any
  internal-network credentials.
- `texture-agent run` exits non-zero with a fatal `RuntimeError` when
  every per-unit texture generation request fails (e.g. expired NIM
  API key returning HTTP 403), instead of reporting "Pipeline complete!"
  with exit 0. Partial failures still log a warning and let the
  pipeline continue.
- The texture-agent public `texture_example.yaml` uses the public NIM
  image-gen default (`nim` / `black-forest-labs/flux_2-klein-4b`), so
  the documented Quick Start runs zero-edit with `NVIDIA_API_KEY`.
- Every package whose source imports `requests` declares it as a
  direct dependency, so the pipeline survives `--no-deps` installs and
  upstream changes to transitive resolution.

## Documentation

- `README_PUBLIC.md` adds a **System Requirements** table covering
  GPU/VRAM, CPU, RAM, OS, and NVIDIA driver for the default
  material/physics deployment, the optional VLM NIM sidecar (material
  agent only), and the texture-agent service base plus its optional
  GPU-backed `image-gen` and `llm` NIM sidecars. Includes a
  `docker login nvcr.io --password-stdin` step so the NIM image pulls
  authenticate without the API key landing in process argv.
- `README_PUBLIC.md` adds a **Where outputs land** section explaining
  the `working_dir` convention and the exact `apply_physics` output
  path; `apps/physics_agent/configs/lightbulb.yaml` documents the same
  in its `apply_physics` step comment.
- `apps/texture_agent_service/docs/api.md` `/results` JSON example
  matches the live wire format (nested `download_urls` / `stats`),
  documents the `202` in-flight response code, and the `/status`
  example matches the live `PipelineStatus` model and includes the
  `cancelling` state held between `POST /cancel` and the worker's
  next checkpoint.
