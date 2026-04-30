"""Regression tests for the public quickstart paths (NVBug 6125716).

QA filed NVBug 6125716 after the public README quickstart for the physics
agent failed end-to-end with only NVIDIA_API_KEY in .env. Two distinct
foot-guns were responsible:

1. ``apps/physics_agent/configs/lightbulb.yaml`` -- the ``identify_asset``
   step had no ``renderer`` block, so it silently fell back to
   ``IDENTIFY_ASSET_DEFAULTS["renderer"]["backend"] = "remote"`` and tried
   to call NVCF, which the public user has not configured.

2. The agent-service ``docker-compose.yml`` files re-listed VLM provider
   API keys under ``environment:`` with ``${VAR:-}`` substitution. Compose
   substitution does NOT read ``env_file:`` -- it reads the project-dir
   ``.env`` (which defaults to the compose file's directory). With the
   user's ``.env`` at the repo root, every key resolved to an empty string
   and clobbered the values that ``env_file: path: ../../.env`` had just
   loaded into the container.

   The same substitution context applies to the ``${VAR:-default}`` lines
   for ``*_VLM_BACKEND``, ``*_VLM_MODEL``, ``*_LLM_BACKEND`` etc. -- those
   resolve to the *built-in default* (e.g. ``nim``) instead of the user's
   ``.env`` override. We don't strip those from ``environment:`` here
   because they're legitimate compose-level defaults; the public READMEs
   document ``--env-file .env`` so substitution finds the repo-root file.

These tests pin the fix in place: the public configs use the local OVRTX
backend by default, the compose files do not list provider API keys
under ``environment:`` (they flow through ``env_file:`` instead), and
the public READMEs document ``--env-file .env`` for the documented
``docker compose`` invocation.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

# Public-shipping configs that a user with only NVIDIA_API_KEY should be
# able to run on a single GPU box. Each entry maps the config path to the
# pipeline steps whose `renderer` block must default to a local backend.
_PUBLIC_CONFIG_LOCAL_RENDER_STEPS: dict[str, tuple[str, ...]] = {
    "apps/physics_agent/configs/lightbulb.yaml": (
        "identify_asset",
        "build_dataset_usd",
    ),
}

# Compose files that load the repo-root .env via long-form `env_file`. The
# fix is to NOT list provider API keys under `environment:` for the same
# service, because Compose substitution can't see env_file contents.
_COMPOSE_FILES = (
    "apps/physics_agent_service/docker-compose.yml",
    "apps/material_agent_service/docker-compose.yml",
    "apps/texture_agent_service/docker-compose.yml",
)

_FORBIDDEN_ENV_KEYS = (
    "NVIDIA_API_KEY",
    "INFERENCE_NVIDIA_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "HF_TOKEN",
    # NGC_API_KEY is also passed via env_file; keeping it under
    # `environment: ${NGC_API_KEY:-}` would clobber .env values too.
    "NGC_API_KEY",
)


def _service_uses_repo_root_env_file(service: dict) -> bool:
    """Return True if the service declares env_file pointing at ../../.env."""
    env_file = service.get("env_file")
    if env_file is None:
        return False
    if isinstance(env_file, str):
        return env_file.endswith("../../.env")
    if isinstance(env_file, list):
        for entry in env_file:
            if isinstance(entry, str) and entry.endswith("../../.env"):
                return True
            if isinstance(entry, dict) and str(entry.get("path", "")).endswith(
                "../../.env"
            ):
                return True
    return False


def _environment_keys(service: dict) -> set[str]:
    """Extract VAR names from the service's `environment:` block."""
    env = service.get("environment")
    if env is None:
        return set()
    keys: set[str] = set()
    if isinstance(env, list):
        for entry in env:
            if not isinstance(entry, str):
                continue
            name, _, _ = entry.partition("=")
            keys.add(name.strip())
    elif isinstance(env, dict):
        keys.update(str(k) for k in env.keys())
    return keys


@pytest.mark.parametrize(
    "config_relpath, steps",
    list(_PUBLIC_CONFIG_LOCAL_RENDER_STEPS.items()),
)
def test_public_config_render_steps_default_to_ovrtx(
    config_relpath: str, steps: tuple[str, ...]
) -> None:
    """Public configs must render with a local backend out of the box.

    A public user with only NVIDIA_API_KEY in .env (no NGC_API_KEY, no
    NVCF function id) must be able to run the quickstart end-to-end on a
    single GPU machine. Every step that renders has to use a local backend
    (ovrtx or warp) -- never `remote`, which requires NVCF.
    """
    config_path = REPO_ROOT / config_relpath
    with config_path.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)

    for step_name in steps:
        step = config["steps"][step_name]
        renderer = step.get("renderer")
        assert renderer is not None, (
            f"{config_relpath}::{step_name} has no `renderer` block, so it "
            "will fall back to the global default which is `remote` "
            "(requires NVCF). Add `renderer: {backend: ovrtx, ...}`."
        )
        backend = renderer.get("backend")
        assert backend in {"ovrtx", "warp"}, (
            f"{config_relpath}::{step_name}.renderer.backend = {backend!r}; "
            "public quickstart configs must use a local backend so users "
            "with only NVIDIA_API_KEY can run end-to-end."
        )


@pytest.mark.parametrize("compose_relpath", _COMPOSE_FILES)
def test_compose_does_not_clobber_env_file_api_keys(compose_relpath: str) -> None:
    """env_file values must not be clobbered by `environment: ${VAR:-}` lines.

    Compose substitution does not read `env_file:` contents -- it reads
    the project-directory `.env` (which defaults to the compose file's
    parent directory). When the user's `.env` lives at the repo root,
    `${NVIDIA_API_KEY:-}` resolves to an empty string and overrides the
    value `env_file: ../../.env` just loaded into the container.

    The fix is to NOT list any provider API key under `environment:` for
    services that already pull from the repo-root env_file. The keys
    flow through env_file directly.
    """
    compose_path = REPO_ROOT / compose_relpath
    with compose_path.open(encoding="utf-8") as f:
        compose = yaml.safe_load(f)

    services = compose.get("services", {})
    offenders: list[str] = []
    for service_name, service in services.items():
        if not _service_uses_repo_root_env_file(service):
            continue
        env_keys = _environment_keys(service)
        for forbidden in _FORBIDDEN_ENV_KEYS:
            if forbidden in env_keys:
                offenders.append(f"{service_name}.{forbidden}")

    assert not offenders, (
        f"{compose_relpath} re-lists API keys under `environment:` for "
        "services that already use repo-root env_file. Compose substitution "
        "would clobber the env_file value with empty when the user's .env "
        "lives at the repo root. Drop these entries:\n  - " + "\n  - ".join(offenders)
    )


# Public READMEs that document a `docker compose -f apps/<svc>/docker-compose.yml`
# invocation. Each must use `--env-file .env` so Compose substitution resolves
# `${VAR:-default}` against the repo-root .env that the README told the user
# to populate -- without it, settings like `MA_VLM_BACKEND=openai` set in .env
# silently get clobbered by the compose default, even though the API key
# alongside them does flow through via the `env_file:` directive.
_PUBLIC_DOCKER_COMPOSE_READMES = (
    "README_PUBLIC.md",
    "apps/physics_agent_service/README.md",
    "apps/material_agent_service/README_PUBLIC.md",
    "apps/texture_agent_service/README.md",
)


@pytest.mark.parametrize("readme_relpath", _PUBLIC_DOCKER_COMPOSE_READMES)
def test_public_readme_compose_invocation_uses_env_file(readme_relpath: str) -> None:
    """Public docker-compose invocations must pass `--env-file .env`.

    Compose's variable substitution (``${VAR:-default}``) resolves against
    the project-directory ``.env``, which defaults to the compose file's
    parent directory. Without ``--env-file .env`` passed on the CLI, a
    user's ``PA_VLM_BACKEND=openai`` (or ``MA_VLM_MODEL=...``, etc.) in
    repo-root ``.env`` is silently ignored and Compose substitutes the
    built-in default -- the container ends up with the user's API key
    set but the wrong backend selected.

    This test scans every ``docker compose ... -f apps/<svc>/...``
    invocation in the public READMEs and requires that the same shell
    block / continued line includes ``--env-file .env``.
    """
    readme_path = REPO_ROOT / readme_relpath
    text = readme_path.read_text(encoding="utf-8")

    # Walk fenced ```bash blocks and look for `docker compose ... -f apps/`
    # invocations. Folded across line continuations (`\\\n`) so the test
    # tolerates the multi-line form the READMEs use.
    in_bash_block = False
    pending: list[str] = []
    invocations: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if line.strip().startswith("```"):
            # Toggle on/off any fenced block; treat shell-ish blocks as bash.
            fence = line.strip()
            if not in_bash_block and (fence == "```bash" or fence == "```sh"):
                in_bash_block = True
            elif in_bash_block:
                in_bash_block = False
                pending = []
            continue
        if not in_bash_block:
            continue
        # Strip leading prompt characters and inline comments.
        stripped = line.lstrip("# ").lstrip("$ ").rstrip()
        if stripped.endswith("\\"):
            pending.append(stripped[:-1].rstrip())
            continue
        pending.append(stripped)
        joined = " ".join(p for p in pending if p)
        pending = []
        if "docker compose" in joined and "-f apps/" in joined:
            invocations.append(joined)

    assert invocations, (
        f"{readme_relpath} declares no `docker compose -f apps/...` "
        "invocation. If the README intentionally moved away from compose, "
        "remove the entry from _PUBLIC_DOCKER_COMPOSE_READMES."
    )

    missing = [inv for inv in invocations if "--env-file" not in inv]
    assert not missing, (
        f"{readme_relpath} contains `docker compose` invocations that "
        "omit `--env-file .env`. Without that flag, Compose's `${{VAR:-...}}` "
        "substitution reads the compose-file-adjacent .env (which the user "
        "did not create) and silently falls back to built-in defaults, "
        "ignoring the user's repo-root .env overrides for backend/model "
        "variables. Offending lines:\n  - " + "\n  - ".join(missing)
    )


def test_texture_agent_cli_bootstraps_dotenv() -> None:
    """texture-agent must load repo-root .env before model calls need keys."""
    package_init = REPO_ROOT / "apps/texture_agent/texture_agent/__init__.py"
    cli_entrypoint = REPO_ROOT / "apps/texture_agent/texture_agent/cli.py"

    init_text = package_init.read_text(encoding="utf-8")
    cli_text = cli_entrypoint.read_text(encoding="utf-8")

    assert "from dotenv import load_dotenv" in init_text
    assert "load_dotenv()" in init_text
    assert "from dotenv import load_dotenv" in cli_text
    assert "load_dotenv()" in cli_text
