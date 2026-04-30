# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared credential resolution helpers."""

from __future__ import annotations

import ipaddress
import os
from typing import Any
from urllib.parse import urlparse

API_KEY_ENV_VAR_MAP: dict[str, tuple[str, ...]] = {
    "nim": ("NVIDIA_API_KEY",),
    "perflab_azure_openai": ("NSTORAGE_API_KEY", "AZURE_OPENAI_API_KEY"),
    "perflab": ("NSTORAGE_API_KEY",),
    "azure_openai": ("AZURE_OPENAI_API_KEY", "NSTORAGE_API_KEY"),
    "nvidia_inference": ("INFERENCE_NVIDIA_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "gemini": ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
}

LOCAL_NIM_API_KEY_PLACEHOLDER = "not-used"
_API_KEY_PLACEHOLDER_VALUES = {
    LOCAL_NIM_API_KEY_PLACEHOLDER,
    "your_api_key",
    "your_api_key_here",
    "your_nvidia_api_key",
    "your_openai_api_key",
    "your_anthropic_api_key",
    "your_google_api_key",
    "your_gemini_api_key",
    "replace_me",
    "changeme",
    "todo",
}
_NVIDIA_HOST_SUFFIXES = ("nvidia.com",)
_OPENAI_HOST_SUFFIXES = ("openai.com",)


def _is_provider_owned_base_url(base_url: Any, host_suffixes: tuple[str, ...]) -> bool:
    if not isinstance(base_url, str) or not base_url.strip():
        return False

    parsed = urlparse(base_url.strip())
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return False

    host = parsed.hostname
    if not host:
        return False
    host_lower = host.lower()
    return any(
        host_lower == suffix or host_lower.endswith(f".{suffix}")
        for suffix in host_suffixes
    )


def is_nvidia_provider_base_url(base_url: Any) -> bool:
    """Return True when ``base_url`` is unset or belongs to NVIDIA."""
    return not base_url or _is_provider_owned_base_url(base_url, _NVIDIA_HOST_SUFFIXES)


def is_openai_provider_base_url(base_url: Any) -> bool:
    """Return True when ``base_url`` is unset or belongs to OpenAI."""
    return not base_url or _is_provider_owned_base_url(base_url, _OPENAI_HOST_SUFFIXES)


def is_placeholder_api_key(value: Any) -> bool:
    """Return True for documented placeholder credential values."""
    if not isinstance(value, str):
        return False
    normalized = value.strip().lower()
    if not normalized:
        return False
    if normalized in _API_KEY_PLACEHOLDER_VALUES:
        return True
    return normalized.startswith("your_") or normalized.startswith("your-")


def is_local_nim_api_key_placeholder(value: Any) -> bool:
    """Return True only for the explicit local NIM no-auth opt-in value."""
    return (
        isinstance(value, str)
        and value.strip().lower() == LOCAL_NIM_API_KEY_PLACEHOLDER
    )


_LLM_NIM_ENV_BASE_URL_VARS = ("MA_LLM_NIM_BASE_URL", "MA_VLM_NIM_BASE_URL")


def get_llm_nim_env_base_url_override() -> str | None:
    """Return the runtime LLM NIM env base-URL override, if any.

    ``MA_LLM_NIM_BASE_URL`` (preferred) and ``MA_VLM_NIM_BASE_URL`` (fallback)
    redirect every LLM call to a NIM endpoint at runtime. The same override
    must be applied at preflight and during model provisioning so the
    selected backend, base URL, and credential resolution are consistent
    across config validation and execution.
    """
    for var in _LLM_NIM_ENV_BASE_URL_VARS:
        value = os.getenv(var)
        if value:
            return value
    return None


def apply_llm_nim_env_override(llm_config: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``llm_config`` with the runtime LLM NIM override applied.

    When ``MA_LLM_NIM_BASE_URL`` or ``MA_VLM_NIM_BASE_URL`` is set, the section
    is forced to ``backend: nim`` and the env-supplied ``base_url``; any
    endpoint-scoped fields from the prior backend are dropped (with the
    explicit local-NIM no-auth placeholder preserved). When neither env var is
    set, ``llm_config`` is returned unchanged (still copied to avoid mutating
    the caller's dict).

    Mock / echo configs and configs with no ``backend`` are left alone — the
    override is a *runtime routing* hint that should not silently turn a
    deliberately-mocked simulate run into a real NIM call when the operator
    happens to also have ``MA_LLM_NIM_BASE_URL`` set in the environment.
    """
    config = dict(llm_config)
    nim_base_url = get_llm_nim_env_base_url_override()
    if not nim_base_url:
        return config
    backend = (config.get("backend") or config.get("provider") or "").strip().lower()
    if backend in ("", "echo", "mock"):
        return config
    drop_stale_endpoint_credentials(config, preserve_local_nim_placeholder=True)
    config["backend"] = "nim"
    config["base_url"] = nim_base_url
    return config


def drop_stale_endpoint_credentials(
    model_config: dict[str, Any],
    *,
    preserve_local_nim_placeholder: bool = False,
) -> None:
    """Drop ``api_key``/``base_url`` left over from a previous backend.

    Whenever ``backend`` or ``base_url`` on a model section is rewritten
    (env override, runtime NIM-base-URL injection, service-side route
    selection), the prior ``api_key`` and ``base_url`` belonged to a
    different endpoint. Leaving them in place can forward one provider's
    credential to another endpoint or send traffic to the old URL while
    validation reports success.

    Args:
        model_config: Model section dict (mutated in place).
        preserve_local_nim_placeholder: If True, keep an existing ``api_key``
            equal to the local NIM no-auth placeholder. Use this when
            reusing the section for a NIM endpoint where the operator has
            explicitly opted into no-auth.
    """
    if not (
        preserve_local_nim_placeholder
        and is_local_nim_api_key_placeholder(model_config.get("api_key"))
    ):
        model_config.pop("api_key", None)
    model_config.pop("base_url", None)


def is_local_base_url(base_url: Any) -> bool:
    """Return True for local or cluster-private service endpoints."""
    if not isinstance(base_url, str) or not base_url.strip():
        return False

    parsed = urlparse(base_url.strip())
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return False

    host = parsed.hostname
    if not host:
        return False
    host_lower = host.lower()
    if host_lower == "localhost":
        return True
    if "." not in host_lower:
        return True
    if host_lower.endswith((".local", ".svc", ".svc.cluster.local")):
        return True

    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return address.is_loopback or address.is_private or address.is_link_local


def get_env_api_key_for_backend(
    backend: str,
    explicit_api_key: Any = None,
) -> str | None:
    """Resolve a backend API key from explicit input or environment aliases."""
    if explicit_api_key is not None:
        explicit_api_key_str = str(explicit_api_key).strip()
        if explicit_api_key_str and not is_placeholder_api_key(explicit_api_key_str):
            return explicit_api_key_str

    for env_var in API_KEY_ENV_VAR_MAP.get(backend, ()):
        api_key = os.getenv(env_var)
        if api_key and not is_placeholder_api_key(api_key):
            return api_key

    return None


def get_nim_api_key_for_base_url(
    base_url: Any,
    explicit_api_key: Any = None,
) -> str | None:
    """Resolve a NIM API key with explicit local sidecar no-auth opt-in.

    Hosted NVIDIA NIM endpoints require a real key from explicit config or
    ``NVIDIA_API_KEY``. Non-hosted NIM endpoints (in-cluster sidecars and
    operator-configured external NIM URLs such as the helm chart's
    ``vlmNim.endpointOverride``) use ``MA_NIM_API_KEY`` as the explicit
    NIM-scoped opt-in; the hosted ``NVIDIA_API_KEY`` is never silently
    forwarded to a non-NVIDIA NIM endpoint. No-auth NIM endpoints may use the
    ``not-used`` placeholder, but only when explicitly supplied via config or
    ``MA_NIM_API_KEY``.
    """
    if explicit_api_key is not None:
        explicit_api_key_str = str(explicit_api_key).strip()
        if explicit_api_key_str and not is_placeholder_api_key(explicit_api_key_str):
            return explicit_api_key_str

    is_local_endpoint = is_local_base_url(base_url)
    if is_local_endpoint and is_local_nim_api_key_placeholder(explicit_api_key):
        return LOCAL_NIM_API_KEY_PLACEHOLDER

    is_nvidia_endpoint = is_nvidia_provider_base_url(base_url)
    ma_nim_api_key = os.getenv("MA_NIM_API_KEY")
    if not is_nvidia_endpoint:
        # Non-hosted NIM (local sidecar or custom remote NIM URL): the
        # operator must opt in with ``MA_NIM_API_KEY`` (real value or the
        # ``not-used`` no-auth placeholder). ``NVIDIA_API_KEY`` is never
        # silently forwarded to non-NVIDIA endpoints.
        if ma_nim_api_key and not is_placeholder_api_key(ma_nim_api_key):
            return ma_nim_api_key
        if is_local_nim_api_key_placeholder(ma_nim_api_key):
            return LOCAL_NIM_API_KEY_PLACEHOLDER
        return None

    nvidia_api_key = os.getenv("NVIDIA_API_KEY")
    if nvidia_api_key and not is_placeholder_api_key(nvidia_api_key):
        return nvidia_api_key

    return None


def resolve_effective_openai_base_url(explicit: Any = None) -> str | None:
    """Return the OpenAI base URL the SDK will actually hit.

    ``langchain_openai.ChatOpenAI`` (and the underlying ``openai`` SDK) fall
    back to ``OPENAI_BASE_URL`` / ``OPENAI_API_BASE`` when the constructor
    receives no explicit ``base_url``. Endpoint-based credential checks must
    use this effective URL or the hosted ``OPENAI_API_KEY`` could be sent to
    an env-redirected custom endpoint.
    """
    if isinstance(explicit, str) and explicit.strip():
        return explicit
    return os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")


def get_openai_api_key_for_base_url(
    base_url: Any,
    explicit_api_key: Any = None,
) -> str | None:
    """Resolve an OpenAI-compatible API key with local no-auth support.

    Hosted OpenAI endpoints require a real key from explicit config or
    ``OPENAI_API_KEY``. Custom remote OpenAI-compatible endpoints require
    explicit config so provider credentials are not sent to arbitrary compatible
    services. Local endpoints may use the documented ``not-used`` dummy key only
    when it is explicitly supplied in config, so authenticated private gateways
    do not get silently treated as no-auth services.

    The check is performed against the *effective* base URL — when no
    ``base_url`` is supplied the OpenAI SDK falls back to ``OPENAI_BASE_URL``
    / ``OPENAI_API_BASE``, and a hosted ``OPENAI_API_KEY`` must not be
    forwarded to that env-redirected endpoint. An explicit ``api_key`` is
    accepted only when the caller also explicitly chose the endpoint (config
    ``base_url``) or the effective endpoint is provider-owned/local; an
    env-redirected custom endpoint paired with an explicit ``api_key`` is
    rejected so the caller's hosted key cannot follow an unintended redirect.
    """
    config_supplied_base_url = isinstance(base_url, str) and bool(base_url.strip())
    effective_base_url = resolve_effective_openai_base_url(base_url)
    is_local_endpoint = is_local_base_url(effective_base_url)
    is_provider_endpoint = is_openai_provider_base_url(effective_base_url)

    explicit_api_key_str = (
        str(explicit_api_key).strip() if explicit_api_key is not None else None
    )
    if explicit_api_key_str is not None:
        if explicit_api_key_str and not is_placeholder_api_key(explicit_api_key_str):
            # Trust an explicit api_key only when the user paired it with an
            # explicit endpoint, or the resolved endpoint is provider-owned.
            # Local endpoints require explicit pairing too — a malicious
            # ``OPENAI_BASE_URL=http://attacker.local/v1`` would otherwise
            # exfiltrate a hosted key to a non-pair-validated host.
            if config_supplied_base_url or is_provider_endpoint:
                return explicit_api_key_str
            return None
        if is_local_endpoint and is_local_nim_api_key_placeholder(explicit_api_key_str):
            return LOCAL_NIM_API_KEY_PLACEHOLDER

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not is_provider_endpoint:
        # Non-provider endpoints (local, custom remote, env-redirected) never
        # silently inherit the hosted ``OPENAI_API_KEY``. The trust boundary
        # for forwarding the hosted key is the OpenAI provider URL set —
        # local servers, even when explicitly paired in config, must opt in
        # via an endpoint-scoped ``api_key`` or the ``not-used`` no-auth
        # placeholder. Callers like ``wu image-gen --base-url <local>``
        # inject ``not-used`` after this returns ``None``.
        return None

    if openai_api_key and not is_placeholder_api_key(openai_api_key):
        return openai_api_key

    return None
