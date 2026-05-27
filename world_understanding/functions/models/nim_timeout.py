# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Timeout helpers for LangChain NVIDIA NIM chat clients."""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _apply_nim_chat_timeout(
    chat_model: Any, timeout: float | None, *, label: str
) -> None:
    """Apply timeout to ChatNVIDIA's underlying HTTP clients when available."""
    if timeout is None:
        return

    timeout_s = float(timeout)
    configured = False
    for client_attr in ("_client", "_async_client"):
        client = getattr(chat_model, client_attr, None)
        if client is None:
            continue
        client.timeout = timeout_s
        configured = True

    if not configured:
        logger.warning(
            "%s could not apply timeout=%s because ChatNVIDIA exposed no HTTP client.",
            label,
            timeout_s,
        )
