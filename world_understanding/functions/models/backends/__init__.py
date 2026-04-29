# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model backends for chat, VLM, and image generation models.

Public backends (nim, gemini, echo) are always available.
Internal backends are loaded if world_understanding_internal is installed
(pip install world-understanding[internal]).
"""

from . import public  # noqa: F401 -- registers public backends

try:
    import world_understanding_internal  # noqa: F401 -- registers internal backends
except ImportError:
    pass
