# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path

PUBLIC_REPLACEMENT_FILENAMES: dict[str, str] = {
    "README_PUBLIC.md": "README.md",
    "AGENTS_PUBLIC.md": "AGENTS.md",
    "CLAUDE_PUBLIC.md": "CLAUDE.md",
    "CHANGELOG_PUBLIC.md": "CHANGELOG.md",
    ".env_example_public": ".env_example",
}


def public_doc_path(repo_root: Path, relpath: str | Path) -> Path:
    """Return the source public doc or its staging-copy replacement."""
    path = repo_root / relpath
    if path.exists():
        return path

    target_name = PUBLIC_REPLACEMENT_FILENAMES.get(path.name)
    if target_name is not None:
        staged_path = path.with_name(target_name)
        if staged_path.exists():
            return staged_path

    return path
