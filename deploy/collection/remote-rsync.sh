#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

usage() {
  printf 'Usage: %s [--execute] <user@host> [remote_dir]\n' "$0"
  printf '\n'
  printf 'Copies this worktree to a remote CPU host for collection deployment.\n'
  printf 'Dry-run is the default. Pass --execute to run rsync.\n'
}

execute=false
if [[ "${1:-}" == "--execute" ]]; then
  execute=true
  shift
fi

remote="${1:-}"
remote_dir="${2:-~/content-agents}"

if [[ -z "$remote" ]]; then
  usage
  exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"

rsync_args=(
  -az
  --exclude=.git/
  --exclude=.venv/
  --exclude=.data/
  --exclude=.mypy_cache/
  --exclude=.pytest_cache/
  --exclude=.ruff_cache/
  --exclude=__pycache__/
  --exclude='*.pyc'
  --exclude=.env
  --exclude=deploy/collection/.collection.generated.env
  --exclude=.codex/plans/
)

if [[ "$execute" != true ]]; then
  rsync_args+=(--dry-run)
  printf 'Dry run. Re-run with --execute to copy files.\n'
fi

printf 'Target: %s:%s\n' "$remote" "$remote_dir"
rsync "${rsync_args[@]}" "$repo_root/" "$remote:$remote_dir/"

printf '\nNext remote command:\n'
printf 'ssh %q "cd %s && ./deploy/collection/deploy.py plan"\n' "$remote" "$remote_dir"
