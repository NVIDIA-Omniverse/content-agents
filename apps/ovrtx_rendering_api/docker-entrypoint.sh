#!/bin/sh
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -eu

display_num="${OVRTX_XVFB_DISPLAY:-99}"
case "$display_num" in
  ''|*[!0-9]*)
    echo "OVRTX_XVFB_DISPLAY must be a numeric display id" >&2
    exit 1
    ;;
  *)
    ;;
esac

startup_checks="${OVRTX_XVFB_STARTUP_CHECKS:-50}"
case "$startup_checks" in
  ''|*[!0-9]*)
    echo "OVRTX_XVFB_STARTUP_CHECKS must be a numeric retry count" >&2
    exit 1
    ;;
  *)
    ;;
esac
if [ "$startup_checks" -le 0 ]; then
  echo "OVRTX_XVFB_STARTUP_CHECKS must be greater than zero" >&2
  exit 1
fi

lock_file="/tmp/.X${display_num}-lock"
socket_file="/tmp/.X11-unix/X${display_num}"
xvfb_pid=""
service_pid=""

remove_owned_display_artifact() {
  path="$1"
  # Keep the -L check in the existence guard so dangling symlinks are refused.
  if [ ! -e "$path" ] && [ ! -L "$path" ]; then
    return 0
  fi
  if [ -L "$path" ]; then
    echo "Refusing to remove symlinked Xvfb display artifact: $path" >&2
    return 1
  fi
  if [ ! -O "$path" ]; then
    echo "Refusing to remove Xvfb display artifact not owned by this user: $path" >&2
    return 1
  fi
  rm -f -- "$path"
  return $?
}

stop_children() {
  if [ -n "$service_pid" ]; then
    kill "$service_pid" 2>/dev/null || true
  fi
  if [ -n "$xvfb_pid" ]; then
    kill "$xvfb_pid" 2>/dev/null || true
  fi
  return 0
}

wait_for_children() {
  if [ -n "$service_pid" ]; then
    wait "$service_pid" 2>/dev/null || true
  fi
  if [ -n "$xvfb_pid" ]; then
    wait "$xvfb_pid" 2>/dev/null || true
  fi
  return 0
}

handle_signal() {
  stop_children
  wait_for_children
  exit 143
  return 0
}

monitor_service() {
  "$@" &
  service_pid="$!"

  while :; do
    if ! kill -0 "$xvfb_pid" 2>/dev/null; then
      xvfb_status=0
      wait "$xvfb_pid" || xvfb_status="$?"
      xvfb_pid=""
      echo "Xvfb exited after startup on display :${display_num} (exit ${xvfb_status})" >&2
      kill "$service_pid" 2>/dev/null || true
      wait "$service_pid" 2>/dev/null || true
      service_pid=""
      exit 1
    fi

    if ! kill -0 "$service_pid" 2>/dev/null; then
      service_status=0
      wait "$service_pid" || service_status="$?"
      service_pid=""
      kill "$xvfb_pid" 2>/dev/null || true
      wait "$xvfb_pid" 2>/dev/null || true
      xvfb_pid=""
      exit "$service_status"
    fi

    sleep "${OVRTX_SUPERVISOR_POLL_DELAY:-1}"
  done
  return 0
}

trap handle_signal INT TERM

mkdir -p /tmp/.X11-unix
artifact_cleanup_failed=0
remove_owned_display_artifact "$lock_file" || artifact_cleanup_failed=1
remove_owned_display_artifact "$socket_file" || artifact_cleanup_failed=1
if [ "$artifact_cleanup_failed" -ne 0 ]; then
  exit 1
fi

Xvfb ":${display_num}" -screen 0 "${OVRTX_XVFB_SCREEN:-1024x768x24}" \
  +extension GLX -nolisten tcp &
xvfb_pid="$!"

attempt=0
while [ "$attempt" -lt "$startup_checks" ]; do
  if ! kill -0 "$xvfb_pid" 2>/dev/null; then
    xvfb_status=0
    wait "$xvfb_pid" || xvfb_status="$?"
    echo "Xvfb failed to start on display :${display_num} (exit ${xvfb_status})" >&2
    exit 1
  fi
  if [ -S "$socket_file" ]; then
    export DISPLAY=":${display_num}"
    monitor_service "$@"
  fi
  attempt=$((attempt + 1))
  sleep "${OVRTX_XVFB_STARTUP_DELAY:-0.1}"
done

echo "Xvfb did not create display socket ${socket_file} after ${startup_checks} checks" >&2
kill "$xvfb_pid" 2>/dev/null || true
wait "$xvfb_pid" 2>/dev/null || true
exit 1
