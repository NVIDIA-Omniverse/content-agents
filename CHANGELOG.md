# Content Agents 0.4.2 (27 May 2026)

Public Content Agents release notes.

This update follows the v0.4.1 public release and covers the delta since
v0.4.1.

## Improvements

- OVRTX rendering now uses the `0.3.0.312915` runtime with an isolated,
  version-checked provisioner shared by local runtime setup and Docker image
  builds.
- OVRTX service and worker paths were updated for the 0.3 renderer API and
  continue to preserve the existing render request contract.
- OVRTX validation now includes opt-in probes for native visibility,
  time-sampled displayColor, sample attributes, and in-process OpenUSD
  coexistence; production safety workarounds remain enabled by default.
- Public source release hygiene now includes complete SPDX headers on shipped
  Python files and a staging guardrail to keep that coverage in place.
- Texture Agent Service now supports S3-backed shared session storage for
  multi-instance deployments, with cross-instance worker reservations,
  cancellation markers, artifact sync, and presigned artifact downloads.
- Material and Physics agent material authoring now types conventional `Looks`
  containers as USD `Scope` prims.

## Fixes

- Texture Agent Service S3 session deletion now preserves metadata and the
  session index on partial artifact-delete failures so cleanup remains
  retryable.
- Texture Agent Service live event streams now return the polling fallback when
  a different instance owns a shared-storage session, preventing clients from
  waiting forever on stale local upload metadata.
- Texture Agent Service queued worker reservations now keep ownership alive
  before execution starts, avoiding duplicate cross-instance processing.
- Texture Agent Service local metadata writes remain atomic when the local
  session store points at the same session directory.
- Material Agent validation with `on_failure: warn` now skips unavailable
  optional USD validation instead of aborting public runs.
- NIM timeout settings are no longer serialized into chat-completion request
  bodies that reject unknown parameters.
- OVRTX render exports now avoid selecting MaterialX-only outputs when a
  universal surface fallback is available, preventing red error-material
  renders.
