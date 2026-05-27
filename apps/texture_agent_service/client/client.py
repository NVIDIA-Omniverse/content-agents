# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


@dataclass(frozen=True)
class SSEMessage:
    """Represents a parsed Server-Sent Event (SSE) message."""

    event: str
    data: str
    id: str | None = None
    retry: int | None = None

    def json(self) -> dict:
        """Returns the message data parsed as JSON."""
        return json.loads(self.data)


class TextureAgentClient:
    """Client for the Texture Agent Service.

    Endpoints:
      - POST /pipeline                         (start pipeline)
      - POST /pipeline/upload-usd              (upload USD, returns session_id)
      - GET  /pipeline/{session_id}/events     (SSE stream)
      - GET  /pipeline/{session_id}/status     (polling status)
      - GET  /pipeline/{session_id}/results    (final results)
      - POST /pipeline/{session_id}/cancel     (cancel run)
      - POST /pipeline/{session_id}/regenerate (re-run specific steps)
      - GET  /artifacts/{session_id}/materials (materials JSON)
      - GET  /artifacts/{session_id}/textures  (textures ZIP)
      - GET  /artifacts/{session_id}/output (textured USDZ)
      - GET  /artifacts/{session_id}/renders   (renders ZIP)
      - GET  /sessions                         (list sessions)
      - DELETE /sessions/{session_id}          (delete session)
      - GET  /health                           (service health)
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8001",
        timeout_seconds: int = 600,
        token: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self._token = token or os.getenv("TEXTURE_AGENT_TOKEN")
        self._http = requests.Session()
        self._http.headers.update({"User-Agent": "texture-agent-client/1.0"})
        if self._token:
            self._http.headers.update({"Authorization": f"Bearer {self._token}"})

    # -------- Core operations
    def upload_usd(
        self, usd_path: str | None = None, *, s3_uri: str | None = None
    ) -> str:
        """Upload a USD file (or reference one on S3) and create a session.

        Returns:
            The session_id of the created session.
        """
        if not usd_path and not s3_uri:
            raise ValueError("Either usd_path or s3_uri must be provided")

        url = f"{self.base_url}/pipeline/upload-usd"

        if s3_uri:
            response = self._http.post(
                url, data={"s3_uri": s3_uri}, timeout=self.timeout_seconds
            )
            response.raise_for_status()
            return response.json()["session_id"]

        with open(usd_path, "rb") as f:
            files = [
                (
                    "usd_file",
                    (Path(usd_path).name, f, "application/octet-stream"),
                )
            ]
            response = self._http.post(url, files=files, timeout=self.timeout_seconds)
            response.raise_for_status()
            return response.json()["session_id"]

    def start_pipeline(
        self,
        session_id: str | None = None,
        usd_path: str | None = None,
        s3_uri: str | None = None,
        material_textures: dict[str, Any] | None = None,
        user_prompt: str | None = None,
        auto_prompt_enabled: bool | None = None,
    ) -> str:
        """Start the pipeline.

        Args:
            session_id: Existing session ID (from upload_usd)
            usd_path: Path to USD file (if not using session_id or s3_uri)
            s3_uri: S3 URI to a USD file
            material_textures: Per-material prompt/opacity config
            user_prompt: Aesthetic direction for auto-prompt generation
            auto_prompt_enabled: Set False for strict material_textures scope.
                None preserves the service default.

        Returns the session_id of the started run.
        """
        url = f"{self.base_url}/pipeline"
        files: list[tuple[str, tuple[str, object, str]]] = []
        file_handles: list = []

        if usd_path:
            uf = open(usd_path, "rb")
            file_handles.append(uf)
            files.append(
                (
                    "usd_file",
                    (Path(usd_path).name, uf, "application/octet-stream"),
                )
            )

        data: dict[str, str] = {}
        if session_id:
            data["session_id"] = session_id
        if s3_uri:
            data["s3_uri"] = s3_uri
        if material_textures:
            data["material_textures_json"] = json.dumps(material_textures)
        if user_prompt:
            data["user_prompt"] = user_prompt
        if auto_prompt_enabled is not None:
            data["auto_prompt_enabled"] = "true" if auto_prompt_enabled else "false"

        try:
            response = self._http.post(
                url,
                data=data or None,
                files=files or None,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            result = response.json()
            return result["session_id"]
        finally:
            for fh in file_handles:
                try:
                    fh.close()
                except Exception:
                    pass

    def regenerate(
        self,
        session_id: str,
        steps: list[str],
        material_textures: dict[str, Any] | None = None,
    ) -> dict:
        """Re-run specific pipeline steps.

        Args:
            session_id: Session to regenerate
            steps: List of step names to re-run
            material_textures: Optional material config override
        """
        url = f"{self.base_url}/pipeline/{session_id}/regenerate"
        body: dict[str, Any] = {"steps": steps}
        if material_textures:
            body["material_textures"] = material_textures
        resp = self._http.post(url, json=body, timeout=self.timeout_seconds)
        resp.raise_for_status()
        return resp.json()

    # -------- Monitoring and results
    def stream_events(
        self, session_id: str, request_timeout: int | None = None
    ) -> Generator[SSEMessage, None, None]:
        """Connect to the SSE endpoint and yield parsed SSEMessage objects."""
        url = f"{self.base_url}/pipeline/{session_id}/events"
        headers = {
            "Accept": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
        timeout = request_timeout or max(self.timeout_seconds, 60)
        with self._http.get(url, headers=headers, stream=True, timeout=timeout) as resp:
            resp.raise_for_status()
            buffer_event: str | None = None
            buffer_data_lines: list[str] = []
            buffer_id: str | None = None
            buffer_retry: int | None = None

            def emit_if_any() -> SSEMessage | None:
                if (
                    buffer_event is None
                    and not buffer_data_lines
                    and buffer_id is None
                    and buffer_retry is None
                ):
                    return None
                data_str = "\n".join(buffer_data_lines) if buffer_data_lines else ""
                return SSEMessage(
                    event=buffer_event or "message",
                    data=data_str,
                    id=buffer_id,
                    retry=buffer_retry,
                )

            for raw_line in resp.iter_lines(decode_unicode=True):
                if raw_line is None:
                    continue
                line = raw_line.rstrip("\r")
                if line == "":
                    msg = emit_if_any()
                    if msg:
                        yield msg
                    buffer_event = None
                    buffer_data_lines = []
                    buffer_id = None
                    buffer_retry = None
                    continue

                if line.startswith(":"):
                    continue

                field, sep, value = line.partition(":")
                if sep:
                    value = value.lstrip(" ")
                else:
                    value = ""

                if field == "event":
                    buffer_event = value
                elif field == "data":
                    buffer_data_lines.append(value)
                elif field == "id":
                    buffer_id = value
                elif field == "retry":
                    try:
                        buffer_retry = int(value)
                    except ValueError:
                        buffer_retry = None

            final_msg = emit_if_any()
            if final_msg:
                yield final_msg

    def get_status(self, session_id: str) -> dict:
        url = f"{self.base_url}/pipeline/{session_id}/status"
        resp = self._http.get(url, timeout=self.timeout_seconds)
        resp.raise_for_status()
        return resp.json()

    def get_results(self, session_id: str) -> dict:
        url = f"{self.base_url}/pipeline/{session_id}/results"
        resp = self._http.get(url, timeout=self.timeout_seconds)
        resp.raise_for_status()
        return resp.json()

    def get_event_log(self, session_id: str) -> dict:
        url = f"{self.base_url}/pipeline/{session_id}/event-log"
        resp = self._http.get(url, timeout=self.timeout_seconds)
        resp.raise_for_status()
        return resp.json()

    def cancel(self, session_id: str) -> None:
        url = f"{self.base_url}/pipeline/{session_id}/cancel"
        resp = self._http.post(url, timeout=self.timeout_seconds)
        resp.raise_for_status()

    # -------- Artifact downloads
    def download_materials(self, session_id: str) -> dict:
        """Download discovered materials JSON."""
        url = f"{self.base_url}/artifacts/{session_id}/materials"
        resp = self._http.get(url, timeout=self.timeout_seconds)
        resp.raise_for_status()
        return resp.json()

    def download_textures(self, session_id: str, output_dir: str) -> list[str]:
        """Download all textures as ZIP and extract to output_dir.

        Returns list of extracted file paths.
        """
        import zipfile
        from io import BytesIO

        url = f"{self.base_url}/artifacts/{session_id}/textures"
        resp = self._http.get(url, timeout=self.timeout_seconds * 2)
        resp.raise_for_status()

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        extracted = []
        with zipfile.ZipFile(BytesIO(resp.content)) as zf:
            for member in zf.namelist():
                member_path = Path(member)
                if member_path.is_absolute() or ".." in member_path.parts:
                    continue
                target = (output_path / member).resolve()
                if not target.is_relative_to(output_path.resolve()):
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(zf.read(member))
                extracted.append(str(target))

        return extracted

    def download_output(self, session_id: str, output_path: str) -> str:
        """Download the self-contained USDZ output (USD + textures bundled).

        Returns the local file path.
        """
        url = f"{self.base_url}/artifacts/{session_id}/output"
        resp = self._http.get(url, timeout=self.timeout_seconds * 2)
        resp.raise_for_status()

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(resp.content)
        return str(out)

    def download_renders(self, session_id: str, output_dir: str) -> list[str]:
        """Download all renders as ZIP and extract to output_dir.

        Returns list of extracted file paths.
        """
        import zipfile
        from io import BytesIO

        url = f"{self.base_url}/artifacts/{session_id}/renders"
        resp = self._http.get(url, timeout=self.timeout_seconds * 2)
        resp.raise_for_status()

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        extracted = []
        with zipfile.ZipFile(BytesIO(resp.content)) as zf:
            for member in zf.namelist():
                member_path = Path(member)
                if member_path.is_absolute() or ".." in member_path.parts:
                    continue
                target = (output_path / member).resolve()
                if not target.is_relative_to(output_path.resolve()):
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(zf.read(member))
                extracted.append(str(target))

        return extracted

    # -------- Utilities
    def sessions(self) -> dict:
        url = f"{self.base_url}/sessions"
        resp = self._http.get(url, timeout=self.timeout_seconds)
        resp.raise_for_status()
        return resp.json()

    def delete_session(self, session_id: str) -> None:
        url = f"{self.base_url}/sessions/{session_id}"
        resp = self._http.delete(url, timeout=self.timeout_seconds)
        resp.raise_for_status()

    def health(self) -> dict:
        url = f"{self.base_url}/health"
        resp = self._http.get(url, timeout=self.timeout_seconds)
        resp.raise_for_status()
        return resp.json()

    # -------- Convenience workflow
    def run_and_monitor(
        self,
        usd_path: str | None = None,
        s3_uri: str | None = None,
        material_textures: dict[str, Any] | None = None,
        user_prompt: str | None = None,
        auto_prompt_enabled: bool | None = None,
        upload_first: bool = False,
        print_stream: bool = True,
        reconnect_attempts: int = 3,
        reconnect_backoff_seconds: float = 2.0,
    ) -> tuple[str, dict | None]:
        """High-level helper that starts the pipeline and monitors it.

        Args:
            usd_path: Path to USD file to process (local).
            s3_uri: S3 URI to a USD file.
            material_textures: Per-material texture config.
            user_prompt: Aesthetic direction for auto-prompt generation.
            auto_prompt_enabled: Set False for strict material_textures scope.
                None preserves the service default.
            upload_first: If True, upload USD first via /upload-usd.
            print_stream: Print progress updates to stdout.
            reconnect_attempts: Number of SSE reconnect attempts.
            reconnect_backoff_seconds: Seconds between reconnect attempts.

        Returns (session_id, status_dict_or_none).
        """
        if not usd_path and not s3_uri:
            raise ValueError("Either usd_path or s3_uri must be provided")

        if s3_uri:
            session_id = self.upload_usd(s3_uri=s3_uri)
            if print_stream:
                print(
                    f"Downloaded USD from S3, session: {session_id}",
                    flush=True,
                )
            session_id = self.start_pipeline(
                session_id=session_id,
                material_textures=material_textures,
                user_prompt=user_prompt,
                auto_prompt_enabled=auto_prompt_enabled,
            )
        elif upload_first:
            session_id = self.upload_usd(usd_path)
            if print_stream:
                print(f"Uploaded USD, session: {session_id}", flush=True)
            session_id = self.start_pipeline(
                session_id=session_id,
                material_textures=material_textures,
                user_prompt=user_prompt,
                auto_prompt_enabled=auto_prompt_enabled,
            )
        else:
            session_id = self.start_pipeline(
                usd_path=usd_path,
                material_textures=material_textures,
                user_prompt=user_prompt,
                auto_prompt_enabled=auto_prompt_enabled,
            )

        if print_stream:
            print(f"Started session: {session_id}", flush=True)

        # Try SSE; if it fails, fall back to polling.
        attempts_left = reconnect_attempts
        saw_done = False
        while attempts_left >= 0 and not saw_done:
            try:
                for msg in self.stream_events(session_id):
                    if msg.event == "ping":
                        continue
                    if msg.event == "progress":
                        try:
                            payload = msg.json()
                        except Exception:
                            payload = {"raw": msg.data}
                        if print_stream:
                            step = payload.get("step")
                            state = payload.get("state")
                            overall = payload.get("overall_percent")
                            message = payload.get("message")
                            print(
                                f"[{step}] {state} overall={overall}% {message or ''}".rstrip(),
                                flush=True,
                            )
                    elif msg.event == "done":
                        saw_done = True
                        break
                if not saw_done:
                    break
            except Exception as e:
                if attempts_left == 0:
                    if print_stream:
                        print(
                            f"SSE failed, falling back to polling: {e}",
                            flush=True,
                        )
                    break
                if print_stream:
                    print(
                        f"SSE error ({e}), retrying in {reconnect_backoff_seconds}s...",
                        flush=True,
                    )
                time.sleep(reconnect_backoff_seconds)
                attempts_left -= 1

        if not saw_done:
            if print_stream:
                print("Polling status...", flush=True)
            max_polls = 300  # 10 minutes at 2s intervals
            for _ in range(max_polls):
                status = self.get_status(session_id)
                st = status.get("status")
                if print_stream:
                    overall = (
                        status.get("overall_percent") or status.get("progress") or "-"
                    )
                    print(f"status={st} overall={overall}", flush=True)
                if st in {"completed", "failed", "cancelled"}:
                    break
                time.sleep(2)
            else:
                if print_stream:
                    print("Timed out waiting for pipeline completion", flush=True)

        try:
            status = self.get_status(session_id)
        except Exception:
            status = None
        return session_id, status


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Texture Agent Service client")
    parser.add_argument(
        "--base-url",
        default="http://localhost:8001",
        help="Service base URL",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Bearer token (or set TEXTURE_AGENT_TOKEN)",
    )
    parser.add_argument(
        "--material-textures",
        default=None,
        help='Per-material config as JSON string, e.g. \'{"Steel": {"prompt": "rusted steel", "opacity": 0.85}}\'',
    )
    parser.add_argument(
        "--user-prompt",
        default=None,
        help="Aesthetic direction for auto-prompt generation (e.g. 'old and weathered')",
    )
    parser.add_argument(
        "--disable-auto-prompt",
        action="store_true",
        help=(
            "Do not auto-generate prompts for materials missing from "
            "--material-textures"
        ),
    )
    parser.add_argument(
        "--upload-first",
        action="store_true",
        help="Upload USD via /upload-usd before starting pipeline",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print streaming updates",
    )

    # USD source: local file or S3 URI
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "usd_path", nargs="?", default=None, help="Path to local USD file"
    )
    source.add_argument(
        "--s3-uri",
        default=None,
        help="S3 URI to a USD file (e.g. s3://bucket/path/scene.usdz)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    client = TextureAgentClient(base_url=args.base_url, token=args.token)

    material_textures = None
    if args.material_textures:
        material_textures = json.loads(args.material_textures)

    session_id, status = client.run_and_monitor(
        usd_path=args.usd_path,
        s3_uri=args.s3_uri,
        material_textures=material_textures,
        user_prompt=args.user_prompt,
        auto_prompt_enabled=False if args.disable_auto_prompt else None,
        upload_first=args.upload_first,
        print_stream=not args.quiet,
    )

    print(f"\nSession: {session_id}")
    if status is not None:
        print(f"Pipeline status: {status['status']}")
        print("\nArtifacts:")
        print(f"- Pipeline Status:  {client.base_url}/pipeline/{session_id}/status")
        print(f"- Materials JSON:   {client.base_url}/artifacts/{session_id}/materials")
        print(f"- Textures ZIP:     {client.base_url}/artifacts/{session_id}/textures")
        print(f"- Output USDZ:      {client.base_url}/artifacts/{session_id}/output")
        print(f"- Renders ZIP:      {client.base_url}/artifacts/{session_id}/renders")
    else:
        print("No results available yet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
