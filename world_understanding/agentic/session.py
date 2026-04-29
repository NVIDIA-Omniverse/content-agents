# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Session management for agent workflows.

This module provides session-based state management for agent workflows,
enabling reproducible runs, debugging, and state persistence.

Key concepts:
- Session ID: Unique identifier for a workflow run
- Session directory: Isolated workspace for all session artifacts
- Path resolution: Automatic path derivation within session structure

Example:
    ```python
    from world_understanding.agentic.session import SessionManager

    # Create new session
    session = SessionManager.create(
        base_dir=Path("outputs"),
        project_name="my_project"
    )

    # Or reuse existing session
    session = SessionManager.from_id(
        session_id="abc-123-def",
        base_dir=Path("outputs")
    )

    # Get session paths
    print(session.session_dir)  # outputs/.abc-123-def
    print(session.get_subdir("dataset"))  # outputs/.abc-123-def/dataset
    print(session.get_subdir("iterations/iteration_1"))  # ...
    ```
"""

import logging
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class SessionManager:
    """Manages session state and directory structure for agent workflows.

    A session represents a single execution of an agent workflow with its own
    isolated workspace. Sessions can be resumed by providing the same session_id.

    Session directory structure:
        .{session_id}/
            dataset/        # Input dataset files
            iterations/     # Iteration outputs (for iterative workflows)
                iteration_1/
                iteration_2/
            output/         # Final outputs
            logs/           # Session logs
            .metadata.json  # Session metadata

    Attributes:
        session_id: Unique identifier for this session
        session_dir: Root directory for all session files
        project_name: Optional project name for metadata
        metadata: Session metadata (creation time, project info, etc.)
    """

    def __init__(
        self,
        session_id: str,
        session_dir: Path,
        project_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """Initialize a session manager.

        Note: Use SessionManager.create() or SessionManager.from_id() instead
        of calling this constructor directly.

        Args:
            session_id: Unique session identifier
            session_dir: Root directory for session files
            project_name: Optional project name
            metadata: Optional session metadata
        """
        self.session_id = session_id
        self.session_dir = Path(session_dir).resolve()
        self.project_name = project_name or "unknown_project"
        self.metadata = metadata or {}

        # Ensure metadata has required fields
        if "session_id" not in self.metadata:
            self.metadata["session_id"] = session_id
        if "project_name" not in self.metadata:
            self.metadata["project_name"] = self.project_name

    @classmethod
    def create(
        cls,
        base_dir: Path,
        project_name: str | None = None,
        session_id: str | None = None,
        prefix: str = ".",
        metadata: dict[str, Any] | None = None,
    ) -> "SessionManager":
        """Create a new session or reuse existing one.

        Args:
            base_dir: Base directory where session directories are created
            project_name: Optional project name for metadata
            session_id: Optional session ID to reuse; if None, generates new UUID
            prefix: Prefix for session directory (default: "." for hidden dirs)
            metadata: Optional additional metadata to store

        Returns:
            SessionManager instance

        Example:
            ```python
            # Create new session
            session = SessionManager.create(
                base_dir=Path("outputs"),
                project_name="material_assignment"
            )

            # Reuse existing session
            session = SessionManager.create(
                base_dir=Path("outputs"),
                session_id="abc-123-def"
            )
            ```
        """
        # Generate or validate session_id
        if session_id is None:
            session_id = str(uuid.uuid4())
            logger.info(f"Generated new session ID: {session_id}")
        else:
            logger.info(f"Using provided session ID: {session_id}")

        # Create session directory name
        session_dir_name = f"{prefix}{session_id}"
        session_dir = Path(base_dir) / session_dir_name

        # Create session directory if it doesn't exist
        session_dir.mkdir(parents=True, exist_ok=True)

        # Initialize metadata
        import datetime

        session_metadata = metadata or {}
        session_metadata.update(
            {
                "session_id": session_id,
                "project_name": project_name or "unknown_project",
                "created_at": datetime.datetime.now().isoformat(),
                "base_dir": str(base_dir),
                "session_dir": str(session_dir),
            }
        )

        return cls(
            session_id=session_id,
            session_dir=session_dir,
            project_name=project_name,
            metadata=session_metadata,
        )

    @classmethod
    def from_id(
        cls,
        session_id: str,
        base_dir: Path,
        prefix: str = ".",
        project_name: str | None = None,
    ) -> "SessionManager":
        """Load an existing session by ID.

        Args:
            session_id: Session ID to load
            base_dir: Base directory where session exists
            prefix: Prefix for session directory (default: ".")
            project_name: Optional project name override

        Returns:
            SessionManager instance

        Raises:
            FileNotFoundError: If session directory doesn't exist

        Example:
            ```python
            # Load existing session
            session = SessionManager.from_id(
                session_id="abc-123-def",
                base_dir=Path("outputs")
            )
            ```
        """
        session_dir_name = f"{prefix}{session_id}"
        session_dir = Path(base_dir) / session_dir_name

        if not session_dir.exists():
            raise FileNotFoundError(
                f"Session directory not found: {session_dir}. "
                f"Session ID '{session_id}' does not exist in {base_dir}."
            )

        # Try to load metadata if it exists
        metadata_file = session_dir / ".metadata.json"
        metadata = {}
        if metadata_file.exists():
            import json

            try:
                with open(metadata_file, encoding="utf-8") as f:
                    metadata = json.load(f)
                logger.debug(f"Loaded session metadata from {metadata_file}")
            except Exception as e:
                logger.warning(f"Failed to load session metadata: {e}")

        # Use project_name from metadata or parameter
        if not project_name and "project_name" in metadata:
            project_name = metadata["project_name"]

        logger.info(f"Loaded existing session: {session_id}")

        return cls(
            session_id=session_id,
            session_dir=session_dir,
            project_name=project_name,
            metadata=metadata,
        )

    def get_subdir(self, subdir: str, create: bool = True) -> Path:
        """Get a subdirectory within the session.

        Args:
            subdir: Subdirectory path relative to session root
            create: Whether to create the directory if it doesn't exist

        Returns:
            Absolute path to the subdirectory

        Example:
            ```python
            dataset_dir = session.get_subdir("dataset")
            iter1_dir = session.get_subdir("iterations/iteration_1")
            ```
        """
        path = self.session_dir / subdir
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path

    def get_file(self, filepath: str) -> Path:
        """Get a file path within the session.

        Args:
            filepath: File path relative to session root

        Returns:
            Absolute path to the file

        Example:
            ```python
            config_file = session.get_file("config.yaml")
            output_file = session.get_file("output/result.json")
            ```
        """
        return self.session_dir / filepath

    def save_metadata(self) -> None:
        """Save session metadata to disk.

        Writes metadata to .metadata.json in the session directory.
        """
        metadata_file = self.session_dir / ".metadata.json"

        import json

        try:
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(self.metadata, f, indent=2)
            logger.debug(f"Saved session metadata to {metadata_file}")
        except Exception as e:
            logger.warning(f"Failed to save session metadata: {e}")

    def update_metadata(self, **kwargs: Any) -> None:
        """Update session metadata with new key-value pairs.

        Args:
            **kwargs: Key-value pairs to add to metadata

        Example:
            ```python
            session.update_metadata(
                status="completed",
                num_predictions=42,
                final_score=0.95
            )
            ```
        """
        self.metadata.update(kwargs)
        self.save_metadata()

    @staticmethod
    def list_sessions(base_dir: Path, prefix: str = ".") -> list[dict[str, Any]]:
        """List all sessions in a base directory.

        Args:
            base_dir: Directory to search for sessions
            prefix: Session directory prefix (default: ".")

        Returns:
            List of session info dicts with session_id, path, and metadata

        Example:
            ```python
            sessions = SessionManager.list_sessions(Path("outputs"))
            for session in sessions:
                print(f"{session['session_id']}: {session['project_name']}")
            ```
        """
        base_path = Path(base_dir)
        if not base_path.exists():
            return []

        sessions = []

        # Find all directories matching the session pattern
        for item in base_path.iterdir():
            if not item.is_dir():
                continue

            # Check if it matches session naming pattern
            if item.name.startswith(prefix):
                # Extract session_id by removing prefix
                potential_session_id = item.name[len(prefix) :]

                # Try to load metadata
                metadata_file = item / ".metadata.json"
                metadata = {}
                if metadata_file.exists():
                    import json

                    try:
                        with open(metadata_file, encoding="utf-8") as f:
                            metadata = json.load(f)
                    except Exception:
                        pass

                sessions.append(
                    {
                        "session_id": potential_session_id,
                        "session_dir": str(item),
                        "project_name": metadata.get("project_name", "unknown"),
                        "created_at": metadata.get("created_at"),
                        "metadata": metadata,
                    }
                )

        # Sort by creation time (newest first)
        sessions.sort(key=lambda s: s.get("created_at", ""), reverse=True)

        return sessions

    def __repr__(self) -> str:
        """String representation of session."""
        return f"SessionManager(id={self.session_id}, dir={self.session_dir})"

    def __str__(self) -> str:
        """Human-readable session info."""
        return f"Session {self.session_id} ({self.project_name}) @ {self.session_dir}"
