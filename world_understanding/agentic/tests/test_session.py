# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for session management."""

import json

import pytest

from world_understanding.agentic.session import SessionManager


class TestSessionManager:
    """Tests for SessionManager class."""

    def test_create_new_session(self, tmp_path):
        """Test creating a new session with auto-generated ID."""
        session = SessionManager.create(base_dir=tmp_path, project_name="test_project")

        # Verify session was created
        assert session.session_id is not None
        assert len(session.session_id) == 36  # UUID format
        assert session.project_name == "test_project"
        assert session.session_dir.exists()
        assert session.session_dir.parent == tmp_path
        assert session.session_dir.name.startswith(".")

        # Verify metadata
        assert session.metadata["session_id"] == session.session_id
        assert session.metadata["project_name"] == "test_project"
        assert "created_at" in session.metadata

    def test_create_session_with_custom_id(self, tmp_path):
        """Test creating a session with a specific ID."""
        custom_id = "my-custom-session-123"

        session = SessionManager.create(
            base_dir=tmp_path, project_name="test_project", session_id=custom_id
        )

        assert session.session_id == custom_id
        assert session.session_dir == tmp_path / f".{custom_id}"
        assert session.session_dir.exists()

    def test_from_id_existing_session(self, tmp_path):
        """Test loading an existing session by ID."""
        # Create a session first
        original = SessionManager.create(
            base_dir=tmp_path,
            project_name="test_project",
            session_id="test-session-456",
        )
        original.save_metadata()

        # Load it by ID
        loaded = SessionManager.from_id(
            session_id="test-session-456", base_dir=tmp_path
        )

        assert loaded.session_id == original.session_id
        assert loaded.session_dir == original.session_dir
        assert loaded.metadata["session_id"] == original.session_id

    def test_from_id_nonexistent_session(self, tmp_path):
        """Test loading a nonexistent session raises error."""
        with pytest.raises(FileNotFoundError, match="Session directory not found"):
            SessionManager.from_id(session_id="nonexistent-session", base_dir=tmp_path)

    def test_get_subdir_creates_directory(self, tmp_path):
        """Test get_subdir creates subdirectories."""
        session = SessionManager.create(base_dir=tmp_path)

        dataset_dir = session.get_subdir("dataset")
        assert dataset_dir.exists()
        assert dataset_dir.parent == session.session_dir
        assert dataset_dir.name == "dataset"

        # Test nested subdirectory
        iter_dir = session.get_subdir("iterations/iteration_1")
        assert iter_dir.exists()
        assert iter_dir.name == "iteration_1"
        assert iter_dir.parent.name == "iterations"

    def test_get_subdir_no_create(self, tmp_path):
        """Test get_subdir with create=False."""
        session = SessionManager.create(base_dir=tmp_path)

        # Get subdirectory without creating
        output_dir = session.get_subdir("output", create=False)
        assert not output_dir.exists()
        assert output_dir.parent == session.session_dir

    def test_get_file(self, tmp_path):
        """Test get_file returns correct path."""
        session = SessionManager.create(base_dir=tmp_path)

        config_file = session.get_file("config.yaml")
        assert config_file.parent == session.session_dir
        assert config_file.name == "config.yaml"

        # Test nested file path
        output_file = session.get_file("output/result.json")
        assert output_file.name == "result.json"
        assert output_file.parent.name == "output"

    def test_save_and_load_metadata(self, tmp_path):
        """Test saving and loading session metadata."""
        # Create session with custom metadata
        session = SessionManager.create(
            base_dir=tmp_path,
            project_name="test_project",
            metadata={"custom_field": "custom_value"},
        )

        # Save metadata
        session.save_metadata()

        # Verify metadata file exists
        metadata_file = session.session_dir / ".metadata.json"
        assert metadata_file.exists()

        # Load and verify contents
        with open(metadata_file, encoding="utf-8") as f:
            saved_metadata = json.load(f)

        assert saved_metadata["session_id"] == session.session_id
        assert saved_metadata["project_name"] == "test_project"
        assert saved_metadata["custom_field"] == "custom_value"

        # Load session from ID and verify metadata
        loaded_session = SessionManager.from_id(
            session_id=session.session_id, base_dir=tmp_path
        )
        assert loaded_session.metadata["custom_field"] == "custom_value"

    def test_update_metadata(self, tmp_path):
        """Test updating session metadata."""
        session = SessionManager.create(base_dir=tmp_path)

        # Update metadata
        session.update_metadata(status="running", num_predictions=42, score=0.95)

        # Verify metadata was updated
        assert session.metadata["status"] == "running"
        assert session.metadata["num_predictions"] == 42
        assert session.metadata["score"] == 0.95

        # Verify it was saved to disk
        metadata_file = session.session_dir / ".metadata.json"
        assert metadata_file.exists()

        with open(metadata_file, encoding="utf-8") as f:
            saved = json.load(f)
        assert saved["status"] == "running"
        assert saved["num_predictions"] == 42

    def test_list_sessions_empty(self, tmp_path):
        """Test listing sessions in empty directory."""
        sessions = SessionManager.list_sessions(tmp_path)
        assert sessions == []

    def test_list_sessions_multiple(self, tmp_path):
        """Test listing multiple sessions."""
        # Create multiple sessions
        session1 = SessionManager.create(
            base_dir=tmp_path, project_name="project_a", session_id="session-001"
        )
        session1.save_metadata()

        session2 = SessionManager.create(
            base_dir=tmp_path, project_name="project_b", session_id="session-002"
        )
        session2.save_metadata()

        # List sessions
        sessions = SessionManager.list_sessions(tmp_path)

        assert len(sessions) == 2
        session_ids = {s["session_id"] for s in sessions}
        assert "session-001" in session_ids
        assert "session-002" in session_ids

        # Verify metadata is included
        project_names = {s["project_name"] for s in sessions}
        assert "project_a" in project_names
        assert "project_b" in project_names

    def test_list_sessions_ignores_non_session_dirs(self, tmp_path):
        """Test that list_sessions ignores non-session directories."""
        # Create a session
        session = SessionManager.create(base_dir=tmp_path, session_id="real-session")
        session.save_metadata()

        # Create some non-session directories
        (tmp_path / "regular_dir").mkdir()
        (tmp_path / "another_dir").mkdir()

        # List sessions - should only find the one with prefix
        sessions = SessionManager.list_sessions(tmp_path)

        assert len(sessions) == 1
        assert sessions[0]["session_id"] == "real-session"

    def test_session_with_custom_prefix(self, tmp_path):
        """Test creating session with custom prefix."""
        session = SessionManager.create(
            base_dir=tmp_path, project_name="test", prefix="session_"
        )

        assert session.session_dir.name.startswith("session_")
        assert session.session_dir.exists()

    def test_repr_and_str(self, tmp_path):
        """Test string representations."""
        session = SessionManager.create(
            base_dir=tmp_path, project_name="test_project", session_id="test-123"
        )

        # Test repr
        repr_str = repr(session)
        assert "SessionManager" in repr_str
        assert "test-123" in repr_str

        # Test str
        str_str = str(session)
        assert "test-123" in str_str
        assert "test_project" in str_str
