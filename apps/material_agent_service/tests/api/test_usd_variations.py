# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for USD file variation support (.usd, .usda, .usdc, .usdz).

Ensures the service correctly handles all supported USD file formats:
- Upload validation (accepts valid extensions, rejects invalid)
- File persistence (files saved with correct extension)
- Input file discovery (finds scene.* with any valid extension)
"""

import pytest

# Minimal valid USD content for each format
# Note: All binary/ascii formats start with the same signature for testing
USD_ASCII_CONTENT = b"#usda 1.0\n"
USD_BINARY_STUB = b"#usdc 1.0\n"  # Simplified - real usdc is binary


@pytest.mark.api
class TestUsdExtensionValidation:
    """Test USD file extension validation at upload time."""

    @pytest.mark.parametrize(
        "extension",
        [".usd", ".usda", ".usdc", ".usdz"],
        ids=["usd", "usda", "usdc", "usdz"],
    )
    async def test_accepts_valid_usd_extension(self, client, extension):
        """Test that all valid USD extensions are accepted."""
        filename = f"scene{extension}"
        files = {"usd_file": (filename, USD_ASCII_CONTENT, "application/octet-stream")}
        data = {"camera_views": "+x+y+z", "user_email": "test@example.com"}

        response = await client.post("/pipeline", files=files, data=data)

        assert response.status_code == 202, f"Failed for extension {extension}"
        body = response.json()
        assert "session_id" in body
        assert body["status"] == "pending"

    @pytest.mark.parametrize(
        "extension",
        [".usd", ".usda", ".usdc", ".usdz"],
        ids=["usd", "usda", "usdc", "usdz"],
    )
    async def test_upload_usd_endpoint_accepts_valid_extensions(
        self, client, extension
    ):
        """Test /upload-usd endpoint accepts all valid USD extensions."""
        filename = f"model{extension}"
        files = {"usd_file": (filename, USD_ASCII_CONTENT, "application/octet-stream")}

        response = await client.post("/pipeline/upload-usd", files=files)

        assert response.status_code == 201, f"Failed for extension {extension}"
        body = response.json()
        assert "session_id" in body
        assert body["status"] == "ready"

    @pytest.mark.parametrize(
        "extension,description",
        [
            (".obj", "Wavefront OBJ format"),
            (".fbx", "FBX format"),
            (".gltf", "glTF format"),
            (".glb", "GLB binary format"),
            (".abc", "Alembic format"),
            (".blend", "Blender format"),
            (".3ds", "3DS Max format"),
            (".txt", "Plain text file"),
            (".json", "JSON file"),
            ("", "No extension"),
        ],
        ids=[
            "obj",
            "fbx",
            "gltf",
            "glb",
            "alembic",
            "blender",
            "3ds",
            "txt",
            "json",
            "no_ext",
        ],
    )
    async def test_rejects_invalid_file_extension(self, client, extension, description):
        """Test that invalid file extensions are rejected with 400 error."""
        filename = f"scene{extension}" if extension else "scene"
        files = {"usd_file": (filename, USD_ASCII_CONTENT, "application/octet-stream")}

        response = await client.post(
            "/pipeline", files=files, data={"user_email": "test@example.com"}
        )

        assert response.status_code == 400, f"Should reject {description}"
        assert "Invalid file type" in response.json()["detail"]

    @pytest.mark.parametrize(
        "extension",
        [".USD", ".USDA", ".USDC", ".USDZ", ".Usd", ".UsDA"],
        ids=["USD", "USDA", "USDC", "USDZ", "Usd_mixed", "UsDA_mixed"],
    )
    async def test_accepts_case_insensitive_extensions(self, client, extension):
        """Test that USD extensions are accepted case-insensitively."""
        filename = f"scene{extension}"
        files = {"usd_file": (filename, USD_ASCII_CONTENT, "application/octet-stream")}

        response = await client.post(
            "/pipeline", files=files, data={"user_email": "test@example.com"}
        )

        assert response.status_code == 202, f"Failed for extension {extension}"


@pytest.mark.api
class TestUsdFilePersistence:
    """Test that uploaded USD files are saved with correct extensions."""

    @pytest.mark.parametrize(
        "extension",
        [".usd", ".usda", ".usdc", ".usdz"],
        ids=["usd", "usda", "usdc", "usdz"],
    )
    async def test_file_saved_with_original_extension(
        self, client, session_manager, extension
    ):
        """Test that uploaded files preserve their original extension."""
        filename = f"my_model{extension}"
        files = {"usd_file": (filename, USD_ASCII_CONTENT, "application/octet-stream")}

        response = await client.post(
            "/pipeline", files=files, data={"user_email": "test@example.com"}
        )
        assert response.status_code == 202

        session_id = response.json()["session_id"]
        session_dir = session_manager.get_session_dir(session_id)

        # File should be saved as scene{extension}
        expected_path = session_dir / "input" / f"scene{extension}"
        assert expected_path.exists(), f"Expected file at {expected_path}"

    async def test_mixed_case_extension_normalized_to_lowercase(
        self, client, session_manager
    ):
        """Test that mixed-case extensions are normalized to lowercase."""
        filename = "model.USDA"
        files = {"usd_file": (filename, USD_ASCII_CONTENT, "application/octet-stream")}

        response = await client.post(
            "/pipeline", files=files, data={"user_email": "test@example.com"}
        )
        assert response.status_code == 202

        session_id = response.json()["session_id"]
        session_dir = session_manager.get_session_dir(session_id)

        # Should be saved with lowercase extension
        expected_path = session_dir / "input" / "scene.usda"
        assert expected_path.exists(), "Extension should be normalized to lowercase"


@pytest.mark.api
class TestExistingSessionUsdLookup:
    """Test USD file discovery for existing sessions (2-step upload flow)."""

    @pytest.mark.parametrize(
        "extension",
        [".usd", ".usda", ".usdc", ".usdz"],
        ids=["usd", "usda", "usdc", "usdz"],
    )
    async def test_pipeline_finds_existing_session_usd(self, client, extension):
        """Test that /pipeline finds USD from /upload-usd for all extensions."""
        # Step 1: Upload USD via immediate upload endpoint
        filename = f"asset{extension}"
        files = {"usd_file": (filename, USD_ASCII_CONTENT, "application/octet-stream")}

        upload_response = await client.post("/pipeline/upload-usd", files=files)
        assert upload_response.status_code == 201
        session_id = upload_response.json()["session_id"]

        # Step 2: Create pipeline using existing session (no new file upload)
        data = {
            "session_id": session_id,
            "camera_views": "+x+y+z",
            "user_email": "test@example.com",
        }

        pipeline_response = await client.post("/pipeline", data=data)

        assert pipeline_response.status_code == 202, (
            f"Failed to find existing USD with {extension} extension"
        )
        assert pipeline_response.json()["session_id"] == session_id


@pytest.mark.unit
class TestFindInputUsdFunction:
    """Unit tests for the _find_input_usd utility function."""

    def test_finds_usd_file(self, tmp_path):
        """Test finding .usd file in session directory."""
        from ...service.routers.pipeline_router import _find_input_usd

        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "scene.usd").write_bytes(USD_ASCII_CONTENT)

        result = _find_input_usd(tmp_path)

        assert result is not None
        assert result.name == "scene.usd"

    def test_finds_usda_file(self, tmp_path):
        """Test finding .usda file in session directory."""
        from ...service.routers.pipeline_router import _find_input_usd

        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "scene.usda").write_bytes(USD_ASCII_CONTENT)

        result = _find_input_usd(tmp_path)

        assert result is not None
        assert result.name == "scene.usda"

    def test_finds_usdc_file(self, tmp_path):
        """Test finding .usdc file in session directory."""
        from ...service.routers.pipeline_router import _find_input_usd

        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "scene.usdc").write_bytes(USD_BINARY_STUB)

        result = _find_input_usd(tmp_path)

        assert result is not None
        assert result.name == "scene.usdc"

    def test_finds_usdz_file(self, tmp_path):
        """Test finding .usdz file in session directory."""
        from ...service.routers.pipeline_router import _find_input_usd

        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "scene.usdz").write_bytes(USD_ASCII_CONTENT)

        result = _find_input_usd(tmp_path)

        assert result is not None
        assert result.name == "scene.usdz"

    def test_returns_none_when_no_usd_found(self, tmp_path):
        """Test that None is returned when no USD file exists."""
        from ...service.routers.pipeline_router import _find_input_usd

        input_dir = tmp_path / "input"
        input_dir.mkdir()
        # No USD files present

        result = _find_input_usd(tmp_path)

        assert result is None

    def test_returns_none_when_input_dir_missing(self, tmp_path):
        """Test that None is returned when input directory doesn't exist."""
        from ...service.routers.pipeline_router import _find_input_usd

        # No input directory at all
        result = _find_input_usd(tmp_path)

        assert result is None

    def test_priority_order_usd_first(self, tmp_path):
        """Test that .usd is found first when multiple formats exist."""
        from ...service.routers.pipeline_router import _find_input_usd

        input_dir = tmp_path / "input"
        input_dir.mkdir()
        # Create multiple formats
        (input_dir / "scene.usd").write_bytes(USD_ASCII_CONTENT)
        (input_dir / "scene.usda").write_bytes(USD_ASCII_CONTENT)
        (input_dir / "scene.usdc").write_bytes(USD_BINARY_STUB)
        (input_dir / "scene.usdz").write_bytes(USD_ASCII_CONTENT)

        result = _find_input_usd(tmp_path)

        # .usd should be found first based on the search order
        assert result is not None
        assert result.name == "scene.usd"


@pytest.mark.unit
class TestConfigAllowedExtensions:
    """Test that config correctly defines allowed USD extensions."""

    def test_allowed_extensions_include_all_usd_formats(self):
        """Test that ServiceConfig includes all USD variations."""
        from ...service.config import ServiceConfig

        config = ServiceConfig()

        expected = {".usd", ".usda", ".usdc", ".usdz"}
        assert config.allowed_extensions == expected

    def test_allowed_extensions_is_case_sensitive_set(self):
        """Test that allowed extensions are stored lowercase for matching."""
        from ...service.config import ServiceConfig

        config = ServiceConfig()

        for ext in config.allowed_extensions:
            assert ext == ext.lower(), f"Extension {ext} should be lowercase"
            assert ext.startswith("."), f"Extension {ext} should start with dot"
