# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for custom materials library upload and validation.

Tests the custom materials ZIP upload functionality:
- Valid ZIP with materials.yaml and USD library
- Icon serving for custom materials
- Validation error handling
- Integration with pipeline execution
"""

import asyncio
import zipfile
from io import BytesIO
from pathlib import Path

import pytest


def _create_materials_zip(
    materials_yaml: str,
    usda_content: str = "#usda 1.0\n",
    subdirectory: str | None = "default_materials",
    icons: dict[str, bytes] | None = None,
) -> BytesIO:
    """Create a materials ZIP file in memory.

    Args:
        materials_yaml: Contents of materials.yaml
        usda_content: Contents of USD library file
        subdirectory: Optional subdirectory name (None for flat structure)
        icons: Optional dict of {icon_relative_path: png_bytes}

    Returns:
        BytesIO containing the ZIP file
    """
    buffer = BytesIO()

    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        prefix = f"{subdirectory}/" if subdirectory else ""

        # Add materials.yaml
        zf.writestr(f"{prefix}materials.yaml", materials_yaml)

        # Add USD library
        zf.writestr(f"{prefix}materials_libs.usda", usda_content)

        # Add icons if provided
        if icons:
            for icon_path, icon_data in icons.items():
                zf.writestr(f"{prefix}{icon_path}", icon_data)

    buffer.seek(0)
    return buffer


# Minimal valid materials.yaml content
VALID_MATERIALS_YAML = """materials:
  library_path: "materials_libs.usda"
  entries:
    - name: "Test_Metal"
      description: "A shiny test metal"
      binding: "/World/Looks/Test_Metal"
      icon: "thumbs/Test_Metal.png"
    - name: "Test_Plastic"
      description: "A colorful test plastic"
      binding: "/World/Looks/Test_Plastic"
      icon: "thumbs/Test_Plastic.png"
"""

# Minimal 1x1 PNG (valid PNG header + IHDR + IEND)
MINIMAL_PNG = (
    b"\x89PNG\r\n\x1a\n"  # PNG signature
    b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"  # IHDR
    b"\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N"  # IDAT
    b"\x00\x00\x00\x00IEND\xaeB`\x82"  # IEND
)


@pytest.mark.api
class TestCustomMaterialsUpload:
    """Test custom materials ZIP upload functionality."""

    async def test_create_pipeline_with_valid_materials_zip(self, client):
        """Test creating a pipeline with a valid custom materials ZIP."""
        # Create USD file
        usd_content = b"#usda 1.0\n"

        # Create materials ZIP with icons
        materials_zip = _create_materials_zip(
            VALID_MATERIALS_YAML,
            subdirectory="default_materials",
            icons={
                "thumbs/Test_Metal.png": MINIMAL_PNG,
                "thumbs/Test_Plastic.png": MINIMAL_PNG,
            },
        )

        files = [
            ("usd_file", ("scene.usda", usd_content, "application/octet-stream")),
            (
                "materials_zip",
                ("custom_materials.zip", materials_zip, "application/zip"),
            ),
        ]
        data = {"camera_views": "+x+y+z", "user_email": "test@example.com"}

        response = await client.post("/pipeline", files=files, data=data)

        assert response.status_code == 202
        body = response.json()
        assert "session_id" in body
        assert body["status"] == "pending"

    async def test_create_pipeline_with_flat_zip_structure(self, client):
        """Test creating a pipeline with materials.yaml at ZIP root (no subdirectory)."""
        usd_content = b"#usda 1.0\n"

        # Create materials ZIP without subdirectory
        materials_zip = _create_materials_zip(
            VALID_MATERIALS_YAML,
            subdirectory=None,  # Flat structure
            icons={
                "thumbs/Test_Metal.png": MINIMAL_PNG,
                "thumbs/Test_Plastic.png": MINIMAL_PNG,
            },
        )

        files = [
            ("usd_file", ("scene.usda", usd_content, "application/octet-stream")),
            ("materials_zip", ("flat_materials.zip", materials_zip, "application/zip")),
        ]

        response = await client.post(
            "/pipeline", files=files, data={"user_email": "test@example.com"}
        )

        assert response.status_code == 202
        assert "session_id" in response.json()

    async def test_create_pipeline_rejects_zip_without_materials_yaml(self, client):
        """Test that ZIP without materials.yaml is rejected."""
        usd_content = b"#usda 1.0\n"

        # Create invalid ZIP (no materials.yaml)
        buffer = BytesIO()
        with zipfile.ZipFile(buffer, "w") as zf:
            zf.writestr("some_file.txt", "not a materials config")
        buffer.seek(0)

        files = [
            ("usd_file", ("scene.usda", usd_content, "application/octet-stream")),
            ("materials_zip", ("invalid.zip", buffer, "application/zip")),
        ]

        response = await client.post(
            "/pipeline", files=files, data={"user_email": "test@example.com"}
        )

        assert response.status_code == 400
        assert "materials.yaml" in response.json()["detail"]

    async def test_create_pipeline_rejects_invalid_yaml(self, client):
        """Test that invalid YAML in materials.yaml is rejected."""
        usd_content = b"#usda 1.0\n"

        # Create ZIP with malformed YAML
        materials_zip = _create_materials_zip(
            "this: is: invalid: yaml: {",
            subdirectory="test",
        )

        files = [
            ("usd_file", ("scene.usda", usd_content, "application/octet-stream")),
            ("materials_zip", ("bad_yaml.zip", materials_zip, "application/zip")),
        ]

        response = await client.post(
            "/pipeline", files=files, data={"user_email": "test@example.com"}
        )

        assert response.status_code == 400
        assert "Invalid materials.yaml" in response.json()["detail"]

    async def test_create_pipeline_rejects_missing_library_path(self, client):
        """Test that materials.yaml without library_path is rejected."""
        usd_content = b"#usda 1.0\n"

        # YAML without library_path
        invalid_yaml = """materials:
  entries:
    - name: "Test"
      description: "Missing library_path"
      binding: "/World/Looks/Test"
"""
        materials_zip = _create_materials_zip(invalid_yaml, subdirectory="test")

        files = [
            ("usd_file", ("scene.usda", usd_content, "application/octet-stream")),
            ("materials_zip", ("no_library.zip", materials_zip, "application/zip")),
        ]

        response = await client.post(
            "/pipeline", files=files, data={"user_email": "test@example.com"}
        )

        assert response.status_code == 400
        assert "library_path" in response.json()["detail"]

    async def test_create_pipeline_rejects_missing_usd_library(self, client):
        """Test that materials.yaml referencing non-existent USD file is rejected."""
        usd_content = b"#usda 1.0\n"

        # YAML referencing non-existent file
        bad_path_yaml = """materials:
  library_path: "nonexistent.usda"
  entries:
    - name: "Test"
      description: "Bad library path"
      binding: "/World/Looks/Test"
"""
        # Create ZIP with only materials.yaml (missing the USD file)
        buffer = BytesIO()
        with zipfile.ZipFile(buffer, "w") as zf:
            zf.writestr("test/materials.yaml", bad_path_yaml)
        buffer.seek(0)

        files = [
            ("usd_file", ("scene.usda", usd_content, "application/octet-stream")),
            ("materials_zip", ("bad_path.zip", buffer, "application/zip")),
        ]

        response = await client.post(
            "/pipeline", files=files, data={"user_email": "test@example.com"}
        )

        assert response.status_code == 400
        assert "USD library file not found" in response.json()["detail"]

    async def test_create_pipeline_rejects_empty_entries(self, client):
        """Test that materials.yaml with empty entries list is rejected."""
        usd_content = b"#usda 1.0\n"

        empty_entries_yaml = """materials:
  library_path: "materials_libs.usda"
  entries: []
"""
        materials_zip = _create_materials_zip(empty_entries_yaml, subdirectory="test")

        files = [
            ("usd_file", ("scene.usda", usd_content, "application/octet-stream")),
            ("materials_zip", ("empty.zip", materials_zip, "application/zip")),
        ]

        response = await client.post(
            "/pipeline", files=files, data={"user_email": "test@example.com"}
        )

        assert response.status_code == 400
        assert "non-empty list" in response.json()["detail"]


@pytest.mark.api
class TestCustomMaterialsIcons:
    """Test custom materials icon serving."""

    async def test_session_material_icon_endpoint(self, client):
        """Test that session-specific material icons can be served."""
        # Create pipeline with custom materials including icons
        usd_content = b"#usda 1.0\n"
        materials_zip = _create_materials_zip(
            VALID_MATERIALS_YAML,
            subdirectory="default_materials",
            icons={
                "thumbs/Test_Metal.png": MINIMAL_PNG,
                "thumbs/Test_Plastic.png": MINIMAL_PNG,
            },
        )

        files = [
            ("usd_file", ("scene.usda", usd_content, "application/octet-stream")),
            ("materials_zip", ("materials.zip", materials_zip, "application/zip")),
        ]

        create_response = await client.post(
            "/pipeline", files=files, data={"user_email": "test@example.com"}
        )
        assert create_response.status_code == 202
        session_id = create_response.json()["session_id"]

        # Wait for pipeline to complete (uses stub executor)
        for _ in range(200):
            status = await client.get(f"/pipeline/{session_id}/status")
            if status.json()["status"] == "completed":
                break
            await asyncio.sleep(0.01)

        # Test icon endpoint
        icon_response = await client.get(
            f"/pipeline/sessions/{session_id}/materials/icon/Test_Metal"
        )

        assert icon_response.status_code == 200
        assert icon_response.headers["content-type"] == "image/png"
        assert len(icon_response.content) > 0

    async def test_session_material_icon_not_found(self, client):
        """Test that missing material icon returns 404."""
        # Create pipeline with custom materials
        usd_content = b"#usda 1.0\n"
        materials_zip = _create_materials_zip(
            VALID_MATERIALS_YAML,
            subdirectory="default_materials",
            icons={
                "thumbs/Test_Metal.png": MINIMAL_PNG,
                "thumbs/Test_Plastic.png": MINIMAL_PNG,
            },
        )

        files = [
            ("usd_file", ("scene.usda", usd_content, "application/octet-stream")),
            ("materials_zip", ("materials.zip", materials_zip, "application/zip")),
        ]

        create_response = await client.post(
            "/pipeline", files=files, data={"user_email": "test@example.com"}
        )
        session_id = create_response.json()["session_id"]

        # Wait for pipeline to start
        await asyncio.sleep(0.1)

        # Request non-existent material
        icon_response = await client.get(
            f"/pipeline/sessions/{session_id}/materials/icon/NonExistent_Material"
        )

        assert icon_response.status_code == 404

    async def test_session_material_icon_invalid_session(self, client):
        """Test that invalid session returns 404."""
        response = await client.get(
            "/pipeline/sessions/00000000-0000-0000-0000-000000000000/materials/icon/Test_Metal"
        )

        assert response.status_code == 404


@pytest.mark.api
class TestDefaultMaterialsTemplate:
    """Test the default materials template download endpoint."""

    async def test_materials_template_download(self, client):
        """Test downloading the default materials template ZIP."""
        # Note: This test requires the default_materials.zip to exist
        # In production, this file should be present
        response = await client.get("/materials/template")

        # If template exists, should return ZIP
        if response.status_code == 200:
            assert response.headers["content-type"] == "application/zip"
            assert "default_materials.zip" in response.headers.get(
                "content-disposition", ""
            )
        else:
            # Template not found is also acceptable in test environment
            assert response.status_code == 404


@pytest.mark.api
class TestCustomMaterialsPipelineIntegration:
    """Test custom materials integration with full pipeline."""

    async def test_pipeline_uses_custom_materials(self, client):
        """Test that pipeline correctly uses custom materials configuration."""
        usd_content = b"#usda 1.0\n"
        materials_zip = _create_materials_zip(
            VALID_MATERIALS_YAML,
            subdirectory="default_materials",
            icons={
                "thumbs/Test_Metal.png": MINIMAL_PNG,
                "thumbs/Test_Plastic.png": MINIMAL_PNG,
            },
        )

        files = [
            ("usd_file", ("scene.usda", usd_content, "application/octet-stream")),
            ("materials_zip", ("materials.zip", materials_zip, "application/zip")),
        ]

        response = await client.post(
            "/pipeline", files=files, data={"user_email": "test@example.com"}
        )
        assert response.status_code == 202
        session_id = response.json()["session_id"]

        # Wait for completion
        for _ in range(200):
            status_r = await client.get(f"/pipeline/{session_id}/status")
            if status_r.json()["status"] == "completed":
                break
            await asyncio.sleep(0.01)

        # Verify completed
        final_status = await client.get(f"/pipeline/{session_id}/status")
        assert final_status.json()["status"] == "completed"

    async def test_pipeline_without_materials_uses_defaults(self, client):
        """Test that pipeline without custom materials uses server defaults."""
        usd_content = b"#usda 1.0\n"
        files = [("usd_file", ("scene.usda", usd_content, "application/octet-stream"))]

        response = await client.post(
            "/pipeline", files=files, data={"user_email": "test@example.com"}
        )
        assert response.status_code == 202

        session_id = response.json()["session_id"]

        # Wait for completion
        for _ in range(200):
            status_r = await client.get(f"/pipeline/{session_id}/status")
            if status_r.json()["status"] == "completed":
                break
            await asyncio.sleep(0.01)

        # Verify completed successfully (using default materials)
        final_status = await client.get(f"/pipeline/{session_id}/status")
        assert final_status.json()["status"] == "completed"


@pytest.mark.api
class TestDefaultMaterialsZip:
    """Test using the actual default_materials.zip bundled with the service.

    These tests verify that the provided default_materials.zip template is valid
    and can be used successfully in the pipeline.
    """

    @pytest.fixture
    def default_materials_zip_path(self) -> Path:
        """Get path to the bundled default_materials.zip."""
        from pathlib import Path

        return (
            Path(__file__).parent.parent.parent
            / "materials"
            / "default"
            / "default_materials.zip"
        )

    async def test_default_materials_zip_is_valid(
        self, client, default_materials_zip_path
    ):
        """Test that the bundled default_materials.zip is valid and can be used.

        This verifies:
        - The ZIP file exists
        - It contains valid materials.yaml
        - It contains a valid USD library
        - It can be uploaded successfully
        """
        if not default_materials_zip_path.exists():
            pytest.skip("default_materials.zip not found (expected in CI)")

        usd_content = b"#usda 1.0\n"

        with open(default_materials_zip_path, "rb") as f:
            zip_content = f.read()

        files = [
            ("usd_file", ("scene.usda", usd_content, "application/octet-stream")),
            (
                "materials_zip",
                ("default_materials.zip", zip_content, "application/zip"),
            ),
        ]

        response = await client.post(
            "/pipeline", files=files, data={"user_email": "test@example.com"}
        )

        assert response.status_code == 202, f"Failed to upload: {response.json()}"
        body = response.json()
        assert "session_id" in body
        assert body["status"] == "pending"

    async def test_default_materials_zip_has_no_default_icons(
        self, client, default_materials_zip_path
    ):
        """The bundled default template intentionally ships without icons."""
        if not default_materials_zip_path.exists():
            pytest.skip("default_materials.zip not found (expected in CI)")

        usd_content = b"#usda 1.0\n"

        with open(default_materials_zip_path, "rb") as f:
            zip_content = f.read()

        files = [
            ("usd_file", ("scene.usda", usd_content, "application/octet-stream")),
            (
                "materials_zip",
                ("default_materials.zip", zip_content, "application/zip"),
            ),
        ]

        # Create pipeline
        create_response = await client.post(
            "/pipeline", files=files, data={"user_email": "test@example.com"}
        )
        assert create_response.status_code == 202
        session_id = create_response.json()["session_id"]

        # Wait for completion
        for _ in range(200):
            status = await client.get(f"/pipeline/{session_id}/status")
            if status.json()["status"] == "completed":
                break
            await asyncio.sleep(0.01)

        # The bundled default template is intentionally icon-free.
        icon_response = await client.get(
            f"/pipeline/sessions/{session_id}/materials/icon/Aluminum"
        )

        assert icon_response.status_code == 404

    async def test_default_materials_zip_structure(self, default_materials_zip_path):
        """Test that default_materials.zip has the expected structure.

        Verifies the ZIP contains:
        - materials.yaml (in subdirectory or root)
        - A USD library file (.usda/.usd/.usdc)
        - No bundled PNG thumbnails in the default release template
        """
        if not default_materials_zip_path.exists():
            pytest.skip("default_materials.zip not found (expected in CI)")

        import zipfile

        import yaml

        with zipfile.ZipFile(default_materials_zip_path, "r") as zf:
            names = zf.namelist()

            # Find materials.yaml
            yaml_files = [n for n in names if n.endswith("materials.yaml")]
            assert yaml_files, "materials.yaml not found in ZIP"

            # Find USD library
            usd_files = [n for n in names if n.endswith((".usda", ".usd", ".usdc"))]
            assert usd_files, "No USD library file found in ZIP"

            # The default release template intentionally omits thumbnails.
            png_files = [n for n in names if n.endswith(".png")]
            assert not png_files, "Default template should not bundle PNG icons"

            # Validate materials.yaml content
            yaml_content = zf.read(yaml_files[0]).decode("utf-8")
            data = yaml.safe_load(yaml_content)

            assert "materials" in data, "materials key missing from YAML"
            materials = data["materials"]
            assert "library_path" in materials, "library_path missing"
            assert "entries" in materials, "entries missing"
            assert len(materials["entries"]) > 0, "entries is empty"

            # Verify each entry has required fields
            for entry in materials["entries"]:
                assert "name" in entry, f"Entry missing name: {entry}"
                assert "binding" in entry, f"Entry missing binding: {entry}"
                assert "icon" not in entry, f"Entry should not ship icon: {entry}"
