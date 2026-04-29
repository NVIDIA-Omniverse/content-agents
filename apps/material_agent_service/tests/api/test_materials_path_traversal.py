# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for path traversal protection in materials icon endpoint.

Verifies that:
- Normal icon requests return 200
- Path traversal attempts (../) return 403
- The security check runs before the existence check (no info leak)
"""

import pytest

# Minimal 1x1 PNG
MINIMAL_PNG = (
    b"\x89PNG\r\n\x1a\n"
    b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"
    b"\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N"
    b"\x00\x00\x00\x00IEND\xaeB`\x82"
)


@pytest.fixture()
def _library_with_icon(tmp_path, monkeypatch):
    """Register a temporary material library with a real icon file on disk."""
    from ...service.config import MaterialLibrary, config

    icons_dir = tmp_path / "thumbs"
    icons_dir.mkdir()
    icon_file = icons_dir / "Metal.png"
    icon_file.write_bytes(MINIMAL_PNG)

    # Create a file outside the library dir to verify traversal is blocked
    secret = tmp_path.parent / "secret.txt"
    secret.write_text("sensitive data")

    lib = MaterialLibrary(
        id="test-lib",
        name="Test Library",
        yaml_path=str(tmp_path / "materials.yaml"),
        library_path=str(tmp_path / "materials_libs.usda"),
        entries=[{"name": "Metal", "icon": "thumbs/Metal.png"}],
        icons={"Metal": "thumbs/Metal.png"},
        base_dir=str(tmp_path),
    )

    original = dict(config.material_libraries)
    config.material_libraries["test-lib"] = lib
    yield tmp_path
    config.material_libraries.clear()
    config.material_libraries.update(original)


@pytest.mark.api
class TestMaterialsPathTraversal:
    """Test path traversal protection on the icon endpoint."""

    async def test_normal_icon_returns_200(self, client, _library_with_icon):
        """Valid material icon request returns the PNG."""
        resp = await client.get("/materials/libraries/test-lib/icon/Metal")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"
        assert resp.content == MINIMAL_PNG

    async def test_traversal_returns_403(self, client, _library_with_icon):
        """Path with ../ that escapes the library directory returns 403."""
        # Use %2e%2e to avoid client-side URL normalization (real attack vector)
        resp = await client.get(
            "/materials/libraries/test-lib/icon/%2e%2e/%2e%2e/%2e%2e/etc/passwd"
        )
        assert resp.status_code == 403

    async def test_traversal_via_icon_name_returns_403(
        self, client, _library_with_icon
    ):
        """Traversal via the direct-path fallback also returns 403."""
        resp = await client.get("/materials/libraries/test-lib/icon/%2e%2e/secret.txt")
        assert resp.status_code == 403

    async def test_nonexistent_icon_returns_404(self, client, _library_with_icon):
        """A safe but nonexistent icon path returns 404, not 403."""
        resp = await client.get("/materials/libraries/test-lib/icon/NoSuchMaterial")
        assert resp.status_code == 404

    async def test_nonexistent_library_returns_404(self, client):
        """Request to unknown library returns 404."""
        resp = await client.get("/materials/libraries/bogus/icon/Metal")
        assert resp.status_code == 404
