# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for materials catalog behavior when the default library has no icons."""

import pytest


@pytest.mark.api
class TestMaterialsCatalog:
    async def test_default_materials_list_keeps_entries_without_icons(self, client):
        """The default catalog should still list materials when icons are absent."""
        response = await client.get("/materials")

        assert response.status_code == 200
        body = response.json()
        assert body["total"] > 0

        aluminum = next(
            item for item in body["materials"] if item["name"] == "Aluminum"
        )
        assert aluminum["description"]
        assert aluminum["binding"]
        assert aluminum["icon_url"] is None
        assert aluminum["icon_path"] is None

    async def test_default_library_materials_return_null_icon_fields(self, client):
        """Per-library listing should return null icon fields when icons are absent."""
        response = await client.get("/materials/libraries/default")

        assert response.status_code == 200
        body = response.json()
        assert body["total"] > 0

        aluminum = next(
            item for item in body["materials"] if item["name"] == "Aluminum"
        )
        assert aluminum["icon_url"] is None
        assert aluminum["icon_path"] is None

    async def test_default_material_icon_returns_404_when_not_shipped(self, client):
        """Requesting a default icon should 404 when the library is icon-free."""
        response = await client.get("/materials/icon/Aluminum")

        assert response.status_code == 404
