# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for async NVCF optimizer with 202 polling support."""

import io
import json
import zipfile
from unittest.mock import AsyncMock, Mock, patch

import pytest

from world_understanding.functions.graphics.scene_optimizer_nvcf import (
    optimize_usd_from_path,
    optimize_usd_from_url,
)
from world_understanding.utils.nvcf_utils import parse_zip_response, poll_nvcf_status


class TestParseZipResponse:
    """Tests for parse_zip_response function."""

    def test_parse_zip_response_valid(self):
        """Test parsing a valid ZIP with .response file."""
        # Create a valid ZIP with a .response file
        result_data = {
            "success": True,
            "optimized_stage_base64": "dGVzdA==",
            "operations_executed": ["deinstance", "split"],
        }

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            zf.writestr("result.response", json.dumps(result_data))
        zip_content = zip_buffer.getvalue()

        # Parse it
        result = parse_zip_response(zip_content)

        assert result is not None
        assert result["success"] is True
        assert result["optimized_stage_base64"] == "dGVzdA=="
        assert "deinstance" in result["operations_executed"]

    def test_parse_zip_response_invalid_zip(self):
        """Test parsing invalid ZIP content."""
        invalid_content = b"not a zip file"
        result = parse_zip_response(invalid_content)
        assert result is None

    def test_parse_zip_response_no_response_file(self):
        """Test ZIP without .response file."""
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            zf.writestr("other_file.txt", "some content")
        zip_content = zip_buffer.getvalue()

        result = parse_zip_response(zip_content)
        assert result is None

    def test_parse_zip_response_invalid_json(self):
        """Test .response file with invalid JSON."""
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            zf.writestr("result.response", "not valid json {")
        zip_content = zip_buffer.getvalue()

        result = parse_zip_response(zip_content)
        assert result is None


class TestPollNvcfStatus:
    """Tests for poll_nvcf_status async function."""

    @pytest.mark.asyncio
    async def test_poll_nvcf_status_200_immediate(self):
        """Test immediate 200 response (no polling needed)."""
        # Mock httpx.AsyncClient
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {
            "success": True,
            "optimized_stage_base64": "dGVzdA==",
        }
        mock_client.get.return_value = mock_response

        # Call poll_nvcf_status
        status_code, result = await poll_nvcf_status(
            client=mock_client,
            req_id="test-req-id",
            api_key="test-key",
            poll_seconds=300,
            timeout=600,
        )

        assert status_code == 200
        assert result is not None
        assert result["success"] is True
        mock_client.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_poll_nvcf_status_202_then_200(self):
        """Test 202 followed by 200 (polling loop)."""
        mock_client = AsyncMock()

        # First call: 202 Accepted
        mock_response_202 = Mock()
        mock_response_202.status_code = 202

        # Second call: 200 OK
        mock_response_200 = Mock()
        mock_response_200.status_code = 200
        mock_response_200.headers = {"content-type": "application/json"}
        mock_response_200.json.return_value = {
            "success": True,
            "optimized_stage_base64": "dGVzdA==",
        }

        mock_client.get.side_effect = [mock_response_202, mock_response_200]

        # Call poll_nvcf_status
        status_code, result = await poll_nvcf_status(
            client=mock_client,
            req_id="test-req-id",
            api_key="test-key",
            poll_seconds=300,
            timeout=600,
        )

        assert status_code == 200
        assert result is not None
        assert result["success"] is True
        assert mock_client.get.call_count == 2

    @pytest.mark.asyncio
    async def test_poll_nvcf_status_timeout(self):
        """Test timeout during polling."""
        mock_client = AsyncMock()

        # Always return 202 Accepted
        mock_response_202 = Mock()
        mock_response_202.status_code = 202
        mock_client.get.return_value = mock_response_202

        # Call with short timeout
        status_code, result = await poll_nvcf_status(
            client=mock_client,
            req_id="test-req-id",
            api_key="test-key",
            poll_seconds=1,
            timeout=0.1,  # Very short timeout
        )

        assert status_code == 504
        assert result is None

    @pytest.mark.asyncio
    async def test_poll_nvcf_status_504_gateway_timeout(self):
        """Test 504 Gateway Timeout response."""
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.status_code = 504
        mock_client.get.return_value = mock_response

        status_code, result = await poll_nvcf_status(
            client=mock_client,
            req_id="test-req-id",
            api_key="test-key",
            poll_seconds=300,
            timeout=600,
        )

        assert status_code == 504
        assert result is None

    @pytest.mark.asyncio
    async def test_poll_nvcf_status_zip_response(self):
        """Test polling with ZIP response."""
        mock_client = AsyncMock()

        # Create a valid ZIP response
        result_data = {
            "success": True,
            "optimized_stage_base64": "dGVzdA==",
        }
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            zf.writestr("result.response", json.dumps(result_data))
        zip_content = zip_buffer.getvalue()

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/zip"}
        mock_response.content = zip_content
        mock_client.get.return_value = mock_response

        status_code, result = await poll_nvcf_status(
            client=mock_client,
            req_id="test-req-id",
            api_key="test-key",
            poll_seconds=300,
            timeout=600,
        )

        assert status_code == 200
        assert result is not None
        assert result["success"] is True


class TestOptimizeUsdFromUrl:
    """Tests for optimize_usd_from_url async function."""

    @pytest.mark.asyncio
    async def test_optimize_usd_from_url_200_direct(self, tmp_path):
        """Test direct 200 OK response (no polling)."""
        output_path = tmp_path / "optimized.usdc"

        # Mock httpx.AsyncClient
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.headers = {"content-type": "application/json"}
            mock_response.json.return_value = {
                "success": True,
                "optimized_stage_base64": "dGVzdCBjb250ZW50",  # "test content"
                "operations_executed": ["deinstance"],
                "report": "Test report",
            }
            mock_client.post.return_value = mock_response

            # Call function
            result = await optimize_usd_from_url(
                input_url="https://example.com/input.usd",
                output_path=output_path,
                api_key="test-key",
                base_url="https://test-function.invocation.api.nvcf.nvidia.com",
            )

            assert result["status"] == "success"
            assert "deinstance" in result["operations_executed"]
            assert output_path.exists()

    @pytest.mark.asyncio
    async def test_optimize_usd_from_url_202_polling(self, tmp_path):
        """Test 202 response with polling."""
        output_path = tmp_path / "optimized.usdc"

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            # Initial POST: 202 Accepted
            mock_post_response = Mock()
            mock_post_response.status_code = 202
            mock_post_response.headers = {"nvcf-reqid": "test-req-id"}
            mock_client.post.return_value = mock_post_response

            # Poll GET: 200 OK
            mock_get_response = Mock()
            mock_get_response.status_code = 200
            mock_get_response.headers = {"content-type": "application/json"}
            mock_get_response.json.return_value = {
                "success": True,
                "optimized_stage_base64": "dGVzdCBjb250ZW50",
                "operations_executed": ["deinstance"],
            }
            mock_client.get.return_value = mock_get_response

            # Call function
            result = await optimize_usd_from_url(
                input_url="https://example.com/input.usd",
                output_path=output_path,
                api_key="test-key",
                base_url="https://test-function.invocation.api.nvcf.nvidia.com",
            )

            assert result["status"] == "success"
            assert mock_client.post.call_count == 1
            assert mock_client.get.call_count >= 1  # Polling occurred


class TestOptimizeUsdFromPath:
    """Tests for optimize_usd_from_path async function."""

    @pytest.mark.asyncio
    async def test_optimize_usd_from_path(self, tmp_path):
        """Test optimize_usd_from_path with S3 upload."""
        input_path = tmp_path / "input.usd"
        output_path = tmp_path / "optimized.usdc"
        input_path.write_text("test usd content")

        with patch(
            "world_understanding.functions.graphics.scene_optimizer_nvcf.upload_file_to_s3"
        ) as mock_upload:
            mock_upload.return_value = "s3://test-bucket/test-key/input.usd"

            with patch(
                "world_understanding.functions.graphics.scene_optimizer_nvcf.delete_s3_path"
            ):
                with patch("httpx.AsyncClient") as mock_client_class:
                    mock_client = AsyncMock()
                    mock_client_class.return_value.__aenter__.return_value = mock_client

                    mock_response = Mock()
                    mock_response.status_code = 200
                    mock_response.headers = {"content-type": "application/json"}
                    mock_response.json.return_value = {
                        "success": True,
                        "optimized_stage_base64": "dGVzdA==",
                        "operations_executed": ["deinstance"],
                    }
                    mock_client.post.return_value = mock_response

                    # Call function
                    result = await optimize_usd_from_path(
                        input_path=input_path,
                        output_path=output_path,
                        api_key="test-key",
                        base_url="https://test-function.invocation.api.nvcf.nvidia.com",
                        use_data_uri=False,
                    )

                    assert result["status"] == "success"
                    assert mock_upload.called
