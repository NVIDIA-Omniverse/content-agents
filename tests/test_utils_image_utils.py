# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for image utility functions."""

import base64
from io import BytesIO

from PIL import Image

from world_understanding.utils.image_utils import extract_base64_strings


class TestExtractBase64Strings:
    """Test cases for the extract_base64_strings function."""

    def create_test_png_base64(self, color=(255, 0, 0), size=(2, 2)):
        """Helper to create a small test PNG image as base64 string."""
        img = Image.new("RGB", size, color)
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode("utf-8")

    def test_single_image_extraction(self):
        """Test extracting a single base64 image."""
        test_image = self.create_test_png_base64()
        result = extract_base64_strings(test_image)

        assert len(result) == 1
        assert result[0] == test_image
        assert result[0].startswith("iVBORw0KGgo")

    def test_multiple_images_concatenated(self):
        """Test extracting multiple concatenated base64 images."""
        image1 = self.create_test_png_base64(color=(255, 0, 0))
        image2 = self.create_test_png_base64(color=(0, 255, 0))
        image3 = self.create_test_png_base64(color=(0, 0, 255))

        # Concatenate images directly
        concatenated = image1 + image2 + image3
        result = extract_base64_strings(concatenated)

        assert len(result) == 3
        assert all(img.startswith("iVBORw0KGgo") for img in result)

    def test_images_with_comma_separators(self):
        """Test extracting images separated by commas."""
        image1 = self.create_test_png_base64(color=(255, 0, 0))
        image2 = self.create_test_png_base64(color=(0, 255, 0))

        # Concatenate with commas
        concatenated = image1 + "," + image2
        result = extract_base64_strings(concatenated)

        assert len(result) == 2
        assert all(img.startswith("iVBORw0KGgo") for img in result)
        # Ensure commas are removed
        assert "," not in result[0]
        assert "," not in result[1]

    def test_images_with_metadata_before(self):
        """Test extracting images with metadata lines before them."""
        image1 = self.create_test_png_base64()

        input_text = (
            """Running with Xvfb for GPU rendering...
Camera:
Renderer plugin: HdStormRendererPlugin
"""
            + image1
        )

        result = extract_base64_strings(input_text)

        assert len(result) == 1
        assert result[0] == image1

    def test_images_with_recording_time_metadata(self):
        """Test extracting images with 'Recording time code' metadata."""
        image1 = self.create_test_png_base64(color=(255, 0, 0))
        image2 = self.create_test_png_base64(color=(0, 255, 0))

        # Test with metadata between images
        input_text = f"Recording time code: 0.000000\n{image1}Recording time code: 1.000000\n{image2}"

        result = extract_base64_strings(input_text)

        assert len(result) == 2
        assert all(img.startswith("iVBORw0KGgo") for img in result)

    def test_images_with_inline_recording_time(self):
        """Test extracting images with recording time embedded in the same line."""
        image1 = self.create_test_png_base64(color=(255, 0, 0))
        image2 = self.create_test_png_base64(color=(0, 255, 0))

        # Simulate the real-world case where metadata is embedded within the line
        input_text = f"{image1}Recording time code: 1.000000{image2}"

        result = extract_base64_strings(input_text)

        assert len(result) == 2
        assert all(img.startswith("iVBORw0KGgo") for img in result)

    def test_complex_real_world_scenario(self):
        """Test a complex scenario mimicking real pipe output."""
        image1 = self.create_test_png_base64(color=(255, 0, 0))
        image2 = self.create_test_png_base64(color=(0, 255, 0))
        image3 = self.create_test_png_base64(color=(0, 0, 255))

        # Complex scenario with various metadata patterns
        input_text = f"""Running with Xvfb for GPU rendering...
Camera:
Renderer plugin: HdStormRendererPlugin
Recording time code: 0.000000
{image1},Recording time code: 1.000000
{image2}
Recording time code: 2.000000
,{image3}"""

        result = extract_base64_strings(input_text)

        assert len(result) == 3
        assert all(img.startswith("iVBORw0KGgo") for img in result)
        # Ensure no metadata or separators remain
        for img in result:
            assert "Recording time code" not in img
            assert "Camera" not in img
            assert "Renderer" not in img
            assert "," not in img
            assert "\n" not in img

    def test_empty_input(self):
        """Test with empty input string."""
        result = extract_base64_strings("")
        assert result == []

    def test_no_images_only_metadata(self):
        """Test input with only metadata, no images."""
        input_text = """Recording time code: 0.000000
Recording time code: 1.000000
Camera: test
Renderer plugin: HdStormRendererPlugin
Running with Xvfb for GPU rendering..."""

        result = extract_base64_strings(input_text)
        assert result == []

    def test_whitespace_and_newlines(self):
        """Test that whitespace and newlines are properly handled."""
        image1 = self.create_test_png_base64()

        # Add various whitespace
        input_text = f"\n\n   {image1}   \n\n"

        result = extract_base64_strings(input_text)

        assert len(result) == 1
        assert result[0] == image1

    def test_partial_base64_data(self):
        """Test with incomplete base64 data (no PNG header)."""
        # Some random base64 that's not a PNG
        non_png_base64 = base64.b64encode(b"This is not a PNG").decode("utf-8")

        result = extract_base64_strings(non_png_base64)
        assert result == []

    def test_mixed_content_with_non_base64(self):
        """Test extraction when there's mixed content including non-base64 characters."""
        image1 = self.create_test_png_base64()

        # Add some non-base64 characters that should be filtered out
        input_text = f"{image1[:50]}!@#${image1[50:]}"

        result = extract_base64_strings(input_text)

        assert len(result) == 1
        # The special characters should be removed
        assert "!" not in result[0]
        assert "@" not in result[0]
        assert "#" not in result[0]
        assert "$" not in result[0]

    def test_preserve_base64_padding(self):
        """Test that base64 padding characters (=) are preserved."""
        # Create an image and ensure it has padding
        image1 = self.create_test_png_base64()

        # If the image doesn't naturally have padding, we'll check that
        # any existing padding is preserved
        if image1.endswith("="):
            result = extract_base64_strings(image1)
            assert result[0].endswith("=")

    def test_multiple_recording_times_same_line(self):
        """Test multiple recording time entries on the same line."""
        image1 = self.create_test_png_base64()

        input_text = f"Recording time code: 0.0Recording time code: 1.0{image1}"

        result = extract_base64_strings(input_text)

        assert len(result) == 1
        assert result[0] == image1

    def test_base64_with_plus_and_slash(self):
        """Test that base64 characters + and / are preserved."""
        # Create a test string with + and /
        test_data = b"Test data with chars to get + and / in base64"
        test_base64 = base64.b64encode(test_data).decode("utf-8")

        # Ensure our test data has + or / (it should)
        if "+" in test_base64 or "/" in test_base64:
            # Create a fake PNG header + our test data
            fake_png = "iVBORw0KGgo" + test_base64

            result = extract_base64_strings(fake_png)

            assert len(result) == 1
            # Check that + and / are preserved if they were in the original
            if "+" in test_base64:
                assert "+" in result[0]
            if "/" in test_base64:
                assert "/" in result[0]
