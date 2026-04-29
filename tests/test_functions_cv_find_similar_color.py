# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the color matcher function."""

import tempfile
from pathlib import Path

import pytest
from PIL import Image as PILImage
from PIL import ImageDraw

from world_understanding.functions.cv.find_similar_color import find_similar_color


class TestFindSimilarColor:
    """Tests for find_similar_color function."""

    @pytest.fixture
    def test_image_path(self):
        """Create a temporary test image with known colors."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            # Create a 100x100 image with specific color regions
            img = PILImage.new("RGB", (100, 100), color=(255, 255, 255))
            draw = ImageDraw.Draw(img)

            # Draw a red rectangle (25% of image)
            draw.rectangle([0, 0, 50, 50], fill=(255, 0, 0))

            # Draw a green rectangle (25% of image)
            draw.rectangle([50, 0, 100, 50], fill=(0, 255, 0))

            # Draw a blue rectangle (25% of image)
            draw.rectangle([0, 50, 50, 100], fill=(0, 0, 255))

            # Bottom right remains white (25% of image)

            img.save(f.name)
            yield f.name

        # Cleanup
        Path(f.name).unlink(missing_ok=True)

    @pytest.fixture
    def test_pil_image(self):
        """Create a PIL Image object for testing."""
        img = PILImage.new("RGB", (50, 50), color=(128, 128, 128))
        draw = ImageDraw.Draw(img)
        # Add a small red square
        draw.rectangle([10, 10, 20, 20], fill=(255, 0, 0))
        return img

    def test_find_similar_color_with_file_path(self, test_image_path):
        """Test matching color with image file path."""
        # Test finding red color
        result = find_similar_color(
            image=test_image_path,
            target_color=[255, 0, 0],
            color_tolerance=10,
            min_percentage=20.0,
        )

        assert result["contains_color"] is True
        assert result["matching_percentage"] >= 25.0
        assert result["target_color_rgb"] == [255, 0, 0]
        assert result["target_color_hex"] == "#ff0000"
        assert "closest_colors" in result
        assert len(result["closest_colors"]) > 0

    def test_find_similar_color_with_pil_image(self, test_pil_image):
        """Test matching color with PIL Image object."""
        result = find_similar_color(
            image=test_pil_image,
            target_color=[255, 0, 0],
            color_tolerance=20,
            min_percentage=1.0,
        )

        assert result["contains_color"] is True
        assert result["matching_percentage"] > 0
        assert result["pixel_count"] > 0
        assert result["total_pixels"] == 2500  # 50x50

    def test_color_not_found(self, test_image_path):
        """Test when target color is not found."""
        # Look for purple with strict tolerance
        result = find_similar_color(
            image=test_image_path,
            target_color=[128, 0, 128],
            color_tolerance=10,
            min_percentage=5.0,
        )

        assert result["contains_color"] is False
        assert result["matching_percentage"] < 5.0

    def test_color_tolerance_variations(self, test_image_path):
        """Test different color tolerance values."""
        # Strict tolerance
        result_strict = find_similar_color(
            image=test_image_path,
            target_color=[250, 5, 5],  # Slightly off from pure red
            color_tolerance=10,
            min_percentage=1.0,
        )

        # Loose tolerance
        result_loose = find_similar_color(
            image=test_image_path,
            target_color=[250, 5, 5],
            color_tolerance=50,
            min_percentage=1.0,
        )

        # Loose tolerance should find more matches
        assert (
            result_loose["matching_percentage"] >= result_strict["matching_percentage"]
        )

    def test_min_percentage_threshold(self, test_image_path):
        """Test minimum percentage threshold."""
        # Red is 25% of the image
        result_below = find_similar_color(
            image=test_image_path,
            target_color=[255, 0, 0],
            color_tolerance=10,
            min_percentage=30.0,  # Higher than actual
        )

        result_above = find_similar_color(
            image=test_image_path,
            target_color=[255, 0, 0],
            color_tolerance=10,
            min_percentage=20.0,  # Lower than actual
        )

        assert result_below["contains_color"] is False
        assert result_above["contains_color"] is True

    def test_invalid_rgb_values(self):
        """Test validation of RGB values."""
        err_msg = "RGB values must be integers between 0-255"
        with pytest.raises(ValueError, match=err_msg):
            find_similar_color(
                image="dummy.jpg",
                target_color=[256, 0, 0],  # Invalid value
            )

        with pytest.raises(ValueError, match=err_msg):
            find_similar_color(
                image="dummy.jpg",
                target_color=[-1, 128, 128],  # Negative value
            )

    def test_invalid_target_color_format(self):
        """Test invalid target color format."""
        err_msg = "target_color must be a list or tuple of 3 values"
        with pytest.raises(ValueError, match=err_msg):
            find_similar_color(
                image="dummy.jpg",
                target_color=[255, 0],  # Only 2 values
            )

        with pytest.raises(ValueError, match=err_msg):
            find_similar_color(
                image="dummy.jpg",
                target_color="red",  # String instead of list
            )

    def test_file_not_found(self):
        """Test handling of non-existent file."""
        with pytest.raises(FileNotFoundError, match="Image file not found"):
            find_similar_color(image="non_existent_file.jpg", target_color=[255, 0, 0])

    def test_invalid_image_file(self, tmp_path):
        """Test handling of invalid image file."""
        # Create a text file with image extension
        invalid_file = tmp_path / "invalid.jpg"
        invalid_file.write_text("This is not an image")

        with pytest.raises(IOError, match="Failed to load image"):
            find_similar_color(image=str(invalid_file), target_color=[255, 0, 0])

    def test_invalid_image_type(self):
        """Test handling of invalid image type."""
        err_msg = "image must be a file path.*or PIL Image"
        with pytest.raises(TypeError, match=err_msg):
            find_similar_color(
                image=123,  # Invalid type
                target_color=[255, 0, 0],
            )

    def test_tuple_color_input(self, test_image_path):
        """Test using tuple for target color."""
        result = find_similar_color(
            image=test_image_path,
            target_color=(255, 0, 0),  # Tuple instead of list
            color_tolerance=10,
            min_percentage=20.0,
        )

        assert result["contains_color"] is True
        assert result["target_color_rgb"] == [255, 0, 0]

    def test_grayscale_image_conversion(self):
        """Test handling of grayscale images."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            # Create grayscale image
            img = PILImage.new("L", (50, 50), color=128)
            img.save(f.name)

            try:
                result = find_similar_color(
                    image=f.name, target_color=[128, 128, 128], color_tolerance=10
                )

                assert result["contains_color"] is True
                assert result["matching_percentage"] > 90.0
            finally:
                Path(f.name).unlink(missing_ok=True)

    def test_closest_colors_output(self, test_image_path):
        """Test the closest colors output format."""
        result = find_similar_color(
            image=test_image_path,
            target_color=[200, 50, 50],  # Pinkish color
            color_tolerance=50,
        )

        assert "closest_colors" in result
        assert isinstance(result["closest_colors"], list)

        if len(result["closest_colors"]) > 0:
            color_info = result["closest_colors"][0]
            assert "rgb" in color_info
            assert "hex" in color_info
            assert "distance" in color_info
            assert "percentage" in color_info

            # Validate format
            assert len(color_info["rgb"]) == 3
            assert color_info["hex"].startswith("#")
            assert isinstance(color_info["distance"], int | float)
            assert 0 <= color_info["percentage"] <= 100

    def test_rgba_image_conversion(self):
        """Test handling of RGBA images."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            # Create RGBA image with transparency
            img = PILImage.new("RGBA", (50, 50), color=(255, 0, 0, 128))
            img.save(f.name)

            try:
                result = find_similar_color(
                    image=f.name, target_color=[255, 0, 0], color_tolerance=10
                )

                # Should successfully convert and match
                assert result["contains_color"] is True
            finally:
                Path(f.name).unlink(missing_ok=True)

    def test_large_tolerance(self, test_image_path):
        """Test with very large color tolerance."""
        result = find_similar_color(
            image=test_image_path,
            target_color=[128, 128, 128],  # Gray
            color_tolerance=255,  # Maximum tolerance
            min_percentage=1.0,
        )

        # With max tolerance, should match almost everything
        assert result["contains_color"] is True
        assert result["matching_percentage"] > 50.0

    def test_exact_color_match(self, test_image_path):
        """Test exact color matching with zero tolerance."""
        result = find_similar_color(
            image=test_image_path,
            target_color=[255, 0, 0],
            color_tolerance=0,  # Exact match only
            min_percentage=1.0,
        )

        # Should still find the red region
        assert result["contains_color"] is True
        assert result["matching_percentage"] >= 25.0
