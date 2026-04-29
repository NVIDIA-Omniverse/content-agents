# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the dominant colors extraction function."""

import tempfile
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image as PILImage
from PIL import ImageDraw

from world_understanding.functions.cv.get_dominant_colors import get_dominant_colors


class TestGetDominantColors:
    """Tests for get_dominant_colors function."""

    @pytest.fixture
    def solid_color_image_path(self):
        """Create a temporary image with a single solid color."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            # Create a 100x100 solid red image
            img = PILImage.new("RGB", (100, 100), color=(255, 0, 0))
            img.save(f.name)
            yield f.name

        Path(f.name).unlink(missing_ok=True)

    @pytest.fixture
    def multi_color_image_path(self):
        """Create a temporary image with multiple distinct colors."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            # Create a 100x100 image with 4 equal quadrants
            img = PILImage.new("RGB", (100, 100), color=(255, 255, 255))
            draw = ImageDraw.Draw(img)

            # Top-left: Red (25%)
            draw.rectangle([0, 0, 50, 50], fill=(255, 0, 0))

            # Top-right: Green (25%)
            draw.rectangle([50, 0, 100, 50], fill=(0, 255, 0))

            # Bottom-left: Blue (25%)
            draw.rectangle([0, 50, 50, 100], fill=(0, 0, 255))

            # Bottom-right: White (25%)

            img.save(f.name)
            yield f.name

        Path(f.name).unlink(missing_ok=True)

    @pytest.fixture
    def gradient_image_path(self):
        """Create a temporary image with a color gradient."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img = PILImage.new("RGB", (100, 100))
            pixels = img.load()

            # Create horizontal gradient from red to blue
            for x in range(100):
                for y in range(100):
                    r = int(255 * (1 - x / 100))
                    b = int(255 * (x / 100))
                    pixels[x, y] = (r, 0, b)

            img.save(f.name)
            yield f.name

        Path(f.name).unlink(missing_ok=True)

    @pytest.fixture
    def test_pil_image(self):
        """Create a PIL Image object for testing."""
        img = PILImage.new("RGB", (50, 50), color=(128, 128, 128))
        draw = ImageDraw.Draw(img)
        # Add colorful rectangles
        draw.rectangle([0, 0, 25, 25], fill=(255, 0, 0))
        draw.rectangle([25, 0, 50, 25], fill=(0, 255, 0))
        draw.rectangle([0, 25, 25, 50], fill=(0, 0, 255))
        return img

    def test_dominant_colors_single_color(self, solid_color_image_path):
        """Test extracting dominant color from solid color image."""
        result = get_dominant_colors(
            image=solid_color_image_path, n_colors=1, analyze_brightness=True
        )

        assert len(result["dominant_colors"]) == 1
        assert result["dominant_colors"][0]["rgb"] == [255, 0, 0]
        assert result["dominant_colors"][0]["hex"] == "#ff0000"
        pct = result["dominant_colors"][0]["percentage"]
        assert pct == pytest.approx(1.0, rel=0.01)
        assert result["n_clusters"] == 1
        assert "average_brightness" in result
        assert "color_diversity" in result

    def test_dominant_colors_multiple(self, multi_color_image_path):
        """Test extracting multiple dominant colors."""
        result = get_dominant_colors(
            image=multi_color_image_path, n_colors=4, analyze_brightness=True
        )

        assert len(result["dominant_colors"]) == 4
        assert result["n_clusters"] == 4

        # Check that percentages sum to 1.0
        total_pct = sum(c["percentage"] for c in result["dominant_colors"])
        assert total_pct == pytest.approx(1.0, rel=0.01)

        # Each color should be approximately 25%
        for color in result["dominant_colors"]:
            assert color["percentage"] == pytest.approx(0.25, rel=0.1)

    def test_dominant_colors_with_pil_image(self, test_pil_image):
        """Test with PIL Image object input."""
        result = get_dominant_colors(
            image=test_pil_image, n_colors=3, analyze_brightness=True
        )

        assert len(result["dominant_colors"]) == 3
        assert all("rgb" in color for color in result["dominant_colors"])
        assert all("hex" in color for color in result["dominant_colors"])
        colors = result["dominant_colors"]
        assert all("percentage" in color for color in colors)

    def test_n_colors_validation(self, solid_color_image_path):
        """Test n_colors parameter validation."""
        # Test minimum boundary
        err_msg = "n_colors must be between 1 and 20"
        with pytest.raises(ValueError, match=err_msg):
            get_dominant_colors(image=solid_color_image_path, n_colors=0)

        # Test maximum boundary
        with pytest.raises(ValueError, match=err_msg):
            get_dominant_colors(image=solid_color_image_path, n_colors=21)

    def test_valid_n_colors_range(self, multi_color_image_path):
        """Test valid n_colors values."""
        # Test minimum
        result_min = get_dominant_colors(image=multi_color_image_path, n_colors=1)
        assert len(result_min["dominant_colors"]) == 1

        # Test maximum - suppress convergence warning when clusters > unique colors
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=UserWarning, message="Number of distinct clusters.*"
            )
            result_max = get_dominant_colors(image=multi_color_image_path, n_colors=20)
        assert len(result_max["dominant_colors"]) <= 20

    def test_brightness_analysis(self, multi_color_image_path):
        """Test brightness analysis feature."""
        # With brightness analysis
        result_with = get_dominant_colors(
            image=multi_color_image_path, n_colors=4, analyze_brightness=True
        )

        assert "average_brightness" in result_with
        assert 0 <= result_with["average_brightness"] <= 255

        # Without brightness analysis
        result_without = get_dominant_colors(
            image=multi_color_image_path, n_colors=4, analyze_brightness=False
        )

        assert result_without["average_brightness"] == 0.0

    def test_color_diversity(self, solid_color_image_path, gradient_image_path):
        """Test color diversity calculation."""
        # Solid color should have low diversity
        result_solid = get_dominant_colors(image=solid_color_image_path, n_colors=1)
        # Note: K-means may still show some diversity even for solid colors
        assert result_solid["color_diversity"] < 0.5

        # Gradient should have higher diversity
        result_gradient = get_dominant_colors(image=gradient_image_path, n_colors=5)
        # Both images should have some diversity
        assert result_gradient["color_diversity"] > 0.1
        assert result_solid["color_diversity"] >= 0

    def test_hex_color_format(self, multi_color_image_path):
        """Test hex color format in output."""
        result = get_dominant_colors(image=multi_color_image_path, n_colors=4)

        for color in result["dominant_colors"]:
            hex_color = color["hex"]
            assert hex_color.startswith("#")
            assert len(hex_color) == 7
            # Verify it's valid hex
            int(hex_color[1:], 16)

    def test_rgb_values_range(self, multi_color_image_path):
        """Test that RGB values are in valid range."""
        result = get_dominant_colors(image=multi_color_image_path, n_colors=4)

        for color in result["dominant_colors"]:
            rgb = color["rgb"]
            assert len(rgb) == 3
            for value in rgb:
                assert 0 <= value <= 255
                assert isinstance(value, int)

    def test_percentage_ordering(self, multi_color_image_path):
        """Test that colors are ordered by percentage (highest first)."""
        result = get_dominant_colors(image=multi_color_image_path, n_colors=4)

        percentages = [c["percentage"] for c in result["dominant_colors"]]
        assert percentages == sorted(percentages, reverse=True)

    def test_file_not_found(self):
        """Test handling of non-existent file."""
        with pytest.raises(FileNotFoundError, match="Image file not found"):
            get_dominant_colors(image="non_existent_file.jpg", n_colors=3)

    def test_invalid_image_file(self, tmp_path):
        """Test handling of invalid image file."""
        # Create a text file with image extension
        invalid_file = tmp_path / "invalid.jpg"
        invalid_file.write_text("This is not an image")

        with pytest.raises(IOError, match="Failed to load image"):
            get_dominant_colors(image=str(invalid_file), n_colors=3)

    def test_invalid_image_type(self):
        """Test handling of invalid image type."""
        err_msg = "image must be a file path.*or PIL Image"
        with pytest.raises(TypeError, match=err_msg):
            get_dominant_colors(
                image=123,  # Invalid type
                n_colors=3,
            )

    def test_grayscale_image_conversion(self):
        """Test handling of grayscale images."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            # Create grayscale image
            img = PILImage.new("L", (50, 50), color=128)
            img.save(f.name)

            try:
                result = get_dominant_colors(image=f.name, n_colors=1)

                # Should convert to RGB
                color = result["dominant_colors"][0]
                assert color["rgb"] == [128, 128, 128]
                assert color["hex"] == "#808080"
            finally:
                Path(f.name).unlink(missing_ok=True)

    def test_rgba_image_conversion(self):
        """Test handling of RGBA images."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            # Create RGBA image with transparency
            img = PILImage.new("RGBA", (50, 50), color=(255, 0, 0, 128))
            img.save(f.name)

            try:
                result = get_dominant_colors(image=f.name, n_colors=1)

                # Should successfully convert and extract color
                color = result["dominant_colors"][0]
                assert color["rgb"] == [255, 0, 0]
            finally:
                Path(f.name).unlink(missing_ok=True)

    def test_more_clusters_than_colors(self, solid_color_image_path):
        """Test requesting more clusters than unique colors in image."""
        # Solid color image has only 1 unique color
        # Suppress convergence warning when clusters > unique colors
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=UserWarning, message="Number of distinct clusters.*"
            )
            result = get_dominant_colors(image=solid_color_image_path, n_colors=10)

        # Should still work, but may have duplicate or similar colors
        assert len(result["dominant_colors"]) <= 10
        assert result["n_clusters"] == 10

    def test_complex_image_clustering(self, gradient_image_path):
        """Test clustering on complex gradient image."""
        result = get_dominant_colors(image=gradient_image_path, n_colors=5)

        assert len(result["dominant_colors"]) == 5

        # Should have good color diversity
        assert result["color_diversity"] > 0.2

        # Colors should represent the gradient
        rgb_values = [color["rgb"] for color in result["dominant_colors"]]

        # Should have colors ranging from reddish to bluish
        red_values = [rgb[0] for rgb in rgb_values]
        blue_values = [rgb[2] for rgb in rgb_values]

        assert max(red_values) > 200  # Some high red values
        assert max(blue_values) > 200  # Some high blue values
        assert min(red_values) < 100  # Some low red values
        assert min(blue_values) < 100  # Some low blue values

    @patch("world_understanding.functions.cv.get_dominant_colors.KMeans")
    def test_kmeans_parameters(self, mock_kmeans, solid_color_image_path):
        """Test that KMeans is called with correct parameters."""
        mock_instance = MagicMock()
        mock_kmeans.return_value = mock_instance
        mock_instance.labels_ = np.zeros(10000)  # 100x100 image
        mock_instance.cluster_centers_ = np.array([[255, 0, 0]])

        # Suppress warning from sklearn about convergence
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            get_dominant_colors(image=solid_color_image_path, n_colors=5)

        # Verify KMeans was called with correct parameters
        mock_kmeans.assert_called_once_with(n_clusters=5, random_state=42, n_init=10)
