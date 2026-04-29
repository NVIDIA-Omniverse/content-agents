# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for image I/O utilities."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from world_understanding.utils.io.image_io import (
    load_image_to_array,
    save_image_from_array,
)


class TestLoadImage:
    """Tests for load_image_to_array function."""

    def test_load_rgb_image(self, tmp_path):
        """Test loading an RGB image."""
        # Create a test RGB image
        test_img = Image.new("RGB", (100, 100), color="red")
        test_path = tmp_path / "test_rgb.png"
        test_img.save(test_path)

        # Load the image
        loaded_img = load_image_to_array(test_path)

        # Verify
        assert isinstance(loaded_img, np.ndarray)
        assert loaded_img.shape == (100, 100, 3)
        assert loaded_img.dtype == np.uint8
        # Red color should be [255, 0, 0]
        assert np.all(loaded_img[0, 0] == [255, 0, 0])

    def test_load_grayscale_image(self, tmp_path):
        """Test loading a grayscale image - should convert to RGB."""
        # Create a test grayscale image
        test_img = Image.new("L", (50, 50), color=128)
        test_path = tmp_path / "test_gray.png"
        test_img.save(test_path)

        # Load the image
        loaded_img = load_image_to_array(test_path)

        # Verify it's converted to RGB
        assert isinstance(loaded_img, np.ndarray)
        assert loaded_img.shape == (50, 50, 3)
        assert loaded_img.dtype == np.uint8
        # Gray value should be replicated across all channels
        assert np.all(loaded_img[0, 0] == [128, 128, 128])

    def test_load_rgba_image(self, tmp_path):
        """Test loading an RGBA image - should convert to RGB."""
        # Create a test RGBA image
        test_img = Image.new("RGBA", (30, 30), color=(255, 0, 0, 128))
        test_path = tmp_path / "test_rgba.png"
        test_img.save(test_path)

        # Load the image
        loaded_img = load_image_to_array(test_path)

        # Verify it's converted to RGB (alpha channel removed)
        assert isinstance(loaded_img, np.ndarray)
        assert loaded_img.shape == (30, 30, 3)
        assert loaded_img.dtype == np.uint8

    def test_load_with_pathlib_path(self, tmp_path):
        """Test loading image with pathlib.Path object."""
        # Create a test image
        test_img = Image.new("RGB", (20, 20), color="blue")
        test_path = Path(tmp_path) / "test_pathlib.png"
        test_img.save(test_path)

        # Load using Path object
        loaded_img = load_image_to_array(test_path)

        # Verify
        assert isinstance(loaded_img, np.ndarray)
        assert loaded_img.shape == (20, 20, 3)
        # Blue color should be [0, 0, 255]
        assert np.all(loaded_img[0, 0] == [0, 0, 255])

    def test_load_with_string_path(self, tmp_path):
        """Test loading image with string path."""
        # Create a test image
        test_img = Image.new("RGB", (20, 20), color="green")
        test_path = tmp_path / "test_string.png"
        test_img.save(test_path)

        # Load using string path
        loaded_img = load_image_to_array(str(test_path))

        # Verify
        assert isinstance(loaded_img, np.ndarray)
        assert loaded_img.shape == (20, 20, 3)
        # Green color should be [0, 255, 0] (but PIL might save as [0, 128, 0])
        assert loaded_img[0, 0, 1] > 0  # Green channel should be non-zero

    def test_load_nonexistent_file(self):
        """Test loading a file that doesn't exist."""
        with pytest.raises(FileNotFoundError):
            load_image_to_array("nonexistent_file.png")

    def test_load_invalid_file(self, tmp_path):
        """Test loading a file that's not an image."""
        # Create a text file
        test_path = tmp_path / "not_an_image.txt"
        test_path.write_text("This is not an image")

        # Should raise an error when trying to load
        from PIL import UnidentifiedImageError

        with pytest.raises(
            UnidentifiedImageError
        ):  # PIL will raise an appropriate exception
            load_image_to_array(test_path)


class TestSaveImage:
    """Tests for save_image_from_array function."""

    def test_save_uint8_image(self, tmp_path):
        """Test saving a uint8 numpy array."""
        # Create a test array
        test_array = np.zeros((50, 50, 3), dtype=np.uint8)
        test_array[:, :, 0] = 255  # Red channel

        # Save the image
        save_path = tmp_path / "saved_uint8.png"
        save_image_from_array(test_array, save_path)

        # Verify the file exists and can be loaded
        assert save_path.exists()
        loaded_img = Image.open(save_path)
        assert loaded_img.size == (50, 50)
        assert loaded_img.mode == "RGB"

        # Verify the content
        loaded_array = np.array(loaded_img)
        assert np.array_equal(loaded_array, test_array)

    def test_save_float_image(self, tmp_path):
        """Test saving a float numpy array (0-1 range)."""
        # Create a test array with float values
        test_array = np.zeros((30, 30, 3), dtype=np.float32)
        test_array[:, :, 1] = 0.5  # Green channel at 50%

        # Save the image
        save_path = tmp_path / "saved_float.png"
        save_image_from_array(test_array, save_path)

        # Verify the file exists and can be loaded
        assert save_path.exists()
        loaded_img = Image.open(save_path)
        assert loaded_img.size == (30, 30)

        # Verify the content (should be converted to uint8)
        loaded_array = np.array(loaded_img)
        expected_green = int(0.5 * 255)
        assert loaded_array[0, 0, 1] == pytest.approx(expected_green, abs=1)

    def test_save_with_pathlib_path(self, tmp_path):
        """Test saving image with pathlib.Path object."""
        # Create a test array
        test_array = np.full((20, 20, 3), 100, dtype=np.uint8)

        # Save using Path object
        save_path = Path(tmp_path) / "saved_pathlib.png"
        save_image_from_array(test_array, save_path)

        # Verify
        assert save_path.exists()
        loaded_img = Image.open(save_path)
        assert loaded_img.size == (20, 20)

    def test_save_with_string_path(self, tmp_path):
        """Test saving image with string path."""
        # Create a test array
        test_array = np.full((20, 20, 3), 150, dtype=np.uint8)

        # Save using string path
        save_path = tmp_path / "saved_string.png"
        save_image_from_array(test_array, str(save_path))

        # Verify
        assert save_path.exists()
        loaded_img = Image.open(save_path)
        assert loaded_img.size == (20, 20)

    def test_save_different_formats(self, tmp_path):
        """Test saving images in different formats."""
        # Create a test array
        test_array = np.random.randint(0, 255, (40, 40, 3), dtype=np.uint8)

        # Test different formats
        formats = [".png", ".jpg", ".jpeg", ".bmp"]
        for fmt in formats:
            save_path = tmp_path / f"saved_image{fmt}"
            save_image_from_array(test_array, save_path)
            assert save_path.exists()

            # Verify can be loaded
            loaded_img = Image.open(save_path)
            assert loaded_img.size == (40, 40)

    def test_save_to_nonexistent_directory(self, tmp_path):
        """Test saving to a directory that doesn't exist."""
        # Create a test array
        test_array = np.zeros((10, 10, 3), dtype=np.uint8)

        # Try to save to non-existent directory
        save_path = tmp_path / "nonexistent_dir" / "image.png"

        # Should raise an error
        with pytest.raises(FileNotFoundError):
            save_image_from_array(test_array, save_path)

    def test_roundtrip_conversion(self, tmp_path):
        """Test that saving and loading preserves the image data."""
        # Create a random test array
        original_array = np.random.randint(0, 255, (60, 80, 3), dtype=np.uint8)

        # Save and load
        save_path = tmp_path / "roundtrip.png"
        save_image_from_array(original_array, save_path)
        loaded_array = load_image_to_array(save_path)

        # Verify arrays are equal
        assert np.array_equal(original_array, loaded_array)

    def test_save_float64_image(self, tmp_path):
        """Test saving a float64 numpy array."""
        # Create a test array with float64 values
        test_array = np.random.rand(25, 25, 3).astype(np.float64)

        # Save the image
        save_path = tmp_path / "saved_float64.png"
        save_image_from_array(test_array, save_path)

        # Verify the file exists
        assert save_path.exists()

        # Load and check conversion
        loaded_img = Image.open(save_path)
        loaded_array = np.array(loaded_img)
        # Should be converted to uint8 range [0, 255]
        assert loaded_array.dtype == np.uint8
        assert np.min(loaded_array) >= 0
        assert np.max(loaded_array) <= 255


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_empty_image(self, tmp_path):
        """Test handling of zero-sized images."""
        # This should raise an error in PIL
        test_array = np.zeros((0, 0, 3), dtype=np.uint8)
        save_path = tmp_path / "empty.png"

        with pytest.raises((SystemError, ValueError)):
            save_image_from_array(test_array, save_path)

    def test_single_pixel_image(self, tmp_path):
        """Test handling of 1x1 images."""
        # Create single pixel image
        test_array = np.array([[[255, 128, 64]]], dtype=np.uint8)

        # Save and load
        save_path = tmp_path / "single_pixel.png"
        save_image_from_array(test_array, save_path)
        loaded_array = load_image_to_array(save_path)

        # Verify
        assert loaded_array.shape == (1, 1, 3)
        assert np.array_equal(loaded_array[0, 0], [255, 128, 64])

    def test_large_image(self, tmp_path):
        """Test handling of large images."""
        # Create a large image (but not too large to avoid memory issues in tests)
        test_array = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)

        # Save and verify it works
        save_path = tmp_path / "large_image.png"
        save_image_from_array(test_array, save_path)
        assert save_path.exists()

        # Verify we can load it back
        loaded_array = load_image_to_array(save_path)
        assert loaded_array.shape == (1000, 1000, 3)
