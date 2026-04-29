# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Image utility functions for encoding/decoding and manipulation."""

import base64
import binascii
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


def load_image(path: str | Path) -> Image.Image:
    """
    Load an image from file as a PIL Image object.

    Args:
        path: Path to the image file (string or Path object)

    Returns:
        PIL Image object

    Raises:
        FileNotFoundError: If the image file doesn't exist
        IOError: If the image cannot be opened

    Example:
        >>> img = load_image("photo.jpg")
        >>> # Also works with Path objects:
        >>> from pathlib import Path
        >>> img = load_image(Path("photo.jpg"))
    """
    try:
        return Image.open(path)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Image file not found: {path}") from e
    except Exception as e:
        raise OSError(f"Failed to open image: {e}") from e


def save_image(image: Image.Image, path: str | Path, **kwargs: Any) -> None:
    """
    Save a PIL Image to file.

    Args:
        image: PIL Image object to save
        path: Path where the image should be saved (string or Path object)
        **kwargs: Additional arguments passed to PIL's save method
                 (e.g., format='JPEG', quality=95, optimize=True)

    Raises:
        IOError: If the image cannot be saved

    Example:
        >>> from PIL import Image
        >>> img = Image.new('RGB', (100, 100), color='red')
        >>> save_image(img, "output.png")
        >>> # With JPEG options:
        >>> save_image(img, "output.jpg", format='JPEG', quality=90)
    """
    try:
        image.save(path, **kwargs)
    except Exception as e:
        raise OSError(f"Failed to save image to {path}: {e}") from e


def image_to_base64(
    image: Image.Image, format: str = "PNG", quality: int | None = None
) -> str:
    """
    Convert a PIL Image to a base64 encoded string.

    Args:
        image: PIL Image object to encode
        format: Image format for encoding (e.g., "PNG", "JPEG").
                Default: "PNG"
        quality: JPEG quality (1-100, higher is better). Only used for JPEG.
                Default: None (uses PIL default of 75)

    Returns:
        Base64 encoded string of the image (without data URL prefix)

    Raises:
        ValueError: If the format is not supported
        IOError: If the image cannot be encoded

    Example:
        >>> from PIL import Image
        >>> img = Image.new('RGB', (100, 100), color='red')
        >>> b64_str = image_to_base64(img)
        >>> # To create data URL: f"data:image/png;base64,{b64_str}"
    """
    try:
        # Create a BytesIO buffer
        buffer = BytesIO()

        # Handle format-specific options
        save_kwargs = {"format": format}
        if format.upper() in ["JPEG", "JPG"] and quality is not None:
            save_kwargs["quality"] = quality
            save_kwargs["optimize"] = True

        # Save image to buffer
        image.save(buffer, **save_kwargs)

        # Get the bytes and encode to base64
        image_bytes = buffer.getvalue()
        base64_bytes = base64.b64encode(image_bytes)

        # Convert to string
        return base64_bytes.decode("utf-8")

    except Exception as e:
        raise OSError(f"Failed to encode image to base64: {e}") from e


def base64_to_image(base64_string: str) -> Image.Image:
    """
    Convert a base64 encoded string to a PIL Image.

    Args:
        base64_string: Base64 encoded image data. Can optionally include
                       data URL prefix (e.g., "data:image/png;base64,...")

    Returns:
        PIL Image object

    Raises:
        ValueError: If the base64 string is invalid
        IOError: If the image data cannot be decoded

    Example:
        >>> b64_str = "iVBORw0KGgoAAAANS..."  # Your base64 string
        >>> img = base64_to_image(b64_str)
        >>> # Also works with data URLs:
        >>> img = base64_to_image("data:image/png;base64,iVBORw0KGgoAAAANS...")
    """
    try:
        # Remove data URL prefix if present
        if base64_string.startswith("data:"):
            base64_string = base64_string.split(",", 1)[1]

        # Decode base64 to bytes
        image_bytes = base64.b64decode(base64_string)

        # Create PIL Image from bytes
        return Image.open(BytesIO(image_bytes))

    except (binascii.Error, ValueError) as e:
        raise ValueError(f"Invalid base64 string: {e}") from e
    except Exception as e:
        raise OSError(f"Failed to decode base64 to image: {e}") from e


def numpy_to_base64(array: np.ndarray, dtype: np.dtype | None = None) -> str:
    """
    Convert a numpy array to a base64 encoded string.

    Args:
        array: Numpy array to encode
        dtype: Data type to cast array to before encoding.
               If None, uses array's current dtype.

    Returns:
        Base64 encoded string of the numpy array

    Raises:
        ValueError: If the array cannot be encoded

    Example:
        >>> import numpy as np
        >>> arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        >>> b64_str = numpy_to_base64(arr)
    """
    try:
        # Cast to specified dtype if provided
        if dtype is not None:
            array = array.astype(dtype)

        # Convert to bytes and encode
        array_bytes = array.tobytes()
        base64_bytes = base64.b64encode(array_bytes)

        return base64_bytes.decode("utf-8")

    except Exception as e:
        raise ValueError(f"Failed to encode numpy array to base64: {e}") from e


def base64_to_numpy(
    base64_string: str,
    dtype: np.dtype = np.float32,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    """
    Convert a base64 encoded string to a numpy array.

    Args:
        base64_string: Base64 encoded numpy array data.
                      Can optionally include data URL prefix.
        dtype: Data type of the array. Default: np.float32
        shape: Shape to reshape the array to. If None, returns 1D array.

    Returns:
        Numpy array

    Raises:
        ValueError: If the base64 string is invalid or reshape fails

    Example:
        >>> b64_str = "AACAPwAAAEAAAEBA"  # Base64 encoded float32 array
        >>> arr = base64_to_numpy(b64_str, dtype=np.float32)
        >>> # With reshaping:
        >>> arr = base64_to_numpy(b64_str, dtype=np.float32, shape=(3,))
    """
    try:
        # Remove data URL prefix if present
        if base64_string.startswith("data:"):
            base64_string = base64_string.split(",", 1)[1]

        # Decode base64 to bytes
        array_bytes = base64.b64decode(base64_string)

        # Create numpy array from bytes
        array = np.frombuffer(array_bytes, dtype=dtype)

        # Reshape if shape is provided
        if shape is not None:
            array = array.reshape(shape)

        return array

    except (binascii.Error, ValueError) as e:
        raise ValueError(f"Invalid base64 string or shape: {e}") from e
    except Exception as e:
        raise ValueError(f"Failed to decode base64 to numpy array: {e}") from e


def save_base64_image(base64_string: str, output_path: str) -> bool:
    """
    Decode a base64 encoded image and save it to a file.

    Args:
        base64_string: The base64 encoded image string
        output_path: The path where the image file should be saved

    Returns:
        True if successful, False otherwise

    Example:
        >>> b64_str = "iVBORw0KGgoAAAANS..."
        >>> success = save_base64_image(b64_str, "output.png")
    """
    try:
        # Decode to PIL Image
        image = base64_to_image(base64_string)

        # Save to file
        image.save(output_path)
        return True

    except Exception:
        return False


def save_base64_numpy(
    base64_string: str,
    output_path: str,
    dtype: np.dtype = np.float32,
    shape: tuple[int, ...] | None = None,
) -> bool:
    """
    Decode a base64 encoded numpy array and save it to a .npy file.

    Args:
        base64_string: The base64 encoded numpy array string
        output_path: The path where the .npy file should be saved
        dtype: Data type of the array. Default: np.float32
        shape: Shape to reshape the array to. If None, saves as 1D array.

    Returns:
        True if successful, False otherwise

    Example:
        >>> b64_str = "AACAPwAAAEAAAEBA"
        >>> success = save_base64_numpy(b64_str, "output.npy", shape=(3,))
    """
    try:
        # Decode to numpy array
        array = base64_to_numpy(base64_string, dtype=dtype, shape=shape)

        # Save to file
        np.save(output_path, array)
        return True

    except Exception:
        return False


def resize_image(
    image: Image.Image, max_long: int = 2000, max_short: int = 768
) -> Image.Image:
    """
    Resize an image to fit within specified dimensions while maintaining aspect ratio.

    Args:
        image: PIL Image to resize
        max_long: Maximum length for the longer dimension
        max_short: Maximum length for the shorter dimension

    Returns:
        Resized PIL Image

    Example:
        >>> from PIL import Image
        >>> img = Image.open("large_image.jpg")
        >>> resized = resize_image(img, max_long=1920, max_short=1080)
    """
    width, height = image.size

    # Determine which dimension is longer
    if width > height:
        long_dim, short_dim = width, height
        max_long_dim, max_short_dim = max_long, max_short
    else:
        long_dim, short_dim = height, width
        max_long_dim, max_short_dim = max_long, max_short

    # Calculate scaling factor
    scale_factor = min(
        max_long_dim / long_dim,
        max_short_dim / short_dim,
        1.0,  # Don't upscale
    )

    if scale_factor < 1.0:
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    return image


def extract_base64_strings(source: str) -> list[str]:
    """
    Extract base64-encoded strings from pipe output.

    Splits concatenated base64 images by looking for PNG headers.
    It also removes any non-base64 content like "Recording time code:" messages
    that may appear before, between, or after the images.

    Args:
        source: The stdout containing concatenated base64 images
                and potentially other metadata interleaved

    Returns:
        List of base64 strings, one for each image

    Example:
        >>> stdout = "Recording time code: 0.0\\niVBORw0KGgo...<image1>...Recording time code: 1.0\\niVBORw0KGgo...<image2>..."
        >>> images = extract_base64_strings(stdout)
        >>> len(images)  # Should be 2
    """
    import re

    # First, remove all metadata patterns wherever they occur in the text
    # Common patterns include:
    # - "Recording time code: X.XXXXXX"
    # - "Camera: " (possibly followed by text)
    # - "Renderer plugin: HdStormRendererPlugin"
    # - "Running with Xvfb for GPU rendering..."
    # Remove metadata patterns that can appear anywhere in the text
    # This handles cases where metadata is embedded within lines containing base64 data
    cleaned_source = source

    # Remove "Recording time code: X.XXXXXX" patterns
    cleaned_source = re.sub(r"Recording time code:\s*[\d.]+\s*", "", cleaned_source)

    # Remove other metadata patterns that appear at the start of lines
    # Split by newlines to handle line-based metadata
    lines = cleaned_source.split("\n")
    filtered_lines = []

    for line in lines:
        # Remove lines that are purely metadata (not containing base64)
        # Skip lines that start with known metadata patterns
        if re.match(r"^(Camera:|Renderer plugin:|Running with)", line.strip()):
            continue
        # Skip empty lines
        if not line.strip():
            continue
        # Keep the line (it may contain base64 data)
        filtered_lines.append(line)

    # Rejoin the filtered lines
    cleaned_source = "\n".join(filtered_lines)

    # Remove any remaining commas and whitespace that might be separators
    # but keep the base64 data intact

    # PNG files start with "iVBORw0KGgo" in base64
    png_header = "iVBORw0KGgo"

    # Find all positions where the PNG header appears
    positions = []
    start = 0
    while True:
        pos = cleaned_source.find(png_header, start)
        if pos == -1:
            break
        positions.append(pos)
        start = pos + 1

    # Split the string at each PNG header position
    if len(positions) > 0:
        images = []
        for i in range(len(positions)):
            start_pos = positions[i]
            end_pos = (
                positions[i + 1] if i + 1 < len(positions) else len(cleaned_source)
            )
            image_data = cleaned_source[start_pos:end_pos]

            # Clean up the image data by removing any non-base64 characters
            # Base64 uses A-Z, a-z, 0-9, +, /, and = for padding
            # Remove any other characters (like commas, newlines, etc.)
            image_data = re.sub(r"[^A-Za-z0-9+/=]", "", image_data)

            if image_data:  # Only add non-empty strings
                images.append(image_data)
        return images
    else:
        # No images found
        return []


def create_data_url(image: Image.Image, format: str = "PNG") -> str:
    """
    Create a data URL from a PIL Image.

    Args:
        image: PIL Image object
        format: Image format (e.g., "PNG", "JPEG")

    Returns:
        Complete data URL string

    Example:
        >>> from PIL import Image
        >>> img = Image.new('RGB', (100, 100), color='blue')
        >>> data_url = create_data_url(img)
        >>> # Returns: "data:image/png;base64,iVBORw0KGgoAAAANS..."
    """
    base64_str = image_to_base64(image, format=format)
    mime_type = f"image/{format.lower()}"
    return f"data:{mime_type};base64,{base64_str}"


def parse_data_url(data_url: str) -> tuple[str, str]:
    """
    Parse a data URL to extract mime type and base64 data.

    Args:
        data_url: Complete data URL string

    Returns:
        Tuple of (mime_type, base64_data)

    Raises:
        ValueError: If the data URL format is invalid

    Example:
        >>> data_url = "data:image/png;base64,iVBORw0KGgoAAAANS..."
        >>> mime_type, b64_data = parse_data_url(data_url)
        >>> # mime_type = "image/png", b64_data = "iVBORw0KGgoAAAANS..."
    """
    if not data_url.startswith("data:"):
        raise ValueError("Invalid data URL: must start with 'data:'")

    try:
        header, data = data_url.split(",", 1)
        mime_type = header.split(":")[1].split(";")[0]
        return mime_type, data
    except (IndexError, ValueError) as e:
        raise ValueError(f"Invalid data URL format: {e}") from e


def extract_color_outline(
    input_image: Image.Image,
    target_channel: str = "r",
    channel_min: int = 150,
    other_channel_max: int = 100,
    thickness: int = 1,
) -> Image.Image:
    """
    Extracts the outline of red pixel regions in a Pillow image with configurable thickness.

    Parameters:
    - input_image: PIL.Image.Image, the input RGB image.
    - target_channel: str, one of "r", "g", or "b".
    - channel_min: int, the minimum value for the target channel.
    - other_channel_max: int, the maximum value for the other channels.
    - thickness: int, the thickness of the outline in pixels.

    Returns:
    - PIL.Image.Image, a grayscale image with the outline in white on black background.
    """
    try:
        from scipy.ndimage import binary_dilation
    except ImportError as e:
        raise ImportError(
            "scipy is required for extract_color_outline function. "
            "Please install it with: pip install scipy>=1.16.1 "
            "or install the complete package with: pip install -e ."
        ) from e

    image_rgb = input_image.convert("RGB")
    image_np = np.array(image_rgb)

    if target_channel == "r":
        # Red pixel mask
        color_mask = (
            (image_np[:, :, 0] > channel_min)
            & (image_np[:, :, 1] < other_channel_max)
            & (image_np[:, :, 2] < other_channel_max)
        )
    elif target_channel == "g":
        # Green pixel mask
        color_mask = (
            (image_np[:, :, 0] < other_channel_max)
            & (image_np[:, :, 1] > channel_min)
            & (image_np[:, :, 2] < other_channel_max)
        )
    elif target_channel == "b":
        # Blue pixel mask
        color_mask = (
            (image_np[:, :, 0] < other_channel_max)
            & (image_np[:, :, 1] < other_channel_max)
            & (image_np[:, :, 2] > channel_min)
        )
    else:
        raise ValueError(f"Invalid target channel: {target_channel}")

    # Outline using dilation with thickness
    dilated = binary_dilation(color_mask, iterations=thickness)
    outline_mask = dilated ^ color_mask

    outline_image = Image.fromarray(np.uint8(outline_mask) * 255, mode="L")
    return outline_image


def extract_red_outline(
    input_image: Image.Image,
    channel_min: int = 150,
    other_channel_max: int = 100,
    thickness: int = 1,
) -> Image.Image:
    """
    Extracts the outline of red pixels in the image.

    Args:
        input_image: The input image to extract the outline from.
        channel_min: The minimum value for the red channel.
        other_channel_max: The maximum value for the other channels.
        thickness: The thickness of the outline.

    Returns:
        The outline of the red pixels in the image.
    """
    return extract_color_outline(
        input_image, "r", channel_min, other_channel_max, thickness
    )


def extract_green_outline(
    input_image: Image.Image,
    channel_min: int = 150,
    other_channel_max: int = 100,
    thickness: int = 1,
) -> Image.Image:
    """
    Extracts the outline of green pixels in the image.

    Args:
        input_image: The input image to extract the outline from.
        channel_min: The minimum value for the green channel.
        other_channel_max: The maximum value for the other channels.
        thickness: The thickness of the outline.

    Returns:
        The outline of the green pixels in the image.
    """
    return extract_color_outline(
        input_image, "g", channel_min, other_channel_max, thickness
    )


def extract_blue_outline(
    input_image: Image.Image,
    channel_min: int = 150,
    other_channel_max: int = 100,
    thickness: int = 1,
) -> Image.Image:
    """
    Extracts the outline of blue pixels in the image.

    Args:
        input_image: The input image to extract the outline from.
        channel_min: The minimum value for the blue channel.
        other_channel_max: The maximum value for the other channels.
        thickness: The thickness of the outline.

    Returns:
        The outline of the blue pixels in the image.

    """
    return extract_color_outline(
        input_image, "b", channel_min, other_channel_max, thickness
    )


def extract_non_black_outline(
    input_image: Image.Image,
    black_threshold: int = 20,
    thickness: int = 1,
) -> Image.Image:
    """
    Extracts the outline of any non-black pixels in the image.

    This function detects any pixel that is not black (or close to black)
    and extracts the outline of those regions.

    Args:
        input_image: The input image to extract the outline from.
        black_threshold: Threshold below which pixels are considered black.
                        Pixels with RGB values all <= this threshold are treated as black.
        thickness: The thickness of the outline in pixels.

    Returns:
        A grayscale image with the outline in white on black background.
    """
    try:
        from scipy.ndimage import binary_dilation
    except ImportError as e:
        raise ImportError(
            "scipy is required for extract_non_black_outline function. "
            "Please install it with: pip install scipy>=1.16.1 "
            "or install the complete package with: pip install -e ."
        ) from e

    image_rgb = input_image.convert("RGB")
    image_np = np.array(image_rgb)

    # Create mask for non-black pixels
    # A pixel is considered non-black if any RGB channel > black_threshold
    non_black_mask = (
        (image_np[:, :, 0] > black_threshold)
        | (image_np[:, :, 1] > black_threshold)
        | (image_np[:, :, 2] > black_threshold)
    )

    # Outline using dilation with thickness
    dilated = binary_dilation(non_black_mask, iterations=thickness)
    outline_mask = dilated ^ non_black_mask

    outline_image = Image.fromarray(np.uint8(outline_mask) * 255, mode="L")
    return outline_image


def paste_outline_to_image(
    image: Image.Image,
    outline: Image.Image,
    outline_color: tuple[int, int, int] = (255, 0, 0),
) -> Image.Image:
    """
    Pastes an outline onto an image.

    Args:
        image: The image to paste the outline onto.
        outline: The outline to paste.
        outline_color: The color of the outline.

    Returns:
        The image with the outline pasted onto it.
    """
    image_rgb = image.convert("RGB")
    image_np = np.array(image_rgb)
    outline_np = np.array(outline)
    image_np[outline_np == 255] = outline_color
    return Image.fromarray(image_np)


def paste_on_background(
    image_rgba: Image.Image,
    background_color: tuple[int, int, int] | tuple[float, float, float] = (
        255,
        255,
        255,
    ),
) -> Image.Image:
    """
    Pastes an RGBA image onto a solid color background and returns an RGB image.

    Args:
        image_rgba: The input image in RGBA mode.
        background_color: The background color as (R,G,B) tuple.

    Returns:
        The RGB image with transparency composited over the background.
    """
    if image_rgba.mode != "RGBA":
        raise ValueError("Input image must be in RGBA mode.")

    if len(background_color) != 3:
        raise ValueError("background_color must have exactly 3 channels (R, G, B).")

    # Accept either normalized floats (0..1) or integer RGB values (0..255).
    if any(isinstance(channel, float) for channel in background_color):
        rgb_color = tuple(
            int(max(0, min(255, round(float(channel) * 255))))
            for channel in background_color
        )
    else:
        rgb_color = tuple(
            int(max(0, min(255, int(channel)))) for channel in background_color
        )

    background = Image.new("RGB", image_rgba.size, rgb_color)
    # Use alpha channel as mask
    background.paste(image_rgba, mask=image_rgba.split()[3])
    return background


def draw_bounding_box_on_color(
    image: Image.Image,
    target_channel: str = "r",
    channel_min: int = 150,
    other_channel_max: int = 100,
    box_width: int = 2,
) -> Image.Image:
    """
    Draws a bounding box around pixels of a specific color in the input image.

    Args:
        image: The input RGB or RGBA image.
        target_channel: The target color channel ("r", "g", or "b").
        channel_min: The minimum value for the target channel.
        other_channel_max: The maximum value for the other channels.
        box_width: The width of the bounding box lines.

    Returns:
        A grayscale mask image containing the bounding box.
    """
    image_rgb = image.convert("RGB")
    image_np = np.array(image_rgb)

    if target_channel == "r":
        # Red pixel mask
        color_mask = (
            (image_np[:, :, 0] > channel_min)
            & (image_np[:, :, 1] < other_channel_max)
            & (image_np[:, :, 2] < other_channel_max)
        )
    elif target_channel == "g":
        # Green pixel mask
        color_mask = (
            (image_np[:, :, 0] < other_channel_max)
            & (image_np[:, :, 1] > channel_min)
            & (image_np[:, :, 2] < other_channel_max)
        )
    elif target_channel == "b":
        # Blue pixel mask
        color_mask = (
            (image_np[:, :, 0] < other_channel_max)
            & (image_np[:, :, 1] < other_channel_max)
            & (image_np[:, :, 2] > channel_min)
        )
    else:
        raise ValueError(f"Invalid target channel: {target_channel}")

    # Get coordinates of red pixels
    coords = np.argwhere(color_mask)

    if coords.size == 0:
        return Image.new("L", image.size, 0)  # Return empty mask

    # Get bounding box (min_row, min_col), (max_row, max_col)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)

    # Draw multiple rectangles to simulate thickness
    for t in range(box_width):
        draw.rectangle((x_min - t, y_min - t, x_max + t, y_max + t), outline=255)

    return mask


def draw_bounding_box_on_red(
    image: Image.Image,
    channel_min: int = 150,
    other_channel_max: int = 100,
    box_width: int = 2,
) -> Image.Image:
    """Draw a bounding box around red pixels in an image.

    Parameters:
        image: Input PIL image
        channel_min: The minimum value for the red channel.
        other_channel_max: The maximum value for the other channels.
        box_width: Width of the bounding box border in pixels

    Returns:
        A grayscale mask image containing the bounding box
    """
    return draw_bounding_box_on_color(
        image, "r", channel_min, other_channel_max, box_width
    )


def draw_bounding_box_on_green(
    image: Image.Image,
    channel_min: int = 150,
    other_channel_max: int = 100,
    box_width: int = 2,
) -> Image.Image:
    """Draw a bounding box around green pixels in an image.

    Parameters:
        image: Input PIL image
        channel_min: The minimum value for the green channel.
        other_channel_max: The maximum value for the other channels.
        box_width: Width of the bounding box border in pixels

    Returns:
        A grayscale mask image containing the bounding box
    """
    return draw_bounding_box_on_color(
        image, "g", channel_min, other_channel_max, box_width
    )


def draw_bounding_box_on_blue(
    image: Image.Image,
    channel_min: int = 150,
    other_channel_max: int = 100,
    box_width: int = 2,
) -> Image.Image:
    """Draw a bounding box around blue pixels in an image.

    Parameters:
        image: Input PIL image
        channel_min: The minimum value for the blue channel.
        other_channel_max: The maximum value for the other channels.
        box_width: Width of the bounding box border in pixels

    Returns:
        A grayscale mask image containing the bounding box
    """
    return draw_bounding_box_on_color(
        image, "b", channel_min, other_channel_max, box_width
    )


def is_prim_visible_in_image(
    image: Image.Image,
    contour_method: str = "red",
    pixel_threshold: int = 10,
    channel_min: int = 150,
    other_channel_max: int = 100,
    black_threshold: int = 20,
) -> bool:
    """Check if a prim is visible in a highlight image.

    This function determines if a prim is visible (not completely occluded) by counting
    the number of visible pixels in the highlight image.

    Args:
        image: The highlight image (with prim rendered in isolation).
        contour_method: Method to detect visible pixels - "red" or "non_black".
        pixel_threshold: Minimum number of visible pixels to consider prim visible.
        channel_min: Minimum value for the target channel in "red" method.
        other_channel_max: Maximum value for other channels in "red" method.
        black_threshold: Threshold for "non_black" method.

    Returns:
        True if the prim has at least pixel_threshold visible pixels, False otherwise.

    Example:
        >>> from PIL import Image
        >>> img = Image.open("highlight.png")
        >>> is_visible = is_prim_visible_in_image(img, "red", pixel_threshold=10)
    """
    image_rgb = image.convert("RGB")
    image_np = np.array(image_rgb)

    if contour_method == "non_black":
        # Count non-black pixels
        visible_mask = (
            (image_np[:, :, 0] > black_threshold)
            | (image_np[:, :, 1] > black_threshold)
            | (image_np[:, :, 2] > black_threshold)
        )
    else:  # Default to "red" method
        # Count red pixels
        visible_mask = (
            (image_np[:, :, 0] > channel_min)
            & (image_np[:, :, 1] < other_channel_max)
            & (image_np[:, :, 2] < other_channel_max)
        )

    # Count visible pixels
    visible_pixel_count = np.sum(visible_mask)

    return visible_pixel_count >= pixel_threshold


def process_depth_map(
    depth_map: np.ndarray, min_output_value: float = 0.01
) -> np.ndarray:
    """Process depth map by inverting it with specific value ranges.

    This works for both depth types, but linear_depth (distance_to_image_plane) is preferred
    as it's the standard Z-depth used in computer vision. The depth sensor (distance_to_camera)
    provides radial distance which varies across the image even for planar surfaces.

    Output mapping:
    - Background (inf) → 0
    - Closest valid object → 1.0
    - Farthest valid object → min_output_value (close to 0 but not 0)

    Args:
        depth_map (np.ndarray): Raw depth data from sensor (linear_depth or depth)
        min_output_value (float): Minimum output value for farthest valid objects.
                                   Should be close to 0 but not 0. Default is 0.01.

    Returns:
        np.ndarray: Processed depth map where background=0, closest=1, farthest=min_output_value
    """
    depth = depth_map.copy()

    # Find background depth values for processing
    background_mask = np.isinf(depth_map)

    # Set background depth values to 0 (black/background)
    depth[background_mask] = 0
    valid_mask = ~background_mask
    if np.any(valid_mask):
        min_depth = np.min(depth[valid_mask])
        max_depth = np.max(depth[valid_mask])

        if max_depth > min_depth:
            # Invert and scale: closest → 1.0, farthest → min_output_value
            # Formula: inverted = (max - depth) / (max - min)
            # Then scale from [0, 1] to [min_output_value, 1.0]
            inverted_normalized = (max_depth - depth[valid_mask]) / (
                max_depth - min_depth
            )
            depth[valid_mask] = min_output_value + inverted_normalized * (
                1.0 - min_output_value
            )
        else:
            # All valid depths are the same, set to 1 (closest)
            depth[valid_mask] = 1.0

    return depth
