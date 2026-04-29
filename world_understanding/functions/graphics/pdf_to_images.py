# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Core function for converting PDF files to images using pypdfium2."""

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def convert_pdf_to_images(
    pdf_path: str | Path,
    output_dir: str | Path | None = None,
    dpi: int = 300,
    fmt: str = "png",
    first_page: int | None = None,
    last_page: int | None = None,
    grayscale: bool = False,
) -> list[dict[str, Any]]:
    """
    Convert PDF pages to images using pypdfium2.

    This function uses pypdfium2 for high-performance PDF rendering.
    It can save images to disk or return them in-memory.

    Args:
        pdf_path: Path to the PDF file to convert
        output_dir: Directory to save images (None = in-memory only)
        dpi: Resolution in dots per inch (72-600, default: 300)
        fmt: Image format - "png", "jpeg", "jpg", "tiff", or "ppm"
        first_page: First page to convert (1-indexed, None = start from first page)
        last_page: Last page to convert (1-indexed, None = convert to last page)
        grayscale: Whether to convert images to grayscale

    Returns:
        List of dictionaries, one per converted page, containing:
        - page_number: Page number (1-indexed)
        - image_path: Path to saved image file (if output_dir provided)
        - image: PIL Image object (in-memory)
        - width: Image width in pixels
        - height: Image height in pixels

    Raises:
        FileNotFoundError: If PDF file does not exist
        ValueError: If parameters are invalid
        RuntimeError: If PDF processing fails

    Example:
        >>> results = convert_pdf_to_images(
        ...     "document.pdf",
        ...     output_dir="output",
        ...     dpi=300,
        ...     fmt="png"
        ... )
        >>> for page in results:
        ...     print(f"Page {page['page_number']}: {page['width']}x{page['height']}")
    """
    # Lazy import pypdfium2
    try:
        import pypdfium2 as pdfium
    except ImportError as e:
        raise RuntimeError(
            "pypdfium2 is required for PDF to image conversion. "
            "Install it with: pip install pypdfium2"
        ) from e

    # Validate inputs
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    if not pdf_path.is_file():
        raise ValueError(f"Path is not a file: {pdf_path}")

    if dpi < 72 or dpi > 600:
        raise ValueError(f"DPI must be between 72 and 600, got: {dpi}")

    # Normalize format
    fmt = fmt.lower()
    valid_formats = ("png", "jpeg", "jpg", "tiff", "ppm")
    if fmt not in valid_formats:
        raise ValueError(f"Invalid format: {fmt}. Must be one of {valid_formats}")

    # Create output directory if needed
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Created output directory: %s", output_dir)

    # Open PDF document
    try:
        doc = pdfium.PdfDocument(pdf_path)
    except Exception as e:
        raise RuntimeError(f"Failed to open PDF: {pdf_path}") from e

    try:
        total_pages = len(doc)
        logger.info("PDF has %d pages", total_pages)

        # Determine page range
        start_page = (first_page - 1) if first_page is not None else 0
        end_page = last_page if last_page is not None else total_pages

        # Validate page range
        if start_page < 0 or start_page >= total_pages:
            raise ValueError(
                f"first_page must be between 1 and {total_pages}, got: {first_page}"
            )

        if end_page < 1 or end_page > total_pages:
            raise ValueError(
                f"last_page must be between 1 and {total_pages}, got: {last_page}"
            )

        if start_page >= end_page:
            raise ValueError(
                f"first_page ({first_page}) must be less than last_page ({last_page})"
            )

        # Calculate scale factor for DPI
        # PDF default is 72 DPI, so scale = target_dpi / 72
        scale = dpi / 72.0

        results = []

        # Process each page
        for page_idx in range(start_page, end_page):
            page_number = page_idx + 1  # 1-indexed for user
            logger.debug("Processing page %d/%d", page_number, total_pages)

            try:
                # Get page
                page = doc[page_idx]

                # Render page to bitmap
                bitmap = page.render(scale=scale)

                # Convert to PIL Image
                img = bitmap.to_pil()

                # Convert to grayscale if requested
                if grayscale:
                    img = img.convert("L")

                # Get dimensions
                width, height = img.size

                # Save to disk if output_dir specified
                image_path = None
                if output_dir is not None:
                    # Create filename: pdf_name_page_001.png
                    filename = f"{pdf_path.stem}_page_{page_number:03d}.{fmt}"
                    image_path = output_dir / filename

                    # Save image
                    img.save(image_path, format=fmt.upper())
                    logger.debug("Saved page %d to %s", page_number, image_path)

                # Add to results
                results.append(
                    {
                        "page_number": page_number,
                        "image_path": str(image_path) if image_path else None,
                        "image": img,
                        "width": width,
                        "height": height,
                    }
                )

            except Exception as e:
                logger.error("Failed to process page %d: %s", page_number, e)
                raise RuntimeError(f"Failed to process page {page_number}") from e

        logger.info(
            "Successfully converted %d pages from %s", len(results), pdf_path.name
        )

        return results

    finally:
        # Always close the document
        doc.close()
