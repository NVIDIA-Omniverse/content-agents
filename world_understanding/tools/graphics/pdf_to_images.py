# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tool wrapper for PDF to images conversion."""

from pydantic import BaseModel, Field
from rich.console import Console

from world_understanding.functions.graphics.pdf_to_images import convert_pdf_to_images
from world_understanding.tools.base import (
    ExecutionPolicy,
    ToolInput,
    ToolOutput,
    register_tool,
)


class ConvertPdfToImagesInput(ToolInput):
    """Input for PDF to images conversion tool."""

    pdf_path: str = Field(..., description="Path to PDF file to convert")
    output_dir: str | None = Field(
        None, description="Directory to save images (None for in-memory only)"
    )
    dpi: int = Field(
        default=300,
        ge=72,
        le=600,
        description="Resolution in DPI (72-600, higher = better quality)",
    )
    format: str = Field(
        default="png",
        description="Image format: png, jpeg, jpg, tiff, or ppm",
    )
    first_page: int | None = Field(
        None, ge=1, description="First page to convert (1-indexed, None = start)"
    )
    last_page: int | None = Field(
        None, ge=1, description="Last page to convert (1-indexed, None = end)"
    )
    grayscale: bool = Field(default=False, description="Convert images to grayscale")


class PageImageInfo(BaseModel):
    """Information about a converted PDF page."""

    page_number: int = Field(..., description="Page number (1-indexed)")
    image_path: str | None = Field(
        None, description="Path to saved image file (if saved)"
    )
    width: int = Field(..., description="Image width in pixels")
    height: int = Field(..., description="Image height in pixels")


class ConvertPdfToImagesOutput(ToolOutput):
    """Output for PDF to images conversion tool."""

    pages: list[PageImageInfo] = Field(
        ..., description="Information about converted pages"
    )
    total_pages: int = Field(..., description="Total pages in PDF")
    converted_pages: int = Field(..., description="Number of pages converted")
    output_directory: str | None = Field(
        None, description="Directory where images were saved"
    )


def _display_pdf_conversion(
    outputs: ConvertPdfToImagesOutput, console: Console, indent: str = ""
) -> None:
    """Display PDF conversion results in a formatted way."""
    console.print(f"{indent}[bold]PDF Conversion Results:[/bold]")
    console.print(
        f"{indent}Converted: {outputs.converted_pages}/{outputs.total_pages} pages"
    )

    if outputs.output_directory:
        console.print(f"{indent}Output Directory: {outputs.output_directory}")

    console.print(f"{indent}[bold]Pages:[/bold]")
    for page in outputs.pages:
        page_num = page.page_number
        width = page.width
        height = page.height
        path = page.image_path if page.image_path else "in-memory"
        console.print(f"{indent}  Page {page_num}: {width}x{height}px - {path}")


@register_tool(
    name="convert_pdf_to_images",
    version="0.1.0",
    description="Convert PDF pages to images with configurable DPI and format",
    tags=["pdf", "conversion", "image", "document", "graphics", "cpu"],
    input_model=ConvertPdfToImagesInput,
    output_model=ConvertPdfToImagesOutput,
    policy=ExecutionPolicy(timeout_s=300.0),  # 5 minutes for large PDFs
)
def convert_pdf_to_images_tool(
    inputs: ConvertPdfToImagesInput,
) -> ConvertPdfToImagesOutput:
    """
    Convert PDF pages to images.

    This tool uses pypdfium2 for high-performance PDF rendering. It supports:
    - Multiple image formats (PNG, JPEG, TIFF)
    - Configurable DPI (72-600)
    - Page range selection
    - Grayscale conversion
    - Save to disk or in-memory processing

    Args:
        inputs: ConvertPdfToImagesInput containing:
            - pdf_path: Path to PDF file
            - output_dir: Directory to save images (optional)
            - dpi: Resolution in DPI (default: 300)
            - format: Image format (default: "png")
            - first_page: First page to convert (optional)
            - last_page: Last page to convert (optional)
            - grayscale: Convert to grayscale (default: False)

    Returns:
        ConvertPdfToImagesOutput containing:
            - pages: List of PageImageInfo with details for each page
            - total_pages: Total pages in the PDF
            - converted_pages: Number of pages converted
            - output_directory: Directory where images were saved (if applicable)

    Example:
        >>> from world_understanding.tools.graphics import convert_pdf_to_images_tool
        >>> from world_understanding.tools.graphics import ConvertPdfToImagesInput
        >>>
        >>> inputs = ConvertPdfToImagesInput(
        ...     pdf_path="document.pdf",
        ...     output_dir="output",
        ...     dpi=300,
        ...     format="png"
        ... )
        >>> output = convert_pdf_to_images_tool(inputs)
        >>> print(f"Converted {output.converted_pages} pages")
    """
    # Call the core function
    results = convert_pdf_to_images(
        pdf_path=inputs.pdf_path,
        output_dir=inputs.output_dir,
        dpi=inputs.dpi,
        fmt=inputs.format,
        first_page=inputs.first_page,
        last_page=inputs.last_page,
        grayscale=inputs.grayscale,
    )

    # Convert results to PageImageInfo objects
    pages = []
    for result in results:
        pages.append(
            PageImageInfo(
                page_number=result["page_number"],
                image_path=result["image_path"],
                width=result["width"],
                height=result["height"],
            )
        )

    # Get total pages (from the last result's page number if we converted to end)
    # Or calculate from the results
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(inputs.pdf_path)
    total_pages = len(doc)
    doc.close()

    return ConvertPdfToImagesOutput(
        pages=pages,
        total_pages=total_pages,
        converted_pages=len(pages),
        output_directory=inputs.output_dir,
    )


# Attach display function to the tool
convert_pdf_to_images_tool._display_function = _display_pdf_conversion
