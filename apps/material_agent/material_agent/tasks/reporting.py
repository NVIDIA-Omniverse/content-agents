# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tasks for generating HTML reports from predictions and evaluation results."""

import base64
import io
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image as PILImage
from world_understanding.agentic.events import get_listener
from world_understanding.agentic.tasks import Task
from world_understanding.agentic.utils.html_report import (
    escape_html,
    format_cost_estimate_section,
    format_images_html,
    format_system_prompt_section,
    validate_image_options,
)
from world_understanding.utils.object_store import ObjectStore

logger = logging.getLogger(__name__)


class BaseReportTask(Task):
    """Base class for report generation tasks with shared image processing utilities."""

    def _process_and_encode_image(
        self,
        img_path: Path,
        image_max_size: int | None = None,
        image_format: str = "png",
        image_quality: int = 85,
    ) -> str:
        """Process an image (resize, convert format) and return base64-encoded data.

        Args:
            img_path: Path to the image file
            image_max_size: Optional max size in pixels (e.g., 256 for max 256x256)
            image_format: Output format ('png' or 'jpeg')
            image_quality: JPEG quality (1-100)

        Returns:
            Base64-encoded image data as string
        """
        try:
            # Load image
            img = PILImage.open(img_path)

            # Convert RGBA to RGB if needed (JPEG doesn't support transparency)
            if image_format == "jpeg" and img.mode in ("RGBA", "LA", "P"):
                # Create white background
                background = PILImage.new("RGB", img.size, (255, 255, 255))
                if img.mode == "P":
                    img = img.convert("RGBA")
                background.paste(
                    img, mask=img.split()[-1] if img.mode == "RGBA" else None
                )
                img = background

            # Resize if max_size is specified
            if image_max_size:
                img.thumbnail(
                    (image_max_size, image_max_size), PILImage.Resampling.LANCZOS
                )

            # Encode to bytes in the specified format
            img_bytes = io.BytesIO()
            save_format = "JPEG" if image_format == "jpeg" else "PNG"

            if image_format == "jpeg":
                # JPEG with quality control
                img.save(
                    img_bytes, format=save_format, quality=image_quality, optimize=True
                )
            else:
                # PNG with optimization
                img.save(img_bytes, format=save_format, optimize=True)

            # Base64 encode
            img_bytes.seek(0)
            img_data = base64.b64encode(img_bytes.getvalue()).decode()
            return img_data

        except Exception as e:
            logger.warning(f"Failed to process image {img_path}: {e}")
            # Fallback: try to read and encode original file
            try:
                with open(img_path, "rb") as f:
                    return base64.b64encode(f.read()).decode()
            except Exception as e2:
                logger.error(f"Failed to encode image {img_path}: {e2}")
                raise

    def _format_images(
        self,
        images: list[str],
        dataset_path: str | None,
        image_metadata: list[dict] | None = None,
        listener=None,
        image_max_size: int | None = None,
        image_format: str = "png",
        image_quality: int = 85,
    ) -> str:
        """Format images as HTML thumbnails with base64 encoding and captions.

        Args:
            images: List of image paths
            dataset_path: Path to dataset file (used to determine base directory)
            image_metadata: Optional list of image metadata with prompts
            listener: Event listener for progress reporting (optional)
            image_max_size: Optional max image size in pixels (e.g., 256 for 256x256)
            image_format: Image format ('png' or 'jpeg', default: 'png')
            image_quality: JPEG quality (1-100, default: 85)

        Returns:
            HTML string with image thumbnails and captions
        """
        if not images:
            return "No images"

        # Determine base directory
        base_dir = None
        if dataset_path:
            base_dir = Path(dataset_path).parent

        images_html = []
        for idx, img in enumerate(images):
            # Get metadata for this image if available
            metadata = None
            image_prompt = None
            if image_metadata and idx < len(image_metadata):
                metadata = image_metadata[idx]
                image_prompt = metadata.get("vlm_prompt", "")

            try:
                # Resolve image path
                if base_dir:
                    img_path = (
                        base_dir / img if not Path(img).is_absolute() else Path(img)
                    )
                else:
                    img_path = Path(img)

                if img_path.exists():
                    # Process image: load, resize (optional), convert format (optional), encode
                    img_data = self._process_and_encode_image(
                        img_path,
                        image_max_size=image_max_size,
                        image_format=image_format,
                        image_quality=image_quality,
                    )

                    # Determine MIME type based on output format
                    mime_type = "image/jpeg" if image_format == "jpeg" else "image/png"

                    data_url = f"data:{mime_type};base64,{img_data}"

                    # Create image with caption if prompt is available
                    if image_prompt:
                        # Escape HTML in prompt
                        prompt_html = (
                            str(image_prompt)
                            .replace("&", "&amp;")
                            .replace("<", "&lt;")
                            .replace(">", "&gt;")
                            .replace('"', "&quot;")
                            .replace("'", "&#39;")
                        )

                        images_html.append(
                            f"""<div class="image-with-caption">
                                <img src="{data_url}" class="image-thumbnail" onclick="showImage(\'{data_url}\')" alt="{img}">
                                <div class="image-caption">{prompt_html}</div>
                            </div>"""
                        )
                    else:
                        images_html.append(
                            f'<img src="{data_url}" class="image-thumbnail" onclick="showImage(\'{data_url}\')" alt="{img}">'
                        )
                else:
                    images_html.append(
                        f'<span style="color: #999;">{img} (not found)</span>'
                    )
            except Exception as e:
                images_html.append(
                    f'<span style="color: #F44336;">{img} (error)</span>'
                )
                if listener:
                    listener.debug(f"Failed to load image {img}: {e}")

        return '<div class="image-container">' + "".join(images_html) + "</div>"


class GeneratePredictionReportTask(BaseReportTask):
    """Generate HTML report from prediction results.

    This task creates a visual HTML report showing prediction results including
    input prompts, images, predicted materials, and success statistics.

    Input context keys:
        - predictions: List of successful predictions
        - failed_predictions: List of failed predictions
        - dataset: Original dataset entries
        - output_dir: Directory to save the HTML report
        - dataset_path: Path to dataset file (for resolving relative image paths)
        - config: Optional configuration dict (for system prompt)
        - report_image_max_size: Optional max image size in pixels (e.g., 256 for 256x256)
        - report_image_format: Optional image format ('png' or 'jpeg', default: 'png')
        - report_image_quality: Optional JPEG quality (1-100, default: 85)

    Output context keys:
        - html_report_path: Path to generated HTML report
    """

    def __init__(
        self,
        image_max_size: int | None = None,
        image_format: str | None = None,
        image_quality: int | None = None,
    ):
        """Initialize the prediction report generation task.

        Args:
            image_max_size: Optional max image size in pixels (e.g., 256 for 256x256).
                           If set, images are resized to fit within this size.
                           Can be overridden by context key 'report_image_max_size'.
            image_format: Optional image format ('png' or 'jpeg', default: 'png').
                         Can be overridden by context key 'report_image_format'.
            image_quality: Optional JPEG quality (1-100, default: 85).
                          Only used if format is 'jpeg'.
                          Can be overridden by context key 'report_image_quality'.
        """
        self.name = "GeneratePredictionReport"
        self.description = "Generate HTML report from prediction results"
        self.image_max_size = image_max_size
        self.image_format = image_format
        self.image_quality = image_quality

    def run(
        self, context: dict[str, Any], object_store: ObjectStore | None = None
    ) -> dict[str, Any]:
        """Generate HTML report from prediction results.

        Args:
            context: Workflow context containing predictions and dataset
            object_store: Optional object store

        Returns:
            Updated context with html_report_path
        """
        # Get event listener (or logger fallback)
        listener = get_listener(context, logger_name=__name__)
        # Get data from context or object store
        predictions = context.get("predictions", [])
        failed = context.get("failed_predictions", [])
        dataset = context.get("dataset", [])

        if object_store:
            predictions = object_store.get("predictions", predictions)
            failed = object_store.get("failed_predictions", failed)
            dataset = object_store.get("dataset", dataset)

        output_dir = Path(context.get("output_dir", "."))
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get and validate image processing options
        image_max_size = (
            context.get("report_image_max_size")
            if context.get("report_image_max_size") is not None
            else self.image_max_size
        )
        image_format = context.get("report_image_format", self.image_format)
        image_quality = context.get("report_image_quality", self.image_quality)

        image_format, image_quality, image_max_size = validate_image_options(
            image_format, image_quality, image_max_size
        )

        try:
            html_file = self._generate_report(
                predictions=predictions,
                failed=failed,
                dataset=dataset,
                output_dir=output_dir,
                context=context,
                listener=listener,
                image_max_size=image_max_size,
                image_format=image_format,
                image_quality=image_quality,
            )

            if html_file:
                context["html_report_path"] = str(html_file)
                listener.info(f"✓ HTML report generated: {html_file}")

        except Exception as e:
            listener.warning(f"Failed to generate HTML report: {e}")

        return context

    def _generate_report(
        self,
        predictions: list[dict],
        failed: list[dict],
        dataset: list[dict],
        output_dir: Path,
        context: dict[str, Any],
        listener,
        image_max_size: int | None = None,
        image_format: str = "png",
        image_quality: int = 85,
    ) -> Path | None:
        """Generate an HTML report for prediction results.

        Args:
            predictions: List of successful predictions
            failed: List of failed predictions
            dataset: Original dataset
            output_dir: Directory to save the HTML report
            context: Workflow context for additional information
            listener: Event listener for progress reporting
            image_max_size: Optional max image size in pixels (e.g., 256 for 256x256)
            image_format: Image format ('png' or 'jpeg', default: 'png')
            image_quality: JPEG quality (1-100, default: 85)

        Returns:
            Path to the generated HTML file or None if error
        """
        try:
            html_file = output_dir / "prediction_report.html"

            # Create mapping from ID to dataset entry for easy lookup
            dataset_map = {entry.get("id"): entry for entry in dataset}

            # Combine predictions and failed into results list
            all_results = []
            for pred in predictions:
                all_results.append(
                    {
                        "id": pred.get("id"),
                        "status": "success",
                        "vlm_response": pred.get("vlm_response", {}),
                    }
                )
            for fail in failed:
                all_results.append(
                    {
                        "id": fail.get("id"),
                        "status": "error",
                        "error": fail.get("error", "Unknown error"),
                    }
                )

            # Generate HTML content
            html_content = self._create_html_content(
                all_results,
                dataset_map,
                context,
                listener,
                image_max_size=image_max_size,
                image_format=image_format,
                image_quality=image_quality,
            )

            # Write HTML file
            with open(html_file, "w", encoding="utf-8") as f:
                f.write(html_content)

            listener.info(f"Generated prediction HTML report: {html_file}")
            return html_file

        except Exception as e:
            listener.error(f"Failed to generate prediction HTML report: {e}")
            return None

    def _format_token_stats_html(self, token_stats: dict[str, Any]) -> str:
        """Format token usage statistics as HTML section.

        Args:
            token_stats: Token statistics dictionary from TokenTracker.get_stats()

        Returns:
            HTML string with token usage section
        """
        if not token_stats or token_stats.get("invocation_count", 0) == 0:
            return ""  # No token stats available

        total_tokens = token_stats.get("total_tokens", 0)
        input_tokens = token_stats.get("total_input_tokens", 0)
        output_tokens = token_stats.get("total_output_tokens", 0)
        invocation_count = token_stats.get("invocation_count", 0)

        # Format by-model breakdown if available
        by_model_html = ""
        by_model = token_stats.get("by_model", {})
        if by_model:
            model_rows = []
            for model_name, stats in by_model.items():
                model_rows.append(
                    f"""
                    <tr>
                        <td>{model_name}</td>
                        <td>{stats["count"]}</td>
                        <td>{stats["input_tokens"]:,}</td>
                        <td>{stats["output_tokens"]:,}</td>
                        <td>{stats["total_tokens"]:,}</td>
                    </tr>
                    """
                )
            by_model_html = f"""
            <h3>By Model</h3>
            <table style="margin-top: 1rem;">
                <thead>
                    <tr>
                        <th>Model</th>
                        <th>Calls</th>
                        <th>Input Tokens</th>
                        <th>Output Tokens</th>
                        <th>Total Tokens</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join(model_rows)}
                </tbody>
            </table>
            """

        return f"""
        <h2>🪙 Token Usage</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Total Tokens</div>
                <div class="metric-value">{total_tokens:,}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Input Tokens</div>
                <div class="metric-value">{input_tokens:,}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Output Tokens</div>
                <div class="metric-value">{output_tokens:,}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">VLM Calls</div>
                <div class="metric-value">{invocation_count}</div>
            </div>
        </div>
        {by_model_html}
        """

    def _create_html_content(
        self,
        results: list[dict],
        dataset_map: dict[str, dict],
        context: dict[str, Any],
        listener,
        image_max_size: int | None = None,
        image_format: str = "png",
        image_quality: int = 85,
    ) -> str:
        """Create the HTML content for the prediction report.

        Args:
            results: List of prediction results (success + failures)
            dataset_map: Mapping from ID to dataset entry
            context: Workflow context
            listener: Event listener for progress reporting
            image_max_size: Optional max image size in pixels
            image_format: Image format ('png' or 'jpeg')
            image_quality: JPEG quality (1-100)

        Returns:
            HTML content as string
        """
        # Generate table rows
        table_rows = []
        for result in results:
            raw_id = result.get("id", "unknown")
            pred_id = escape_html(raw_id)

            # Get dataset entry
            dataset_entry = dataset_map.get(raw_id, {})

            # Extract and escape prompt (support both v0.1 and v0.2 schemas)
            # v0.2: user_prompt, v0.1: text
            prompt_text = dataset_entry.get("user_prompt") or dataset_entry.get("text")
            prompt = escape_html(prompt_text)

            # Handle images with metadata
            # Determine base directory for images
            base_dir = None
            if context.get("dataset_path"):
                base_dir = Path(context["dataset_path"]).parent

            # Support both v0.1 and v0.2 schemas
            # v0.2: media.images with path and metadata fields
            # v0.1: images (list of paths) and image_metadata (parallel list)
            media_section = dataset_entry.get("media", {})
            if media_section and "images" in media_section:
                # v0.2 schema: extract paths and metadata from media.images
                media_images = media_section.get("images", [])
                image_paths = [img.get("path") for img in media_images]
                image_metadata = [img.get("metadata", {}) for img in media_images]
            else:
                # v0.1 schema: separate images and image_metadata lists
                image_paths = dataset_entry.get("images", [])
                image_metadata = dataset_entry.get("image_metadata", [])

            images_html = format_images_html(
                image_paths,
                base_dir,
                image_metadata,
                image_max_size=image_max_size,
                image_format=image_format,
                image_quality=image_quality,
            )

            # Extract predicted material and response
            if result["status"] == "success":
                vlm_response = result.get("vlm_response", {})
                if isinstance(vlm_response, dict):
                    predicted_material = vlm_response.get("material", "N/A")
                    original_response = vlm_response.get("original_response", "")
                else:
                    predicted_material = str(vlm_response)
                    original_response = str(vlm_response)

                # Escape HTML
                predicted_material_html = escape_html(predicted_material)
                original_response_html = escape_html(original_response)

                status_badge = (
                    '<span style="color: #4CAF50; font-weight: bold;">✓ Success</span>'
                )
            else:
                predicted_material_html = "N/A"
                error_msg = result.get("error", "Unknown error")
                original_response_html = f"Error: {escape_html(error_msg)}"
                status_badge = (
                    '<span style="color: #F44336; font-weight: bold;">✗ Failed</span>'
                )

            # Generate table row
            row = f"""
                <tr>
                    <td>{pred_id}</td>
                    <td><div class="prompt-cell">{prompt}</div></td>
                    <td>{images_html}</td>
                    <td><strong>{predicted_material_html}</strong></td>
                    <td>{status_badge}</td>
                    <td><div class="response-cell">{original_response_html}</div></td>
                </tr>
            """
            table_rows.append(row)

        table_rows_html = "\n".join(table_rows)

        # Calculate statistics
        total_cases = len(results)
        successful_cases = sum(1 for r in results if r["status"] == "success")
        failed_cases = total_cases - successful_cases
        success_rate = (successful_cases / total_cases * 100) if total_cases > 0 else 0

        # Get prim counts from context
        original_prim_count = context.get("original_prim_count", 0)
        num_prims = context.get("num_prims", 0)
        num_images = context.get("num_images", 0)

        # Get system prompt if available
        # Prefer actual_system_prompt_used (includes critique) over base config prompt
        system_prompt = context.get("actual_system_prompt_used") or context.get(
            "config", {}
        ).get("system_prompt", "")
        system_prompt_section = format_system_prompt_section(system_prompt)

        # Get token usage statistics if available
        token_stats = context.get("token_stats", {})
        token_section = self._format_token_stats_html(token_stats)
        cost_section = format_cost_estimate_section(token_stats)

        # Fill template
        html_content = self._HTML_TEMPLATE.format(
            original_prim_count=original_prim_count,
            prims_processed=num_prims,
            images_generated=num_images,
            total_cases=total_cases,
            successful_cases=successful_cases,
            failed_cases=failed_cases,
            success_rate=f"{success_rate:.1f}",
            system_prompt_section=system_prompt_section,
            token_section=token_section,
            cost_section=cost_section,
            table_rows=table_rows_html,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

        return html_content

    # HTML template for prediction report (without ground truth and scores)
    _HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Material Agent Prediction Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            padding: 30px;
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #2196F3;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: 30px;
            border-bottom: 1px solid #ddd;
            padding-bottom: 8px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            padding: 15px;
            background: #f9f9f9;
            border-radius: 6px;
            border: 1px solid #e0e0e0;
        }}
        .metric-label {{
            font-size: 12px;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #333;
            margin-top: 5px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
            font-size: 12px;
        }}
        th {{
            background: #2196F3;
            color: white;
            text-align: left;
            padding: 12px;
            font-weight: 600;
            position: sticky;
            top: 0;
            z-index: 10;
        }}
        th:first-child {{
            width: 180px;
            min-width: 150px;
            max-width: 180px;
        }}
        td {{
            padding: 12px;
            border-bottom: 1px solid #e0e0e0;
            vertical-align: top;
        }}
        td:first-child {{
            width: 180px;
            min-width: 150px;
            max-width: 180px;
            font-size: 10px;
            word-wrap: break-word;
            word-break: break-all;
        }}
        tr:hover {{
            background: #f9f9f9;
        }}
        .image-container {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 5px;
        }}
        .image-with-caption {{
            display: flex;
            flex-direction: column;
            align-items: center;
            max-width: 150px;
        }}
        .image-caption {{
            font-size: 11px;
            color: #666;
            margin-top: 4px;
            text-align: center;
            line-height: 1.2;
            word-wrap: break-word;
        }}
        .image-thumbnail {{
            max-width: 100px;
            max-height: 100px;
            object-fit: cover;
            border: 1px solid #ddd;
            border-radius: 4px;
            cursor: pointer;
        }}
        .prompt-cell {{
            max-width: 300px;
            max-height: 200px;
            overflow-y: auto;
            font-size: 12px;
            background: #f9f9f9;
            padding: 8px;
            border-radius: 4px;
            white-space: pre-wrap;
            word-wrap: break-word;
        }}
        .response-cell {{
            max-width: 400px;
            max-height: 200px;
            overflow-y: auto;
            font-size: 12px;
            background: #f9f9f9;
            padding: 8px;
            border-radius: 4px;
            white-space: pre-wrap;
            word-wrap: break-word;
        }}
        .timestamp {{
            color: #666;
            font-size: 12px;
            margin-top: 10px;
        }}
        .system-prompt-section {{
            margin: 30px 0;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
            border-left: 4px solid #2196F3;
        }}
        .system-prompt-section h3 {{
            margin-top: 0;
            color: #333;
            font-size: 14px;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .system-prompt-content {{
            margin-top: 15px;
            padding: 15px;
            background: white;
            border-radius: 4px;
            font-family: monospace;
            font-size: 10px;
            white-space: pre-wrap;
            word-wrap: break-word;
            max-height: 300px;
            overflow-y: auto;
            border: 1px solid #e0e0e0;
        }}
        .modal {{
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            overflow: auto;
            background-color: rgba(0,0,0,0.8);
        }}
        .modal-content {{
            margin: auto;
            display: block;
            max-width: 90%;
            max-height: 90%;
            margin-top: 50px;
        }}
        .close {{
            position: absolute;
            top: 15px;
            right: 35px;
            color: #f1f1f1;
            font-size: 40px;
            font-weight: bold;
            cursor: pointer;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔮 Material Agent Prediction Report</h1>

        <h2>📊 Summary Statistics</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Original Prims</div>
                <div class="metric-value">{original_prim_count}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Prims Processed</div>
                <div class="metric-value">{prims_processed}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Images Generated</div>
                <div class="metric-value">{images_generated}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Total Cases</div>
                <div class="metric-value">{total_cases}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Successful</div>
                <div class="metric-value">{successful_cases}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Failed</div>
                <div class="metric-value">{failed_cases}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Success Rate</div>
                <div class="metric-value">{success_rate}%</div>
            </div>
        </div>

        {token_section}

        {cost_section}

        {system_prompt_section}

        <h2>📋 Detailed Prediction Results</h2>
        <table>
            <thead>
                <tr>
                    <th>ID</th>
                    <th>Input Prompt</th>
                    <th>Input Images</th>
                    <th>Predicted Material</th>
                    <th>Status</th>
                    <th>Full Response</th>
                </tr>
            </thead>
            <tbody>
                {table_rows}
            </tbody>
        </table>

        <div class="timestamp">
            Generated at: {timestamp}
        </div>
    </div>

    <!-- Modal for image preview -->
    <div id="imageModal" class="modal">
        <span class="close" onclick="closeModal()">&times;</span>
        <img class="modal-content" id="modalImage">
    </div>

    <script>
        function showImage(src) {{
            var modal = document.getElementById('imageModal');
            var modalImg = document.getElementById('modalImage');
            modal.style.display = "block";
            modalImg.src = src;
        }}

        function closeModal() {{
            document.getElementById('imageModal').style.display = "none";
        }}

        // Close modal when clicking outside the image
        window.onclick = function(event) {{
            var modal = document.getElementById('imageModal');
            if (event.target == modal) {{
                modal.style.display = "none";
            }}
        }}
    </script>
</body>
</html>
    """


class GenerateEvaluationReportTask(BaseReportTask):
    """Generate HTML report from evaluation results.

    This task creates a visual HTML report showing evaluation results including
    predictions, ground truth, LLM judge scores, and detailed metrics.

    Input context keys:
        - evaluations: List of evaluation results
        - metrics: Dictionary of calculated metrics (FCS, success rate, etc.)
        - predictions: List of original predictions
        - output_dir: Directory to save the HTML report
        - dataset_path: Path to dataset file (for resolving relative image paths)
        - config: Optional configuration dict (for system prompt)
        - report_image_max_size: Optional max image size in pixels (e.g., 256 for 256x256)
        - report_image_format: Optional image format ('png' or 'jpeg', default: 'png')
        - report_image_quality: Optional JPEG quality (1-100, default: 85)

    Output context keys:
        - evaluation_html_report_path: Path to generated HTML report
    """

    def __init__(
        self,
        image_max_size: int | None = None,
        image_format: str | None = None,
        image_quality: int | None = None,
    ):
        """Initialize the evaluation report generation task.

        Args:
            image_max_size: Optional max image size in pixels (e.g., 256 for 256x256).
                           Can be overridden by context key 'report_image_max_size'.
            image_format: Optional image format ('png' or 'jpeg', default: 'png').
                         Can be overridden by context key 'report_image_format'.
            image_quality: Optional JPEG quality (1-100, default: 85).
                          Can be overridden by context key 'report_image_quality'.
        """
        self.name = "GenerateEvaluationReport"
        self.description = "Generate HTML report from evaluation results"
        self.image_max_size = image_max_size
        self.image_format = image_format
        self.image_quality = image_quality

    def run(
        self, context: dict[str, Any], object_store: ObjectStore | None = None
    ) -> dict[str, Any]:
        """Generate HTML report from evaluation results.

        Args:
            context: Workflow context containing evaluations and metrics
            object_store: Optional object store

        Returns:
            Updated context with evaluation_html_report_path
        """
        # Get event listener (or logger fallback)
        listener = get_listener(context, logger_name=__name__)
        # Get data from context or object store
        evaluations = context.get("evaluations", [])
        metrics = context.get("metrics", {})
        predictions = context.get("predictions", [])

        if object_store:
            evaluations = object_store.get("evaluations", evaluations)
            metrics = object_store.get("metrics", metrics)
            predictions = object_store.get("predictions", predictions)

        # If predictions are not available or incomplete, try loading from predictions_path
        predictions_path = context.get("predictions_path")
        if predictions_path and (not predictions or not predictions[0].get("images")):
            try:
                import json

                with open(predictions_path, encoding="utf-8") as f:
                    file_predictions = [json.loads(line) for line in f if line.strip()]
                if file_predictions:
                    listener.debug(
                        f"Loaded {len(file_predictions)} predictions from file for report"
                    )
                    predictions = file_predictions
            except Exception as e:
                listener.warning(f"Failed to load predictions from file: {e}")

        output_dir = Path(context.get("output_dir", "."))
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get and validate image processing options
        image_max_size = (
            context.get("report_image_max_size")
            if context.get("report_image_max_size") is not None
            else self.image_max_size
        )
        image_format = context.get("report_image_format", self.image_format)
        image_quality = context.get("report_image_quality", self.image_quality)

        image_format, image_quality, image_max_size = validate_image_options(
            image_format, image_quality, image_max_size
        )

        try:
            html_file = self._generate_report(
                evaluations=evaluations,
                metrics=metrics,
                output_dir=output_dir,
                predictions=predictions,
                context=context,
                listener=listener,
                image_max_size=image_max_size,
                image_format=image_format,
                image_quality=image_quality,
            )

            if html_file:
                context["evaluation_html_report_path"] = str(html_file)
                listener.info(f"✓ Evaluation HTML report generated: {html_file}")

        except Exception as e:
            listener.warning(f"Failed to generate evaluation HTML report: {e}")

        return context

    def _generate_report(
        self,
        evaluations: list[dict],
        metrics: dict[str, Any],
        output_dir: Path,
        predictions: list[dict],
        context: dict[str, Any],
        listener,
        image_max_size: int | None = None,
        image_format: str = "png",
        image_quality: int = 85,
    ) -> Path | None:
        """Generate an HTML report for the evaluation results.

        Args:
            evaluations: List of evaluation results
            metrics: Calculated metrics
            output_dir: Directory to save the HTML report
            predictions: Original predictions with input data
            context: Optional workflow context for additional information
            listener: Event listener for progress reporting
            image_max_size: Optional max image size in pixels (e.g., 256 for 256x256)
            image_format: Image format ('png' or 'jpeg', default: 'png')
            image_quality: JPEG quality (1-100, default: 85)

        Returns:
            Path to the generated HTML file or None if error
        """
        try:
            html_file = output_dir / "evaluation_report.html"

            # Create a mapping from prediction ID to prediction data for easy lookup
            pred_map = {pred["id"]: pred for pred in predictions}

            # Load dataset if available to enrich predictions with text prompts
            dataset_path = context.get("dataset_path")
            dataset_map = {}
            if dataset_path:
                dataset_path_obj = Path(dataset_path)
                if dataset_path_obj.exists():
                    try:
                        import json

                        with open(dataset_path_obj, encoding="utf-8") as f:
                            dataset = [json.loads(line) for line in f if line.strip()]
                        dataset_map = {
                            entry["id"]: entry for entry in dataset if "id" in entry
                        }
                        listener.debug(
                            f"Loaded {len(dataset_map)} dataset entries for report"
                        )
                    except Exception as e:
                        listener.warning(f"Failed to load dataset for report: {e}")

            # Generate HTML content
            html_content = self._create_html_content(
                evaluations,
                metrics,
                pred_map,
                dataset_map,
                context,
                listener,
                image_max_size=image_max_size,
                image_format=image_format,
                image_quality=image_quality,
            )

            # Write HTML file
            with open(html_file, "w", encoding="utf-8") as f:
                f.write(html_content)

            listener.info(f"Generated evaluation HTML report: {html_file}")
            return html_file

        except Exception as e:
            listener.error(f"Failed to generate evaluation HTML report: {e}")
            return None

    def _create_html_content(
        self,
        evaluations: list[dict],
        metrics: dict[str, Any],
        pred_map: dict[str, dict],
        dataset_map: dict[str, dict],
        context: dict[str, Any],
        listener,
        image_max_size: int | None = None,
        image_format: str = "png",
        image_quality: int = 85,
    ) -> str:
        """Create the HTML content for the evaluation report.

        Args:
            evaluations: List of evaluation results
            metrics: Calculated metrics
            pred_map: Mapping from prediction ID to prediction data
            dataset_map: Mapping from entry ID to dataset entry (for text prompts)
            context: Optional workflow context for additional information
            listener: Event listener for progress reporting
            image_max_size: Optional max image size in pixels (e.g., 256 for 256x256)
            image_format: Image format ('png' or 'jpeg', default: 'png')
            image_quality: JPEG quality (1-100, default: 85)

        Returns:
            HTML content as string
        """
        # Generate table rows
        table_rows = []
        for eval_result in evaluations:
            raw_pred_id = eval_result["id"]
            pred_data = pred_map.get(raw_pred_id, {})
            dataset_entry = dataset_map.get(raw_pred_id, {})

            # Escape HTML in ID
            pred_id = escape_html(raw_pred_id)

            # Extract prompt from prediction (enriched), dataset, or fallback to "N/A"
            # Try multiple sources: pred_data.prompt, pred_data.text, dataset_entry.text
            raw_prompt = (
                pred_data.get("prompt")
                or pred_data.get("text")
                or dataset_entry.get("text")
                or "N/A"
            )
            prompt = (
                escape_html(raw_prompt) if raw_prompt and raw_prompt != "N/A" else "N/A"
            )

            # Handle images - pass base directory for relative paths
            # Get images from pred_data or dataset_entry (with fallback)
            images = pred_data.get("images", []) or dataset_entry.get("images", [])
            image_metadata = pred_data.get("image_metadata", []) or dataset_entry.get(
                "image_metadata", []
            )

            # Determine base directory
            base_dir = None
            if context.get("dataset_path"):
                base_dir = Path(context["dataset_path"]).parent
            elif context.get("config_path"):
                base_dir = Path(context["config_path"]).parent

            # Format images with metadata
            images_html = format_images_html(
                images,
                base_dir,
                image_metadata,
                image_max_size=image_max_size,
                image_format=image_format,
                image_quality=image_quality,
            )

            # Get predicted material - prioritize eval_result, then pred_data
            # eval_result has the predicted_material from evaluation
            raw_predicted = eval_result.get("predicted_material", "N/A")
            if raw_predicted == "N/A":
                # Fall back to pred_data materials
                materials_data = pred_data.get("materials", {})
                if isinstance(materials_data, dict):
                    raw_predicted = materials_data.get("material", "N/A")
                elif materials_data:
                    raw_predicted = str(materials_data)

            # Get original response from pred_data materials
            materials_data = pred_data.get("materials", {})
            if isinstance(materials_data, dict):
                original_response = materials_data.get("original_response", "N/A")
            else:
                original_response = "N/A"

            # Escape HTML
            predicted = escape_html(raw_predicted)
            ground_truth = escape_html(eval_result.get("ground_truth", "N/A"))

            # Get score and determine badge class
            score = eval_result.get("score", 0)
            score_class = f"score-{score}"
            score_badge = f'<span class="score-badge {score_class}">{score}</span>'

            # Exact match indicator
            is_exact_match = eval_result.get("exact_match", False)
            match_indicator = (
                '<span class="exact-match">✓ Match</span>'
                if is_exact_match
                else '<span class="no-match">✗ No Match</span>'
            )

            # Escape evaluation explanation and original response
            explanation = escape_html(eval_result.get("explanation"))
            original_response_html = escape_html(original_response)

            # Generate table row
            row = f"""
                <tr>
                    <td>{pred_id}</td>
                    <td><div class="prompt-cell">{prompt}</div></td>
                    <td>{images_html}</td>
                    <td><strong>{predicted}</strong></td>
                    <td><strong>{ground_truth}</strong></td>
                    <td>{score_badge}</td>
                    <td>{match_indicator}</td>
                    <td><div class="explanation-cell">{explanation}</div></td>
                    <td><div class="response-cell">{original_response_html}</div></td>
                </tr>
            """
            table_rows.append(row)

        table_rows_html = "\n".join(table_rows)

        # Get prim counts from context
        original_prim_count = context.get("original_prim_count", 0)
        num_prims = context.get("num_prims", 0)
        num_images = context.get("num_images", 0)

        # Get system prompt if available
        # Prefer actual_system_prompt_used (includes critique) over base config prompt
        system_prompt = context.get("actual_system_prompt_used") or context.get(
            "config", {}
        ).get("system_prompt", "")
        system_prompt_section = format_system_prompt_section(system_prompt)

        # Fill template
        # Note: metrics uses 'functional_correctness_score' as key
        fcs_score = metrics.get("functional_correctness_score", metrics.get("fcs", 0))
        html_content = self._HTML_TEMPLATE.format(
            original_prim_count=original_prim_count,
            prims_processed=num_prims,
            images_generated=num_images,
            fcs=f"{fcs_score:.2f}",
            success_rate=f"{metrics.get('success_rate', 0):.1f}",
            exact_match_rate=f"{metrics.get('exact_match_rate', 0):.1f}",
            total_cases=metrics.get("total_cases", 0),
            valid_cases=metrics.get("valid_cases", 0),
            exact_matches=metrics.get("exact_matches", 0),
            system_prompt_section=system_prompt_section,
            table_rows=table_rows_html,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

        return html_content

    # HTML template for evaluation report
    _HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Material Agent Evaluation Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            padding: 30px;
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: 30px;
            border-bottom: 1px solid #ddd;
            padding-bottom: 8px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            padding: 15px;
            background: #f9f9f9;
            border-radius: 6px;
            border: 1px solid #e0e0e0;
        }}
        .metric-label {{
            font-size: 12px;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #333;
            margin-top: 5px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
            font-size: 12px;
        }}
        th {{
            background: #4CAF50;
            color: white;
            text-align: left;
            padding: 12px;
            font-weight: 600;
            position: sticky;
            top: 0;
            z-index: 10;
        }}
        th:first-child {{
            width: 180px;
            min-width: 150px;
            max-width: 180px;
        }}
        td {{
            padding: 12px;
            border-bottom: 1px solid #e0e0e0;
            vertical-align: top;
        }}
        td:first-child {{
            width: 180px;
            min-width: 150px;
            max-width: 180px;
            font-size: 10px;
            word-wrap: break-word;
            word-break: break-all;
        }}
        tr:hover {{
            background: #f9f9f9;
        }}
        .score-badge {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-weight: bold;
            text-align: center;
            min-width: 30px;
        }}
        .score-5 {{ background: #4CAF50; color: white; }}
        .score-4 {{ background: #8BC34A; color: white; }}
        .score-3 {{ background: #FFC107; color: black; }}
        .score-2 {{ background: #FF9800; color: white; }}
        .score-1 {{ background: #F44336; color: white; }}
        .score-0 {{ background: #9E9E9E; color: white; }}
        .exact-match {{
            color: #4CAF50;
            font-weight: bold;
        }}
        .no-match {{
            color: #F44336;
        }}
        .image-container {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 5px;
        }}
        .image-with-caption {{
            display: flex;
            flex-direction: column;
            align-items: center;
            max-width: 150px;
        }}
        .image-caption {{
            font-size: 11px;
            color: #666;
            margin-top: 4px;
            text-align: center;
            line-height: 1.2;
            word-wrap: break-word;
        }}
        .image-thumbnail {{
            max-width: 100px;
            max-height: 100px;
            object-fit: cover;
            border: 1px solid #ddd;
            border-radius: 4px;
            cursor: pointer;
        }}
        .prompt-cell {{
            max-width: 300px;
            max-height: 200px;
            overflow-y: auto;
            font-size: 12px;
            background: #f9f9f9;
            padding: 8px;
            border-radius: 4px;
            white-space: pre-wrap;
            word-wrap: break-word;
        }}
        .response-cell {{
            max-width: 400px;
            max-height: 200px;
            overflow-y: auto;
            font-size: 12px;
            background: #f9f9f9;
            padding: 8px;
            border-radius: 4px;
            white-space: pre-wrap;
            word-wrap: break-word;
        }}
        .explanation-cell {{
            max-width: 400px;
            max-height: 200px;
            overflow-y: auto;
            font-size: 12px;
            background: #f9f9f9;
            padding: 8px;
            border-radius: 4px;
            white-space: pre-wrap;
            word-wrap: break-word;
        }}
        .timestamp {{
            color: #666;
            font-size: 12px;
            margin-top: 10px;
        }}
        .system-prompt-section {{
            margin: 30px 0;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
            border-left: 4px solid #4CAF50;
        }}
        .system-prompt-section h3 {{
            margin-top: 0;
            color: #333;
            font-size: 14px;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .system-prompt-content {{
            margin-top: 15px;
            padding: 15px;
            background: white;
            border-radius: 4px;
            font-family: monospace;
            font-size: 10px;
            white-space: pre-wrap;
            word-wrap: break-word;
            max-height: 300px;
            overflow-y: auto;
            border: 1px solid #e0e0e0;
        }}
        .modal {{
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            overflow: auto;
            background-color: rgba(0,0,0,0.8);
        }}
        .modal-content {{
            margin: auto;
            display: block;
            max-width: 90%;
            max-height: 90%;
            margin-top: 50px;
        }}
        .close {{
            position: absolute;
            top: 15px;
            right: 35px;
            color: #f1f1f1;
            font-size: 40px;
            font-weight: bold;
            cursor: pointer;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 Material Agent Evaluation Report</h1>

        <h2>📊 Summary Metrics</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Original Prims</div>
                <div class="metric-value">{original_prim_count}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Prims Processed</div>
                <div class="metric-value">{prims_processed}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Images Generated</div>
                <div class="metric-value">{images_generated}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">FCS (Functional Correctness)</div>
                <div class="metric-value">{fcs}/5.0</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Success Rate</div>
                <div class="metric-value">{success_rate}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Exact Match Rate</div>
                <div class="metric-value">{exact_match_rate}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Total Cases</div>
                <div class="metric-value">{total_cases}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Valid Cases</div>
                <div class="metric-value">{valid_cases}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Exact Matches</div>
                <div class="metric-value">{exact_matches}</div>
            </div>
        </div>

        {system_prompt_section}

        <h2>📋 Detailed Evaluation Results</h2>
        <table>
            <thead>
                <tr>
                    <th>ID</th>
                    <th>Input Prompt</th>
                    <th>Input Images</th>
                    <th>Predicted Material</th>
                    <th>Ground Truth</th>
                    <th>Score</th>
                    <th>Match</th>
                    <th>Eval Reason</th>
                    <th>Full Response</th>
                </tr>
            </thead>
            <tbody>
                {table_rows}
            </tbody>
        </table>

        <div class="timestamp">
            Generated at: {timestamp}
        </div>
    </div>

    <!-- Modal for image preview -->
    <div id="imageModal" class="modal">
        <span class="close" onclick="closeModal()">&times;</span>
        <img class="modal-content" id="modalImage">
    </div>

    <script>
        function showImage(src) {{
            var modal = document.getElementById('imageModal');
            var modalImg = document.getElementById('modalImage');
            modal.style.display = "block";
            modalImg.src = src;
        }}

        function closeModal() {{
            document.getElementById('imageModal').style.display = "none";
        }}

        // Close modal when clicking outside the image
        window.onclick = function(event) {{
            var modal = document.getElementById('imageModal');
            if (event.target == modal) {{
                modal.style.display = "none";
            }}
        }}
    </script>
</body>
</html>
    """
