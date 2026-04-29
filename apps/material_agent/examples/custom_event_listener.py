#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Example: Using Custom Event Listeners with Material Agent API.

This example demonstrates how to create and use custom event listeners
to handle workflow progress. It runs a real pipeline on the ladder USD file.
"""

from dotenv import load_dotenv

from material_agent.api import (
    build_unified_pipeline_config,
    pipeline,
)

# Load environment variables (for API keys, etc.)
load_dotenv()


# ============================================================================
# Example 1: Simple Progress Tracker
# ============================================================================


class SimpleProgressListener:
    """A simple custom listener that tracks progress."""

    def __init__(self):
        self.started = False
        self.completed = False
        self.progress_updates = []

    def info(self, message: str, **kwargs):
        """Just print info messages."""
        print(f"ℹ️  {message}", flush=True)

    def debug(self, message: str, **kwargs):
        """Ignore debug messages."""
        pass

    def warning(self, message: str, **kwargs):
        """Print warnings with icon."""
        print(f"⚠️  {message}")

    def error(self, message: str, **kwargs):
        """Print errors with icon."""
        print(f"❌ {message}")

    def event(self, event_type: str, data: dict, **kwargs):
        """Track structured events."""
        if event_type == "pipeline.started":
            # NEW: Show session ID at pipeline start
            session_id = data.get("session_id", "unknown")
            project = data.get("project_name", "unknown")
            print(f"\n🚀 Starting pipeline: {project}")
            print(f"   🔑 Session ID: {session_id}")
            print(f"   📁 Working Dir: {data.get('working_dir', 'unknown')}")
            self.started = True

        elif event_type == "workflow.started":
            self.started = True
            print(f"\n🚀 Starting {data.get('workflow_type', 'workflow')}...")

        elif event_type == "step.started":
            # Show step header when step begins
            step_name = data.get("step_name", "unknown")
            step_idx = data.get("step_index", 0)
            total_steps = data.get("total_steps", 0)
            print(f"\n📍 [{step_idx}/{total_steps}] Starting step: {step_name}")

        elif event_type == "step.completed":
            # Show when step completes
            step_name = data.get("step_name", "unknown")
            print(f"   ✅ Step '{step_name}' completed")

        elif event_type == "step.failed":
            # Show when step fails
            step_name = data.get("step_name", "unknown")
            error = data.get("error", "unknown error")
            print(f"   ❌ Step '{step_name}' failed: {error}")

        elif event_type == "pipeline.completed":
            # NEW: Show session ID at pipeline completion
            session_id = data.get("session_id", "unknown")
            output_usd = data.get("output_usd", "unknown")
            working_dir = data.get("working_dir", "unknown")
            print("\n✅ Pipeline completed successfully!")
            print(f"   🔑 Session ID: {session_id}")
            print(f"   📦 Output USD: {output_usd}")
            print(f"   📁 Find outputs in: {working_dir}/output/")
            self.completed = True

        elif event_type == "workflow.completed":
            self.completed = True
            print("\n✅ Workflow completed!")

        elif event_type == "workflow.failed":
            print(f"\n❌ Workflow failed: {data.get('error')}")

        elif event_type == "task.started":
            # Show when a task begins
            task_name = data.get("task_name", "unknown")
            total = data.get("total_entries", data.get("total", 0))
            if total:
                print(f"   ⚙️  Starting {task_name} ({total} items)...")

        elif event_type == "task.progress":
            self.progress_updates.append(data)
            # Show live progress updates
            pct = data.get("percentage", 0)
            current = data.get("current", 0)
            total = data.get("total", 0)
            task = data.get("task_name", "Task")
            print(f"   {task}: {current}/{total} ({pct:.1f}%)", end="\r")

        elif event_type == "task.completed":
            # Show when task completes
            task_name = data.get("task_name", "unknown")
            successful = data.get("successful", 0)
            total = data.get("total", 0)
            print(f"\n   ✅ {task_name} completed: {successful}/{total} successful")

        elif event_type == "prediction.completed":
            # Show individual material predictions
            entry_id = data.get("entry_id", "unknown")
            material = data.get("material", "unknown")
            confidence = data.get("confidence")
            print(
                f"   📦 Prediction for {entry_id}: material='{material}'"
                + (f" (confidence: {confidence})" if confidence else "")
            )

        elif event_type == "rendering.completed":
            # Show individual render completions (camera or prim)
            # Check if this is a per-camera event or per-prim event
            if "camera_corner" in data:
                # Per-camera event (from RenderTask)
                camera_corner = data.get("camera_corner", "unknown")
                output_path = data.get("output_path", "unknown")
                backend = data.get("backend", "unknown")
                width = data.get("image_width", 0)
                height = data.get("image_height", 0)
                print(
                    f"   🎬 Rendered {camera_corner}: "
                    f"{width}x{height} → {output_path.split('/')[-1]} "
                    f"(backend: {backend})"
                )
            elif "prim_path" in data:
                # Per-prim event (from USDPrimTraversalAndRenderingTask)
                prim_path = data.get("prim_path", "unknown")
                camera_view = data.get("camera_view", "unknown")
                render_mode = data.get("render_mode", "unknown")
                output_path = data.get("output_path", "unknown")
                # Extract prim name from path
                prim_name = prim_path.split("/")[-1] if "/" in prim_path else prim_path
                print(
                    f"   🎬 Rendered {prim_name} ({camera_view}, {render_mode}) → "
                    f"{output_path.split('/')[-1]}"
                )

        elif event_type == "rendering.all_completed":
            # Show overall rendering completion summary
            # Check if this is from RenderTask or USDPrimTraversalAndRenderingTask
            if "backend" in data:
                # From RenderTask (camera-based rendering)
                total = data.get("total_images", 0)
                failed = data.get("failed_renders", 0)
                backend = data.get("backend", "unknown")
                corners = data.get("camera_corners", [])
                print(
                    f"\n   ✅ All rendering completed: "
                    f"{total} images rendered "
                    f"({len(corners)} views) "
                    f"using {backend}"
                )
                if failed > 0:
                    print(f"   ⚠️  {failed} render(s) failed")
            elif "total_prims" in data:
                # From USDPrimTraversalAndRenderingTask (prim-based rendering)
                total_prims = data.get("total_prims", 0)
                total_images = data.get("total_images", 0)
                skipped = data.get("skipped_prims", 0)
                modes = data.get("rendering_modes", [])
                views = data.get("num_views", 0)
                failed = data.get("failed_batches", 0)
                print(
                    f"\n   ✅ All rendering completed: "
                    f"{total_images} images from {total_prims} prims "
                    f"({views} views × {len(modes)} modes)"
                )
                if skipped > 0:
                    print(f"   ℹ️  Skipped {skipped} existing prims")
                if failed > 0:
                    print(f"   ⚠️  {failed} batch(es) failed")


def example_simple_progress():
    """Example: Run full pipeline with custom progress tracking."""
    print("=" * 70)
    print("Example: Full Pipeline with Custom Event Listener")
    print("=" * 70)

    # Create custom listener
    listener = SimpleProgressListener()

    # Build minimal config for ladder pipeline
    # Note: output_usd_path is auto-derived as .{session_id}/output/output.usd
    config = build_unified_pipeline_config(
        project_name="ladder_event_example",
        input_usd_path="apps/material_agent/data/examples/ladder/sources/usd/ladder.usd",
        materials_library_path="apps/material_agent/data/materials/material_libs/material_libs.usd",
        materials_entries=[
            {
                "name": "Aluminum Polished",
                "description": "Polished aluminum for structural parts",
                "binding": "/World/metal_library/Looks/Aluminum_Polished",
            },
            {
                "name": "Polycarbonate Blue",
                "description": "Blue polycarbonate for decorative components",
                "binding": "/World/plastic_library/Looks/Polycarbonate_Blue",
            },
            {
                "name": "Polyethylene Cloudy Rough Black",
                "description": "Black plastic for non-slip surfaces",
                "binding": "/World/plastic_library/Looks/Polyethylene_Cloudy_Rough",
            },
        ],
        enabled_steps=[
            "build_dataset_usd",
            "build_dataset_prepare_dataset",
            "predict",
            "apply",
            "render",
        ],
    )

    # Configure steps
    config["steps"]["build_dataset_usd"]["renderer"] = {
        "backend": "remote",
        "image_width": 512,
        "image_height": 512,
    }
    config["steps"]["render"] = {"enabled": True}

    # Add report compression configuration
    # This reduces report file size from 17-22 MB to 1-2 MB
    config["steps"]["predict"]["report"] = {
        "image_max_size": 128,  # Downscale to 128×128 max
        "image_format": "jpeg",  # Use JPEG instead of PNG
        "image_quality": 60,  # JPEG quality (1-100)
    }

    print("\n🎯 Running full pipeline on ladder.usd...")
    print("   Steps: build_dataset_usd → prepare_dataset → predict → apply → render")
    print(f"   Materials: {len(config['materials']['entries'])}")
    print("   Using custom event listener to track progress")
    print("   📊 Report compression: 128×128 JPEG (quality=60) for smaller file size")

    # Run pipeline with custom listener
    try:
        result = pipeline(
            config, event_listener=listener, clean=False
        )  # Keep output for inspection

        if result.success:
            print("\n✅ Pipeline completed successfully!")
            print(f"  Started: {listener.started}")
            print(f"  Completed: {listener.completed}")
            print(f"  Progress updates: {len(listener.progress_updates)}")

            # Show session ID and output location
            session_id = result.raw_result.get("session_id", "unknown")
            print(f"\n📁 Session ID: {session_id}")
            print(f"📦 Find outputs in: .{session_id}/output/")
            print(f"  Completed steps: {result.completed_steps}")

            # Show outputs
            if result.step_results:
                print("\n📁 Pipeline outputs:")
                for step, outputs in result.step_results.items():
                    print(f"  {step}:")
                    for key, value in outputs.items():
                        if value and len(str(value)) < 100:
                            print(f"    - {key}: {value}")
        else:
            print("\n❌ Pipeline FAILED!")
            print(f"  Error: {result.error}")
            print(f"  Completed steps: {result.completed_steps}")
            print(
                f"  Failed at: {result.error.split(':')[0] if ':' in str(result.error) else 'unknown'}"
            )

    except Exception as e:
        print("\n❌ Exception during pipeline execution!")
        print(f"  Error: {str(e)}")
        print(f"  Type: {type(e).__name__}")


# ============================================================================
# Main
# ============================================================================


if __name__ == "__main__":
    print()
    print("=" * 70)
    print("CUSTOM EVENT LISTENER EXAMPLE")
    print("=" * 70)
    print()
    print("This example runs a full Material Agent pipeline on ladder.usd")
    print("using a custom event listener to track progress.")
    print()

    example_simple_progress()

    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print()
    print("✓ Custom listeners can track workflow progress")
    print("✓ Both log messages AND structured events available")
    print("✓ Perfect for building REST APIs, dashboards, etc.")
    print("✓ Default behavior unchanged (backward compatible)")
    print()
    print("=" * 70)
    print()
