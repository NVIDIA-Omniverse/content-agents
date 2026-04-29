#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Minimal reproduction of ovrtx 0.2.0 segfault on time-sampled visibility.

ovrtx crashes with SIGSEGV (exit code -11) when rendering a USD stage that
contains time-sampled visibility attributes on 3+ prims. The crash occurs
inside libovrtx.dylib.so in std::vector::_M_realloc_insert.

Reproduction:
    # Requires ovrtx==0.2.0.280040 from PyPI
    pip install ovrtx==0.2.0.280040 numpy pillow
    python ovrtx_bug_repro.py

The USD file (ovrtx_bug_repro.usda) contains:
- 3 cubes at different positions
- Time-sampled visibility: each cube visible at one frame, invisible at others
  - Part0: visible at frame 0, invisible at frame 1+
  - Part1: visible at frame 1, invisible at frame 2+
  - Part2: visible at frame 2
- A fixed camera

Expected: renders 3 frames (one per visible cube)
Actual: SIGSEGV in libovrtx.dylib.so

Notes:
- 2 prims with visibility toggling works fine
- 3+ prims crashes
- Rotation animation (no visibility changes) works fine with any number of prims
- Single-frame render of a stage containing visibility keyframes works at frame 0
  but crashes at frame 1+

Workaround: strip time-sampled visibility from the USD before loading into ovrtx,
then use renderer.write_attribute() to toggle visibility per frame via the API.
"""

import os
import sys
import tempfile


def main():
    import ovrtx
    from PIL import Image

    print(f"ovrtx version: {ovrtx.__version__}")

    usd_path = os.path.join(os.path.dirname(__file__), "ovrtx_bug_repro.usda")
    if not os.path.exists(usd_path):
        print(f"ERROR: {usd_path} not found")
        sys.exit(1)

    # Build render products USDA
    render_products = """#usda 1.0

def Scope "Render"
{
    def RenderProduct "Product_Cameras_Cam"
    {
        rel camera = </Cameras/Cam>
        rel orderedVars = [</Render/Vars/LdrColor>]
        uniform int2 resolution = (512, 512)
    }

    def Scope "Vars"
    {
        def RenderVar "LdrColor"
        {
            uniform string sourceName = "LdrColor"
        }
    }
}
"""

    tmp_dir = tempfile.mkdtemp(prefix="ovrtx_bug_")
    rp_path = os.path.join(tmp_dir, "render_products.usda")
    with open(rp_path, "w") as f:
        f.write(render_products)

    # Sublayer scene + render products
    from pxr import Sdf

    combined_path = os.path.join(tmp_dir, "combined.usda")
    combined = Sdf.Layer.CreateNew(combined_path)
    combined.subLayerPaths = [usd_path, rp_path]
    combined.Save()

    print(f"Loading {combined_path}")
    renderer = ovrtx.Renderer()
    renderer.add_usd(combined_path)

    product_paths = {"/Render/Product_Cameras_Cam"}
    fps = 24.0

    for frame in range(3):
        print(f"Rendering frame {frame}...", flush=True)
        renderer.update_from_usd_time(float(frame) / fps)
        renderer.reset()
        products = renderer.step(render_products=product_paths, delta_time=1.0 / 60.0)

        if products:
            for _name, product in products.items():
                for f in product.frames:
                    if "LdrColor" in f.render_vars:
                        with f.render_vars["LdrColor"].map(
                            device=ovrtx.Device.CPU
                        ) as var:
                            pixels = var.tensor.numpy().copy()
                        out = os.path.join(tmp_dir, f"frame_{frame}.png")
                        Image.fromarray(pixels).save(out)
                        print(f"  Saved {out}")

    print("All frames rendered (bug not triggered)")
    del renderer


if __name__ == "__main__":
    main()
