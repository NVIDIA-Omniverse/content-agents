# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scene Optimizer UV generation subprocess worker.

Runs in an isolated subprocess with packman USD + SO bindings (same
ABI isolation as ``so_worker.py``).  Executes ``generateAtlasUVs`` or
``generateProjectionUVs`` on a USD stage and exports the result.

Must NOT be imported by the main process.
"""

import json
import os
import sys
import time
import traceback


def main():
    if len(sys.argv) < 2:
        sys.stderr.write("Usage: so_uv_worker.py '<json_params>'\n")
        sys.exit(1)

    try:
        params = json.loads(sys.argv[1])
    except json.JSONDecodeError as exc:
        sys.stderr.write(f"Error: Invalid JSON in arguments: {exc}\n")
        sys.exit(1)

    input_usd_path = params["input_usd_path"]
    output_usd_path = params["output_usd_path"]
    operation = params.get("operation", "unknown")
    op_params = params.get("op_params", {})
    manifest_path = params["manifest_path"]

    total_start = time.time()

    try:
        from omni.scene.optimizer.core import ExecutionContext, SceneOptimizerCore
        from pxr import Usd, UsdGeom

        stage = Usd.Stage.Open(input_usd_path)
        if stage is None:
            raise RuntimeError(f"Failed to open USD stage: {input_usd_path}")

        # Count meshes before
        mesh_count = sum(
            1
            for p in Usd.PrimRange(stage.GetPseudoRoot())
            if not p.IsPseudoRoot() and p.IsA(UsdGeom.Mesh)
        )

        ctx = ExecutionContext()
        ctx.set_stage(stage)
        so = SceneOptimizerCore.getInstance()

        op_start = time.time()
        so.executeOperation(operation, ctx, op_params)
        op_time = time.time() - op_start

        # Count meshes with primvars:st after
        meshes_with_uvs = 0
        for prim in Usd.PrimRange(stage.GetPseudoRoot()):
            if prim.IsPseudoRoot():
                continue
            if prim.IsA(UsdGeom.Mesh):
                st_attr = prim.GetAttribute("primvars:st")
                if st_attr and st_attr.HasAuthoredValue():
                    meshes_with_uvs += 1

        stage.GetRootLayer().Export(output_usd_path)
        ctx.remove_stage()

        output_size = (
            os.path.getsize(output_usd_path) if os.path.exists(output_usd_path) else 0
        )

        manifest = {
            "status": "success",
            "operation": operation,
            "operation_time": op_time,
            "total_time": time.time() - total_start,
            "stage_size_bytes": output_size,
            "mesh_count": mesh_count,
            "meshes_with_uvs": meshes_with_uvs,
        }

    except Exception:  # noqa: BLE001 — subprocess must always write manifest
        manifest = {
            "status": "error",
            "operation": operation,
            "total_time": time.time() - total_start,
            "error": traceback.format_exc()[-2000:],
        }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f)


if __name__ == "__main__":
    main()
