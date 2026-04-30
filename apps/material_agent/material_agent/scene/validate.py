# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scene validation — check material bindings for scene pipeline outputs.

Provides data classes and functions for validating per-asset outputs,
payload groups, and the composed scene.  Used by both the CLI
(``material-agent scene validate``) and the standalone script
(``scripts/validate_scene.py``).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SAFE_SCENE_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def _validate_scene_session_id(value: Any) -> str | None:
    """Return a safe scene session ID or None if it is unsafe for path use."""
    session_id = str(value).strip()
    if not _SAFE_SCENE_SESSION_ID_RE.fullmatch(session_id):
        return None
    return session_id


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class AssetReport:
    name: str
    status: str = "unknown"
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    input_meshes: int = 0
    output_meshes: int = 0
    bindings_in_layer: int = 0
    bindings_our: int = 0
    bindings_old: int = 0
    bindings_none: int = 0
    material_defs: int = 0
    deinstanced: int = 0
    sublayers_optimized: bool = False
    topology_match: bool = True
    hierarchy_match: bool = True
    input_instances: int = 0
    output_instances: int = 0
    instances_kept: int = 0
    instances_deinstanced: int = 0
    predictions_count: int = 0
    has_predictions: bool = False

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0


@dataclass
class PayloadReport:
    name: str
    status: str = "unknown"
    depth: int = 0
    predictions_count: int = 0
    has_predictions: bool = False
    has_output_usd: bool = False
    has_material_layer: bool = False
    instance_count: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0


@dataclass
class SceneReport:
    assets: list[AssetReport] = field(default_factory=list)
    payloads: list[PayloadReport] = field(default_factory=list)
    total_bindings: int = 0
    total_deinstanced: int = 0
    composed_our: int = 0
    composed_old: int = 0
    composed_none: int = 0
    composed_instance_our: int = 0
    composed_instance_old: int = 0
    composed_instances_checked: int = 0
    composed_subset_our: int = 0
    composed_subset_old: int = 0
    composed_subsets_checked: int = 0
    composed_scene_path: str = ""
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Layer-level checks (fast, no stage composition)
# ---------------------------------------------------------------------------


def count_layer_bindings(layer: Any) -> tuple[int, int, int]:
    """Count material bindings, material defs, and de-instanced prims in a layer."""
    bindings = 0
    mat_defs = 0
    deinstanced = 0

    def _walk(spec: Any) -> None:
        nonlocal bindings, mat_defs, deinstanced
        if spec.typeName == "Material":
            mat_defs += 1
        for rel_name in spec.relationships.keys():
            if "material:binding" in rel_name:
                bindings += 1
        if spec.HasInfo("instanceable") and spec.GetInfo("instanceable") is False:
            deinstanced += 1
        for child_name in spec.nameChildren.keys():
            _walk(spec.nameChildren[child_name])

    _walk(layer.pseudoRoot)
    return bindings, mat_defs, deinstanced


def check_layer_sublayers(layer: Any) -> tuple[bool, list[str]]:
    """Check if layer has sublayers and whether they point to optimized USDs."""
    has_optimized = False
    sublayers = list(layer.subLayerPaths)
    for sl in sublayers:
        if "optimized" in sl:
            has_optimized = True
    return has_optimized, sublayers


# ---------------------------------------------------------------------------
# Stage-level checks (slower, requires composition)
# ---------------------------------------------------------------------------


def check_stage_bindings(stage: Any) -> tuple[int, int, int]:
    """Count prims bound to our materials, old materials, or nothing."""
    our = 0
    old = 0
    none_ = 0

    for prim in stage.Traverse():
        if prim.GetTypeName() not in ("Mesh", "GeomSubset"):
            continue
        rel = prim.GetRelationship("material:binding")
        if rel and rel.GetTargets():
            target = str(rel.GetTargets()[0])
            if "/Looks/" in target:
                our += 1
            else:
                old += 1
        else:
            none_ += 1

    return our, old, none_


def check_topology_match(input_usd: str, output_usd: str) -> tuple[bool, int, int]:
    """Check if output preserves the input mesh topology."""
    from pxr import Usd

    stage_in = Usd.Stage.Open(input_usd)
    stage_out = Usd.Stage.Open(output_usd)

    in_meshes = {
        str(p.GetPath()) for p in stage_in.Traverse() if p.GetTypeName() == "Mesh"
    }
    out_meshes = {
        str(p.GetPath()) for p in stage_out.Traverse() if p.GetTypeName() == "Mesh"
    }

    return in_meshes == out_meshes, len(in_meshes), len(out_meshes)


def check_hierarchy_match(
    input_usd: str, output_usd: str
) -> tuple[bool, list[str], list[str]]:
    """Check if output preserves the full prim hierarchy."""
    from pxr import Usd

    stage_in = Usd.Stage.Open(input_usd)
    stage_out = Usd.Stage.Open(output_usd)

    in_prims = {str(p.GetPath()) for p in stage_in.Traverse()}
    out_prims = {str(p.GetPath()) for p in stage_out.Traverse()}
    out_prims_filtered = {p for p in out_prims if not p.startswith("/World/Looks/")}

    only_in = sorted(in_prims - out_prims_filtered)
    only_out = sorted(out_prims_filtered - in_prims)

    return len(only_in) == 0 and len(only_out) == 0, only_in, only_out


def check_instances(
    input_usd: str, output_usd: str
) -> tuple[int, int, int, int, list[str]]:
    """Check instance preservation between input and output."""
    from pxr import Usd

    stage_in = Usd.Stage.Open(input_usd)
    stage_out = Usd.Stage.Open(output_usd)

    in_instances = {str(p.GetPath()) for p in stage_in.Traverse() if p.IsInstance()}
    out_instances = {str(p.GetPath()) for p in stage_out.Traverse() if p.IsInstance()}

    deinstanced_paths = sorted(in_instances - out_instances)
    kept = len(in_instances & out_instances)

    return (
        len(in_instances),
        len(out_instances),
        kept,
        len(deinstanced_paths),
        deinstanced_paths,
    )


# ---------------------------------------------------------------------------
# Asset validation
# ---------------------------------------------------------------------------


def _count_predictions(path: Path) -> int:
    """Count valid predictions in a JSONL file."""
    count = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                pred = json.loads(line)
                if pred.get("id"):
                    count += 1
            except json.JSONDecodeError:
                continue
    return count


def validate_asset(wdir: Path, verbose: bool = False) -> AssetReport:
    """Validate a single asset working directory."""
    from pxr import Sdf, Usd

    name = wdir.name.lstrip(".")
    report = AssetReport(name=name)

    state_file = wdir / ".pipeline_state.json"
    if not state_file.exists():
        report.errors.append("No pipeline state file")
        report.status = "no_state"
        return report

    state = json.loads(state_file.read_text())
    completed = state.get("completed_steps", [])
    failed = state.get("failed_steps", [])

    has_apply = "apply" in completed
    has_predict = "predict" in completed

    simulate_marker = wdir / ".simulate"
    is_simulated = simulate_marker.exists()

    if failed:
        report.status = "failed"
        report.errors.append(f"Failed steps: {failed}")
    elif has_apply:
        report.status = "completed"
    elif has_predict:
        report.status = "completed"
    elif is_simulated:
        report.status = "completed"
    else:
        report.status = "incomplete"
        report.errors.append(
            f"Neither predict nor apply completed (steps: {completed})"
        )
        return report

    if is_simulated:
        report.warnings.append(
            "Processed in simulate mode (mock predictions, not real VLM inference)"
        )

    restored_path = wdir / "restored" / "restored_predictions.jsonl"
    raw_path = wdir / "predictions" / "predictions.jsonl"
    predictions_path = restored_path if restored_path.exists() else raw_path

    if predictions_path.exists():
        report.has_predictions = True
        report.predictions_count = _count_predictions(predictions_path)
        if report.predictions_count == 0:
            report.warnings.append("Predictions file exists but contains 0 entries")
    else:
        report.errors.append("No predictions file found")
        return report

    output_path = wdir / "output" / "output.usd"
    if has_apply:
        if not output_path.exists():
            report.errors.append("Apply ran but output USD not found")
            return report

        layer = Sdf.Layer.FindOrOpen(str(output_path))
        if not layer:
            report.errors.append("Cannot open output layer")
            return report

        report.bindings_in_layer, report.material_defs, report.deinstanced = (
            count_layer_bindings(layer)
        )
        report.sublayers_optimized, sublayers = check_layer_sublayers(layer)

        if report.bindings_in_layer == 0:
            report.errors.append("Zero material bindings in output layer")
        if report.material_defs == 0:
            report.errors.append("Zero material definitions in output layer")
        if report.sublayers_optimized:
            report.warnings.append(
                "Output sublayers optimized USD (should be original for scene mode)"
            )
        for sl in sublayers:
            if "optimized" in sl:
                report.errors.append(
                    f"Sublayer points to optimized USD: {Path(sl).name}"
                )

        try:
            stage = Usd.Stage.Open(str(output_path))
            report.bindings_our, report.bindings_old, report.bindings_none = (
                check_stage_bindings(stage)
            )
            report.output_meshes = sum(
                1 for p in stage.Traverse() if p.GetTypeName() == "Mesh"
            )
        except Exception as e:
            report.warnings.append(f"Could not open output stage: {e}")

        config_path = wdir.parent / f"{name}.yaml"
        if config_path.exists():
            _check_topology_and_instances(config_path, output_path, report)

    return report


def _check_topology_and_instances(
    config_path: Path, output_path: Path, report: AssetReport
) -> None:
    """Run topology, hierarchy, and instance checks for a per-asset output."""
    try:
        import yaml

        with open(config_path) as f:
            config = yaml.safe_load(f)
        input_usd = config.get("input", {}).get("usd_path", "")
        if input_usd and not Path(input_usd).is_absolute():
            input_usd = str((config_path.parent / input_usd).resolve())
        if not (input_usd and Path(input_usd).exists()):
            return

        match, in_count, out_count = check_topology_match(input_usd, str(output_path))
        report.input_meshes = in_count
        report.topology_match = match
        if not match:
            report.warnings.append(
                f"Topology mismatch: input={in_count} output={out_count} meshes"
            )

        hier_match, only_in, only_out = check_hierarchy_match(
            input_usd, str(output_path)
        )
        report.hierarchy_match = hier_match
        if not hier_match:
            if only_in:
                report.warnings.append(
                    f"Hierarchy: {len(only_in)} prims only in input "
                    f"(e.g. {only_in[0].split('/')[-1]})"
                )
            if only_out:
                report.warnings.append(
                    f"Hierarchy: {len(only_out)} prims only in output "
                    f"(e.g. {only_out[0].split('/')[-1]})"
                )

        (
            report.input_instances,
            report.output_instances,
            report.instances_kept,
            report.instances_deinstanced,
            _deinstanced_paths,
        ) = check_instances(input_usd, str(output_path))

        if report.instances_deinstanced > 0:
            report.errors.append(
                f"{report.instances_deinstanced} instances lost "
                f"(should inherit from prototype, not be de-instanced)"
            )
        unexpected = report.output_instances - report.instances_kept
        if unexpected > 0:
            report.warnings.append(f"{unexpected} unexpected new instances in output")
    except Exception as e:
        report.warnings.append(f"Could not check topology: {e}")


# ---------------------------------------------------------------------------
# Payload validation
# ---------------------------------------------------------------------------


def _validate_payload_group(
    pg: dict, manifest_dir: Path, verbose: bool = False
) -> PayloadReport:
    """Validate a single payload group."""
    from pxr import Sdf

    name = pg.get("group_name", "unknown")
    report = PayloadReport(
        name=name,
        instance_count=pg.get("instance_count", 0),
        depth=pg.get("depth", 0),
    )
    report.status = pg.get("status", "unknown")

    working_dir = pg.get("working_dir")
    if working_dir:
        working_path = Path(working_dir)
        simulate_marker = working_path / ".simulate"
        if simulate_marker.exists():
            report.warnings.append("Processed in simulate mode (mock predictions)")
        predictions_path = working_path / "predictions" / "predictions.jsonl"
        if predictions_path.exists():
            report.has_predictions = True
            report.predictions_count = _count_predictions(predictions_path)
            if report.predictions_count == 0:
                report.warnings.append("Predictions file exists but contains 0 entries")
        elif report.status == "completed":
            report.errors.append("Status is completed but no predictions file found")

    output_usd = pg.get("output_usd_path")
    if output_usd and Path(output_usd).exists():
        report.has_output_usd = True
        layer = Sdf.Layer.FindOrOpen(output_usd)
        if layer:
            bindings, mat_defs, deinstanced = count_layer_bindings(layer)
            if bindings == 0 and report.status == "completed":
                report.warnings.append("Output USD has 0 material bindings")
            if mat_defs == 0 and report.status == "completed":
                report.warnings.append("Output USD has 0 material definitions")
            if deinstanced > 0:
                report.errors.append(
                    f"Output USD contains {deinstanced} de-instanced prims "
                    f"(instanceable=false) — instancing must be preserved"
                )
            if not layer.subLayerPaths:
                report.warnings.append(
                    "Output USD has no sublayers (expected to sublayer original)"
                )
    elif report.status == "completed":
        report.warnings.append("No output.usd found for completed payload")

    return report


# ---------------------------------------------------------------------------
# Scene validation
# ---------------------------------------------------------------------------


def validate_scene(scene_config_path: Path, verbose: bool = False) -> SceneReport:
    """Validate all assets in a scene pipeline."""
    import yaml
    from pxr import Sdf, Usd

    report = SceneReport()

    with open(scene_config_path) as f:
        config = yaml.safe_load(f)

    project = config.get("project", {})
    raw_session_id = project.get("session_id") or project.get("name") or "scene"
    session_id = _validate_scene_session_id(raw_session_id)
    if session_id is None:
        report.errors.append(
            "Unsafe scene session_id/name for manifest directory: "
            f"{raw_session_id!r}. Use 1-128 characters from "
            "A-Z, a-z, 0-9, underscore, hyphen, or dot, starting with "
            "an alphanumeric character."
        )
        return report
    manifest_dir = scene_config_path.parent / f".{session_id}_scene"
    manifest_path = manifest_dir / "manifest.json"

    if not manifest_path.exists():
        report.errors.append(f"Manifest not found: {manifest_path}")
        return report

    manifest = json.loads(manifest_path.read_text())

    sa_by_id = {sa["id"]: sa for sa in manifest.get("sub_assets", [])}

    # Build instance group representative lookup
    ig_representative: dict[str, str] = {}
    for ig in manifest.get("instance_groups", []):
        rep_id = ig.get("representative_id")
        if rep_id:
            rep_sa = sa_by_id.get(rep_id)
            if rep_sa:
                ig_representative[ig["group_name"]] = rep_sa["name"]
    for sa in manifest.get("sub_assets", []):
        ig_name = sa.get("instance_group")
        if ig_name and sa.get("config_path") and ig_name not in ig_representative:
            ig_representative[ig_name] = sa["name"]

    # Check instance group representatives
    for ig in manifest.get("instance_groups", []):
        rep_id = ig.get("representative_id")
        group_name = ig.get("group_name", "?")
        member_count = ig.get("instance_count", 0)
        if not rep_id:
            report.warnings.append(
                f"Instance group '{group_name}' has no representative "
                f"({member_count} members not in sub-assets)"
            )
            continue
        rep_sa = sa_by_id.get(rep_id)
        if not rep_sa:
            report.errors.append(
                f"Instance group '{group_name}' representative '{rep_id}' "
                f"not found in manifest"
            )
            continue
        if rep_sa.get("status") != "completed":
            report.errors.append(
                f"Instance group '{group_name}' representative "
                f"'{rep_sa.get('name', rep_id)}' has status "
                f"'{rep_sa.get('status')}' (expected 'completed') — "
                f"{member_count} members will have no materials"
            )

    # First pass: validate assets with configs
    rep_reports: dict[str, AssetReport] = {}
    for sa in manifest.get("sub_assets", []):
        config_path_str = sa.get("config_path")
        if not config_path_str:
            continue
        config_path = Path(config_path_str)
        if not config_path.exists():
            ar = AssetReport(
                name=sa["name"], status="no_config", errors=["Config not found"]
            )
            rep_reports[sa["name"]] = ar
            continue

        wdir = config_path.parent / f".{config_path.stem}"
        asset_report = validate_asset(wdir, verbose)

        # Check if processed via payload group
        manifest_status = sa.get("status", "")
        if asset_report.status == "incomplete" and manifest_status == "completed":
            prim_path = sa.get("prim_path", "")
            for pg in manifest.get("payload_groups", []):
                if pg.get("status") == "completed" and prim_path in pg.get(
                    "instance_paths", []
                ):
                    asset_report.status = "completed"
                    asset_report.errors.clear()
                    asset_report.warnings.append(
                        f"Processed via payload group '{pg['group_name']}'"
                    )
                    break

        if asset_report.status != "completed":
            asset_report.status = sa.get("status", asset_report.status)
        rep_reports[sa["name"]] = asset_report

    # Second pass: build final list with instance group inheritance
    for sa in manifest.get("sub_assets", []):
        config_path_str = sa.get("config_path")
        ig = sa.get("instance_group")

        if config_path_str:
            ar = rep_reports.get(sa["name"])
            if ar:
                report.assets.append(ar)
                report.total_bindings += ar.bindings_in_layer
                report.total_deinstanced += ar.deinstanced
            continue

        rep_name = ig_representative.get(ig, "") if ig else ""
        rep_report = rep_reports.get(rep_name) if rep_name else None

        if rep_report and rep_report.ok:
            ar = AssetReport(
                name=sa["name"],
                status="inherited",
                predictions_count=rep_report.predictions_count,
                has_predictions=rep_report.has_predictions,
            )
            ar.warnings.append(f"Instance group member (inherits from {rep_name})")
            report.assets.append(ar)
        elif ig:
            ar = AssetReport(name=sa["name"], status="inherited")
            ar.warnings.append(
                f"Instance group '{ig}' member (representative not validated)"
            )
            report.assets.append(ar)
        else:
            report.assets.append(
                AssetReport(
                    name=sa["name"],
                    status="no_config",
                    errors=["Config not found"],
                )
            )

    # Validate payload groups
    for pg in manifest.get("payload_groups", []):
        pg_report = _validate_payload_group(pg, manifest_dir, verbose)
        report.payloads.append(pg_report)

    # Check payload material layers
    payload_layers_dir = manifest_dir / "output" / "payload_layers"
    if manifest.get("payload_groups") and not payload_layers_dir.exists():
        report.warnings.append(
            f"Payload layers directory not found: {payload_layers_dir}"
        )

    # Check composed scene
    composed_path = manifest_dir / "output" / "composed_scene.usd"
    if composed_path.exists():
        report.composed_scene_path = str(composed_path)
        layer = Sdf.Layer.FindOrOpen(str(composed_path))
        if layer:
            bindings, mat_defs, _deinst = count_layer_bindings(layer)
            if bindings == 0:
                report.warnings.append("Composed layer contains 0 material bindings")
            if mat_defs == 0:
                report.warnings.append("Composed layer contains 0 material definitions")

        try:
            stage = Usd.Stage.Open(str(composed_path))
            our, old, none_ = check_stage_bindings(stage)
            report.composed_our = our
            report.composed_old = old
            report.composed_none = none_

            instance_with_our = 0
            instance_with_old = 0
            instance_checked = 0
            subset_with_our = 0
            subset_with_old = 0
            subset_checked = 0
            for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
                if not prim.IsInstanceProxy():
                    continue
                typ = prim.GetTypeName()
                if typ not in ("Mesh", "GeomSubset"):
                    continue

                rel = prim.GetRelationship("material:binding")
                has_our = False
                if rel and rel.GetTargets():
                    target = str(rel.GetTargets()[0])
                    has_our = "/Looks/" in target

                if typ == "Mesh":
                    instance_checked += 1
                    if has_our:
                        instance_with_our += 1
                    elif rel and rel.GetTargets():
                        instance_with_old += 1
                else:
                    subset_checked += 1
                    if has_our:
                        subset_with_our += 1
                    elif rel and rel.GetTargets():
                        subset_with_old += 1

            report.composed_instance_our = instance_with_our
            report.composed_instance_old = instance_with_old
            report.composed_instances_checked = instance_checked
            report.composed_subset_our = subset_with_our
            report.composed_subset_old = subset_with_old
            report.composed_subsets_checked = subset_checked

            instance_missing = instance_checked - instance_with_our - instance_with_old
            if instance_with_our < instance_checked:
                report.errors.append(
                    f"Composed scene: {instance_checked - instance_with_our}/"
                    f"{instance_checked} instance proxies missing our materials "
                    f"({instance_with_old} old, {instance_missing} none)"
                )
            subset_missing = subset_checked - subset_with_our - subset_with_old
            if subset_with_our < subset_checked:
                report.errors.append(
                    f"Composed scene: {subset_checked - subset_with_our}/"
                    f"{subset_checked} GeomSubsets missing our materials "
                    f"({subset_with_old} old, {subset_missing} none)"
                )
        except Exception as e:
            report.warnings.append(f"Could not validate composed scene: {e}")

    return report


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------


def format_asset_report(r: AssetReport, verbose: bool = False) -> list[str]:
    """Format an asset report as lines of text."""
    lines: list[str] = []

    if r.status == "inherited":
        if verbose:
            lines.append(f"  [ IG ] {r.name:35s} predictions={r.predictions_count:>5d}")
            for w in r.warnings:
                lines.append(f"         INFO:  {w}")
        return lines

    status_icon = "PASS" if r.ok else "FAIL"
    if r.bindings_in_layer > 0:
        lines.append(
            f"  [{status_icon}] {r.name:35s} "
            f"bindings={r.bindings_in_layer:>5d}  "
            f"mats={r.material_defs:>3d}  "
            f"inst={r.input_instances:>3d}->{r.output_instances:>3d}  "
            f"our={r.bindings_our:>5d}  old={r.bindings_old:>5d}  "
            f"none={r.bindings_none:>3d}"
        )
    else:
        lines.append(
            f"  [{status_icon}] {r.name:35s} predictions={r.predictions_count:>5d}"
        )

    if verbose or not r.ok:
        for e in r.errors:
            lines.append(f"         ERROR: {e}")
        if not r.topology_match:
            lines.append("         WARN:  Mesh topology mismatch")
        if not r.hierarchy_match:
            lines.append("         WARN:  Prim hierarchy mismatch")
        if r.instances_deinstanced > 0:
            lines.append(
                f"         INFO:  {r.instances_deinstanced} instances de-instanced "
                f"for material overrides"
            )
        for w in r.warnings:
            lines.append(f"         WARN:  {w}")

    return lines
