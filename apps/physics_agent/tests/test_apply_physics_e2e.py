"""End-to-end tests for `apply_physics` against a real USD asset.

These exercise the full `apply_physics` function on the bundled lightbulb
asset and validate that the output USD has the expected UsdPhysics schemas
authored correctly.

Skipped if `pxr` isn't importable.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pxr = pytest.importorskip("pxr", reason="USD (pxr) not available in this env")
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade  # noqa: E402

from physics_agent.functions.apply_physics import (  # noqa: E402
    PhysicsAuthoringError,
    apply_physics,
)

LIGHTBULB_DIR = (
    Path(__file__).resolve().parents[1] / "data" / "examples" / "Lightbulb01"
)
LIGHTBULB_USDZ = LIGHTBULB_DIR / "light_bulb_01.usdz"


@pytest.fixture
def lightbulb_usdz() -> Path:
    if not LIGHTBULB_USDZ.exists():
        pytest.skip(f"Lightbulb example asset missing: {LIGHTBULB_USDZ}")
    return LIGHTBULB_USDZ


def _write_jsonl(path: Path, entries: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


def _glass_props() -> dict:
    return {
        "density": 2500,
        "estimated_mass_kg": 0.008,
        "static_friction": 0.4,
        "dynamic_friction": 0.3,
        "restitution": 0.5,
    }


def _metal_props() -> dict:
    return {
        "density": 8900,
        "estimated_mass_kg": 0.01,
        "static_friction": 0.6,
        "dynamic_friction": 0.5,
        "restitution": 0.3,
    }


def _write_instanced_usd(tmp_path: Path) -> Path:
    asset_path = tmp_path / "referenced_model.usda"
    asset_stage = Usd.Stage.CreateNew(str(asset_path))
    model = UsdGeom.Xform.Define(asset_stage, "/Model")
    asset_stage.SetDefaultPrim(model.GetPrim())
    UsdGeom.Cube.Define(asset_stage, "/Model/Cube")
    asset_stage.GetRootLayer().Save()

    stage_path = tmp_path / "instanced_scene.usda"
    stage = Usd.Stage.CreateNew(str(stage_path))
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    instance = UsdGeom.Xform.Define(stage, "/World/Inst").GetPrim()
    instance.GetReferences().AddReference(str(asset_path), "/Model")
    instance.SetInstanceable(True)
    stage.GetRootLayer().Save()
    return stage_path


def test_apply_physics_authors_expected_schemas(
    tmp_path: Path, lightbulb_usdz: Path
) -> None:
    """Full-pipeline equivalent: applies physics to three prims and verifies
    schemas, attributes, PhysicsScene gravity, material cache, and binding
    purpose are all authored correctly on the output USD.
    """
    predictions_path = tmp_path / "predictions.jsonl"
    output_usd = tmp_path / "light_bulb_01_physics.usda"

    _write_jsonl(
        predictions_path,
        [
            {
                "id": "/light_bulb_01/Geometry/Bulb_Main",
                "classification": {"physical_properties": _glass_props()},
            },
            {
                "id": "/light_bulb_01/Geometry/Bulb_Screw_Cap",
                "classification": {"physical_properties": _metal_props()},
            },
            {
                "id": "/light_bulb_01/Geometry/Filament_Big",
                "classification": {"physical_properties": _metal_props()},
            },
        ],
    )

    out = apply_physics(
        usd_path=str(lightbulb_usdz),
        predictions_path=str(predictions_path),
        output_path=str(output_usd),
    )

    # apply_physics returns an absolute path per its docstring.
    assert Path(out).is_absolute()
    assert Path(out).exists()

    stage = Usd.Stage.Open(str(output_usd))
    assert stage is not None

    # PhysicsScene created at the default-prim root, with gravity matching Z-up.
    scenes = [p for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)]
    assert len(scenes) == 1, f"expected 1 PhysicsScene, got {len(scenes)}"
    scene = UsdPhysics.Scene(scenes[0])
    assert scene.GetGravityMagnitudeAttr().Get() == pytest.approx(9.81, rel=1e-3)
    up_axis = UsdGeom.GetStageUpAxis(stage)
    expected_dir = (0, 0, -1) if up_axis == UsdGeom.Tokens.z else (0, -1, 0)
    gravity_dir = scene.GetGravityDirectionAttr().Get()
    assert tuple(gravity_dir) == expected_dir

    # Each target prim carries the five expected applied schemas.
    expected_schemas = {
        "PhysicsRigidBodyAPI",
        "PhysicsCollisionAPI",
        "PhysicsMeshCollisionAPI",
        "PhysicsMassAPI",
        "MaterialBindingAPI",
    }
    for prim_path, props in [
        ("/light_bulb_01/Geometry/Bulb_Main", _glass_props()),
        ("/light_bulb_01/Geometry/Bulb_Screw_Cap", _metal_props()),
        ("/light_bulb_01/Geometry/Filament_Big", _metal_props()),
    ]:
        prim = stage.GetPrimAtPath(prim_path)
        assert prim.IsValid(), f"missing prim {prim_path}"
        applied = set(prim.GetAppliedSchemas())
        assert expected_schemas.issubset(applied), (
            f"{prim_path} applied schemas {applied} missing {expected_schemas - applied}"
        )
        assert UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Get() is True
        mass_api = UsdPhysics.MassAPI(prim)
        assert mass_api.GetMassAttr().Get() == pytest.approx(
            props["estimated_mass_kg"], rel=1e-3
        )
        assert mass_api.GetDensityAttr().Get() == pytest.approx(
            props["density"], rel=1e-3
        )
        assert (
            UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
            == "convexHull"
        )
        # Physics-purpose material binding preserves visual bindings on the prim.
        rel = UsdShade.MaterialBindingAPI(prim).GetDirectBindingRel("physics")
        assert rel, f"{prim_path} has no physics-purpose material binding"
        targets = rel.GetTargets()
        assert len(targets) == 1

    # Material cache de-duplicated the two metal prims onto a single PhysMat.
    looks = stage.GetPrimAtPath("/light_bulb_01/Looks")
    assert looks.IsValid()
    phys_mats = [c for c in looks.GetChildren() if c.GetName().startswith("PhysMat_")]
    assert len(phys_mats) == 2, (
        f"expected 2 cached PhysMats (glass + metal), got {len(phys_mats)}: "
        f"{[m.GetName() for m in phys_mats]}"
    )


def test_apply_physics_output_is_self_contained_and_relocatable(
    tmp_path: Path, lightbulb_usdz: Path
) -> None:
    """Output must open cleanly from a directory with no source-sibling files.

    Regression: apply_physics previously exported just the root layer, which
    preserved the source asset's relative payload refs (`./Payload/...`)
    that only resolved against the *source* directory. When written to
    `{working_dir}/physics/` — with no Payload sibling — reopening the
    output produced composition errors and missing prims. The fix flattens
    the stage before export so the output is self-contained.
    """
    predictions_path = tmp_path / "predictions.jsonl"
    produce_dir = tmp_path / "physics"
    produce_dir.mkdir()
    output_usd = produce_dir / "light_bulb_01_physics.usda"
    _write_jsonl(
        predictions_path,
        [
            {
                "id": "/light_bulb_01/Geometry/Bulb_Main",
                "classification": {"physical_properties": _glass_props()},
            },
        ],
    )

    apply_physics(
        usd_path=str(lightbulb_usdz),
        predictions_path=str(predictions_path),
        output_path=str(output_usd),
    )

    # The output directory has ONLY the generated usda — no Payload sibling,
    # nothing that would make relative refs in the source asset resolve.
    assert list(produce_dir.iterdir()) == [output_usd]

    stage = Usd.Stage.Open(str(output_usd))
    assert stage is not None

    # A prim that only exists inside the source payload must still be present
    # and valid on reopen — proves the payload was actually composed in.
    payload_only_prim = stage.GetPrimAtPath("/light_bulb_01/Geometry/Internal_Chamber")
    assert payload_only_prim.IsValid(), (
        "payload-provided prim should be composed into a self-contained output"
    )

    # And the prim we authored physics onto should carry the schemas.
    bulb = stage.GetPrimAtPath("/light_bulb_01/Geometry/Bulb_Main")
    assert bulb.IsValid()
    assert "PhysicsRigidBodyAPI" in bulb.GetAppliedSchemas()


def test_apply_physics_respects_custom_output_key(
    tmp_path: Path, lightbulb_usdz: Path
) -> None:
    """When `predict.output_key` != 'classification', apply_physics should read
    physical_properties from the configured key rather than silently skipping.
    """
    predictions_path = tmp_path / "predictions.jsonl"
    output_usd = tmp_path / "light_bulb_01_physics.usda"

    _write_jsonl(
        predictions_path,
        [
            {
                "id": "/light_bulb_01/Geometry/Bulb_Main",
                # Stored under "analysis" — default "classification" would miss it
                "analysis": {"physical_properties": _glass_props()},
            },
        ],
    )

    apply_physics(
        usd_path=str(lightbulb_usdz),
        predictions_path=str(predictions_path),
        output_path=str(output_usd),
        output_key="analysis",
    )

    stage = Usd.Stage.Open(str(output_usd))
    prim = stage.GetPrimAtPath("/light_bulb_01/Geometry/Bulb_Main")
    assert prim.IsValid()
    assert UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Get() is True
    assert UsdPhysics.MassAPI(prim).GetMassAttr().Get() == pytest.approx(
        _glass_props()["estimated_mass_kg"], rel=1e-3
    )


def test_apply_physics_default_output_key_on_mismatched_data_fails_loudly(
    tmp_path: Path, lightbulb_usdz: Path
) -> None:
    """Sanity companion to the custom-output-key test: when data is under
    `analysis` but we read with the default `classification`, fail instead
    of exporting a USD with no authored physics schemas.
    """
    predictions_path = tmp_path / "predictions.jsonl"
    output_usd = tmp_path / "light_bulb_01_physics.usda"

    _write_jsonl(
        predictions_path,
        [
            {
                "id": "/light_bulb_01/Geometry/Bulb_Main",
                "analysis": {"physical_properties": _glass_props()},
            },
        ],
    )

    with pytest.raises(PhysicsAuthoringError, match="No physics schemas were applied"):
        apply_physics(
            usd_path=str(lightbulb_usdz),
            predictions_path=str(predictions_path),
            output_path=str(output_usd),
            # default output_key="classification" — mismatched, so no physics applies
        )

    assert not output_usd.exists()


def test_apply_physics_skips_nonexistent_prims_and_missing_properties(
    tmp_path: Path, lightbulb_usdz: Path
) -> None:
    """A prediction targeting a prim that isn't on the stage, or one whose
    classification lacks `physical_properties`, should be skipped (warned in
    logs, no schema authored) without failing the run.
    """
    predictions_path = tmp_path / "predictions.jsonl"
    output_usd = tmp_path / "light_bulb_01_physics.usda"

    _write_jsonl(
        predictions_path,
        [
            # Valid
            {
                "id": "/light_bulb_01/Geometry/Bulb_Main",
                "classification": {"physical_properties": _glass_props()},
            },
            # Missing prim
            {
                "id": "/light_bulb_01/Geometry/Nonexistent",
                "classification": {"physical_properties": _glass_props()},
            },
            # Missing physical_properties
            {
                "id": "/light_bulb_01/Geometry/Internal_Chamber",
                "classification": {},
            },
        ],
    )

    apply_physics(
        usd_path=str(lightbulb_usdz),
        predictions_path=str(predictions_path),
        output_path=str(output_usd),
    )

    stage = Usd.Stage.Open(str(output_usd))
    # Valid prim got physics
    assert (
        "PhysicsRigidBodyAPI"
        in stage.GetPrimAtPath("/light_bulb_01/Geometry/Bulb_Main").GetAppliedSchemas()
    )
    # Skipped prim stayed clean
    internal = stage.GetPrimAtPath("/light_bulb_01/Geometry/Internal_Chamber")
    assert internal.IsValid()
    assert "PhysicsRigidBodyAPI" not in internal.GetAppliedSchemas()


def test_apply_physics_fails_on_instance_proxy_targets(tmp_path: Path) -> None:
    """Direct authoring onto instance proxy descendants should not export a
    partial-success USD. Optimized pipeline runs must author onto the
    deinstanced USD instead.
    """
    instanced_usd = _write_instanced_usd(tmp_path)
    predictions_path = tmp_path / "predictions.jsonl"
    output_usd = tmp_path / "instanced_scene_physics.usda"
    _write_jsonl(
        predictions_path,
        [
            {
                "id": "/World/Inst/Cube",
                "classification": {"physical_properties": _glass_props()},
            }
        ],
    )

    with pytest.raises(PhysicsAuthoringError, match="instance proxy"):
        apply_physics(
            usd_path=str(instanced_usd),
            predictions_path=str(predictions_path),
            output_path=str(output_usd),
        )

    assert not output_usd.exists()
