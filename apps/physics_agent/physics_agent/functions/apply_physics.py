"""Apply physics properties from predictions JSONL to a USD file."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

logger = logging.getLogger(__name__)


class PhysicsAuthoringError(RuntimeError):
    """Raised when physics schemas cannot be authored safely."""


def load_predictions(jsonl_path: str) -> list[dict]:
    predictions = []
    with open(jsonl_path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                predictions.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(
                    "Skipping malformed JSON in %s at line %d: %s",
                    jsonl_path,
                    lineno,
                    e,
                )
    return predictions


def _create_physics_material(
    stage: Usd.Stage,
    material_path: str,
    static_friction: float,
    dynamic_friction: float,
    restitution: float,
) -> UsdShade.Material:
    material = UsdShade.Material.Define(stage, material_path)
    physics_mat = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics_mat.CreateStaticFrictionAttr(static_friction)
    physics_mat.CreateDynamicFrictionAttr(dynamic_friction)
    physics_mat.CreateRestitutionAttr(restitution)
    return material


def _apply_physics_to_prim(
    stage: Usd.Stage,
    prim_path: str,
    physics_props: dict,
    collision_approx: str,
    materials_root: str,
    material_cache: dict[str, UsdShade.Material],
) -> bool:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        logger.warning("Prim not found: %s", prim_path)
        return False

    if prim.IsInstanceProxy():
        raise PhysicsAuthoringError(
            f"Prim {prim_path} is an instance proxy and cannot be authored on. "
            "Apply physics to a deinstanced USD instead, for example by "
            "running optimize_usd with enable_deinstance and using the raw "
            "predict output keyed to the optimized USD."
        )

    mass = physics_props.get("estimated_mass_kg", 0.0)
    density = physics_props.get("density", 0.0)
    static_friction = physics_props.get("static_friction", 0.5)
    dynamic_friction = physics_props.get("dynamic_friction", 0.4)
    restitution = physics_props.get("restitution", 0.3)

    rigid_body = UsdPhysics.RigidBodyAPI.Apply(prim)
    rigid_body.CreateRigidBodyEnabledAttr(True)

    collision = UsdPhysics.CollisionAPI.Apply(prim)
    collision.CreateCollisionEnabledAttr(True)

    if collision_approx != "none":
        mesh_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
        mesh_api.CreateApproximationAttr(collision_approx)

    mass_api = UsdPhysics.MassAPI.Apply(prim)
    if mass > 0:
        mass_api.CreateMassAttr(mass)
    if density > 0:
        mass_api.CreateDensityAttr(density)

    mat_key = f"sf{static_friction:.2f}_df{dynamic_friction:.2f}_r{restitution:.2f}"
    if mat_key not in material_cache:
        safe_name = mat_key.replace(".", "_").replace("-", "m")
        mat_path = f"{materials_root}/PhysMat_{safe_name}"
        material = _create_physics_material(
            stage, mat_path, static_friction, dynamic_friction, restitution
        )
        material_cache[mat_key] = material
        logger.info(
            "  Created physics material: %s (sf=%.2f df=%.2f r=%.2f)",
            mat_path,
            static_friction,
            dynamic_friction,
            restitution,
        )

    binding_api = UsdShade.MaterialBindingAPI.Apply(prim)
    binding_api.Bind(
        material_cache[mat_key],
        UsdShade.Tokens.weakerThanDescendants,
        "physics",
    )

    logger.info(
        "  Applied physics to %s: mass=%.3f kg density=%.0f "
        "friction=%.2f/%.2f restitution=%.2f",
        prim_path,
        mass,
        density,
        static_friction,
        dynamic_friction,
        restitution,
    )

    return True


def apply_physics(
    usd_path: str,
    predictions_path: str,
    output_path: str,
    collision_approx: str = "convexHull",
    output_key: str = "classification",
) -> str:
    """Apply physics properties from predictions to a USD file.

    Reads a predictions JSONL file (from the physics-agent pipeline) and applies
    UsdPhysics schemas (RigidBodyAPI, CollisionAPI, MassAPI, MaterialAPI) to the
    matching prims in a USD stage, producing a simulation-ready USD file.

    Args:
        usd_path: Path to input USD file.
        predictions_path: Path to predictions JSONL from physics-agent predict step.
        output_path: Path for the output USD file.
        collision_approx: Collision approximation — "convexHull", "convexDecomposition",
            "boundingCube", "boundingSphere", "meshSimplification", or "none".
        output_key: Key under which the VLM classification dict is stored in
            each prediction entry. Must match `predict.output_key` from the
            upstream step (defaults to "classification").

    Returns:
        Absolute path to the created USD file.
    """
    predictions = load_predictions(predictions_path)
    logger.info("Loaded %d predictions from %s", len(predictions), predictions_path)

    stage = Usd.Stage.Open(usd_path)
    if not stage:
        raise RuntimeError(f"Failed to open USD stage: {usd_path}")

    # Add a PhysicsScene prim if none exists
    physics_scenes = [p for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)]
    if not physics_scenes:
        default_prim = stage.GetDefaultPrim()
        if default_prim and default_prim.IsValid():
            scene_path = default_prim.GetPath().AppendChild("PhysicsScene")
        else:
            scene_path = Sdf.Path("/PhysicsScene")
        physics_scene = UsdPhysics.Scene.Define(stage, scene_path)
        physics_scene.CreateGravityMagnitudeAttr(9.81)
        up_axis = UsdGeom.GetStageUpAxis(stage)
        if up_axis == UsdGeom.Tokens.y:
            physics_scene.CreateGravityDirectionAttr(Gf.Vec3f(0, -1, 0))
        else:
            physics_scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
        logger.info("Created PhysicsScene at %s", scene_path)

    default_prim = stage.GetDefaultPrim()
    materials_root = (
        str(default_prim.GetPath()) + "/Looks"
        if (default_prim and default_prim.IsValid())
        else "/Looks"
    )

    material_cache: dict[str, UsdShade.Material] = {}
    applied = 0
    skipped = 0

    for pred in predictions:
        prim_path = pred.get("id", "")
        classification = pred.get(output_key, {})
        physics_props = classification.get("physical_properties", {})

        if not prim_path:
            logger.warning("Prediction missing 'id', skipping")
            skipped += 1
            continue

        if not physics_props:
            logger.warning("No physical_properties for %s, skipping", prim_path)
            skipped += 1
            continue

        if _apply_physics_to_prim(
            stage,
            prim_path,
            physics_props,
            collision_approx,
            materials_root,
            material_cache,
        ):
            applied += 1
        else:
            skipped += 1

    if predictions and applied == 0:
        raise PhysicsAuthoringError(
            f"No physics schemas were applied from {len(predictions)} prediction(s); "
            f"{skipped} were skipped. Check prediction prim paths, "
            f"output_key={output_key!r}, and whether the target USD contains "
            "authorable prims."
        )

    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    # Flatten the full composition before export so the result is
    # self-contained. Exporting the root layer alone preserves relative
    # payload/reference paths (e.g. "./Payload/Contents.usda") that resolved
    # against the source location but break at the new output location —
    # notably when the input is a USDZ package with bundled payloads.
    # Flattening bakes all composed geometry, materials, and the physics
    # schemas authored above into one relocatable layer.
    stage.Flatten().Export(str(output))

    logger.info(
        "Saved %s: %d prims with physics, %d skipped, %d materials created",
        output,
        applied,
        skipped,
        len(material_cache),
    )
    return str(output)
