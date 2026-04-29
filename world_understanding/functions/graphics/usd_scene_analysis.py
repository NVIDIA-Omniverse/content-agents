# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""USD scene analysis functions.

Higher-level analysis functions for detecting and classifying objects
in USD scenes using a feature-scoring algorithm.
"""

from __future__ import annotations

import logging
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pxr import Usd

logger = logging.getLogger(__name__)

_SKIP_TYPES = frozenset({"Material", "Shader"})
_MATERIAL_SCOPE_NAMES = frozenset({"Looks", "Materials", "materials", "looks"})


def _find_content_root(prim: Usd.Prim, max_depth: int = 5) -> Usd.Prim:
    """Descend through thin hierarchy nodes to find the actual content root.

    Many USD scenes wrap their content in a chain of thin containers
    (e.g. ``/Root/Root/HumanoidsDemo``). This helper walks down the
    hierarchy as long as the current node has very few children and the
    content is concentrated in a single child (not spread across siblings).

    Args:
        prim: Starting prim to descend from.
        max_depth: Maximum number of levels to descend.

    Returns:
        The deepest prim that looks like the content root.
    """
    from pxr import Usd

    if max_depth <= 0:
        return prim
    children = list(prim.GetFilteredChildren(Usd.TraverseInstanceProxies()))
    # Exclude material scopes (Looks, Materials) from content root search
    children = [c for c in children if c.GetName() not in _MATERIAL_SCOPE_NAMES]
    if len(children) > 5:
        return prim
    if not children:
        return prim

    # Count grandchildren for each child to gauge content distribution
    gc_counts = [
        (c, len(list(c.GetFilteredChildren(Usd.TraverseInstanceProxies()))))
        for c in children
    ]
    gc_counts.sort(key=lambda x: x[1], reverse=True)
    best_child, best_gc_count = gc_counts[0]

    # Stop if multiple children have significant content — the content is
    # spread across siblings rather than concentrated in one subtree.
    if len(gc_counts) > 1:
        _, second_gc_count = gc_counts[1]
        if second_gc_count >= 2 and second_gc_count >= best_gc_count * 0.3:
            return prim

    if best_gc_count >= len(children):
        return _find_content_root(best_child, max_depth - 1)
    return prim


# ------------------------------------------------------------------
# Phase 0 helpers: pre-computation caches
# ------------------------------------------------------------------


def _build_mesh_ancestry_cache(root_prim: Usd.Prim) -> set[str]:
    """Build a set of all prim paths that have at least one Mesh descendant.

    Walks every prim under *root_prim*; for each Mesh found, marks all
    ancestor paths up to (and including) the root.
    """
    from pxr import Usd

    paths_with_meshes: set[str] = set()
    root_path = str(root_prim.GetPath())
    for prim in Usd.PrimRange(root_prim, Usd.TraverseInstanceProxies()):
        if str(prim.GetTypeName()) == "Mesh":
            # Walk upward from the mesh's parent to root_path
            cur = prim.GetParent()
            while cur and cur.IsValid():
                cp = str(cur.GetPath())
                if cp in paths_with_meshes:
                    break  # ancestors already cached
                paths_with_meshes.add(cp)
                if cp == root_path:
                    break
                cur = cur.GetParent()
    return paths_with_meshes


def _build_subtree_refs_cache(
    prim_refs: dict[str, list[str]], root_path: str
) -> dict[str, set[str]]:
    """Build a mapping of prim_path -> set of all sub-USD asset paths in subtree.

    Only considers prim paths that are under *root_path*.  The result is
    built bottom-up by sorting paths longest-first and propagating each
    prim's own refs to all its ancestors.
    """
    cache: dict[str, set[str]] = {}

    # Collect only refs under the content root
    relevant: list[tuple[str, list[str]]] = []
    prefix = root_path + "/"
    for p, refs in prim_refs.items():
        if p == root_path or p.startswith(prefix):
            relevant.append((p, refs))

    # Sort deepest first so children are processed before parents
    relevant.sort(key=lambda x: x[0].count("/"), reverse=True)

    for p, refs in relevant:
        own = set(refs)
        cache.setdefault(p, set()).update(own)
        # Propagate upward to each ancestor down to root_path
        parts = p.split("/")
        # Build ancestor paths from parent up to root
        for i in range(len(parts) - 1, 0, -1):
            ancestor = "/".join(parts[:i])
            if len(ancestor) < len(root_path):
                break
            cache.setdefault(ancestor, set()).update(cache[p])
            if ancestor == root_path:
                break

    return cache


def _compute_sibling_homogeneity_map(
    parent_prim: Usd.Prim, prim_refs: dict[str, list[str]]
) -> dict[str, float]:
    """Compute sibling_homogeneity for all children of *parent_prim*.

    Groups siblings by their frozen direct-ref sets, then for each child
    returns the fraction of siblings sharing the same ref signature.
    """
    from pxr import Usd

    children = list(parent_prim.GetFilteredChildren(Usd.TraverseInstanceProxies()))
    if not children:
        return {}

    # Map each child path to its frozen ref set
    ref_sigs: dict[str, frozenset[str]] = {}
    for child in children:
        cp = str(child.GetPath())
        refs = prim_refs.get(cp, [])
        ref_sigs[cp] = frozenset(refs)

    # Count how many siblings share each signature
    from collections import Counter

    sig_counts = Counter(ref_sigs.values())

    n_siblings = len(children)
    result: dict[str, float] = {}
    for child in children:
        cp = str(child.GetPath())
        sig = ref_sigs[cp]
        result[cp] = sig_counts[sig] / n_siblings
    return result


# ------------------------------------------------------------------
# Phase 2: CandidateFeatures dataclass
# ------------------------------------------------------------------


@dataclass
class CandidateFeatures:
    """Feature vector for a candidate prim."""

    path: str
    subtree_ref_diversity: int = 0
    max_subtree_reuse: int = 0
    direct_ref_reuse: int = 0
    sibling_homogeneity: float = 0.0
    child_count: int = 0
    child_type_diversity: int = 0
    has_skel_root: bool = False
    rel_depth: int = 0
    has_mesh_descendants: bool = False
    classification: str = "component"
    subtree_refs: set[str] = field(default_factory=set)


# ------------------------------------------------------------------
# Phase 3: Classification rules
# ------------------------------------------------------------------

_CLASSIFICATION_OBJECT_ROOT = "object_root"
_CLASSIFICATION_CATEGORY = "category"
_CLASSIFICATION_BUILDING_BLOCK = "building_block"
_CLASSIFICATION_COMPONENT = "component"


def _classify_candidate(feat: CandidateFeatures, threshold: int) -> str:
    """Apply priority-ordered classification rules to a candidate."""
    # Rule 1: building block (only prims that directly reference a
    # highly-reused asset — containers of building blocks are NOT excluded)
    if feat.subtree_ref_diversity <= 1 and feat.direct_ref_reuse >= threshold:
        return _CLASSIFICATION_BUILDING_BLOCK

    # Rule 2: depth-1 container (no direct refs → organizational grouping)
    if feat.rel_depth == 1 and feat.direct_ref_reuse == 0:
        return _CLASSIFICATION_CATEGORY

    # Rule 3: multi-asset assembly
    if feat.subtree_ref_diversity >= 2:
        return _CLASSIFICATION_OBJECT_ROOT

    # Rule 4: single-asset, low-reuse
    if feat.direct_ref_reuse > 0 and feat.direct_ref_reuse < threshold:
        return _CLASSIFICATION_OBJECT_ROOT

    # Rule 5: SkelRoot presence
    if feat.has_skel_root:
        return _CLASSIFICATION_OBJECT_ROOT

    # Rule 6: inline geometry (no sub-USD refs but has mesh descendants)
    if feat.subtree_ref_diversity == 0 and feat.has_mesh_descendants:
        return _CLASSIFICATION_OBJECT_ROOT

    # Rule 7: category node (many children, shallow depth, no direct refs)
    if feat.child_count >= 3 and feat.rel_depth <= 2 and feat.direct_ref_reuse == 0:
        return _CLASSIFICATION_CATEGORY

    # Default
    return _CLASSIFICATION_COMPONENT


# ------------------------------------------------------------------
# Phase 4: Non-overlap resolution
# ------------------------------------------------------------------


def _resolve_overlaps(
    classifications: dict[str, str],
) -> dict[str, str]:
    """Resolve overlapping object_root claims.

    Greedy shallowest-first claiming: sorts object_roots by depth then
    alphabetically.  An object_root is demoted to component if any
    ancestor is already claimed (by another object_root) or if any
    ancestor is a building_block.
    """
    resolved = dict(classifications)

    # Building blocks block their entire subtree
    blocked: set[str] = {
        p for p, c in resolved.items() if c == _CLASSIFICATION_BUILDING_BLOCK
    }

    object_roots = sorted(
        [p for p, c in resolved.items() if c == _CLASSIFICATION_OBJECT_ROOT],
        key=lambda p: (p.count("/"), p),
    )

    claimed: set[str] = set()
    for path in object_roots:
        inside_blocked = any(path.startswith(b + "/") for b in blocked)
        if inside_blocked:
            resolved[path] = _CLASSIFICATION_COMPONENT
            continue
        ancestor_claimed = any(path.startswith(c + "/") for c in claimed)
        if ancestor_claimed:
            resolved[path] = _CLASSIFICATION_COMPONENT
            continue
        claimed.add(path)

    return resolved


# ------------------------------------------------------------------
# Main detection function
# ------------------------------------------------------------------


def detect_objects(
    stage: Usd.Stage,
    composition_data: dict[str, Any],
    geometry_stats: dict[str, Any],
    skip_geometry: bool = False,
    building_block_min_reuse: int = 20,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Detect objects in a USD scene using a feature-scoring algorithm.

    Implements a 6-phase detection strategy:
      0. Pre-computation -- build lookup caches
      1. Candidate selection -- walk hierarchy for viable prims
      2. Feature extraction -- compute feature vector per candidate
      3. Classification -- apply priority-ordered rules
      4. Non-overlap resolution -- resolve ancestor/descendant conflicts
      5. Instance grouping & source classification

    Args:
        stage: The USD stage to analyze.
        composition_data: Output of
            :func:`~world_understanding.utils.usd.composition.collect_composition_arcs`.
        geometry_stats: Output of
            :func:`~world_understanding.utils.usd.prim.collect_mesh_geometry_stats`.
        skip_geometry: If True, skip vertex/face counting in subtree stats.
        building_block_min_reuse: Minimum reuse count for building block
            classification.  The actual threshold is
            ``max(building_block_min_reuse, median_ref_count * 10)``.

    Returns:
        Tuple of ``(objects, instance_groups)`` where each object is a dict
        with keys: id, name, path, parent_group, source_classification,
        source_files, mesh_count, vertex_count, face_count,
        prim_type_breakdown, bounding_box, instance_group,
        llm_classification, llm_description.
    """
    from pxr import Pcp, Usd

    from world_understanding.utils.usd.prim import (
        get_bbox_from_prim,
        get_subtree_geometry_stats,
    )

    # ==================================================================
    # Setup
    # ==================================================================
    default_prim = stage.GetDefaultPrim()
    scene_root_prim = (
        default_prim
        if (default_prim and default_prim.IsValid())
        else stage.GetPseudoRoot()
    )
    scene_root_prim = _find_content_root(scene_root_prim)
    scene_root_path = str(scene_root_prim.GetPath())
    logger.info(f"Scene root detected: {scene_root_path}")

    # ==================================================================
    # Phase 0: Pre-computation
    # ==================================================================

    # prim_path -> [asset_paths]
    prim_refs: dict[str, list[str]] = {}
    for sub_usd in composition_data.get("sub_usd_files", []):
        for ref_prim in sub_usd.get("referencing_prims", []):
            prim_refs.setdefault(ref_prim, []).append(sub_usd["asset_path"])

    # asset_path -> global reference count
    asset_reuse_count: dict[str, int] = {}
    for sub_usd in composition_data.get("sub_usd_files", []):
        asset_reuse_count[sub_usd["asset_path"]] = sub_usd.get(
            "reference_count", len(sub_usd.get("referencing_prims", []))
        )

    # Building block threshold
    ref_counts = sorted(asset_reuse_count.values()) if asset_reuse_count else [0]
    median_ref = statistics.median(ref_counts) if ref_counts else 0
    bb_threshold = max(building_block_min_reuse, int(median_ref * 10))
    logger.info(
        f"Building block threshold: {bb_threshold} "
        f"(median_ref={median_ref}, min_reuse={building_block_min_reuse})"
    )

    # Mesh ancestry cache
    paths_with_meshes = _build_mesh_ancestry_cache(scene_root_prim)

    # Subtree refs cache
    subtree_refs_cache = _build_subtree_refs_cache(prim_refs, scene_root_path)

    # ==================================================================
    # Phase 1: Candidate Selection
    # ==================================================================
    candidates: dict[str, CandidateFeatures] = {}
    prefix = scene_root_path + "/"

    for prim in Usd.PrimRange(scene_root_prim, Usd.TraverseInstanceProxies()):
        prim_path = str(prim.GetPath())
        # Skip the content root itself
        if prim_path == scene_root_path:
            continue
        # Must be under content root
        if not prim_path.startswith(prefix):
            continue
        # Skip instance proxy descendants — instance roots (IsInstance)
        # stay as candidates but their proxy children don't expand into
        # separate candidates.  Prototypes are handled via Signal 0.
        if prim.IsInstanceProxy():
            continue
        # Must have at least one child (not a leaf)
        children = prim.GetFilteredChildren(Usd.TraverseInstanceProxies())
        child_list = list(children)
        if not child_list:
            continue
        # Skip Material/Shader types
        type_name = str(prim.GetTypeName())
        if type_name in _SKIP_TYPES:
            continue
        # Must have mesh descendants OR sub-USD references in subtree
        has_meshes = prim_path in paths_with_meshes
        has_subtree_refs = bool(subtree_refs_cache.get(prim_path))
        if not has_meshes and not has_subtree_refs:
            continue

        candidates[prim_path] = CandidateFeatures(path=prim_path)

    logger.info(f"Phase 1: {len(candidates)} candidates selected")

    # ==================================================================
    # Phase 2: Feature Extraction
    # ==================================================================
    # Pre-compute sibling homogeneity per parent
    sibling_homo_cache: dict[str, float] = {}
    parents_computed: set[str] = set()

    for path in candidates:
        prim = stage.GetPrimAtPath(path)
        parent = prim.GetParent()
        if not parent or not parent.IsValid():
            continue
        parent_path = str(parent.GetPath())
        if parent_path not in parents_computed:
            parents_computed.add(parent_path)
            homo_map = _compute_sibling_homogeneity_map(parent, prim_refs)
            sibling_homo_cache.update(homo_map)

    for path, feat in candidates.items():
        prim = stage.GetPrimAtPath(path)

        # Relative depth
        rel_part = path[len(scene_root_path) + 1 :]
        feat.rel_depth = rel_part.count("/") + 1

        # Subtree refs
        feat.subtree_refs = subtree_refs_cache.get(path, set())
        feat.subtree_ref_diversity = len(feat.subtree_refs)

        # Max subtree reuse
        if feat.subtree_refs:
            feat.max_subtree_reuse = max(
                asset_reuse_count.get(a, 0) for a in feat.subtree_refs
            )

        # Direct ref reuse
        direct_refs = prim_refs.get(path, [])
        if direct_refs:
            feat.direct_ref_reuse = max(
                asset_reuse_count.get(a, 0) for a in direct_refs
            )

        # Sibling homogeneity
        feat.sibling_homogeneity = sibling_homo_cache.get(path, 0.0)

        # Child count and type diversity
        child_list = list(prim.GetFilteredChildren(Usd.TraverseInstanceProxies()))
        feat.child_count = len(child_list)
        child_types = {str(c.GetTypeName()) for c in child_list}
        feat.child_type_diversity = len(child_types)

        # SkelRoot among direct children
        feat.has_skel_root = any(str(c.GetTypeName()) == "SkelRoot" for c in child_list)

        # Mesh descendants
        feat.has_mesh_descendants = path in paths_with_meshes

    logger.info("Phase 2: features extracted")

    # ==================================================================
    # Phase 3: Classification
    # ==================================================================
    for _path, feat in candidates.items():
        feat.classification = _classify_candidate(feat, bb_threshold)

    classifications = {p: f.classification for p, f in candidates.items()}

    obj_count = sum(
        1 for c in classifications.values() if c == _CLASSIFICATION_OBJECT_ROOT
    )
    cat_count = sum(
        1 for c in classifications.values() if c == _CLASSIFICATION_CATEGORY
    )
    bb_count = sum(
        1 for c in classifications.values() if c == _CLASSIFICATION_BUILDING_BLOCK
    )
    logger.info(
        f"Phase 3: {obj_count} object_roots, {cat_count} categories, "
        f"{bb_count} building_blocks (pre-overlap)"
    )

    # ==================================================================
    # Phase 4: Non-overlap Resolution
    # ==================================================================
    resolved = _resolve_overlaps(classifications)

    # Leaf-category promotion (post-overlap): if a category has no
    # remaining descendant object_roots, promote it to object_root so it
    # appears as an object for material assignment.
    for path in list(resolved):
        if resolved[path] != _CLASSIFICATION_CATEGORY:
            continue
        cat_prefix = path + "/"
        has_descendant_obj = any(
            c == _CLASSIFICATION_OBJECT_ROOT
            for p, c in resolved.items()
            if p.startswith(cat_prefix)
        )
        if not has_descendant_obj:
            resolved[path] = _CLASSIFICATION_OBJECT_ROOT

    obj_count = sum(1 for c in resolved.values() if c == _CLASSIFICATION_OBJECT_ROOT)
    cat_count = sum(1 for c in resolved.values() if c == _CLASSIFICATION_CATEGORY)
    logger.info(
        f"Phase 4: {obj_count} object_roots, {cat_count} categories (post-overlap)"
    )

    # ==================================================================
    # Phase 5 (part 1): Object Assembly
    # ==================================================================
    # Only compute expensive geometry stats for final object_roots
    final_roots = sorted(
        [p for p, c in resolved.items() if c == _CLASSIFICATION_OBJECT_ROOT],
        key=lambda p: (p.count("/"), p),
    )

    obj_counter = 0

    def _next_id() -> str:
        nonlocal obj_counter
        obj_counter += 1
        return f"obj_{obj_counter:03d}"

    def _get_parent_group(path: str) -> str | None:
        if scene_root_path and path.startswith(prefix):
            rel = path[len(prefix) :]
        else:
            rel = path
        parts = [p for p in rel.split("/") if p]
        if len(parts) >= 2:
            return parts[0]
        return None

    def _compute_bbox_dict(path: str) -> dict[str, Any] | None:
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            return None
        try:
            bbox = get_bbox_from_prim(prim)
            rng = bbox.ComputeAlignedRange()
            mn = rng.GetMin()
            mx = rng.GetMax()
            return {"min": [mn[0], mn[1], mn[2]], "max": [mx[0], mx[1], mx[2]]}
        except Exception:
            return None

    def _collect_subtree_refs_sorted(root_path: str) -> list[str]:
        return sorted(subtree_refs_cache.get(root_path, set()))

    objects: list[dict[str, Any]] = []
    for path in final_roots:
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            continue
        stats = get_subtree_geometry_stats(stage, path, skip_geometry=skip_geometry)
        source_files = _collect_subtree_refs_sorted(path)
        bbox = _compute_bbox_dict(path)
        parent_group = _get_parent_group(path)

        obj: dict[str, Any] = {
            "id": _next_id(),
            "name": prim.GetName(),
            "path": path,
            "parent_group": parent_group,
            "source_classification": None,
            "source_files": source_files,
            "mesh_count": stats["mesh_count"],
            "vertex_count": stats["vertex_count"],
            "face_count": stats["face_count"],
            "prim_type_breakdown": stats["prim_type_breakdown"],
            "bounding_box": bbox,
            "instance_group": None,
            "llm_classification": None,
            "llm_description": None,
        }
        objects.append(obj)

    logger.info(f"Phase 5: {len(objects)} objects assembled")

    # ==================================================================
    # Phase 5 (part 2): Instance Grouping & Source Classification
    # ==================================================================
    instance_groups: list[dict[str, Any]] = []
    assigned: set[str] = set()  # object paths already in a group

    # Signal 0: USD native prototype grouping (strongest signal)
    # Instances sharing the same prototype have identical geometry and
    # need the same materials — group them before any other signal.
    proto_groups: dict[str, list[dict[str, Any]]] = {}
    for obj in objects:
        prim = stage.GetPrimAtPath(obj["path"])
        if prim and prim.IsValid() and prim.IsInstance():
            proto = prim.GetPrototype()
            if proto and proto.IsValid():
                proto_path = str(proto.GetPath())
                proto_groups.setdefault(proto_path, []).append(obj)

    for proto_path, members in proto_groups.items():
        if len(members) < 2:
            continue
        group_name = members[0]["name"]
        ig: dict[str, Any] = {
            "group_name": group_name,
            "source_file": proto_path,
            "instance_count": len(members),
            "member_paths": [m["path"] for m in members],
        }
        instance_groups.append(ig)
        for m in members:
            m["instance_group"] = group_name
            assigned.add(m["path"])

    # Signal 1: Same direct sub-USD (strongest)
    source_groups: dict[str, list[dict[str, Any]]] = {}
    for obj in objects:
        if len(obj["source_files"]) == 1:
            sf = obj["source_files"][0]
            source_groups.setdefault(sf, []).append(obj)
    for sf, members in source_groups.items():
        if len(members) < 2:
            continue
        group_name = Path(sf).stem
        ig = {
            "group_name": group_name,
            "source_file": sf,
            "instance_count": len(members),
            "member_paths": [m["path"] for m in members],
        }
        instance_groups.append(ig)
        for m in members:
            m["instance_group"] = group_name
            assigned.add(m["path"])

    # Signal 1b: Reference-source duplicate detection via PcpPrimIndex
    # Catches prims that reference the same USD file but whose source_files
    # lists diverged (e.g. collected at different layer levels).
    def _get_reference_source(prim: Usd.Prim) -> str | None:
        """Extract the primary reference source file from a prim's PcpPrimIndex."""
        try:
            prim_index = prim.GetPrimIndex()
            root_node = prim_index.rootNode
            for child in root_node.children:
                if child.arcType == Pcp.ArcTypeReference:
                    layer = (
                        child.layerStack.layers[0] if child.layerStack.layers else None
                    )
                    if layer:
                        identifier = layer.identifier
                        if identifier and not identifier.startswith("anon:"):
                            return identifier
        except Exception:
            pass
        return None

    ref_source_groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for obj in objects:
        if obj["path"] in assigned:
            continue
        prim = stage.GetPrimAtPath(obj["path"])
        if not prim or not prim.IsValid():
            continue
        ref_src = _get_reference_source(prim)
        if ref_src:
            key = (ref_src, obj["mesh_count"])
            ref_source_groups.setdefault(key, []).append(obj)

    for (ref_src, _mc), members in ref_source_groups.items():
        if len(members) < 2:
            continue
        group_name = Path(ref_src).stem
        ig = {
            "group_name": group_name,
            "source_file": ref_src,
            "instance_count": len(members),
            "member_paths": [m["path"] for m in members],
        }
        instance_groups.append(ig)
        for m in members:
            m["instance_group"] = group_name
            assigned.add(m["path"])

    # Signal 1b fallback: same name + exact topology match.
    # Catches objects referencing different source files that contain
    # identical geometry (e.g. fixture.usd vs fixture_loaded.usd).
    name_topo_groups: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
    for obj in objects:
        if obj["path"] in assigned:
            continue
        if obj["mesh_count"] > 0:
            key = (obj["name"], obj["mesh_count"], obj["vertex_count"])
            name_topo_groups.setdefault(key, []).append(obj)

    for (name, _mc, _vc), members in name_topo_groups.items():
        if len(members) < 2:
            continue
        source = members[0]["source_files"]
        ig = {
            "group_name": name,
            "source_file": source[0]
            if len(source) == 1
            else (source if source else None),
            "instance_count": len(members),
            "member_paths": [m["path"] for m in members],
        }
        instance_groups.append(ig)
        for m in members:
            m["instance_group"] = name
            assigned.add(m["path"])

    # Signal 2: Name pattern
    instance_pattern = re.compile(r"^(.+?)(?:__I?\d+|_?\d+)$")
    name_groups: dict[str, list[dict[str, Any]]] = {}
    for obj in objects:
        if obj["path"] in assigned:
            continue
        m = instance_pattern.match(obj["name"])
        if m:
            base = m.group(1)
            name_groups.setdefault(base, []).append(obj)

    for base_name, members in name_groups.items():
        if len(members) < 2:
            continue
        # Sub-group by mesh_count to avoid grouping prims with different topology
        mc_subgroups: dict[int, list[dict[str, Any]]] = {}
        for m in members:
            mc_subgroups.setdefault(m["mesh_count"], []).append(m)
        for mc, sub_members in mc_subgroups.items():
            if len(sub_members) < 2:
                continue
            group_name = base_name if len(mc_subgroups) == 1 else f"{base_name}_{mc}m"
            source = sub_members[0]["source_files"]
            ig = {
                "group_name": group_name,
                "source_file": source[0]
                if len(source) == 1
                else (source if source else None),
                "instance_count": len(sub_members),
                "member_paths": [m["path"] for m in sub_members],
            }
            instance_groups.append(ig)
            for m in sub_members:
                m["instance_group"] = group_name
                assigned.add(m["path"])

    # Signal 3: Subtree reference fingerprint
    fingerprint_groups: dict[frozenset[str], list[dict[str, Any]]] = {}
    for obj in objects:
        if obj["path"] in assigned:
            continue
        sf = frozenset(obj["source_files"])
        if sf:  # skip objects with no source files
            fingerprint_groups.setdefault(sf, []).append(obj)

    for sf_set, members in fingerprint_groups.items():
        if len(members) < 2:
            continue
        # Sub-group by mesh_count to avoid grouping prims with different topology
        mc_subgroups_fp: dict[int, list[dict[str, Any]]] = {}
        for m in members:
            mc_subgroups_fp.setdefault(m["mesh_count"], []).append(m)
        for mc, sub_members in mc_subgroups_fp.items():
            if len(sub_members) < 2:
                continue
            # Use shortest common stem as group name
            stems = [Path(f).stem for f in sorted(sf_set)]
            group_name = stems[0] if stems else "unnamed_group"
            if len(mc_subgroups_fp) > 1:
                group_name = f"{group_name}_{mc}m"
            ig = {
                "group_name": group_name,
                "source_file": sorted(sf_set)[0]
                if len(sf_set) == 1
                else sorted(sf_set),
                "instance_count": len(sub_members),
                "member_paths": [m["path"] for m in sub_members],
            }
            instance_groups.append(ig)
            for m in sub_members:
                m["instance_group"] = group_name
                assigned.add(m["path"])

    # Source classification
    for obj in objects:
        n_source_files = len(obj["source_files"])
        if n_source_files == 0:
            obj["source_classification"] = "INLINE"
        elif n_source_files == 1:
            obj["source_classification"] = "FILE"
        else:
            obj["source_classification"] = "MIXED"

    return objects, instance_groups
