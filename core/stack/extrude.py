"""Geometry extrusion module.

Extrudes 2D layout polygons into 3D meshes using layer stack parameters.
"""

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import shapely.geometry
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
import trimesh

from core.stack.layer_stack import Layer

logger = logging.getLogger(__name__)


def _signed_area(pts: Sequence[Sequence[float]]) -> float:
    """Calculate signed area of a 2D closed polygon ring."""
    n = len(pts)
    if n < 3:
        return 0.0
    return 0.5 * sum(
        pts[i][0] * pts[(i + 1) % n][1] - pts[(i + 1) % n][0] * pts[i][1]
        for i in range(n)
    )


def _point_in_triangle(
    pt: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray
) -> bool:
    """Check if point pt lies inside triangle (a, b, c)."""

    def sign(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

    d1 = sign(pt, a, b)
    d2 = sign(pt, b, c)
    d3 = sign(pt, c, a)

    has_neg = (d1 < -1e-12) or (d2 < -1e-12) or (d3 < -1e-12)
    has_pos = (d1 > 1e-12) or (d2 > 1e-12) or (d3 > 1e-12)
    return not (has_neg and has_pos)


def triangulate_polygon(
    polygon: Polygon, layer_name: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Triangulate a 2D shapely Polygon, supporting interior holes.

    Uses an ear-clipping algorithm with bridge-edge insertion for holes.

    Args:
        polygon: Shapely Polygon to triangulate.
        layer_name: Optional layer name for error reporting.

    Returns:
        Tuple of (vertices, faces) as numpy arrays. Degenerate polygons with zero
        area or empty geometry return empty arrays (0 vertices, 0 faces).
    """
    if polygon.is_empty or polygon.area <= 1e-12:
        return np.zeros((0, 2), dtype=float), np.zeros((0, 3), dtype=int)

    if not polygon.is_valid:
        polygon = polygon.buffer(0)
        if polygon.is_empty or polygon.area <= 1e-12:
            return np.zeros((0, 2), dtype=float), np.zeros((0, 3), dtype=int)

    exterior_coords = list(polygon.exterior.coords)[:-1]
    if len(exterior_coords) < 3:
        return np.zeros((0, 2), dtype=float), np.zeros((0, 3), dtype=int)

    # Ensure counter-clockwise exterior orientation
    if _signed_area(exterior_coords) < 0:
        exterior_coords.reverse()

    ring = list(exterior_coords)

    # Sort interior holes by maximum x-coordinate descending to bridge from right to left
    sorted_interiors = sorted(
        polygon.interiors,
        key=lambda ring_geom: max(c[0] for c in ring_geom.coords),
        reverse=True,
    )

    # Bridge each interior hole into the outer ring
    for interior in sorted_interiors:
        hole_coords = list(interior.coords)[:-1]
        if len(hole_coords) < 3:
            continue
        # Ensure clockwise hole orientation
        if _signed_area(hole_coords) > 0:
            hole_coords.reverse()

        # Find hole vertex with maximum x coordinate
        max_hole_idx = max(range(len(hole_coords)), key=lambda i: hole_coords[i][0])
        hx, hy = hole_coords[max_hole_idx]

        best_ring_idx: Optional[int] = None
        min_dist = float("inf")
        n_r = len(ring)
        for r_idx, (rx, ry) in enumerate(ring):
            seg = LineString([(hx, hy), (rx, ry)])
            if seg.crosses(polygon.boundary):
                continue
            mid_pt = Point(((hx + rx) / 2.0, (hy + ry) / 2.0))
            if not polygon.contains(mid_pt):
                continue

            # Ensure candidate bridge does not cross any existing ring edge
            crosses_ring = False
            for i in range(n_r):
                if i == r_idx or (i + 1) % n_r == r_idx:
                    continue
                r_edge = LineString([ring[i], ring[(i + 1) % n_r]])
                if seg.crosses(r_edge):
                    crosses_ring = True
                    break
            if crosses_ring:
                continue

            dist = (hx - rx) ** 2 + (hy - ry) ** 2
            if dist < min_dist:
                min_dist = dist
                best_ring_idx = r_idx

        if best_ring_idx is None:
            layer_info = f" on layer '{layer_name}'" if layer_name else ""
            raise ValueError(
                f"Triangulation failed: unable to bridge interior hole to outer boundary{layer_info}."
            )

        ordered_hole = hole_coords[max_hole_idx:] + hole_coords[:max_hole_idx]
        ring = (
            ring[: best_ring_idx + 1]
            + ordered_hole
            + [ordered_hole[0], ring[best_ring_idx]]
            + ring[best_ring_idx + 1 :]
        )

    pts = [np.array(p, dtype=float) for p in ring]
    indices = list(range(len(pts)))

    def is_convex(p_prev: np.ndarray, p_curr: np.ndarray, p_next: np.ndarray) -> bool:
        cross = (p_curr[0] - p_prev[0]) * (p_next[1] - p_curr[1]) - (
            p_curr[1] - p_prev[1]
        ) * (p_next[0] - p_curr[0])
        return cross > 1e-12

    triangles: List[List[int]] = []
    max_iters = len(indices) * 4

    while len(indices) > 2 and max_iters > 0:
        max_iters -= 1
        ear_found = False
        n = len(indices)
        for i in range(n):
            prev_i = indices[(i - 1) % n]
            curr_i = indices[i]
            next_i = indices[(i + 1) % n]

            p_prev = pts[prev_i]
            p_curr = pts[curr_i]
            p_next = pts[next_i]

            if not is_convex(p_prev, p_curr, p_next):
                continue

            has_inside = False
            for j in range(n):
                if j in ((i - 1) % n, i, (i + 1) % n):
                    continue
                test_pt = pts[indices[j]]
                if (
                    np.allclose(test_pt, p_prev)
                    or np.allclose(test_pt, p_curr)
                    or np.allclose(test_pt, p_next)
                ):
                    continue
                if _point_in_triangle(test_pt, p_prev, p_curr, p_next):
                    has_inside = True
                    break

            if not has_inside:
                triangles.append([prev_i, curr_i, next_i])
                indices.pop(i)
                ear_found = True
                break

        if not ear_found:
            layer_info = f" on layer '{layer_name}'" if layer_name else ""
            raise ValueError(
                f"Triangulation failed: no valid ear found during polygon ear clipping{layer_info}."
            )

    v = np.array(pts, dtype=float)
    f = np.array(triangles, dtype=int)
    if len(v) == 0 or len(f) == 0:
        return np.zeros((0, 2), dtype=float), np.zeros((0, 3), dtype=int)

    # Merge duplicate 2D vertices for watertight boundary topology
    _, unique_indices, inverse = np.unique(
        np.round(v, 8), axis=0, return_index=True, return_inverse=True
    )
    v_unique = v[unique_indices]
    f_unique = inverse[f]
    return v_unique, f_unique


def _to_shapely_polygons(item: Any) -> List[Polygon]:
    """Convert an arbitrary polygon-like input into a list of shapely Polygons."""
    if isinstance(item, Polygon):
        return [item]
    if isinstance(item, MultiPolygon):
        return list(item.geoms)
    if hasattr(item, "points"):
        # e.g. gdstk.Polygon
        pts = item.points
        if len(pts) >= 3:
            return [Polygon(pts)]
    elif isinstance(item, (list, tuple, np.ndarray)):
        # Check if given as (shell, [holes])
        if (
            len(item) == 2
            and isinstance(item[1], (list, tuple))
            and len(item[1]) > 0
            and isinstance(item[1][0], (list, tuple, np.ndarray))
        ):
            return [Polygon(item[0], holes=item[1])]
        arr = np.asarray(item)
        if arr.ndim == 2 and len(arr) >= 3:
            return [Polygon(arr)]
    return []


def extrude_polygon_to_mesh(
    poly: Polygon, z_base: float, thickness: float, layer_name: Optional[str] = None
) -> Optional[trimesh.Trimesh]:
    """Extrude a 2D shapely Polygon into a 3D Trimesh."""
    v_2d, f_2d = triangulate_polygon(poly, layer_name=layer_name)
    if len(v_2d) < 3 or len(f_2d) == 0:
        return None

    mesh = trimesh.creation.extrude_triangulation(v_2d, f_2d, height=thickness)
    if z_base != 0.0:
        mesh.apply_translation([0.0, 0.0, z_base])
    return mesh


def extrude(
    polygons_by_layer: Dict[Any, Sequence[Any]],
    stack: Sequence[Layer],
) -> Dict[str, trimesh.Trimesh]:
    """Extrude 2D layout polygons into 3D meshes based on a PDK layer stack.

    Args:
        polygons_by_layer: Mapping from (layer, datatype) or layer name to
                           a sequence of polygon-like objects.
        stack: Sequence of Layer dataclass objects.

    Returns:
        Dictionary mapping layer name to trimesh.Trimesh objects.
        Degenerate polygons with zero area (area <= 1e-12 or empty) are skipped.
    """
    stack_by_gds: Dict[Tuple[int, int], Layer] = {layer.gds: layer for layer in stack}
    stack_by_name: Dict[str, Layer] = {layer.name: layer for layer in stack}

    result: Dict[str, trimesh.Trimesh] = {}

    for key, poly_items in polygons_by_layer.items():
        layer_def: Optional[Layer] = None
        if key in stack_by_gds:
            layer_def = stack_by_gds[key]
        elif isinstance(key, str) and key in stack_by_name:
            layer_def = stack_by_name[key]
        elif isinstance(key, (list, tuple)) and len(key) == 2:
            tup_key = (int(key[0]), int(key[1]))
            if tup_key in stack_by_gds:
                layer_def = stack_by_gds[tup_key]

        # Skip layers absent from the stack and log a warning
        if layer_def is None:
            logger.warning(
                "Skipping layer %s present in layout: not found in layer stack", key
            )
            continue

        if not poly_items:
            continue

        layer_meshes: List[trimesh.Trimesh] = []
        for p in poly_items:
            shapely_polys = _to_shapely_polygons(p)
            for sp in shapely_polys:
                if sp.is_empty or sp.area <= 1e-12:
                    continue
                mesh = extrude_polygon_to_mesh(
                    sp, layer_def.z, layer_def.thickness, layer_name=layer_def.name
                )
                if mesh is not None and len(mesh.faces) > 0:
                    layer_meshes.append(mesh)

        if not layer_meshes:
            continue

        if len(layer_meshes) == 1:
            combined_mesh = layer_meshes[0]
        else:
            combined_mesh = trimesh.util.concatenate(layer_meshes)

        result[layer_def.name] = combined_mesh

    return result
