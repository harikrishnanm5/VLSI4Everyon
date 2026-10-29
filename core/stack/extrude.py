"""Geometry extrusion module.

Extrudes 2D layout polygons into 3D meshes using layer stack parameters.
"""

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import shapely.geometry
from shapely.geometry import MultiPolygon, Polygon
import trimesh

from core.stack.layer_stack import Layer

logger = logging.getLogger(__name__)


def triangulate_polygon(
    polygon: Polygon, layer_name: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Triangulate a 2D shapely Polygon using mapbox-earcut.

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

    try:
        vertices, faces = trimesh.creation.triangulate_polygon(
            polygon, engine="earcut"
        )
        return np.asarray(vertices, dtype=float), np.asarray(faces, dtype=int)
    except Exception as exc:
        layer_info = f" on layer '{layer_name}'" if layer_name else ""
        raise ValueError(f"Triangulation failed{layer_info}: {exc}") from exc


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
    """Extrude a 2D shapely Polygon into a 3D Trimesh using trimesh.creation.extrude_polygon."""
    if poly.is_empty or poly.area <= 1e-12:
        return None

    if not poly.is_valid:
        poly = poly.buffer(0)
        if poly.is_empty or poly.area <= 1e-12:
            return None

    try:
        mesh = trimesh.creation.extrude_polygon(
            poly, height=thickness, engine="earcut"
        )
    except Exception as exc:
        logger.debug(
            "Failed to extrude polygon on layer %s: %s",
            layer_name or "unknown",
            exc,
        )
        return None

    if mesh is None or len(mesh.faces) == 0:
        return None

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
