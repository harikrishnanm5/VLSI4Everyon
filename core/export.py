"""3D scene export utilities.

Serializes extruded layer meshes and PDK layer stack metadata to scene.json format,
and provides a CLI to convert GDSII + stack YAML into scene.json.
"""

import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import trimesh

from core.layout.gds_loader import load_gds
from core.stack.extrude import extrude
from core.stack.layer_stack import Layer, load_stack


def export_gltf(mesh: Any, path: str) -> None:
    """Export 3D mesh geometry to glTF format (stub)."""
    raise NotImplementedError


def export_scene(
    meshes: Dict[str, trimesh.Trimesh],
    stack: Sequence[Layer],
    path: Union[str, Path],
) -> None:
    """Export extruded 3D layer meshes and metadata to scene.json.

    Args:
        meshes: Mapping of layer name to trimesh.Trimesh from extrude().
        stack: Ordered sequence of Layer dataclass objects.
        path: Destination file path for scene.json.

    Raises:
        ValueError: If any mesh layer name is not present in the stack.
    """
    stack_by_name: Dict[str, Layer] = {layer.name: layer for layer in stack}

    # Validate that all mesh names exist in the stack
    for mesh_name in meshes.keys():
        if mesh_name not in stack_by_name:
            raise ValueError(
                f"Mesh layer '{mesh_name}' is not defined in the layer stack"
            )

    exported_layers: List[Dict[str, Any]] = []
    all_mins: List[np.ndarray] = []
    all_maxs: List[np.ndarray] = []

    # Export layers in the exact order defined by the stack, skipping empty ones
    for layer in stack:
        if layer.name not in meshes:
            continue

        mesh = meshes[layer.name]
        if mesh is None or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            continue

        vertices = [
            [round(float(c), 6) for c in vertex]
            for vertex in mesh.vertices
        ]
        faces = [
            [int(idx) for idx in face]
            for face in mesh.faces
        ]

        bounds = mesh.bounds
        all_mins.append(bounds[0])
        all_maxs.append(bounds[1])

        exported_layers.append(
            {
                "name": str(layer.name),
                "material": str(layer.material),
                "z": float(layer.z),
                "thickness": float(layer.thickness),
                "approx": bool(layer.approx),
                "vertices": vertices,
                "faces": faces,
            }
        )

    if all_mins and all_maxs:
        overall_min = np.min(all_mins, axis=0)
        overall_max = np.max(all_maxs, axis=0)
        scene_bounds = {
            "min": [round(float(c), 6) for c in overall_min],
            "max": [round(float(c), 6) for c in overall_max],
        }
    else:
        scene_bounds = {
            "min": [0.0, 0.0, 0.0],
            "max": [0.0, 0.0, 0.0],
        }

    scene_data: Dict[str, Any] = {
        "version": 1,
        "units": "um",
        "layers": exported_layers,
        "bounds": scene_bounds,
    }

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(scene_data, f, indent=2)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entry point to chain load_gds -> load_stack -> extrude -> export_scene."""
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 3:
        print("Usage: python -m core.export <gds> <stack.yaml> <out.json>")
        sys.exit(1)

    gds_path = Path(args[0])
    stack_path = Path(args[1])
    out_path = Path(args[2])

    polygons_by_layer = load_gds(gds_path)
    stack = load_stack(stack_path)
    meshes = extrude(polygons_by_layer, stack)
    export_scene(meshes, stack, out_path)
    print(f"Exported scene with {len(meshes)} layer(s) to {out_path}")


if __name__ == "__main__":
    main()
