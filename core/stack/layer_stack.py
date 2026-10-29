"""Layer stack representation and loader.

Parses and validates PDK layer stack YAML definitions.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import yaml


@dataclass(frozen=True)
class Layer:
    """Represents a fabrication process layer definition.

    Attributes:
        name: Unique layer identifier.
        gds: GDSII (layer, datatype) tuple.
        z: Base vertical position (z-offset).
        thickness: Vertical thickness (must be positive).
        material: Material identifier for the layer.
        approx: Whether z or thickness is an approximate value.
    """

    name: str
    gds: Tuple[int, int]
    z: float
    thickness: float
    material: str
    approx: bool = False


REQUIRED_FIELDS = ("name", "gds", "z", "thickness", "material")


def load_stack(path: Union[str, Path]) -> List[Layer]:
    """Load and validate a layer stack definition from a YAML file.

    Args:
        path: Path to the YAML file.

    Returns:
        List of validated Layer dataclass instances.

    Raises:
        FileNotFoundError: If the stack YAML file does not exist.
        ValueError: If required fields are missing, thickness <= 0, or duplicate
                    names / GDS pairs are present.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Layer stack file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict) or "layers" not in data:
        raise ValueError("Invalid layer stack YAML: missing 'layers' root key")

    layers_raw = data["layers"]
    if not isinstance(layers_raw, list):
        raise ValueError("Invalid layer stack YAML: 'layers' must be a list")

    layers: List[Layer] = []
    seen_names = set()
    seen_gds = set()

    for idx, item in enumerate(layers_raw):
        if not isinstance(item, dict):
            raise ValueError(f"Layer entry at index {idx} must be a dictionary")

        for field in REQUIRED_FIELDS:
            if field not in item:
                raise ValueError(
                    f"Layer entry at index {idx} missing required field: '{field}'"
                )

        name = str(item["name"]).strip()
        if not name:
            raise ValueError(f"Layer entry at index {idx} has empty name")

        if name in seen_names:
            raise ValueError(f"Duplicate layer name found: '{name}'")
        seen_names.add(name)

        gds_val = item["gds"]
        if (
            not isinstance(gds_val, (list, tuple))
            or len(gds_val) != 2
            or not all(isinstance(x, int) for x in gds_val)
        ):
            raise ValueError(
                f"Layer '{name}' has invalid GDS specifier: {gds_val} (expected [layer, datatype])"
            )

        gds_tuple = (int(gds_val[0]), int(gds_val[1]))
        if gds_tuple in seen_gds:
            raise ValueError(
                f"Duplicate GDS (layer, datatype) pair {gds_tuple} for layer '{name}'"
            )
        seen_gds.add(gds_tuple)

        try:
            z = float(item["z"])
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Layer '{name}' has invalid 'z' coordinate: {item['z']}") from exc

        try:
            thickness = float(item["thickness"])
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"Layer '{name}' has invalid 'thickness': {item['thickness']}"
            ) from exc

        if thickness <= 0:
            raise ValueError(
                f"Layer '{name}' thickness must be positive (> 0), got {thickness}"
            )

        material = str(item["material"]).strip()
        if not material:
            raise ValueError(f"Layer '{name}' has empty material")

        approx = bool(item.get("approx", False))

        layers.append(
            Layer(
                name=name,
                gds=gds_tuple,
                z=z,
                thickness=thickness,
                material=material,
                approx=approx,
            )
        )

    return layers
