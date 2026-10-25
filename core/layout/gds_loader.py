"""GDS layout loader.

Loads GDSII layout files into dictionaries mapping (layer, datatype) tuples
to lists of 2D polygons.
"""

from pathlib import Path
import sys
from typing import Dict, List, Tuple, Union

import gdstk


def load_gds(path: Union[str, Path]) -> Dict[Tuple[int, int], List[gdstk.Polygon]]:
    """Load polygons from a GDSII file grouped by (layer, datatype).

    Args:
        path: Path to the GDSII file.

    Returns:
        Dictionary mapping (layer, datatype) tuples to lists of gdstk.Polygon objects.

    Raises:
        FileNotFoundError: If the specified GDSII file does not exist.
    """
    gds_path = Path(path)
    if not gds_path.is_file():
        raise FileNotFoundError(f"GDSII file not found: {gds_path}")

    lib = gdstk.read_gds(str(gds_path))
    top_cells = lib.top_level()
    cells = top_cells if top_cells else lib.cells

    layer_dict: Dict[Tuple[int, int], List[gdstk.Polygon]] = {}
    for cell in cells:
        polygons = cell.get_polygons(apply_repetitions=True, include_paths=True, depth=-1)
        for poly in polygons:
            key = (int(poly.layer), int(poly.datatype))
            if key not in layer_dict:
                layer_dict[key] = []
            layer_dict[key].append(poly)

    return layer_dict


def main() -> None:
    """CLI entry point to inspect layer and polygon counts of a GDSII file."""
    if len(sys.argv) < 2:
        print("Usage: python -m core.layout.gds_loader <path_to_gds>")
        sys.exit(1)

    file_path = Path(sys.argv[1])
    layers = load_gds(file_path)

    for (layer, datatype), polys in sorted(layers.items()):
        print(f"Layer ({layer}, {datatype}): {len(polys)} polygon(s)")


if __name__ == "__main__":
    main()
