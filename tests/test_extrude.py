"""Unit tests for 2D to 3D geometry extrusion."""

import logging
import random
import gdstk
import numpy as np
import pytest
from shapely.geometry import MultiPoint, Polygon, box

from core.stack.extrude import extrude, triangulate_polygon
from core.stack.layer_stack import Layer


def test_extrude_rectangle_volume_and_bounds() -> None:
    """Test: a 10x20 rectangle with thickness 0.5 must give volume 100 and bounds z in [z, z+thickness]."""
    rect = box(0, 0, 10, 20)
    z_base = 2.0
    thickness = 0.5
    layer = Layer(
        name="poly",
        gds=(66, 20),
        z=z_base,
        thickness=thickness,
        material="polysilicon",
        approx=True,
    )

    polygons_by_layer = {(66, 20): [rect]}
    meshes = extrude(polygons_by_layer, [layer])

    assert "poly" in meshes
    mesh = meshes["poly"]

    # Known volume: 10 * 20 * 0.5 = 100.0
    assert pytest.approx(mesh.volume, rel=1e-3) == 100.0

    # Bounds z in [z, z + thickness] -> [2.0, 2.5]
    z_min, z_max = mesh.bounds[:, 2]
    assert pytest.approx(z_min, rel=1e-3) == z_base
    assert pytest.approx(z_max, rel=1e-3) == z_base + thickness


def test_extrude_square_with_hole() -> None:
    """Test that a 10x10 square with a 2x2 hole has volume (100 - 4) * thickness."""
    outer = [(0, 0), (10, 0), (10, 10), (0, 10)]
    hole = [(4, 4), (6, 4), (6, 6), (4, 6)]
    polygon_with_hole = Polygon(shell=outer, holes=[hole])

    thickness = 2.0
    layer = Layer(
        name="met1",
        gds=(68, 20),
        z=0.0,
        thickness=thickness,
        material="metal1",
    )

    polygons_by_layer = {(68, 20): [polygon_with_hole]}
    meshes = extrude(polygons_by_layer, [layer])

    assert "met1" in meshes
    mesh = meshes["met1"]

    expected_volume = (100.0 - 4.0) * thickness  # 192.0
    assert pytest.approx(mesh.volume, rel=1e-3) == expected_volume
    assert mesh.is_watertight
    assert mesh.volume > 0


def test_extrude_concave_u_shape_with_hole() -> None:
    """Test concave U-shape with a hole: mesh volume == polygon.area * thickness."""
    outer = [
        (0.0, 0.0),
        (10.0, 0.0),
        (10.0, 10.0),
        (8.0, 10.0),
        (8.0, 3.0),
        (2.0, 3.0),
        (2.0, 10.0),
        (0.0, 10.0),
    ]
    hole = [(4.0, 1.0), (6.0, 1.0), (6.0, 2.0), (4.0, 2.0)]
    u_poly = Polygon(shell=outer, holes=[hole])

    thickness = 3.0
    layer = Layer(
        name="nwell",
        gds=(64, 20),
        z=0.0,
        thickness=thickness,
        material="silicon",
    )

    meshes = extrude({"nwell": [u_poly]}, [layer])
    assert "nwell" in meshes
    mesh = meshes["nwell"]

    expected_volume = u_poly.area * thickness
    assert pytest.approx(mesh.volume, rel=1e-5) == expected_volume
    assert mesh.is_watertight
    assert mesh.volume > 0


def test_extrude_polygon_with_two_holes() -> None:
    """Test polygon with TWO holes: volume == (area - hole areas) * thickness."""
    outer = [(0.0, 0.0), (20.0, 0.0), (20.0, 20.0), (0.0, 20.0)]
    h1 = [(2.0, 5.0), (6.0, 5.0), (6.0, 9.0), (2.0, 9.0)]
    h2 = [(10.0, 5.0), (14.0, 5.0), (14.0, 9.0), (10.0, 9.0)]
    poly_two_holes = Polygon(shell=outer, holes=[h1, h2])

    thickness = 2.5
    layer = Layer(
        name="met1",
        gds=(68, 20),
        z=0.5,
        thickness=thickness,
        material="metal1",
    )

    meshes = extrude({(68, 20): [poly_two_holes]}, [layer])
    assert "met1" in meshes
    mesh = meshes["met1"]

    expected_volume = poly_two_holes.area * thickness
    assert pytest.approx(mesh.volume, rel=1e-5) == expected_volume
    assert mesh.is_watertight
    assert mesh.volume > 0


def test_extrude_watertight_and_positive_volume() -> None:
    """Verify that every returned mesh is watertight and has positive volume."""
    shapes = [
        Polygon([(0, 0), (5, 0), (5, 5), (0, 5)]),
        Polygon([(0, 0), (10, 0), (5, 8)]),
        Polygon([(0, 0), (8, 0), (8, 8), (0, 8)], holes=[[(2, 2), (4, 2), (4, 4), (2, 4)]]),
    ]
    layer = Layer(
        name="poly",
        gds=(66, 20),
        z=0.0,
        thickness=1.5,
        material="polysilicon",
    )

    meshes = extrude({"poly": shapes}, [layer])
    mesh = meshes["poly"]
    assert mesh.is_watertight
    assert mesh.volume > 0


def test_extrude_degenerate_polygon_skipped() -> None:
    """Test that degenerate zero-area polygons are gracefully skipped."""
    collinear_line = Polygon([(0, 0), (5, 0), (10, 0), (0, 0)])
    zero_area_point = Polygon([(1, 1), (1, 1), (1, 1)])
    valid_square = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])

    layer_poly = Layer(name="poly", gds=(66, 20), z=0.0, thickness=1.0, material="poly")
    layer_met = Layer(name="met1", gds=(68, 20), z=1.0, thickness=1.0, material="metal1")

    # Layer with only degenerate shapes -> skipped entirely
    meshes = extrude(
        {
            (66, 20): [collinear_line, zero_area_point],
            (68, 20): [collinear_line, valid_square],
        },
        [layer_poly, layer_met],
    )

    assert "poly" not in meshes
    assert "met1" in meshes
    assert pytest.approx(meshes["met1"].volume, rel=1e-3) == 4.0


def test_extrude_property_random_polygons() -> None:
    """Property test: 50 random simple polygons with random inner holes."""
    random.seed(42)
    np.random.seed(42)

    tested_count = 0
    iteration = 0

    while tested_count < 50 and iteration < 200:
        iteration += 1
        num_pts = random.randint(5, 16)
        pts = np.random.uniform(0.0, 100.0, (num_pts, 2))
        hull = MultiPoint(pts).convex_hull

        if not isinstance(hull, Polygon) or hull.area < 25.0:
            continue

        minx, miny, maxx, maxy = hull.bounds
        cx, cy = (minx + maxx) / 2.0, (miny + maxy) / 2.0
        w = min(maxx - minx, maxy - miny) * random.uniform(0.15, 0.3)
        candidate_hole = box(cx - w / 2.0, cy - w / 2.0, cx + w / 2.0, cy + w / 2.0)

        # Include hole when it fits completely inside the polygon
        if hull.contains(candidate_hole):
            test_poly = Polygon(hull.exterior.coords, [candidate_hole.exterior.coords])
        else:
            test_poly = hull

        thickness = random.uniform(0.5, 5.0)
        z_base = random.uniform(0.0, 10.0)

        layer = Layer(
            name=f"rand_{tested_count}",
            gds=(100 + tested_count, 0),
            z=z_base,
            thickness=thickness,
            material="test_mat",
        )

        meshes = extrude({layer.gds: [test_poly]}, [layer])
        mesh = meshes[layer.name]

        expected_volume = test_poly.area * thickness
        rel_diff = abs(mesh.volume - expected_volume) / expected_volume
        assert (
            rel_diff < 1e-6
        ), f"Volume mismatch at iteration {iteration}: {mesh.volume} vs {expected_volume}"
        assert mesh.is_watertight, f"Mesh at iteration {iteration} not watertight"
        assert mesh.volume > 0

        tested_count += 1

    assert tested_count == 50


def test_extrude_skip_absent_layers_and_logs_warning(caplog: pytest.LogCaptureFixture) -> None:
    """Test that layers absent from the stack are skipped and a warning is logged."""
    square = Polygon([(0, 0), (5, 5), (5, 0)])
    layer = Layer(
        name="poly",
        gds=(66, 20),
        z=0.0,
        thickness=1.0,
        material="polysilicon",
    )

    polygons_by_layer = {
        (66, 20): [square],
        (99, 99): [square],  # Absent layer
        "unknown_layer": [square],
    }

    with caplog.at_level(logging.WARNING):
        meshes = extrude(polygons_by_layer, [layer])

    assert "poly" in meshes
    assert len(meshes) == 1
    assert "Skipping layer (99, 99)" in caplog.text
    assert "Skipping layer unknown_layer" in caplog.text


def test_extrude_by_layer_name() -> None:
    """Test that polygons keyed by layer name string are supported."""
    square = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
    layer = Layer(
        name="nwell",
        gds=(64, 20),
        z=0.5,
        thickness=1.0,
        material="silicon",
    )

    polygons_by_layer = {"nwell": [square]}
    meshes = extrude(polygons_by_layer, [layer])

    assert "nwell" in meshes
    assert pytest.approx(meshes["nwell"].volume, rel=1e-3) == 4.0


def test_extrude_gdstk_polygons() -> None:
    """Test that gdstk.Polygon objects are accepted and properly extruded."""
    gds_poly = gdstk.rectangle((0, 0), (10, 10), layer=66, datatype=20)
    layer = Layer(
        name="poly",
        gds=(66, 20),
        z=0.0,
        thickness=2.0,
        material="polysilicon",
    )

    polygons_by_layer = {(66, 20): [gds_poly]}
    meshes = extrude(polygons_by_layer, [layer])

    assert "poly" in meshes
    assert pytest.approx(meshes["poly"].volume, rel=1e-3) == 200.0


def test_extrude_multiple_polygons_concatenation() -> None:
    """Test that multiple polygons on the same layer are combined."""
    sq1 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    sq2 = Polygon([(20, 0), (30, 0), (30, 10), (20, 10)])

    layer = Layer(
        name="met1",
        gds=(68, 20),
        z=1.0,
        thickness=2.0,
        material="metal1",
    )

    polygons_by_layer = {(68, 20): [sq1, sq2]}
    meshes = extrude(polygons_by_layer, [layer])

    assert "met1" in meshes
    assert pytest.approx(meshes["met1"].volume, rel=1e-3) == 400.0
