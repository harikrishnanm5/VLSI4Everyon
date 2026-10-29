"""Unit tests for scene.json serialization, export, and CLI."""

import json
from pathlib import Path
import subprocess
import sys
import gdstk
import pytest
from shapely.geometry import box
import trimesh

from core.export import export_scene, main
from core.stack.extrude import extrude
from core.stack.layer_stack import Layer


def test_export_scene_rectangle_schema_and_counts(tmp_path: Path) -> None:
    """Test exporting a 10x20 rectangle layer and validating the schema and bounds."""
    rect = box(0, 0, 10, 20)
    z_base = 1.5
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

    out_file = tmp_path / "scene.json"
    export_scene(meshes, [layer], out_file)

    assert out_file.is_file()
    with open(out_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Check top-level schema keys and metadata
    assert set(data.keys()) == {"version", "units", "layers", "bounds"}
    assert data["version"] == 1
    assert data["units"] == "um"

    # Check layers array
    assert len(data["layers"]) == 1
    layer_entry = data["layers"][0]
    expected_layer_keys = {
        "name",
        "material",
        "z",
        "thickness",
        "approx",
        "vertices",
        "faces",
    }
    assert set(layer_entry.keys()) == expected_layer_keys
    assert layer_entry["name"] == "poly"
    assert layer_entry["material"] == "polysilicon"
    assert layer_entry["z"] == 1.5
    assert layer_entry["thickness"] == 0.5
    assert layer_entry["approx"] is True

    # Check vertex and face counts match trimesh
    mesh = meshes["poly"]
    assert len(layer_entry["vertices"]) == len(mesh.vertices)
    assert len(layer_entry["faces"]) == len(mesh.faces)
    assert len(layer_entry["faces"]) > 0

    # Check bounds
    assert data["bounds"]["min"] == [0.0, 0.0, 1.5]
    assert data["bounds"]["max"] == [10.0, 20.0, 2.0]


def test_export_scene_order_and_skipping(tmp_path: Path) -> None:
    """Test that layer order follows stack order and stack layers without mesh are skipped."""
    stack = [
        Layer(name="nwell", gds=(64, 20), z=0.0, thickness=1.0, material="silicon", approx=True),
        Layer(name="poly", gds=(66, 20), z=1.0, thickness=0.2, material="poly", approx=True),
        Layer(name="li1", gds=(67, 20), z=1.5, thickness=0.1, material="ti", approx=True),
        Layer(name="met1", gds=(68, 20), z=2.0, thickness=0.5, material="al", approx=False),
    ]

    # Meshes only provided for met1 and poly (passed in reversed order)
    rect1 = box(0, 0, 5, 5)
    rect2 = box(10, 10, 15, 15)
    poly_mesh = extrude({(66, 20): [rect1]}, stack)["poly"]
    met1_mesh = extrude({(68, 20): [rect2]}, stack)["met1"]

    meshes = {
        "met1": met1_mesh,
        "poly": poly_mesh,
    }

    out_file = tmp_path / "scene_order.json"
    export_scene(meshes, stack, out_file)

    with open(out_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Only poly and met1 present, in stack order (poly first, met1 second)
    exported_names = [l["name"] for l in data["layers"]]
    assert exported_names == ["poly", "met1"]
    assert data["layers"][0]["approx"] is True
    assert data["layers"][1]["approx"] is False


def test_export_scene_unknown_mesh_raises_value_error(tmp_path: Path) -> None:
    """Test that a mesh name not in the stack raises a clear ValueError."""
    stack = [
        Layer(name="poly", gds=(66, 20), z=0.0, thickness=1.0, material="poly", approx=True)
    ]
    dummy_mesh = trimesh.Trimesh(
        vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        faces=[[0, 1, 2]],
    )
    meshes = {"unknown_layer": dummy_mesh}

    out_file = tmp_path / "scene_err.json"
    with pytest.raises(ValueError, match="Mesh layer 'unknown_layer' is not defined in the layer stack"):
        export_scene(meshes, stack, out_file)


def test_export_scene_rounding_6_decimals(tmp_path: Path) -> None:
    """Test that vertex coordinates are rounded to 6 decimal places."""
    stack = [
        Layer(name="poly", gds=(66, 20), z=0.0, thickness=1.0, material="poly", approx=True)
    ]
    mesh = trimesh.Trimesh(
        vertices=[[0.123456789, 1.987654321, 2.555555555]],
        faces=[[0, 0, 0]],
    )
    out_file = tmp_path / "scene_rounded.json"
    export_scene({"poly": mesh}, stack, out_file)

    with open(out_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    v = data["layers"][0]["vertices"][0]
    assert v == [0.123457, 1.987654, 2.555556]


def test_cli_end_to_end(tmp_path: Path) -> None:
    """Test CLI python -m core.export end-to-end with a generated tiny GDS."""
    gds_path = tmp_path / "tiny.gds"
    lib = gdstk.Library(name="TINY_LIB")
    cell = lib.new_cell("TOP")
    rect1 = gdstk.rectangle((0, 0), (10, 20), layer=66, datatype=20)
    rect2 = gdstk.rectangle((5, 5), (15, 25), layer=68, datatype=20)
    cell.add(rect1, rect2)
    lib.write_gds(str(gds_path))

    stack_yaml = tmp_path / "stack.yaml"
    stack_content = """
layers:
  - name: poly
    gds: [66, 20]
    z: 0.2
    thickness: 0.18
    material: polysilicon
    approx: true
  - name: met1
    gds: [68, 20]
    z: 1.0
    thickness: 0.35
    material: metal1
    approx: true
"""
    stack_yaml.write_text(stack_content, encoding="utf-8")

    out_json = tmp_path / "exported_scene.json"

    # Invoke CLI via subprocess
    cmd = [
        sys.executable,
        "-m",
        "core.export",
        str(gds_path),
        str(stack_yaml),
        str(out_json),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert "Exported scene with 2 layer(s)" in result.stdout

    # Validate generated scene.json
    assert out_json.is_file()
    with open(out_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["version"] == 1
    assert data["units"] == "um"
    assert len(data["layers"]) == 2
    layer_names = [l["name"] for l in data["layers"]]
    assert layer_names == ["poly", "met1"]

    for l in data["layers"]:
        assert len(l["vertices"]) > 0
        assert len(l["faces"]) > 0
        assert l["approx"] is True

    # Check bounds encompass both rectangles
    bounds = data["bounds"]
    assert pytest.approx(bounds["min"][0]) == 0.0
    assert pytest.approx(bounds["min"][1]) == 0.0
    assert pytest.approx(bounds["min"][2]) == 0.2
    assert pytest.approx(bounds["max"][0]) == 15.0
    assert pytest.approx(bounds["max"][1]) == 25.0
    assert pytest.approx(bounds["max"][2]) == 1.35


def test_cli_missing_arguments(capsys: pytest.CaptureFixture[str]) -> None:
    """Test CLI usage output and exit code when arguments are missing."""
    with pytest.raises(SystemExit) as exc_info:
        main([])
    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "Usage: python -m core.export" in captured.out
