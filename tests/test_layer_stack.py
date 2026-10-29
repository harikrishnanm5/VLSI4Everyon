"""Unit tests for layer stack loader and schema validation."""

from pathlib import Path
import pytest
import yaml

from core.stack.layer_stack import Layer, load_stack


def yaml_dump_inline(d: dict) -> str:
    """Helper to format a dictionary as YAML lines."""
    return yaml.dump(d).replace("\n", "\n    ").strip()


def test_load_stack_valid(tmp_path: Path) -> None:
    """Test loading a valid stack YAML file."""
    yaml_content = """
layers:
  - name: poly
    gds: [66, 20]
    z: 0.1
    thickness: 0.2
    material: polysilicon
    approx: true
  - name: met1
    gds: [68, 20]
    z: 0.8
    thickness: 0.35
    material: metal1
    approx: false
"""
    yaml_file = tmp_path / "valid_stack.yaml"
    yaml_file.write_text(yaml_content, encoding="utf-8")

    layers = load_stack(yaml_file)

    assert len(layers) == 2
    assert layers[0] == Layer(
        name="poly",
        gds=(66, 20),
        z=0.1,
        thickness=0.2,
        material="polysilicon",
        approx=True,
    )
    assert layers[1] == Layer(
        name="met1",
        gds=(68, 20),
        z=0.8,
        thickness=0.35,
        material="metal1",
        approx=False,
    )


def test_load_sky130_stack() -> None:
    """Test loading the PDK Sky130 stack definition with required 4 layers."""
    sky130_path = Path("pdks/sky130/stack.yaml")
    layers = load_stack(sky130_path)

    assert len(layers) == 4
    layer_map = {layer.name: layer for layer in layers}
    assert set(layer_map.keys()) == {"nwell", "poly", "li1", "met1"}

    assert layer_map["nwell"].gds == (64, 20)
    assert layer_map["poly"].gds == (66, 20)
    assert layer_map["li1"].gds == (67, 20)
    assert layer_map["met1"].gds == (68, 20)

    for layer in layers:
        assert layer.approx is True
        assert layer.thickness > 0
        assert layer.material


@pytest.mark.parametrize("missing_field", ["name", "gds", "z", "thickness", "material"])
def test_load_stack_missing_field(tmp_path: Path, missing_field: str) -> None:
    """Test that missing any required field raises a clear ValueError."""
    data = {
        "name": "poly",
        "gds": [66, 20],
        "z": 0.0,
        "thickness": 0.2,
        "material": "polysilicon",
    }
    del data[missing_field]

    yaml_content = f"""
layers:
  - {yaml_dump_inline(data)}
"""
    yaml_file = tmp_path / "missing_field.yaml"
    yaml_file.write_text(yaml_content, encoding="utf-8")

    with pytest.raises(ValueError, match=f"missing required field: '{missing_field}'"):
        load_stack(yaml_file)


@pytest.mark.parametrize("bad_thickness", [0.0, -0.5, -2])
def test_load_stack_bad_thickness(tmp_path: Path, bad_thickness: float) -> None:
    """Test that non-positive thickness raises ValueError."""
    yaml_content = f"""
layers:
  - name: poly
    gds: [66, 20]
    z: 0.0
    thickness: {bad_thickness}
    material: polysilicon
"""
    yaml_file = tmp_path / "bad_thickness.yaml"
    yaml_file.write_text(yaml_content, encoding="utf-8")

    with pytest.raises(ValueError, match="thickness must be positive"):
        load_stack(yaml_file)


def test_load_stack_duplicate_name(tmp_path: Path) -> None:
    """Test that duplicate layer names raise ValueError."""
    yaml_content = """
layers:
  - name: poly
    gds: [66, 20]
    z: 0.0
    thickness: 0.2
    material: polysilicon
  - name: poly
    gds: [68, 20]
    z: 0.5
    thickness: 0.3
    material: polysilicon
"""
    yaml_file = tmp_path / "dup_name.yaml"
    yaml_file.write_text(yaml_content, encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate layer name"):
        load_stack(yaml_file)


def test_load_stack_duplicate_gds(tmp_path: Path) -> None:
    """Test that duplicate GDS (layer, datatype) specifiers raise ValueError."""
    yaml_content = """
layers:
  - name: poly1
    gds: [66, 20]
    z: 0.0
    thickness: 0.2
    material: polysilicon
  - name: poly2
    gds: [66, 20]
    z: 0.5
    thickness: 0.3
    material: polysilicon
"""
    yaml_file = tmp_path / "dup_gds.yaml"
    yaml_file.write_text(yaml_content, encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate GDS"):
        load_stack(yaml_file)


def test_load_stack_invalid_gds_format(tmp_path: Path) -> None:
    """Test that malformed GDS specifiers raise ValueError."""
    yaml_content = """
layers:
  - name: poly
    gds: [66]
    z: 0.0
    thickness: 0.2
    material: polysilicon
"""
    yaml_file = tmp_path / "bad_gds.yaml"
    yaml_file.write_text(yaml_content, encoding="utf-8")

    with pytest.raises(ValueError, match="invalid GDS specifier"):
        load_stack(yaml_file)


def test_load_stack_file_not_found(tmp_path: Path) -> None:
    """Test that non-existent file raises FileNotFoundError."""
    missing = tmp_path / "non_existent.yaml"
    with pytest.raises(FileNotFoundError):
        load_stack(missing)


def test_load_stack_invalid_root(tmp_path: Path) -> None:
    """Test that YAML without 'layers' key raises ValueError."""
    yaml_file = tmp_path / "no_root.yaml"
    yaml_file.write_text("other_key: 123", encoding="utf-8")

    with pytest.raises(ValueError, match="missing 'layers' root key"):
        load_stack(yaml_file)
