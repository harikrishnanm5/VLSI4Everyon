"""Unit tests for GDS layout loader."""

from pathlib import Path
import pytest
import gdstk

from core.layout.gds_loader import load_gds, main


def test_load_gds_basic(tmp_path: Path) -> None:
    """Test loading a basic GDS file with polygons on multiple layers."""
    gds_path = tmp_path / "basic.gds"

    lib = gdstk.Library(name="TEST_LIB")
    top_cell = lib.new_cell("TOP")

    # Layer (1, 0): 2 rectangles
    rect1 = gdstk.rectangle((0, 0), (10, 10), layer=1, datatype=0)
    rect2 = gdstk.rectangle((20, 0), (30, 10), layer=1, datatype=0)
    top_cell.add(rect1, rect2)

    # Layer (68, 20): 1 polygon
    poly1 = gdstk.Polygon([(0, 0), (5, 0), (5, 5)], layer=68, datatype=20)
    top_cell.add(poly1)

    lib.write_gds(str(gds_path))

    result = load_gds(gds_path)

    assert (1, 0) in result
    assert len(result[(1, 0)]) == 2

    assert (68, 20) in result
    assert len(result[(68, 20)]) == 1

    # Verify polygon coordinate extraction
    pts = result[(68, 20)][0].points
    assert len(pts) == 3


def test_load_gds_hierarchy(tmp_path: Path) -> None:
    """Test that cell references are flattened and geometry transformed."""
    gds_path = tmp_path / "hierarchy.gds"

    lib = gdstk.Library(name="HIERARCHY_LIB")
    sub_cell = lib.new_cell("SUB")
    sub_cell.add(gdstk.rectangle((0, 0), (2, 2), layer=2, datatype=0))

    top_cell = lib.new_cell("TOP")
    # Reference sub_cell with translation (10, 20)
    ref = gdstk.Reference(sub_cell, origin=(10, 20))
    top_cell.add(ref)

    lib.write_gds(str(gds_path))

    result = load_gds(gds_path)

    assert (2, 0) in result
    assert len(result[(2, 0)]) == 1

    pts = result[(2, 0)][0].points
    # Check transformed coordinates (bounding box minimum should be at (10, 20))
    min_x = pts[:, 0].min()
    min_y = pts[:, 1].min()
    assert pytest.approx(min_x) == 10.0
    assert pytest.approx(min_y) == 20.0


def test_load_gds_file_not_found(tmp_path: Path) -> None:
    """Test that loading a non-existent file raises FileNotFoundError."""
    missing_path = tmp_path / "non_existent.gds"
    with pytest.raises(FileNotFoundError):
        load_gds(missing_path)


def test_main_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Test the CLI entry point prints layer and polygon counts."""
    gds_path = tmp_path / "cli.gds"

    lib = gdstk.Library(name="CLI_LIB")
    cell = lib.new_cell("TOP")
    cell.add(gdstk.rectangle((0, 0), (1, 1), layer=1, datatype=0))
    cell.add(gdstk.rectangle((0, 0), (2, 2), layer=2, datatype=0))
    lib.write_gds(str(gds_path))

    monkeypatch.setattr("sys.argv", ["gds_loader.py", str(gds_path)])
    main()

    captured = capsys.readouterr()
    assert "Layer (1, 0): 1 polygon(s)" in captured.out
    assert "Layer (2, 0): 1 polygon(s)" in captured.out
