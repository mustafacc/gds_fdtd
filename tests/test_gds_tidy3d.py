#!/usr/bin/env python

"""Tests for `gds_tidy3d` package."""

import pytest
import klayout.db as pya
import os
import numpy as np
from gds_tidy3d import core, lyprocessor, simprocessor
import tidy3d as td


@pytest.fixture
def response_is_point_inside_polygon():
    """Sample pytest fixture that returns a point and polygon."""
    point = [0, 0]
    polygon = [[-1, -1], [1, -1], [1, 1], [-1, 1]]
    return {"point": point, "polygon": polygon}


def test_is_point_inside_polygon(response_is_point_inside_polygon):
    """Sample pytest test function with the pytest fixture as an argument."""
    point = response_is_point_inside_polygon["point"]
    polygon = response_is_point_inside_polygon["polygon"]
    assert core.is_point_inside_polygon(point, polygon)


def test_parse_yaml_tech():
    file_path = os.path.join(os.path.dirname(__file__), "tech.yaml")

    expected_output = {
        "name": "EBeam",
        "substrate": [
            {"z_base": 0.0, "z_span": -2, "material_type": "nk", "material": 1.48}
        ],
        "superstrate": [
            {"z_base": 0.0, "z_span": 3, "material_type": "nk", "material": 1.48}
        ],
        "pinrec": [{"layer": [1, 10]}],
        "devrec": [{"layer": [68, 0]}],
        "device": [
            {
                "layer": [1, 0],
                "z_base": 0.0,
                "z_span": 0.22,
                "material_type": "tidy3d_db",
                "material": ["cSi", "Li1993_293K"],
                "sidewall_angle": 85,
            },
            {
                "layer": [1, 5],
                "z_base": 0.3,
                "z_span": 0.22,
                "material_type": "tidy3d_db",
                "material": ["Si3N4", "Philipp1973Sellmeier"],
                "sidewall_angle": 80,
            },
        ],
    }

    parsed_data = core.parse_yaml_tech(file_path)
    assert parsed_data == expected_output


def test_layout_initialization():
    ly = pya.Layout()
    cell = pya.Cell()
    test_layout = core.layout("TestLayout", ly, cell)

    assert test_layout.name == "TestLayout"
    assert test_layout.ly == ly
    assert test_layout.cell == cell


def test_layout_dbu_property():
    ly = pya.Layout()
    ly.dbu = 0.001  # Set a specific dbu value
    test_layout = core.layout("TestLayout", ly, ly.create_cell("TestCell"))

    assert test_layout.dbu == 0.001


def test_port_initialization():
    test_port = core.port("opt1", [0, 0, 0], 0.5, "+")

    assert test_port.name == "opt1"
    assert test_port.center == [0, 0, 0]
    assert test_port.width == 0.5
    assert test_port.direction == "+"
    assert test_port.height is None
    assert test_port.material is None


def test_port_properties():
    test_port = core.port("opt1", [0, 0, 0], 0.5, "+")

    assert test_port.x == 0
    assert test_port.y == 0
    assert test_port.z == 0


def test_port_idx_property():
    test_port = test_port = core.port("opt1", [0, 0, 0], 0.5, "+")

    assert test_port.idx == 1  # assuming idx is the reverse of the digits in the name


def test_structure_initialization_with_all_parameters():
    test_polygon = [[0, 0], [1, 1], [2, 2]]
    test_structure = core.structure("structure", test_polygon, 0.0, 0.22, "Si", 85)

    assert test_structure.name == "structure"
    assert test_structure.polygon == test_polygon
    assert test_structure.z_base == 0.0
    assert test_structure.z_span == 0.22
    assert test_structure.material == "Si"
    assert test_structure.sidewall_angle == 85


def test_structure_initialization_with_default_sidewall_angle():
    test_polygon = [[0, 0], [1, 1], [2, 2]]
    test_structure = core.structure("structure", test_polygon, 0.0, 0.22, "Si", 85)

    assert test_structure.sidewall_angle == 85  # Checking default value


def test_dilate_with_default_extension():
    vertices = [[-1, -1], [1, -1], [1, 1], [-1, 1]]
    expected_result = [[-2, -2], [2, -2], [2, 2], [-2, 2]]

    assert lyprocessor.dilate(vertices, extension=1) == expected_result


def test_load_layout_with_single_top_cell():
    test_file = "tests/si_sin_escalator.gds"  # Replace with actual path
    result = lyprocessor.load_layout(test_file)

    assert isinstance(result, core.layout)  # Assuming layout is a class
    assert (
        result.name == "si_sin_escalator"
    )  # Replace with the actual expected cell name
    # Further assertions can be made depending on the properties of the layout and cell


def test_load_layout_with_multiple_top_cells():
    test_file = "tests/bad_topcells.gds"  # Replace with actual path

    with pytest.raises(ValueError) as excinfo:
        lyprocessor.load_layout(test_file)

    assert "More than one top cell found" in str(excinfo.value)


def test_load_region():
    test_file = "tests/si_sin_escalator.gds"  # Replace with actual path
    layout = lyprocessor.load_layout(test_file)
    region = lyprocessor.load_region(layout=layout, layer=[68, 0])
    assert region.y_center == 0
    assert isinstance(region, core.region)


def test_load_structure():
    test_file = "tests/si_sin_escalator.gds"  # Replace with actual path
    layout = lyprocessor.load_layout(test_file)
    device_si = lyprocessor.load_structure(
        layout,
        name="Si",
        layer=[1, 0],
        z_base=0,
        z_span=0.22,
        material=td.Medium(permittivity=3.47**2),
    )

    assert device_si[0].name == "Si_0"
    assert isinstance(device_si[0], core.structure)


def test_load_structure_from_bounds():
    test_file = "tests/si_sin_escalator.gds"  # Replace with actual path
    layout = lyprocessor.load_layout(test_file)

    bounds = lyprocessor.load_region(layout, layer=[68, 0], z_center=0 / 2, z_span=4)

    device_super = lyprocessor.load_structure_from_bounds(
        bounds,
        name="Superstrate",
        z_base=0,
        z_span=2,
        material=td.Medium(permittivity=1.48**2),
    )
    assert isinstance(device_super, core.structure)
    assert device_super.sidewall_angle == 90  # angle is always supposed to be 90 from structure from bounds

def test_load_ports():
    test_file = "tests/si_sin_escalator.gds"  # Replace with actual path
    layout = lyprocessor.load_layout(test_file)
    ports_si = lyprocessor.load_ports(layout, layer=[1, 10])

    assert len(ports_si) == 2
    assert ports_si[0].name == 'opt1'
    assert ports_si[0].width == .5
    assert all(p.y == 0 for p in ports_si)

def test_make_source():
    test_file = "tests/si_sin_escalator.gds"  # Replace with actual path
    layout = lyprocessor.load_layout(test_file)
    ports_si = lyprocessor.load_ports(layout, layer=[1, 10])
    for p in ports_si:
        p.height == 0.22
        p.center[2] = 0

    for p in ports_si:
        source = simprocessor.make_source(
            port=p,
        )

    assert isinstance(source, td.ModeSource)
    assert source.mode_index == 0

def test_make_structure():
    test_file = "tests/si_sin_escalator.gds"  # Replace with actual path
    layout = lyprocessor.load_layout(test_file)

    # load all the ports in the device and (optional) initialize each to have a center
    ports = lyprocessor.load_ports(layout, layer=[1, 10])

    # load the device simulation region
    bounds = lyprocessor.load_region(
        layout, layer=[68, 0], z_center=.22 / 2, z_span=4
    )

    # load the silicon structures in the device in layer (1,0)
    device_si = lyprocessor.load_structure(
        layout,
        name="Si",
        layer=[1, 0],
        z_base=0,
        z_span=.22,
        material=td.material_library["cSi"]["Li1993_293K"]
    )

    # load the silicon structures in the device in layer (1,0)
    device_sin = lyprocessor.load_structure(
        layout,
        name="SiN",
        layer=[1, 5],
        z_base=.3,
        z_span=.22,
        material=td.material_library["cSi"]["Li1993_293K"]
    )

    device_super = lyprocessor.load_structure_from_bounds(
        bounds, name="Superstrate", z_base=0, z_span=2, material=td.Medium(permittivity=1.48**2)
    )
    device_sub = lyprocessor.load_structure_from_bounds(
        bounds, name="Substrate", z_base=0, z_span=-3, material=td.Medium(permittivity=1.48**2)
    )
    # create the device by loading the structures
    device = core.component(
        name=layout.name,
        structures= [device_sub, device_super, device_si, device_sin],
        ports=ports,
        bounds=bounds,
    )
    # define structures from device
    structures = simprocessor.make_structures(device)

def test_make_port_monitor():
    test_file = "tests/si_sin_escalator.gds"  # Replace with actual path
    layout = lyprocessor.load_layout(test_file)

    # load all the ports in the device and (optional) initialize each to have a center
    ports = lyprocessor.load_ports(layout, layer=[1, 10])

    # load the device simulation region
    bounds = lyprocessor.load_region(
        layout, layer=[68, 0], z_center=.22 / 2, z_span=4
    )

    # load the silicon structures in the device in layer (1,0)
    device_si = lyprocessor.load_structure(
        layout,
        name="Si",
        layer=[1, 0],
        z_base=0,
        z_span=.22,
        material=td.material_library["cSi"]["Li1993_293K"]
    )

    # load the silicon structures in the device in layer (1,0)
    device_sin = lyprocessor.load_structure(
        layout,
        name="SiN",
        layer=[1, 5],
        z_base=.3,
        z_span=.22,
        material=td.material_library["cSi"]["Li1993_293K"]
    )

    device_super = lyprocessor.load_structure_from_bounds(
        bounds, name="Superstrate", z_base=0, z_span=2, material=td.Medium(permittivity=1.48**2)
    )
    device_sub = lyprocessor.load_structure_from_bounds(
        bounds, name="Substrate", z_base=0, z_span=-3, material=td.Medium(permittivity=1.48**2)
    )
    # create the device by loading the structures
    device = core.component(
        name=layout.name,
        structures= [device_sub, device_super, device_si, device_sin],
        ports=ports,
        bounds=bounds,
    )
    # define monitors
    monitors = []
    for p in device.ports:
        monitors.append(
            simprocessor.make_port_monitor(
                p,
                freqs=td.C_0 / np.linspace(1.5, 1.6, 101),
                depth=2,
                width=3,
            )
        )
    assert isinstance(monitors[0].mode_spec, td.ModeSpec)
    assert all(m.name.startswith('opt') for m in monitors)

if __name__ == "__main__":
    pytest.main([__file__])
