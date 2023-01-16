#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wrapper methods to build simulations with Tidy3D.

@author: Mustafa Hammood
"""

def make_structures(polygons_device, devrec, thick_dev, thick_sub, thick_super, mat_dev, mat_sub, mat_super, sidewall_angle=85):
    import tidy3d as td
    import numpy as np
    structures = []

    structures.append(td.Structure(
        geometry=td.PolySlab(
            vertices=devrec,
            slab_bounds=(0, thick_super),
            axis=2,
        ),
        medium=mat_super,
    ))

    for poly in polygons_device:
        structures.append(td.Structure(
            geometry=td.PolySlab(
                vertices=poly,
                slab_bounds=(0, thick_dev),
                axis=2,
                sidewall_angle=(90-sidewall_angle) * (np.pi/180),
            ),
            medium=mat_dev,
        ))

    structures.append(td.Structure(
        geometry=td.PolySlab(
            vertices=devrec,
            slab_bounds=(-thick_sub, 0),
            axis=2,
        ),
        medium=mat_sub,
    ))

    return structures


def make_source(port, width=3, depth=2, thick_dev=0.22, freq0=2e14, fwidth=1e13, buffer=0.25):
    import tidy3d as td
    if port['direction'] == 0:
        x_buffer = -buffer
        y_buffer = 0
    elif port['direction'] == 180:
        x_buffer = buffer
        y_buffer = 0
    elif port['direction'] == 90:
        x_buffer = 0
        y_buffer = -buffer
    elif port['direction'] == 270:
        x_buffer = 0
        y_buffer = buffer
    if port['direction'] in [180, 270]:
        direction = "+"
    else:
        direction = "-"
    msource = td.ModeSource(
        center=[port['x']+x_buffer, port['y']+y_buffer, thick_dev/2],
        size=[0, width, depth],
        direction=direction,
        source_time=td.GaussianPulse(freq0=freq0, fwidth=fwidth),
        mode_spec=td.ModeSpec(),
        mode_index=0,
    )
    return msource


def make_monitors(ports, thick_dev=0.22, freq0=2e14, freqs=2e14, buffer=0.5):
    """Create monitors for a given list of ports."""
    import tidy3d as td
    monitors = []
    for p in ports:
        if ports[p]['direction'] == 0:
            x_buffer = -buffer
            y_buffer = 0
        elif ports[p]['direction'] == 180:
            x_buffer = buffer
            y_buffer = 0
        elif ports[p]['direction'] == 90:
            x_buffer = 0
            y_buffer = -buffer
        elif ports[p]['direction'] == 270:
            x_buffer = 0
            y_buffer = buffer
        # mode monitors
        monitors.append(td.ModeMonitor(
            center=[ports[p]['x']+x_buffer, ports[p]
                    ['y']+y_buffer, thick_dev/2],
            size=[0, ports[p]['width']*5, thick_dev*10],
            freqs=freqs,
            mode_spec=td.ModeSpec(),
            name=str(p),
        ))
    # field monitor
    monitors.append(td.FieldMonitor(
        center=[0, 0, thick_dev/2],
        size=[td.inf, td.inf, 0],
        freqs=freqs,
        name="field",
    ))
    return monitors
