"""
GDS_Tidy3D integration toolbox.

Lumerical tools module.
@author: Mustafa Hammood, 2024
"""

from gds_tidy3d.core import component
import logging
import lumapi
import numpy as np

m_to_um = 1e-6

def to_lumerical(c: component, lum: lumapi.FDTD, tech: dict, buffer: float=2.):
    """Add an input component with a given tech to a lumerical instance

    Args:
        c (component): input component.
        lum (lumapi.FDTD): lumerical FDTD instance.
        tech (dict): technology dictionary
    """

    # TODO fix box tox handling here
    structures = []
    for s in c.structures:
        # if structure is a list then its a device (could have multiple polygons inside)
        if type(s) == list:
            for i in s:
                if i.z_span < 0:
                    bounds = (i.z_base + i.z_span, i.z_base)
                else:
                    bounds = (i.z_base, i.z_base + i.z_span)

                lum.putv("polygon_vertices", m_to_um*np.array(i.polygon))

                make_poly = f" \
                addpoly; set('vertices',polygon_vertices); \
                set('material', '{i.material}'); set('name', '{i.name}'); set('z min', {m_to_um*bounds[0]}); set('x',0); set('y',0); set('z max',{m_to_um*bounds[1]});    \
                addtogroup('device'); \
                ?'Device Polygons added';"

                lum.eval(make_poly)
        # if structure is not a list then its a region
        else:
            if s.z_span < 0:
                bounds = (s.z_base + s.z_span, s.z_base)
            else:
                bounds = (s.z_base, s.z_base + s.z_span)
            lum.putv("polygon_vertices", m_to_um*np.array(s.polygon))

            make_poly = f" \
            addpoly; set('vertices',polygon_vertices); \
            set('material', '{s.material}'); set('name', '{s.name}'); set('z min', {m_to_um*bounds[0]}); set('x',0); set('y',0); set('z max',{m_to_um*bounds[1]});    \
            set('alpha',0.5); ?'Process regions added';"

            lum.eval(make_poly)

    # extend ports beyond sim region
    for p in c.ports:
        if p.direction == 0:
            pts = [
                [p.center[0], p.center[1] + p.width / 2],
                [p.center[0] + buffer, p.center[1] + p.width / 2],
                [p.center[0] + buffer, p.center[1] - p.width / 2],
                [p.center[0], p.center[1] - p.width / 2],
            ]
        elif p.direction == 180:
            pts = [
                [p.center[0], p.center[1] + p.width / 2],
                [p.center[0] - buffer, p.center[1] + p.width / 2],
                [p.center[0] - buffer, p.center[1] - p.width / 2],
                [p.center[0], p.center[1] - p.width / 2],
            ]
        elif p.direction == 90:
            pts = [
                [p.center[0] - p.width / 2, p.center[1]],
                [p.center[0] - p.width / 2, p.center[1] + buffer],
                [p.center[0] + p.width / 2, p.center[1] + buffer],
                [p.center[0] + p.width / 2, p.center[1]],
            ]
        elif p.direction == 270:
            pts = [
                [p.center[0] - p.width / 2, p.center[1]],
                [p.center[0] - p.width / 2, p.center[1] - buffer],
                [p.center[0] + p.width / 2, p.center[1] - buffer],
                [p.center[0] + p.width / 2, p.center[1]],
            ]

        lum.putv("polygon_vertices", m_to_um*np.array(pts))

        make_poly = f" \
        addpoly; set('vertices',polygon_vertices); \
        set('material', '{p.material}'); set('name', '{p.name}'); set('z min', {m_to_um*(p.center[2] - p.height / 2)}); set('x',0); set('y',0); set('z max',{m_to_um*(p.center[2] + p.height / 2)});    \
        addtogroup('ports'); ?'Process regions added';"

        lum.eval(make_poly)
    """
        structures.append(
            td.Structure(
                geometry=td.PolySlab(
                    vertices=pts,
                    slab_bounds=(
                        p.center[2] - p.height / 2,
                        p.center[2] + p.height / 2,
                    ),
                    axis=2,
                    sidewall_angle=(90 - device.structures[0].sidewall_angle)
                    * (np.pi / 180),
                ),
                medium=p.material,
                name=f"port_{p.name}",
            )
        )
    """
    return structures



