"""
GDS_Tidy3D integration toolbox.

Lumerical tools module.
@author: Mustafa Hammood, 2024
"""

from gds_tidy3d.core import structure, component
import logging
import lumapi
import numpy as np

m_to_um = 1e-6

def structure_to_lum_poly(
    s: structure, 
    lum: lumapi.FDTD,
    alpha: float=1.,
    group: bool=False,
    group_name: str='group',
):
    """import a structure objecto to a lumerical instance.

    Args:
        s (structure): structure to instantiate
        lum (lumapi.FDTD): lumerical instance
        alpha (float, optional): transperancy setting. Defaults to 1..
        group (bool, optional): flag to add the structure to a given group. Defaults to False.
        group_name (str, optional): group name, if group is True. Defaults to 'group'.
    """
    if s.z_span < 0:
        bounds = (s.z_base + s.z_span, s.z_base)
    else:
        bounds = (s.z_base, s.z_base + s.z_span)

    poly = lum.addpoly(
        vertices=m_to_um*np.array(s.polygon),
        x=0,
        y=0,
        z_min=m_to_um*bounds[0],
        z_max=m_to_um*bounds[1],
        name=s.name,
        material=s.material["lum"] if isinstance(s.material, dict) else s.material,
        alpha=alpha,
    )

    if group:
        lum.addtogroup(group_name)
    lum.eval(f"?'Polygons {s.name} added';")

    return poly

def to_lumerical(c: component, lum: lumapi.FDTD, buffer: float=2.):
    """Add an input component with a given tech to a lumerical instance.

    Args:
        c (component): input component.
        lum (lumapi.FDTD): lumerical FDTD instance.
    """

    # TODO fix box tox handling here
    structures = []
    for s in c.structures:
        # if structure is a list then its a device (could have multiple polygons inside)
        if type(s) == list:
            for i in s:
                structures.append(structure_to_lum_poly(s=i, lum=lum, group=True, group_name='device'))

        # if structure is not a list then its a region
        else:
            structures.append(structure_to_lum_poly(s=s, lum=lum, alpha=0.5))

    # extend ports beyond sim region
    for p in c.ports:
        structures.append(lum.addpoly(
            vertices=m_to_um*np.array(p.polygon_extension(buffer=buffer)),
            x=0,
            y=0,
            z_min=m_to_um*(p.center[2] - p.height / 2),
            z_max=m_to_um*(p.center[2] + p.height / 2),
            name=p.name,
            material=p.material["lum"] if isinstance(p.material, dict) else p.material,
        ))

        lum.addtogroup('ports')
        lum.eval(f"?'port {p.name} added';")

    return structures

def setup_lum_fdtd(
    c: component,
    lum: lumapi.FDTD,
):
    pass