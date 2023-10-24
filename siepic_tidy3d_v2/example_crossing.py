# %%!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 12:19:10 2023

@author: mustafah
"""
import tidy3d as td
from tidy3d import web
import siepic_tidy3d as sitd

fname_gds = "crossing.gds"

# define 3D geometry paramters not captured in the 2D layout
thickness_si = 0.22
thickness_sub = 2
thickness_super = 3
sidewall_angle = 88  # sidewall angle of the structures

# define simulation parameters
z_span = 4  # Z-span of the simulation

# define materials structures
mat_si = td.material_library["cSi"]["Li1993_293K"]
mat_sub = td.Medium(permittivity=1.48**2)
mat_super = td.Medium(permittivity=1.48**2)

# frequency and bandwidth of pulsed excitation
in_port = "opt1"  # input port
in_pol = "TE"  # input polarization state (options are TE, TM, TETM)
wavl_min = 1.45  # simulation wavelength start (microns)
wavl_max = 1.65  # simulation wavelength end (microns)
wavl_pts = 101

# define symmetry across Z axis (TE mode) - set to -1 for anti symmetric
symmetry = (0, 0, 1)

# %% load and process the layout file
layout = sitd.lyprocessor.load_layout(fname_gds)

# load all the ports in the device and (optional) initialize each to have a center
ports_si = sitd.lyprocessor.load_ports(
    layout, layer=[1, 10], z_center=thickness_si / 2, z_span=thickness_si
)

# load the device simulation region
bounds = sitd.lyprocessor.load_region(
    layout, layer=[68, 0], z_center=thickness_si / 2, z_span=z_span
)

# load the silicon structures in the device in layer (1,0)
device_si = sitd.lyprocessor.load_structure(
    layout, name="Si", layer=[1, 0], z_base=0, z_span=thickness_si, material=mat_si
)

# make the superstrate and substrate based on device bounds
# this information isn't typically captured in a 2D layer stack
device_super = sitd.lyprocessor.load_structure_from_bounds(
    bounds, name="Superstrate", z_base=0, z_span=thickness_super, material=mat_super
)
device_sub = sitd.lyprocessor.load_structure_from_bounds(
    bounds, name="Substrate", z_base=0, z_span=-thickness_sub, material=mat_sub
)

# create the device by loading the structures
device = sitd.core.component(
    name=layout.name,
    structures=[device_sub, device_super, device_si],
    ports=ports_si,
    bounds=bounds,
)

#%%
simulation = sitd.simprocessor.make_sim(device=device, field_monitor=True)
#%%
job = web.Job(simulation=simulation, task_name='test')

# %%