# %%!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 12:19:10 2023

@author: mustafah
"""
import tidy3d as td
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

# load the silicon structur#%%es in the device in layer (1,0)
device_si = sitd.lyprocessor.load_structure(
    layout, name="Si", layer=[1, 0], z_base=0, z_span=thickness_si, material=mat_si
)

# make the superstrate and substrate based on device bounds
# this information isn't typically captured in a 2D layer stack
device_super = sitd.lyprocessor.load_structure_from_bounds(
    bounds, name="Superstrate", thickness=thickness_super, z_base=0, material=mat_super
)

# %%


device_sub = sitd.lyprocessor.load_structure_from_bounds(
    bounds, name="Substrate", thickness=-thickness_sub, z_base=0, material=mat_sub
)

# create the device by loading the structures
device = sitd.geometry.component(
    name=layout.name,
    structures=[device_si, device_sub, device_super],
    ports=ports,
    bounds=bounds,
)

# %% build the simulation object
simulation = sitd.sim.make_sim(device=device, symmetry=symmetry)

# %% create a wavelength sweep simulation with single port excitation
simulation.wavl_sweep(
    wavl_min=wavl_min,
    wavl_max=wavl_max,
    wavl_pts=wavl_pts,
    in_port=in_port,
    in_pol=in_pol,
)
simulation.wavl_sweep.visualize()

# %% send the simulation job to the cluser
simulation.wavl_sweep.upload()
# visualizing the simulation job in the tidy3d web UI is probably a good idea!
# %% execute the simulation
simulation.wavl_sweep.run()
# %% visualize the simulation results
simulation.wavl_sweep.visualize_rslts()
