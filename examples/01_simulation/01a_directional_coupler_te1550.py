# %%!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Mustafa Hammood
Example defining a tidy3d simulation from manually defined geometry.
"""
import tidy3d as td
import gds_fdtd as gtd
import os

# Define the path to the GDS file
file_gds = os.path.join(os.path.dirname(os.path.dirname(__file__)), "devices.gds")

# Define 3D geometry parameters not captured in the 2D layout
thickness_si = 0.22  # Silicon thickness
thickness_sub = 2    # Substrate thickness
thickness_super = 2  # Superstrate thickness
sidewall_angle = 88  # Sidewall angle of the structures

# Define simulation parameters
z_span = 4  # Z-span of the simulation

# Define material structures
mat_si = td.material_library["cSi"]["Li1993_293K"]
mat_sub = td.Medium(permittivity=1.48**2)
mat_super = td.Medium(permittivity=1.48**2)

# Frequency and bandwidth of pulsed excitation
in_pol = "TE"  # Input polarization state (options are TE, TM, TETM)
wavl_min = 1.45  # Simulation wavelength start (microns)
wavl_max = 1.65  # Simulation wavelength end (microns)
wavl_pts = 101   # Number of wavelength points

# Define symmetry across Z axis (TE mode) - set to -1 for anti-symmetric
# Warning: Ensure structure is symmetric across symmetry axis!
symmetry = (0, 0, 1)

# Load and process the layout file
layout = gtd.lyprocessor.load_layout(file_gds, top_cell='directional_coupler_te1550')

# Load all the ports in the device and (optional) initialize each to have a center
ports_si = gtd.lyprocessor.load_ports(layout, layer=[1, 10])

# Load the device simulation region
bounds = gtd.lyprocessor.load_region(
    layout, layer=[68, 0], z_center=thickness_si / 2, z_span=z_span
)

# Load the silicon structures in the device in layer (1,0)
device_si = gtd.lyprocessor.load_structure(
    layout, name="Si", layer=[1, 0], z_base=0, z_span=thickness_si, material=mat_si
)

# Make the superstrate and substrate based on device bounds
# This information isn't typically captured in a 2D layer stack
device_super = gtd.lyprocessor.load_structure_from_bounds(
    bounds, name="Superstrate", z_base=0, z_span=thickness_super, material=mat_super
)
device_sub = gtd.lyprocessor.load_structure_from_bounds(
    bounds, name="Substrate", z_base=0, z_span=-thickness_sub, material=mat_sub
)

# Create the device by loading the structures
device = gtd.core.component(
    name=layout.name,
    structures=[device_sub, device_super, device_si],
    ports=ports_si,
    bounds=bounds,
)

# Create the simulation object
simulation = gtd.simprocessor.make_sim(
    device=device,
    wavl_min=wavl_min,
    wavl_max=wavl_max,
    wavl_pts=wavl_pts,
    symmetry=symmetry,
    in_port=device.ports[1],
    z_span=z_span,
    field_monitor_axis='z',
)

# Upload and run the simulation
# Create job, upload sim to server to begin running
simulation.upload()

# Run the simulation. CHECK THE SIMULATION IN THE UI BEFORE RUNNING!
simulation.execute()

# Visualize the results
simulation.visualize_results()

# %%
