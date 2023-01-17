#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example of creating a directional coupler simulation flow with tiny3d-klayout

@author: Mustafa Hammood, 2023
"""

from siepic_tidy3d import geometry, extend, sim
import tidy3d as td
from tidy3d import web
import klayout.db as pya
import numpy as np
import matplotlib.pyplot as plt

# %% draw the device
# define device parameters (in microns)
wg_width = 0.5
dc_gap = 0.2
dc_length = 12.5
sbend_r = 5
sbend_h = 2
sbend_l = 7

# define a layout object
ly = pya.Layout()
dbu = 0.001  # layout's database unit (in microns)

# create a cell object
cell = ly.create_cell('DirectionalCoupler')

# define layers
layer_device = ly.layer(1, 0)  # layer to define device's objects
layer_pinrec = ly.layer(69, 0)  # layer to define device's ports
layer_devrec = ly.layer(68, 0)  # layer to define device's boundaries

objects_device = []
# create coupling region waveguides
box = pya.Box(-geometry.to_dbu(dc_length/2, dbu), geometry.to_dbu(dc_gap/2, dbu),
              geometry.to_dbu(dc_length/2, dbu), geometry.to_dbu(wg_width + dc_gap/2, dbu))
objects_device.append(cell.shapes(layer_device).insert(box))

box = pya.Box(-geometry.to_dbu(dc_length/2, dbu), -geometry.to_dbu(dc_gap/2, dbu),
              geometry.to_dbu(dc_length/2, dbu), -geometry.to_dbu(wg_width + dc_gap/2, dbu))
objects_device.append(cell.shapes(layer_device).insert(box))

# create s-bend fan-out/in sections
s = geometry.sbend(dc_length/2, dc_gap/2+wg_width/2, wg_width, sbend_r, sbend_h,
                   sbend_l, direction='east', verbose=True)
objects_device.append(cell.shapes(layer_device).insert(s))

s = geometry.sbend(dc_length/2, -dc_gap/2-wg_width/2, wg_width, sbend_r, sbend_h,
                   sbend_l, direction='east', flip=True, verbose=True)
objects_device.append(cell.shapes(layer_device).insert(s))

s = geometry.sbend(-dc_length/2, -dc_gap/2-wg_width/2, wg_width, sbend_r, sbend_h,
                   sbend_l, direction='west', verbose=True)
objects_device.append(cell.shapes(layer_device).insert(s))

s = geometry.sbend(-dc_length/2, dc_gap/2+wg_width/2, wg_width, sbend_r, sbend_h,
                   sbend_l, direction='west', flip=True, verbose=True)
objects_device.append(cell.shapes(layer_device).insert(s))

# create straight regions near the ports
l_extra = 0.1
box = pya.Box(
    geometry.to_dbu(-dc_length/2-sbend_l-l_extra, dbu),
    geometry.to_dbu(sbend_h + dc_gap/2, dbu),
    geometry.to_dbu(-dc_length/2-sbend_l, dbu),
    geometry.to_dbu(sbend_h + wg_width + dc_gap/2, dbu))
objects_device.append(cell.shapes(layer_device).insert(box))

box = pya.Box(
    geometry.to_dbu(-dc_length/2-sbend_l-l_extra, dbu),
    -geometry.to_dbu(sbend_h + dc_gap/2, dbu),
    geometry.to_dbu(-dc_length/2-sbend_l, dbu),
    -geometry.to_dbu(sbend_h + wg_width + dc_gap/2, dbu))
objects_device.append(cell.shapes(layer_device).insert(box))

box = pya.Box(
    -geometry.to_dbu(-dc_length/2-sbend_l-l_extra, dbu),
    geometry.to_dbu(sbend_h + dc_gap/2, dbu),
    -geometry.to_dbu(-dc_length/2-sbend_l, dbu),
    geometry.to_dbu(sbend_h + wg_width + dc_gap/2, dbu))
objects_device.append(cell.shapes(layer_device).insert(box))

box = pya.Box(
    -geometry.to_dbu(-dc_length/2-sbend_l-l_extra, dbu),
    -geometry.to_dbu(sbend_h + dc_gap/2, dbu),
    -geometry.to_dbu(-dc_length/2-sbend_l, dbu),
    -geometry.to_dbu(sbend_h + wg_width + dc_gap/2, dbu))
objects_device.append(cell.shapes(layer_device).insert(box))


# create device boundary region
h_buffer = 2
box = pya.Box(
    geometry.to_dbu(-dc_length/2-sbend_l-l_extra, dbu),
    -geometry.to_dbu(h_buffer+sbend_h + wg_width + dc_gap/2, dbu),
    -geometry.to_dbu(-dc_length/2-sbend_l-l_extra, dbu),
    geometry.to_dbu(h_buffer+sbend_h + wg_width + dc_gap/2, dbu))
objects_devrec = cell.shapes(layer_devrec).insert(box)

# create port definition regions
geometry.make_pin(cell, 'opt1', [-dc_length/2-sbend_l-l_extra, sbend_h +
                                 wg_width/2 + dc_gap/2], wg_width, layer_pinrec, direction=180)

geometry.make_pin(cell, 'opt2', [-dc_length/2-sbend_l-l_extra, -sbend_h -
                                 wg_width/2 - dc_gap/2], wg_width, layer_pinrec, direction=180)

geometry.make_pin(cell, 'opt3', [dc_length/2+sbend_l+l_extra, sbend_h +
                                 wg_width/2 + dc_gap/2], wg_width, layer_pinrec, direction=0)

geometry.make_pin(cell, 'opt4', [dc_length/2+sbend_l+l_extra, -sbend_h -
                                 wg_width/2 - dc_gap/2], wg_width, layer_pinrec, direction=0)

# export layout
fname = "ex3_DirectionalCoupler.oas"
gzip = False
options = pya.SaveLayoutOptions()
ly.write(fname, gzip, options)

# %% extract the gds cell parameters for simulation and setup simulation
# define geometry
thick_dev = 0.22  # device layer thickness
thick_sub = 2  # substrate thickness
thick_super = 3  # superstrate thickness (microns)
sidewall_angle = 85  # device layer sidewall angle (degrees)
z_span = 4  # simulation z-span

# frequency and bandwidth of pulsed excitation
in_port = 'opt1'  # input port
wavl_min = 1.5  # simulation wavelength start (microns)
wavl_max = 1.6  # simulation wavelength end (microns)
wavl_pts = 11

# define materials structures
mat_dev = td.material_library["cSi"]["Li1993_293K"]
mat_sub = td.Medium(permittivity=1.48**2)
mat_super = td.Medium(permittivity=1.48**2)

simulation = sim.make_sim(cell, ly, layer_device, layer_devrec, layer_pinrec,
                          wavl_min=wavl_min, wavl_max=wavl_max, wavl_pts=wavl_pts,
                          in_port=in_port, z_span=z_span, thick_dev=thick_dev,
                          thick_sub=thick_sub, thick_super=thick_super,
                          mat_dev=mat_dev, mat_sub=mat_sub, mat_super=mat_super,
                          angle=sidewall_angle, visualize=True)

# %% upload and run the simulation
# create job, upload sim to server to begin running
job = web.Job(simulation=simulation, task_name=fname)

# %% run the simulation. CHECK THE SIMULATION IN THE UI BEFORE RUNNING!
sim_data = job.run(path=f"{fname}/sim_data.hdf5")

# %%

sim.visualize_results(sim_data, cell, ly, layer_pinrec, in_port)
