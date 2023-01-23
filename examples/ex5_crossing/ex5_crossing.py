#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 18:10:33 2023

@author: mustafa
"""
from siepic_tidy3d import geometry, extend, sim
import klayout.db as pya
import tidy3d as td
from tidy3d import web
import numpy as np
import matplotlib.pyplot as plt

fname = 'crossing'

# import gds and define layers
cell, ly = sim.load_gds(fname)

layer_device = ly.layer(1, 0)  # layer to define device's objects
layer_pinrec = ly.layer(69, 0)  # layer to define device's ports
layer_devrec = ly.layer(68, 0)  # layer to define device's boundaries

# %% extract the gds cell parameters for simulation and setup simulation
# define geometry
thick_dev = 0.22  # device layer thickness
thick_sub = 2  # substrate thickness
thick_super = 3  # superstrate thickness (microns)
sidewall_angle = 85  # device layer sidewall angle (degrees)
z_span = 4  # simulation z-span

# frequency and bandwidth of pulsed excitation
in_port = 'opt1'  # input port
wavl_min = 1.25  # simulation wavelength start (microns)
wavl_max = 1.6  # simulation wavelength end (microns)
wavl_pts = 51

# define materials structures
mat_dev = td.material_library["cSi"]["Li1993_293K"]
mat_sub = td.Medium(permittivity=1.48**2)
mat_super = td.Medium(permittivity=1.48**2)

# define symmetry across Z axis (TE mode) - set to -1 for anti symmetric
symmetry = (0, 0, 1)  # probably not a good idea to set symmetry with sidewall angle?

simulation = sim.make_sim(cell, ly, layer_device, layer_devrec, layer_pinrec,
                          wavl_min=wavl_min, wavl_max=wavl_max, wavl_pts=wavl_pts,
                          in_port=in_port, z_span=z_span, thick_dev=thick_dev,
                          thick_sub=thick_sub, thick_super=thick_super,
                          mat_dev=mat_dev, mat_sub=mat_sub, mat_super=mat_super,
                          angle=sidewall_angle, symmetry=symmetry, visualize=True)


# %% upload and run the simulation
# create job, upload sim to server to begin running
job = web.Job(simulation=simulation, task_name=fname)

# %% run the simulation. CHECK THE SIMULATION IN THE UI BEFORE RUNNING!
sim_data = job.run(path=f"{fname}/sim_data.hdf5")

# %%

sim.visualize_results(sim_data, cell, ly, layer_pinrec, in_port)
