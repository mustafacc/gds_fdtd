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
sidewall_angle = 82  # device layer sidewall angle (degrees)
z_span = 4  # simulation z-span

# frequency and bandwidth of pulsed excitation
input_port = 1  # input port
wavl_min = 1.5  # simulation wavelength start (microns)
wavl_max = 1.6  # simulation wavelength end (microns)
wavl_pts = 11
freq0 = td.C_0/((wavl_max+wavl_min)/2)
freqs = td.C_0/np.linspace(wavl_min, wavl_max, wavl_pts)
fwidth = 0.5*freq0
# sim. time in secs
run_time = 85/fwidth

# define materials structures
mat_dev = td.material_library["cSi"]["Li1993_293K"]
mat_sub = td.Medium(permittivity=1.48**2)
mat_super = td.Medium(permittivity=1.48**2)

# resolution control: minimum number of grid cells per wavelength in each material
grid_cells_per_wvl = 16

# apply pml in all directions
boundary_spec = td.BoundarySpec.all_sides(boundary=td.PML())
polygons_device = extend.get_polygons(cell, layer_device, dbu)

# define device structures
devrec, sim_x, sim_y = extend.get_devrec(cell, layer_devrec, dbu)
ports = extend.get_ports(cell, layer_pinrec, dbu)
structures = sim.make_structures(polygons_device, devrec, thick_dev, thick_sub,
                                 thick_super, mat_dev, mat_sub, mat_super, sidewall_angle)
# define source on a given port
source = sim.make_source(
    ports[input_port], thick_dev=thick_dev, freq0=freq0, fwidth=fwidth)
# define monitors
monitors = sim.make_monitors(ports, thick_dev, freq0=freq0, freqs=freqs)

# simulation domain size (in microns)
sim_size = [sim_x, sim_y, 4]

# initialize the simulation
simulation = td.Simulation(
    size=sim_size,
    grid_spec=td.GridSpec.auto(min_steps_per_wvl=grid_cells_per_wvl),
    structures=structures,
    sources=[source],
    monitors=monitors,
    run_time=run_time,
    boundary_spec=boundary_spec,
)

# %% visualize the simulation and send the job to web ui

for m in simulation.monitors:
    m.help()

# visualize the source
source.source_time.plot(np.linspace(0, run_time, 1001))
plt.show()

# visualize the simulation
fig, ax = plt.subplots(1, 3, figsize=(13, 4))
simulation.plot_eps(z=0, freq=freq0, ax=ax[0])
simulation.plot_eps(y=0, freq=freq0, ax=ax[1])
simulation.plot_eps(x=0, freq=freq0, ax=ax[2])

# visualize geometry
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
simulation.plot(z=thick_dev/2, ax=ax1)
simulation.plot(x=0.1, ax=ax2)
ax2.set_xlim([-3, 3])
plt.show()

# %% upload and run the simulation
# create job, upload sim to server to begin running
job = web.Job(simulation=simulation, task_name="CouplerVerify")
sim_data = job.run(path="data/sim_data.hdf5")
# %% download the results and load them into a simulation
#sim_data = web.load("data/sim_data.hdf5")


def measure_transmission(sim_data):
    """Constructs a "row" of the scattering matrix when sourced from top left port"""

    input_amp = sim_data['1'].amps.sel(direction="+")

    amps = np.zeros((4, 11), dtype=complex)
    directions = ("-", "-", "+", "+")
    for i, (monitor, direction) in enumerate(
        zip(sim_data.simulation.monitors[:4], directions)
    ):
        amp = sim_data[monitor.name].amps.sel(direction=direction)
        amp_normalized = amp / input_amp
        amps[i] = np.squeeze(amp_normalized.values)

    return amps


# monitor and test out the measure_transmission function the results of the single run

amps_arms = measure_transmission(sim_data)
print("mode amplitudes in each port: \n")
fig, ax = plt.subplots(1, 1)
wavl = np.linspace(wavl_min, wavl_max, wavl_pts)
ax.set_xlabel('Wavelength [microns]')
ax.set_ylabel('Transmission [dB]')
for amp, monitor in zip(amps_arms, sim_data.simulation.monitors[:-1]):
    print(f'\tmonitor     = "{monitor.name}"')
    plt.plot(wavl, [10*np.log10(abs(i)**2) for i in amp], label=f"S1{monitor.name}")
    print(f"\tamplitude^2 = {[abs(i)**2 for i in amp]}")
    print(f"\tphase       = {[np.angle(i)**2 for i in amp]} (rad)\n")
fig.legend()

fig, ax = plt.subplots(1, 1, figsize=(16, 3))
sim_data.plot_field("field", "Ey", z=thick_dev/2, freq=freq0, ax=ax)
plt.show()
