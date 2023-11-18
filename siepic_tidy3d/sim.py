#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wrapper methods to build simulations with Tidy3D.

@author: Mustafa Hammood
"""
import tidy3d as td


def load_gds(fname):
    import klayout.db as pya
    ly = pya.Layout()
    lmap = ly.read(fname+'.gds')
    if ly.cells() > 1:
        ValueError('More than one top cell found, ensure only 1 top cell exists.')
    else:
        cell = ly.top_cell()
    return cell, ly


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


def make_source(port, width=3, depth=2, thick_dev=0.22, freq0=2e14, num_freqs=5, fwidth=1e13, buffer=0.25):
    import tidy3d as td
    if port['direction'] == 0:
        x_buffer = -buffer
        y_buffer = 0
        size = [0, width, depth]
    elif port['direction'] == 180:
        x_buffer = buffer
        y_buffer = 0
        size = [0, width, depth]
    elif port['direction'] == 90:
        x_buffer = 0
        y_buffer = -buffer
        size = [width, 0, depth]
    elif port['direction'] == 270:
        x_buffer = 0
        y_buffer = buffer
        size = [width, 0, depth]
    if port['direction'] in [180, 270]:
        direction = "+"
    else:
        direction = "-"
    msource = td.ModeSource(
        center=[port['x']+x_buffer, port['y']+y_buffer, thick_dev/2],
        size=size,
        direction=direction,
        source_time=td.GaussianPulse(freq0=freq0, fwidth=fwidth),
        mode_spec=td.ModeSpec(),
        mode_index=0,
        num_freqs=num_freqs,
    )
    return msource


def make_monitors(ports, thick_dev=0.22, freqs=2e14, buffer=0.5, port_scale=5, z_span=4):
    """Create monitors for a given list of ports."""
    import tidy3d as td
    monitors = []
    for p in ports:
        if ports[p]['direction'] == 0:
            x_buffer = -buffer
            y_buffer = 0
            size = [0, ports[p]['width']*port_scale, z_span]
        elif ports[p]['direction'] == 180:
            x_buffer = buffer
            y_buffer = 0
            size = [0, ports[p]['width']*port_scale, z_span]
        elif ports[p]['direction'] == 90:
            x_buffer = 0
            y_buffer = -buffer
            size = [ports[p]['width']*port_scale, 0, z_span]
        elif ports[p]['direction'] == 270:
            x_buffer = 0
            y_buffer = buffer
            size = [ports[p]['width']*port_scale, 0, z_span]
        # mode monitors
        monitors.append(td.ModeMonitor(
            center=[ports[p]['x']+x_buffer, ports[p]
                    ['y']+y_buffer, thick_dev/2],
            size=size,
            freqs=freqs,
            mode_spec=td.ModeSpec(),
            name=ports[p]['name'],
        ))
    # field monitor
    monitors.append(td.FieldMonitor(
        center=[0, 0, thick_dev/2],
        size=[td.inf, td.inf, 0],
        freqs=freqs,
        name="field",
    ))
    return monitors


def find_port(ports, in_port):
    for p in ports:
        if ports[p]['name'] == in_port:
            return p
    return


def make_sim(cell, ly, layer_device, layer_devrec, layer_pinrec, in_port='opt1',
             thick_dev=0.22, thick_sub=2, thick_super=3, angle=88, fwidth=9.6e13,
             mat_dev=td.material_library["cSi"]["Li1993_293K"],
             mat_sub=td.Medium(permittivity=1.48**2),
             mat_super=td.Medium(permittivity=1.48**2),
             wavl_min=1.5, wavl_max=1.6, wavl_pts=11, grid_cells_per_wvl=15,
             boundary=td.BoundarySpec.all_sides(boundary=td.PML()), z_span=4,
             symmetry=(0, 0, 1), num_freqs=3, visualize=False,
             run_time_factor=50):
    import tidy3d as td
    import numpy as np
    import matplotlib.pyplot as plt
    from . import extend

    lda0 = (wavl_max+wavl_min)/2
    lda_bw = (wavl_max-wavl_min)
    freq0 = td.C_0/lda0
    freqs = td.C_0/np.linspace(wavl_min, wavl_max, wavl_pts)
    fwidth = td.C_0*lda_bw/(lda0**2)  # 0.5*freq0

    polygons_device = extend.get_polygons(
        cell, layer_device, layer_pinrec, ly.dbu)
    devrec, sim_x, sim_y, center_x, center_y = extend.get_devrec(
        cell, layer_devrec, ly.dbu)
    ports = extend.get_ports(cell, layer_pinrec, ly.dbu)
    structures = make_structures(polygons_device, devrec, thick_dev,
                                 thick_sub, thick_super, mat_dev, mat_sub,
                                 mat_super, angle)

    # define source on a given port
    input_port = find_port(ports, in_port)
    source = make_source(
        ports[input_port], depth=z_span, thick_dev=thick_dev, freq0=freq0, num_freqs=num_freqs, fwidth=fwidth)
    # define monitors|
    monitors = make_monitors(ports, thick_dev, z_span=z_span, freqs=freqs)

    # simulation domain size (in microns)
    sim_size = [sim_x, sim_y, z_span]

    run_time = run_time_factor*max(sim_size)/td.C_0  # 85/fwidth  # sim. time in secs

    # initialize the simulation
    simulation = td.Simulation(
        size=sim_size,
        grid_spec=td.GridSpec.auto(min_steps_per_wvl=grid_cells_per_wvl),
        structures=structures,
        sources=[source],
        monitors=monitors,
        run_time=run_time,
        boundary_spec=boundary,
        center=(center_x, center_y, thick_dev/2),
        symmetry=symmetry,
    )

    if visualize:
        for m in simulation.monitors:
            m.help()

        source.source_time.plot(np.linspace(0, run_time, 1001))
        plt.show()

        # visualize geometry
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
        simulation.plot(z=thick_dev/2, ax=ax1)
        simulation.plot(x=0., ax=ax2)
        ax2.set_xlim([-sim_y/2, sim_y/2])
        plt.show()
    return simulation



def visualize_results(sim_data, sim):
    import matplotlib.pyplot as plt
    import numpy as np

    def get_directions(ports):
        directions = []
        for p in ports:
            if p.direction in [0, 90]:
                directions.append("+")
            else:
                directions.append("-")
        return tuple(directions)

    def get_port_name(port):
        index = ""
        if type(port) is not str:
            for char in port.name:
                if char.isdigit():
                    index += char
        else:
            for char in port:
                if char.isdigit():
                    index += char
        return int(index)

    def measure_transmission(sim_data, ports, num_ports, sim):
        """Constructs a "row" of the scattering matrix when sourced from top left port"""
        input_amp = sim_data[sim.in_port.name].amps.sel(direction="+")
        amps = np.zeros((num_ports, sim.wavl_pts), dtype=complex)
        directions = get_directions(ports)
        for i, (monitor, direction) in enumerate(
            zip(sim_data.simulation.monitors[:num_ports], directions)
        ):
            amp = sim_data[monitor.name].amps.sel(direction=direction)
            amp_normalized = amp / input_amp
            amps[i] = np.squeeze(amp_normalized.values)

        return amps

    def get_field_monitor_z(sim_data):
        for i in sim_data.simulation.monitors:
            if i.type == "FieldMonitor":
                return i.center[2]


    amps_arms = measure_transmission(sim_data, sim.device.ports, np.size(sim.device.ports), sim)
    print("mode amplitudes in each port: \n")
    fig, ax = plt.subplots(1, 1)
    wavl = np.linspace(sim.wavl_min, sim.wavl_max, sim.wavl_pts)
    ax.set_xlabel("Wavelength [microns]")
    ax.set_ylabel("Transmission [dB]")
    for amp, monitor in zip(amps_arms, sim_data.simulation.monitors[:]):
        print(f'\tmonitor     = "{monitor.name}"')
        plt.plot(
            wavl,
            [10 * np.log10(abs(i) ** 2) for i in amp],
            label=f"S{get_port_name(monitor.name)}{get_port_name(sim.in_port)}",
        )
        print(f"\tamplitude^2 = {[abs(i)**2 for i in amp]}")
        print(f"\tphase       = {[np.angle(i)**2 for i in amp]} (rad)\n")
    fig.legend()


    for i in sim_data.simulation.monitors:
        if i.type == "FieldMonitor":
            fig, ax = plt.subplots(1, 1, figsize=(16, 3))
            sim_data.plot_field(
                "field",
                "Ey",
                z=get_field_monitor_z(sim_data),
                freq=td.C_0 / ((sim.wavl_max + sim.wavl_min) / 2),
                ax=ax,
            )
            plt.show()
