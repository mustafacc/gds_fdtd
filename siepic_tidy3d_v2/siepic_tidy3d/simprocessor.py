"""
SiEPIC-Tidy3D integration toolbox.

Tidy3D simulation processing module.
@author: Mustafa Hammood, 2023
"""
import tidy3d as td


def make_source(
    port, width=3, depth=2, freq0=2e14, num_freqs=5, fwidth=1e13, buffer=0.1
):
    import tidy3d as td

    if port.direction == 0:
        x_buffer = -buffer
        y_buffer = 0
        size = [0, width, depth]
    elif port.direction == 180:
        x_buffer = buffer
        y_buffer = 0
        size = [0, width, depth]
    elif port.direction == 90:
        x_buffer = 0
        y_buffer = -buffer
        size = [width, 0, depth]
    elif port.direction == 270:
        x_buffer = 0
        y_buffer = buffer
        size = [width, 0, depth]
    if port.direction in [180, 270]:
        direction = "+"
    else:
        direction = "-"
    msource = td.ModeSource(
        center=[port.x + x_buffer, port.y + y_buffer, port.z],
        size=size,
        direction=direction,
        source_time=td.GaussianPulse(freq0=freq0, fwidth=fwidth),
        mode_spec=td.ModeSpec(),
        mode_index=0,
        num_freqs=num_freqs,
    )
    return msource


def make_structures(device, buffer = 2):
    import tidy3d as td
    import numpy as np

    structures = []
    for s in device.structures:
        if type(s) == list:
            for i in s:
                if i.z_span<0:
                    bounds = (i.z_span, i.z_base)
                else:
                    bounds = (i.z_base, i.z_span)
                structures.append(
                    td.Structure(
                        geometry=td.PolySlab(
                            vertices=i.polygon,
                            slab_bounds=bounds,
                            axis=2,
                            sidewall_angle=(90 - i.sidewall_angle) * (np.pi / 180),
                        ),
                        medium=i.material,
                    )
                )
        else:
            if s.z_span<0:
                bounds = (s.z_span, s.z_base)
            else:
                bounds = (s.z_base, s.z_span)
            structures.append(
                td.Structure(
                    geometry=td.PolySlab(
                        vertices=s.polygon,
                        slab_bounds=bounds,
                        axis=2,
                        sidewall_angle=(90 - s.sidewall_angle) * (np.pi / 180),
                    ),
                    medium=s.material,
                )
            )

    # extend ports beyond sim region
    for p in device.ports:
        if p.direction == 0:
            pts = [
                [p.center[0], p.center[1]+p.width/2],
                [p.center[0]+buffer, p.center[1]+p.width/2],
                [p.center[0]+buffer, p.center[1]-p.width/2],
                [p.center[0], p.center[1]-p.width/2]
                ]
        elif p.direction == 180:
            pts = [
                [p.center[0], p.center[1]+p.width/2],
                [p.center[0]-buffer, p.center[1]+p.width/2],
                [p.center[0]-buffer, p.center[1]-p.width/2],
                [p.center[0], p.center[1]-p.width/2]
                ]
        elif p.direction == 90:
            pts = [
                [p.center[0]-p.width/2, p.center[1]],
                [p.center[0]-p.width/2, p.center[1]+buffer],
                [p.center[0]+p.width/2, p.center[1]+buffer],
                [p.center[0]+p.width/2, p.center[1]]
                ]
        elif p.direction == 270:
            pts = [
                [p.center[0]-p.width/2, p.center[1]],
                [p.center[0]-p.width/2, p.center[1]-buffer],
                [p.center[0]+p.width/2, p.center[1]-buffer],
                [p.center[0]+p.width/2, p.center[1]]
                ]
        
        structures.append(
            td.Structure(
                geometry=td.PolySlab(
                    vertices=pts,
                    slab_bounds=(p.center[2]-p.height/2, p.center[2]+p.height/2),
                    axis=2,
                    sidewall_angle=(90 - device.structures[0].sidewall_angle) * (np.pi / 180),
                ),
                medium=s[0].material,
            )
        )
    return structures


def make_port_monitor(port, freqs=2e14, buffer=0.2, depth=2, width=3):
    """Create monitors for a given list of ports."""
    import tidy3d as td

    if port.direction == 0:
        x_buffer = -buffer
        y_buffer = 0
        size = [0, width, depth]
    elif port.direction == 180:
        x_buffer = buffer
        y_buffer = 0
        size = [0, width, depth]
    elif port.direction == 90:
        x_buffer = 0
        y_buffer = -buffer
        size = [width, 0, depth]
    elif port.direction == 270:
        x_buffer = 0
        y_buffer = buffer
        size = [width, 0, depth]
    # mode monitors
    monitors = td.ModeMonitor(
            center=[port.x + x_buffer, port.y + y_buffer, port.z],
            size=size,
            freqs=freqs,
            mode_spec=td.ModeSpec(),
            name=port.name,
        )


    return monitors

def make_field_monitor(device, freqs=2e14, z_center=None):
    import numpy as np
    # identify a device field z_center if None
    if z_center is None:
        z_center = []
        for s in device.structures:
            if type(s) == list:  # i identify non sub/superstrate if s is a list
                s = s[0]
                z_center.append(s.z_base+s.z_span/2)
        z_center = np.average(z_center)
    return td.FieldMonitor(
        center=[0, 0, z_center],
        size=[td.inf, td.inf, 0],
        freqs=freqs,
        name="field",
    )

def make_sim(
    device,
    wavl_min=1.45,
    wavl_max=1.65,
    wavl_pts=101,
    width_ports=3,
    depth_ports=2,
    symmetry=(0, 0, 0),
    num_freqs=5,
    in_port=None,
    boundary=td.BoundarySpec.all_sides(boundary=td.PML()),
    grid_cells_per_wvl=15,
    run_time_factor=50,
    z_span=None,
    field_monitor=False,
    visualize=True,
):
    import tidy3d as td
    import numpy as np
    import matplotlib.pyplot as plt

    if in_port is None:
        in_port = device.ports[0]

    lda0 = (wavl_max + wavl_min) / 2
    lda_bw = wavl_max - wavl_min
    freq0 = td.C_0 / lda0
    freqs = td.C_0 / np.linspace(wavl_min, wavl_max, wavl_pts)
    fwidth = 0.5 * (np.max(freqs) - np.min(freqs))

    # define structures from device
    structures = make_structures(device)

    # define source on a given port
    source = make_source(
        device.ports[0],
        depth= depth_ports,
        width= width_ports,
        freq0=freq0,
        num_freqs=num_freqs,
        fwidth=fwidth,
    )

    # define monitors
    monitors = []
    for p in device.ports:
        monitors.append(make_port_monitor(
            p,
            freqs=freqs,
            depth= depth_ports,
            width= width_ports,
        ))

    if field_monitor:
        monitors.append(make_field_monitor(device, freqs=freqs))
    # simulation domain size (in microns)
    if z_span == None:
        sim_size = [device.bounds.x_span, device.bounds.y_span, device.bounds.z_span]
    else:
        sim_size = [device.bounds.x_span, device.bounds.y_span, z_span]
    run_time = (
        run_time_factor * max(sim_size) / td.C_0
    )  # 85/fwidth  # sim. time in secs

    # initialize the simulation
    simulation = td.Simulation(
        size=sim_size,
        grid_spec=td.GridSpec.auto(min_steps_per_wvl=grid_cells_per_wvl, wavelength=lda0),
        structures=structures,
        sources=[source],
        monitors=monitors,
        run_time=run_time,
        boundary_spec=boundary,
        center=(device.bounds.x_center, device.bounds.y_center, device.bounds.z_center),
        symmetry=symmetry,
    )

    if visualize:
        for m in simulation.monitors:
            m.help()

        source.source_time.plot(np.linspace(0, run_time, 1001))
        plt.show()

        # visualize geometry
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
        simulation.plot(z=device.bounds.z_center, ax=ax1)
        simulation.plot(x=0.0, ax=ax2)
        ax2.set_xlim([-device.bounds.x_span / 2, device.bounds.y_span / 2])
        plt.show()
    return simulation