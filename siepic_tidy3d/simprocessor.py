"""
SiEPIC-Tidy3D integration toolbox.

Tidy3D simulation processing module.
@author: Mustafa Hammood, 2023
"""
import tidy3d as td


def make_source(
    port, width=3, depth=2, freq0=2e14, num_freqs=5, fwidth=1e13, buffer=0.1
):
    """Create a simulation mode source on an input port.

    Args:
        port (port object): Port to add source to.
        width (int, optional): Source width. Defaults to 3 microns.
        depth (int, optional): Source depth. Defaults to 2 microns.
        freq0 (_type_, optional): Source's centre frequency. Defaults to 2e14 hz.
        num_freqs (int, optional): Frequency evaluation mode ports. Defaults to 5.
        fwidth (_type_, optional): Source's bandwidth. Defaults to 1e13 hz.
        buffer (float, optional): Distance between edge of simulation and source. Defaults to 0.1 microns.

    Returns:
        _type_: _description_
    """
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


def make_structures(device, buffer=2):
    """Create a tidy3d structure object from a device objcet.

    Args:
        device (device object): Device to create the structure from.
        buffer (int, optional): Extension of ports beyond simulation region . Defaults to 2 microns.

    Returns:
        list: list of structures generated from the device.
    """
    import tidy3d as td
    import numpy as np

    structures = []
    for s in device.structures:
        if type(s) == list:
            for i in s:
                if i.z_span < 0:
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
            if s.z_span < 0:
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
                medium=s[0].material,
            )
        )
    return structures


def make_port_monitor(port, freqs=2e14, buffer=0.2, depth=2, width=3):
    """
    Create mode monitor object for a given port.

    Args:
        port (port object): Port to create the monitor for.
        freqs (float, optional): Mode monitor's central frequency. Defaults to 2e14 hz.
        buffer (float, optional): Distance between monitor and port location. Defaults to 0.2 microns.
        depth (int, optional): Monitor's depth. Defaults to 2 microns.
        width (int, optional): Monitors width. Defaults to 3 microns.

    Returns:
        monitor: Generated ModeMonitor object.
    """

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
    # mode monitor
    monitor = td.ModeMonitor(
        center=[port.x + x_buffer, port.y + y_buffer, port.z],
        size=size,
        freqs=freqs,
        mode_spec=td.ModeSpec(),
        name=port.name,
    )

    return monitor


def make_field_monitor(device, freqs=2e14, axis="z", z_center=None):
    """Make a field monitor for an input device

    Args:
        device (device object): Device to create field monitor for.
        freqs (float, optional): _description_. Defaults to 2e14.
        z_center (float, optional): Z center for field monitor. Defaults to None.
        axis (string, optional): Field monitor's axis. Valid options are 'x', 'y', 'z' Defaults to 'z'.

    Returns:
        FieldMonitor: Generated Tidy3D field monitor object
    """
    import numpy as np

    # identify a device field z_center if None
    if z_center is None:
        z_center = []
        for s in device.structures:
            if type(s) == list:  # i identify non sub/superstrate if s is a list
                s = s[0]
                z_center.append(s.z_base + s.z_span / 2)
        center = np.average(z_center)
    return td.FieldMonitor(
        center=[0, 0, center],
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
    field_monitor_axis=None,
    visualize=True,
):
    """Generate a single port excitation simulation.

    Args:
        device (device object): Device to simulate.
        wavl_min (float, optional): Start wavelength. Defaults to 1.45 microns.
        wavl_max (float, optional): End wavelength. Defaults to 1.65 microns.
        wavl_pts (int, optional): Number of wavelength evaluation pts. Defaults to 101.
        width_ports (int, optional): Width of source and monitors. Defaults to 3 microns.
        depth_ports (int, optional): Depth of source and monitors. Defaults to 2 microns.
        symmetry (tuple, optional): Enforcing symmetry along axes. Defaults to (0, 0, 0).
        num_freqs (int, optional): Number of source's frequency mode evaluation pts. Defaults to 5 microns.
        in_port (port object, optional): Input port. Defaults to None.
        boundary (td.BonudarySpec object, optional): Configure boundary conditions. Defaults to td.BoundarySpec.all_sides(boundary=td.PML()).
        grid_cells_per_wvl (int, optional): Mesh settings, grid cells per wavelength. Defaults to 15.
        run_time_factor (int, optional): Runtime multiplier factor. Set larger if runtime is insufficient. Defaults to 50.
        z_span (float, optional): Simulation's depth. Defaults to None.
        field_monitor_axis (str, optional): Flag to create a field monitor. Options are 'x', 'y', 'z', or none. Defaults to None.
        visualize (bool, optional): Simulation visualization flag. Defaults to True.

    Returns:
        _type_: _description_
    """
    from .core import Simulation

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
        in_port,
        depth=depth_ports,
        width=width_ports,
        freq0=freq0,
        num_freqs=num_freqs,
        fwidth=fwidth,
    )

    # define monitors
    monitors = []
    for p in device.ports:
        monitors.append(
            make_port_monitor(
                p,
                freqs=freqs,
                depth=depth_ports,
                width=width_ports,
            )
        )

    if field_monitor_axis is not None:
        monitors.append(
            make_field_monitor(device, freqs=freqs, axis=field_monitor_axis)
        )
    # simulation domain size (in microns)
    if z_span == None:
        sim_size = [device.bounds.x_span, device.bounds.y_span, device.bounds.z_span]
    else:
        sim_size = [device.bounds.x_span, device.bounds.y_span, z_span]
    run_time = (
        run_time_factor * max(sim_size) / td.C_0
    )  # 85/fwidth  # sim. time in secs

    # initialize the simulation
    simulation = Simulation(
        in_port=in_port,
        wavl_max=wavl_max,
        wavl_min=wavl_min,
        wavl_pts=wavl_pts,
        device=device,
        sim=td.Simulation(
            size=sim_size,
            grid_spec=td.GridSpec.auto(
                min_steps_per_wvl=grid_cells_per_wvl, wavelength=lda0
            ),
            structures=structures,
            sources=[source],
            monitors=monitors,
            run_time=run_time,
            boundary_spec=boundary,
            center=(
                device.bounds.x_center,
                device.bounds.y_center,
                device.bounds.z_center,
            ),
            symmetry=symmetry,
        ),
    )

    if visualize:
        for m in simulation.sim.monitors:
            m.help()

        source.source_time.plot(np.linspace(0, run_time, 1001))
        plt.show()

        # visualize geometry
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
        simulation.sim.plot(z=device.bounds.z_center, ax=ax1)
        simulation.sim.plot(x=0.0, ax=ax2)
        ax2.set_xlim([-device.bounds.x_span / 2, device.bounds.y_span / 2])
        plt.show()
    return simulation


def visualize_results(sim_data, sim):
    import matplotlib.pyplot as plt
    import numpy as np

    def get_directions(ports, in_port, sim_data):
        directions = []
        for p in ports:
            if p.direction in [0, 90]:
                directions.append('-')
            else:
                directions.append('+')
        return tuple(directions)

    def get_port_name(port):
        return [int(i) for i in port if i.isdigit()][0]

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

    ports = sim.device.ports
    amps_arms = measure_transmission(sim_data, ports, np.size(ports), sim)
    print("mode amplitudes in each port: \n")
    fig, ax = plt.subplots(1, 1)
    wavl = np.linspace(sim.wavl_min, sim.wavl_max, sim.wavl_pts)
    ax.set_xlabel("Wavelength [microns]")
    ax.set_ylabel("Transmission [dB]")
    for amp, monitor in zip(amps_arms, sim_data.simulation.monitors[:-1]):
        print(f'\tmonitor     = "{monitor.name}"')
        plt.plot(
            wavl,
            [10 * np.log10(abs(i) ** 2) for i in amp],
            label=f"S{get_port_name(sim.in_port)}{get_port_name(monitor.name)}",
        )
        print(f"\tamplitude^2 = {[abs(i)**2 for i in amp]}")
        print(f"\tphase       = {[np.angle(i)**2 for i in amp]} (rad)\n")
    fig.legend()

    fig, ax = plt.subplots(1, 1, figsize=(16, 3))
    sim_data.plot_field(
        "field",
        "Ey",
        z=get_field_monitor_z(sim_data),
        freq=td.C_0 / ((sim.wavl_max + sim.wavl_min) / 2),
        ax=ax,
    )
    plt.show()
