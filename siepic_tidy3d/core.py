"""
SiEPIC-Tidy3D integration toolbox.

Core objects module.
@author: Mustafa Hammood, 2023
"""
import tidy3d as td


def is_point_inside_polygon(point, polygon_points):
    """Identify if a point inside a polygon using Shapely.

    Args:
        point (list): Point for test [x, y]
        polygon_points (list): List of points defining a polygon [[x1, y1], [x2,y2], ..]

    Returns:
        bool: Test result.
    """
    from shapely.geometry import Point
    from shapely.geometry.polygon import Polygon

    # Create a Shapely Point object for the given coordinate
    point = Point(point)

    # Create a Shapely Polygon object from the list of polygon points
    polygon = Polygon(polygon_points)

    # Check if the point is inside the polygon
    return point.within(polygon) or polygon.touches(point)


class layout:
    def __init__(self, name, ly, cell):
        self.name = name
        self.ly = ly
        self.cell = cell

    @property
    def dbu(self):
        return self.ly.dbu


class port:
    def __init__(self, name, center, width, direction):
        self.name = name
        self.center = center
        self.width = width
        self.direction = direction
        # initialize height as none
        # will be assigned upon component __init__
        self.height = None
        self.material = None

    @property
    def x(self):
        return self.center[0]

    @property
    def y(self):
        return self.center[1]

    @property
    def z(self):
        return self.center[2]

    @property
    def idx(self):
        """index of the port, extracted from name."""
        return int("".join(char for char in reversed(self.name) if char.isdigit()))


class structure:
    def __init__(self, name, polygon, z_base, z_span, material, sidewall_angle=90):
        self.name = name
        self.polygon = polygon
        self.z_base = z_base
        self.z_span = z_span
        self.material = material
        self.sidewall_angle = sidewall_angle


class region:
    def __init__(self, vertices, z_center, z_span):
        self.vertices = vertices
        self.z_center = z_center
        self.z_span = z_span

    @property
    def x(self):
        return [i[0] for i in self.vertices]

    @property
    def y(self):
        return [i[1] for i in self.vertices]

    @property
    def x_span(self):
        return abs(min(self.x) - max(self.x))

    @property
    def y_span(self):
        return abs(min(self.y) - max(self.y))

    @property
    def x_center(self):
        return (min(self.x) + max(self.x)) / 2

    @property
    def y_center(self):
        return (min(self.y) + max(self.y)) / 2


class component:
    def __init__(self, name, structures, ports, bounds):
        self.name = name
        self.structures = structures
        self.ports = ports
        self.bounds = bounds
        self.get_port_z()  # initialize ports z center and z span

    def get_port_z(self):
        # iterate through each port
        for p in self.ports:
            # check if port location is within any structure
            for s in self.structures:
                # hack: if s is a list then it's not a box/clad region, find a better way to identify this..
                if type(s) == list:
                    if is_point_inside_polygon(p.center[:2], s[0].polygon):
                        p.center[2] = s[0].z_base + s[0].z_span/2
                        p.height = s[0].z_span
                        p.material = s[0].material
        return


class Simulation:
    def __init__(
        self, in_port, device, wavl_min=1.45, wavl_max=1.65, wavl_pts=101, sim=None
    ):
        self.in_port = in_port
        self.device = device
        self.wavl_min = wavl_min
        self.wavl_max = wavl_max
        self.wavl_pts = wavl_pts
        self.sim = sim
        self.results = None

    def visualize_results(self):
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
            return [int(i) for i in port if i.isdigit()][0]

        def measure_transmission(ports, num_ports):
            """Constructs a "row" of the scattering matrix when sourced from top left port"""
            input_amp = self.results[self.in_port.name].amps.sel(direction="+")
            amps = np.zeros((num_ports, self.wavl_pts), dtype=complex)
            directions = get_directions(ports)
            for i, (monitor, direction) in enumerate(
                zip(self.results.simulation.monitors[:num_ports], directions)
            ):
                amp = self.results[monitor.name].amps.sel(direction=direction)
                amp_normalized = amp / input_amp
                amps[i] = np.squeeze(amp_normalized.values)

            return amps

        def get_field_monitor_z():
            for i in self.results.simulation.monitors:
                if i.type == "FieldMonitor":
                    return i.center[2]

        ports = self.device.ports
        amps_arms = measure_transmission(ports, np.size(ports))
        print("mode amplitudes in each port: \n")
        fig, ax = plt.subplots(1, 1)
        wavl = np.linspace(self.wavl_min, self.wavl_max, self.wavl_pts)
        ax.set_xlabel("Wavelength [microns]")
        ax.set_ylabel("Transmission [dB]")
        for amp, monitor in zip(amps_arms, self.results.simulation.monitors):
            print(f'\tmonitor     = "{monitor.name}"')
            plt.plot(
                wavl,
                [10 * np.log10(abs(i) ** 2) for i in amp],
                label=f"S{self.in_port.idx}{get_port_name(monitor.name)}",
            )
            print(f"\tamplitude^2 = {[abs(i)**2 for i in amp]}")
            print(f"\tphase       = {[np.angle(i)**2 for i in amp]} (rad)\n")
        fig.legend()

        fig, ax = plt.subplots(1, 1, figsize=(16, 3))
        self.results.plot_field(
            "field",
            "Ey",
            z=get_field_monitor_z(self.results),
            freq=td.C_0 / ((self.wavl_max + self.wavl_min) / 2),
            ax=ax,
        )
        plt.show()
