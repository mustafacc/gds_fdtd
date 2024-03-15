"""
GDS_Tidy3D integration toolbox.

Core objects module.
@author: Mustafa Hammood, 2024
"""

import tidy3d as td
import logging
import os


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
    def __init__(
        self, name: str, center: list[float, float], width: float, direction: float
    ):
        self.name = name
        self.center = center
        self.width = width
        self.direction = direction
        # initialize height as none
        # will be assigned upon component __init__
        # TODO: feels like a better way to do this..
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

    def polygon_extension(self, buffer: float = 2.0):
        if self.direction == 0:
            return [
                [self.center[0], self.center[1] + self.width / 2],
                [self.center[0] + buffer, self.center[1] + self.width / 2],
                [self.center[0] + buffer, self.center[1] - self.width / 2],
                [self.center[0], self.center[1] - self.width / 2],
            ]
        elif self.direction == 180:
            return [
                [self.center[0], self.center[1] + self.width / 2],
                [self.center[0] - buffer, self.center[1] + self.width / 2],
                [self.center[0] - buffer, self.center[1] - self.width / 2],
                [self.center[0], self.center[1] - self.width / 2],
            ]
        elif self.direction == 90:
            return [
                [self.center[0] - self.width / 2, self.center[1]],
                [self.center[0] - self.width / 2, self.center[1] + buffer],
                [self.center[0] + self.width / 2, self.center[1] + buffer],
                [self.center[0] + self.width / 2, self.center[1]],
            ]
        elif self.direction == 270:
            return [
                [self.center[0] - self.width / 2, self.center[1]],
                [self.center[0] - self.width / 2, self.center[1] - buffer],
                [self.center[0] + self.width / 2, self.center[1] - buffer],
                [self.center[0] + self.width / 2, self.center[1]],
            ]


class structure:
    def __init__(
        self,
        name: str,
        polygon: list[list[float, float]],
        z_base: float,
        z_span: float,
        material: str,
        sidewall_angle: float = 90.0,
    ):
        self.name = name
        self.polygon = polygon  # polygon should be in the form of list of list of 2 pts, i.e. [[0,0],[0,1],[1,1]]
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

    @property
    def x_min(self):
        return min(self.x)

    @property
    def x_max(self):
        return max(self.x)

    @property
    def y_min(self):
        return min(self.y)

    @property
    def y_max(self):
        return max(self.y)


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
                # TODO: hack: if s is a list then it's not a box/clad region, find a better way to identify this..
                if type(s) == list:
                    for poly in s:
                        if is_point_inside_polygon(p.center[:2], poly.polygon):
                            p.center[2] = s[0].z_base + s[0].z_span / 2
                            p.height = s[0].z_span
                            p.material = s[0].material
            if p.height == None:
                logging.warning(f"Cannot find height for port {p.name}")
        return


class Simulation:
    def __init__(
        self, in_port, device, wavl_min=1.45, wavl_max=1.65, wavl_pts=101, sim_jobs=None
    ):
        self.in_port = in_port
        self.device = device
        self.wavl_min = wavl_min
        self.wavl_max = wavl_max
        self.wavl_pts = wavl_pts
        self.sim_jobs = sim_jobs
        self.results = None

    def upload(self):
        from tidy3d import web

        # divide between job and sim, how to attach them?

        for sim_job in self.sim_jobs:
            sim = sim_job["sim"]
            name = sim_job["name"]
            sim_job["job"] = web.Job(simulation=sim, task_name=name)

    def execute(self):
        import numpy as np

        def get_directions(ports):
            directions = []
            for p in ports:
                if p.direction in [0, 90]:
                    directions.append("+")
                else:
                    directions.append("-")
            return tuple(directions)

        def get_source_direction(port):
            if port.direction in [0, 90]:
                return "-"
            else:
                return "+"

        def get_port_name(port):
            return [int(i) for i in port if i.isdigit()][0]

        def measure_transmission(
            in_port: port, in_mode_idx: int, out_mode_idx: int
        ):
            """
            Constructs a "row" of the scattering matrix.
            """
            num_ports = np.size(self.device.ports)
            input_amp = self.results[in_port.name].amps.sel(
                direction=get_source_direction(in_port),
                mode_index=in_mode_idx,
            )
            amps = np.zeros((num_ports, self.wavl_pts), dtype=complex)
            directions = get_directions(self.device.ports)
            for i, (monitor, direction) in enumerate(
                zip(self.results.simulation.monitors[:num_ports], directions)
            ):
                amp = self.results[monitor.name].amps.sel(
                    direction=direction, mode_index=out_mode_idx
                )
                amp_normalized = amp / input_amp
                amps[i] = np.squeeze(amp_normalized.values)

            return amps

        self.s_parameters = s_parameters()

        for sim_job in self.sim_jobs:
            if not os.path.exists(self.device.name):
                os.makedirs(self.device.name)
            self.results = sim_job["job"].run(
                path=os.path.join(self.device.name, f"{sim_job['name']}.hdf5")
            )
            for mode in range(sim_job["num_modes"]):
                amps_arms = measure_transmission(
                    in_port=sim_job["in_port"],
                    in_mode_idx=sim_job["source"].mode_index,
                    out_mode_idx=mode,
                )

                logging.info("Mode amplitudes in each port: \n")
                wavl = np.linspace(self.wavl_min, self.wavl_max, self.wavl_pts)
                for amp, monitor in zip(amps_arms, self.results.simulation.monitors):
                    logging.info(f'\tmonitor     = "{monitor.name}"')
                    logging.info(f"\tamplitude^2 = {[abs(i)**2 for i in amp]}")
                    logging.info(
                        f"\tphase       = {[np.angle(i)**2 for i in amp]} (rad)\n"
                    )

                    self.s_parameters.add_param(
                        sparam(
                            idx_in=sim_job["in_port"].idx,
                            idx_out=get_port_name(monitor.name),
                            mode_in=sim_job["source"].mode_index,
                            mode_out=mode,
                            freq=td.C_0 / wavl,
                            s=amp,
                        )
                    )

    def visualize_results(self):
        import matplotlib.pyplot as plt

        self.s_parameters.plot()

        try:
            fig, ax = plt.subplots(1, 1, figsize=(16, 3))
            self.results.plot_field(
                "field",
                "Ey",
                freq=td.C_0 / ((self.wavl_max + self.wavl_min) / 2),
                ax=ax,
            )
            fig.show()
        except:
            return


class s_parameters:
    def __init__(self, entries=None):
        if entries is None:
            self._entries = []
        else:
            self._entries = entries
        return

    def add_param(self, sparam):
        self._entries.append(sparam)

    def entries_in_mode(self, mode_in=0, mode_out=0):
        entries = []
        for s in self._entries:
            if s.mode_in == mode_in and s.mode_out == mode_out:
                entries.append(s)
        return entries

    def entries_in_ports(self, input_entries=None, idx_in=0, idx_out=0):
        entries = []
        if input_entries == None:
            input_entries = self._entries

        for s in input_entries:
            if s.idx_in == idx_in and s.idx_out == idx_out:
                entries.append(s)
        return entries

    def plot(self):
        import matplotlib.pyplot as plt
        import numpy as np

        fig, ax = plt.subplots(1, 1)
        ax.set_xlabel("Wavelength [microns]")
        ax.set_ylabel("Transmission [dB]")
        for i in self._entries:
            logging.info("Mode amplitudes in each port: \n")
            mag = [10 * np.log10(abs(i) ** 2) for i in i.s]
            phase = [np.angle(i) ** 2 for i in i.s]
            ax.plot(td.C_0 / i.freq, mag, label=i.label)
        ax.legend()
        return fig, ax


class sparam:
    def __init__(self, idx_in, idx_out, mode_in, mode_out, freq, s):
        self.idx_in = idx_in
        self.idx_out = idx_out
        self.mode_in = mode_in
        self.mode_out = mode_out
        self.freq = freq
        self.s = s

    @property
    def label(self):
        return f"S{self.idx_out}{self.idx_in}_idx{self.mode_out}{self.mode_in}"


def parse_yaml_tech(file_path):
    import yaml

    with open(file_path, "r") as file:
        data = yaml.safe_load(file)

    technology = data.get("technology", {})
    parsed_data = {
        "name": technology.get("name", "Unknown"),
        "substrate": [],
        "superstrate": [],
        "pinrec": [],
        "devrec": [],
        "device": [],
    }

    # Parsing substrate layer
    substrate = technology.get("substrate", {})
    parsed_data["substrate"].append(
        {
            "z_base": substrate.get("z_base"),
            "z_span": substrate.get("z_span"),
            "material": substrate.get("material"),
        }
    )

    # Parsing superstrate layer
    superstrate = technology.get("superstrate", {})
    parsed_data["superstrate"].append(
        {
            "z_base": superstrate.get("z_base"),
            "z_span": superstrate.get("z_span"),
            "material": superstrate.get("material"),
        }
    )

    # Parsing pinrec layers
    parsed_data["pinrec"] = [
        {"layer": pinrec.get("layer")} for pinrec in technology.get("pinrec", [])
    ]

    # Parsing devrec layers
    parsed_data["devrec"] = [
        {"layer": devrec.get("layer")} for devrec in technology.get("devrec", [])
    ]

    # Parsing device layers
    parsed_data["device"] = [
        {
            "layer": device.get("layer"),
            "z_base": device.get("z_base"),
            "z_span": device.get("z_span"),
            "material": device.get("material"),
            "sidewall_angle": device.get("sidewall_angle"),
        }
        for device in technology.get("device", [])
    ]

    return parsed_data
