"""
SiEPIC-Tidy3D integration toolbox.

Core objects module.
@author: Mustafa Hammood, 2023
"""


class layout:
    def __init__(self, name, ly, cell):
        self.name = name
        self.ly = ly
        self.cell = cell

    @property
    def dbu(self):
        return self.ly.dbu


class port:
    def __init__(self, name, center, width, height, direction):
        self.name = name
        self.center = center
        self.width = width
        self.height = height
        self.direction = direction

    @property
    def x(self):
        return self.center[0]

    @property
    def y(self):
        return self.center[1]

    @property
    def z(self):
        return self.center[2]


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
