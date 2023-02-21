#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SiEPIC-Tidy3D integration toolbox.

Layout processing module.
@author: Mustafa Hammood, 2023
"""


class layout:
    def __init__(self, name, ly, cell):
        self.name = name
        self.ly = ly
        self.cell = cell
        return


class port:
    def __init__(self, name, center, width, height, direction):
        self.name = name
        self.center = center
        self.width = width
        self.height = height
        self.direction = direction
        return


class structure:
    def __init__(self, name, polygon, thickness, z_base, material, sidewall_angle=90):
        return


class component:
    def __init__(self, name, structures, ports):
        return


def load_layout(fname):
    import klayout.db as pya
    ly = pya.Layout()
    ly.read(fname)
    if ly.cells() > 1:
        ValueError(
            'More than one top cell found, ensure only 1 top cell exists.')
    else:
        cell = ly.top_cell()
        name = cell.name
    return layout(name, ly, cell)


def load_ports(layout, layer=[1, 10], z_center=0):
    import klayout.db as pya

    def get_direction(path):
        """Determine orientation of a pin path."""
        if path.points > 2:
            return ValueError('Number of points in a pin path are > 2.')
        p = path.each_point()
        p1 = p.__next__()
        p2 = p.__next__()
        if p1.x == p2.x:
            if p1.y > p2.y:  # north/south
                return 270
            else:
                return 90
        elif p1.y == p2.y:  # east/west
            if p1.x > p2.x:
                return 180
            else:
                return 0

    def get_center(path, dbu):
        """Determine center of a pin path."""
        p = path.each_point()
        p1 = p.__next__()
        p2 = p.__next__()
        direction = get_direction(path)
        if direction in [0, 180]:
            x = dbu*(p1.x + p2.x)/2
            y = dbu*p1.y
        elif direction in [90, 270]:
            x = dbu*p1.x
            y = dbu*(p1.y + p2.y)/2
        return x, y

    def get_name(c, x, y, dbu):
        s = c.begin_shapes_rec(layer)
        while not(s.at_end()):
            if s.shape().is_text():
                label_x = s.shape().text.x*dbu
                label_y = s.shape().text.y*dbu
                if label_x == x and label_y == y:
                    return s.shape().text.string
            s.next()

    ports = []
    s = layout.cell.begin_shapes_rec(layout.ly.layer(layer[0], layer[1]))
    while not(s.at_end()):
        if s.shape().is_path():
            
            width = s.shape().path_dwidth
            direction = get_direction(s.shape().path)
            center = get_center(s.shape().path, layout.ly.dbu)+[z_center]
            name = get_name(layout.cell, p.center, layout.ly.dbu)
            p = port(name=name, center=center, width=width, height=)
        s.next()
    return ports
