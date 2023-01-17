#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KLayout-API geometry building assisting module.

@author: Mustafa Hammood, 2023
"""


def to_dbu(f, dbu):
    """Convert numerical value to dbu-valued integer."""
    return int(round(float(f)/dbu))


def pts_per_circle(r, dbu=0.001):
    """
    Calculate the number of points per circle. Adopted from SiEPIC-Tools.

    Parameters
    ----------
    r : Float
        Radius of the circle (in microns).
    dbu : Float, optional
        Layout's database unit (in microns). The default is 0.001 (1 nm)

    Returns
    -------
    int: number of points in the circle.

    """
    from math import pi, acos, ceil
    err = dbu/2
    return int(ceil(pi/acos(1-err/2))) if r > 1 else 10


def sbend(x, y, w, r, h, l, dbu=0.001, direction='east', flip=False, verbose=False):
    """
    Generate an S-bend polygon. Adopted from SiEPIC-Tools, LukasC.

    Parameters
    ----------
    x : Float
        X point of the start of the S-bend.
    y : Float
        Y point of the start of the S-bend.
    w : Float
        Width of the S-bend polygon.
    r : Float
        Bend radius.
    h : Float
        Height offset of the S-bend.
    l : Float
        Desired length of the S-bend.
    dbu : Float, optional
        Layout's database unit (in microns). The default is 0.001 (1 nm)
    direction : String, optional
        Direction of the S-bend. The default is 'east'. Options include:
            'east'
            'west'
            'north'
            'south'
    flip : Boolean, optional
        Mirror the S-bend axis. The default is False.
    verbose : Boolean, optional
        Verbose messages flag for debugging. The default is False.

    Returns
    -------
    shape_sbend : KLayout.db polygon object
        Generated polygon object of the S-bend

    """
    import klayout.db as pya
    from math import pi, acos, sin, cos
    start_x = to_dbu(x, dbu)
    start_y = to_dbu(y, dbu)
    w = to_dbu(w, dbu)
    r = to_dbu(r, dbu)
    h = to_dbu(h, dbu)
    l = to_dbu(l, dbu)

    theta = acos(float(r-abs(h/2))/r)*180/pi
    x = int(2*r*sin(theta/180.0*pi))
    straight_l = int((l-x)/2)

    if straight_l < 0:
        if verbose:
            print(f"Warning: S-bend too short. Expected length: {straight_l}")
        l = x
        straight_l = 0
    else:
        circle_fraction = abs(theta)/360.0
        npts = int(pts_per_circle(r*circle_fraction))
        if npts == 0:
            npts = 1
        da = 2*pi/npts*circle_fraction  # increment, in rads
        x1 = straight_l
        x2 = l-straight_l
        if h > 0:
            y1 = r
            y2 = h-r
            theta_start1 = 270
            theta_start2 = 90
            pts = []
            th1 = theta_start1/360.0*2*pi
            th2 = theta_start2/360.0*2*pi
            pts.append(pya.Point.from_dpoint(pya.DPoint(0, w/2)))
            pts.append(pya.Point.from_dpoint(pya.DPoint(0, -w/2)))
            for i in range(0, npts+1):  # lower left
                pts.append(pya.Point.from_dpoint(
                    pya.DPoint((x1+(r+w/2)*cos(i*da+th1))/1,
                               (y1+(r+w/2)*sin(i*da+th1))/1)))
            for i in range(npts, -1, -1):  # lower right
                pts.append(pya.Point.from_dpoint(
                    pya.DPoint((x2+(r-w/2)*cos(i*da+th2))/1,
                               (y2+(r-w/2)*sin(i*da+th2))/1)))
            pts.append(pya.Point.from_dpoint(pya.DPoint(l, h-w/2)))
            pts.append(pya.Point.from_dpoint(pya.DPoint(l, h+w/2)))
            for i in range(0, npts+1):  # upper right
                pts.append(pya.Point.from_dpoint(pya.DPoint(
                    (x2+(r+w/2)*cos(i*da+th2))/1, (y2+(r+w/2)*sin(i*da+th2))/1)))
            for i in range(npts, -1, -1):  # upper left
                pts.append(pya.Point.from_dpoint(pya.DPoint(
                    (x1+(r-w/2)*cos(i*da+th1))/1, (y1+(r-w/2)*sin(i*da+th1))/1)))
        trans = pya.Trans(start_x, start_y)

        if direction == 'east':
            shape_bend = pya.Polygon(pts).transformed(trans.R0)
        elif direction == 'west':
            shape_bend = pya.Polygon(pts).transformed(trans.R180)
        elif direction == 'north':
            shape_bend = pya.Polygon(pts).transformed(trans.R90)
        elif direction == 'south':
            shape_bend = pya.Polygon(pts).transformed(trans.R270)
        else:
            raise ValueError(
                'Invalid direction! Valid options: east, west, north, south')
        if flip:
            shape_bend = shape_bend.transformed(trans.M0)
        shape_bend = shape_bend.transformed(trans)
        return shape_bend


def make_pin(cell, name, center, w, layer, direction, pin_length=10, verbose=False):
    """
    Makes a pin that SiEPIC-Tools will recognize.

    Parameters
    ----------
    cell : klayout.db (pya) Cell type
        Which cell to draw the path in.
    name : String
        Name of the pin (text label).
    center : List [x (int), y(int)]
        Center location of the pin.
    w : Int
        Width of the pin.
    layer : klayout.db (pya) layout.layer() type
        Layer to place the pin object into.
    direction : Int
        Direction of the pin. Valid options:
            0: right
            90: up
            180: left
            270: down
    verbose : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    None.

    """
    '''
    Makes a pin that SiEPIC-Tools will recognize
    cell: which cell to draw it in
    name: text label for the pin
    center: location, int [x,y]
    w: pin width
    layer: layout.layer() type
    direction =
        0: right
        90: up
        180: left
        270: down

    Units: intput can be float for microns, or int for nm
    '''
    import klayout.db as pya
    from klayout.db import Point, DPoint
    import numpy
    dbu = cell.layout().dbu
    if type(w) == type(float()):
        w = to_dbu(w, dbu)
        if verbose:
            print(f'SiEPIC.utils.layout.make_pin: w converted to {w}')
    else:
        if verbose:
            print(f'SiEPIC.utils.layout.make_pin: w {w}')
    if type(center) == type(Point()) or type(center) == type(DPoint()):
        center = [center.x, center.y]
    if type(center[0]) == type(float()) or type(center[0]) == type(numpy.float64()):
        center[0] = to_dbu(center[0], dbu)
        center[1] = to_dbu(center[1], dbu)
        if verbose:
            print(
                f'SiEPIC.utils.layout.make_pin: center converted to {center}')
    else:
        if verbose:
            print(f'SiEPIC.utils.layout.make_pin: center {center}')

    direction = direction % 360
    if direction not in [0, 90, 180, 270]:
        raise Exception(
            'error in make_pin: direction (%s) must be one of [0, 90, 180, 270]' % direction)

    # text label
    t = pya.Trans(pya.Trans.R0, center[0], center[1])
    text = pya.Text(name, t)
    shape = cell.shapes(layer).insert(text)
    shape.text_dsize = float(w*dbu/2)
    shape.text_valign = 1

    if direction == 0:
        p1 = pya.Point(center[0]-pin_length/2, center[1])
        p2 = pya.Point(center[0]+pin_length/2, center[1])
        shape.text_halign = 2
    if direction == 90:
        p1 = pya.Point(center[0], center[1]-pin_length/2)
        p2 = pya.Point(center[0], center[1]+pin_length/2)
        shape.text_halign = 2
        shape.text_rot = 1
    if direction == 180:
        p1 = pya.Point(center[0]+pin_length/2, center[1])
        p2 = pya.Point(center[0]-pin_length/2, center[1])
        shape.text_halign = 3
    if direction == 270:
        p1 = pya.Point(center[0], center[1]+pin_length/2)
        p2 = pya.Point(center[0], center[1]-pin_length/2)
        shape.text_halign = 3
        shape.text_rot = 1

    pin = pya.Path([p1, p2], w)
    cell.shapes(layer).insert(pin)
