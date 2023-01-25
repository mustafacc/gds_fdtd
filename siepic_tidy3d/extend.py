#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GDS cell processing helper methods.

@author: Mustafa Hammood, 2023
"""


def get_ports(c, layer, dbu=0.001):
    """
    Get the ports of a cell.

    Parameters
    ----------
    cell : klayout.db (pya) Cell type
        Cell to extract the polygons from.
    layer : klayout.db (pya) layout.layer() type
        Layer to place the pin object into.
    dbu : Float, optional
        Layout's database unit (in microns). The default is 0.001 (1 nm)

    Returns
    -------
    Dictionary
        Dictionary containing id, width, direction, and center of each port.

    """
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

    ports = dict()
    s = c.begin_shapes_rec(layer)
    port_counter = 1
    while not(s.at_end()):
        if s.shape().is_path():
            ports[port_counter] = dict()
            ports[port_counter]['width'] = s.shape().path_dwidth
            ports[port_counter]['direction'] = get_direction(s.shape().path)
            ports[port_counter]['x'], ports[port_counter]['y'] = get_center(
                s.shape().path, dbu)
            ports[port_counter]['name'] = get_name(
                c, ports[port_counter]['x'], ports[port_counter]['y'], dbu)
        s.next()
        port_counter += 1
    return ports


def get_devrec(c, layer, dbu=0.001):
    """
    Get device bounds.

    Parameters
    ----------
    cell : klayout.db (pya) Cell type
        Cell to extract the polygons from.
    layer : klayout.db (pya) layout.layer() type
        Layer to place the pin object into.
    dbu : Float, optional
        Layout's database unit (in microns). The default is 0.001 (1 nm)

    Returns
    -------
    polygons_vertices : list [lists[x,y]]
        list of bounding box coordinates.
    sim_x : Float
        x-span of the device.
    sim_y : Float
        y-span of the device.

    """
    import klayout.db as pya
    iter1 = c.begin_shapes_rec(layer)
    # DevRec must be either a Box or a Polygon:
    if iter1.shape().is_box():
        box = iter1.shape().box.transformed(iter1.itrans())
        polygon = pya.Polygon(box)  # Save the component outline polygon
        DevRec_polygon = pya.Polygon(iter1.shape().box)
    if iter1.shape().is_polygon():
        polygon = iter1.shape().polygon.transformed(
            iter1.itrans())  # Save the component outline polygon
        DevRec_polygon = iter1.shape().polygon
    polygons_vertices = [[[vertex.x*dbu, vertex.y*dbu] for vertex in p.each_point()]
                         for p in [p.to_simple_polygon() for p in [DevRec_polygon]]][0]
    x = [i[0] for i in polygons_vertices]
    y = [i[1] for i in polygons_vertices]
    sim_x = abs(min(x)-max(x))
    sim_y = abs(min(y)-max(y))
    center_x = (min(x)+max(x))/2
    center_y = (min(y)+max(y))/2
    return polygons_vertices, sim_x, sim_y, center_x, center_y


def get_polygons(c, layer, layer_pinrec, dbu=0.001):
    """
    Extract polygons from a given cell on a given layer.

    Parameters
    ----------
    cell : klayout.db (pya) Cell type
        Cell to extract the polygons from.
    layer : klayout.db (pya) layout.layer() type
        Layer to place the pin object into.
    dbu : Float, optional
        Layout's database unit (in microns). The default is 0.001 (1 nm)

    Returns
    -------
    polygons_vertices : list [lists[x,y]]
        list of polygons from the cell.

    """
    def make_port_extension(c, layer, ports, dbu=0.001):
        import klayout.db as pya
        from . import geometry
        ext_out = 1 # extended structure beyond port
        ext_in = 0.25
        # create boxes at each port
        for p in ports:
            if ports[p]['direction'] == 0:
                p1 = pya.Point(geometry.to_dbu(
                    ports[p]['x']-ext_in, dbu), geometry.to_dbu(ports[p]['y'], dbu))
                p2 = pya.Point(geometry.to_dbu(
                    ports[p]['x']+ext_out, dbu), geometry.to_dbu(ports[p]['y'], dbu))
            elif ports[p]['direction'] == 180:
                p1 = pya.Point(geometry.to_dbu(
                    ports[p]['x']+ext_in, dbu), geometry.to_dbu(ports[p]['y'], dbu))
                p2 = pya.Point(geometry.to_dbu(
                    ports[p]['x']-ext_out, dbu), geometry.to_dbu(ports[p]['y'], dbu))
            elif ports[p]['direction'] == 90:
                p1 = pya.Point(geometry.to_dbu(
                    ports[p]['x'], dbu), geometry.to_dbu(ports[p]['y']-ext_in, dbu))
                p2 = pya.Point(geometry.to_dbu(ports[p]['x'], dbu), geometry.to_dbu(
                    ports[p]['y']+ext_out, dbu))
            elif ports[p]['direction'] == 270:
                p1 = pya.Point(geometry.to_dbu(
                    ports[p]['x'], dbu), geometry.to_dbu(ports[p]['y']+ext_in, dbu))
                p2 = pya.Point(geometry.to_dbu(ports[p]['x'], dbu), geometry.to_dbu(
                    ports[p]['y']-ext_out, dbu))
            pin = pya.Path([p1, p2], geometry.to_dbu(ports[p]['width'], dbu))
            c.shapes(layer).insert(pin)
        #mergeReg = pya.Region(c.begin_shapes_rec(layer))
        #mergeReg.merge()
        #c.clear(layer)
        #c.shapes(layer).insert(mergeReg)
        return

    import klayout.db as pya
    # create additional material to extend beyond device bounds (needed for convergance)
    ports = get_ports(c, layer_pinrec, dbu)
    make_port_extension(c, layer, ports, dbu)

    r = pya.Region()
    s = c.begin_shapes_rec(layer)
    while not(s.at_end()):
        if s.shape().is_polygon() or s.shape().is_box() or s.shape().is_path():
            r.insert(s.shape().polygon.transformed(s.itrans()))
        s.next()

    r.merge()
    polygons = [p for p in r.each_merged()]
    polygons_vertices = [[[vertex.x*dbu, vertex.y*dbu] for vertex in p.each_point()]
                         for p in [p.to_simple_polygon() for p in polygons]]

    return polygons_vertices
