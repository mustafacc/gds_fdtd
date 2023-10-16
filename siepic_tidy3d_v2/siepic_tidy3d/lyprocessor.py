"""
SiEPIC-Tidy3D integration toolbox.

Layout processing module.
@author: Mustafa Hammood, 2023
"""

from .core import layout, port, structure, region, component


def load_layout(fname):
    import klayout.db as pya

    ly = pya.Layout()
    ly.read(fname)
    if ly.cells() > 1:
        ValueError("More than one top cell found, ensure only 1 top cell exists.")
    else:
        cell = ly.top_cell()
        name = cell.name
    return layout(name, ly, cell)


def load_region(layout, layer=[68, 0], z_center=0, z_span=5):
    """
    Get device bounds.

    Parameters
    ----------
    layout : SiEPIC Tidy3d layout type
        layout to extract the polygons from.
    layer : klayout.db (pya) layout.layer() type
        Layer to place detect the devrec object from.

    Returns
    -------
    region : region object type
    """

    def get_kdb_layer(layer):
        return layout.ly.layer(layer[0], layer[1])

    import klayout.db as pya

    c = layout.cell
    dbu = layout.dbu
    layer = get_kdb_layer(layer)
    iter1 = c.begin_shapes_rec(layer)
    # DevRec must be either a Box or a Polygon:
    if iter1.shape().is_box():
        box = iter1.shape().box.transformed(iter1.itrans())
        polygon = pya.Polygon(box)  # Save the component outline polygon
        DevRec_polygon = pya.Polygon(iter1.shape().box)
    if iter1.shape().is_polygon():
        polygon = iter1.shape().polygon.transformed(
            iter1.itrans()
        )  # Save the component outline polygon
        DevRec_polygon = iter1.shape().polygon
    polygons_vertices = [
        [[vertex.x * dbu, vertex.y * dbu] for vertex in p.each_point()]
        for p in [p.to_simple_polygon() for p in [DevRec_polygon]]
    ][0]

    return region(vertices=polygons_vertices, z_center=z_center, z_span=z_span)


def load_structure(layout, name, layer, z_base, z_span, material, sidewall_angle=90):
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

    def get_kdb_layer(layer):
        return layout.ly.layer(layer[0], layer[1])

    import klayout.db as pya

    c = layout.cell
    dbu = layout.dbu
    layer = get_kdb_layer(layer)

    r = pya.Region()
    s = c.begin_shapes_rec(layer)
    while not (s.at_end()):
        if s.shape().is_polygon() or s.shape().is_box() or s.shape().is_path():
            r.insert(s.shape().polygon.transformed(s.itrans()))
        s.next()

    r.merge()
    polygons = [p for p in r.each_merged()]
    polygons_vertices = [
        [[vertex.x * dbu, vertex.y * dbu] for vertex in p.each_point()]
        for p in [p.to_simple_polygon() for p in polygons]
    ]
    structures = []
    for idx, s in enumerate(polygons_vertices):
        name = f"{name}_{idx}"
        structures.append(
            structure(
                name=name,
                polygon=s,
                z_base=z_base,
                z_span=z_span,
                material=material,
                sidewall_angle=sidewall_angle,
            )
        )
    return structures


def load_structure_from_bounds(bounds, name, z_base, z_span, material, extension=1):
    # TODO: incorporate extension to grow polygon vertices
    return structure(
        name=name,
        polygon=bounds.vertices,
        z_base=z_base,
        z_span=z_span,
        material=material,
    )


def load_ports(layout, layer=[1, 10], z_center=0, z_span=0.22):
    import klayout.db as pya

    def get_kdb_layer(layer):
        return layout.ly.layer(layer[0], layer[1])

    def get_direction(path):
        """Determine orientation of a pin path."""
        if path.points > 2:
            return ValueError("Number of points in a pin path are > 2.")
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
            x = dbu * (p1.x + p2.x) / 2
            y = dbu * p1.y
        elif direction in [90, 270]:
            x = dbu * p1.x
            y = dbu * (p1.y + p2.y) / 2
        return x, y

    def get_name(c, x, y, dbu):
        s = c.begin_shapes_rec(get_kdb_layer(layer))
        while not (s.at_end()):
            if s.shape().is_text():
                label_x = s.shape().text.x * dbu
                label_y = s.shape().text.y * dbu
                if label_x == x and label_y == y:
                    return s.shape().text.string
            s.next()

    ports = []
    s = layout.cell.begin_shapes_rec(get_kdb_layer(layer))
    while not (s.at_end()):
        if s.shape().is_path():
            width = s.shape().path_dwidth
            direction = get_direction(s.shape().path)
            center = list(get_center(s.shape().path, layout.ly.dbu)) + [z_center]
            name = get_name(layout.cell, center[0], center[1], layout.ly.dbu)
            ports.append(
                port(
                    name=name,
                    center=center,
                    width=width,
                    height=z_span,
                    direction=direction,
                )
            )
        s.next()
    return ports
