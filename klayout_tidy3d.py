# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 15:38:04 2023.

Create a directional coupler.

Automated SiEPIC-Tools (Klayout gds) to tidy3d simulation maker flow.

@author: Mustafa Hammood, Dream Photonics 2023
"""
# %%
import tidy3d as td
import matplotlib.pyplot as plt
import numpy as np
import klayout.db as pya

# define device parameters (in microns)
wg_width = 0.5
dc_gap = 0.1
dc_length = 12.5
sbend_r = 5
sbend_h = 2
sbend_l = 7

# below are helper methods in creating layouts in klayout.db package.


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


# define a layout object
ly = pya.Layout()
dbu = 0.001  # layout's database unit (in microns)

# create a cell object
cell = ly.create_cell('DirectionalCoupler')

# define layers
layer_device = ly.layer(1, 0)  # layer to define device's objects
layer_pinrec = ly.layer(69, 0)  # layer to define device's ports
layer_devrec = ly.layer(68, 0)  # layer to define device's boundaries

objects_device = []
# create coupling region waveguides
box = pya.Box(-to_dbu(dc_length/2, dbu), to_dbu(dc_gap/2, dbu),
              to_dbu(dc_length/2, dbu), to_dbu(wg_width + dc_gap/2, dbu))
objects_device.append(cell.shapes(layer_device).insert(box))

box = pya.Box(-to_dbu(dc_length/2, dbu), -to_dbu(dc_gap/2, dbu),
              to_dbu(dc_length/2, dbu), -to_dbu(wg_width + dc_gap/2, dbu))
objects_device.append(cell.shapes(layer_device).insert(box))

# create s-bend fan-out/in sections
s = sbend(dc_length/2, dc_gap/2+wg_width/2, wg_width, sbend_r, sbend_h,
          sbend_l, direction='east', verbose=True)
objects_device.append(cell.shapes(layer_device).insert(s))

s = sbend(dc_length/2, -dc_gap/2-wg_width/2, wg_width, sbend_r, sbend_h,
          sbend_l, direction='east', flip=True, verbose=True)
objects_device.append(cell.shapes(layer_device).insert(s))

s = sbend(-dc_length/2, -dc_gap/2-wg_width/2, wg_width, sbend_r, sbend_h,
          sbend_l, direction='west', verbose=True)
objects_device.append(cell.shapes(layer_device).insert(s))

s = sbend(-dc_length/2, dc_gap/2+wg_width/2, wg_width, sbend_r, sbend_h,
          sbend_l, direction='west', flip=True, verbose=True)
objects_device.append(cell.shapes(layer_device).insert(s))

# create straight regions near the ports
l_extra = 0.1
box = pya.Box(
    to_dbu(-dc_length/2-sbend_l-l_extra, dbu),
    to_dbu(sbend_h + dc_gap/2, dbu),
    to_dbu(-dc_length/2-sbend_l, dbu),
    to_dbu(sbend_h + wg_width + dc_gap/2, dbu))
objects_device.append(cell.shapes(layer_device).insert(box))

box = pya.Box(
    to_dbu(-dc_length/2-sbend_l-l_extra, dbu),
    -to_dbu(sbend_h + dc_gap/2, dbu),
    to_dbu(-dc_length/2-sbend_l, dbu),
    -to_dbu(sbend_h + wg_width + dc_gap/2, dbu))
objects_device.append(cell.shapes(layer_device).insert(box))

box = pya.Box(
    -to_dbu(-dc_length/2-sbend_l-l_extra, dbu),
    to_dbu(sbend_h + dc_gap/2, dbu),
    -to_dbu(-dc_length/2-sbend_l, dbu),
    to_dbu(sbend_h + wg_width + dc_gap/2, dbu))
objects_device.append(cell.shapes(layer_device).insert(box))

box = pya.Box(
    -to_dbu(-dc_length/2-sbend_l-l_extra, dbu),
    -to_dbu(sbend_h + dc_gap/2, dbu),
    -to_dbu(-dc_length/2-sbend_l, dbu),
    -to_dbu(sbend_h + wg_width + dc_gap/2, dbu))
objects_device.append(cell.shapes(layer_device).insert(box))


# create device boundary region
h_buffer = 2
box = pya.Box(
    to_dbu(-dc_length/2-sbend_l-l_extra, dbu),
    -to_dbu(h_buffer+sbend_h + wg_width + dc_gap/2, dbu),
    -to_dbu(-dc_length/2-sbend_l-l_extra, dbu),
    to_dbu(h_buffer+sbend_h + wg_width + dc_gap/2, dbu))
objects_devrec = cell.shapes(layer_devrec).insert(box)

# create port definition regions
make_pin(cell, 'opt1', [-dc_length/2-sbend_l-l_extra, sbend_h +
         wg_width/2 + dc_gap/2], wg_width, layer_pinrec, direction=180)

make_pin(cell, 'opt2', [-dc_length/2-sbend_l-l_extra, -sbend_h -
         wg_width/2 - dc_gap/2], wg_width, layer_pinrec, direction=180)

make_pin(cell, 'opt3', [dc_length/2+sbend_l+l_extra, sbend_h +
         wg_width/2 + dc_gap/2], wg_width, layer_pinrec, direction=0)

make_pin(cell, 'opt4', [dc_length/2+sbend_l+l_extra, -sbend_h -
         wg_width/2 - dc_gap/2], wg_width, layer_pinrec, direction=0)

# export layout
fname = "ex2_DirectionalCoupler.oas"
gzip = False
options = pya.SaveLayoutOptions()
ly.write(fname, gzip, options)

# %% helper functions to deconstruct a SiEPIC-Tidy3D cell


def get_ports(c, layer, dbu=0.001):
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
    sim_x = abs(min(x))+abs(max(x))
    sim_y = abs(min(y))+abs(max(y))
    return polygons_vertices, sim_x, sim_y


def get_polygons(c, layer, dbu=0.001):
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
    import klayout.db as pya
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


polygons_device = get_polygons(cell, layer_device, dbu)
devrec, sim_x, sim_y = get_devrec(cell, layer_devrec, dbu)
ports = get_ports(cell, layer_pinrec, dbu)

# %% setup simulation


def make_structures(polygons_device, devrec, thick_dev, thick_sub, thick_super, mat_dev, mat_sub, mat_super, sidewall_angle=85):
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


def make_source(port, width=3, depth=2, thick_dev=0.22, freq0=2e14, fwidth=1e13, buffer=0.25):
    if port['direction'] == 0:
        x_buffer = -buffer
        y_buffer = 0
    elif port['direction'] == 180:
        x_buffer = buffer
        y_buffer = 0
    elif port['direction'] == 90:
        x_buffer = 0
        y_buffer = -buffer
    elif port['direction'] == 270:
        x_buffer = 0
        y_buffer = buffer
    if port['direction'] in [180, 270]:
        direction = "+"
    else:
        direction = "-"
    msource = td.ModeSource(
        center=[port['x']+x_buffer, port['y']+y_buffer, thick_dev/2],
        size=[0, width, depth],
        direction=direction,
        source_time=td.GaussianPulse(freq0=freq0, fwidth=fwidth),
        mode_spec=td.ModeSpec(),
        mode_index=0,
    )
    return msource


def make_monitors(ports, thick_dev=0.22, freq0=2e14, buffer=0.5):
    """Create monitors for a given list of ports."""
    monitors = []
    for p in ports:
        if ports[p]['direction'] == 0:
            x_buffer = -buffer
            y_buffer = 0
        elif ports[p]['direction'] == 180:
            x_buffer = buffer
            y_buffer = 0
        elif ports[p]['direction'] == 90:
            x_buffer = 0
            y_buffer = -buffer
        elif ports[p]['direction'] == 270:
            x_buffer = 0
            y_buffer = buffer
        # mode monitors
        monitors.append(td.ModeMonitor(
            center=[ports[p]['x']+x_buffer, ports[p]
                    ['y']+y_buffer, thick_dev/2],
            size=[0, ports[p]['width']*5, thick_dev*10],
            freqs=[freq0],
            mode_spec=td.ModeSpec(),
            name=str(p),
        ))
    # field monitor
    monitors.append(td.FieldMonitor(
        center=[0, 0, thick_dev/2],
        size=[td.inf, td.inf, 0],
        freqs=[freq0],
        name="field",
    ))
    return monitors


# frequency and bandwidth of pulsed excitation
freq0 = 3e8/(1550e-9)
fwidth = 1.9e13
# sim. time in secs
run_time = 20/fwidth
thick_dev = 0.22
thick_sub = 2
thick_super = 3

# apply pml in all directions
boundary_spec = td.BoundarySpec.all_sides(boundary=td.PML())

# define materials structures
mat_dev = td.Medium(permittivity=3.48**2)
mat_sub = td.Medium(permittivity=1.48**2)
mat_super = td.Medium(permittivity=1.48**2)

# define geometry
sidewall_angle = 82  # degrees
structures = make_structures(polygons_device, devrec, thick_dev, thick_sub,
                             thick_super, mat_dev, mat_sub, mat_super, sidewall_angle)
# define source on a given port
source = make_source(ports[1], thick_dev=thick_dev, freq0=freq0, fwidth=fwidth)
# define monitors
monitors = make_monitors(ports, thick_dev, freq0)


# simulation domain size (in microns)
sim_size = [sim_x, sim_y, 4]

# resolution control: minimum number of grid cells per wavelength in each material
grid_cells_per_wvl = 16

# initialize the simulation
sim = td.Simulation(
    size=sim_size,
    grid_spec=td.GridSpec.auto(min_steps_per_wvl=grid_cells_per_wvl),
    structures=structures,
    sources=[source],
    monitors=monitors,
    run_time=run_time,
    boundary_spec=boundary_spec,
)

for m in sim.monitors:
    m.help()

# visualize the source
source.source_time.plot(np.linspace(0, run_time, 1001))
plt.show()

# visualize the simulation
fig, ax = plt.subplots(1, 3, figsize=(13, 4))
sim.plot_eps(z=0, freq=freq0, ax=ax[0])
sim.plot_eps(y=0, freq=freq0, ax=ax[1])
sim.plot_eps(x=0, freq=freq0, ax=ax[2])

# visualize geometry
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
sim.plot(z=thick_dev/2, ax=ax1)
sim.plot(x=0.1, ax=ax2)
ax2.set_xlim([-3, 3])
plt.show()

# %% upload and run the simulation
# create job, upload sim to server to begin running
job = td.web.Job(simulation=sim, task_name="CouplerVerify")

# download the results and load them into a simulation
sim_data = job.run(path="data/sim_data.hdf5")

#%%
def measure_transmission(sim_data):
    """Constructs a "row" of the scattering matrix when sourced from top left port"""

    input_amp = sim_data['1'].amps.sel(direction="+")

    amps = np.zeros(4, dtype=complex)
    directions = ("-", "-", "+", "+")
    for i, (monitor, direction) in enumerate(
        zip(sim_data.simulation.monitors[:4], directions)
    ):
        amp = sim_data[monitor.name].amps.sel(direction=direction)
        amp_normalized = amp / input_amp
        amps[i] = np.squeeze(amp_normalized.values)

    return amps


# monitor and test out the measure_transmission function the results of the single run
amps_arms = measure_transmission(sim_data)
print("mode amplitudes in each port: \n")
for amp, monitor in zip(amps_arms, sim_data.simulation.monitors[:-1]):
    print(f'\tmonitor     = "{monitor.name}"')
    print(f"\tamplitude^2 = {abs(amp)**2:.2f}")
    print(f"\tphase       = {(np.angle(amp)):.2f} (rad)\n")
#%%
fig, ax = plt.subplots(1, 1, figsize=(16, 3))
# sim_data['field'].Ey.real.interp(z=0).plot()
sim_data.plot_field("field", "Ey", z=thick_dev/2, freq=freq0, ax=ax)
plt.show()
