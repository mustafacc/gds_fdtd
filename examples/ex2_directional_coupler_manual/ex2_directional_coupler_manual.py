# %%
from tidy3d import web
import matplotlib.pyplot as plt
import numpy as np
import tidy3d as td
import klayout.db as pya

# define device parameters (in microns)
wg_width = 0.5
dc_gap = 0.2
dc_length = 11
sbend_r = 5
sbend_h = 1
sbend_l = 6

# define a layout object
ly = pya.Layout()
dbu = 0.001  # layout's database unit (in microns)

# create a cell object
cell = ly.create_cell('DirectionalCoupler')

# define layers
layer_device = ly.layer(1, 0)  # layer to define device's objects

# %% Define the couling region


def to_dbu(f, dbu):
    """Convert numerical value to dbu-valued integer."""
    return int(round(float(f)/dbu))


objects_device = []
# create coupling region waveguides
box = pya.Box(-to_dbu(dc_length/2, dbu), to_dbu(dc_gap/2, dbu),
              to_dbu(dc_length/2, dbu), to_dbu(wg_width + dc_gap/2, dbu))
objects_device.append(cell.shapes(layer_device).insert(box))

box = pya.Box(-to_dbu(dc_length/2, dbu), -to_dbu(dc_gap/2, dbu),
              to_dbu(dc_length/2, dbu), -to_dbu(wg_width + dc_gap/2, dbu))
objects_device.append(cell.shapes(layer_device).insert(box))

# %% Define the S-bends


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


def sbend(x, y, w, r, h, l, dbu=0.001, direction=0, flip=False, verbose=False):
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
        Direction of the S-bend. The default is 0. Options include:
            0 -> 'east'
            180 -> 'west'
            90 -> 'north'
            270 -> 'south'
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

        if direction == 0:
            shape_bend = pya.Polygon(pts).transformed(trans.R0)
        elif direction == 180:
            shape_bend = pya.Polygon(pts).transformed(trans.R180)
        elif direction == 90:
            shape_bend = pya.Polygon(pts).transformed(trans.R90)
        elif direction == 270:
            shape_bend = pya.Polygon(pts).transformed(trans.R270)
        else:
            raise ValueError(
                'Invalid direction! Valid options: 0, 180, 90, 270')
        if flip:
            shape_bend = shape_bend.transformed(trans.M0)
        shape_bend = shape_bend.transformed(trans)
        return shape_bend


# create s-bend fan-out/in sections
s = sbend(dc_length/2, dc_gap/2+wg_width/2, wg_width, sbend_r, sbend_h,
          sbend_l, direction=0, verbose=True)
objects_device.append(cell.shapes(layer_device).insert(s))

s = sbend(dc_length/2, -dc_gap/2-wg_width/2, wg_width, sbend_r, sbend_h,
          sbend_l, direction=0, flip=True, verbose=True)
objects_device.append(cell.shapes(layer_device).insert(s))

s = sbend(-dc_length/2, -dc_gap/2-wg_width/2, wg_width, sbend_r, sbend_h,
          sbend_l, direction=180, verbose=True)
objects_device.append(cell.shapes(layer_device).insert(s))

s = sbend(-dc_length/2, dc_gap/2+wg_width/2, wg_width, sbend_r, sbend_h,
          sbend_l, direction=180, flip=True, verbose=True)
objects_device.append(cell.shapes(layer_device).insert(s))

# %% create additional waveguide extension near the ports and save the layout
# create straight regions near the ports
l_extra = 1
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

# export layout
fname = "DirectionalCoupler.oas"
ly.write(fname)

# %% Define the materials to be used

mat_dev = td.material_library["cSi"]["Li1993_293K"]
mat_sub = td.Medium(permittivity=1.48**2)
mat_super = td.Medium(permittivity=1.48**2)

# %% define the structures in the simulation

thick_dev = 0.22  # thickness of the device layer (microns)
thick_sub = 2  # thickness of the substrate (microns)
thick_super = 3  # thickness of the superstrate (microns)


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

structures = []


# define the polygon bounds for the substrate and superstrate
x_min = -dc_length/2-sbend_l-l_extra/2
x_max = -x_min
y_buffer = 2  # y buffer space away from the center of the port (microns)
y_max = y_buffer+sbend_h + wg_width + dc_gap/2
y_min = -y_max

# define the superstrate
structures.append(td.Structure(
    geometry=td.PolySlab(
        vertices=[[x_min, y_min], [x_min, y_max],
                  [x_max, y_max], [x_max, y_min]],
        slab_bounds=(0, thick_super),
        axis=2,
    ),
    medium=mat_super,
))

# define the directional coupler structure
for poly in polygons_device:
    structures.append(td.Structure(
        geometry=td.PolySlab(
            vertices=poly,
            slab_bounds=(0, thick_dev),
            axis=2,
            sidewall_angle=(90-88) * (np.pi/180),
        ),
        medium=mat_dev,
    ))

# define the substrate
structures.append(td.Structure(
    geometry=td.PolySlab(
        vertices=[[x_min, y_min], [x_min, y_max],
                  [x_max, y_max], [x_max, y_min]],
        slab_bounds=(-thick_sub, 0),
        axis=2,
    ),
    medium=mat_sub,
))
# %% define the source on the north-west port
wavl_min = 1500e-9  # broadband simulation start wavelength
wavl_max = 1600e-9  # broadband simulation end wavelength

wavl0 = (wavl_max+wavl_min)/2
wavl_bw = (wavl_max-wavl_min)
freq0 = 3e8/wavl0  # define source frequency as center of broadband source
fwidth = 0.5 * 3e8*wavl_bw/(wavl0**2)

port_x = -dc_length/2-sbend_l
port_y = sbend_h + wg_width/2 + dc_gap/2
port_y_span = 5*wg_width
port_z_span = 4
msource = td.ModeSource(
    center=[port_x-l_extra/3, port_y, thick_dev/2],
    size=[0, port_y_span, port_z_span],
    direction='+',
    source_time=td.GaussianPulse(freq0=freq0, fwidth=fwidth),
    mode_spec=td.ModeSpec(),
    mode_index=0,
)
# %% define the monitors
monitors = []
# define 4 ModeMonitors at each port to capture the S-parameters
# setup the monitor to capture multiple wavelength points
wavl_pts = 101
freqs = 3e8/np.linspace(wavl_min, wavl_max, wavl_pts)
monitors.append(td.ModeMonitor(
    center=[port_x, port_y, thick_dev/2],
    size=[0, port_y_span, port_z_span],
    freqs=freqs,
    mode_spec=td.ModeSpec(),
    name='opt1',
))

monitors.append(td.ModeMonitor(
    center=[-port_x, port_y, thick_dev/2],
    size=[0, port_y_span, port_z_span],
    freqs=freqs,
    mode_spec=td.ModeSpec(),
    name='opt2',
))

monitors.append(td.ModeMonitor(
    center=[port_x, -port_y, thick_dev/2],
    size=[0, port_y_span, port_z_span],
    freqs=freqs,
    mode_spec=td.ModeSpec(),
    name='opt3',
))

monitors.append(td.ModeMonitor(
    center=[-port_x, -port_y, thick_dev/2],
    size=[0, port_y_span, port_z_span],
    freqs=freqs,
    mode_spec=td.ModeSpec(),
    name='opt4',
))

# define a FieldMonitor at mid-way through the device thickness
monitors.append(td.FieldMonitor(
    center=[0, 0, thick_dev/2],
    size=[td.inf, td.inf, 0],
    freqs=freqs,
    name="field",
))

# define a ModeSolverMonitor at the coupling region
monitors.append(td.ModeSolverMonitor(
    center=[0, 0, thick_dev/2],
    size=[0, port_y_span*2, port_z_span],
    freqs=[freq0],
    mode_spec=td.ModeSpec(num_modes=4),
    name="mode_monitor",
))
# %% define and build the simulation
# set boundary condition as PML on all sides
boundary_spec = td.BoundarySpec.all_sides(boundary=td.PML())

sim_x = 2*(dc_length/2+sbend_l+l_extra/2)
sim_size = [sim_x, y_max*2, port_z_span]

run_time = 3*4.5*sim_x*1e-6/3e8  # t = L/c = 3*ng*simulation span / c

symmetry = (0, 0, 1)  # define z symmetric simulation volume

sim = td.Simulation(
    size=sim_size,
    center=[0, 0, thick_dev/2],
    grid_spec=td.GridSpec.auto(min_steps_per_wvl=16),
    structures=structures,
    sources=[msource],
    monitors=monitors,
    run_time=run_time,
    boundary_spec=boundary_spec,
    symmetry=symmetry,
)

# %% visualize the simulation build

for m in sim.monitors:
    m.help()

# visualize the source
msource.source_time.plot(np.linspace(0, run_time, 1001))
plt.show()

# visualize the simulation
fig, ax = plt.subplots(1, 3, figsize=(13, 4))
sim.plot_eps(z=thick_dev/2, freq=freq0, ax=ax[0])
sim.plot_eps(y=0, freq=freq0, ax=ax[1])
sim.plot_eps(x=0, freq=freq0, ax=ax[2])

# visualize geometry
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
sim.plot(z=thick_dev/2, ax=ax1)
sim.plot(x=0.0, ax=ax2)
plt.show()


# %% upload the simulation
# create job, upload sim to server to begin running
job = web.Job(simulation=sim, task_name='DirectionalCoupler')
# check the simulation in the web viewer at this point before executing!
# %% run the simulation and save the simulation build and results into an hdf5 file
sim_data = job.run(path=f"results/sim_data.hdf5")

# %% fetch and visualize the results


def measure_transmission(sim_data, wavl):
    """Constructs a "row" of the scattering matrix when sourced from top left port"""
    in_port = 'opt1'
    input_amp = sim_data[in_port].amps.sel(direction="+")
    wavl_min, wavl_max, wavl_pts = wavl
    amps = np.zeros((4, wavl_pts), dtype=complex)
    directions = ('-', '+', '-', '+')
    for i, (monitor, direction) in enumerate(
        zip(sim_data.simulation.monitors[:4], directions)
    ):
        amp = sim_data[monitor.name].amps.sel(direction=direction)
        amp_normalized = amp / input_amp
        amps[i] = np.squeeze(amp_normalized.values)

    return amps


# monitor and test out the measure_transmission function the results of the single run
amps_arms = measure_transmission(sim_data, [wavl_min, wavl_max, wavl_pts])
fig, ax = plt.subplots(1, 1)
wavl = np.linspace(wavl_min, wavl_max, wavl_pts)
ax.set_xlabel('Wavelength [microns]')
ax.set_ylabel('Transmission [dB]')
for amp, monitor in zip(amps_arms, sim_data.simulation.monitors[:-1]):
    print(f'\tmonitor     = "{monitor.name}"')
    plt.plot(wavl, [10*np.log10(abs(i)**2)
             for i in amp], label=f"S{1}_{monitor.name}")
    print(f"\tamplitude^2 = {[abs(i)**2 for i in amp]}")
    print(f"\tphase       = {[np.angle(i)**2 for i in amp]} (rad)\n")
fig.legend()

fig, ax = plt.subplots(1, 1, figsize=(16, 3))
# sim_data['field'].Ey.real.interp(z=0).plot()
sim_data.plot_field("field", "Ey", z=thick_dev/2, freq=freq0, ax=ax)
plt.show()

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
sim_data.plot_field("mode_monitor", "Ey", f=freq0, val="real", mode_index=0, ax=ax[0])
sim_data.plot_field("mode_monitor", "Ey", f=freq0, val="real", mode_index=1, ax=ax[1])