# %%!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convergence testing examples using an unideal S-bend test case. 
@author: Mustafa Hammood
"""
import gds_tidy3d as gtd
import tidy3d as td
import os
import matplotlib.pyplot as plt
import numpy as np

def convergence_z_span(
    layout: gtd.core.layout,
    tech: dict,
    z_span: list=np.logspace(np.log10(0.221), np.log10(2), num=12)):

    sims_z_span = []
    sims = []

    for z in z_span:
        device = gtd.simprocessor.load_component_from_tech(ly=layout, tech=tech, z_span=z)
        sims.append(gtd.simprocessor.build_sim_from_tech(
            tech=tech,
            layout=layout,
            in_port=0,
            wavl_min=1.500,
            wavl_max=1.600,
            wavl_pts=51,
            mode_index=[0,1],
            num_modes=2,
            symmetry=(
                0,
                0,
                0,
            ),  # ensure structure is symmetric across symmetry axis before triggering this!
            z_span=z,
            visualize=False,
        ))
    
        sims[-1].upload()
        # run the simulation. CHECK THE SIMULATION IN THE UI BEFORE RUNNING!
        sims[-1].execute()
        #  visualize the results
        #sims[-1].visualize_results()     

    fig, ax1 = plt.subplots()
    te_log = []
    tm_log = []
    s12_te = 'S12_idx00'
    s12_tm = 'S12_idx11'
    for idx, sim in enumerate(sims):
        s_te = sim.s_parameters.S[s12_te].s
        s_tm = sim.s_parameters.S[s12_tm].s
        te_log.append(10*np.log10(np.abs(s_te[len(s_te) // 2])**2))  # middle
        tm_log.append(10*np.log10(np.abs(s_tm[len(s_tm) // 2])**2))  # middle  entry
    ax1.plot(z_span, te_log, 'x-', label="TE", color='r')
    ax1.set_xlabel('Z_span [um]')
    ax1.set_ylabel(f'Transmission [dB]', color='r')

    ax2 = ax1.twinx()
    ax2.plot(z_span, tm_log, 'x-', label="TM", color='b')
    ax2.set_ylabel(f'Transmission [dB]', color='b')

    fig.tight_layout()
    fig.legend()
    fig.show()

    return sims


def convergence_port_width(
    layout: gtd.core.layout,
    tech: dict,
    port_width: list=np.logspace(np.log10(0.51), np.log10(3), num=12)):

    sims_port_width = []
    sims = []

    for w in port_width:
        device = gtd.simprocessor.load_component_from_tech(ly=layout, tech=tech, z_span=2)
        sims.append(gtd.simprocessor.build_sim_from_tech(
            tech=tech,
            layout=layout,
            in_port=0,
            wavl_min=1.500,
            wavl_max=1.600,
            wavl_pts=51,
            mode_index=[0,1],
            num_modes=2,
            width_ports=w,
            symmetry=(
                0,
                0,
                0,
            ),  # ensure structure is symmetric across symmetry axis before triggering this!
            z_span=2,
            visualize=False,
        ))
    
        sims[-1].upload()
        # run the simulation. CHECK THE SIMULATION IN THE UI BEFORE RUNNING!
        sims[-1].execute()
        #  visualize the results
        #sims[-1].visualize_results()     


    fig, ax = plt.subplots()
    te_log = []
    tm_log = []
    s12_te = 'S12_idx00'
    s12_tm = 'S12_idx11'
    for idx, sim in enumerate(sims):
        s_te = sim.s_parameters.S[s12_te].s
        s_tm = sim.s_parameters.S[s12_tm].s
        te_log.append(10*np.log10(np.abs(s_te[len(s_te) // 2])**2))  # middle
        tm_log.append(10*np.log10(np.abs(s_tm[len(s_tm) // 2])**2))  # middle  entry
    ax.plot(port_width, te_log, 'x-', label="TE", color='r')
    ax.plot(port_width, tm_log, 'x-', label="TM", color='b')
    ax.legend()
    ax.set_xlabel('port_width [um]')
    ax.set_ylabel(f'Transmission [dB]')

    return sims


def convergence_mesh(
    layout: gtd.core.layout,
    tech: dict,
    mesh: list=np.linspace(6, 60, 20)
    ):

    sims_mesh = []
    sims = []

    for m in mesh:
        device = gtd.simprocessor.load_component_from_tech(ly=layout, tech=tech, z_span=2)
        sims.append(gtd.simprocessor.build_sim_from_tech(
            tech=tech,
            layout=layout,
            in_port=0,
            wavl_min=1.500,
            wavl_max=1.600,
            wavl_pts=51,
            mode_index=[0,1],
            num_modes=2,
            grid_cells_per_wvl=m,
            symmetry=(
                0,
                0,
                0,
            ),  # ensure structure is symmetric across symmetry axis before triggering this!
            z_span=2,
            visualize=False,
        ))
    
        sims[-1].upload()
        # run the simulation. CHECK THE SIMULATION IN THE UI BEFORE RUNNING!
        sims[-1].execute()
        #  visualize the results
        #sims[-1].visualize_results()     

    fig, ax1 = plt.subplots()
    te_log = []
    tm_log = []
    s12_te = 'S12_idx00'
    s12_tm = 'S12_idx11'
    for idx, sim in enumerate(sims):
        s_te = sim.s_parameters.S[s12_te].s
        s_tm = sim.s_parameters.S[s12_tm].s
        te_log.append(10*np.log10(np.abs(s_te[len(s_te) // 2])**2))  # middle
        tm_log.append(10*np.log10(np.abs(s_tm[len(s_tm) // 2])**2))  # middle  entry
    ax1.plot(mesh, te_log, 'x-', label="TE", color='r')
    ax1.set_xlabel('Mesh [grid cells / wavl]')
    ax1.set_ylabel(f'Transmission [dB]', color='r')

    ax2 = ax1.twinx()
    ax2.plot(mesh, tm_log, 'x-', label="TM", color='b')
    ax2.set_ylabel(f'Transmission [dB]', color='b')

    fig.tight_layout()
    fig.legend()
    fig.show()

    return sims


if __name__ == "__main__":

    tech_path = os.path.join(os.path.dirname(__file__), "tech.yaml")
    technology = gtd.core.parse_yaml_tech(tech_path)

    file_gds = os.path.join(os.path.dirname(__file__), "sbend.gds")

    layout = gtd.lyprocessor.load_layout(file_gds)

    """
    # log sampling space, i expect output transission to be log too..
    z_span_sweep = convergence_z_span(
        layout=layout,
        tech=technology,
        z_span=np.logspace(np.log10(0.221), np.log10(4), num=14),
        )
    """
    port_width_sweep = convergence_port_width(
        layout=layout,
        tech=technology,
        port_width=np.logspace(np.log10(0.51), np.log10(3), num=12),
        )
    """
    mesh_sweep = convergence_mesh(
        layout=layout,
        tech=technology,
        mesh=np.linspace(6, 40, 20)
        )
    """
# %%
