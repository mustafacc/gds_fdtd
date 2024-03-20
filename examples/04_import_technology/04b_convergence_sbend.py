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
        device = gtd.simprocessor.load_component_from_tech(ly=layout, tech=technology, z_span=z)
        sims.append(gtd.simprocessor.build_sim_from_tech(
            tech=technology,
            layout=layout,
            in_port=0,
            wavl_min=1.545,
            wavl_max=1.555,
            wavl_pts=51,
            mode_index=[0,1],
            num_modes=2,
            symmetry=(
                0,
                0,
                0,
            ),  # ensure structure is symmetric across symmetry axis before triggering this!
            z_span=z,
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
        f = sim.s_parameters.S[s12_te].freq
        te_log.append(10*np.log10(np.abs(s_te[len(s_te) // 2])**2))  # middle
        tm_log.append(10*np.log10(np.abs(s_tm[len(s_tm) // 2])**2))  # middle  entry        ax.plot(z_span, te_log, 'x-', label="TE", color='b')
    ax.plot(z_span, te_log, 'x-', label="TE", color='r')
    ax.plot(z_span, tm_log, 'x-', label="TM", color='r')
    ax.legend()
    ax.set_xlabel('z_span [um]')
    ax.set_ylabel(f'Transmission [dB]')

    return sims


if __name__ == "__main__":

    tech_path = os.path.join(os.path.dirname(__file__), "tech.yaml")
    technology = gtd.core.parse_yaml_tech(tech_path)

    file_gds = os.path.join(os.path.dirname(__file__), "sbend.gds")

    layout = gtd.lyprocessor.load_layout(file_gds)

    # log sampling space, i expect output transission to be log too..
    z_span_sweep = convergence_z_span(
        layout=layout,
        tech=technology,
        z_span=np.logspace(np.log10(0.221), np.log10(2), num=12),
        )