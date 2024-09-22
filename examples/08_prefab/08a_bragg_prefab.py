# %%!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Mustafa Hammood
"""
import gds_fdtd as gtd
import tidy3d as td
import os


if __name__ == "__main__":

    tech_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tech.yaml")
    technology = gtd.core.parse_yaml_tech(tech_path)

    # Define the path to the GDS file
    file_gds = os.path.join(os.path.dirname(os.path.dirname(__file__)), "devices.gds")

    device = gtd.lyprocessor.load_device(file_gds, top_cell='bragg_te1550', tech=technology, prefab="ANT_NanoSOI_ANF1_d9")

    simulation = gtd.simprocessor.build_sim_from_tech(
        tech=technology,
        layout=layout,
        in_port=0,
        wavl_min=1.,
        wavl_max=1.4,
        wavl_pts=501,
        grid_cells_per_wvl=6,
        symmetry=(
            0,
            0,
            0,
        ),  # ensure structure is symmetric across symmetry axis before triggering this!
        z_span=4,
        field_monitor_axis="y",
    )

    simulation.upload()
    # run the simulation. CHECK THE SIMULATION IN THE UI BEFORE RUNNING!
    simulation.execute()
    #  visualize the results
    simulation.visualize_results()

# %%
