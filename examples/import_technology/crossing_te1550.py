# %%!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Mustafa Hammood
"""
import gds_tidy3d as gtd
import tidy3d as td


if __name__ == "__main__":
    # Example usage
    file_path = "tech.yaml"
    technology = gtd.core.parse_yaml_tech(file_path)
    print(technology)

    fname_gds = "crossing_te1550.gds"
    layout = gtd.lyprocessor.load_layout(fname_gds)

    simulation = gtd.simprocessor.build_sim_from_tech(
        tech=technology,
        layout=layout,
        in_port=0,
        wavl_min=1.5,
        wavl_max=1.6,
        wavl_pts=101,
        symmetry=(0, 0, 1),  # ensure structure is symmetric across symmetry axis!
        z_span=4,
        field_monitor_axis="z",
    )
    simulation.upload()
    # run the simulation. CHECK THE SIMULATION IN THE UI BEFORE RUNNING!
    simulation.execute()
    #  visualize the results
    simulation.visualize_results()
