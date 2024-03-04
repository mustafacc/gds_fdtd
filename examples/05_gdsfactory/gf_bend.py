# %%
import gds_tidy3d as gtd
import gdsfactory as gf
import os
from gds_tidy3d.simprocessor import from_gdsfactory


if __name__ == '__main__':
    tech_file_path = os.path.join(os.path.dirname(__file__), "tech.yaml")
    technology = gtd.core.parse_yaml_tech(tech_file_path)

    c = gf.components.bend_circular()

    simulation = from_gdsfactory(c=c, tech=technology)
    # %%
    simulation.upload()
    # run the simulation. CHECK THE SIMULATION IN THE UI BEFORE RUNNING!
    simulation.execute()
    #  visualize the results
    simulation.visualize_results()
    # %%
