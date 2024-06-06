#%% send a component to a lumerical instance
import os
import lumapi
from gds_tidy3d.lum_tools import to_lumerical, setup_lum_fdtd
from gds_tidy3d.core import parse_yaml_tech
from gds_tidy3d.simprocessor import load_component_from_tech
from gds_tidy3d.lyprocessor import load_layout

os.environ['QT_QPA_PLATFORM'] = 'xcb'  # i need to do this to get my lumerical gui to work in linux... comment out if not necessary

if __name__ == "__main__":
    tech_path = os.path.join(os.path.dirname(__file__), "tech.yaml")  # note materials definition format in yaml
    technology = parse_yaml_tech(tech_path)

    file_gds = os.path.join(os.path.dirname(__file__), "si_sin_escalator.gds")
    layout = load_layout(file_gds)
    component = load_component_from_tech(ly=layout, tech=technology)

    fdtd = lumapi.FDTD()  # can also be mode/device
    print(type(fdtd))

    to_lumerical(c=component, lum=fdtd)

    #setup_lum_fdtd(c=component, lum=fdtd)

    input('Proceed to terminate the GUI?')
# %%
