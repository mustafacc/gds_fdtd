# gds_tidy3d

![alternative text](/docs/banner.png)

![codecov](https://codecov.io/gh/mustafacc/gds_tidy3d/branch/main/graph/badge.svg)
![build](https://github.com/mustafacc/gds_tidy3d/actions/workflows/main.yml/badge.svg)

Minimal Python module to assist in setting up Tidy3D FDTD simulation on planar nanophotonic devices.

## Features
- Generate tidy3d simulation of a gds device systematically.
- Generate tidy3d simulations from GDS device by identifying ports and simulation region from an input technology stack or [gdsfactory](https://github.com/gdsfactory/gdsfactory).
- Generate S-parameters of devices and export them to standard formats.
- Generate multimode (or dual polarization) simulations.

## Installation

```bash
git clone git@github.com:mustafacc/gds_tidy3d.git
cd gds_tidy3d
pip install -e .[dev]
