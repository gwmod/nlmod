# nlmod

[![nlmod](https://github.com/gwmod/nlmod/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/gwmod/nlmod/actions/workflows/ci.yml)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/f1797b66e98b42b294bc1c5fc233dbf3)](https://app.codacy.com/gh/gwmod/nlmod/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
[![Codacy Badge](https://app.codacy.com/project/badge/Coverage/f1797b66e98b42b294bc1c5fc233dbf3)](https://app.codacy.com/gh/gwmod/nlmod/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_coverage)
[![PyPI version](https://badge.fury.io/py/nlmod.svg)](https://badge.fury.io/py/nlmod)
[![Documentation Status](https://readthedocs.org/projects/nlmod/badge/?version=stable)](https://nlmod.readthedocs.io/en/stable/?badge=stable)

Python package to build, run and visualize MODFLOW 6 groundwater models in the Netherlands.

`nlmod` was built to allow users to write scripts to quickly download relevant data
from publicly available sources, and build and post-process groundwater flow and
transport models at different spatial and temporal scales to answer specific
geohydrological questions. Scripting these steps, from downloading data to building
groundwater models, makes models more reproducible and transparent.

The functions in `nlmod` have four main objectives:

1. Create and adapt the temporal and spatial discretization of a MODFLOW model using an 
   xarray Dataset (`nlmod.dims`).
2. Download and read data from external sources, project this data on the modelgrid and 
   add this data to an xarray Dataset (`nlmod.read`).
3. Use data in an xarray Dataset to build modflow packages for both groundwater flow 
   and transport models using FloPy (`nlmod.sim`, `nlmod.gwf` and `nlmod.gwt` for 
   Modflow 6 and `nlmod.modpath` for Modpath).
4. Visualise modeldata in Python (`nlmod.plot`) or GIS software (`nlmod.gis`).

More information can be found on the documentation-website:
https://nlmod.readthedocs.io/.

<p align="center">
  <img src="docs/_static/logo_10000_2.png" width="256"/>
</p>

## Installation

Install the module with pip:

`pip install nlmod`

`nlmod` has the following required dependencies: 

* `flopy`
* `xarray`
* `netcdf4`
* `rasterio`
* `rioxarray`
* `affine`
* `geopandas`
* `owslib`
* `hydropandas`
* `shapely`
* `pyshp`
* `rtree`
* `matplotlib`
* `dask`
* `colorama`
* `joblib`
* `bottleneck`

There are some optional dependecies, only needed (and imported) in a single method.
Examples of this are `geocube`, `rasterstats` (both used in nlmod.util.zonal_statistics),
`h5netcdf` (used for hdf5 files backend in xarray), `scikit-image`
(used in nlmod.read.rws.calculate_sea_coverage).
To install `nlmod` with the optional dependencies use:

`pip install nlmod[full]`

When using pip the dependencies are automatically installed. Some dependencies are
notoriously hard to install on certain platforms. Please see the
[dependencies](https://github.com/ArtesiaWater/hydropandas#dependencies) section of the
`hydropandas` package for more information on how to install these packages manually.

## Getting started

Start with the Jupyter Notebooks in the examples folder. These notebooks illustrate how to use the `nlmod` package.
