# nlmod
<img src="docs/_static/logo_10000_2.png" width="256"/>

[![nlmod](https://github.com/ArtesiaWater/nlmod/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/ArtesiaWater/nlmod/actions/workflows/ci.yml)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/6fadea550ea04ea28b6ccde88fc56f35)](https://www.codacy.com/gh/ArtesiaWater/nlmod/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=ArtesiaWater/nlmod&amp;utm_campaign=Badge_Grade)
[![Codacy Badge](https://app.codacy.com/project/badge/Coverage/6fadea550ea04ea28b6ccde88fc56f35)](https://www.codacy.com/gh/ArtesiaWater/nlmod/dashboard?utm_source=github.com&utm_medium=referral&utm_content=ArtesiaWater/nlmod&utm_campaign=Badge_Coverage)
[![PyPI version](https://badge.fury.io/py/nlmod.svg)](https://badge.fury.io/py/nlmod)

Python package with functions to process, build and visualise MODFLOW models in the Netherlands. 

The functions in nlmod have four main objectives:
1. Create and adapt the temporal and spatial discretization of a MODFLOW model using an xarray Dataset. These functions are contained in `nlmod.dims`
2. Read data from external sources, project this data on the modelgrid and add this data to an xarray Dataset. These functions are contained in `nlmod.read`
3. Use data in an xarray Dataset to build modflow packages using flopy.  The functions for modflow 6 packages are in `nlmod.gwf`, for modpath in `nlmod.modpath`.
4. Visualise modeldata in Python or GIS software. These functions are contained in `nlmod.plot` and `nlmod.gis`.

External data sources that can be read are:
- AHN, digital elevation model
- bgt, surface water level geometries
- Geotop, subsurface model
- Jarkus, bathymetry
- KNMI, precipitation and evaporation
- REGIS, subsurface model
- Rijkswaterstaat, surface water polygons
- multiple waterboards, surface water level data

## Installation

Install the module with pip:

`pip install nlmod`

`nlmod` has many dependencies `xarray`, `flopy`, `rasterio`, `rioxarray`, `owslib`, `hydropandas`, `netcdf4`, `pyshp`, `rtree`, `openpyxl` and `matplotlib`.

When using pip the dependencies are automatically installed. Some dependencies are notoriously hard to install on certain platforms. 
Please see the [dependencies](https://github.com/ArtesiaWater/hydropandas#dependencies) section of the `hydropandas` package for more information on how to install these packages manually. 


## Getting started
If you are using nlmod for the first time you need to download the MODFLOW executables. You can easily download these executables by running this Python code:

	import nlmod
	nlmod.util.download_mfbinaries()

After you've downloaded the executables you can run the Jupyter Notebooks in the examples folder. These notebooks illustrate how you to use the nlmod package. 
