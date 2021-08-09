[![nlmod](https://github.com/ArtesiaWater/nlmod/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/ArtesiaWater/nlmod/actions/workflows/ci.yml)

# nlmod

Python package with functions to process, build and visualise MODFLOW models in the Netherlands. 

The functions in nlmod have three main objectives:
1. Create and adapt the temporal and spatial discretization of a MODFLOW model using an xarray Dataset. These functions are contained in `nlmod.mdims`
2. Read data from external sources, project this data on the modelgrid and add this data to an xarray Dataset. These functions are contained in `nlmod.read`
3. Use data in an xarray Dataset to build modflow packages using flopy.  These functions are contained in `nlmod.mfpackages`

External data sources that can be read are:
- regis
- geotop
- knmi
- jarkus
- jarkus

## Installation

The package is not (yet) available on PyPi. 

You can install the package by cloning the repository and
typing `pip install -e .` from the package root directory.

When using pip the dependencies are automatically installed. Some dependencies are notoriously hard to install on certain platforms. Please see the [dependencies](https://github.com/ArtesiaWater/hydropandas#dependencies) section of the `hydropandas` package for more information on how to install these packages manually. 


## Getting started
In the examples folder you find some Jupyter Notebooks which illustrate how you to use the nlmod package. 

