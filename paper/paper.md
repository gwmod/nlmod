---
title: 'NLMOD: From open data to MODFLOW models with Python'
tags:
  - hydrology
  - groundwater
  - Python
authors: # ORDER?
  - name: Ruben Caljé
    orcid:
    affiliation: "1"
  - name: Davíd A. Brakenhoff
    orcid: 0000-0002-2993-2202
    affiliation: "1"
  - name: Bas F. des Tombe
    orcid: 0000-0002-3302-7387
    affiliation: "2"
  - name: Martin A. Vonk
    orcid: 0009-0007-3528-2991
    affiliation: "1, 3"
  - name: Onno Ebbens
    orcid:
    affiliation: "1"
affiliations:
 - name: Artesia B.V., Schoonhoven, South Holland, The Netherlands
   index: 1
 - name: N.V. PWN Waterleidingbedrijf Noord-Holland, Velserbroek, North Holland, The Netherlands
   index: 2
 - name: Department of Water Management, Faculty of Civil Engineering and Geosciences, Delft University of Technology, Delft, South Holland, The Netherlands
   index: 3

date: 06 August 2025
bibliography: paper.bib
---

# Summary
`NLMOD` is a Python package to build, run and visualize MODFLOW 6 [@MODFLOW6] groundwater models.
It handles open-data, mainly from the Netherlands, efficiently via the popular Python package `xarray` [@hoyer_xarray_2017].
`FloPy` [@bakker_flopy_2016,hughes_flopy_2024] is used for creating and running the MODFLOW-based groundwater flow, transport and particle tracking models.

# Statement of need
`NLMOD` was built to allow users to write scripts to quickly download relevant data from publicly available sources, and build and post-process groundwater flow [@MODFLOW6GFM] and transport [@MODFLOW6GTM] models at different spatial and temporal scales to answer geohydrological questions.
Scripting these steps, from downloading data to building groundwater models, makes models more reproducible and transparent.
As groundwater systems face increasing pressure from climate change and overuse, the ability to rapidly develop fit-for-purpose models with the latest open data is essential for decision making and sustainable water management.

# Workflow from open data to groundwater model
The core of `NLMOD` is built around an `xarray.Dataset` [@hoyer_xarray_2017], a multi-dimensional, dictionary-like container of labeled arrays (`xarray.DataArray` objects) with aligned dimensions, conceptually similar to a `pandas.DataFrame` [@mckinney_pandas_2010].
`Dataset`s serve as an in-memory representation of the Network Common Data Form (netCDF) data model, making them ideal for handling structured geospatial and temporal data.
An `xarray.Dataset` forms the foundational data structure that supports the four main objectives of NLMOD:

## Define model discretization: `nlmod.dims`
The `nlmod.dims` module allows users to create and adapt both the spatial discretization (e.g., extent, layers, grid) and temporal discretization using an `xarray.Dataset`.

## Downloading and integrating open data: `nlmod.read`
Once the model grid is defined, the `nlmod.read` module supports downloading relevant geohydrological data from public sources, reprojecting it to fit the model grid, and adding it directly to the `xarray.Dataset`.
These public sources can be very broad such as digital terrain models [@AHN], large-scale base maps [@BGT], bathymetry models, geological or geomorphological schematizations [@REGIS;@GEOTOP;@BOFEK;@BRO], meteorological datasets [@KNMI] etc.

At the moment, predominantly open data sources from the Netherlands are supported. However, users can (download and) add any dataset needed to the `xarray.Dataset`.

## Building groundwater models: `nlmod.sim`, `nlmod.gwf`, `nlmod.gwt`, `nlmod.prt`, and `nlmod.modpath`
Using the data in the xarray.Dataset, NLMOD provides tools to build MODFLOW 6 [@MODFLOW6] models via FloPy [@bakker_flopy_2016;@hughes_flopy_2024].
The modules `nlmod.sim`, `nlmod.gwf`, `nlmod.gwt`, and `nlmod.prt` are designed for MODFLOW 6, while `nlmod.modpath` enables particle tracking using MODPATH [@MODPATH7].
This modular approach supports full scriptability which enables complex model optimization schemes.

## Visualize model data: `nlmod.plot` and `nlmod.gis`
To interpret and communicate model inputs and outputs, `NLMOD` offers built-in plotting functionality (`nlmod.plot`) for use in Python, as well as GIS tools (`nlmod.gis`) for exporting data to commonly used geospatial formats.
These visualization options support both technical analysis and clear communication with stakeholders.

# Note of thanks
We thank the following institutions and contributors for their support to the development of `NLMOD`.

# References