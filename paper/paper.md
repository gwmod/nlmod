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
It handles open-data, mainly from the Netherlands, efficiently via the popular Python packages `xarray` [@hoyer_xarray_2017] and `GeoPandas` [@bossche_geopandas_2025].
`FloPy` [@bakker_flopy_2016;@hughes_flopy_2024] is used for creating and running the MODFLOW-based groundwater flow, transport and particle tracking models.

# Statement of need
Building, running and changing groundwater models is often a time consuming, error prone task. Finding, cleaning and discretizing the data usually takes the majority of the time leaving less resources for analyzing, discussing and evaluating model outcomes. This often leads to poor model performance, insufficient knowledge about the effect of assumptions and unreliable results [REFERENTIE].

The goal of NLMOD is to reduce the development time and errors in the data preparation. Therefore leaving more time to evaluate model results, perform sensitivity analysis and assess the effect of assumptions. Meanwhile scripting these steps, from downloading data to building groundwater models, makes these models more reproducible and transparent. As groundwater systems face increasing pressure from climate change and overuse, the ability to rapidly develop fit-for-purpose models with the latest open data is essential for decision making and sustainable water management.

# Workflow from open data to groundwater model
In general an nlmod script to build a groundwater flow model contain these steps:
1. Download relevant data within a certain extent from available online sources and
gather data from local sources.
2. Specify spatial and temporal model dimensions and discritize the data from step 1.
Store all raster data in a single model dataset and vector data in one or multiple `geopandas.GeoDataFrame`s.
3. Build and run a groundwater flow model from the model dataset and Geodataframes using the flopy package.
This step
4. Post process model results.

## Downloading and integrating open data: `nlmod.read`
The `nlmod.read` module supports downloading relevant geohydrological data from public sources. reprojecting it to fit the model grid, and adding it directly to the `xarray.Dataset`. Raster data is stored in `xarray.Dataset` [@hoyer_xarray_2017], vector data in `geopandas.GeoDataFrame` [@bossche_geopandas_2025]. These public sources can be very broad such as digital terrain models [@AHN], large-scale base maps [@BGT], bathymetry models, geological or geomorphological schematizations [@REGIS;@GEOTOP;@BOFEK;@BRO], meteorological datasets [@KNMI] etc.

At the moment, predominantly open data sources from the Netherlands are supported. However, users can (download and) add any local dataset manually.

## Discretizate: `nlmod.dims`
The `nlmod.dims` module allows users to create and adapt both the spatial discretization (e.g., extent, layers, grid) and temporal discretization using an `xarray.Dataset`. Functionality includes data manipulation, such as calculating weighted means and filling not-a-number values, and discritisation methods for both structured and vertex grids. Model data is stored in a single xarray.Dataset (for raster) and a number of geodataframes (for vector data).

## Building groundwater models: `nlmod.sim`, `nlmod.gwf`, `nlmod.gwt`, `nlmod.prt`, and `nlmod.modpath`
Using the data in the xarray.Dataset, NLMOD provides tools to build MODFLOW 6 [@MODFLOW6] models via `FloPy` [@bakker_flopy_2016;@hughes_flopy_2024].
The modules `nlmod.sim`, `nlmod.gwf`, `nlmod.gwt`, and `nlmod.prt` are designed for MODFLOW 6, while `nlmod.modpath` enables particle tracking using MODPATH [@MODPATH7].
This modular approach supports full scriptability which enables complex model optimization schemes.

## Visualize model data: `nlmod.plot` and `nlmod.gis`
To interpret and communicate model inputs and outputs, `NLMOD` offers built-in reading methods(`nlmod.mfoutput`), plotting functionality (`nlmod.plot`) and GIS tools (`nlmod.gis`) for exporting data to commonly used geospatial formats. These visualization options support both technical analysis and clear communication with stakeholders.

# Note of thanks
We thank the following institutions and contributors for their support to the development of `NLMOD`.

# References