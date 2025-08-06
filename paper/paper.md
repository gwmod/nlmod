---
title: 'NLMOD: From open data to MODFLOW models with Python'
tags:
  - hydrology
  - groundwater
  - Python
authors:
  - name: Ruben Caljé
    orcid:
    affiliation: "1"
  - name: Onno Ebbens
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
`NLMOD` is a Python package to build, run and visualize MODFLOW 6 [@MODFLOW6] groundwater models in the Netherlands.
It handles open-data efficiently via the popular Python package `xarray` [@hoyer_xarray_2017].
`FloPy` [@bakker_flopy_2016,hughes_flopy_2024] is used for for creating and running the MODFLOW-based groundwater flow, transport and particle tracking models.

# Statement of need
`NLMOD` was built to allow users to write scripts to quickly download relevant data from publicly available sources, and build and post-process groundwater flow [@MODFLOW6GFM] and transport [@MODFLOW6GTM] models at different spatial and temporal scales to answer specific geohydrological questions. Scripting these steps, from downloading data to building groundwater models, makes models more reproducible and transparent.

# Package setup
The functions in `NLMOD` have four main objectives:

- Download and read data from external sources, project this data on the modelgrid and add this data to an `xarray.Dataset`: `nlmod.read`.
- Create and adapt the temporal and spatial discretization of a MODFLOW model using an `xarray.Dataset`: `nlmod.dims`
- Use data in an `xarray.Dataset` to build modflow packages for both groundwater flow and transport models using FloPy [@bakker_flopy_2016,hughes_flopy_2024]: `nlmod.sim`, `nlmod.gwf`, `nlmod.gwt` & `nlmod.prt` for MODFLOW 6 [@MODFLOW6] and `nlmod.modpath` for Modpath [@MODPATH7].
- Visualise modeldata in Python: `nlmod.plot` or GIS software: `nlmod.gis`


# Note of thanks
We thank the following institutions and contributors for their support to the development of `NLMOD`.

# References