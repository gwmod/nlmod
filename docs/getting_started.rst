===============
Getting Started
===============
On this page you will find information on how to get started with nlmod.

Getting Python
--------------
To install nlmod, a working version of Python 3.7 or higher has to be
installed on your computer. We recommend using the
`Anaconda Distribution <https://www.continuum.io/downloads>`_
of Python.

Installing nlmod
----------------
Install the module by typing::
  
    pip install nlmod


For installing in development mode, clone the repository and install by
typing ``pip install -e .`` from the module root directory.

Using nlmod
-----------
This section provides a brief introduction to nlmod. To start using nlmod,
start Python and import the module::

    import nlmod

nlmod contains many methods for loading data from online data sources, .e.g.
from the hydrogeological subsurface model REGIS::

    extent = [116_500, 120_000, 439_000, 442_000]  # define extent
    regis_ds = nlmod.read.regis.get_regis(extent)  # download REGIS data

These methods are accessible through ``nlmod.read``. Supported data sources include:

* REGIS, hydrogeological model (raster)
* GeoTOP, geological model for the shallow subsurface (raster)
* AHN, the digital elevation model (raster)
* Jarkus, bathymetry data for the North Sea (raster)
* BGT, surface water features (shapefile)
* Rijkswaterstaat, surface water polygons for large water bodies (shapefile)
* Waterboards, surface water level data (shapefile)
* KNMI, precipitation and evaporation data (timeseries)

The basic building block for groundwater models in nlmod is the model Dataset.
A model dataset can either be created from scratch, or derived from a
hydrogeological subsurface model such as REGIS::

    ds = nlmod.get_ds(extent)  # start from scratch OR
    ds = nlmod.to_model_ds(regis_ds)  # convert REGIS data to a model dataset

This dataset should eventually contain all the (rasterized) information
required to build the groundwater model. Manipulating the spatial
discretization is possible through all kinds of functions in ``nlmod.grid`` and
``nlmod.resample``. Adding time settings to the model dataset is done
through functions in ``nlmod.time``.

Building the groundwater model is performed using flopy. nlmod is not a
replacement of flopy, but does offer convenience methods to build MODFLOW
models using the model dataset as input. An minimal example of code to build a
MODFLOW 6 model given a model dataset::

    sim = nlmod.sim.sim(ds, model_ws="./my_model") # MFSimulation
    tdis = nlmod.sim.tdis(ds)  # time discretization
    ims = nlmod.sim.ims(ds)    # ims solver
    gwf = nlmod.gwf.gwf(ds)    # groundwater flow model
    dis = nlmod.gwf.dis(ds)    # spatial discretization
    npf = nlmod.gwf.npf(ds)    # node property flow
    oc = nlmod.gwf.oc(ds)      # output control

    # ... add some boundary condition packages (GHB, RIV, DRN, ...)

Running the model requires the modflow binaries provided by the USGS. Those can
be downloaded with::

    nlmod.util.download_mfbinaries()

Writing and running the model can then be done using::

    nlmod.sim.write_and_run(ds, sim)

The output from a model can be read using::

    head = nlmod.gwf.output(ds)

And plotting the mean head in the top model layer::

    nlmod.plot.data_array(head.sel(layer=0).mean("time"))
   
This was a very brief overview of some of the features in nlmod. There is a lot
more to discover, so we recommend taking a look at :ref:`examples` section for
more detailed examples.

Dependencies
------------

This module has the following dependencies that should be installed
automatically when installing nlmod. If you run into any trouble with any of 
these packages during installation, refer to 
`this page <https://github.com/ArtesiaWater/hydropandas#dependencies>`_ for
potential solutions.

- xarray
- geopandas
- shapely
- flopy
- rasterio
- rioxarray
- affine
- owslib
- netcdf4
- mfpymake
- hydropandas
- dask
- colorama
- matplotlib
