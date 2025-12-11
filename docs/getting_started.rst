===============
Getting Started
===============
On this page you will find information on how to get started with `nlmod`.

Getting Python
--------------
To install `nlmod`, a working version of Python 3.10 or higher has to be
installed on your computer. We recommend using the
`Anaconda Distribution <https://www.continuum.io/downloads>`_
of Python.

Installing `nlmod`
----------------
Install the module by typing::

    pip install nlmod


For installing in development mode, clone the repository and install by
typing ``pip install -e .`` from the module root directory.

Using `nlmod`
-----------
This section provides a brief introduction to `nlmod`. To start using `nlmod`,
start Python and import the module::

    import nlmod

`nlmod` contains many methods for loading data from online data sources, .e.g.
from the hydrogeological subsurface model REGIS (see :ref:`Data Sources` for more
information on available data sources)::

    extent = [116_500, 120_000, 439_000, 442_000]  # define extent
    regis_ds = nlmod.read.regis.download_regis(extent)  # download REGIS data

The basic building block for groundwater models in nlmod is the model Dataset.
A model dataset can either be created from scratch, or derived from a
hydrogeological subsurface model such as REGIS::

    # start from scratch
    ds = nlmod.get_ds(extent, model_ws="./my_model", model_name="my_model")
	
    # OR convert REGIS data to a model dataset
    ds = nlmod.to_model_ds(regis_ds, model_ws="./my_model", model_name="my_model")

This dataset should eventually contain all the (rasterized) information
required to build the groundwater model. Manipulating the spatial
discretization is possible through all kinds of functions in ``nlmod.grid`` and
``nlmod.resample``. Adding time settings to the model dataset is done
through functions in ``nlmod.time``.

Building the groundwater model is performed using flopy. `nlmod` is not a
replacement of flopy, but does offer convenience methods to build MODFLOW
models using the model dataset as input. An minimal example of code to build a
MODFLOW 6 model given a model dataset::

    sim = nlmod.sim.sim(ds)                       # simulation
    tdis = nlmod.sim.tdis(ds, sim)                # time discretization
    ims = nlmod.sim.ims(sim)                      # ims solver
    gwf = nlmod.gwf.gwf(ds, sim)                  # groundwater flow model
    dis = nlmod.gwf.dis(ds, gwf)                  # spatial discretization
    npf = nlmod.gwf.npf(ds, gwf)                  # node property flow
    sto = nlmod.gwf.sto(ds, gwf)                  # storage (if transient)
    ic = nlmod.gwf.ic(ds, gwf, starting_head=0.0) # initial conditions
    oc = nlmod.gwf.oc(ds, gwf)                    # output control

    # ... add some boundary condition packages (RCH, GHB, RIV, DRN, ...)

The MODFLOW 6 executable is automatically downloaded and installed to your system
when building the first model.

Writing and running the model can then be done using::

    nlmod.sim.write_and_run(sim, ds)

The output from a model can be read using::

    head = nlmod.gwf.get_heads_da(ds)

And plotting the mean head in one of the layers::

    nlmod.plot.map_array(head.sel(layer="BXz2").mean("time"), ds=ds)

This was a very brief overview of some of the features in `nlmod`. There is a lot
more to discover, so we recommend taking a look at the  sections :ref:`Data Sources`,
:ref:`Example Models`, :ref:`Utilities`, :ref:`Workflows`, and :ref:`Advanced Stress Packages`.

Dependencies
------------

This module has the following dependencies that should be installed
automatically when installing `nlmod`. If you run into any trouble with any of
these packages during installation, refer to
`this page <https://github.com/ArtesiaWater/hydropandas#dependencies>`_ for
potential solutions.

- flopy
- xarray
- netcdf4
- rioxarray
- geopandas
- matplotlib
- dask
- requests
- scipy
- bottleneck

On top of that there are some optional dependecies:

- geocube (used in nlmod.util.zonal_statistics)
- rasterstats (used in nlmod.util.zonal_statistics)
- contextily (nlmod.plot.add_background_map)
- scikit-image (used in nlmod.read.rws.calculate_sea_coverage)
- py7zr (used in nlmod.read.bofek.download_bofek_gdf)
- joblib (used in nlmod.cache)
- colorama (used in nlmod.util.get_color_logger)
- tqdm (used for showing progress in long-running methods)
- hydropandas (used in nlmod.read.knmi and nlmod.read.bro)
- owslib (used in nlmod.read.ahn.get_latest_ahn_from_wcs)
- pyshp (used in nlmod.grid.refine)
- h5netcdf (used in nlmod.read.knmi_data_platform)

These dependencies are only needed (and imported) in a single module or method.
They can be installed using ``pip install nlmod[full]`` or ``pip install -e .[full]``.
