"""
functions to create modflow models


"""


import os
from nlmod import (mtime, mgrid, recharge, surface_water, util,
                   mfpackages, regis, ahn)


def gen_model_structured(model_ws,
                         model_name,
                         use_cache=False,
                         verbose=False,
                         steady_state=False,
                         start_time='2015-1-1',
                         transient_timesteps=5,
                         perlen=1.,
                         steady_start=True,
                         extent=[95000., 150000., 487000., 553500.],
                         delr=100.,
                         delc=100.,
                         angrot=0,
                         length_units='METERS',
                         use_regis=True,
                         regis_botm_layer=b'AKc',
                         use_geotop=True,
                         remove_nan_layers=True,
                         add_northsea=True, 
                         add_surface_water_ghb=True,
                         add_surface_drn=True, 
                         surface_drn_cond=1000,
                         add_chd_edges=True, 
                         add_recharge=True,
                         anisotropy=10, 
                         icelltype=0,
                         fill_value_kh=1.,
                         fill_value_kv=0.1,
                         starting_head=1.0,
                         write_sim=False,
                         run_sim=False):
    """ generate a model with structured grid


    Parameters
    ----------
    model_ws : str
        model workspace, model data is written to this directory.
    model_name : str
        name of the model.
    use_cache : bool, optional
        if True an attempt is made to read model data from cache.
        The default is False.
    verbose : bool, optional
        print additional information to the console. default is False
    steady_state : bool
        if True the model is steady state with one time step.
    start_time : str or datetime
        start time of the model. This is the start_time of the transient time
        steps and the end_time of the steady_state periods (if steady_state is
        True or steady_start is True).
    transient_timesteps : int, optional
        number of transient time steps. The default is 0.
    perlen : float, int, list or np.array, optional
        length of each timestep depending on the type:
            - float or int: this is the length of all the time steps. If 
            steady_start is True the length of the first time step is defined
            by steady_perlen
            - list or array: the items are the length per timestep.
            the length of perlen should match the number of transient 
            timesteps (or transient timesteps +1 if steady_start is True) 
        The default is 1.0.
    steady_start : bool
        if True the model is transient with a steady state start time step.
    extent : list, tuple or np.array
        extent (xmin, xmax, ymin, ymax) of the desired grid.
    delr : int or float
        cell size along rows of the desired grid (dx).
    delc : int or float
        cell size along columns of the desired grid (dy).
    angrot : int or float, optional
        rotation angle. The default is 0.
    length_units : str, optional
        length unit. The default is 'METERS'.
    use_regis : bool, optional
        True if part of the layer model should be REGIS. The default is True.
    regis_botm_layer : binary str, optional
        regis layer that is used as the bottom of the model. This layer is
        included in the model. the Default is b'AKc' which is the bottom
        layer of regis. call nlmod.regis.get_layer_names() to get a list of
        regis names.
    use_geotop : bool, optional
        True if part of the layer model should be geotop. The default is True.
    remove_nan_layers : bool, optional
        if True regis and geotop layers with only nans are removed from the 
        model. if False nan layers are kept which might be usefull if you want 
        to keep some layers that exist in other models. The default is True.
    add_northsea : bool, optional
        if True:
            - the nan values at the northsea are filled with a certain value
            - the seabed is added to the layer model using the bathymetry from
            jarkus
            - the model layers below the seabed are extrapolated under the
            seabed
    add_surface_water_ghb : bool, optional
        if True surface water is read from a shapefile and added to the model
        as a general head boundary. ghb conductance and level are computed
        from values in the attribute table of the shapefile. The default is
        True.
    add_surface_drn : bool, optional
        if True a surface drain (maaivelddrainage) is added to the model. The
        default is True.
    surface_drn_cond : int or float, optional
        conductance of the surface drain (maaivelddrainage) only used if
        add_surface_drn is True. The default is 1000.
    add_chd_edges : bool, optional
        add constant head boundary at the model boundaries. Use the starting
        head as default chd level. The default is False.
    add_recharge : bool, optional
        if True recharge is added based on precipitation and evaporation from
        the knmi. The default is True.
    anisotropy : int or float
        factor to calculate kv from kh or the other way around
    icelltype : int, optional
        celltype, 0 is used for confined cells and the only celltype currently
        supported. The default is 0.
    fill_value_kh : int or float, optional
        use this value for kh if there is no data in regis. The default is 1.0.
    fill_value_kv : int or float, optional
        use this value for kv if there is no data in regis. The default is 0.1.
    starting_head : float, optional
        starting head. The default is 1.0.
    write_sim : bool, optional
        if True model data is written to the model_ws directory. The default
        is False.
    run_sim : bool, optional
        if True run the model, fails if write_sim is False. The default is
        False.

    Returns
    -------
    model_ds : xarray Dataset
        dataset with model data.
    gwf : flopy.mf6.modflow.mfgwf.ModflowGwf
        groundwater flow model.

    """
    
    #checks
    if length_units!='METERS':
        raise NotImplementedError()

    gridtype = 'structured'

    # Model directories
    figdir, cachedir = util.get_model_dirs(model_ws)

    # create model time dataset
    model_ds = mtime.get_model_ds_time(model_name, model_ws, start_time,
                                       steady_state,
                                       steady_start,
                                       transient_timesteps=transient_timesteps,
                                       perlen=perlen)

    sim, gwf = mfpackages.sim_tdis_gwf_ims_from_model_ds(model_ds,
                                                         verbose)

    extent, nrow, ncol = regis.fit_extent_to_regis(extent,
                                                   delr,
                                                   delc,
                                                   verbose=verbose)

    # layer model
    layer_model = regis.get_layer_models(extent, delr, delc,
                                         use_regis=use_regis,
                                         regis_botm_layer=regis_botm_layer,
                                         use_geotop=use_geotop,
                                         remove_nan_layers=remove_nan_layers,
                                         cachedir=cachedir,
                                         fname_netcdf='combined_layer_ds.nc',
                                         use_cache=use_cache,
                                         verbose=verbose)

    # update model_ds from layer model
    model_ds = mgrid.update_model_ds_from_ml_layer_ds(
        model_ds,
        layer_model,
        keep_vars=['x', 'y'],
        gridtype=gridtype,
        anisotropy=anisotropy,
        fill_value_kh=fill_value_kh,
        fill_value_kv=fill_value_kv,
        add_northsea=add_northsea,
        verbose=verbose)

    # Create discretization
    mfpackages.dis_from_model_ds(model_ds, gwf,
                                 angrot=angrot,
                                 length_units=length_units)

    # create node property flow
    mfpackages.npf_from_model_ds(model_ds, gwf, icelltype=icelltype)

    # Create the initial conditions package
    mfpackages.ic_from_model_ds(model_ds, gwf, starting_head=starting_head)

    # Create the output control package
    mfpackages.oc_from_model_ds(model_ds, gwf)

    # voeg grote oppervlaktewaterlichamen toe
    if add_surface_water_ghb:
        da_name = 'surface_water'
        model_ds = surface_water.get_general_head_boundary(model_ds,
                                                           gwf.modelgrid,
                                                           da_name,
                                                           cachedir=cachedir,
                                                           use_cache=use_cache,
                                                           verbose=verbose)
        mfpackages.ghb_from_model_ds(model_ds, gwf, da_name)

    # surface level drain
    if add_surface_drn:
        model_ds = ahn.get_ahn_dataset(model_ds, use_cache=use_cache,
                                       cachedir=cachedir, verbose=verbose)

        mfpackages.surface_drain_from_model_ds(
            model_ds,
            gwf,
            surface_drn_cond=surface_drn_cond
        )

    if add_recharge:
        # add knmi recharge to the model datasets
        model_ds = recharge.get_recharge(model_ds,
                                         verbose=verbose,
                                         cachedir=cachedir,
                                         use_cache=use_cache)
        # create recharge package
        mfpackages.rch_from_model_ds(model_ds, gwf)

    # add constant head cells at model boundaries
    if add_chd_edges:
        mfpackages.chd_at_model_edge_from_model_ds(model_ds, gwf,
                                                   head='starting_head')

    # save model_ds
    model_ds.to_netcdf(os.path.join(cachedir, 'full_model_ds.nc'))

    if write_sim:
        sim.write_simulation()

    if run_sim:
        success, buff = sim.run_simulation()
        print('\nSuccess is: ', success)

    return model_ds, gwf


def gen_model_unstructured(model_ws,
                           model_name,
                           refine_shp_fname='',
                           use_cache=False,
                           verbose=False,
                           levels=2,
                           steady_state=False,
                           start_time='2015-1-1',
                           transient_timesteps=5,
                           perlen=1.,
                           steady_start=True,
                           extent=[95000., 150000., 487000., 553500.],
                           delr=100.,
                           delc=100.,
                           angrot=0,
                           length_units='METERS',
                           use_regis=True,
                           use_geotop=True,
                           add_northsea=True,
                           add_surface_water_ghb=True,
                           add_surface_drn=True,
                           surface_drn_cond=1000,
                           add_chd_edges=True,
                           add_recharge=True,
                           anisotropy=10,
                           icelltype=0,
                           fill_value_kh=1.,
                           fill_value_kv=0.1,
                           starting_head=1.0,
                           write_sim=False,
                           run_sim=False):
    """ generate a model with an unstructured grid


    Parameters
    ----------
    model_ws : str
        model workspace, model data is written to this directory.
    model_name : str
        name of the model.
    refine_shp_fname : str, optional
        full path of the shapefile that is used to locally refine the grid
        using gridgen
    levels : int, optional
        number of levels to refine. If the original cellsize is 100x100 then
        the refined cells get the following cell size per level:
            _________________________________
           |level   | cell size refined cells|
           -----------------------------------
           |1       |         50x50          |
           |2       |         25x25          |
           |3       |       12.5x12.5        |
           |4       |       6.25x6.25        |
           -----------------------------------

        The default is 2
    use_cache : bool, optional
        if True an attempt is made to read model data from cache.
        The default is False.
    verbose : bool, optional
        print additional information to the console. default is False
    steady_state : bool
        if True the model is steady state with one time step.
    start_time : str or datetime
        start time of the model.
    transient_timesteps : int, optional
        number of transient time steps. The default is 0.
    perlen : float, int, list or np.array, optional
        length of each timestep depending on the type:
            - float or int: this is the length of all the time steps. If 
            steady_start is True the length of the first time step is defined
            by steady_perlen
            - list or array: the items are the length per timestep.
            the length of perlen should match the number of transient 
            timesteps (or transient timesteps +1 if steady_start is True) 
        The default is 1.0.
    steady_start : bool
        if True the model is transient with a steady state start time step.
    extent : list, tuple or np.array
        extent (xmin, xmax, ymin, ymax) of the desired grid.
    delr : int or float
        cell size along rows of the desired grid (dx). The size along rows of
        refined cells will be a factor of this defined by levels.
    delc : int or float
        cell size along columns of the desired grid (dy). The size along
        columns of refined cells will be a factor of this defined by levels.
    angrot : int or float, optional
        rotation angle. The default is 0.
    length_units : str, optional
        length unit. The default is 'METERS'.
    use_regis : bool, optional
        True if part of the layer model should be REGIS. The default is True.
    use_geotop : bool, optional
        True if part of the layer model should be geotop. The default is True.
    add_northsea : bool, optional
        if True:
            - the nan values at the northsea are filled with a certain value
            - the seabed is added to the layer model using the bathymetry from
            jarkus
            - the model layers below the seabed are extrapolated under the
            seabed
    add_surface_water_ghb : bool, optional
        if True surface water is read from a shapefile and added to the model
        as a general head boundary. ghb conductance and level are computed
        from values in the attribute table of the shapefile. The default is
        True.
    add_surface_drn : bool, optional
        if True a surface drain (maaivelddrainage) is added to the model. The
        default is True.
    surface_drn_cond : int or float, optional
        conductance of the surface drain (maaivelddrainage) only used if
        add_surface_drn is True. The default is 1000.
    add_chd_edges : bool, optional
        add constant head boundary at the model boundaries. Use the starting
        head as default chd level. The default is False.
    add_recharge : bool, optional
        if True recharge is added based on precipitation and evaporation from
        the knmi. The default is True.
    anisotropy : int or float
        factor to calculate kv from kh or the other way around
    icelltype : int, optional
        celltype, 0 is used for confined cells and the only celltype currently
        supported. The default is 0.
    fill_value_kh : int or float, optional
        use this value for kh if there is no data in regis. The default is 1.0.
    fill_value_kv : int or float, optional
        use this value for kv if there is no data in regis. The default is 0.1.
    starting_head : float, optional
        starting head. The default is 1.0.
    write_sim : bool, optional
        if True model data is written to the model_ws directory. The default
        is False.
    run_sim : bool, optional
        if True run the model, fails if write_sim is False. The default is
        False.

    Returns
    -------
    model_ds : xarray Dataset
        dataset with model data.
    gwf : flopy.mf6.modflow.mfgwf.ModflowGwf
        groundwater flow model.
    gridprops : dictionary
        dictionary with grid properties output from gridgen.

    """

    gridtype = 'unstructured'

    # Model directories
    figdir, cachedir, gridgen_ws = util.get_model_dirs(model_ws,
                                                       gridtype=gridtype)

    # create model time dataset
    model_ds = mtime.get_model_ds_time(model_name, model_ws, start_time,
                                       steady_state,
                                       steady_start,
                                       transient_timesteps=transient_timesteps,
                                       perlen=perlen)

    # create model simulation packages
    sim, gwf = mfpackages.sim_tdis_gwf_ims_from_model_ds(model_ds,
                                                         verbose)

    extent, nrow, ncol = regis.fit_extent_to_regis(extent, delr, delc,
                                                   verbose=verbose)

    # layer model
    layer_model = regis.get_layer_models(extent, delr, delc,
                                         use_regis=use_regis,
                                         use_geotop=use_geotop,
                                         cachedir=cachedir,
                                         fname_netcdf='combined_layer_ds.nc',
                                         use_cache=use_cache,
                                         verbose=verbose)

    # use gridgen to create unstructured grid
    gridprops = mgrid.create_unstructured_grid(gridgen_ws, model_name, gwf,
                                               refine_shp_fname, levels, extent,
                                               layer_model.dims['layer'],
                                               nrow, ncol,
                                               delr, delc,
                                               cachedir=cachedir, use_cache=use_cache,
                                               verbose=verbose)

    # add layer model to unstructured grid
    layer_model_unstr = mgrid.get_ml_layer_dataset_unstruc(raw_ds=layer_model,
                                                           extent=extent,
                                                           gridprops=gridprops,
                                                           cachedir=cachedir,
                                                           fname_netcdf='layer_model_unstr.nc',
                                                           use_cache=use_cache,
                                                           verbose=verbose)

    # combine model time dataset with layer model dataset
    model_ds = mgrid.update_model_ds_from_ml_layer_ds(model_ds,
                                                      layer_model_unstr,
                                                      gridtype,
                                                      keep_vars=['x', 'y'],
                                                      gridprops=gridprops,
                                                      add_northsea=add_northsea,
                                                      verbose=verbose)

    # Create discretization
    mfpackages.disv_from_model_ds(model_ds, gwf, gridprops,
                                  angrot=angrot,
                                  length_units=length_units)

    # create node property flow
    mfpackages.npf_from_model_ds(model_ds, gwf, icelltype=icelltype)

    # Create the initial conditions package
    mfpackages.ic_from_model_ds(model_ds, gwf, starting_head=starting_head)

    # Create the storage package
    mfpackages.sto_from_model_ds(model_ds, gwf)

    # Create the output control package
    mfpackages.oc_from_model_ds(model_ds, gwf)

    # add surface water (only big lakes and northsea)
    if add_surface_water_ghb:
        model_ds = surface_water.get_general_head_boundary(model_ds,
                                                           gwf.modelgrid,
                                                           'surface_water',
                                                           cachedir=cachedir,
                                                           use_cache=use_cache,
                                                           verbose=verbose)

        # create ghb package
        mfpackages.ghb_from_model_ds(model_ds, gwf, 'surface_water')

    # surface level drain
    if add_surface_drn:

        model_ds = ahn.get_ahn_dataset(model_ds, gridprops=gridprops,
                                       use_cache=use_cache,
                                       cachedir=cachedir, verbose=verbose)

        mfpackages.surface_drain_from_model_ds(model_ds, gwf,
                                               surface_drn_cond=surface_drn_cond
                                               )

    # add constant head cells at model boundaries
    if add_chd_edges:
        mfpackages.chd_at_model_edge_from_model_ds(model_ds, gwf,
                                                   head='starting_head')

    if add_recharge:
        # add knmi recharge to the model datasets
        model_ds = recharge.get_recharge(model_ds,
                                         verbose=verbose,
                                         cachedir=cachedir,
                                         use_cache=use_cache)
        # create recharge package
        mfpackages.rch_from_model_ds(model_ds, gwf)

    # save model_ds
    model_ds.to_netcdf(os.path.join(cachedir, 'full_model_ds.nc'))

    if write_sim:
        sim.write_simulation()

    if run_sim:
        success, buff = sim.run_simulation()
        print('\nSuccess is: ', success)

    return model_ds, gwf, gridprops
