"""
functions to create modflow models


"""


import os
from nlmod import (mtime, mgrid, recharge, surface_water, util,
                   northsea, mfpackages, regis, ahn)


def gen_model_structured(model_ws, model_name, use_cache=False,
                         verbose=False,
                         steady_state=False, start_time='2015-1-1',
                         transient_timesteps=5, steady_start=True,
                         extent=[95000., 150000., 487000., 553500.],
                         delr=100., delc=100., angrot=0,
                         length_units='METERS',
                         use_regis=True, use_geotop=True,
                         add_northsea=True,
                         anisotropy=10, icelltype=0,
                         fill_value_kh=1.,
                         fill_value_kv=0.1, surface_drn=True,
                         surface_drn_cond=1000, starting_head=1.0,
                         constant_head_edges=False, write_sim=False,
                         run_sim=False):

    gridtype = 'structured'

    # Model directories
    figdir, cachedir = util.get_model_dirs(model_ws)

    # create model time dataset
    model_ds = mtime.get_model_ds_time(model_name, model_ws, start_time,
                                       steady_state,
                                       steady_start,
                                       transient_timesteps=transient_timesteps)

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

    # update model_ds from layer model
    model_ds = mgrid.update_model_ds_from_ml_layer_ds(model_ds, layer_model,
                                                      keep_vars=['x', 'y'],
                                                      gridtype=gridtype,
                                                      anisotropy=anisotropy,
                                                      fill_value_kh=fill_value_kh,
                                                      fill_value_kv=fill_value_kv,
                                                      add_northsea=add_northsea,
                                                      verbose=verbose)

    # Create discretization
    mfpackages.dis_from_model_ds(model_ds, gwf, angrot=angrot,
                                 length_units=length_units)

    # create node property flow
    mfpackages.npf_from_model_ds(model_ds, gwf, icelltype=icelltype)

    # voeg grote oppervlaktewaterlichamen toe
    da_name = 'surface_water'
    model_ds = surface_water.get_general_head_boundary(model_ds,
                                                       gwf.modelgrid, da_name,
                                                       cachedir=cachedir,
                                                       use_cache=use_cache,
                                                       verbose=verbose)
    mfpackages.ghb_from_model_ds(model_ds, gwf, da_name)

    # surface level drain
    if surface_drn:

        model_ds = ahn.get_ahn_dataset(model_ds, use_cache=use_cache,
                                       cachedir=cachedir, verbose=verbose)

        mfpackages.surface_drain_from_model_ds(model_ds, gwf,
                                               surface_drn_cond=surface_drn_cond
                                               )

    # Create the initial conditions package
    mfpackages.ic_from_model_ds(model_ds, gwf, starting_head=starting_head)

    # add constant head cells at model boundaries
    if constant_head_edges:
        mfpackages.chd_at_model_edge_from_model_ds(model_ds, gwf,
                                                   head='starting_head')

    # add knmi recharge to the model datasets
    model_ds = recharge.get_recharge(model_ds,
                                     verbose=verbose,
                                     cachedir=cachedir,
                                     use_cache=use_cache)
    # create recharge package
    mfpackages.recharge_from_model_ds(model_ds, gwf)

    # Create the output control package
    mfpackages.oc_from_model_ds(model_ds, gwf)

    # save model_ds
    model_ds.to_netcdf(os.path.join(cachedir, 'full_model_ds.nc'))

    if write_sim:
        sim.write_simulation()

    if run_sim:
        success, buff = sim.run_simulation()
        print('\nSuccess is: ', success)

    return model_ds, gwf


def gen_model_unstructured(model_ws, model_name,
                           refine_shp_fname='',
                           use_cache=False,
                           verbose=False,
                           levels=2,
                           steady_state=False, start_time='2015-1-1',
                           transient_timesteps=5, steady_start=True,
                           extent=[95000., 150000., 487000., 553500.],
                           delr=100., delc=100., angrot=0,
                           length_units='METERS',
                           use_regis=True, use_geotop=True,
                           add_northsea=True,
                           anisotropy=10, icelltype=0,
                           fill_value_kh=1.,
                           fill_value_kv=0.1, surface_drn=True,
                           surface_drn_cond=1000, starting_head=1.0,
                           constant_head_edges=False, write_sim=False,
                           run_sim=False):

    gridtype = 'unstructured'

    # Model directories
    figdir, cachedir, gridgen_ws = util.get_model_dirs(model_ws,
                                                       gridtype=gridtype)

    # create model time dataset
    model_ds = mtime.get_model_ds_time(model_name, model_ws, start_time,
                                       steady_state,
                                       steady_start,
                                       transient_timesteps=transient_timesteps)

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

    # add surface water (only big lakes and northsea)
    model_ds = surface_water.get_general_head_boundary(model_ds,
                                                       gwf.modelgrid,
                                                       'surface_water',
                                                       cachedir=cachedir,
                                                       use_cache=use_cache,
                                                       verbose=verbose)

    # create ghb package
    mfpackages.ghb_from_model_ds(model_ds, gwf, 'surface_water')

    # surface level drain
    if surface_drn:

        model_ds = ahn.get_ahn_dataset(model_ds, gridprops=gridprops,
                                       use_cache=use_cache,
                                       cachedir=cachedir, verbose=verbose)

        mfpackages.surface_drain_from_model_ds(model_ds, gwf,
                                               surface_drn_cond=surface_drn_cond
                                               )

    # Create the initial conditions package
    mfpackages.ic_from_model_ds(model_ds, gwf, starting_head=starting_head)

    # Create the storage package
    mfpackages.sto_from_model_ds(model_ds, gwf)

    # add constant head cells at model boundaries
    if constant_head_edges:
        mfpackages.chd_at_model_edge_from_model_ds(model_ds, gwf,
                                                   head='starting_head')

    # add knmi recharge to the model datasets
    model_ds = recharge.get_recharge(model_ds,
                                     verbose=verbose,
                                     cachedir=cachedir,
                                     use_cache=use_cache)
    # create recharge package
    mfpackages.recharge_from_model_ds(model_ds, gwf)

    # Create the output control package
    mfpackages.oc_from_model_ds(model_ds, gwf)

    # save model_ds
    model_ds.to_netcdf(os.path.join(cachedir, 'full_model_ds.nc'))

    if write_sim:
        sim.write_simulation()

    if run_sim:
        success, buff = sim.run_simulation()
        print('\nSuccess is: ', success)

    return model_ds, gwf, gridprops
