# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 12:11:27 2021

@author: oebbe
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 12:58:58 2020

@author: oebbe
"""

# flopy

# other




from shapely.geometry import box
from scipy.interpolate import griddata
from scipy import interpolate
import xarray as xr
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
import numpy as np
import datetime as dt
import os
from flopy.discretization.structuredgrid import StructuredGrid
import flopy.discretization as fgrid
from flopy.utils.gridintersect import GridIntersect
import flopy

import nlmod
from nlmod import (mtime, mgrid, recharge, surface_water, util, well,
                   mfpackages, regis, northsea)


def gen_model_structured(model_ws, model_name, use_cache=False,
                         verbose=False,
                         steady_state=False, start_time='2015-1-1',
                         transient_timesteps=5, steady_start=True,
                         extent=[95000., 150000., 487000., 553500.],
                         delr=100., delc=100., angrot=0,
                         length_units='METERS', regis=True, geotop=True,
                         anisotropy=10,
                         confined=True, fill_value_kh=1.,
                         fill_value_kv=0.1, maaivelddrainage=True,
                         maaiveldrn_cond=1000, starting_head=1.0,
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

    # layer model
    layer_model = regis.get_layer_models(extent, delr, delc,
                                         regis=regis, geotop=geotop,
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
                                                      verbose=verbose)

    # find grid cells with sea
    model_ds = northsea.get_modelgrid_sea(model_ds,
                                          cachedir=cachedir,
                                          use_cache=use_cache, 
                                          verbose=verbose)

    # fill top, bot, kh, kv at sea cells
    fill_mask = (model_ds['first_active_layer'] == model_ds.nodata) * model_ds['sea']
    model_ds = mgrid.fill_top_bot_kh_kv_at_mask(model_ds, fill_mask)

    # add bathymetry noordzee
    model_ds = northsea.get_modelgrid_bathymetry(model_ds,
                                                 cachedir=cachedir,
                                                 use_cache=use_cache,
                                                 verbose=verbose)
    model_ds = mgrid.add_bathymetry_to_top_bot_kh_kv(model_ds,
                                                     model_ds['bathymetry'], fill_mask)

    # Create discretization
    # update idomain on adjusted tops and bots
    model_ds['thickness'] = mgrid.get_thickness_from_topbot(model_ds['top'],
                                                            model_ds['bot'])
    model_ds['idomain'] = mgrid.update_idomain_from_thickness(model_ds['idomain'],
                                                              model_ds['thickness'],
                                                              model_ds['sea'])
    model_ds['first_active_layer'] = mgrid.get_first_active_layer_from_idomain(model_ds['idomain'])

    flopy.mf6.ModflowGwfdis(gwf,
                            pname='dis',
                            length_units=length_units,
                            xorigin=model_ds.extent[0],
                            yorigin=model_ds.extent[2],
                            angrot=angrot,
                            nlay=model_ds.dims['layer'],
                            nrow=model_ds.dims['y'],
                            ncol=model_ds.dims['x'],
                            delr=model_ds.delr,
                            delc=model_ds.delc,
                            top=model_ds['top'].data,
                            botm=model_ds['bot'].data,
                            idomain=model_ds['idomain'].data,
                            filename='{}.dis'.format(model_name))
    
    
    
    # define flow parameters
    if confined:
        icelltype = 0
    else:
        raise NotImplementedError()
        
    # create node property flow
    flopy.mf6.modflow.mfgwfnpf.ModflowGwfnpf(gwf,
                                            pname='npf',
                                            icelltype=icelltype,
                                            k=model_ds['kh'].data,
                                            k33=model_ds['kv'].data,
                                            save_flows=True)

    # voeg grote oppervlaktewaterlichamen toE
    name = 'oppervlaktewater'
    model_ds = surface_water.get_general_head_boundary(model_ds, gdf_opp_water,
                                                       gwf.modelgrid, name,
                                                       cachedir=cachedir,
                                                       use_cache=use_cache,
                                                       verbose=verbose)

    ghb_rec = mgrid.data_array_2d_to_rec_list(model_ds,
                                              model_ds[f'{name}_cond'] != 0,
                                              col1=f'{name}_peil',
                                              col2=f'{name}_cond',
                                              first_active_layer=True,
                                              only_active_cells=False,
                                              layer=0)
    if len(ghb_rec) > 0:
        flopy.mf6.modflow.mfgwfghb.ModflowGwfghb(gwf, print_input=True,
                                                 maxbound=len(ghb_rec),
                                                 stress_period_data=ghb_rec,
                                                 save_flows=True)

    # maaivelddrainage
    if maaivelddrainage:
        model_ds = ahn.get_ahn_dataset(model_ds, extent, delr, use_cache=use_cache,
                                       cachedir=cachedir, verbose=verbose)
        model_ds.attrs['maaiveldrn_cond'] = maaiveldrn_cond
        mask = model_ds['ahn'].notnull()
        drn_rec = mgrid.data_array_2d_to_rec_list(model_ds, mask, col1='ahn',
                                                  first_active_layer=True,
                                                  only_active_cells=False,
                                                  col2=model_ds.maaiveldrn_cond)

        flopy.mf6.modflow.mfgwfdrn.ModflowGwfdrn(gwf, print_input=True,
                                                maxbound=len(drn_rec),
                                                stress_period_data={0: drn_rec},
                                                save_flows=True)

    # initial conditions

    # Create the initial conditions package
    model_ds['starting_head'] = starting_head * xr.ones_like(model_ds['idomain'])
    model_ds['starting_head'][0] = xr.where(model_ds['oppervlaktewater_cond'] != 0,
                                            model_ds['oppervlaktewater_peil'],
                                            model_ds['starting_head'][0])

    flopy.mf6.modflow.mfgwfic.ModflowGwfic(gwf, pname='ic',
                                           strt=model_ds['starting_head'].data)

    # add constant head cells at model boundaries
    if constant_head_edges:
        # get mask with grid edges
        xmin = model_ds['x'] == model_ds['x'].min()
        xmax = model_ds['x'] == model_ds['x'].max()
        ymin = model_ds['y'] == model_ds['y'].min()
        ymax = model_ds['y'] == model_ds['y'].max()
        mask2d = (ymin | ymax | xmin | xmax)

        # assign 1 to cells that are on the edge and have an active idomain
        model_ds['chd'] = xr.zeros_like(model_ds['idomain'])
        for lay in model_ds.layer:
            model_ds['chd'].loc[lay] = np.where(mask2d & (model_ds['idomain'].loc[lay] == 1), 1, 0)

        # get the stress_period_data
        chd_rec = mgrid.data_array_3d_to_rec_list(model_ds, model_ds['chd'] != 0,
                                                  col1='starting_head')

        flopy.mf6.modflow.mfgwfchd.ModflowGwfchd(gwf, pname='chd',
                                                maxbound=len(chd_rec),
                                                stress_period_data=chd_rec,
                                                save_flows=True)

    # rch
    # add knmi data to the model datasets
    model_ds = recharge.get_recharge(model_ds,
                                     verbose=verbose,
                                     cachedir=cachedir,
                                     use_cache=use_cache)

    # create recharge package
    recharge.model_datasets_to_rch(gwf, model_ds)

    # Create the output control package
    headfile = '{}.hds'.format(model_name)
    head_filerecord = [headfile]
    budgetfile = '{}.cbb'.format(model_name)
    budget_filerecord = [budgetfile]
    saverecord = [('HEAD', 'ALL'),
                  ('BUDGET', 'ALL')]
    printrecord = [('HEAD', 'LAST')]

    flopy.mf6.modflow.mfgwfoc.ModflowGwfoc(gwf, pname='oc',
                                            saverecord=saverecord,
                                            head_filerecord=head_filerecord,
                                            budget_filerecord=budget_filerecord,
                                            printrecord=printrecord)

    # save model_ds
    model_ds.to_netcdf(os.path.join(cachedir, 'full_model_ds.nc'))

    if write_sim:
        sim.write_simulation()

    if run_sim:
        success, buff = sim.run_simulation()
        print('\nSuccess is: ', success)

    return model_ds
