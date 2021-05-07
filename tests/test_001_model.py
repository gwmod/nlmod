# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 16:22:58 2021

@author: oebbe

Extents uit nhflo:

# entire model domain
extent = [95000., 150000., 487000., 553500.]

# alkmaar
#extent = [104000.0, 121500. ,510000., 528000.] 

# alle infiltratiepanden
extent = [100350., 106000. ,500800., 508000.] 

# zelfde als koster doorsneden
extent = [100000., 109000. ,497000., 515000.] 


# extent pwn model
extent = [ 95800., 109000., 496700., 515100.]

# # xmax ligt buiten pwn_model
# extent = [100000., 115000. ,497000., 515000.] 

# xmax, ymin en ymax liggen buiten pwn_model
extent = [100000., 115000. ,496000., 516000.] 

# hoekje met zee
extent = [95000., 100000., 487000., 500000.]

# klein (300m x 300m)
# extent = [102000.0, 102300.0, 505800.0, 506100.0]

# extent = [102000.0, 102300.0, 505800.0, 506800.0]


test modellen hebben volgende eigenschappen:
    
    basic_sea_model:
        transient
        100x60 cellen, 
        delr=delc=100, 
        structured grid,
        zee aanwezig (ter hoogte van noordzeekanaal),
        geotop+regis, 
        zee niet opgevuld
        [95000., 105000., 494000., 500000.]
    
    sea_model_grid:
        transient
        100x60 cellen, 
        delr=delc=100, 
        structured grid,
        zee aanwezig (ter hoogte van noordzeekanaal),
        geotop+regis, 
        noordzee_opgevuld
        [95000., 105000., 494000., 500000.]
        
    full_sea_model:
        transient
        100x60 cellen, 
        delr=delc=100, 
        structured grid,
        zee aanwezig (ter hoogte van noordzeekanaal),
        geotop+regis, 
        noordzee_opgevuld, 
        packages:
            - drn maaivelddrainage
            - chd boundary
            - rch knmi 
            - ghb surface_water
        [95000., 105000., 494000., 500000.]
        
    small_model_grid:
        transient
        3x3 cellen, delr=delc=100, geen zee, geotop+regis, noordzee opgevuld
        [98700., 99000., 489500., 489700.]
        
    infpanden_model:
        transient
        
        delr=delc=100 
        unstructured grid, levels 2, refinment on -> panden.shp
        zee aanwezig
        geotop+regis
        noordzee opgevuld
        packages:
            - drn maaivelddrainage
            - chd boundary
            - rch knmi
            - ghb surface_water
        [100350., 106000., 500800., 508000.]
        
    
"""

import test_002_regis_geotop
import nlmod
import os
import geopandas as gpd
import xarray as xr
import datetime as dt
import flopy
import pytest
import tempfile
tmpdir = tempfile.gettempdir()


tst_model_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             'data')


def test_model_directories(tmpdir):
    model_ws = os.path.join(tmpdir, 'test_model')
    figdir, cachedir = nlmod.util.get_model_dirs(model_ws)

    return model_ws, figdir, cachedir


def test_model_ds_time_steady(tmpdir, modelname='test'):
    model_ws = os.path.join(tmpdir, 'test_model')
    model_ds = nlmod.mdims.get_model_ds_time(modelname, model_ws,
                                             '2015-1-1',
                                             steady_state=True)

    return model_ds


def test_model_ds_time_transient(tmpdir, modelname='test'):
    model_ws = os.path.join(tmpdir, 'test_model')
    model_ds = nlmod.mdims.get_model_ds_time(modelname, model_ws,
                                             '2015-1-1',
                                             steady_state=False,
                                             steady_start=True,
                                             transient_timesteps=10)
    nlmod.util.get_model_dirs(model_ds.model_ws)
    return model_ds

# %% creating model grids


@pytest.mark.slow
def test_create_small_model_grid_only(tmpdir):
    model_ds = test_model_ds_time_transient(tmpdir)

    extent = [98700., 99000., 489500., 489700.]
    extent, nrow, ncol = nlmod.read.regis.fit_extent_to_regis(extent, 100, 100)
    regis_geotop_ds = nlmod.read.regis.get_layer_models(extent, 100., 100.,
                                                        regis_botm_layer=b'KRz5',
                                                        use_regis=True,
                                                        use_geotop=True,
                                                        verbose=True)
    assert regis_geotop_ds.dims['layer'] == 5

    model_ds = nlmod.mdims.update_model_ds_from_ml_layer_ds(model_ds,
                                                            regis_geotop_ds,
                                                            keep_vars=['x', 'y'],
                                                            gridtype='structured',
                                                            verbose=True)

    sim, gwf = nlmod.mfpackages.sim_tdis_gwf_ims_from_model_ds(model_ds,
                                                               verbose=True)

    # Create discretization
    nlmod.mfpackages.dis_from_model_ds(model_ds, gwf)

    # save model_ds
    model_ds.to_netcdf(os.path.join(tst_model_dir, 'small_model.nc'))

    return model_ds, gwf


@pytest.mark.slow
def test_create_sea_model_grid_only(tmpdir):
    model_ds = test_model_ds_time_transient(tmpdir)
    extent = [95000., 105000., 494000., 500000.]
    extent, nrow, ncol = nlmod.read.regis.fit_extent_to_regis(extent, 100, 100)
    regis_geotop_ds = nlmod.read.regis.get_layer_models(extent, 100., 100.,
                                                   use_regis=True,
                                                   use_geotop=True,
                                                   verbose=True)
    model_ds = nlmod.mdims.update_model_ds_from_ml_layer_ds(model_ds,
                                                            regis_geotop_ds,
                                                            keep_vars=['x', 'y'],
                                                            gridtype='structured',
                                                            verbose=True)
    # save model_ds
    model_ds.to_netcdf(os.path.join(tst_model_dir, 'sea_model_grid.nc'))

    return model_ds


@pytest.mark.slow
def test_create_seamodel_grid_only_without_northsea(tmpdir):
    model_ds = test_model_ds_time_transient(tmpdir)
    extent = [95000., 105000., 494000., 500000.]
    extent, nrow, ncol = nlmod.read.regis.fit_extent_to_regis(extent, 100, 100)
    regis_geotop_ds = nlmod.read.regis.get_layer_models(extent, 100., 100.,
                                                   use_regis=True,
                                                   use_geotop=True,
                                                   verbose=True)

    model_ds = nlmod.mdims.update_model_ds_from_ml_layer_ds(model_ds,
                                                            regis_geotop_ds,
                                                            keep_vars=['x', 'y'],
                                                            gridtype='structured',
                                                            add_northsea=False,
                                                            verbose=True)

    # save model_ds
    model_ds.to_netcdf(os.path.join(tst_model_dir, 'basic_sea_model.nc'))

    return model_ds


@pytest.mark.slow
def test_create_sea_model_grid_only_delr_delc_50(tmpdir):
    model_ds = test_model_ds_time_transient(tmpdir)
    extent = [95000., 105000., 494000., 500000.]
    extent, nrow, ncol = nlmod.read.regis.fit_extent_to_regis(extent, 50., 50.)
    regis_geotop_ds = nlmod.read.regis.get_layer_models(extent, 50., 50.,
                                                   use_regis=True,
                                                   use_geotop=True,
                                                   verbose=True)
    model_ds = nlmod.mdims.update_model_ds_from_ml_layer_ds(model_ds,
                                                            regis_geotop_ds,
                                                            keep_vars=['x', 'y'],
                                                            gridtype='structured',
                                                            verbose=True)
    # save model_ds
    model_ds.to_netcdf(os.path.join(tst_model_dir, 'sea_model_grid_50.nc'))

    return model_ds

# %% create models using create_model module


@pytest.mark.slow
def test_create_sea_model(tmpdir):
    model_ds = xr.open_dataset(os.path.join(tst_model_dir,
                                            'basic_sea_model.nc'))
    # create modflow packages
    sim, gwf = nlmod.mfpackages.sim_tdis_gwf_ims_from_model_ds(model_ds,
                                                               verbose=True)
    # Create discretization
    nlmod.mfpackages.dis_from_model_ds(model_ds, gwf)
    
    # create node property flow
    nlmod.mfpackages.npf_from_model_ds(model_ds, gwf)
    
    # Create the initial conditions package
    nlmod.mfpackages.ic_from_model_ds(model_ds, gwf, 
                                      starting_head=1.0)
    
    # Create the output control package
    nlmod.mfpackages.oc_from_model_ds(model_ds, gwf)
    
    # voeg grote oppervlaktewaterlichamen toe
    da_name = 'surface_water'
    model_ds = nlmod.mfpackages.surface_water.get_general_head_boundary(model_ds,
                                                                        gwf.modelgrid,
                                                                        da_name)
    ghb = nlmod.mfpackages.ghb_from_model_ds(model_ds, gwf, da_name)
    
    # surface level drain
    model_ds = nlmod.read.ahn.get_ahn_dataset(model_ds)
    drn = nlmod.mfpackages.surface_drain_from_model_ds(model_ds, gwf)

    # add constant head cells at model boundaries
    chd = nlmod.mfpackages.chd_at_model_edge_from_model_ds(model_ds, gwf, head='starting_head')
    
    # add knmi recharge to the model datasets
    model_ds = nlmod.read.knmi.get_recharge(model_ds)
    # create recharge package
    nlmod.mfpackages.rch_from_model_ds(model_ds, gwf)
    
    gwf.simulation.write_simulation()
    
    assert gwf.simulation.run_simulation()[0]

    # save model_ds
    model_ds.to_netcdf(os.path.join(tst_model_dir, 'full_sea_model.nc'))

    return model_ds, gwf


@pytest.mark.slow
def test_create_sea_model_perlen_list(tmpdir):
    model_ds = xr.open_dataset(os.path.join(tst_model_dir,
                                            'basic_sea_model.nc'))
    
    # create transient with perlen list
    perlen = [3650, 14, 10, 11]  # length of the time steps
    transient_timesteps=3
    
    model_ds_time = nlmod.mdims.get_model_ds_time(model_ds.model_name,
                                                  model_ds.model_ws,
                                                  model_ds.start_time,
                                                  steady_state=False,
                                                  steady_start=True,
                                                  perlen=perlen,
                                                  transient_timesteps=transient_timesteps)
    
    model_ds = model_ds.drop_dims('time')
    model_ds.update(model_ds_time)
    model_ds.attrs['nper'] = model_ds_time.nper
    model_ds.attrs['perlen'] = model_ds_time.perlen
    
    # create modflow packages
    sim, gwf = nlmod.mfpackages.sim_tdis_gwf_ims_from_model_ds(model_ds,
                                                               verbose=True)
    # Create discretization
    nlmod.mfpackages.dis_from_model_ds(model_ds, gwf)
    
    # create node property flow
    nlmod.mfpackages.npf_from_model_ds(model_ds, gwf)
    
    # Create the initial conditions package
    nlmod.mfpackages.ic_from_model_ds(model_ds, gwf, 
                                      starting_head=1.0)
    
    # Create the output control package
    nlmod.mfpackages.oc_from_model_ds(model_ds, gwf)
    
    # voeg grote oppervlaktewaterlichamen toe
    da_name = 'surface_water'
    model_ds = nlmod.mfpackages.surface_water.get_general_head_boundary(model_ds,
                                                                        gwf.modelgrid,
                                                                        da_name)
    ghb = nlmod.mfpackages.ghb_from_model_ds(model_ds, gwf, da_name)
    
    # surface level drain
    model_ds = nlmod.read.ahn.get_ahn_dataset(model_ds)
    drn = nlmod.mfpackages.surface_drain_from_model_ds(model_ds, gwf)

    # add constant head cells at model boundaries
    chd = nlmod.mfpackages.chd_at_model_edge_from_model_ds(model_ds, gwf, head='starting_head')
    
    # add knmi recharge to the model datasets
    model_ds = nlmod.read.knmi.get_recharge(model_ds)
    # create recharge package
    nlmod.mfpackages.rch_from_model_ds(model_ds, gwf)
    
    gwf.simulation.write_simulation()
    
    assert gwf.simulation.run_simulation()[0]

    # save model_ds
    model_ds.to_netcdf(os.path.join(tst_model_dir, 'full_sea_model.nc'))


    return model_ds, gwf


@pytest.mark.slow
def test_create_sea_model_perlen_14(tmpdir):
    model_ds = xr.open_dataset(os.path.join(tst_model_dir,
                                            'basic_sea_model.nc'))
    
    # create transient with perlen list
    perlen = 14  # length of the time steps
    transient_timesteps=3
    
    model_ds_time = nlmod.mdims.get_model_ds_time(model_ds.model_name,
                                                  model_ds.model_ws,
                                                  model_ds.start_time,
                                                  steady_state=False,
                                                  steady_start=True,
                                                  perlen=perlen,
                                                  transient_timesteps=transient_timesteps)
    
    model_ds = model_ds.drop_dims('time')
    model_ds.update(model_ds_time)
    model_ds.attrs['nper'] = model_ds_time.nper
    model_ds.attrs['perlen'] = model_ds_time.perlen
    
    # create modflow packages
    sim, gwf = nlmod.mfpackages.sim_tdis_gwf_ims_from_model_ds(model_ds,
                                                               verbose=True)
    # Create discretization
    nlmod.mfpackages.dis_from_model_ds(model_ds, gwf)
    
    # create node property flow
    nlmod.mfpackages.npf_from_model_ds(model_ds, gwf)
    
    # Create the initial conditions package
    nlmod.mfpackages.ic_from_model_ds(model_ds, gwf, 
                                      starting_head=1.0)
    
    # Create the output control package
    nlmod.mfpackages.oc_from_model_ds(model_ds, gwf)
    
    # voeg grote oppervlaktewaterlichamen toe
    da_name = 'surface_water'
    model_ds = nlmod.mfpackages.surface_water.get_general_head_boundary(model_ds,
                                                                        gwf.modelgrid,
                                                                        da_name)
    ghb = nlmod.mfpackages.ghb_from_model_ds(model_ds, gwf, da_name)
    
    # surface level drain
    model_ds = nlmod.read.ahn.get_ahn_dataset(model_ds)
    drn = nlmod.mfpackages.surface_drain_from_model_ds(model_ds, gwf)

    # add constant head cells at model boundaries
    chd = nlmod.mfpackages.chd_at_model_edge_from_model_ds(model_ds, gwf, head='starting_head')
    
    # add knmi recharge to the model datasets
    model_ds = nlmod.read.knmi.get_recharge(model_ds)
    # create recharge package
    nlmod.mfpackages.rch_from_model_ds(model_ds, gwf)
    
    gwf.simulation.write_simulation()
    
    assert gwf.simulation.run_simulation()[0]

    # save model_ds
    model_ds.to_netcdf(os.path.join(tst_model_dir, 'full_sea_model.nc'))

    return model_ds, gwf

# %% obtaining the test models

def test_get_model_ds_from_cache(name='small_model'):

    model_ds = xr.open_dataset(os.path.join(tst_model_dir, name + '.nc'))

    return model_ds

# %% other functions


def _check_tmpdir(tmpdir):

    # pytest uses a LocalPath object for the tmpdir argument when testing
    # this function convert a LocalPath object to a string

    if isinstance(tmpdir, str):
        return tmpdir
    else:
        return str(tmpdir)
