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
    model_ds = nlmod.mtime.get_model_ds_time(modelname, model_ws,
                                             '2015-1-1',
                                             steady_state=True)

    return model_ds


def test_model_ds_time_transient(tmpdir, modelname='test'):
    model_ws = os.path.join(tmpdir, 'test_model')
    model_ds = nlmod.mtime.get_model_ds_time(modelname, model_ws,
                                             '2015-1-1',
                                             steady_state=False,
                                             steady_start=True,
                                             transient_timesteps=10)

    return model_ds

# %% creating test models and saving


@pytest.mark.slow
def test_create_small_model(tmpdir):
    model_ds = test_model_ds_time_transient(tmpdir)
    extent = [98700., 99000., 489500., 489700.]
    regis_geotop_ds = nlmod.regis.get_layer_models(extent, 100., 100.,
                                                   use_regis=True,
                                                   use_geotop=True,
                                                   verbose=True)

    model_ds = nlmod.mgrid.update_model_ds_from_ml_layer_ds(model_ds,
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
def test_create_basic_seamodel(tmpdir):
    model_ds = test_model_ds_time_transient(tmpdir)
    extent = [95000., 105000., 494000., 500000.]
    regis_geotop_ds = nlmod.regis.get_layer_models(extent, 100., 100.,
                                                   use_regis=True,
                                                   use_geotop=True,
                                                   verbose=True)

    model_ds = nlmod.mgrid.update_model_ds_from_ml_layer_ds(model_ds,
                                                            regis_geotop_ds,
                                                            keep_vars=['x', 'y'],
                                                            gridtype='structured',
                                                            add_northsea=False,
                                                            verbose=True)

    # save model_ds
    model_ds.to_netcdf(os.path.join(tst_model_dir, 'basic_sea_model.nc'))

    return model_ds


@pytest.mark.slow
def test_create_sea_model_grid(tmpdir):
    model_ds = test_model_ds_time_transient(tmpdir)
    extent = [95000., 105000., 494000., 500000.]

    regis_geotop_ds = nlmod.regis.get_layer_models(extent, 100., 100.,
                                                   use_regis=True,
                                                   use_geotop=True,
                                                   verbose=True)
    model_ds = nlmod.mgrid.update_model_ds_from_ml_layer_ds(model_ds,
                                                            regis_geotop_ds,
                                                            keep_vars=['x', 'y'],
                                                            gridtype='structured',
                                                            verbose=True)
    # save model_ds
    model_ds.to_netcdf(os.path.join(tst_model_dir, 'sea_model_grid.nc'))

    return model_ds


@pytest.mark.slow
def test_create_full_sea_model(tmpdir):
    tmpdir = _check_tmpdir(tmpdir)
    extent = [95000., 105000., 494000., 500000.]
    model_ds, gwf = nlmod.create_model.gen_model_structured(tmpdir,
                                                       'full_sea_model',
                                                       extent=extent,
                                                       steady_state=False,
                                                       transient_timesteps=10,
                                                       steady_start=True,
                                                       write_sim=True,
                                                       run_sim=True)

    # save model_ds
    model_ds.to_netcdf(os.path.join(tst_model_dir, 'full_sea_model.nc'))

    return model_ds, gwf

@pytest.mark.slow
def test_create_unstructured_model(tmpdir):
    tmpdir = _check_tmpdir(tmpdir)
    refine_shp = os.path.join(nlmod.nlmod_datadir, 
                              'shapes', 'planetenweg_ijmuiden')
    extent=[95000., 105000., 494000., 500000.]
    
    model_ds, gwf, gridprops = nlmod.create_model.gen_model_unstructured(tmpdir, 
                                                                         'IJm_planeten',
                                                                         refine_shp_fname=refine_shp,
                                                                         levels=2,
                                                                         extent=extent
                                                                         )
    # save model_ds 
    model_ds.to_netcdf(os.path.join(tst_model_dir, 'IJm_planeten.nc'))
    
    return model_ds, gwf, gridprops

@pytest.mark.skip
def test_create_inf_panden_model(tmpdir):
    tmpdir = _check_tmpdir(tmpdir)
    extent = [100350., 106000., 500800., 508000.]
    shpname = os.path.join(nlmod.nlmod_datadir, 'nhflo', 'panden')
    model_ds, gwf, gridprops = nlmod.create_model.gen_model_unstructured(tmpdir,
                                                                         'infpanden_model',
                                                                         refine_shp_fname=shpname,
                                                                         extent=extent,
                                                                         steady_state=False,
                                                                         steady_start=True,
                                                                         constant_head_edges=True,
                                                                         surface_drn=True,
                                                                         write_sim=True,
                                                                         run_sim=True)

    # save model_ds
    model_ds.to_netcdf(os.path.join(tst_model_dir, 'infpanden_model.nc'))

    return model_ds, gwf, gridprops


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
