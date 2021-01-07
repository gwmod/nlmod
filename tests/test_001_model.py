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
    tst_model1
    klein, 300x300 m
    
    tst_model2
"""

import nlmod
import os
import geopandas as gpd
import xarray as xr
import datetime as dt
import flopy
import pytest

import test_002_regis_geotop

tst_model_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             'data')


def test_model_directories(tmpdir):
    model_ws = os.path.join(tmpdir, 'test_model')
    figdir, cachedir = nlmod.util.get_model_dirs(model_ws)

    return model_ws, figdir, cachedir


def test_model_ds_time_steady(tmpdir):
    model_ws = os.path.join(tmpdir, 'test_model')
    model_ds = nlmod.mtime.get_model_ds_time('test', model_ws,
                                             dt.datetime.now(),
                                             steady_state=True)

    return model_ds


def test_model_ds_time_transient(tmpdir):
    model_ws = os.path.join(tmpdir, 'test_model')
    model_ds = nlmod.mtime.get_model_ds_time('test', model_ws,
                                             dt.datetime.now(),
                                             steady_state=False,
                                             steady_start=True,
                                             transient_timesteps=10)

    return model_ds


@pytest.mark.slow
def test_model_ds_time_grid_regis_small(tmpdir):
    model_ds = test_model_ds_time_transient(tmpdir)
    layer_model = test_002_regis_geotop.test_get_regis_geotop()
    model_ds = nlmod.mgrid.update_model_ds_from_ml_layer_ds(model_ds, layer_model,
                                                            keep_vars=['x', 'y'],
                                                            gridtype='structured',
                                                            verbose=True)

    return model_ds

@pytest.mark.slow
def test_model_ds_time_grid_regis_sea(tmpdir):
    model_ds = test_model_ds_time_transient(tmpdir)
    layer_model = test_002_regis_geotop.test_get_regis_geotop(extent=[95000., 105000., 494000., 500000.])
    model_ds = nlmod.mgrid.update_model_ds_from_ml_layer_ds(model_ds, 
                                                            layer_model,
                                                            keep_vars=['x', 'y'],
                                                            gridtype='structured',
                                                            verbose=True)

    return model_ds


def test_get_model_ds_from_cache(name='small_model'):
    
    model_ds = xr.open_dataset(os.path.join(tst_model_dir, name+'.nc'))

    
    return model_ds
