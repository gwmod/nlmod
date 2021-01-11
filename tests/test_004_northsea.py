# -*- coding: utf-8 -*-

from nlmod import util
import test_001_model, test_002_regis_geotop
import pandas as pd

import os
import nlmod
import pytest


def test_get_gdf_opp_water():
    model_ds = test_001_model.test_get_model_ds_from_cache()
    gdf_surface_water = nlmod.surface_water.get_gdf_surface_water(model_ds)

    return gdf_surface_water


def test_surface_water_to_dataset():
    
    # model with sea
    model_ds = test_001_model.test_get_model_ds_from_cache('sea_model_grid')
    sim, gwf = nlmod.mfpackages.sim_tdis_gwf_ims_from_model_ds(model_ds)
    nlmod.mfpackages.dis_from_model_ds(model_ds, gwf)
    
    name = 'surface_water'
    model_ds_surfwat = nlmod.surface_water.surface_water_to_model_dataset(model_ds, 
                                                                          gwf.modelgrid, 
                                                                          name)
    
    return model_ds_surfwat


def test_get_northsea_seamodel():

    # model with sea
    model_ds = test_001_model.test_get_model_ds_from_cache('basic_sea_model')
    modelgrid = nlmod.mgrid.modelgrid_from_model_ds(model_ds)
    model_ds_sea = nlmod.northsea.find_sea_cells(model_ds, modelgrid)

    assert (model_ds_sea.northsea == 1).sum() > 0

    return model_ds_sea


def test_get_northsea_nosea():

    # model without sea
    model_ds = test_001_model.test_get_model_ds_from_cache('small_model')
    modelgrid = nlmod.mgrid.modelgrid_from_model_ds(model_ds)
    model_ds_sea = nlmod.northsea.find_sea_cells(model_ds, modelgrid)

    assert (model_ds_sea.northsea == 1).sum() == 0

    return model_ds_sea


def test_fill_top_bot_kh_kv_seamodel():

    # model with sea
    model_ds = test_001_model.test_get_model_ds_from_cache('basic_sea_model')
    modelgrid = nlmod.mgrid.modelgrid_from_model_ds(model_ds)
    model_ds = nlmod.northsea.get_modelgrid_sea(model_ds, modelgrid)

    fill_mask = (model_ds['first_active_layer'] == model_ds.nodata) * model_ds['northsea']
    model_ds = nlmod.mgrid.fill_top_bot_kh_kv_at_mask(model_ds, fill_mask)

    return model_ds


def test_fill_top_bot_kh_kv_nosea():

    # model with sea
    model_ds = test_001_model.test_get_model_ds_from_cache('small_model')
    modelgrid = nlmod.mgrid.modelgrid_from_model_ds(model_ds)
    model_ds = nlmod.northsea.get_modelgrid_sea(model_ds, modelgrid)

    fill_mask = (model_ds['first_active_layer'] == model_ds.nodata) * model_ds['northsea']
    model_ds = nlmod.mgrid.fill_top_bot_kh_kv_at_mask(model_ds, fill_mask)

    return model_ds


def test_get_bathymetrie_seamodel():

    # model with sea
    model_ds = test_001_model.test_get_model_ds_from_cache('basic_sea_model')
    modelgrid = nlmod.mgrid.modelgrid_from_model_ds(model_ds)
    model_ds = nlmod.northsea.get_modelgrid_sea(model_ds, modelgrid)
    model_ds_bathymetry = nlmod.northsea.bathymetry_to_model_dataset(model_ds)

    assert (~model_ds_bathymetry.bathymetry.isnull()).sum() > 0

    return model_ds_bathymetry


def test_get_bathymetrie_nosea():

    # model without sea
    model_ds = test_001_model.test_get_model_ds_from_cache('small_model')
    modelgrid = nlmod.mgrid.modelgrid_from_model_ds(model_ds)
    model_ds = nlmod.northsea.find_sea_cells(model_ds, modelgrid)
    model_ds_bathymetry = nlmod.northsea.bathymetry_to_model_dataset(model_ds)

    assert (~model_ds_bathymetry.bathymetry.isnull()).sum() == 0

    return model_ds_bathymetry


def test_add_bathymetrie_to_top_bot_kh_kv_seamodel():

    # model with sea
    model_ds = test_001_model.test_get_model_ds_from_cache('basic_sea_model')
    modelgrid = nlmod.mgrid.modelgrid_from_model_ds(model_ds)
    model_ds = nlmod.northsea.get_modelgrid_sea(model_ds, modelgrid)
    model_ds = nlmod.northsea.get_modelgrid_bathymetry(model_ds, modelgrid)
    
    fill_mask = (model_ds['first_active_layer'] == model_ds.nodata) * model_ds['northsea']

    model_ds = nlmod.northsea.add_bathymetry_to_top_bot_kh_kv(model_ds,
                                                              model_ds['bathymetry'], 
                                                              fill_mask)

    return model_ds


