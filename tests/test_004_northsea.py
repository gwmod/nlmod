# -*- coding: utf-8 -*-

from nlmod import util
import test_001_model
import pandas as pd

import os
import nlmod
import pytest


def test_get_gdf_opp_water():
    model_ds = test_001_model.test_get_model_ds_from_cache()
    gdf_opp_water = nlmod.surface_water.get_gdf_opp_water(model_ds)
    
    return gdf_opp_water

def test_get_northsea():
    
    # model without sea
    model_ds = test_001_model.test_get_model_ds_from_cache('small_model')
    model_ds_sea = nlmod.northsea.find_sea_cells(model_ds)
    
    assert (model_ds_sea.northsea == 1 ).sum() == 0
    
    # model with sea
    model_ds = test_001_model.test_get_model_ds_from_cache('sea_model')
    model_ds_sea = nlmod.northsea.find_sea_cells(model_ds)
    
    assert (model_ds.northsea == 1 ).sum() > 0
    
    return model_ds_sea

def test_get_bathymetrie():
    
    # model without sea
    model_ds = test_001_model.test_get_model_ds_from_cache('small_model')
    model_ds = nlmod.northsea.find_sea_cells(model_ds)
    model_ds_bathymetry = nlmod.northsea.bathymetry_to_model_dataset(model_ds)
    
    assert (~model_ds_bathymetry.bathymetry.isnull()).sum() == 0

    # model with sea
    model_ds = test_001_model.test_get_model_ds_from_cache('sea_model')
    model_ds = nlmod.northsea.get_modelgrid_sea(model_ds)
    model_ds_bathymetry = nlmod.northsea.bathymetry_to_model_dataset(model_ds)
    
    assert (~model_ds_bathymetry.bathymetry.isnull()).sum() > 0
    
    return model_ds_bathymetry




