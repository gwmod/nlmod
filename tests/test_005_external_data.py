# -*- coding: utf-8 -*-

from nlmod import recharge, ahn, surface_water
import test_001_model, test_002_regis_geotop
import pandas as pd

import os
import nlmod
import pytest



def test_get_recharge():
    
    # model with sea
    model_ds = test_001_model.test_get_model_ds_from_cache('sea_model_grid')

    # add knmi recharge to the model datasets
    model_ds = recharge.add_knmi_to_model_dataset(model_ds,
                                                  verbose=True)
    
    return model_ds



def test_get_ahn():
    
    # model with sea
    model_ds = test_001_model.test_get_model_ds_from_cache('sea_model_grid')
    

    # add ahn data to the model datasets
    model_ds = ahn.get_ahn_at_grid(model_ds)
    
    return model_ds


def test_get_surface_water_ghb():
    
    # model with sea
    model_ds = test_001_model.test_get_model_ds_from_cache('sea_model_grid')
    sim, gwf = nlmod.mfpackages.sim_tdis_gwf_ims_from_model_ds(model_ds)
    nlmod.mfpackages.dis_from_model_ds(model_ds, gwf)
    

    # add knmi recharge to the model datasets
    model_ds = surface_water.surface_water_to_model_dataset(model_ds,
                                                            gwf.modelgrid,
                                                            'surface_water')
    
    return model_ds

