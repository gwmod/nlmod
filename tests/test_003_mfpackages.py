# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 16:24:03 2021

@author: oebbe
"""

import datetime as dt
import nlmod

import test_001_model


def test_sim_tdis_gwf_ims_from_model_ds(tmpdir):
    
    model_ds = test_001_model.test_get_model_ds_from_cache()
    
    sim, gwf = nlmod.mfpackages.sim_tdis_gwf_ims_from_model_ds(model_ds)
    
    return sim, gwf

