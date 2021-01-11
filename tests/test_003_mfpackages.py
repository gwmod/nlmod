# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 16:24:03 2021

@author: oebbe
"""

import datetime as dt
import nlmod
import pytest

import test_001_model


def test_sim_tdis_gwf_ims_from_model_ds(tmpdir):
    
    model_ds = test_001_model.test_get_model_ds_from_cache('basic_sea_model')
    
    sim, gwf = nlmod.mfpackages.sim_tdis_gwf_ims_from_model_ds(model_ds)
    
    return sim, gwf

def dis_from_model_ds(tmpdir):
    
    model_ds = test_001_model.test_get_model_ds_from_cache('small_model')
    
    dis = nlmod.mfpackages.dis_from_model_ds(model_ds)
    
    return dis

@pytest.mark.slow
def disv_from_model_ds(tmpdir):
    
    model_ds, gwf, gridprops = test_001_model.test_create_inf_panden_model(tmpdir)
    
    disv = nlmod.mfpackages.disv_from_model_ds(model_ds, gwf, gridprops)
    
    return disv

def npf_from_model_ds(tmpdir):
    
    model_ds = test_001_model.test_get_model_ds_from_cache('small_model')
    sim, gwf = nlmod.mfpackages.sim_tdis_gwf_ims_from_model_ds(model_ds)
    nlmod.mfpackages.dis_from_model_ds(model_ds)
    
    npf = nlmod.mfpackages.npf_from_model_ds(model_ds, gwf)
    
    return npf

def oc_from_model_ds(tmpdir):
    
    model_ds = test_001_model.test_get_model_ds_from_cache('small_model')
    sim, gwf = nlmod.mfpackages.sim_tdis_gwf_ims_from_model_ds(model_ds)
    
    oc = nlmod.mfpackages.oc_from_model_ds(model_ds, gwf)
    
    return oc

def sto_from_model_ds(tmpdir):
    
    model_ds = test_001_model.test_get_model_ds_from_cache('small_model')
    sim, gwf = nlmod.mfpackages.sim_tdis_gwf_ims_from_model_ds(model_ds)
    
    sto = nlmod.mfpackages.sto_from_model_ds(model_ds, gwf)
    
    return sto


def ghb_from_model_ds(tmpdir):
    
    model_ds = test_001_model.test_get_model_ds_from_cache('full_sea_model')
    sim, gwf = nlmod.mfpackages.sim_tdis_gwf_ims_from_model_ds(model_ds)
    nlmod.mfpackages.dis_from_model_ds(model_ds)
    
    ghb = nlmod.mfpackages.ghb_from_model_ds(model_ds, gwf,
                                             'surface_water')
    
    return ghb

def rch_from_model_ds(tmpdir):
    model_ds = test_001_model.test_get_model_ds_from_cache('full_sea_model')
    sim, gwf = nlmod.mfpackages.sim_tdis_gwf_ims_from_model_ds(model_ds)
    nlmod.mfpackages.dis_from_model_ds(model_ds)
    
    rch = nlmod.mfpackages.rch_from_model_ds(model_ds, gwf)
    
    return rch

def drn_from_model_ds(tmpdir):
    model_ds = test_001_model.test_get_model_ds_from_cache('full_sea_model')
    sim, gwf = nlmod.mfpackages.sim_tdis_gwf_ims_from_model_ds(model_ds)
    nlmod.mfpackages.dis_from_model_ds(model_ds)
    
    
    
    drn = nlmod.mfpackages.surface_drain_from_model_ds(model_ds, gwf)
    
    return drn

def chd_from_model_ds(tmpdir):
    model_ds = test_001_model.test_get_model_ds_from_cache('small_model')
    sim, gwf = nlmod.mfpackages.sim_tdis_gwf_ims_from_model_ds(model_ds)
    nlmod.mfpackages.dis_from_model_ds(model_ds)
    
    nlmod.mfpackages.ic_from_model_ds(model_ds, gwf,
                     starting_head=1.0)
    
    chd = nlmod.mfpackages.chd_at_model_edge_from_model_ds(model_ds, gwf)
    
    return chd

