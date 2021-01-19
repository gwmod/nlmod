# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 16:23:35 2021

@author: oebbe
"""

from nlmod import regis, geotop
import test_001_model
import pandas as pd
import os
import nlmod
import pytest

@pytest.mark.skip(reason="too slow")
def test_get_regis(extent=[98700., 99000., 489500., 489700.],
                    delr=100., delc=100.):

    regis_ds = regis.get_regis_dataset(extent, delr, delc, 
                                       cachedir=None,
                                       use_cache=False,
                                       verbose=True)
    
    assert regis_ds.dims['layer'] == 132

    return regis_ds


@pytest.mark.skip(reason="too slow")
def test_get_regis_botm_layer_BEk1(extent=[98700., 99000., 489500., 489700.],
                                   delr=100., delc=100., botm_layer=b'BEk1'):

    regis_ds = regis.get_regis_dataset(extent, delr, delc, 
                                       botm_layer,
                                       cachedir=None,
                                       use_cache=False,
                                       verbose=True)
    
    assert regis_ds.dims['layer'] == 18
    
    assert regis_ds.layer.values[-1] == 'BEk1'

    return regis_ds

@pytest.mark.skip(reason="too slow")
def test_get_geotop(extent=[98700., 99000., 489500., 489700.],
                    delr=100., delc=100.):

    
    regis_ds = test_get_regis(extent=extent,
                              delr=delr, delc=delc)

    geotop_ds = geotop.get_geotop_dataset(extent, delr, delc, 
                                          regis_ds,
                                          cachedir=None,
                                          use_cache=False,
                                          verbose=True)

    return geotop_ds

@pytest.mark.skip(reason="too slow")
def test_get_regis_geotop(extent=[98700., 99000., 489500., 489700.],
                          delr=100., delc=100.):
    
    
    regis_geotop_ds = regis.get_layer_models(extent, delr, delc,
                                             use_regis=True, use_geotop=True,
                                             cachedir=None,
                                             use_cache=False, verbose=True)
    
    assert regis_geotop_ds.dims['layer'] == 19
    
    
    return regis_geotop_ds

@pytest.mark.skip(reason="too slow")
def test_get_regis_geotop_keep_all_layers(extent=[98700., 99000., 489500., 489700.],
                                          delr=100., delc=100.):
    
    
    regis_geotop_ds = regis.get_layer_models(extent, delr, delc,
                                             use_regis=True, use_geotop=True,
                                             remove_nan_layers=False,
                                             cachedir=None,
                                             use_cache=False, verbose=True)
    
    assert regis_geotop_ds.dims['layer'] == 135
    
    
    
    return regis_geotop_ds