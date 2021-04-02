# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 16:23:35 2021

@author: oebbe
"""

from nlmod.read import regis, geotop
import test_001_model
import pandas as pd
import os
import nlmod
import pytest

#@pytest.mark.skip(reason="too slow")
def test_get_regis(extent=[98650.0, 99050.0, 489450.0, 489750.0],
                    delr=100., delc=100.):

    regis_ds = regis.get_regis_dataset(extent, delr, delc, 
                                       cachedir=None,
                                       use_cache=False,
                                       verbose=True)
    
    assert regis_ds.dims['layer'] == 132

    return regis_ds


#@pytest.mark.skip(reason="too slow")
def test_fit_regis_extent(extent = [128000. , 141400., 468500., 481400. ],
                          delr=100., delc=100.):
    
    
    try:
        regis_ds = regis.get_regis_dataset(extent, delr, delc, 
                                           cachedir=None,
                                           use_cache=False,
                                           verbose=True)
    except ValueError:
        return True
    
    raise RuntimeError('regis fit does not work as expected')
    

    return regis_ds


#@pytest.mark.skip(reason="too slow")
def test_get_regis_botm_layer_BEk1(extent=[98700., 99000., 489500., 489700.],
                                   delr=100., delc=100., botm_layer=b'BEk1'):
    
    extent, nrow, ncol = regis.fit_extent_to_regis(extent, delr, delc)
    
    regis_ds = regis.get_regis_dataset(extent, delr, delc, 
                                       botm_layer,
                                       cachedir=None,
                                       use_cache=False,
                                       verbose=True)
    
    assert regis_ds.dims['layer'] == 18
    
    assert regis_ds.layer.values[-1] == 'BEk1'

    return regis_ds


#@pytest.mark.skip(reason="too slow")
def test_get_geotop(extent=[98650.0, 99050.0, 489450.0, 489750.0],
                    delr=100., delc=100.):

    

    regis_ds = test_get_regis(extent=extent,
                              delr=delr, delc=delc)

    geotop_ds = geotop.get_geotop_dataset(extent, delr, delc, 
                                          regis_ds,
                                          cachedir=None,
                                          use_cache=False,
                                          verbose=True)

    return geotop_ds

#@pytest.mark.skip(reason="too slow")
def test_get_regis_geotop(extent=[98650.0, 99050.0, 489450.0, 489750.0],
                          delr=100., delc=100.):
    
    
    regis_geotop_ds = regis.get_layer_models(extent, delr, delc,
                                             use_regis=True, use_geotop=True,
                                             cachedir=None,
                                             use_cache=False, verbose=True)
    
    assert regis_geotop_ds.dims['layer'] == 23
    
    
    return regis_geotop_ds

#@pytest.mark.skip(reason="too slow")
def test_get_regis_geotop_keep_all_layers(extent=[98650.0, 99050.0, 489450.0, 489750.0],
                                          delr=100., delc=100.):
    
    
    regis_geotop_ds = regis.get_layer_models(extent, delr, delc,
                                             use_regis=True, use_geotop=True,
                                             remove_nan_layers=False,
                                             cachedir=None,
                                             use_cache=False, verbose=True)
    
    assert regis_geotop_ds.dims['layer'] == 135
    
    
    
    return regis_geotop_ds