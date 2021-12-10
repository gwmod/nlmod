# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 16:23:35 2021

@author: oebbe
"""

import os

import nlmod
import pandas as pd
import pytest
from nlmod.read import geotop, regis

import test_001_model

# @pytest.mark.skip(reason="too slow")


def test_get_regis(extent=[98600.0, 99000.0, 489400.0, 489700.0],
                   delr=100., delc=100.):

    regis_ds = regis.get_regis(extent, delr, delc)

    assert regis_ds.dims['layer'] == 132

    return regis_ds


# @pytest.mark.skip(reason="too slow")
def test_fit_regis_extent(extent=[128050., 141450., 468550., 481450.],
                          delr=100., delc=100.):

    try:
        regis_ds = regis.get_regis(extent, delr, delc)
    except ValueError:
        return True

    raise RuntimeError('regis fit does not work as expected')

    return regis_ds


# @pytest.mark.skip(reason="too slow")
def test_get_regis_botm_layer_BEk1(extent=[98700., 99000., 489500., 489700.],
                                   delr=100., delc=100., botm_layer=b'BEk1'):

    extent, nrow, ncol = regis.fit_extent_to_regis(extent, delr, delc)

    regis_ds = regis.get_regis(extent, delr, delc,
                               botm_layer)

    assert regis_ds.dims['layer'] == 18

    assert regis_ds.layer.values[-1] == 'BEk1'

    return regis_ds


# @pytest.mark.skip(reason="too slow")
def test_get_geotop(extent=[98600.0, 99000.0, 489400.0, 489700.0],
                    delr=100., delc=100.):

    regis_ds = test_get_regis(extent=extent,
                              delr=delr, delc=delc)

    geotop_ds = geotop.get_geotop(extent, delr, delc,
                                  regis_ds)

    return geotop_ds

# @pytest.mark.skip(reason="too slow")


def test_get_regis_geotop(extent=[98600.0, 99000.0, 489400.0, 489700.0],
                          delr=100., delc=100.):

    regis_geotop_ds = regis.get_combined_layer_models(extent, delr, delc,
                                                      use_regis=True, 
                                                      use_geotop=True)

    assert regis_geotop_ds.dims['layer'] == 24

    return regis_geotop_ds

# @pytest.mark.skip(reason="too slow")


def test_get_regis_geotop_keep_all_layers(extent=[98600.0, 99000.0, 489400.0, 489700.0],
                                          delr=100., delc=100.):

    regis_geotop_ds = regis.get_combined_layer_models(extent, delr, delc,
                                                     use_regis=True, 
                                                     use_geotop=True,
                                                     remove_nan_layers=False)

    assert regis_geotop_ds.dims['layer'] == 135

    return regis_geotop_ds
