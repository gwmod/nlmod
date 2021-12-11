# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 12:26:16 2021

@author: oebbe
"""

import tempfile

import nlmod
import pytest

import test_001_model

tmpdir = tempfile.gettempdir()


def test_model_ds_check_true():

    # two models with the same grid and time dicretisation
    model_ds = test_001_model.test_get_model_ds_from_cache('small_model')
    model_ds2 = model_ds.copy()

    check = nlmod.cache._check_ds(model_ds, model_ds2)

    assert check


def test_model_ds_check_time_false():

    # two models with a different time discretisation
    model_ds = test_001_model.test_get_model_ds_from_cache('small_model')
    model_ds2 = test_001_model.test_model_ds_time_steady(tmpdir)

    check = nlmod.cache._check_ds(model_ds, model_ds2)

    assert check == False


@pytest.mark.slow
def test_model_ds_check_grid_false(tmpdir):

    # two models with a different grid and same time dicretisation
    model_ds = test_001_model.test_get_model_ds_from_cache('small_model')
    model_ds2 = test_001_model.test_model_ds_time_transient(tmpdir)
    extent = [99100., 99400., 489100., 489400.]
    extent, nrow, ncol = nlmod.read.regis.fit_extent_to_regis(extent,
                                                              50.,
                                                              50.)
    regis_ds = nlmod.read.regis.get_combined_layer_models(extent,
                                                          50., 50.,
                                                          use_regis=True,
                                                          use_geotop=False,
                                                          cachedir=tmpdir,
                                                          cachename='comb.nc'
                                                          )

    model_ds2 = nlmod.mdims.update_model_ds_from_ml_layer_ds(model_ds2,
                                                             regis_ds,
                                                             keep_vars=[
                                                                 'x', 'y'],
                                                             gridtype='structured')

    check = nlmod.cache._check_ds(model_ds, model_ds2)

    assert check == False


@pytest.mark.skip("too slow")
def test_use_cached_regis(tmpdir):

    extent = [98700., 99000., 489500., 489700.]
    delr = 100.
    delc = 100.
    extent, nrow, ncol = nlmod.read.regis.fit_extent_to_regis(
        extent, delr, delc)

    regis_ds1 = nlmod.read.regis.get_regis(extent, delr, delc,
                                           cachedir=tmpdir,
                                           cachename='reg.nc')

    regis_ds2 = nlmod.read.regis.get_regis(extent, delr, delc,
                                           cachedir=tmpdir,
                                           cachename='reg.nc')

    assert regis_ds1.equals(regis_ds2)

    return regis_ds2


@pytest.mark.skip("too slow")
def test_do_not_use_cached_regis(tmpdir):
    # cache regis
    extent = [98700., 99000., 489500., 489700.]
    delr = 100.
    delc = 100.
    extent, nrow, ncol = nlmod.read.regis.fit_extent_to_regis(
        extent, delr, delc)
    regis_ds1 = nlmod.read.regis.get_regis(extent, delr, delc,
                                           cachedir=tmpdir,
                                           cachename='regis.nc')

    # do not use cache because extent is different
    extent = [99100., 99400., 489100., 489400.]
    delr = 100.
    delc = 100.
    extent, nrow, ncol = nlmod.read.regis.fit_extent_to_regis(
        extent, delr, delc)
    regis_ds2 = nlmod.read.regis.get_regis(extent, delr, delc,
                                           cachedir=tmpdir,
                                           cachename='regis.nc')

    assert not regis_ds1.equals(regis_ds2)

    # do not use cache because delr is different
    extent = [99100., 99400., 489100., 489400.]
    delr = 50.
    delc = 100.
    extent, nrow, ncol = nlmod.read.regis.fit_extent_to_regis(
        extent, delr, delc)
    regis_ds3 = nlmod.read.regis.get_regis(extent, delr, delc,
                                           cachedir=tmpdir,
                                           cachename='regis.nc')

    assert not regis_ds2.equals(regis_ds3)

    # do not use cache because delc is different
    extent = [99100., 99400., 489100., 489400.]
    delr = 50.
    delc = 50.
    extent, nrow, ncol = nlmod.read.regis.fit_extent_to_regis(
        extent, delr, delc)
    regis_ds4 = nlmod.read.regis.get_regis(extent, delr, delc,
                                           cachedir=tmpdir,
                                           cachename='regis.nc')

    assert not regis_ds3.equals(regis_ds4)

    return regis_ds4
