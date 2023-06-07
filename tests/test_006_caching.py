# -*- coding: utf-8 -*-
"""Created on Mon Jan 11 12:26:16 2021.

@author: oebbe
"""

import tempfile

import pytest
import test_001_model

import nlmod

tmpdir = tempfile.gettempdir()


def test_ds_check_true():
    # two models with the same grid and time dicretisation
    ds = test_001_model.get_ds_from_cache("small_model")
    ds2 = ds.copy()

    check = nlmod.cache._check_ds(ds, ds2)

    assert check


def test_ds_check_time_false():
    # two models with a different time discretisation
    ds = test_001_model.get_ds_from_cache("small_model")
    ds2 = test_001_model.get_ds_time_steady(tmpdir)

    check = nlmod.cache._check_ds(ds, ds2)

    assert not check


def test_ds_check_time_attributes_false():
    # two models with a different time discretisation
    ds = test_001_model.get_ds_from_cache("small_model")
    ds2 = ds.copy()

    ds2.time.attrs["time_units"] = "MONTHS"

    check = nlmod.cache._check_ds(ds, ds2)

    assert not check


@pytest.mark.slow
def test_ds_check_grid_false(tmpdir):
    # two models with a different grid and same time dicretisation
    ds = test_001_model.get_ds_from_cache("small_model")
    ds2 = test_001_model.get_ds_time_transient(tmpdir)
    extent = [99100.0, 99400.0, 489100.0, 489400.0]
    regis_ds = nlmod.read.regis.get_combined_layer_models(
        extent,
        use_regis=True,
        use_geotop=False,
        cachedir=tmpdir,
        cachename="comb.nc",
    )
    ds2 = nlmod.base.to_model_ds(regis_ds, delr=50.0, delc=50.0)

    check = nlmod.cache._check_ds(ds, ds2)

    assert not check


@pytest.mark.skip("too slow")
def test_use_cached_regis(tmpdir):
    extent = [98700.0, 99000.0, 489500.0, 489700.0]
    regis_ds1 = nlmod.read.regis.get_regis(extent, cachedir=tmpdir, cachename="reg.nc")

    regis_ds2 = nlmod.read.regis.get_regis(extent, cachedir=tmpdir, cachename="reg.nc")

    assert regis_ds1.equals(regis_ds2)


@pytest.mark.skip("too slow")
def test_do_not_use_cached_regis(tmpdir):
    # cache regis
    extent = [98700.0, 99000.0, 489500.0, 489700.0]
    regis_ds1 = nlmod.read.regis.get_regis(
        extent, cachedir=tmpdir, cachename="regis.nc"
    )

    # do not use cache because extent is different
    extent = [99100.0, 99400.0, 489100.0, 489400.0]
    regis_ds2 = nlmod.read.regis.get_regis(
        extent, cachedir=tmpdir, cachename="regis.nc"
    )

    assert not regis_ds1.equals(regis_ds2)
