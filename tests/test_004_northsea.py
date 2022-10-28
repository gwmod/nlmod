# -*- coding: utf-8 -*-

import nlmod

import test_001_model


def test_get_gdf_opp_water():
    ds = test_001_model.test_get_ds_from_cache()
    gdf_surface_water = nlmod.read.rws.get_gdf_surface_water(ds)

    return gdf_surface_water


def test_surface_water_to_dataset():

    # model with sea
    ds = test_001_model.test_get_ds_from_cache("sea_model_grid")
    name = "surface_water"
    ds_surfwat = nlmod.read.rws.get_surface_water(ds, name)

    return ds_surfwat


def test_get_northsea_seamodel():

    # model with sea
    ds = test_001_model.test_get_ds_from_cache("basic_sea_model")
    ds_sea = nlmod.read.rws.get_northsea(ds)

    assert (ds_sea.northsea == 1).sum() > 0

    return ds_sea


def test_get_northsea_nosea():

    # model without sea
    ds = test_001_model.test_get_ds_from_cache("small_model")
    ds_sea = nlmod.read.rws.get_northsea(ds)

    assert (ds_sea.northsea == 1).sum() == 0

    return ds_sea


def test_fill_top_bot_kh_kv_seamodel():

    # model with sea
    ds = test_001_model.test_get_ds_from_cache("basic_sea_model")
    ds.update(nlmod.read.rws.get_northsea(ds))

    fill_mask = (ds["first_active_layer"] == ds.nodata) * ds["northsea"]
    ds = nlmod.mdims.fill_top_bot_kh_kv_at_mask(ds, fill_mask)

    return ds


def test_fill_top_bot_kh_kv_nosea():

    # model with sea
    ds = test_001_model.test_get_ds_from_cache("small_model")
    ds.update(nlmod.read.rws.get_northsea(ds))

    fill_mask = (ds["first_active_layer"] == ds.nodata) * ds["northsea"]
    ds = nlmod.mdims.fill_top_bot_kh_kv_at_mask(ds, fill_mask)

    return ds


def test_get_bathymetry_seamodel():

    # model with sea
    ds = test_001_model.test_get_ds_from_cache("basic_sea_model")
    ds.update(nlmod.read.rws.get_northsea(ds))
    ds_bathymetry = nlmod.read.jarkus.get_bathymetry(ds, ds["northsea"])

    assert (~ds_bathymetry.bathymetry.isnull()).sum() > 0

    return ds_bathymetry


def test_get_bathymetrie_nosea():

    # model without sea
    ds = test_001_model.test_get_ds_from_cache("small_model")
    ds.update(nlmod.read.rws.get_northsea(ds))
    ds_bathymetry = nlmod.read.jarkus.get_bathymetry(ds, ds["northsea"])

    assert (~ds_bathymetry.bathymetry.isnull()).sum() == 0

    return ds_bathymetry


def test_add_bathymetrie_to_top_bot_kh_kv_seamodel():

    # model with sea
    ds = test_001_model.test_get_ds_from_cache("basic_sea_model")
    ds.update(nlmod.read.rws.get_northsea(ds))
    ds.update(nlmod.read.jarkus.get_bathymetry(ds, ds["northsea"]))

    fill_mask = (ds["first_active_layer"] == ds.nodata) * ds["northsea"]

    ds = nlmod.read.jarkus.add_bathymetry_to_top_bot_kh_kv(
        ds, ds["bathymetry"], fill_mask
    )

    return ds
