import test_001_model

import nlmod


def test_get_gdf_opp_water():
    ds = test_001_model.get_ds_from_cache()
    nlmod.read.rws.get_gdf_surface_water(ds)


def test_surface_water_to_dataset():
    # model with sea
    ds = test_001_model.get_ds_from_cache("basic_sea_model")
    name = "surface_water"
    nlmod.read.rws.get_surface_water(ds, name)


def test_get_northsea_seamodel():
    # model with sea
    ds = test_001_model.get_ds_from_cache("basic_sea_model")
    ds_sea = nlmod.read.rws.get_northsea(ds)

    assert (ds_sea.northsea == 1).sum() > 0


def test_get_northsea_nosea():
    # model without sea
    ds = test_001_model.get_ds_from_cache("small_model")
    ds_sea = nlmod.read.rws.get_northsea(ds)

    assert (ds_sea.northsea == 1).sum() == 0


def test_fill_top_bot_kh_kv_seamodel():
    # model with sea
    ds = test_001_model.get_ds_from_cache("basic_sea_model")
    ds.update(nlmod.read.rws.get_northsea(ds))

    fal = nlmod.layers.get_first_active_layer(ds)
    fill_mask = (fal == fal.nodata) * ds["northsea"]
    nlmod.layers.fill_top_bot_kh_kv_at_mask(ds, fill_mask)


def test_fill_top_bot_kh_kv_nosea():
    # model with sea
    ds = test_001_model.get_ds_from_cache("small_model")
    ds.update(nlmod.read.rws.get_northsea(ds))

    fal = nlmod.layers.get_first_active_layer(ds)
    fill_mask = (fal == fal.nodata) * ds["northsea"]
    nlmod.layers.fill_top_bot_kh_kv_at_mask(ds, fill_mask)


def test_get_bathymetry_seamodel():
    # model with sea
    ds = test_001_model.get_ds_from_cache("basic_sea_model")
    ds.update(nlmod.read.rws.get_northsea(ds))
    ds_bathymetry = nlmod.read.jarkus.get_bathymetry(ds, ds["northsea"])

    assert (~ds_bathymetry.bathymetry.isnull()).sum() > 0


def test_get_bathymetrie_nosea():
    # model without sea
    ds = test_001_model.get_ds_from_cache("small_model")
    ds.update(nlmod.read.rws.get_northsea(ds))
    ds_bathymetry = nlmod.read.jarkus.get_bathymetry(ds, ds["northsea"])

    assert (~ds_bathymetry.bathymetry.isnull()).sum() == 0


def test_add_bathymetrie_to_top_bot_kh_kv_seamodel():
    # model with sea
    ds = test_001_model.get_ds_from_cache("basic_sea_model")
    ds.update(nlmod.read.rws.get_northsea(ds))
    ds.update(nlmod.read.jarkus.get_bathymetry(ds, ds["northsea"]))

    fal = nlmod.layers.get_first_active_layer(ds)
    fill_mask = (fal == fal.nodata) * ds["northsea"]

    nlmod.read.jarkus.add_bathymetry_to_top_bot_kh_kv(ds, ds["bathymetry"], fill_mask)
