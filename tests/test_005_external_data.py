import nlmod

import test_001_model


def test_get_recharge():
    # model with sea
    ds = test_001_model.test_get_ds_from_cache("basic_sea_model")

    # add knmi recharge to the model dataset
    ds.update(nlmod.read.knmi.get_recharge(ds))

    return ds


def test_get_recharge_steady_state():
    # model with sea
    ds = test_001_model.test_get_ds_from_cache("basic_sea_model")

    # modify mtime
    ds = ds.drop_dims("time")
    ds = nlmod.time.set_ds_time(ds, start_time="2000-1-1", perlen=3650)

    # add knmi recharge to the model dataset
    ds.update(nlmod.read.knmi.get_recharge(ds))

    return ds


def test_ahn_within_extent():
    extent = [95000.0, 105000.0, 494000.0, 500000.0]
    da = nlmod.read.ahn.get_ahn_from_wcs(extent)

    assert not da.isnull().all(), "AHN only has nan values"

    return da


def test_ahn_split_extent():
    extent = [95000.0, 105000.0, 494000.0, 500000.0]
    da = nlmod.read.ahn.get_ahn_from_wcs(extent, maxsize=1000)

    assert not da.isnull().all(), "AHN only has nan values"

    return da


def test_get_ahn3():
    extent = [98000.0, 100000.0, 494000.0, 496000.0]
    da = nlmod.read.ahn.get_ahn3(extent)

    assert not da.isnull().all(), "AHN only has nan values"


def test_get_ahn4():
    extent = [98000.0, 100000.0, 494000.0, 496000.0]
    da = nlmod.read.ahn.get_ahn4(extent)

    assert not da.isnull().all(), "AHN only has nan values"


def test_get_ahn():
    # model with sea
    ds = test_001_model.test_get_ds_from_cache("basic_sea_model")

    # add ahn data to the model dataset
    ahn_ds = nlmod.read.ahn.get_ahn(ds)

    assert not ahn_ds["ahn"].isnull().all(), "AHN only has nan values"

    return ahn_ds


def test_get_surface_water_ghb():
    # model with sea
    ds = test_001_model.test_get_ds_from_cache("basic_sea_model")

    # create simulation
    sim = nlmod.sim.sim(ds)

    # create time discretisation
    tdis = nlmod.sim.tdis(ds, sim)

    # create groundwater flow model
    gwf = nlmod.gwf.gwf(ds, sim)

    # create ims
    ims = nlmod.sim.ims(sim)

    nlmod.gwf.dis(ds, gwf)

    # add surface water levels to the model dataset
    ds.update(nlmod.read.rws.get_surface_water(ds, "surface_water"))

    return ds


def test_get_brp():
    extent = [116500, 120000, 439000, 442000]
    return nlmod.read.brp.get_percelen(extent)
