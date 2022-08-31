import nlmod

import test_001_model


def test_get_recharge():

    # model with sea
    model_ds = test_001_model.test_get_model_ds_from_cache("sea_model_grid")

    # add knmi recharge to the model dataset
    model_ds.update(nlmod.read.knmi.get_recharge(model_ds))

    return model_ds


def test_get_recharge_steady_state():

    # model with sea
    model_ds = test_001_model.test_get_model_ds_from_cache("sea_model_grid")

    # modify mtime
    model_ds = model_ds.drop_dims("time")
    model_ds = nlmod.mdims.set_model_ds_time(
        model_ds, start_time="2000-1-1", perlen=3650
    )

    # add knmi recharge to the model dataset
    model_ds.update(nlmod.read.knmi.get_recharge(model_ds))

    return model_ds


def test_ahn_within_extent():

    extent = [95000.0, 105000.0, 494000.0, 500000.0]
    da = nlmod.read.ahn.get_ahn_within_extent(extent)

    assert not da.isnull().all(), "AHN only has nan values"

    return da


def test_ahn_split_extent():

    extent = [95000.0, 105000.0, 494000.0, 500000.0]
    da = nlmod.read.ahn.get_ahn_within_extent(extent, maxsize=1000)

    assert not da.isnull().all(), "AHN only has nan values"

    return da


def test_get_ahn():

    # model with sea
    model_ds = test_001_model.test_get_model_ds_from_cache("sea_model_grid")

    # add ahn data to the model dataset
    ahn_ds = nlmod.read.ahn.get_ahn(model_ds)

    assert not ahn_ds["ahn"].isnull().all(), "AHN only has nan values"

    return ahn_ds


def test_get_surface_water_ghb():

    # model with sea
    model_ds = test_001_model.test_get_model_ds_from_cache("sea_model_grid")

    # create simulation
    sim = nlmod.gwf.sim_from_model_ds(model_ds)

    # create time discretisation
    tdis = nlmod.gwf.tdis_from_model_ds(model_ds, sim)

    # create groundwater flow model
    gwf = nlmod.gwf.gwf_from_model_ds(model_ds, sim)

    # create ims
    ims = nlmod.gwf.ims_to_sim(sim)

    nlmod.gwf.dis_from_model_ds(model_ds, gwf)

    # add surface water levels to the model dataset
    model_ds.update(nlmod.read.rws.get_surface_water(model_ds, "surface_water"))

    return model_ds


def test_get_brp():
    extent = [116500, 120000, 439000, 442000]
    return nlmod.read.brp.get_percelen(extent)
