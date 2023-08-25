import numpy as np
import test_001_model
import xarray as xr

import nlmod


def sim_tdis_gwf_ims_from_ds(tmpdir):
    ds = test_001_model.get_ds_from_cache("basic_sea_model")

    # create simulation
    sim = nlmod.sim.sim(ds)

    # create time discretisation
    _ = nlmod.sim.tdis(ds, sim)

    # create ims
    _ = nlmod.sim.ims(sim)

    # create groundwater flow model
    gwf = nlmod.gwf.gwf(ds, sim)

    return sim, gwf


def dis_from_ds(tmpdir):
    ds = test_001_model.get_ds_from_cache("small_model")

    _, gwf = sim_tdis_gwf_ims_from_ds(tmpdir)

    nlmod.gwf.dis(ds, gwf)


def npf_from_ds(tmpdir):
    ds = test_001_model.get_ds_from_cache("small_model")
    _, gwf = sim_tdis_gwf_ims_from_ds(tmpdir)
    nlmod.gwf.dis(ds)
    nlmod.gwf.npf(ds, gwf)


def oc_from_ds(tmpdir):
    ds = test_001_model.get_ds_from_cache("small_model")
    _, gwf = sim_tdis_gwf_ims_from_ds(tmpdir)
    nlmod.gwf.oc(ds, gwf)


def sto_from_ds(tmpdir):
    ds = test_001_model.get_ds_from_cache("small_model")
    _, gwf = sim_tdis_gwf_ims_from_ds(tmpdir)
    nlmod.gwf.sto(ds, gwf)


def ghb_from_ds(tmpdir):
    ds = test_001_model.get_ds_from_cache("full_sea_model")
    _, gwf = sim_tdis_gwf_ims_from_ds(tmpdir)
    _ = nlmod.gwf.dis(ds, gwf)

    nlmod.gwf.ghb(ds, gwf, bhead="surface_water_stage", cond="surface_water_cond")


def rch_from_ds(tmpdir):
    ds = test_001_model.get_ds_from_cache("full_sea_model")
    _, gwf = sim_tdis_gwf_ims_from_ds(tmpdir)
    _ = nlmod.gwf.dis(ds, gwf)

    nlmod.gwf.rch(ds, gwf)


def drn_from_ds(tmpdir):
    ds = test_001_model.get_ds_from_cache("full_sea_model")
    _, gwf = sim_tdis_gwf_ims_from_ds(tmpdir)
    _ = nlmod.gwf.dis(ds, gwf)
    nlmod.gwf.surface_drain_from_ds(ds, gwf, 1.0)


def chd_from_ds(tmpdir):
    ds = test_001_model.get_ds_from_cache("small_model")
    _, gwf = sim_tdis_gwf_ims_from_ds(tmpdir)
    _ = nlmod.gwf.dis(ds, gwf)

    _ = nlmod.gwf.ic(ds, gwf, starting_head=1.0)

    # add constant head cells at model boundaries
    ds.update(nlmod.grid.mask_model_edge(ds, ds["idomain"]))
    nlmod.gwf.chd(ds, gwf, mask="edge_mask", head="starting_head")


def get_value_from_ds_datavar():
    ds = xr.Dataset(
        coords={
            "layer": [0, 1],
            "y": np.arange(10, -1, -1),
            "x": np.arange(10),
        },
    )
    shape = list(ds.dims.values())
    ds["test_var"] = ("layer", "y", "x"), np.arange(np.product(shape)).reshape(shape)

    # get value from ds
    v0 = nlmod.util._get_value_from_ds_datavar(
        ds, "test_var", "test_var", return_da=True
    )
    xr.testing.assert_equal(ds["test_var"], v0)

    # get value from ds, variable and stored name are different
    v1 = nlmod.util._get_value_from_ds_datavar(ds, "test", "test_var")
    xr.testing.assert_equal(ds["test_var"].values, v1)

    # do not get value from ds, value is Data Array, should log info msg
    v2 = nlmod.util._get_value_from_ds_datavar(ds, "test", v0, return_da=True)
    xr.testing.assert_equal(ds["test_var"], v2)

    # do not get value from ds, value is Data Array, no msg
    v0.name = "test2"
    v3 = nlmod.util._get_value_from_ds_datavar(ds, "test", v0, return_da=True)
    assert (v0 == v3).all()

    # return None, value is str but not in dataset, should log warning
    v4 = nlmod.util._get_value_from_ds_datavar(ds, "test", "test")
    assert v4 is None, "should be None"

    # return None, no warning
    v5 = nlmod.util._get_value_from_ds_datavar(ds, "test", None)
    assert v5 is None, "should be None."


def get_value_from_ds_attr():
    ds = xr.Dataset(
        coords={
            "layer": [0, 1],
            "y": np.arange(10, -1, -1),
            "x": np.arange(10),
        },
    )
    ds.attrs["test_float"] = 1.0
    ds.attrs["test_str"] = "test"

    # get float value, log debug msg
    v0 = nlmod.util._get_value_from_ds_attr(ds, "test_float")
    assert v0 == 1.0

    # get string value, log debug msg
    v1 = nlmod.util._get_value_from_ds_attr(ds, "test_str")
    assert v1 == "test"

    # get string value, different datavar name, log debug msg
    v2 = nlmod.util._get_value_from_ds_attr(ds, "test", "test_str")
    assert v2 == "test"

    # use user-provided value, log info msg
    v3 = nlmod.util._get_value_from_ds_attr(ds, "test_float", value=2.0)
    assert v3 == 2.0

    # use user-provided str value, log info msg
    v4 = nlmod.util._get_value_from_ds_attr(ds, "test", "test_str", value="test")
    assert v4 == "test"

    # use user-provided value, no msg, since "test" is not in attrs
    v5 = nlmod.util._get_value_from_ds_attr(ds, "test", "test", value=3.0)
    assert v5 == 3.0

    # user user-provided str value, no msg, since "test" is not in attrs
    v6 = nlmod.util._get_value_from_ds_attr(ds, "test", "test", value="test")
    assert v6 == "test"

    # return None, log warning
    v7 = nlmod.util._get_value_from_ds_attr(ds, "test")
    assert v7 is None

    # return None, log no warning, None is intended value
    v8 = nlmod.util._get_value_from_ds_attr(ds, "test", value=None)
    assert v8 is None
