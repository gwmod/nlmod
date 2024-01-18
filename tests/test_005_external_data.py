import pandas as pd
import xarray as xr
from shapely.geometry import LineString
import pytest
import test_001_model

import nlmod


def test_get_recharge():
    # model with sea
    ds = test_001_model.get_ds_from_cache("basic_sea_model")

    # add knmi recharge to the model dataset
    ds.update(nlmod.read.knmi.get_recharge(ds))


def test_get_recharge_most_common():
    # model with sea
    ds = nlmod.get_ds([100000, 110000, 420000, 430000])
    ds = nlmod.time.set_ds_time(ds, start="2021", time=pd.date_range("2022", "2023"))

    # add knmi recharge to the model dataset
    ds.update(nlmod.read.knmi.get_recharge(ds, most_common_station=True))


def test_get_recharge_steady_state():
    # model with sea
    ds = test_001_model.get_ds_from_cache("basic_sea_model")

    # modify mtime
    ds = ds.drop_dims("time")
    ds = nlmod.time.set_ds_time(ds, time=[3650], start="2000-1-1")

    # add knmi recharge to the model dataset
    ds.update(nlmod.read.knmi.get_recharge(ds))


def test_get_recharge_not_available():
    ds = nlmod.get_ds([100000, 110000, 420000, 430000])
    time = pd.date_range("2010", pd.Timestamp.now())
    ds = nlmod.time.set_ds_time(ds, start="2000", time=time)
    with pytest.raises(ValueError):
        ds.update(nlmod.read.knmi.get_recharge(ds))


def test_ahn_within_extent():
    extent = [104000.0, 105000.0, 494000.0, 494600.0]
    da = nlmod.read.ahn.get_latest_ahn_from_wcs(extent)

    assert not da.isnull().all(), "AHN only has nan values"


def test_ahn_split_extent():
    extent = [104000.0, 105000.0, 494000.0, 494600.0]
    da = nlmod.read.ahn.get_latest_ahn_from_wcs(extent, maxsize=1000)

    assert not da.isnull().all(), "AHN only has nan values"


def test_get_ahn3():
    extent = [98000.0, 100000.0, 494000.0, 496000.0]
    da = nlmod.read.ahn.get_ahn3(extent)

    assert not da.isnull().all(), "AHN only has nan values"


def test_get_ahn4():
    extent = [98000.0, 100000.0, 494000.0, 496000.0]
    ahn = nlmod.read.ahn.get_ahn4(extent)
    assert isinstance(ahn, xr.DataArray)
    assert not ahn.isnull().all(), "AHN only has nan values"

    line = LineString([(99000, 495000), (100000, 496000)])
    ahn_line = nlmod.read.ahn.get_ahn_along_line(line, ahn=ahn)
    assert isinstance(ahn_line, xr.DataArray)


def test_get_ahn():
    # model with sea
    ds = test_001_model.get_ds_from_cache("basic_sea_model")

    # add ahn data to the model dataset
    ahn_ds = nlmod.read.ahn.get_ahn(ds)

    assert not ahn_ds["ahn"].isnull().all(), "AHN only has nan values"


def test_get_ahn_at_point():
    nlmod.read.ahn.get_ahn_at_point(100010, 400010)


def test_get_surface_water_ghb():
    # model with sea
    ds = test_001_model.get_ds_from_cache("basic_sea_model")

    # create simulation
    sim = nlmod.sim.sim(ds)

    # create time discretisation
    nlmod.sim.tdis(ds, sim)

    # create groundwater flow model
    gwf = nlmod.gwf.gwf(ds, sim)

    # create ims
    nlmod.sim.ims(sim)

    nlmod.gwf.dis(ds, gwf)

    # add surface water levels to the model dataset
    ds.update(nlmod.read.rws.get_surface_water(ds, "surface_water"))


def test_get_brp():
    extent = [116500, 120000, 439000, 442000]
    nlmod.read.brp.get_percelen(extent)
