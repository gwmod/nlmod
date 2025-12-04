import logging
import os

import pandas as pd
import pytest
import test_001_model
import xarray as xr
from shapely.geometry import LineString

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
    time = [pd.Timestamp.now().normalize()]
    ds = nlmod.time.set_ds_time(ds, start=time[0] - pd.Timedelta(days=21), time=time)
    with pytest.raises((KeyError, ValueError)):
        ds.update(nlmod.read.knmi.get_recharge(ds))


def test_get_recharge_add_stn_dimensions():
    ds = nlmod.get_ds(
        [100000, 110000, 420000, 430000],
        model_ws=os.path.join("models", "test_get_recharge_add_stn_dimensions"),
        model_name="test",
    )
    # set the top left cell to incactive, to test this functionality as well
    ds["active_domain"] = ds["area"] > 0
    ds["active_domain"].data[0, 0] = False
    time = pd.date_range("2024", "2025")
    ds = nlmod.time.set_ds_time(ds, start="2023", time=time)
    ds.update(nlmod.read.knmi.get_recharge(ds, add_stn_dimensions=True))

    # create simulation
    sim = nlmod.sim.sim(ds)
    _ = nlmod.sim.tdis(ds, sim)
    gwf = nlmod.gwf.gwf(ds, sim)
    _ = nlmod.sim.ims(sim)
    _ = nlmod.gwf.dis(ds, gwf)
    _ = nlmod.gwf.npf(ds, gwf)
    _ = nlmod.gwf.ic(ds, gwf, starting_head=1.0)
    _ = nlmod.gwf.rch(ds, gwf)
    _ = nlmod.gwf.evt(ds, gwf)

    # do not run, as this takes a lot of time
    # _ = nlmod.gwf.drn(ds, gwf, elev='top', cond='area')
    # nlmod.sim.write_and_run(sim, ds, write_ds=False)

    spd = gwf.rch.stress_period_data.data
    assert len(spd) == 1
    assert len(spd[0]) == 10000 - 1  # one inactive cell
    assert spd[0]["recharge"].dtype == object

    spd = gwf.evt.stress_period_data.data
    assert len(spd) == 1
    assert len(spd[0]) == 10000 - 1  # one inactive cell
    assert spd[0]["rate"].dtype == object


def test_add_recharge_as_float():
    ds = nlmod.get_ds(
        [100000, 110000, 420000, 430000],
        model_ws=os.path.join("models", "test_add_recharge_as_float"),
        model_name="test",
    )
    time = pd.date_range("2024", "2025")
    ds = nlmod.time.set_ds_time(ds, start="2023", time=time)

    sim = nlmod.sim.sim(ds)
    _ = nlmod.sim.tdis(ds, sim)
    gwf = nlmod.gwf.gwf(ds, sim)
    _ = nlmod.sim.ims(sim)
    _ = nlmod.gwf.dis(ds, gwf)
    _ = nlmod.gwf.rch(ds, gwf, recharge=0.1)

    spd = gwf.rch.stress_period_data.data
    assert len(spd) == 1
    assert len(spd[0]) == 10000
    assert (spd[0]["recharge"] == 0.1).all()


def test_ahn_within_extent():
    extent = [104000.0, 105000.0, 494000.0, 494600.0]
    da = nlmod.read.ahn.download_latest_ahn_from_wcs(extent)

    assert not da.isnull().all(), "AHN only has nan values"


def test_ahn_split_extent():
    extent = [104000.0, 105000.0, 494000.0, 494600.0]
    da = nlmod.read.ahn.download_latest_ahn_from_wcs(extent, maxsize=1000)

    assert not da.isnull().all(), "AHN only has nan values"


def test_get_ahn3():
    extent = [98000.0, 100000.0, 494000.0, 496000.0]
    da = nlmod.read.ahn.download_ahn3(extent)

    assert not da.isnull().all(), "AHN only has nan values"


def test_get_ahn4():
    extent = [98000.0, 100000.0, 494000.0, 496000.0]
    ahn = nlmod.read.ahn.download_ahn4(extent)
    assert isinstance(ahn, xr.DataArray)
    assert not ahn.isnull().all(), "AHN only has nan values"

    line = LineString([(99000, 495000), (100000, 496000)])
    ahn_line = nlmod.read.ahn.download_ahn_along_line(line, ahn=ahn)
    assert isinstance(ahn_line, xr.DataArray)


def test_get_ahn5():
    extent = [99500.0, 100000.0, 494500.0, 495000.0]
    da = nlmod.read.ahn.download_ahn5(extent)

    assert not da.isnull().all(), "AHN only has nan values"


def test_get_ahn():
    # model with sea
    ds = test_001_model.get_ds_from_cache("basic_sea_model")

    # add ahn data to the model dataset
    ahn_ds = nlmod.read.ahn.get_ahn(ds)

    assert not ahn_ds["ahn"].isnull().all(), "AHN only has nan values"


def test_get_ahn_at_point():
    nlmod.read.ahn.download_ahn_at_point(100010, 400010)


def test_check_ahn_files_up_to_date():
    try:
        tiles_new = nlmod.read.ahn._download_tiles_ellipsis()
    except:
        msg = "Cannot download ahn tiles. Will skip test to see if tiles are up to date"
        logging.warning(msg)
        return
    fname = os.path.join(nlmod.NLMOD_DATADIR, "ahn", "ellipsis_tiles.geojson")
    tiles_old = nlmod.read.ahn._get_tiles_from_file(fname)
    assert tiles_old.shape == tiles_new.shape
    assert tiles_old.eq(tiles_new).all(None)


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
    ds.update(nlmod.read.rws.get_surface_water(ds, da_basename="surface_water"))


def test_get_brp():
    extent = [116500, 120000, 439000, 442000]
    nlmod.read.brp.download_percelen_gdf(extent)


# disable because slow (~35 seconds depending on internet connection)
@pytest.mark.skip(reason="slow")
def test_get_bofek():
    # model with sea
    ds = test_001_model.get_ds_from_cache("basic_sea_model")

    # add knmi recharge to the model dataset
    gdf_bofek = nlmod.read.bofek.download_bofek_gdf(ds)

    assert not gdf_bofek.empty, "Bofek geodataframe is empty"
