import os
import tempfile

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import nlmod

tmpdir = tempfile.gettempdir()
tst_model_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")


def test_model_directories(tmpdir):
    model_ws = os.path.join(tmpdir, "test_model")
    figdir, cachedir = nlmod.util.get_model_dirs(model_ws)


def test_snap_extent():
    extent = (0.22, 1056.12, 7.43, 1101.567)
    new_extent = nlmod.dims.snap_extent(extent, 10, 20)
    assert new_extent == [0.0, 1060.0, 0.0, 1120.0]

    extent = (1000, 2000, 8000, 10000)
    new_extent = nlmod.dims.snap_extent(extent, 250, 55)
    assert new_extent == [1000.0, 2000.0, 7975.0, 10010.0]


def get_ds_time_steady(tmpdir, modelname="test"):
    model_ws = os.path.join(tmpdir, "test_model")
    ds = nlmod.base.set_ds_attrs(xr.Dataset(), modelname, model_ws)
    ds = nlmod.time.set_ds_time(ds, time=["2015-1-2"], start="2015-1-1", steady=True)
    return ds


def get_ds_time_transient(tmpdir, modelname="test"):
    model_ws = os.path.join(tmpdir, "test_model")
    ds = nlmod.base.set_ds_attrs(xr.Dataset(), modelname, model_ws)
    nper = 11
    time = pd.date_range(start="2015-1-2", periods=nper, freq="D")
    steady = np.zeros(nper)
    ds = nlmod.time.set_ds_time(ds, time=time, start="2015-1-1", steady=steady)
    return ds


def test_get_ds():
    model_ws = os.path.join(tmpdir, "test_model_ds")
    nlmod.get_ds(
        [-500, 500, -500, 500],
        delr=10.0,
        layer=3,
        top=0.0,
        botm=[-10, -20, -30],
        kh=[100, 1, 5],
        kv=[10, 0.1, 0.5],
        model_ws=model_ws,
        model_name="test_ds",
    )


def test_get_ds_variable_delrc():
    model_ws = os.path.join(tmpdir, "test_model_ds")
    nlmod.get_ds(
        extent=[-500, 500, -500, 500],
        delr=[100] * 5 + [20] * 5 + [100] * 4,
        delc=[100] * 4 + [20] * 5 + [100] * 5,
        layer=3,
        top=0.0,
        botm=[-10, -20, -30],
        kh=[100, 1, 5],
        kv=[10, 0.1, 0.5],
        model_ws=model_ws,
        model_name="test_ds",
    )


@pytest.mark.slow
def test_create_small_model_grid_only(tmpdir, model_name="test"):
    extent = [98700.0, 99000.0, 489500.0, 489700.0]
    # extent, nrow, ncol = nlmod.read.regis.fit_extent_to_regis(extent, 100, 100)
    regis_geotop_ds = nlmod.read.regis.get_combined_layer_models(
        extent, regis_botm_layer="KRz5", use_regis=True, use_geotop=True
    )
    model_ws = os.path.join(tmpdir, model_name)
    ds = nlmod.base.to_model_ds(
        regis_geotop_ds, model_name, model_ws, delr=100.0, delc=100.0
    )
    assert ds.sizes["layer"] == 5

    nper = 11
    steady = np.zeros(nper, dtype=int)
    steady[0] = 1
    ds = nlmod.time.set_ds_time(
        ds,
        time=pd.date_range("2015-1-2", periods=nper, freq="D"),
        start="2015-1-1",
        steady=steady,
    )

    # create simulation
    sim = nlmod.sim.sim(ds)

    # create time discretisation
    _ = nlmod.sim.tdis(ds, sim)

    # create groundwater flow model
    gwf = nlmod.gwf.gwf(ds, sim)

    # create ims
    _ = nlmod.sim.ims(sim)

    # Create discretization
    _ = nlmod.gwf.dis(ds, gwf)

    # save ds
    ds.to_netcdf(os.path.join(tst_model_dir, "small_model.nc"))


@pytest.mark.slow
def test_create_sea_model_grid_only(tmpdir, model_name="test"):
    extent = [95000.0, 105000.0, 494000.0, 500000.0]
    # extent, nrow, ncol = nlmod.read.regis.fit_extent_to_regis(extent, 100, 100)
    regis_geotop_ds = nlmod.read.regis.get_combined_layer_models(
        extent, use_regis=True, use_geotop=True
    )
    model_ws = os.path.join(tmpdir, model_name)
    ds = nlmod.base.to_model_ds(
        regis_geotop_ds, model_name, model_ws, delr=100.0, delc=100.0
    )

    nper = 11
    steady = np.zeros(nper, dtype=int)
    steady[0] = 1
    ds = nlmod.time.set_ds_time(
        ds,
        time=pd.date_range("2015-1-2", periods=nper, freq="D"),
        start="2005-1-1",
        steady=steady,
    )

    # save ds
    ds.to_netcdf(os.path.join(tst_model_dir, "basic_sea_model.nc"))


@pytest.mark.slow
def test_create_sea_model_grid_only_delr_delc_50(tmpdir, model_name="test"):
    ds = get_ds_time_transient(tmpdir)
    extent = [95000.0, 105000.0, 494000.0, 500000.0]
    # extent, nrow, ncol = nlmod.read.regis.fit_extent_to_regis(extent, 50.0, 50.0)
    regis_geotop_ds = nlmod.read.regis.get_combined_layer_models(
        extent, use_regis=True, use_geotop=True
    )
    model_ws = os.path.join(tmpdir, model_name)
    ds = nlmod.base.to_model_ds(
        regis_geotop_ds, model_name, model_ws, delr=50.0, delc=50.0
    )

    # save ds
    ds.to_netcdf(os.path.join(tst_model_dir, "sea_model_grid_50.nc"))


@pytest.mark.slow
def test_create_sea_model(tmpdir):
    ds = xr.open_dataset(
        os.path.join(tst_model_dir, "basic_sea_model.nc"), mask_and_scale=False
    )
    # create simulation
    sim = nlmod.sim.sim(ds)

    # create time discretisation
    _ = nlmod.sim.tdis(ds, sim)

    # create groundwater flow model
    gwf = nlmod.gwf.gwf(ds, sim)

    # create ims
    _ = nlmod.sim.ims(sim)

    # Create discretization
    _ = nlmod.gwf.dis(ds, gwf)

    # create node property flow
    _ = nlmod.gwf.npf(ds, gwf, save_flows=True)

    # Create the initial conditions package
    _ = nlmod.gwf.ic(ds, gwf, starting_head=1.0)

    # Create the output control package
    _ = nlmod.gwf.oc(ds, gwf)

    # voeg grote oppervlaktewaterlichamen toe
    da_name = "surface_water"
    gdf_surface_water = nlmod.read.rws.get_gdf_surface_water(ds=ds)
    ds.update(
        nlmod.read.rws.get_surface_water(ds, gdf=gdf_surface_water, da_basename=da_name)
    )
    _ = nlmod.gwf.ghb(ds, gwf, bhead=f"{da_name}_stage", cond=f"{da_name}_cond")

    # surface level drain
    ds.update(nlmod.read.ahn.get_ahn(ds))
    _ = nlmod.gwf.surface_drain_from_ds(ds, gwf, 0.1)

    # add constant head cells at model boundaries
    ds.update(nlmod.grid.mask_model_edge(ds))
    _ = nlmod.gwf.chd(ds, gwf, mask="edge_mask", head="starting_head")

    # add knmi recharge to the model datasets
    ds.update(nlmod.read.knmi.get_recharge(ds))
    # create recharge package
    _ = nlmod.gwf.rch(ds, gwf)

    _ = nlmod.sim.write_and_run(sim, ds)


@pytest.mark.slow
def test_create_sea_model_perlen_list(tmpdir):
    ds = xr.open_dataset(os.path.join(tst_model_dir, "basic_sea_model.nc"))

    # update model_ws
    model_ws = os.path.join(tmpdir, "test_model_perlen_list")
    ds = nlmod.base.set_ds_attrs(ds, ds.model_name, model_ws)

    # create transient with perlen list
    start = ds.time.start
    perlen = [3650, 14, 10, 11]  # length of the time steps
    steady = np.zeros(len(perlen), dtype=int)
    steady[0] = 1

    # drop time dimension before setting time
    ds = ds.drop_dims("time")

    # update current ds with new time dicretisation
    ds = nlmod.time.set_ds_time(
        ds,
        time=np.cumsum(perlen),
        start=start,
        steady=steady,
    )

    # create simulation
    sim = nlmod.sim.sim(ds)

    # create time discretisation
    _ = nlmod.sim.tdis(ds, sim)

    # create groundwater flow model
    gwf = nlmod.gwf.gwf(ds, sim)

    # create ims
    _ = nlmod.sim.ims(sim)

    # Create discretization
    nlmod.gwf.dis(ds, gwf)

    # create node property flow
    nlmod.gwf.npf(ds, gwf)

    # Create the initial conditions package
    nlmod.gwf.ic(ds, gwf, starting_head=1.0)

    # Create the output control package
    nlmod.gwf.oc(ds, gwf)

    # voeg grote oppervlaktewaterlichamen toe
    da_name = "surface_water"
    gdf_surface_water = nlmod.read.rws.get_gdf_surface_water(ds=ds)
    ds.update(
        nlmod.read.rws.get_surface_water(ds, gdf=gdf_surface_water, da_basename=da_name)
    )
    _ = nlmod.gwf.ghb(ds, gwf, bhead=f"{da_name}_stage", cond=f"{da_name}_cond")

    # surface level drain
    ds.update(nlmod.read.ahn.get_ahn(ds))
    nlmod.gwf.surface_drain_from_ds(ds, gwf, 1.0)

    # add constant head cells at model boundaries
    ds.update(nlmod.grid.mask_model_edge(ds))
    nlmod.gwf.chd(ds, gwf, mask="edge_mask", head="starting_head")

    # add knmi recharge to the model datasets
    ds.update(nlmod.read.knmi.get_recharge(ds))
    # create recharge package
    nlmod.gwf.rch(ds, gwf)

    nlmod.sim.write_and_run(sim, ds)


@pytest.mark.slow
def test_create_sea_model_perlen_14(tmpdir):
    ds = xr.open_dataset(os.path.join(tst_model_dir, "basic_sea_model.nc"))

    # update model_ws
    model_ws = os.path.join(tmpdir, "test_model_perlen_14")
    ds = nlmod.base.set_ds_attrs(ds, ds.model_name, model_ws)

    # create transient with perlen list
    perlen = 14  # length of the transient time steps
    nper = 4
    start = ds.time.start
    perlen = perlen * np.ones(nper)
    perlen[0] = 3652.0  # length of the steady state step
    steady = np.zeros(nper, dtype=int)
    steady[0] = 1
    time = nlmod.time.ds_time_idx_from_tdis_settings(start, perlen=perlen)

    # drop time dimension before setting time
    ds = ds.drop_dims("time")

    # update current ds with new time discretization
    ds = nlmod.time.set_ds_time(
        ds,
        time=time,
        start=start,
        steady=steady,
    )

    # create simulation
    sim = nlmod.sim.sim(ds)

    # create time discretisation
    _ = nlmod.sim.tdis(ds, sim)

    # create ims
    _ = nlmod.sim.ims(sim)

    # create groundwater flow model
    gwf = nlmod.gwf.gwf(ds, sim)

    # Create discretization
    nlmod.gwf.dis(ds, gwf)

    # create node property flow
    nlmod.gwf.npf(ds, gwf)

    # Create the initial conditions package
    nlmod.gwf.ic(ds, gwf, starting_head=1.0)

    # Create the output control package
    nlmod.gwf.oc(ds, gwf)

    # voeg grote oppervlaktewaterlichamen toe
    da_name = "surface_water"
    gdf_surface_water = nlmod.read.rws.get_gdf_surface_water(ds=ds)
    ds.update(
        nlmod.read.rws.get_surface_water(ds, gdf=gdf_surface_water, da_basename=da_name)
    )
    _ = nlmod.gwf.ghb(ds, gwf, bhead=f"{da_name}_stage", cond=f"{da_name}_cond")

    # surface level drain
    ds.update(nlmod.read.ahn.get_ahn(ds))
    nlmod.gwf.surface_drain_from_ds(ds, gwf, 1.0)

    # add constant head cells at model boundaries
    ds.update(nlmod.grid.mask_model_edge(ds))
    nlmod.gwf.chd(ds, gwf, mask="edge_mask", head="starting_head")

    # add knmi recharge to the model datasets
    ds.update(nlmod.read.knmi.get_recharge(ds))
    # create recharge package
    nlmod.gwf.rch(ds, gwf)

    nlmod.sim.write_and_run(sim, ds)


# obtaining the test models
def get_ds_from_cache(name="small_model"):
    ds = xr.open_dataset(os.path.join(tst_model_dir, name + ".nc"))
    return ds


# other functions
def _check_tmpdir(tmpdir):
    # pytest uses a LocalPath object for the tmpdir argument when testing
    # this function convert a LocalPath object to a string

    if isinstance(tmpdir, str):
        return tmpdir
    else:
        return str(tmpdir)
