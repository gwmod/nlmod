import os

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import nlmod
import util


def _get_model_data_path(name):
    return os.path.join(util.get_model_data_dir(), name)


def _get_model_ws(name):
    return os.path.join(util.get_model_data_dir(), name)


def _ensure_cached_model(name):
    model_data_path = _get_model_data_path(name + ".nc")
    if os.path.exists(model_data_path):
        return model_data_path

    if name == "small_model":
        _build_small_model_grid_only(model_name="small_model")
    elif name == "sea_model_grid_only":
        _build_sea_model_grid_only(model_name="sea_model")
    elif name == "sea_model":
        _ensure_cached_model("sea_model_grid_only")
        _build_sea_model()
    else:
        raise FileNotFoundError(
            f"No cached model generator is configured for '{name}'."
        )

    if not os.path.exists(model_data_path):
        raise FileNotFoundError(
            f"Cached model '{name}' was not created at expected path: {model_data_path}"
        )

    return model_data_path


def test_model_directories(tmp_path):
    model_ws = os.path.join(tmp_path, "test_model")
    figdir, cachedir = nlmod.util.get_model_dirs(model_ws)


def test_snap_extent():
    extent = (0.22, 1056.12, 7.43, 1101.567)
    new_extent = nlmod.dims.snap_extent(extent, 10, 20)
    assert new_extent == [0.0, 1060.0, 0.0, 1120.0]

    extent = (1000, 2000, 8000, 10000)
    new_extent = nlmod.dims.snap_extent(extent, 250, 55)
    assert new_extent == [1000.0, 2000.0, 7975.0, 10010.0]


def test_get_ds(tmp_path):
    model_ws = os.path.join(tmp_path, "test_model_ds")
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


def test_get_ds_variable_delrc(tmp_path):
    model_ws = os.path.join(tmp_path, "test_model_ds")
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


def _build_small_model_grid_only(model_name="small_model"):
    extent = [98700.0, 99000.0, 489500.0, 489700.0]
    # extent, nrow, ncol = nlmod.read.regis.fit_extent_to_regis(extent, 100, 100)
    regis_geotop_ds = nlmod.read.regis.get_combined_layer_models(
        extent, regis_botm_layer="KRz5", use_regis=True, use_geotop=True
    )
    model_ws = _get_model_ws(model_name)
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
    ds.to_netcdf(_get_model_data_path("small_model.nc"))


@pytest.mark.slow
def test_create_small_model_grid_only():
    _build_small_model_grid_only(model_name="small_model")


def _build_sea_model_grid_only(model_name="sea_model"):
    extent = [95000.0, 105000.0, 494000.0, 500000.0]
    # extent, nrow, ncol = nlmod.read.regis.fit_extent_to_regis(extent, 100, 100)
    regis_geotop_ds = nlmod.read.regis.get_combined_layer_models(
        extent, use_regis=True, use_geotop=True
    )
    model_ws = _get_model_ws("sea_model_grid_only")
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

    ds.attrs["model_name"] = model_name
    ds.attrs["model_ws"] = model_ws

    # save ds
    ds.to_netcdf(_get_model_data_path("sea_model_grid_only.nc"))


@pytest.mark.slow
def test_create_sea_model_grid_only():
    _build_sea_model_grid_only(model_name="sea_model")


def _build_sea_model():
    ds = xr.open_dataset(
        _get_model_data_path("sea_model_grid_only.nc"), mask_and_scale=False
    )
    ds = nlmod.base.set_ds_attrs(
        ds, model_name="sea_model", model_ws=_get_model_ws("sea_model")
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
        nlmod.read.rws.discretize_surface_water(
            ds, gdf=gdf_surface_water, da_basename=da_name
        )
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

    ds.to_netcdf(_get_model_data_path("sea_model.nc"))

    _ = nlmod.sim.write_and_run(sim, ds)


@pytest.mark.slow
def test_create_sea_model():
    _build_sea_model()


# obtaining the test models
def get_ds_from_cache(name="small_model"):
    model_data_path = _ensure_cached_model(name)
    # Load into memory so the source file handle can be closed immediately.
    with xr.open_dataset(model_data_path) as ds:
        return ds.load()
