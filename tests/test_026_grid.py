import os
import tempfile

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import nlmod

model_ws = os.path.join(tempfile.gettempdir(), "test_grid")
extent = [98000.0, 99000.0, 489000.0, 490000.0]


def get_bgt():
    fname = os.path.join(model_ws, "bgt.gpkg")
    if not os.path.isfile(fname):
        if not os.path.isdir(model_ws):
            os.makedirs(model_ws)
        bgt = nlmod.read.bgt.get_bgt(extent)
        bgt.to_file(fname)
    return gpd.read_file(fname)


def get_regis():
    fname = os.path.join(model_ws, "regis.nc")
    if not os.path.isfile(fname):
        if not os.path.isdir(model_ws):
            os.makedirs(model_ws)
        regis = nlmod.read.regis.get_regis(extent)
        regis.to_netcdf(fname)
    return xr.open_dataset(fname, decode_coords="all")


def get_structured_model_ds():
    model_ws = os.path.join(tempfile.gettempdir(), "test_grid_structured")
    fname = os.path.join(model_ws, "ds.nc")
    if not os.path.isfile(fname):
        if not os.path.isdir(model_ws):
            os.makedirs(model_ws)
        ds = nlmod.get_ds(extent, model_name="test_grid", model_ws=model_ws)
        ds.to_netcdf(fname)
    return xr.open_dataset(fname)


def get_structured_model_ds_rotated():
    model_ws = os.path.join(tempfile.gettempdir(), "test_grid_structured_rotated")
    fname = os.path.join(model_ws, "ds.nc")
    if not os.path.isfile(fname):
        if not os.path.isdir(model_ws):
            os.makedirs(model_ws)
        ds = nlmod.get_ds(extent, model_name="test_grid", model_ws=model_ws, angrot=15)
        ds.to_netcdf(fname)
    return xr.open_dataset(fname)


def get_vertex_model_ds(bgt=None):
    model_ws = os.path.join(tempfile.gettempdir(), "test_grid_vertex")
    fname = os.path.join(model_ws, "ds.nc")
    if not os.path.isfile(fname):
        if not os.path.isdir(model_ws):
            os.makedirs(model_ws)
        ds = get_structured_model_ds()
        if bgt is None:
            bgt = get_bgt()
        ds = nlmod.grid.refine(ds, model_ws=model_ws, refinement_features=[(bgt, 1)])
        ds.to_netcdf(fname)
    return xr.open_dataset(fname)


def get_vertex_model_ds_rotated(bgt=None):
    model_ws = os.path.join(tempfile.gettempdir(), "test_grid_vertex_rotated")
    fname = os.path.join(model_ws, "ds.nc")
    if not os.path.isfile(fname):
        if not os.path.isdir(model_ws):
            os.makedirs(model_ws)
        ds = get_structured_model_ds_rotated()
        if bgt is None:
            bgt = get_bgt()
        ds = nlmod.grid.refine(ds, model_ws=model_ws, refinement_features=[(bgt, 1)])
        ds.to_netcdf(fname)
    return xr.open_dataset(fname)


def test_get_ds_rotated():
    ds0 = get_structured_model_ds_rotated()
    assert ds0.extent[0] == 0 and ds0.extent[2] == 0
    assert ds0.xorigin == extent[0] and ds0.yorigin == extent[2]

    # test refine method, by refining in all cells that contain surface water polygons
    ds = get_vertex_model_ds_rotated()
    assert len(ds.area) > np.prod(ds0.area.shape)
    assert ds.extent[0] == 0 and ds.extent[2] == 0
    assert ds.xorigin == extent[0] and ds.yorigin == extent[2]

    f0, ax0 = plt.subplots()
    nlmod.plot.modelgrid(ds0, ax=ax0)
    f, ax = plt.subplots()
    nlmod.plot.modelgrid(ds, ax=ax)
    assert (np.array(ax.axis()) == np.array(ax0.axis())).all()


def test_vertex_da_to_ds():
    # for a normal grid
    ds0 = get_structured_model_ds()
    ds = get_vertex_model_ds()
    da = nlmod.resample.vertex_da_to_ds(ds["top"], ds0, method="linear")
    assert not da.isnull().all()
    da = nlmod.resample.vertex_da_to_ds(ds["botm"], ds0, method="linear")
    assert not da.isnull().all()

    # for a rotated grid
    ds0 = get_structured_model_ds_rotated()
    ds = get_vertex_model_ds_rotated()
    da = nlmod.resample.vertex_da_to_ds(ds["top"], ds0, method="linear")
    assert not da.isnull().all()
    da = nlmod.resample.vertex_da_to_ds(ds["botm"], ds0, method="linear")
    assert not da.isnull().all()


def test_fillnan_da():
    # for a structured grid
    ds = get_structured_model_ds()
    ds["top"][5, 5] = np.nan
    top = nlmod.resample.fillnan_da(ds["top"], ds=ds)
    assert not np.isnan(top[5, 5])

    # also for a vertex grid
    ds = get_vertex_model_ds()
    ds["top"][100] = np.nan
    mask = ds["top"].isnull()
    assert mask.any()
    top = nlmod.resample.fillnan_da(ds["top"], ds=ds)
    assert not top[mask].isnull().any()


def test_interpolate_gdf_to_array():
    bgt = get_bgt()
    bgt.geometry = bgt.centroid
    bgt["values"] = range(len(bgt))

    regis = get_regis()
    ds = nlmod.to_model_ds(regis, model_ws=model_ws)
    sim = nlmod.sim.sim(ds)
    gwf = nlmod.gwf.gwf(ds, sim)
    nlmod.gwf.dis(ds, gwf)

    nlmod.grid.interpolate_gdf_to_array(bgt, gwf, field="values", method="linear")


def test_gdf_to_da_methods():
    bgt = get_bgt()
    regis = get_regis()
    ds = nlmod.to_model_ds(regis)
    bgt["values"] = range(len(bgt))

    for agg_method in [
        "nearest",
        "area_weighted",
        "max_area",
        # "length_weighted",
        # "max_length",
        # "center_grid",
        "max",
        "min",
        "mean",
        "sum",
    ]:
        nlmod.grid.gdf_to_da(bgt, ds, column="values", agg_method=agg_method)


def test_gdf_to_bool_da():
    bgt = get_bgt()

    # test for a structured grid
    ds = get_structured_model_ds()
    da = nlmod.grid.gdf_to_bool_da(bgt, ds)
    assert da.any()

    # test for a vertex grid
    ds = get_vertex_model_ds()
    da = nlmod.grid.gdf_to_bool_da(bgt, ds)
    assert da.any()

    # tets for a slightly rotated structured grid
    ds = get_structured_model_ds_rotated()
    da = nlmod.grid.gdf_to_bool_da(bgt, ds)
    assert da.any()

    # test for a rotated vertex grid
    ds = get_vertex_model_ds_rotated()
    da = nlmod.grid.gdf_to_bool_da(bgt, ds)
    assert da.any()


def test_gdf_to_count_da():
    bgt = get_bgt()

    # test for a structured grid
    ds = get_structured_model_ds()
    da = nlmod.grid.gdf_to_count_da(bgt, ds)
    assert da.any()

    # test for a vertex grid
    ds = get_vertex_model_ds()
    da = nlmod.grid.gdf_to_count_da(bgt, ds)
    assert da.any()

    # tets for a slightly rotated structured grid
    ds = get_structured_model_ds_rotated()
    da = nlmod.grid.gdf_to_count_da(bgt, ds)
    assert da.any()

    # test for a rotated vertex grid
    ds = get_vertex_model_ds_rotated()
    da = nlmod.grid.gdf_to_count_da(bgt, ds)
    assert da.any()


def test_gdf_to_da():
    bgt = get_bgt()

    # test for a structured grid
    ds = get_structured_model_ds()
    da = nlmod.grid.gdf_to_da(bgt, ds, "relatieveHoogteligging", agg_method="max_area")
    assert not da.isnull().all()

    # test for a vertex grid
    ds = get_vertex_model_ds()
    da = nlmod.grid.gdf_to_da(bgt, ds, "relatieveHoogteligging", agg_method="max_area")
    assert not da.isnull().all()

    # tets for a slightly rotated structured grid
    ds = get_structured_model_ds_rotated()
    da = nlmod.grid.gdf_to_da(bgt, ds, "relatieveHoogteligging", agg_method="max_area")
    assert not da.isnull().all()

    # test for a rotated vertex grid
    ds = get_vertex_model_ds_rotated()
    da = nlmod.grid.gdf_to_da(bgt, ds, "relatieveHoogteligging", agg_method="max_area")
    assert not da.isnull().all()


def test_update_ds_from_layer_ds():
    bgt = get_bgt()
    regis = get_regis()

    # test for a structured grid
    ds = nlmod.get_ds(extent, delr=200)
    ds = nlmod.grid.update_ds_from_layer_ds(ds, regis, method="nearest")
    assert len(np.unique(ds["top"])) > 1
    ds = nlmod.grid.update_ds_from_layer_ds(ds, regis, method="average")
    assert len(np.unique(ds["top"])) > 1

    # test for a vertex grid
    model_ws = os.path.join(tempfile.gettempdir(), "test_grid_vertex_200")
    ds = nlmod.grid.refine(ds, model_ws=model_ws, refinement_features=[(bgt, 1)])
    ds = nlmod.grid.update_ds_from_layer_ds(ds, regis, method="nearest")
    assert len(np.unique(ds["top"])) > 1
    ds = nlmod.grid.update_ds_from_layer_ds(ds, regis, method="average")
    assert len(np.unique(ds["top"])) > 1

    # tets for a slightly rotated structured grid
    ds = nlmod.get_ds(extent, delr=200, angrot=15)
    ds = nlmod.grid.update_ds_from_layer_ds(ds, regis, method="nearest")
    assert len(np.unique(ds["top"])) > 1
    ds = nlmod.grid.update_ds_from_layer_ds(ds, regis, method="average")
    assert len(np.unique(ds["top"])) > 1

    # test for a rotated vertex grid
    model_ws = os.path.join(tempfile.gettempdir(), "test_grid_vertex_200_rotated")
    ds = nlmod.grid.refine(ds, model_ws=model_ws, refinement_features=[(bgt, 2)])
    ds = nlmod.grid.update_ds_from_layer_ds(ds, regis, method="nearest")
    assert len(np.unique(ds["top"])) > 1
    ds = nlmod.grid.update_ds_from_layer_ds(ds, regis, method="average")
    assert len(np.unique(ds["top"])) > 1
