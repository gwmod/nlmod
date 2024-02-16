import tempfile
import os
import numpy as np
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
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
    return xr.open_dataset(fname)


def test_get_ds_rotated():
    ds0 = nlmod.get_ds(extent, angrot=15)
    assert ds0.extent[0] == 0 and ds0.extent[2] == 0
    assert ds0.xorigin == extent[0] and ds0.yorigin == extent[2]

    # test refine method, by refining in all cells that contain surface water polygons
    bgt = get_bgt()
    ds = nlmod.grid.refine(ds0, model_ws=model_ws, refinement_features=[(bgt, 1)])
    assert len(ds.area) > np.prod(ds0.area.shape)
    assert ds.extent[0] == 0 and ds.extent[2] == 0
    assert ds.xorigin == extent[0] and ds.yorigin == extent[2]

    f0, ax0 = plt.subplots()
    nlmod.plot.modelgrid(ds0, ax=ax0)
    f, ax = plt.subplots()
    nlmod.plot.modelgrid(ds, ax=ax)
    assert (np.array(ax.axis()) == np.array(ax0.axis())).all()


def test_gdf_to_bool_da():
    bgt = get_bgt()

    # test for a structured grid
    ds = nlmod.get_ds(extent)
    da = nlmod.grid.gdf_to_bool_da(bgt, ds)
    assert da.any()

    # test for a vertex grid
    ds = nlmod.grid.refine(ds, model_ws=model_ws, refinement_features=[(bgt, 1)])
    da = nlmod.grid.gdf_to_bool_da(bgt, ds)
    assert da.any()

    # tets for a slightly rotated structured grid
    ds = nlmod.get_ds(extent, angrot=15)
    da = nlmod.grid.gdf_to_bool_da(bgt, ds)
    assert da.any()

    # test for a rotated vertex grid
    ds = nlmod.grid.refine(ds, model_ws=model_ws, refinement_features=[(bgt, 1)])
    da = nlmod.grid.gdf_to_bool_da(bgt, ds)
    assert da.any()


def test_gdf_to_da():
    bgt = get_bgt()

    # test for a structured grid
    ds = nlmod.get_ds(extent)
    da = nlmod.grid.gdf_to_da(bgt, ds, "relatieveHoogteligging", agg_method="max_area")
    assert not da.isnull().all()

    # test for a vertex grid
    ds = nlmod.grid.refine(ds, model_ws=model_ws, refinement_features=[(bgt, 1)])
    da = nlmod.grid.gdf_to_da(bgt, ds, "relatieveHoogteligging", agg_method="max_area")
    assert not da.isnull().all()

    # tets for a slightly rotated structured grid
    ds = nlmod.get_ds(extent, angrot=15)
    da = nlmod.grid.gdf_to_da(bgt, ds, "relatieveHoogteligging", agg_method="max_area")
    assert not da.isnull().all()

    # test for a rotated vertex grid
    ds = nlmod.grid.refine(ds, model_ws=model_ws, refinement_features=[(bgt, 1)])
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
    ds = nlmod.grid.refine(ds, model_ws=model_ws, refinement_features=[(bgt, 2)])
    ds = nlmod.grid.update_ds_from_layer_ds(ds, regis, method="nearest")
    assert len(np.unique(ds["top"])) > 1
    ds = nlmod.grid.update_ds_from_layer_ds(ds, regis, method="average")
    assert len(np.unique(ds["top"])) > 1
