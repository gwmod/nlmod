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
        bgt = nlmod.read.bgt.download_bgt(extent)
        bgt.to_file(fname)
    return gpd.read_file(fname)


def get_regis():
    fname = os.path.join(model_ws, "regis.nc")
    if not os.path.isfile(fname):
        if not os.path.isdir(model_ws):
            os.makedirs(model_ws)
        regis = nlmod.read.regis.download_regis(extent)
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
    """Test fillnan_da function improvements from PR."""
    # Test structured grid with uniform spacing (uses distance_transform_edt opt)
    ds = get_structured_model_ds()
    original = ds["top"].copy()

    # Test single NaN - should use fast distance_transform_edt path
    ds["top"][5, 5] = np.nan
    expected_value = original[5, 4].values  # nearest neighbor value

    top_nearest = nlmod.resample.fillnan_da(ds["top"], ds=ds, method="nearest")
    assert not np.isnan(top_nearest[5, 5])
    assert np.isclose(top_nearest[5, 5], expected_value, rtol=1e-10)

    # Test linear method (should use griddata fallback)
    top_linear = nlmod.resample.fillnan_da(ds["top"], ds=ds, method="linear")
    assert not np.isnan(top_linear[5, 5])

    # Test that original values are preserved where not NaN
    mask_valid = ~ds["top"].isnull()
    np.testing.assert_allclose(
        top_nearest.where(mask_valid), original.where(mask_valid), equal_nan=True
    )

    # Test vertex grid with improved coordinate handling
    ds_vertex = get_vertex_model_ds()
    ds_vertex["top"][100] = np.nan

    # Test with ds parameter (should extract x,y from ds)
    top_vertex_ds = nlmod.resample.fillnan_da(ds_vertex["top"], ds=ds_vertex)
    assert not np.isnan(top_vertex_ds[100])

    # Test vertex grid with coordinates in DataArray
    x_coords = ds_vertex["x"].values
    y_coords = ds_vertex["y"].values
    ds_vertex_coords = ds_vertex.copy()
    ds_vertex_coords["top"] = ds_vertex_coords["top"].assign_coords(
        x=("icell2d", x_coords), y=("icell2d", y_coords)
    )
    top_vertex_coords = nlmod.resample.fillnan_da(ds_vertex_coords["top"])
    assert not np.isnan(top_vertex_coords[100])


def test_fillnan_da_vertex_grid_coordinates():
    """Test improved coordinate handling in fillnan_da_vertex_grid."""
    for method in ("nearest", "linear"):
        ds_vertex = get_vertex_model_ds()
        ds_vertex["top"][100] = np.nan

        # Test with ds parameter
        for ds in (ds_vertex, None):
            top_ds = nlmod.resample.fillnan_da_vertex_grid(
                ds_vertex["top"], method=method, ds=ds
            )
            assert not np.isnan(top_ds[100])

        # Test with explicit x,y coordinates
        x_coords = ds_vertex["x"].values
        y_coords = ds_vertex["y"].values
        for ds in (ds_vertex, None):
            top_xy = nlmod.resample.fillnan_da_vertex_grid(
                ds_vertex["top"], x=x_coords, y=y_coords, method=method, ds=ds
            )
            assert not np.isnan(top_xy[100])
            assert np.isclose(top_ds[100], top_xy[100])

        # Test with coordinates in DataArray
        vertex_da_with_coords = ds_vertex["top"].assign_coords(
            x=("icell2d", x_coords), y=("icell2d", y_coords)
        )
        top_coords = nlmod.resample.fillnan_da_vertex_grid(
            vertex_da_with_coords, method=method
        )
        assert not np.isnan(top_coords[100])
        assert np.isclose(top_ds[100], top_coords[100])


def test_fillnan_da_uniform_vs_nonuniform():
    """Test optimization path selection for uniform vs non-uniform grids."""
    ds = get_structured_model_ds()

    # Create test data with known pattern
    test_values = np.arange(ds["top"].size).reshape(ds["top"].shape)
    ds["top"].values = test_values

    # Add NaN in center
    center_y, center_x = ds["top"].shape[0] // 2, ds["top"].shape[1] // 2
    ds["top"][center_y, center_x] = np.nan

    # Test uniform grid (should use distance_transform_edt)
    result_uniform = nlmod.resample.fillnan_da(ds["top"], ds=ds, method="nearest")

    # Create non-uniform grid by adjusting coordinates
    ds_nonuniform = ds.copy()
    x_coords = ds.x.values
    x_coords[5:] += 10  # Make spacing non-uniform
    ds_nonuniform = ds_nonuniform.assign_coords(x=x_coords)

    # Test non-uniform grid (should use griddata)
    result_nonuniform = nlmod.resample.fillnan_da(
        ds_nonuniform["top"], ds=ds_nonuniform, method="nearest"
    )

    # Both should fill the NaN but may give different results
    assert not np.isnan(result_uniform[center_y, center_x])
    assert not np.isnan(result_nonuniform[center_y, center_x])


def test_fillnan_da_error_handling():
    """Test improved error handling."""
    import pytest
    import xarray as xr

    # Test vertex grid with wrong dimensions
    ds_vertex = get_vertex_model_ds()
    wrong_vertex_da = ds_vertex["top"].rename({"icell2d": "wrong_dim"})

    with pytest.raises(ValueError, match="is not a valid GridTypeDims"):
        nlmod.resample.fillnan_da_vertex_grid(wrong_vertex_da, ds=ds_vertex)

    # Test vertex grid without coordinates (improved error handling from PR)
    # Create a clean DataArray without x,y coordinates
    clean_vertex_da = xr.DataArray(
        ds_vertex["top"].values,
        dims=("icell2d",),
        coords={"icell2d": ds_vertex.icell2d},
    )

    with pytest.raises(ValueError, match="x or ds must be provided"):
        nlmod.resample.fillnan_da_vertex_grid(clean_vertex_da)

    # Test y coordinate error
    with pytest.raises(ValueError, match="y or ds must be provided"):
        nlmod.resample.fillnan_da_vertex_grid(clean_vertex_da, x=ds_vertex["x"].values)


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

    bgt_line = bgt.copy()
    bgt_line.geometry = bgt.boundary

    for agg_method in [
        "nearest",
        "area_weighted",
        "max_area",
        "length_weighted",
        "max_length",
        # "center_grid",
        "max",
        "min",
        "mean",
        "sum",
    ]:
        if agg_method in ["length_weighted", "max_length"]:
            gdf = bgt_line
        else:
            gdf = bgt
        nlmod.grid.gdf_to_da(gdf, ds, column="values", agg_method=agg_method)


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


def test_gdf_area_per_index_to_da():
    bgt = get_bgt()
    ds = get_structured_model_ds()
    ds["bgt"] = nlmod.grid.gdf_area_to_da(bgt, ds)
    assert (ds["bgt"].sum("index") > 0).any()

    ds = get_vertex_model_ds()
    ds["bgt"] = nlmod.grid.gdf_area_to_da(bgt, ds)
    assert (ds["bgt"].sum("index") > 0).any()
