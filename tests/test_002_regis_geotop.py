import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import nlmod
from nlmod.read.geotop import split_layers_on_geul


def test_get_regis(extent=None):
    if extent is None:
        extent = [98600.0, 99000.0, 489400.0, 489700.0]
    regis_ds = nlmod.read.regis.download_regis(extent)

    assert regis_ds.sizes["layer"] == 20


def test_get_regis_botm_layer_BEk1(
    extent=None,
    botm_layer="MSc",
):
    if extent is None:
        extent = [98700.0, 99000.0, 489500.0, 489700.0]
    regis_ds = nlmod.read.regis.download_regis(extent, botm_layer)
    assert regis_ds.sizes["layer"] == 15
    assert regis_ds.layer.values[-1] == botm_layer


def test_get_regis_only_c(extent=None):
    if extent is None:
        extent = [98700.0, 99000.0, 489500.0, 489700.0]
    regis_ds = nlmod.read.regis.download_regis(extent, variables="c")
    assert np.all([x == "c" for x in regis_ds.data_vars])
    assert regis_ds.sizes["layer"] == 8


def test_get_regis_only_c_and_kd(extent=None):
    if extent is None:
        extent = [98700.0, 99000.0, 489500.0, 489700.0]
    regis_ds = nlmod.read.regis.download_regis(extent, variables=["c", "kD"])
    assert np.all([x in ["c", "kD"] for x in regis_ds.data_vars])
    assert regis_ds.sizes["layer"] == 18


def test_get_geotop(extent=None):
    if extent is None:
        extent = [98600.0, 99000.0, 489400.0, 489700.0]
    geotop_ds = nlmod.read.geotop.download_geotop(extent)
    line = [(extent[0], extent[2]), (extent[1], extent[3])]

    # also test the plot-methods
    f, ax = plt.subplots()
    nlmod.plot.geotop_lithok_in_cross_section(line, geotop_ds, ax=ax)

    f, ax = plt.subplots()
    nlmod.plot.geotop_lithok_on_map(geotop_ds, z=-20.2, ax=ax)


def test_get_regis_geotop(extent=None):
    if extent is None:
        extent = [98600.0, 99000.0, 489400.0, 489700.0]
    regis_geotop_ds = nlmod.read.regis.get_combined_layer_models(
        extent, use_regis=True, use_geotop=True
    )
    regis_geotop_ds = nlmod.base.to_model_ds(regis_geotop_ds)
    assert regis_geotop_ds.sizes["layer"] == 24


def test_get_regis_geotop_keep_all_layers(
    extent=None,
):
    if extent is None:
        extent = [98600.0, 99000.0, 489400.0, 489700.0]
    regis_geotop_ds = nlmod.read.regis.get_combined_layer_models(
        extent, use_regis=True, use_geotop=True, remove_nan_layers=False
    )
    assert regis_geotop_ds.sizes["layer"] == 137


def test_add_kh_and_kv():
    gt = nlmod.read.geotop.download_geotop(
        [118200, 118300, 439800, 439900], probabilities=True
    )

    # test with a value for kh for each lithoclass
    df = nlmod.read.geotop.get_lithok_props()
    # stochastic = None/False is allready tested in methods above, so we onlt test stochastic=True
    gt = nlmod.read.geotop.add_kh_and_kv(gt, df, stochastic=True)

    # test with a value for kh and kv for each combination of lithoclass and stratigraphic unit
    df = nlmod.read.geotop.get_kh_kv_table()
    gt = nlmod.read.geotop.add_kh_and_kv(gt, df)
    # again, but using the stochastic method
    gt = nlmod.read.geotop.add_kh_and_kv(gt, df, stochastic=True)


def test_geulen_geotop():

    #  example geul at top
    strat = xr.DataArray(
        np.array([[0, 10, 10], [1, 1, 1], [2, 2, 2]])[:, None, :],
        coords={"z": [0, -0.5, -1.0], "y": [100], "x": [100, 200, 300]},
        dims=("z", "y", "x"),
    )
    units_no_geul = [0, 1, 2]
    geulen = [10]

    strat_modified, new_unit_order = split_layers_on_geul(strat, units_no_geul, geulen)
    assert np.array_equal(np.unique(strat_modified), np.unique(new_unit_order))

    # example geul at bottom
    strat = xr.DataArray(
        np.array([[0, 0, 0], [1, 1, 1], [2, 10, 10]])[:, None, :],
        coords={"z": [0, -0.5, -1.0], "y": [100], "x": [100, 200, 300]},
        dims=("z", "y", "x"),
    )
    units_no_geul = [0, 1, 2]
    geulen = [10]

    strat_modified, new_unit_order = split_layers_on_geul(strat, units_no_geul, geulen)
    assert np.array_equal(np.unique(strat_modified), np.unique(new_unit_order))

    # example split single layer
    strat = xr.DataArray(
        np.array([[0, 0, 0], [0, 10, 10], [0, 0, 0]])[:, None, :],
        coords={"z": [0, -0.5, -1.0], "y": [100], "x": [100, 200, 300]},
        dims=("z", "y", "x"),
    )
    units_no_geul = [0]
    geulen = [10]

    strat_modified, new_unit_order = split_layers_on_geul(strat, units_no_geul, geulen)
    assert np.array_equal(np.unique(strat_modified), np.unique(new_unit_order))

    # example simple add between layers
    strat = xr.DataArray(
        np.array([[0, 0, 0], [0, 10, 10], [1, 1, 1]])[:, None, :],
        coords={"z": [0, -0.5, -1.0], "y": [100], "x": [100, 200, 300]},
        dims=("z", "y", "x"),
    )
    units_no_geul = [0, 1]
    geulen = [10]

    strat_modified, new_unit_order = split_layers_on_geul(strat, units_no_geul, geulen)
    assert np.array_equal(np.unique(strat_modified), np.unique(new_unit_order))

    # example add between layers with geul at top and bottom
    strat = xr.DataArray(
        np.array([[0, 0, 10], [10, 10, 10], [10, 1, 1]])[:, None, :],
        coords={"z": [0, -0.5, -1.0], "y": [100], "x": [100, 200, 300]},
        dims=("z", "y", "x"),
    )
    units_no_geul = [0, 1]
    geulen = [10]

    strat_modified, new_unit_order = split_layers_on_geul(strat, units_no_geul, geulen)

    # example geul between and below
    strat = xr.DataArray(
        np.array([[0, 0, 0], [10, 1, 1], [0, 10, 10]])[:, None, :],
        coords={"z": [0, -0.5, -1.0], "y": [100], "x": [100, 200, 300]},
        dims=("z", "y", "x"),
    )
    units_no_geul = [0, 1]
    geulen = [10]

    strat_modified, new_unit_order = split_layers_on_geul(strat, units_no_geul, geulen)
    assert np.array_equal(np.unique(strat_modified), np.unique(new_unit_order))

    # example 2 geulen
    strat = xr.DataArray(
        np.array([[0, 10, 0], [1, 20, 1], [2, 2, 2]])[:, None, :],
        coords={"z": [0, -0.5, -1.0], "y": [100], "x": [100, 200, 300]},
        dims=("z", "y", "x"),
    )
    units_no_geul = [0, 1, 2]
    geulen = [10, 20]

    strat_modified, new_unit_order = split_layers_on_geul(strat, units_no_geul, geulen)
    assert np.array_equal(np.unique(strat_modified), np.unique(new_unit_order))
