# %%
import os

import matplotlib.pyplot as plt
import numpy as np
import test_001_model
from pandas import DataFrame
from shapely.geometry import LineString

import nlmod
from nlmod.plot import DatasetCrossSection


def get_regis_horstermeer(cachedir=None, cachename="regis_horstermeer"):
    extent = [131000, 136800, 471500, 475700]
    if cachedir is None:
        cachedir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")
    if not os.path.isdir(cachedir):
        os.makedirs(cachedir)
    regis = nlmod.read.download_regis(extent, cachedir=cachedir, cachename=cachename)
    return regis


def get_regis_unstructured():
    regis = get_regis_horstermeer()
    cachedir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")
    if not os.path.isdir(cachedir):
        os.makedirs(cachedir)
    return nlmod.grid.refine(regis, cachedir)


def compare_layer_models(
    ds1,
    line,
    colors,
    ds2=None,
    zmin=-200,
    zmax=10,
    min_label_area=1000,
    title1="REGIS original",
    title2="Modified layers",
    xlabel="Distance along x-sec (m)",
    ylabel="m NAP",
):
    if ds2 is None:
        fig, ax1 = plt.subplots(1, 1, figsize=(14, 6))
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
    dcs1 = DatasetCrossSection(ds1, line=line, ax=ax1, zmin=zmin, zmax=zmax)
    dcs1.plot_layers(colors=colors, min_label_area=min_label_area)
    dcs1.plot_grid(linewidth=0.5, vertical=False)
    ax1.set_ylabel(ylabel)

    if ds2 is not None:
        ax1.set_title(title1)
        dcs2 = DatasetCrossSection(ds2, line=line, ax=ax2, zmin=zmin, zmax=zmax)
        dcs2.plot_layers(colors=colors, min_label_area=min_label_area)
        dcs2.plot_grid(linewidth=0.5, vertical=False)
        ax2.set_ylabel(ylabel)
        ax2.set_xlabel(xlabel)
        ax2.set_title(title2)
    else:
        ax1.set_xlabel(xlabel)


def plot_test(ds, ds_new, line=None, colors=None):
    if line is None:
        line = LineString([(ds.extent[0], ds.extent[2]), (ds.extent[1], ds.extent[3])])
    if colors is None:
        colors = nlmod.read.regis.get_legend()["color"].to_dict()
    compare_layer_models(ds, line, colors, ds_new)


def test_split_layers(plot=False):
    regis = get_regis_horstermeer()
    split_dict = {"PZWAz2": (0.3, 0.3, 0.4), "PZWAz3": (0.2, 0.2, 0.2, 0.2, 0.2)}
    regis_split, split_reindexer = nlmod.layers.split_layers_ds(
        regis, split_dict, return_reindexer=True
    )

    th_new = nlmod.layers.calculate_thickness(regis_split)
    assert (th_new >= 0).all()
    # make sure the total thickness of the two models is equal (does not assert to True yet)
    # assert th_new.sum() == nlmod.layers.calculate_thickness(regis).sum()

    if plot:
        colors = nlmod.read.regis.get_legend()["color"].to_dict()
        for j, i in split_reindexer.items():
            if j not in colors:
                colors[j] = colors[i]
        plot_test(regis, regis_split, colors=colors)


def test_add_layer_dim_to_top():
    regis = get_regis_horstermeer()
    ds = nlmod.layers.add_layer_dim_to_top(regis)
    assert "layer" in ds["top"].dims
    assert ds["botm"].isnull().any()


def test_combine_layers(plot=False):
    regis = get_regis_horstermeer()
    combine_layers = [
        regis.layer[regis.layer.str.startswith("URz")].data.tolist(),
        ["PZWAz2", "PZWAz3"],
    ]
    regis_combined = nlmod.layers.combine_layers_ds(
        regis, combine_layers, kD=None, c=None
    )

    th_new = nlmod.layers.calculate_thickness(regis_combined)
    assert (th_new >= 0).all()

    if plot:
        plot_test(regis, regis_combined)


def test_combine_layers_unstructured(plot=False):
    regis_u = get_regis_unstructured()
    combine_layers = [
        regis_u.layer[regis_u.layer.str.startswith("URz")].data.tolist(),
        ["PZWAz2", "PZWAz3"],
    ]
    regis_combined = nlmod.layers.combine_layers_ds(
        regis_u, combine_layers, kD=None, c=None
    )

    th_new = nlmod.layers.calculate_thickness(regis_combined)
    assert (th_new >= 0).all()

    if plot:
        plot_test(regis_u, regis_combined)


def test_set_layer_thickness(plot=False):
    regis = get_regis_horstermeer()
    ds_new = nlmod.layers.set_layer_thickness(regis.copy(deep=True), "WAk1", 10.0)

    th_new = nlmod.layers.calculate_thickness(ds_new)
    assert (np.abs(th_new.sel(layer="WAk1") - 10.0) < 0.001).all()
    assert (th_new >= 0).all()
    assert th_new.sum() == nlmod.layers.calculate_thickness(ds_new).sum()

    if plot:
        plot_test(regis, ds_new)


def test_set_minimum_layer_thickness(plot=False):
    regis = get_regis_horstermeer()
    ds_new = nlmod.layers.set_minimum_layer_thickness(
        regis.copy(deep=True), "WAk1", 5.0
    )

    th_new = nlmod.layers.calculate_thickness(ds_new)
    assert (th_new.sel(layer="WAk1") > 5.0 - 0.001).all()
    assert (th_new >= 0).all()
    assert th_new.sum() == nlmod.layers.calculate_thickness(ds_new).sum()

    if plot:
        plot_test(regis, ds_new)


def test_calculate_transmissivity():
    regis = get_regis_horstermeer()
    nlmod.layers.calculate_transmissivity(regis)


def test_calculate_resistance():
    regis = get_regis_horstermeer()
    # with the default value of between_layers=True
    nlmod.layers.calculate_resistance(regis)
    # and also with between_layers=False
    nlmod.layers.calculate_resistance(regis, between_layers=False)


def test_get_layer_of_z():
    regis = get_regis_horstermeer()
    z = -100

    layer = nlmod.layers.get_layer_of_z(regis, z)

    assert (regis["botm"].isel(layer=layer) < z).all()
    top = regis["botm"] + nlmod.layers.calculate_thickness(regis)
    assert (top.isel(layer=layer) > z).all()


def test_get_layer_of_z_above_model():
    ds = nlmod.get_ds([0, 1000, 0, 500], top=0, botm=[-10, -20])
    layer = nlmod.layers.get_layer_of_z(ds, 10, below_model=-999, above_model=999)
    assert (layer == layer.attrs["above_model"]).all()


def test_get_layer_of_z_below_model():
    ds = nlmod.get_ds([0, 1000, 0, 500], top=0, botm=[-10, -20])
    layer = nlmod.layers.get_layer_of_z(ds, -30, below_model=-999, above_model=999)
    assert (layer == layer.attrs["below_model"]).all()


def test_aggregate_by_weighted_mean_to_ds():
    regis = get_regis_horstermeer()
    regis2 = regis.copy(deep=True)

    # botm needs to have the name "bottom'
    regis2["bottom"] = regis2["botm"]
    # top needs to be 3d
    regis2["top"] = regis2["botm"] + nlmod.layers.calculate_thickness(regis2)
    kh_new = nlmod.layers.aggregate_by_weighted_mean_to_ds(regis, regis2, "kh")
    assert np.abs(kh_new - regis["kh"]).max() < 1e-5
    # assert (kh_new.isnull() == regis["kh"].isnull()).all() # does not assert to True...


def test_check_elevations_consistency(caplog):
    regis = get_regis_horstermeer()
    # there are no inconsistencies in this dataset, let's check for that:
    nlmod.layers.check_elevations_consistency(regis)
    assert len(caplog.text) == 0

    # add an inconsistency by lowering the top of the model in part of the model domain
    regis["top"][10:20, 20:25] = -5
    nlmod.layers.check_elevations_consistency(regis)
    assert "check_elevations_consistency" not in caplog.text
    assert len(caplog.text) > 0
    assert "Thickness of layers is negative in 50 cells" in caplog.text


def test_get_first_and_last_active_layer():
    regis = get_regis_horstermeer()
    thickness = nlmod.layers.calculate_thickness(regis)

    fal = nlmod.layers.get_first_active_layer(regis)
    assert (thickness[fal] > 0).all()

    lal = nlmod.layers.get_last_active_layer(regis)
    assert (thickness[lal] > 0).all()


def test_set_model_top(plot=False):
    regis = get_regis_horstermeer()
    ds_new = nlmod.layers.set_model_top(regis.copy(deep=True), 5.0)

    assert (ds_new["top"] == 5.0).all()
    th_new = nlmod.layers.calculate_thickness(ds_new)
    assert (th_new >= 0).all()
    if plot:
        plot_test(regis, ds_new)


def test_set_layer_top(plot=False):
    regis = get_regis_horstermeer()
    ds_new = nlmod.layers.set_layer_top(regis.copy(deep=True), "WAk1", -40.0)

    th_new = nlmod.layers.calculate_thickness(ds_new)
    top_new = ds_new["botm"].loc["WAk1"] + th_new.loc["WAk1"]
    assert (top_new == -40).all()
    assert (th_new >= 0).all()
    assert th_new.sum() == nlmod.layers.calculate_thickness(ds_new).sum()
    if plot:
        plot_test(regis, ds_new)


def test_set_layer_botm(plot=False):
    regis = get_regis_horstermeer()
    ds_new = nlmod.layers.set_layer_botm(regis.copy(deep=True), "WAk1", -75.0)

    th_new = nlmod.layers.calculate_thickness(ds_new)
    assert (ds_new["botm"].loc["WAk1"] == -75).all()
    assert (th_new >= 0).all()
    assert th_new.sum() == nlmod.layers.calculate_thickness(ds_new).sum()
    if plot:
        plot_test(regis, ds_new)


def test_insert_layer():
    # download regis
    ds1 = get_regis_horstermeer()

    # just replace the 2nd layer by a new insertion
    layer = ds1.layer.data[1]
    new_layer = "test"
    ds1_top3d = ds1["botm"] + nlmod.layers.calculate_thickness(ds1)
    ds2 = nlmod.layers.insert_layer(
        ds1, "test", ds1_top3d.loc[layer], ds1["botm"].loc[layer]
    )

    # make sure the total thickness of the two models is equal (does not assert to True yet)
    # total_thickness1 = float(nlmod.layers.calculate_thickness(ds1).sum())
    # total_thickness2 = float(nlmod.layers.calculate_thickness(ds2).sum())
    # assert total_thickness1 == total_thickness2

    # msake sure the original layer has no thickness left
    assert nlmod.layers.calculate_thickness(ds2).loc[layer].sum() == 0

    # make sure the top of the new layer is equal to the top of the original layer
    ds2_top3d = ds2["botm"] + nlmod.layers.calculate_thickness(ds2)
    assert (ds2_top3d.loc[new_layer] == ds1_top3d.loc[layer]).all()

    # make sure the top of the new layer is equal to the top of the original layer
    assert (ds2["botm"].loc[new_layer] == ds1["botm"].loc[layer]).all()


def test_remove_thin_layers():
    # download regis and define min_thickness
    regis = get_regis_horstermeer()

    min_thickness = 1.0

    # test update_thickness_every_layer = False
    ds_new = nlmod.layers.remove_thin_layers(regis, min_thickness)

    th = nlmod.layers.calculate_thickness(regis)
    assert th.where(th > 0).min() < min_thickness

    th_new = nlmod.layers.calculate_thickness(ds_new)
    assert th_new.where(th_new > 0).min() <= min_thickness

    # the following does not assert to True, as there are negative thicknesses in the original regis data
    # assert th_new.sum() == th.sum()

    # test if all active cells in the new dataset were active in the original dataset
    assert (th.data[th_new > 0] > 0).all()

    # test update_thickness_every_layer = True
    ds_new2 = nlmod.layers.remove_thin_layers(
        regis, min_thickness, update_thickness_every_layer=True
    )
    th_new2 = nlmod.layers.calculate_thickness(ds_new2)

    assert th_new2.where(th_new2 > 0).min() <= min_thickness

    assert th_new2.sum() == th_new.sum()

    # test if all active cells in the new dataset were active in the original dataset
    assert (th.data[th_new2 > 0] > 0).all()


def test_get_modellayers_screens():
    ds = test_001_model.get_ds_from_cache("small_model")
    xy = [
        [98900, 489600],
        [98800, 489500],
        [98980, 489680],
        [98980, 489680],
    ]
    screen_top = [10, -1, -35, 1000]
    screen_bottom = [9, -20, -100, -1000]
    modellayers = nlmod.layers.get_modellayers_screens(
        ds, screen_top, screen_bottom, xy=xy
    )
    assert np.isnan(modellayers[0])
    assert modellayers[1] == 1.0
    assert modellayers[2] == ds.sizes["layer"] - 1

    ds_ref = nlmod.grid.refine(ds, refinement_features=[])
    modellayers_ref = nlmod.layers.get_modellayers_screens(
        ds_ref, screen_top, screen_bottom, xy=xy
    )
    assert modellayers == modellayers_ref


def test_get_modellayers_indexer():
    ds = test_001_model.get_ds_from_cache("small_model")
    data = {
        "x": [98900, 98800, 98980, 98980],
        "y": [489600, 489500, 489680, 489680],
        "screen_top": [10, -1, -35, 1000],
        "screen_bottom": [9, -20, -100, -1000],
    }
    df = DataFrame(data)

    # structured grid
    idx = nlmod.layers.get_modellayers_indexer(ds, df)
    # check result
    assert idx["layer"].values[0] == ds["layer"].values[1]
    assert idx["layer"].values[1] == ds["layer"].values[ds.sizes["layer"] - 1]
    # test getting bottom elevations using indexer
    _ = ds["botm"].sel(**idx)

    # structured grid keep nan
    idx2 = nlmod.layers.get_modellayers_indexer(ds, df, drop_nan_layers=False)
    # check result
    assert np.isnan(idx2["layer"].values[0])

    # drop nans
    idx2 = idx2.dropna("index")
    # get layer names (step is unnecesary if you keep drop_nan_layers=True)
    idx2["layer"].values = ds.layer[idx2["layer"].astype(int)].values
    # test getting bottom elevations using indexer
    _ = ds["botm"].sel(**idx2)

    # vertex grid
    ds_ref = nlmod.grid.refine(ds, refinement_features=[])
    idx3 = nlmod.layers.get_modellayers_indexer(ds_ref, df)
    _ = ds_ref["botm"].sel(**idx3)

    assert (idx3["layer"] == idx["layer"]).all()

    # full output
    idxfull = nlmod.layers.get_modellayers_indexer(ds, df, full_output=True)
    assert (idxfull["modellayer_top"] == np.array([1, 4, 0])).all()
    assert (idxfull["modellayer_bot"] == np.array([2, 4, 4])).all()
