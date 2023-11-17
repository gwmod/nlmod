import os

import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString

import nlmod
from nlmod.plot import DatasetCrossSection


def get_regis_horstermeer():
    extent = [131000, 136800, 471500, 475700]
    cachedir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")
    if not os.path.isdir(cachedir):
        os.makedirs(cachedir)
    regis = nlmod.read.get_regis(
        extent, cachedir=cachedir, cachename="regis_horstermeer"
    )
    return regis


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


def test_split_layers():
    regis = get_regis_horstermeer()
    split_dict = {"PZWAz2": (0.3, 0.3, 0.4), "PZWAz3": (0.2, 0.2, 0.2, 0.2, 0.2)}
    regis_split, split_reindexer = nlmod.layers.split_layers_ds(
        regis, split_dict, return_reindexer=True
    )
    assert (
        nlmod.layers.calculate_thickness(regis).sum()
        == nlmod.layers.calculate_thickness(regis_split).sum()
    )

    # diagonal line through extent
    colors = nlmod.read.regis.get_legend()["color"].to_dict()
    for j, i in split_reindexer.items():
        if j not in colors:
            colors[j] = colors[i]
    plot_test(regis, regis_split, colors=colors)


def test_add_layer_dim_to_top():
    regis = get_regis_horstermeer()
    nlmod.layers.add_layer_dim_to_top(regis)


def test_combine_layers():
    regis = get_regis_horstermeer()
    combine_layers = [
        tuple(np.argwhere(regis.layer.str.startswith("URz").data).squeeze().tolist()),
        tuple(
            np.argwhere(regis.layer.isin(["PZWAz2", "PZWAz3"]).data).squeeze().tolist()
        ),
    ]
    regis_combined = nlmod.layers.combine_layers_ds(
        regis, combine_layers, kD=None, c=None
    )

    # plot comparison
    plot_test(regis, regis_combined)


def test_set_layer_thickness(plot=False):
    regis = get_regis_horstermeer()
    ds = nlmod.to_model_ds(regis)

    ds_new = nlmod.layers.set_layer_thickness(ds.copy(deep=True), "WAk1", 10.0)
    th = nlmod.layers.calculate_thickness(ds_new)
    assert (th >= 0).all()
    # assert (th.sel(layer="WAk1") == 10.0).all()

    if plot:
        plot_test(ds, ds_new)


def test_set_minimum_layer_thickness(plot=False):
    regis = get_regis_horstermeer()
    ds = nlmod.to_model_ds(regis)

    ds_new = nlmod.layers.set_minimum_layer_thickness(ds.copy(deep=True), "WAk1", 5.0)
    th = nlmod.layers.calculate_thickness(ds_new)
    assert (th >= 0).all()
    # assert (th.sel(layer="WAk1") == 10.0).all()

    if plot:
        plot_test(ds, ds_new)


def test_set_layer_top(plot=False):
    regis = get_regis_horstermeer()
    ds = nlmod.to_model_ds(regis)

    ds_new = nlmod.layers.set_layer_top(ds.copy(deep=True), "WAk1", -40.0)
    assert (nlmod.layers.calculate_thickness(ds_new) >= 0).all()

    if plot:
        plot_test(ds, ds_new)


def test_set_layer_botm(plot=False):
    regis = get_regis_horstermeer()
    ds = nlmod.to_model_ds(regis)

    ds_new = nlmod.layers.set_layer_botm(ds.copy(deep=True), "WAk1", -75.0)
    assert (nlmod.layers.calculate_thickness(ds_new) >= 0).all()

    if plot:
        plot_test(ds, ds_new)


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

    # make sure the top of the new layer is euqal to the top of the original layer
    ds2_top3d = ds2["botm"] + nlmod.layers.calculate_thickness(ds2)
    assert (ds2_top3d.loc[new_layer] == ds1_top3d.loc[layer]).all()

    # make sure the top of the new layer is euqal to the top of the original layer
    assert (ds2["botm"].loc[new_layer] == ds1["botm"].loc[layer]).all()
