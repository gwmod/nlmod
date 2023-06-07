import os

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
    polys2 = dcs1.plot_layers(colors=colors, min_label_area=min_label_area)
    dcs1.plot_grid(linewidth=0.5, vertical=False)
    ax1.set_ylabel(ylabel)

    if ds2 is not None:
        ax1.set_title(title1)
        dcs2 = DatasetCrossSection(ds2, line=line, ax=ax2, zmin=zmin, zmax=zmax)
        polys1 = dcs2.plot_layers(colors=colors, min_label_area=min_label_area)
        dcs2.plot_grid(linewidth=0.5, vertical=False)
        ax2.set_ylabel(ylabel)
        ax2.set_xlabel(xlabel)
        ax2.set_title(title2)
    else:
        ax1.set_xlabel(xlabel)


def plot_test(ds, ds_new):
    line = LineString([(ds.extent[0], ds.extent[2]), (ds.extent[1], ds.extent[3])])
    colors = nlmod.read.regis.get_legend()["color"].to_dict()
    compare_layer_models(ds, line, colors, ds_new)


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
