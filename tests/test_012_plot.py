import util

import nlmod


def test_plot_modelgrid():
    ds = util.get_ds_structured()
    nlmod.plot.modelgrid(ds)


def test_plot_surface_water_empty():
    ds = util.get_ds_structured()
    nlmod.plot.surface_water(ds)


def test_plot_data_array_structured():
    # also test colorbar_inside and title_inside
    ds = util.get_ds_structured()
    pcm = nlmod.plot.data_array(ds["top"], edgecolor="k")
    nlmod.plot.colorbar_inside(pcm)
    nlmod.plot.title_inside("top")


def test_plot_data_array_vertex():
    ds = util.get_ds_vertex()
    nlmod.plot.data_array(ds["top"], ds=ds, edgecolor="k")
    nlmod.plot.modelgrid(ds)


def test_plot_get_map():
    nlmod.plot.get_map([100000, 101000, 400000, 401000], backgroundmap=True, figsize=3)
