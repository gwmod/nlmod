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
    nlmod.plot.get_map([100000, 101000, 400000, 401000], background=True, figsize=3)


def test_map_array():
    ds = util.get_ds_structured()
    nlmod.plot.map_array(ds["kh"], ds, ilay=0)

    gwf = util.get_gwf(ds)
    nlmod.plot.flopy.map_array(ds["kh"], gwf, ilay=0)


def test_flopy_contour_array():
    ds = util.get_ds_structured()
    gwf = util.get_gwf(ds)
    nlmod.plot.flopy.contour_array(ds["kh"], gwf, ilay=0)


def test_flopy_animate_map():
    # no test implemented yet
    pass


def test_flopy_facet_plot():
    ds = util.get_ds_structured()
    gwf = util.get_gwf(ds)
    nlmod.plot.flopy.facet_plot(gwf, ds["kh"])
