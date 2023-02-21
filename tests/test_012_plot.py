import nlmod
from shapely.geometry import LineString


def test_plot_modelgrid():
    ds = nlmod.get_ds([0, 1000, 0, 1000])
    nlmod.plot.modelgrid(ds)


def test_plot_surface_water_empty():
    ds = nlmod.get_ds([0, 1000, 0, 1000])
    nlmod.plot.surface_water(ds)


def test_plot_data_array_structured():
    # also test colorbar_inside and title_inside
    ds = nlmod.get_ds([0, 1000, 0, 1000])
    pcm = nlmod.plot.data_array(ds["top"], edgecolor="k")
    nlmod.plot.colorbar_inside(pcm)
    nlmod.plot.title_inside("top")


def test_plot_data_array_vertex():
    ds = nlmod.get_ds([0, 1000, 0, 1000])
    refinement_features = [([LineString([(0, 1000), (1000, 0)])], "line", 1)]
    ds = nlmod.grid.refine(ds, "refine", refinement_features=refinement_features)
    nlmod.plot.data_array(ds["top"], ds=ds, edgecolor="k")


def test_plot_get_map():
    nlmod.plot.get_map([100000, 101000, 400000, 401000], background=True, figsize=3)
