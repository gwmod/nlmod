import nlmod
from shapely.geometry import LineString


def test_dcs_structured():
    ds = nlmod.get_ds([0, 1000, 0, 1000])
    line = [(0, 0), (1000, 1000)]
    dcs = nlmod.dcs.DatasetCrossSection(ds, line)
    dcs.plot_layers()
    dcs.plot_array(ds["kh"], alpha=0.5)
    dcs.plot_grid()


def test_dcs_vertex():
    ds = nlmod.get_ds([0, 1000, 0, 1000])
    refinement_features = [([LineString([(0, 1000), (1000, 0)])], "line", 1)]
    ds = nlmod.grid.refine(ds, "refine", refinement_features=refinement_features)
    line = [(0, 0), (1000, 1000)]
    dcs = nlmod.dcs.DatasetCrossSection(ds, line)
    dcs.plot_layers()
    dcs.plot_array(ds["kh"], alpha=0.5)
    dcs.plot_grid()
