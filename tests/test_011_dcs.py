import util

import nlmod
import matplotlib.pyplot as plt


def test_dcs_structured():
    ds = util.get_ds_structured()
    line = [(0, 0), (1000, 1000)]
    dcs = nlmod.plot.DatasetCrossSection(ds, line)
    dcs.plot_layers()
    dcs.label_layers()
    dcs.plot_array(ds["kh"], alpha=0.5)
    dcs.plot_grid()


def test_dcs_vertex():
    ds = util.get_ds_vertex()
    line = [(0, 0), (1000, 1000)]
    dcs = nlmod.plot.DatasetCrossSection(ds, line)
    dcs.plot_layers()
    dcs.label_layers()
    dcs.plot_array(ds["kh"], alpha=0.5)
    dcs.plot_grid(vertical=False)


def test_cross_section_utils():
    ds = util.get_ds_vertex()
    line = [(0, 0), (1000, 1000)]
    _, ax = plt.subplots(1, 1, figsize=(10, 4))
    dcs = nlmod.plot.DatasetCrossSection(ds, line, ax=ax)
    dcs.plot_layers()
    dcs.label_layers()
    dcs.plot_array(ds["kh"], alpha=0.5)
    dcs.plot_grid(vertical=False)
    mapax = nlmod.plot.inset_map(ax, ds.extent, provider=None)
    nlmod.plot.add_xsec_line_and_labels(line, ax, mapax)
