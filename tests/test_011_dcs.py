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

    # add a map at the bottom right corner, 35 % of axes hight
    axes_bounds = nlmod.plot.plotutil.get_inset_map_bounds(
        ax=ax, extent=ds.extent, height=0.35, right=True
    )
    mapax = nlmod.plot.inset_map(ax, ds.extent, provider=None, axes_bounds=axes_bounds)
    nlmod.plot.add_xsec_line_and_labels(line, ax, mapax)

    # add a map at the top left corner, 20 % of axes width
    axes_bounds = nlmod.plot.get_inset_map_bounds(
        ax=ax,
        extent=ds.extent,
        width=0.2,
        right=False,
        bottom=False,
    )
    mapax = nlmod.plot.inset_map(ax, ds.extent, provider=None, axes_bounds=axes_bounds)
    nlmod.plot.add_xsec_line_and_labels(line, ax, mapax)
