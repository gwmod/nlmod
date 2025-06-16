import os

import matplotlib
import numpy as np
import pytest

import nlmod

# def test_download_polygons(): # is tested in test_024_administrative.test_get_waterboards
#     nlmod.read.waterboard.get_polygons()


def test_get_config():
    nlmod.read.waterboard.get_configuration()


def test_bgt_waterboards():
    extent = [116500, 120000, 439000, 442000]
    bgt = nlmod.read.bgt.download_bgt(extent)

    la = nlmod.gwf.surface_water.download_level_areas(
        bgt, extent=extent, raise_exceptions=False
    )
    bgt = nlmod.gwf.surface_water.add_stages_from_waterboards(bgt, la=la)


def get_ahn_colormap(name="ahn", N=256):
    colors = np.array(
        [
            [0, 98, 177],  # dark blue
            [0, 156, 224],  # medium blue
            [115, 216, 255],  # light blue
            [128, 137, 0],  # dark green
            [164, 221, 0],  # light green: center
            [252, 220, 0],  # yellow
            [251, 158, 0],  # orange
            [211, 49, 21],  # light red
            [159, 5, 0],  # dark red
        ]
    )
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(name, colors / 255, N=N)
    return cmap


@pytest.mark.skip("too slow")
def test_download_peilgebieden(
    data_kind="level_areas", plot=True, save=True, figdir=r"..\docs\_static"
):
    waterboards = nlmod.read.waterboard.get_polygons()

    gdf = {}
    for wb in waterboards.index:
        print(wb)
        try:
            # xmin, ymin, xmax, ymax = waterboards.at[wb, "geometry"].bounds
            # extent = [xmin, xmax, ymin, ymax]
            gdf[wb] = nlmod.read.waterboard.get_data(
                wb, data_kind, max_record_count=1000
            )
        except Exception as e:
            if str(e) == f"{data_kind} not available for {wb}":
                print(e)
            else:
                raise

    if plot:
        if data_kind == "level_areas":
            columns = ["winter_stage", "summer_stage"]
            labels = ["Winter stage (m NAP)", "Summer stage (m NAP)"]
        elif data_kind == "watercourses":
            columns = ["bottom_height"]
            labels = ["Bottom height (m NAP)"]
        else:
            raise (Exception(f"Unknown data_kind: {data_kind}"))
        # plot the winter_stage and summer_stage
        for column, label in zip(columns, labels):
            f, ax = nlmod.plot.get_map([9000, 279000, 304000, 623000], base=100000)
            waterboards.plot(edgecolor="k", facecolor="none", ax=ax)
            norm = matplotlib.colors.Normalize(-10.0, 20.0)
            cmap = get_ahn_colormap()
            for wb in waterboards.index:
                if wb in gdf:
                    try:
                        # gdf[wb].plot(ax=ax, zorder=0)
                        gdf[wb].plot(column, ax=ax, zorder=0, norm=norm, cmap=cmap)
                    except Exception as e:
                        print(f"plotting of {data_kind} for {wb} failed: {e}")
                c = waterboards.at[wb, "geometry"].centroid
                ax.text(c.x, c.y, wb.replace(" ", "\n"), ha="center", va="center")
            nlmod.plot.colorbar_inside(
                ax=ax,
                norm=norm,
                cmap=cmap,
                bounds=[0.05, 0.55, 0.02, 0.4],
                label=label,
            )
            if save:
                f.savefig(os.path.join(figdir, column))


@pytest.mark.skip("too slow")
def test_download_waterlopen(plot=True):
    def get_extent(waterboards, wb, buffer=1000.0):
        c = waterboards.at[wb, "geometry"].centroid
        extent = [c.x - buffer, c.x + buffer, c.y - buffer, c.y + buffer]
        if wb == "Vallei & Veluwe":
            extent = [170000, 172000, 460000, 462000]
        # elif wb == "Aa en Maas":
        #    extent = [132500, 147500, 408000, 416000]
        # elif wb == "HH Hollands Noorderkwartier":
        #    extent = [120000, 123000, 510000, 513000]
        # elif wb == "Waterschap Limburg":
        #    extent = [190000, 196000, 358000, 364000]
        # elif wb == "Brabantse Delta":
        #    extent = [100000, 105000, 405000, 410000]
        # elif wb == "Waterschap Scheldestromen":
        #    extent = [57000, 58000, 378000, 379000]
        return extent

    data_kind = "watercourses"
    # data_kind = "level_areas"
    waterboards = nlmod.read.waterboard.get_polygons()
    gdf = {}
    for wb in waterboards.index:
        print(wb)
        extent = get_extent(waterboards, wb)
        try:
            gdf[wb] = nlmod.read.waterboard.get_data(wb, data_kind, extent)
        except Exception as e:
            if str(e) == f"{data_kind} not available for {wb}":
                print(e)
            else:
                print(e)
                raise

    if plot:
        for wb in gdf:
            ax = gdf[wb].plot()
            ax.axis(get_extent(waterboards, wb))
            ax.set_title(wb)
