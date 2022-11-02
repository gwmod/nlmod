# -*- coding: utf-8 -*-
"""Created on Tue Aug 16 10:29:13 2022.

@author: Ruben
"""

import nlmod
import pytest


def test_download_polygons():
    return nlmod.read.waterboard.get_polygons()


def test_get_config():
    return nlmod.read.waterboard.get_configuration()


def test_bgt_waterboards():
    extent = [116500, 120000, 439000, 442000]
    bgt = nlmod.read.bgt.get_bgt(extent)
    pg = nlmod.gwf.surface_water.download_level_areas(bgt, extent=extent)
    bgt = nlmod.gwf.surface_water.add_stages_from_waterboards(bgt, pg=pg)
    return bgt


@pytest.mark.skip("too slow")
def test_download_peilgebieden(plot=True):
    waterboards = nlmod.read.waterboard.get_polygons()
    data_kind = "level_areas"

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
        # plot the winter_stage
        ax = waterboards.plot(edgecolor="k", facecolor="none")
        for wb in waterboards.index:
            if wb in gdf:
                # gdf[wb].plot(ax=ax, zorder=0)
                gdf[wb].plot("winter_stage", ax=ax, zorder=0, vmin=-10, vmax=20)
            c = waterboards.at[wb, "geometry"].centroid
            ax.text(c.x, c.y, wb.replace(" ", "\n"), ha="center", va="center")


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
    # data_kind = "peilgebieden"
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
